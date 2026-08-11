"""Gate for per-component learning rates and checkpoint pruning.

    .venv/bin/python anima/parity/test_param_groups.py

CPU-only, no model load: `build_param_groups` / `build_adapter_param_groups` only ever call
`named_parameters()`, so duck-typed stubs exercise the real code paths at full fidelity and let the
LoRA naming be spelled out explicitly rather than inferred from a 2B checkpoint.

Two regressions this pins:
  * `component_lr` under LoRA used to be **silently ignored** for everything except adaln/base --
    the config guard checked only those two, so a `component_lr.mlp` was accepted and dropped;
  * `keep_last_n` was declared in `TrainConfig`, accepted by the loader, and never read.
"""

from __future__ import annotations

import os
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from anima.training.config import TrainConfig  # noqa: E402
from anima.training.params import (  # noqa: E402
    ComponentLRs,
    build_adapter_param_groups,
    build_param_groups,
    classify,
)
from anima.training.train import Trainer  # noqa: E402


class Fake:
    """Anything with `named_parameters()` is enough for the group builders."""

    def __init__(self, names, requires_grad=True):
        self._p = [
            (n, nn.Parameter(torch.zeros(4), requires_grad=requires_grad)) for n in names
        ]

    def named_parameters(self):
        return list(self._p)


# Real peft naming: the adapter suffix sits *after* the part `classify()` keys on.
LORA_NAMES = [
    "transformer_blocks.0.attn1.to_q.lora_A.default.weight",
    "transformer_blocks.0.attn1.to_out.0.lora_B.default.weight",
    "transformer_blocks.0.attn2.to_k.lora_A.default.weight",
    "transformer_blocks.1.ff.net.0.proj.lora_A.default.weight",
    "transformer_blocks.1.ff.net.2.lora_B.default.weight",
    "transformer_blocks.0.norm1.linear_1.lora_A.default.weight",
]
FULL_NAMES = [
    "transformer_blocks.0.attn1.to_q.weight",
    "transformer_blocks.0.attn2.to_k.weight",
    "transformer_blocks.0.ff.net.0.proj.weight",
    "transformer_blocks.0.norm1.linear_1.weight",
    "patch_embed.proj.weight",
    "norm_out.linear_1.weight",
    "time_embed.t_embedder.linear_1.weight",
]


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    return bool(ok)


def lrs_of(report):
    return {g["component"]: g["lr"] for g in report.groups}


def main() -> int:
    r = []

    # ------------------------------------------------------------- classify on LoRA names
    got = {n.split(".")[2] if n.startswith("transformer_blocks") else n: classify(n)
           for n in LORA_NAMES}
    r.append(check("classify() handles peft-suffixed names",
                   got == {"attn1": "self_attn", "attn2": "cross_attn", "ff": "mlp",
                           "norm1": "adaln"}, str(got)))

    # ------------------------------------------------------------- LoRA grouping
    rep = build_adapter_param_groups(
        Fake(LORA_NAMES), Fake(["a.weight"]), Fake(["b.weight"]),
        ComponentLRs(self_attn=3e-5, cross_attn=1.5e-5, mlp=4.5e-5, adaln=2e-5,
                     llm_adapter=1e-5, text_encoder=5e-6),
        default_lr=1e-4,
    )
    r.append(check("LoRA groups split by component",
                   lrs_of(rep) == {"self_attn": 3e-5, "cross_attn": 1.5e-5, "mlp": 4.5e-5,
                                   "adaln": 2e-5, "llm_adapter": 1e-5, "text_encoder": 5e-6},
                   str(lrs_of(rep))))
    r.append(check("text_encoder is its own group (the kohya text_encoder_lr gap)",
                   "text_encoder" in lrs_of(rep) and lrs_of(rep)["text_encoder"] == 5e-6))

    # Unset components inherit the global LR -- the previous single-group behaviour.
    flat = build_adapter_param_groups(Fake(LORA_NAMES), None, None,
                                      ComponentLRs(adaln=None), default_lr=2e-5)
    r.append(check("unset components fall back to optimizer.lr",
                   set(lrs_of(flat).values()) == {2e-5}, str(lrs_of(flat))))
    total = sum(flat.counts.values())
    r.append(check("no parameter is lost when splitting",
                   total == 4 * len(LORA_NAMES), f"{total} elems"))

    # 0.0 freezes rather than making a zero-LR group.
    frz = build_adapter_param_groups(Fake(LORA_NAMES), None, None,
                                     ComponentLRs(mlp=0.0), default_lr=2e-5)
    r.append(check("component_lr 0.0 freezes the adapter tensors",
                   "mlp" not in lrs_of(frz) and frz.frozen.get("mlp", 0) == 8,
                   f"frozen={frz.frozen}"))
    try:
        build_adapter_param_groups(
            Fake(LORA_NAMES), None, None,
            ComponentLRs(self_attn=0.0, cross_attn=0.0, mlp=0.0, adaln=0.0), default_lr=1e-5)
        r.append(check("all-frozen adapter raises", False, "accepted"))
    except ValueError:
        r.append(check("all-frozen adapter raises", True))

    # ------------------------------------------------------------- full-FT regression
    full = build_param_groups(Fake(FULL_NAMES), Fake(["c.weight"]),
                              ComponentLRs(self_attn=1e-5, mlp=1.5e-5), default_lr=1e-5)
    r.append(check("full FT: adaln/base/llm_adapter still frozen by default",
                   set(lrs_of(full)) == {"self_attn", "cross_attn", "mlp"}, str(lrs_of(full))))
    r.append(check("full FT: mlp keeps its own LR", lrs_of(full)["mlp"] == 1.5e-5))

    # ------------------------------------------------------------- explicit()
    e = ComponentLRs(mlp=2e-5, adaln=0.0).explicit()
    r.append(check("explicit() reports only user-set values (adaln=0.0 is the default)",
                   e == {"mlp": 2e-5}, str(e)))

    # ------------------------------------------------------------- LoKr export
    # peft absorbed the LyCORIS methods, so `kind = "lokr"` trains -- but it stores each factor as
    # a bare nn.Parameter named `<marker>.<adapter_name>`, with no `.weight`. The export used to
    # leave that `.default` in place and emit `...lokr_w1.default`, which loads nowhere: ComfyUI
    # looks up `<key>.lokr_w1` (comfy/weight_adapter/lokr.py:211) and diffusers'
    # AnimaLoraLoaderMixin has no LoKr support at all.
    from anima.modeling.convert import ConversionError, _strip_peft_key, lora_to_native

    strips = {
        "transformer_blocks.0.attn1.to_q.lora_A.default.weight":
            "transformer_blocks.0.attn1.to_q.lora_A.weight",
        "transformer_blocks.0.attn1.to_q.lokr_w1.default":
            "transformer_blocks.0.attn1.to_q.lokr_w1",
        "transformer_blocks.0.attn1.to_q.lokr_w2_a.default":
            "transformer_blocks.0.attn1.to_q.lokr_w2_a",
        "transformer_blocks.0.attn1.to_q.lokr_t2.default":
            "transformer_blocks.0.attn1.to_q.lokr_t2",
        # already clean -- the second rule must not strip this back to `lora_A`
        "transformer_blocks.0.attn1.to_q.lora_A.weight":
            "transformer_blocks.0.attn1.to_q.lora_A.weight",
    }
    bad = {k: _strip_peft_key(k) for k in strips if _strip_peft_key(k) != strips[k]}
    r.append(check("peft adapter suffixes stripped for both LoRA and LoKr", not bad, str(bad)))

    lokr_sd = {f"transformer_blocks.0.attn1.to_q.{n}.default": torch.zeros(2, 2)
               for n in ("lokr_w1", "lokr_w2_a", "lokr_w2_b")}
    got = set(lora_to_native(transformer_lora=lokr_sd))
    base = "diffusion_model.blocks.0.self_attn.q_proj"
    want = {f"{base}.lokr_w1", f"{base}.lokr_w2_a", f"{base}.lokr_w2_b"}
    r.append(check("LoKr export keys match what ComfyUI looks up", got == want,
                   str(sorted(got - want) or "exact")))

    # kohya format is LoRA-only by construction (lora_down/lora_up have no LoKr analogue), and
    # must say so rather than emit something unloadable.
    from anima.modeling.convert import native_lora_to_kohya

    try:
        native_lora_to_kohya({f"{base}.lokr_w1": torch.zeros(2, 2)}, alpha=8.0)
        r.append(check("kohya export rejects LoKr", False, "accepted"))
    except ConversionError:
        r.append(check("kohya export rejects LoKr", True))

    # ------------------------------------------------------------- DDP + compile wrappers
    # A real crash, at the first epoch checkpoint of a 2-GPU compiled LoRA run. Accelerate's
    # `compile_regions` stamps `_orig_mod` on the root object it returns; regional compilation runs
    # after the DDP wrap, so that root IS the DDP module, and `unwrap_model`'s default
    # keep_torch_compile=True then re-attaches the wrapper it had just removed. Result: `module.`
    # at the root and `_orig_mod` inside every block, 560 unmapped keys, run dead after the epoch.
    from anima.modeling.convert import _strip_wrapper_key, strip_wrappers

    wrapped = {
        "module.transformer_blocks.0._orig_mod.attn1.to_q.lora_A.default.weight": torch.zeros(2, 2),
        "module.transformer_blocks.0._orig_mod.attn1.to_q.lora_B.default.weight": torch.zeros(2, 2),
    }
    got = set(lora_to_native(transformer_lora=wrapped))
    want = {f"{base}.lora_A.weight", f"{base}.lora_B.weight"}
    r.append(check("DDP + regional-compile keys still export (the 560-unmapped crash)",
                   got == want, str(sorted(got - want) or "exact")))

    cases = {
        # exactly the shape from the traceback
        "module.transformer_blocks.0._orig_mod.attn1.to_q.weight":
            "transformer_blocks.0.attn1.to_q.weight",
        # whole-model compile instead of regional
        "_orig_mod.transformer_blocks.0.attn1.to_q.weight":
            "transformer_blocks.0.attn1.to_q.weight",
        # both, and nested DDP
        "module.module._orig_mod.transformer_blocks.0._orig_mod.ff.net.2.weight":
            "transformer_blocks.0.ff.net.2.weight",
        # already clean: must be left exactly alone
        "transformer_blocks.0.attn1.to_q.weight": "transformer_blocks.0.attn1.to_q.weight",
        "patch_embed.proj.weight": "patch_embed.proj.weight",
    }
    bad = {k: _strip_wrapper_key(k) for k in cases if _strip_wrapper_key(k) != cases[k]}
    r.append(check("wrapper stripping handles every nesting", not bad, str(bad)))

    # Full finetune goes through the same choke point, and would have hit the same wall.
    from anima.modeling.convert import transformer_to_native

    native = transformer_to_native(
        {"module.transformer_blocks.0._orig_mod.attn1.to_q.weight": torch.zeros(2, 2)})
    r.append(check("full-FT native export strips wrappers too",
                   list(native) == ["net.blocks.0.self_attn.q_proj.weight"], str(list(native))))
    r.append(check("strip_wrappers is a no-op on clean keys",
                   strip_wrappers({"a.b.weight": 1}) == {"a.b.weight": 1}))

    # ------------------------------------------------------------- progress / ETA
    from anima.training.train import _emit, _fmt_duration

    r.append(check("duration formatting",
                   [_fmt_duration(x) for x in (0, 5, 65, 3600, 3725)]
                   == ["0s", "5s", "1m05s", "1h00m", "1h02m"]))

    # The ETA is a cumulative mean from the anchor, not the last interval: with bucketing (and far
    # more so with multi-res) step cost swings with whichever bucket came up, so an ETA off `dt`
    # would jump several-fold between log lines.
    now = time.time()
    eta_stub = SimpleNamespace(
        _eta_anchor=(now - 100.0, 10), global_step=110, total_steps=210)
    got = Trainer._eta(eta_stub)
    r.append(check("ETA extrapolates the mean step time", got == "1m40s", str(got)))
    r.append(check("progress_pct", abs(Trainer.progress_pct(eta_stub) - 52.38) < 0.01))

    r.append(check("no ETA before the anchor exists",
                   Trainer._eta(SimpleNamespace(_eta_anchor=None)) is None))
    r.append(check("no ETA on the anchor step itself (no division by zero)",
                   Trainer._eta(SimpleNamespace(_eta_anchor=(now, 10), global_step=10,
                                                total_steps=210)) is None))
    r.append(check("no ETA once the run is complete",
                   Trainer._eta(SimpleNamespace(_eta_anchor=(now - 100.0, 10), global_step=210,
                                                total_steps=210)) is None))

    # `_emit` is what save/prune use so a checkpoint message does not tear the bar. It must also
    # work on an object with no `bar` at all -- which is exactly the stub below.
    _emit(SimpleNamespace(), "  (emit without a bar: printed above)")
    r.append(check("_emit works on an object with no bar attribute", True))

    for bad in ("yes", "tqdm", ""):
        try:
            TrainConfig(progress=bad)
            r.append(check(f"train.progress={bad!r} rejected", False, "accepted"))
            break
        except ValueError:
            pass
    else:
        r.append(check("invalid train.progress rejected", True))
    r.append(check("valid train.progress accepted",
                   all(TrainConfig(progress=p).progress == p
                       for p in ("auto", "bar", "plain", "off"))))

    # ------------------------------------------------------------- keep_last_n
    tmp = Path(tempfile.mkdtemp(prefix="anima_prune_"))
    try:
        for name, step in (("epoch001", 10), ("epoch002", 20), ("epoch003", 30),
                           ("step000040", 40), ("final", 99)):
            d = tmp / name
            d.mkdir()
            (d / "state.json").write_text(json.dumps({"global_step": step}))
        (tmp / "notacheckpoint").mkdir()
        (tmp / "epoch009_nostate").mkdir()          # no state.json -> not a candidate

        stub = SimpleNamespace(cfg=SimpleNamespace(train=TrainConfig(keep_last_n=2)), out_dir=tmp)
        Trainer._prune_checkpoints(stub)
        left = sorted(p.name for p in tmp.iterdir())
        r.append(check("keep_last_n keeps the newest N by global_step",
                       left == ["epoch003", "epoch009_nostate", "final", "notacheckpoint",
                                "step000040"], str(left)))
        r.append(check("keep_last_n never deletes `final` or foreign dirs",
                       (tmp / "final").exists() and (tmp / "notacheckpoint").exists()))

        stub.cfg.train.keep_last_n = None
        before = sorted(p.name for p in tmp.iterdir())
        Trainer._prune_checkpoints(stub)
        r.append(check("keep_last_n unset prunes nothing",
                       sorted(p.name for p in tmp.iterdir()) == before))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ------------------------------------------------------------- precision has ONE control
    # Left unset, Accelerate reads mixed_precision from `~/.cache/huggingface/accelerate/
    # default_config.yaml`, but only under `accelerate launch` -- so the CLI and the GUI (which
    # always launches through accelerate) ran different numerics from the same config file. With
    # `mixed_precision: bf16` there, SDNQ's training-mode Linear -- fp32 masters, own compiled
    # backward -- died at the first backward() with "expected mat1 and mat2 to have the same
    # dtype". Source-checked because a functional check would need a GPU, and because the failure
    # is an ABSENT argument, which no amount of exercising the object can detect.
    src = (Path(__file__).resolve().parents[1] / "training" / "train.py").read_text()
    r.append(check("the Accelerator pins mixed_precision instead of inheriting it",
                   'mixed_precision="no"' in src))
    r.append(check("  and says so when the environment asked for something else",
                   "ACCELERATE_MIXED_PRECISION" in src))

    # ------------------------------------------------------------- quantized-matmul resolution
    # Pure logic, so it belongs in a CPU gate -- it had none, and the training-mode case below was
    # a live defect: every bucket of a real full-FT run sat above the token crossover, so `auto`
    # said yes, and the run measured 113-178 s/it against 5-13 with it off.
    from anima.training.quant import QMM_TOKEN_CROSSOVER, QuantConfig, resolve_quantized_matmul

    hi = [QMM_TOKEN_CROSSOVER + 500] * 8
    lo = [QMM_TOKEN_CROSSOVER - 500] * 8
    r.append(check("frozen: auto follows the token crossover",
                   resolve_quantized_matmul(QuantConfig(mode="frozen"), hi) is True
                   and resolve_quantized_matmul(QuantConfig(mode="frozen"), lo) is False))
    r.append(check("training: auto is OFF even far above the crossover",
                   resolve_quantized_matmul(QuantConfig(mode="training"), hi) is False,
                   "the crossover was measured on frozen weights and does not transfer"))
    r.append(check("  an explicit true still wins, so the A/B is still runnable",
                   resolve_quantized_matmul(
                       QuantConfig(mode="training", use_quantized_matmul=True), hi) is True))
    r.append(check("  and an explicit false is honoured in frozen mode too",
                   resolve_quantized_matmul(
                       QuantConfig(mode="frozen", use_quantized_matmul=False), hi) is False))
    r.append(check("no tokens known means no quantized matmul",
                   resolve_quantized_matmul(QuantConfig(mode="frozen"), None) is False
                   and resolve_quantized_matmul(QuantConfig(mode="frozen"), []) is False))
    # Weighted by each bucket's share of the WORK, not by sample count. Ninety-nine tiny samples
    # plus one enormous one average to 398 tokens by sample -- below the crossover -- while the big
    # one dominates the wall clock, which is what the decision should turn on.
    lopsided = [200] * 99 + [20000]
    r.append(check("auto weights buckets by wall-clock share, not sample count",
                   resolve_quantized_matmul(QuantConfig(mode="frozen"), lopsided) is True
                   and sum(lopsided) / len(lopsided) < QMM_TOKEN_CROSSOVER,
                   f"sample mean {sum(lopsided)/len(lopsided):.0f} tokens"))

    # ------------------------------------------------------------- keep_last_n, flat layout
    # sd-scripts naming: one file per checkpoint in the run directory. Everything belonging to a
    # checkpoint must go together -- the file, its `_te` sibling and its `-state` directory -- or
    # pruning leaves orphaned optimizer state that is larger than the checkpoints themselves.
    from safetensors.torch import save_file

    tmp = Path(tempfile.mkdtemp(prefix="anima_prune_flat_"))
    try:
        run = "myrun"
        for step in (10, 20, 30):
            save_file({"w": torch.zeros(1)}, tmp / f"{run}-step{step:06d}.safetensors",
                      metadata={"step": str(step), "run": run})
            save_file({"w": torch.zeros(1)}, tmp / f"{run}-step{step:06d}_te.safetensors",
                      metadata={"step": str(step), "run": run})
            state = tmp / f"{run}-step{step:06d}-state"
            state.mkdir()
            (state / "state.json").write_text(json.dumps({"global_step": step}))
        # `final` is the bare run name with no tag, so it must never be a candidate.
        save_file({"w": torch.zeros(1)}, tmp / f"{run}.safetensors", metadata={"step": "30"})
        save_file({"w": torch.zeros(1)}, tmp / "someone_elses.safetensors")
        (tmp / f"{run}.toml").write_text("# the config this run used\n")

        stub = SimpleNamespace(cfg=SimpleNamespace(train=TrainConfig(keep_last_n=2, run_name=run)),
                               out_dir=tmp, steps_per_epoch=100)
        Trainer._prune_checkpoints(stub)
        left = sorted(p.name for p in tmp.iterdir())
        r.append(check("flat layout: oldest checkpoint pruned with ALL of its pieces",
                       not any("step000010" in n for n in left), str(left)))
        r.append(check("  the two newest survive whole",
                       all(f"{run}-step{s:06d}{suf}" in left
                           for s in (20, 30)
                           for suf in (".safetensors", "_te.safetensors", "-state")), str(left)))
        r.append(check("  `final`, the saved TOML and foreign files are never touched",
                       f"{run}.safetensors" in left and f"{run}.toml" in left
                       and "someone_elses.safetensors" in left, str(left)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # --- skip-key policies name real parameters -------------------------------------------------
    # SDNQ's key matching is exact (`check_param_name_in`, sdnq/utils.py:29), so a typo does not
    # raise -- it skips nothing and looks like it worked. That is the failure mode the quant module
    # docstring warns about, so pin the two curated lists against the actual checkpoint. Header-only
    # read: no weights are loaded and no GPU is touched.
    import glob
    import struct

    from anima.training.quant import _ANIMA_INT8_MM, _MLP_DOWN_ALL

    shards = sorted(glob.glob(os.path.join(os.environ.get("ANIMA_MODEL", "../anima-diffusers"),
                                     "transformer", "*.safetensors")))
    if not shards:
        print("  (skipped: converted transformer not present)")
    else:
        shapes = {}
        for f in shards:
            with open(f, "rb") as fh:
                header = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
            shapes.update({k: v["shape"] for k, v in header.items() if k != "__metadata__"})

        for policy, keys, want_params in (("anima_int8_mm", _ANIMA_INT8_MM, 377.5),
                                          ("mlp_down", _MLP_DOWN_ALL, 469.8)):
            missing = [k for k in keys if k not in shapes]
            r.append(check(f"{policy}: every key names a real parameter",
                           not missing, f"missing {missing}"))
            got = sum(shapes[k][0] * shapes[k][1] for k in keys if k in shapes) / 1e6
            r.append(check(f"  and costs the documented {want_params:.1f}M params",
                           abs(got - want_params) < 0.5, f"{got:.1f}M"))

        # The author's list is deliberately narrower than "skip the MLP" -- if that ever stops
        # being true the docstring's whole argument for keeping it separate is gone.
        ff = [k for k in _ANIMA_INT8_MM if ".ff." in k]
        r.append(check("anima_int8_mm is not just the MLP",
                       len(ff) == 19 and len(_ANIMA_INT8_MM) - len(ff) == 14,
                       f"{len(ff)} ff, {len(_ANIMA_INT8_MM) - len(ff)} attn"))
        r.append(check("  and is a strict subset of mlp_down on the down-projections",
                       {k for k in _ANIMA_INT8_MM if k.endswith("ff.net.2.weight")}
                       < set(_MLP_DOWN_ALL)))

    n = sum(r)
    print(f"\n{n}/{len(r)} passed")
    return 0 if n == len(r) else 1


if __name__ == "__main__":
    sys.exit(main())
