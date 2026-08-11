"""Gate for single-file Anima -> diffusers conversion.

    venv/bin/python anima/parity/test_convert_to_diffusers.py

Runs on CPU with synthetic tensors, so it needs no checkpoints. Two of the checks DO use the real
files when they are present, because they are the ones that cannot be faked:

    ANIMA_NATIVE_CKPT   the released anima-base-v1.0.safetensors
    ANIMA_COMFY_VAE     the Qwen-Image VAE in ComfyUI naming
    ANIMA_MODEL         an already-converted repo to compare against

The failure this exists to prevent is not a crash. A rename table that is wrong in one entry
produces a repo that loads, trains, and generates noise -- so the checks are about the mapping
being total and reversible, not about the code running.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from anima.modeling.convert import (  # noqa: E402
    _BLOCK,
    _TOPLEVEL,
    ConversionError,
    text_conditioner_to_native,
    transformer_to_native,
)
from anima.modeling.to_diffusers import (  # noqa: E402
    TEXT_CONDITIONER_CONFIG,
    TRANSFORMER_CONFIG,
    VAE_KEY_MAP,
    convert_text_encoder,
    convert_vae,
    split_native,
)

PASS, FAIL = [], []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")


def _fake_native() -> dict[str, torch.Tensor]:
    """A native state dict covering every rename rule at least once."""
    sd = {}
    for i in range(TRANSFORMER_CONFIG["num_layers"]):
        for native in _BLOCK.values():
            sd[f"net.blocks.{i}.{native}.weight"] = torch.randn(4)
    for native in _TOPLEVEL.values():
        sd[f"net.{native}.weight"] = torch.randn(4)
    for i in range(TEXT_CONDITIONER_CONFIG["num_layers"]):
        sd[f"net.llm_adapter.blocks.{i}.self_attn.q_proj.weight"] = torch.randn(4)
    return sd


def _tokenizers_load(repo: Path) -> bool:
    """Both tokenizers load, as the classes `AnimaModularPipeline` declares in its ComponentSpec.

    Files of the right size in the right place still fail if the class is wrong, and that failure
    would only surface at the first training step.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return False
    want = {"tokenizer": "Qwen2Tokenizer", "t5_tokenizer": "T5Tokenizer"}
    try:
        return all(type(AutoTokenizer.from_pretrained(str(repo / sub))).__name__.startswith(cls)
                   for sub, cls in want.items())
    except Exception:                                  # noqa: BLE001 - a load failure is the answer
        return False


def main() -> int:
    # --- the split is total, and round-trips ------------------------------------------------
    native = _fake_native()
    transformer, conditioner = split_native(native)
    check("split_native maps every key (no silent drops)",
          len(transformer) + len(conditioner) == len(native),
          f"{len(transformer)}+{len(conditioner)} vs {len(native)}")
    check("  the conditioner is separated from the trunk",
          len(conditioner) == TEXT_CONDITIONER_CONFIG["num_layers"] and
          not any(k.startswith("llm_adapter") for k in transformer),
          f"{len(conditioner)} conditioner tensors")

    # The strongest property available without a checkpoint: this converter and `convert.py` are
    # inverses, so native -> diffusers -> native must be the identity. A one-entry error in either
    # table breaks it, and neither direction can drift without the other noticing.
    back = transformer_to_native(transformer) | text_conditioner_to_native(conditioner)
    check("native -> diffusers -> native is the identity",
          set(back) == set(native) and all(torch.equal(back[k], native[k]) for k in native),
          f"{len(set(back) ^ set(native))} keys differ")

    # --- refusals ----------------------------------------------------------------------------
    for bad, why in (
        ({"blocks.0.self_attn.q_proj.weight": torch.zeros(1)}, "keys without the net. prefix"),
        ({"net.blocks.0.nonsense.weight": torch.zeros(1)}, "an unknown block submodule"),
        ({"net.mystery.weight": torch.zeros(1)}, "an unknown top-level module"),
    ):
        try:
            split_native(bad)
            check(f"rejects {why}", False, "accepted")
        except ConversionError:
            check(f"rejects {why}", True)

    # --- text encoder -------------------------------------------------------------------------
    te = convert_text_encoder({"model.embed_tokens.weight": torch.zeros(2, 2),
                               "model.layers.0.mlp.up_proj.weight": torch.zeros(2, 2),
                               "lm_head.weight": torch.zeros(2, 2)})
    check("text encoder strips the `model.` prefix", set(te) ==
          {"embed_tokens.weight", "layers.0.mlp.up_proj.weight"}, str(sorted(te)))
    check("  and drops the LM head (Anima uses Qwen3 as an encoder)", "lm_head.weight" not in te)
    try:
        convert_text_encoder({"transformer.h.0.weight": torch.zeros(1)})
        check("rejects a non-Qwen3 text encoder", False, "accepted")
    except ConversionError:
        check("rejects a non-Qwen3 text encoder", True)

    # --- VAE table --------------------------------------------------------------------------
    check("the VAE table is 1:1 (no two comfy keys collide on one diffusers key)",
          len(set(VAE_KEY_MAP.values())) == len(VAE_KEY_MAP), f"{len(VAE_KEY_MAP)} entries")
    fake_vae = {k: torch.zeros(1) for k in VAE_KEY_MAP}
    check("  and converts a full VAE without loss",
          set(convert_vae(fake_vae)) == set(VAE_KEY_MAP.values()))
    try:
        convert_vae({**fake_vae, "surprise.weight": torch.zeros(1)})
        check("rejects a VAE with unknown keys", False, "accepted")
    except ConversionError:
        check("rejects a VAE with unknown keys", True)
    try:
        convert_vae({k: torch.zeros(1) for k in list(VAE_KEY_MAP)[:-1]})
        check("rejects a VAE missing tensors", False, "accepted")
    except ConversionError:
        check("rejects a VAE missing tensors", True)

    # --- against the real files, when they are present -----------------------------------------
    ckpt = os.environ.get("ANIMA_NATIVE_CKPT", "")
    comfy_vae = os.environ.get("ANIMA_COMFY_VAE", "")
    ref = os.environ.get("ANIMA_MODEL", "../anima-diffusers")

    if comfy_vae and Path(comfy_vae).is_file() and Path(ref, "vae").is_dir():
        # Re-derive the table the way it was originally built -- by content hash -- and demand it
        # matches byte for byte. This is what makes the hardcoded table trustworthy rather than
        # merely plausible: it is checked against the weights, not against someone's reading.
        from safetensors.torch import load_file
        a = load_file(comfy_vae)
        b = load_file(str(Path(ref, "vae", "diffusion_pytorch_model.safetensors")))

        def h(t):
            return hashlib.blake2b(t.contiguous().view(-1).view(torch.uint8).numpy().tobytes(),
                                   digest_size=16).hexdigest()

        by = {h(v): k for k, v in a.items()}
        derived = {by[h(v)]: k for k, v in b.items() if h(v) in by}
        check("the VAE table re-derives exactly from the real weights",
              derived == VAE_KEY_MAP,
              f"{len(set(derived.items()) ^ set(VAE_KEY_MAP.items()))} entries differ")
    else:
        print("  (skipped: set ANIMA_COMFY_VAE and ANIMA_MODEL to re-derive the VAE table)")

    if ckpt and Path(ckpt).is_file() and Path(ref, "transformer").is_dir():
        from safetensors.torch import load_file
        t_new, c_new = split_native(load_file(ckpt))
        t_ref = load_file(str(Path(ref, "transformer", "diffusion_pytorch_model.safetensors")))
        c_ref = load_file(str(Path(ref, "text_conditioner", "diffusion_pytorch_model.safetensors")))
        for label, new, old in (("transformer", t_new, t_ref), ("text_conditioner", c_new, c_ref)):
            same = set(new) == set(old) and all(torch.equal(new[k], old[k]) for k in old)
            check(f"real checkpoint: {label} matches the reference repo bit for bit", same,
                  f"{len(new)} vs {len(old)} tensors")
    else:
        print("  (skipped: set ANIMA_NATIVE_CKPT and ANIMA_MODEL for the real-checkpoint check)")

    # --- the writer refuses to merge into a populated directory ---------------------------------
    from anima.modeling.to_diffusers import convert_to_diffusers
    tmp = Path(tempfile.mkdtemp(prefix="anima_conv_"))
    try:
        (tmp / "stale.txt").write_text("something already here")
        try:
            convert_to_diffusers("a", "b", "c", tmp)
            check("refuses to write into a non-empty directory", False, "accepted")
        except FileExistsError:
            check("refuses to write into a non-empty directory", True)
        except FileNotFoundError:
            # Input validation runs first, which is also fine -- but then the emptiness guard is
            # untested, so say so rather than counting a pass.
            check("refuses to write into a non-empty directory", False,
                  "input check ran first; guard untested")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # --- the tokenizer warning ----------------------------------------------------------------
    # The one part of the conversion whose ABSENCE is the failure. A repo without tokenizers holds
    # every weight and lists correctly, then dies inside `load_components` -- after loading the
    # whole model -- with huggingface_hub's "Repo id must be in the form ...", which names neither
    # the problem nor the fix. This trainer never caches text embeddings (tag shuffling and
    # caption dropout change the caption every epoch), so both tokenizers run on every step.
    from anima.modeling.to_diffusers import (
        DEFAULT_TOKENIZER_REPO,
        TOKENIZER_DIRS,
        TOKENIZER_NONE,
        _looks_like_repo_id,
        copy_tokenizers,
    )

    # A local directory must always beat a repo-id reading, or someone holding a real `./org/model`
    # checkout silently gets a download instead of their own files.
    check("a hub repo id is told apart from a path",
          _looks_like_repo_id("Bluvoll/Anima-v1.0-Base-Diffusers")
          and not _looks_like_repo_id("/abs/path")
          and not _looks_like_repo_id("./rel/path")
          and not _looks_like_repo_id("a/b/c")
          and not _looks_like_repo_id("."))
    real_dir = Path(tempfile.mkdtemp(prefix="anima_org_")) / "org" / "model"
    real_dir.mkdir(parents=True)
    try:
        check("  and an existing directory shaped like a repo id is treated as a path",
              not _looks_like_repo_id(str(real_dir.relative_to(Path.cwd()))
                                      if real_dir.is_relative_to(Path.cwd()) else real_dir))
    finally:
        shutil.rmtree(real_dir.parents[1], ignore_errors=True)

    tmp = Path(tempfile.mkdtemp(prefix="anima_tok_"))
    try:
        msgs: list[str] = []
        copied = copy_tokenizers(tmp, TOKENIZER_NONE, msgs.append)
        check("'none' -> nothing copied, and it says so loudly",
              copied == [] and any("CANNOT TRAIN" in m for m in msgs), str(msgs)[:90])
        check("  the warning explains WHY they are needed, not just that they are missing",
              any("not cached" in m and "every step" in m for m in msgs))

        src = Path(tempfile.mkdtemp(prefix="anima_toksrc_"))
        try:
            for name in TOKENIZER_DIRS:
                (src / name).mkdir()
                (src / name / "tokenizer_config.json").write_text("{}")
            msgs.clear()
            copied = copy_tokenizers(tmp, src, msgs.append)
            check("both tokenizers copied from a source repo",
                  sorted(copied) == sorted(TOKENIZER_DIRS)
                  and all((tmp / n / "tokenizer_config.json").is_file() for n in TOKENIZER_DIRS),
                  str(copied))
            check("  and then there is no warning", not any("CANNOT TRAIN" in m for m in msgs))

            # Half a source is the nastiest case: it looks like it worked.
            shutil.rmtree(tmp)
            tmp.mkdir()
            shutil.rmtree(src / "t5_tokenizer")
            msgs.clear()
            copied = copy_tokenizers(tmp, src, msgs.append)
            check("a source missing one tokenizer still warns",
                  copied == ["tokenizer"] and any("CANNOT TRAIN" in m for m in msgs), str(copied))
            check("  and names the one that is absent",
                  any("t5_tokenizer" in m and "CANNOT TRAIN" in m for m in msgs))
        finally:
            shutil.rmtree(src, ignore_errors=True)

        # An unreachable repo must not kill a conversion that has already read several GB. It
        # degrades to the same warning as "no source", which names the fix.
        shutil.rmtree(tmp)
        tmp.mkdir()
        msgs.clear()
        try:
            copied = copy_tokenizers(tmp, "anima-trainer-test/definitely-not-a-real-repo",
                                     msgs.append)
            check("an unreachable repo warns instead of raising",
                  copied == [] and any("CANNOT TRAIN" in m for m in msgs)
                  and any("could not fetch" in m for m in msgs), str(msgs)[-90:])
        except Exception as exc:                       # noqa: BLE001 - that is the failure
            check("an unreachable repo warns instead of raising", False,
                  f"raised {type(exc).__name__}")

        # The default path, when there is a network (or a warm cache). Skipped rather than failed
        # offline, since the point of the cache is that this works offline *after* one fetch.
        shutil.rmtree(tmp)
        tmp.mkdir()
        msgs.clear()
        copied = copy_tokenizers(tmp, None, msgs.append)
        if copied:
            check(f"default fetches both tokenizers from {DEFAULT_TOKENIZER_REPO}",
                  sorted(copied) == sorted(TOKENIZER_DIRS)
                  and all((tmp / n / "tokenizer.json").stat().st_size > 1_000_000
                          for n in TOKENIZER_DIRS),
                  str(copied))
            check("  and loads as the classes the pipeline declares", _tokenizers_load(tmp))
        elif any("could not fetch" in m for m in msgs):
            print("  (skipped: no network and no cached copy, so the default fetch is untested)")
        else:
            # The distinction matters: a silent no-op here is exactly the `str(None) == "none"`
            # bug that made the default download nothing while reporting no error at all.
            check("default fetches both tokenizers", False,
                  "copied nothing and reported no fetch failure -- the default did not even try")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} passed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
