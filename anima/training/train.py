"""Anima training loop: full finetune or LoRA/LoKr, single GPU or Accelerate DDP.

    .venv/bin/accelerate launch --num_processes 2 -m anima.training.train configs/lora.toml

Two structural decisions worth stating up front.

**Text embeddings are computed every step, not cached.** Tag shuffling and dropout produce a
different caption for the same image on every epoch, so a cached embedding would be wrong. The
cost is one frozen 0.6B forward per step -- which is precisely the trade the LLMAdapter exists to
make cheap, since the alternative architecture would be a 4.7B T5-XXL forward instead.

**Bucketing is enforced end to end.** The sampler emits bucket-homogeneous batches, `collate`
re-checks it, and under DDP every rank draws from the same seeded batch list, so ranks stay in
lockstep on step count. That last part is not cosmetic: a rank that runs out of batches early
deadlocks the all-reduce rather than failing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

# Must precede `import torch`: the caching allocator parses this once, at first CUDA use.
# Bucketing hands the allocator a different tensor shape per bucket (55 of them on a two-tier
# texture run), which is the textbook way to fragment it -- measured 3.35 GB of reserved-but-
# unallocated on a shape-varying probe, versus 0.74 GB with expandable segments. That gap is the
# difference between fitting and OOM at the batch sizes this trainer targets. `setdefault` so an
# explicitly-set value always wins.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.dataset import AnimaDataset, BucketBatchSampler, collate
from ..modeling.convert import save_lora_checkpoint, save_native_checkpoint
from ..modeling.loader import load_components
from .config import Config, load_config
from .flow import flow_loss, hf_loss, prepare_flow_batch
from .optim import build_optimizer, build_scheduler, estimate_optimizer_bytes
from .params import (
    apply_adapter,
    build_adapter_param_groups,
    build_param_groups,
    count_parameters,
)
from .quant import (
    QMM_TOKEN_CROSSOVER,
    dequantize_state_dict,
    quantize_module,
    quantized_layer_report,
    resolve_quantized_matmul,
    straddles_crossover,
    tokens_for_bucket,
)

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _peft_state_dict(module) -> dict[str, torch.Tensor]:
    """Adapter tensors only. `get_peft_model_state_dict` is not usable here because diffusers'
    `add_adapter` injects layers without wrapping the module in a PeftModel."""
    return {k: v for k, v in module.state_dict().items() if "lora_" in k or "lokr_" in k}


def _emit(obj, msg: str) -> None:
    """Print without tearing the progress bar.

    Module-level and `getattr`-guarded on purpose: `_prune_checkpoints` is exercised by the gate
    against a duck-typed stub that has no `bar`, and a bound method would make that stub carry
    display state it has nothing to do with.
    """
    if getattr(obj, "bar", None) is not None:
        tqdm.write(msg)
    else:
        print(msg, flush=True)


def _fmt_duration(seconds: float) -> str:
    s = int(max(0.0, seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


UNCONDITIONAL = " "


def substitute_empty(captions: list[str], filler: str = UNCONDITIONAL) -> list[str]:
    """Replace captions that would tokenize to nothing. See `Trainer._encode` for why."""
    return [c if c.strip() else filler for c in captions]


def require_cuda() -> None:
    """Refuse to start on CPU.

    Torch does not fail on a bad `CUDA_VISIBLE_DEVICES`; it reports zero devices, Accelerate falls
    back to gloo, and a 2B model trains on the CPU at roughly a thousandth of the speed while
    printing a completely normal-looking startup summary. That is the worst failure mode available
    -- it looks like it is working.

    Observed with `CUDA_VISIBLE_DEVICES=all`, which is not a device list. The message names the
    variable and its value because that is nearly always what is wrong.
    """
    import os

    if torch.cuda.is_available():
        return
    if os.environ.get("ANIMA_ALLOW_CPU"):
        print("WARNING: no CUDA device; continuing because ANIMA_ALLOW_CPU is set.", flush=True)
        return

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    detail = ""
    if cvd is not None:
        detail = (f"\nCUDA_VISIBLE_DEVICES is set to {cvd!r}. It must be a comma-separated list of "
                  f"device indices (e.g. '0' or '0,1'); anything else -- including 'all' and an "
                  f"empty string -- hides every GPU.")
    raise RuntimeError(
        "no CUDA device is visible, so this would train on the CPU." + detail +
        "\nSet ANIMA_ALLOW_CPU=1 to override."
    )


class Trainer:
    def __init__(self, cfg: Config, config_path: str | Path | None = None):
        require_cuda()
        self.cfg = cfg
        self.config_path = Path(config_path) if config_path else None
        self.dtype = _DTYPES[cfg.train.dtype]

        # `mixed_precision="no"` is explicit, not a default being restated. Left unset, Accelerate
        # reads `~/.cache/huggingface/accelerate/default_config.yaml`, and a `mixed_precision: bf16`
        # there silently wraps every forward in autocast -- but ONLY under `accelerate launch`, so
        # `python -m anima.training.train` and the GUI (which always launches through accelerate)
        # would run different numerics from the same config file.
        #
        # That is not merely redundant here -- every module is already loaded in `train.dtype`, so
        # autocast has nothing to convert -- it is broken. SDNQ's training-mode Linear keeps fp32
        # master weights and supplies its own compiled backward; under autocast that backward gets
        # a bf16 grad_output against an fp32 input and dies inside inductor with "expected mat1 and
        # mat2 to have the same dtype, but got: c10::BFloat16 != float", at the first
        # `backward()`. Reproduced on one GPU with `accelerate launch`, and fixed by
        # `--mixed_precision no`.
        #
        # Precision is `train.dtype`'s job. Making that the single control is what keeps the CLI
        # and the GUI running the same arithmetic.
        env_mp = os.environ.get("ACCELERATE_MIXED_PRECISION")
        if env_mp and env_mp.lower() != "no":
            print(f"note: ignoring mixed_precision={env_mp!r} from the accelerate environment; "
                  f"this trainer sets precision through train.dtype (={cfg.train.dtype}) and "
                  f"autocast breaks SDNQ's training-mode backward.", flush=True)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
            log_with=None,
            mixed_precision="no",
            dynamo_plugin=self._dynamo_plugin(),
        )
        # `device_specific=True` adds the process index to the seed, so ranks do not start from an
        # identical torch RNG state. Without it every rank calls `torch.manual_seed(seed)` with the
        # same value, and since the timestep draw is shape (B,) regardless of image shape, rank 0
        # and rank 1 sample the SAME timestep wherever their streams are in step -- always at step
        # 1, and thereafter whenever the two ranks happen to draw the same texture canvas (~22% of
        # batches with the default preset list, since a matching shape consumes the stream
        # identically and keeps them synchronised).
        #
        # This is the same defect the caption RNG was fixed for -- every DDP rank forking from one
        # seed -- missed in the noise/timestep path. Single-GPU behaviour is unchanged: with one
        # process the index is 0, so the seed is identical to before and existing runs reproduce.
        set_seed(cfg.train.seed, device_specific=True)

        self.out_dir = Path(cfg.train.output_dir) / cfg.train.run_name
        if self.accelerator.is_main_process:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            # The config that produced these checkpoints, copied verbatim so its comments survive.
            # Checkpoints outlive the configs directory -- a file gets edited for the next
            # experiment and the run that produced a LoRA becomes unreconstructable. Copied rather
            # than re-serialised for exactly that reason: a round trip through the dataclasses
            # would emit the resolved values and drop every comment explaining them.
            if self.config_path and self.config_path.is_file():
                dest = self.out_dir / f"{cfg.train.run_name}.toml"
                if self.config_path.resolve() != dest.resolve():
                    shutil.copy2(self.config_path, dest)

        # None when no curriculum is configured, which is the default and is exactly today's
        # behaviour: `t_range=None` reaches the sampler and no range mapping happens at all.
        self.phase = None
        # NCCL has no Windows build, so torch falls back to gloo, which does not support the GPU
        # collectives DDP needs here. Caught with a sentence rather than left to surface as a
        # backend error from inside `accelerator.prepare`.
        if self.accelerator.num_processes > 1 and sys.platform == "win32":
            raise RuntimeError(
                f"multi-GPU training is not available on Windows: DDP requires NCCL, which is "
                f"Linux-only, and gloo cannot carry the GPU collectives. Got "
                f"{self.accelerator.num_processes} processes.\n"
                f"Run one process and pick a device with CUDA_VISIBLE_DEVICES=<index>."
            )
        # Refused rather than warned: a texture curriculum under DDP produced progressively
        # worsening anatomy in two runs, already worse than single-GPU at matched step 480. A
        # warning printed at startup scrolls away in the first few seconds of a 25-minute run,
        # which is exactly how those two were lost. Checked here rather than in `load_config`
        # because only the Accelerator knows the process count.
        if (self.accelerator.num_processes > 1
                and any(p.mode == "texture" for p in cfg.curriculum.phases)
                and not cfg.train.allow_multi_gpu_texture):
            raise ValueError(
                f"texture curricula are not supported on {self.accelerator.num_processes} "
                f"processes.\n\nTwo runs here degraded anatomy progressively against a single-GPU "
                f"run at MATCHED optimizer steps, so this is not the halved step count and more "
                f"epochs do not fix it. The cause is not identified -- ranks seeding identically "
                f"was found and fixed, but was never shown to be the mechanism.\n\n"
                f"Run on one process (CUDA_VISIBLE_DEVICES=<one gpu>, no accelerate launch), or "
                f"set train.allow_multi_gpu_texture = true to re-test deliberately against a "
                f"single-GPU control."
            )
        self._build_data()
        self._build_model()
        self._build_optimizer()
        self._prepare()

        self.global_step = 0
        self.start_epoch = 0
        if cfg.train.resume_from:
            self._resume(cfg.train.resume_from)

    def _dynamo_plugin(self):
        """Accelerate applies this inside `prepare`, so compilation happens *under* the DDP wrapper
        rather than around it -- the ordering torch documents for distributed runs."""
        from accelerate.utils import TorchDynamoPlugin

        cfg = self.cfg.train
        if cfg.compile is None:
            return TorchDynamoPlugin(backend="no")

        # Aspect-ratio bucketing hands the trunk a different sequence length per bucket, and dynamo
        # recompiles per shape up to `recompile_limit` (8) before silently falling back to eager for
        # the remainder of the run. Dynamic shapes are what keep that from happening; the limit is
        # raised as well so a partially-specialised graph still has room.
        import torch._dynamo

        if cfg.compile_dynamic:
            torch._dynamo.config.recompile_limit = max(
                torch._dynamo.config.recompile_limit, 32
            )
        return TorchDynamoPlugin(
            backend="inductor",
            mode=cfg.compile,
            dynamic=cfg.compile_dynamic,
            fullgraph=False,          # the trunk has graph breaks we do not control
            use_regional_compilation=cfg.compile_regional,
        )

    # ---------------------------------------------------------------- data

    def _build_data(self) -> None:
        cfg = self.cfg
        self.dataset = AnimaDataset(cfg.dataset, caption_seed=cfg.train.seed)

        self.sampler = BucketBatchSampler(
            self.dataset,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            drop_last=False,
            seed=cfg.train.seed,
        )
        if cfg.curriculum:
            self.sampler.set_curriculum(cfg.curriculum, cfg.train.epochs, cfg.dataset.texture)
        self.loader = DataLoader(
            self.dataset,
            batch_sampler=self.sampler,
            collate_fn=collate,
            num_workers=cfg.train.num_workers,
            pin_memory=True,
            # Workers must be re-forked each epoch: the sampler's epoch is read at __iter__ time,
            # and a persistent worker would keep serving batches built from a stale epoch.
            persistent_workers=False,
        )

        # `self.sampler` is unsharded; Accelerate's BatchSamplerShard splits it across ranks and
        # pads with even_batches, so each rank sees ceil(total / num_processes) batches. The
        # trailing partial accumulation group still steps (end_of_dataloader sync), hence ceil()
        # again -- nothing is dropped.
        total_batches = len(self.sampler)
        per_rank = math.ceil(total_batches / self.accelerator.num_processes)
        steps_per_epoch = math.ceil(per_rank / cfg.train.gradient_accumulation_steps)

        self.total_batches = total_batches
        self.total_steps = cfg.train.max_steps or steps_per_epoch * cfg.train.epochs
        self.steps_per_epoch = steps_per_epoch

        # Progress display. The bar is main-process-only: under DDP every rank runs this loop, and
        # N ranks writing \r to the same terminal is unreadable.
        mode = cfg.train.progress
        if mode == "auto":
            mode = "bar" if sys.stdout.isatty() else "plain"
        self.progress_mode = mode if self.accelerator.is_main_process else "off"
        self.bar: tqdm | None = None
        # (wall clock, global_step) taken *after* the first step of the run, so the ETA average
        # excludes torch.compile and the first dataloader fork -- both one-off costs that would
        # otherwise inflate every estimate for the rest of the run.
        self._eta_anchor: tuple[float, int] | None = None

    # --------------------------------------------------------------- model

    def _build_model(self) -> None:
        cfg = self.cfg
        needs_te = True  # captions are dynamic, so Qwen3 runs every step
        # Normally latents are cached and the VAE is 243MB of dead weight. Under
        # `dataset.source = "encode"` it is the whole point: pixels arrive per step and have to be
        # encoded before anything else can happen.
        needs_vae = cfg.dataset.source == "encode"

        self.components = load_components(
            cfg.train.model_path,
            dtype=self.dtype,
            load_text_encoder=needs_te,
            load_vae=needs_vae,
            load_tokenizers=True,
        )
        self.transformer = self.components.transformer
        self.text_conditioner = self.components.text_conditioner
        self.text_encoder = self.components.text_encoder

        # Reuse LatentCacher rather than re-deriving the encode: it owns the `(x - mean) / std`
        # normalisation, so an encode-on-the-fly run is numerically the same as a cached one.
        self.latent_encoder = None
        if needs_vae:
            from ..data.cache import LatentCacher
            self.latent_encoder = LatentCacher(
                self.components.vae, device=self.accelerator.device, dtype=self.dtype
            )
            self.components.vae.requires_grad_(False)

        # The hf loss tokenizes exactly as the model patchifies, so the patch size is read off the
        # model rather than configured -- a mismatch would weight the wrong spatial extent.
        # patch_size is (T, H, W); T is 1 for images and the spatial dims are square.
        self.hf_patch = None
        if cfg.flow.hf_scale > 0.0:
            _, ph, pw = self.transformer.config.patch_size
            if ph != pw:
                raise ValueError(f"hf loss assumes a square patch, got {ph}x{pw}")
            self.hf_patch = ph

        if cfg.train.gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()
            self.text_conditioner.enable_gradient_checkpointing()

        self._quantize()

        if cfg.is_lora:
            apply_adapter(
                self.transformer, self.text_conditioner, cfg.adapter, self.text_encoder
            )
            # Per-component LRs apply to adapters too. A single group across the trunk *and* a
            # Qwen3 adapter is the specific case that goes wrong quietly -- a text encoder trained
            # at the trunk's LR is the classic way to fry a LoRA, which is why sd-scripts has
            # always exposed `text_encoder_lr` separately.
            self.param_report = build_adapter_param_groups(
                self.transformer,
                self.text_conditioner if cfg.adapter.train_llm_adapter else None,
                self.text_encoder if cfg.adapter.train_text_encoder else None,
                cfg.component_lr,
                default_lr=cfg.optimizer.lr,
                weight_decay=cfg.optimizer.weight_decay,
            )
            groups = self.param_report.groups
        else:
            self.param_report = build_param_groups(
                self.transformer,
                self.text_conditioner,
                cfg.component_lr,
                default_lr=cfg.optimizer.lr,
                weight_decay=cfg.optimizer.weight_decay,
            )
            groups = self.param_report.groups
            self.text_encoder.requires_grad_(False)

        self.groups = groups
        self.train_text_encoder = cfg.is_lora and cfg.adapter.train_text_encoder

    def _quantize(self) -> None:
        """Quantize before adapters are injected, so LoRA layers wrap quantized bases rather than
        being quantized themselves."""
        qcfg = self.cfg.quant
        if qcfg.mode == "none":
            self.quant_info = None
            return

        device = self.accelerator.device
        # One token count per sample, so `auto` weights buckets by how much of the run they are,
        # not by which is biggest. Under a multi-resolution ladder those differ sharply.
        tokens = [
            tokens_for_bucket(b)
            for b, idx in self.sampler.bucket_indices.items()
            for _ in idx
        ]
        max_tokens = max(tokens)
        use_qmm = resolve_quantized_matmul(qcfg, tokens)

        # The crossover only governs frozen weights, so a straddle warning in training mode would
        # point at a threshold that no longer decides anything.
        below, above = straddles_crossover(tokens) if qcfg.mode != "training" else (0, 0)
        if below and above:
            import warnings

            warnings.warn(
                f"buckets straddle the quantized-matmul crossover ({QMM_TOKEN_CROSSOVER} tokens): "
                f"{below} sample(s) below, {above} at or above. `use_quantized_matmul` is one flag "
                f"for the whole model, so whichever way it resolves ({'ON' if use_qmm else 'OFF'} "
                f"here) is wrong for the other side -- roughly 2x slow if ON below the crossover, "
                f"~1.35x slow if OFF above it. Narrow the resolution ladder to avoid the split."
            )

        self.transformer = quantize_module(self.transformer, qcfg, device, self.dtype, use_qmm)
        if qcfg.quantize_text_conditioner:
            self.text_conditioner = quantize_module(
                self.text_conditioner, qcfg, device, self.dtype, use_qmm
            )
        if qcfg.quantize_text_encoder:
            self.text_encoder = quantize_module(
                self.text_encoder, qcfg, device, self.dtype, use_qmm
            )

        nq, ntotal, _ = quantized_layer_report(self.transformer)
        wmean = sum(t * t for t in tokens) / sum(tokens)
        self.quant_info = (nq, ntotal, use_qmm, max_tokens, wmean)

    def _all_modules_params(self):
        for m in (self.transformer, self.text_conditioner, self.text_encoder):
            if m is not None:
                yield from m.parameters()

    # ----------------------------------------------------------- optimizer

    def _build_optimizer(self) -> None:
        cfg = self.cfg
        self.optimizer = build_optimizer(self.groups, cfg.optimizer)
        # Accelerate's scheduler wrapper advances the inner scheduler `num_processes` times per
        # `step()` call (accelerate/scheduler.py: it assumes the dataloader batch was multiplied by
        # world size). Our horizon is counted in optimizer steps, so without this the schedule
        # burns through `num_processes`x too fast -- on 2 GPUs a cosine decay reached its floor at
        # the halfway point and trained the rest of the run at lr 0.
        scale = self.accelerator.num_processes
        schedule = replace(cfg.schedule, warmup_steps=cfg.schedule.warmup_steps * scale)
        self.scheduler = build_scheduler(self.optimizer, schedule, self.total_steps * scale)

    def _prepare(self) -> None:
        # The text encoder is prepared only when it has trainable parameters; wrapping a frozen
        # module in DDP would allocate gradient buckets for weights that never receive gradients.
        modules = [self.transformer, self.text_conditioner]
        if self.train_text_encoder:
            modules.append(self.text_encoder)
        else:
            self.text_encoder.to(self.accelerator.device).eval()

        prepared = self.accelerator.prepare(*modules, self.optimizer, self.scheduler, self.loader)
        *modules, self.optimizer, self.scheduler, self.loader = prepared
        self.transformer, self.text_conditioner = modules[0], modules[1]
        if self.train_text_encoder:
            self.text_encoder = modules[2]

    # ----------------------------------------------------------- one step

    # Qwen3 tokenizes "" to ZERO tokens -- not to a BOS/EOS pair. Padded to max_length that gives
    # an all-zero attention mask, `hidden` is then masked_fill'd to all zeros, and the conditioner
    # is asked to attend over nothing. It happens not to NaN here, but it is the degenerate corner
    # of the attention kernel and nothing guarantees that holds across versions.
    #
    # Three separate paths produce an empty caption: `caption_dropout_percent` returns "" by
    # design (10% of every batch in the shipped configs), a texture phase with no trigger, and an
    # empty tag file. TrainTrain hits none of them -- it substitutes a single space, `(trigger or
    # " ")` -- so a run here was also diverging from the reference on ~10% of its fullres samples.
    # Substituted centrally rather than at each call site so a future caption path cannot
    # reintroduce it. Measured: emb norm 6.73 for "" against 5.85 for " ", cosine 0.969.
    UNCONDITIONAL = UNCONDITIONAL

    def _encode(self, captions: list[str]) -> torch.Tensor:
        captions = substitute_empty(captions)
        device = self.accelerator.device
        tok = self.components.tokenizer(
            captions, return_tensors="pt", truncation=True,
            padding="max_length", max_length=512,
        )
        t5 = self.components.t5_tokenizer(
            captions, return_tensors="pt", truncation=True,
            padding="max_length", max_length=512,
        )
        qwen_mask = tok.attention_mask.to(device)

        with torch.set_grad_enabled(self.train_text_encoder):
            hidden = self.text_encoder(
                input_ids=tok.input_ids.to(device), attention_mask=qwen_mask
            ).last_hidden_state
        # Zero the padded rows: they carry no text and must not leak into cross-attention.
        hidden = hidden.masked_fill(~qwen_mask.bool().unsqueeze(-1), 0.0)

        return self.text_conditioner(
            source_hidden_states=hidden.to(self.dtype),
            target_input_ids=t5.input_ids.to(device),
            target_attention_mask=t5.attention_mask.to(device),
            source_attention_mask=qwen_mask,
        )

    def _encode_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        """Pixels -> normalised latents, in chunks, with nothing left alive afterwards.

        Chunked so the encode's transient activations are bounded by `vae_encode_chunk` rather than
        by the training batch size. Because this runs under `no_grad` and everything but the (tiny)
        latents is freed before the transformer forward begins, the step's peak is
        `max(train_peak, encode_peak)` -- not their sum. That is the whole reason encode-on-the-fly
        does not force the batch size down to a couple of images.
        """
        chunk = max(1, self.cfg.train.vae_encode_chunk)
        out = [
            self.latent_encoder.encode_tensor(pixels[i:i + chunk])
            for i in range(0, pixels.shape[0], chunk)
        ]
        return torch.cat(out).float()

    def _step(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor | None]:
        device = self.accelerator.device
        if "pixels" in batch:
            latents = self._encode_pixels(batch["pixels"].to(device, self.dtype))
        else:
            latents = batch["latents"].to(device, torch.float32)

        stats: dict = {}
        noisy, timesteps, target = prepare_flow_batch(
            latents, self.cfg.flow, stats=stats, t_range=self.phase.t_range if self.phase else None
        )
        self.last_stats = stats
        encoder_hidden_states = self._encode(batch["captions"])

        bucket_w, bucket_h = batch["bucket"]
        # (1, 1, H, W) in *pixels*: the transformer resizes it and repeats over the batch itself.
        padding_mask = noisy.new_zeros(1, 1, bucket_h, bucket_w, dtype=self.dtype)

        pred = self.transformer(
            hidden_states=noisy.to(self.dtype),
            timestep=timesteps.to(self.dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.dtype),
            padding_mask=padding_mask,
            return_dict=False,
        )[0]

        # (B,1,h,w) -> (B,1,1,h,w) to broadcast over the latent's channel and time axes. Texture
        # batches supervise only a feathered sub-region; the rest is real context that conditions
        # the forward pass but carries no target.
        mask = batch.get("mask")
        if mask is not None:
            mask = mask.to(device, torch.float32).unsqueeze(2)

        loss = flow_loss(pred, target, mask=mask)
        if self.hf_patch is None:
            return loss, None

        # Reported separately and already scaled, so the log decomposes as total = mse + hf.
        # The mask goes through here too: weighting unsupervised context by its detail content
        # would put a target back on exactly the region the mask exists to exclude.
        hf = self.cfg.flow.hf_scale * hf_loss(
            pred, noisy, latents, timesteps, self.hf_patch, self.cfg.flow.hf_exponent, mask=mask
        )
        return loss + hf, hf

    # ------------------------------------------------------------ curriculum

    def _set_phase(self) -> None:
        if not self.cfg.curriculum:
            return
        prev = self.phase
        self.phase = self.cfg.curriculum.resolve(self.global_step / max(1, self.total_steps))
        if prev is not self.phase and self.accelerator.is_main_process:
            _emit(self, f"phase   {self.phase.label()}  @ step {self.global_step}")

    def _apply_lr_mul(self) -> None:
        """Scale the scheduler's current LR by the phase multiplier.

        Set absolutely from `get_last_lr()` rather than multiplied in place: this runs on every
        iteration but the scheduler only advances on sync boundaries, so an in-place multiply
        would compound `lr_mul` once per accumulation micro-step -- silently turning lr_mul=0.5
        into 0.5^accum.
        """
        if not self.phase or self.phase.lr_mul == 1.0:
            return
        last = getattr(self.scheduler, "get_last_lr", None)
        if last is None:
            return
        for pg, lr in zip(self.optimizer.param_groups, last()):
            pg["lr"] = lr * self.phase.lr_mul

    # ---------------------------------------------------------------- run

    def train(self) -> None:
        cfg = self.cfg
        acc = self.accelerator
        self._report()

        trainable = [p for p in self._all_modules_params() if p.requires_grad]
        done = False

        if self.progress_mode == "bar":
            self.bar = tqdm(
                total=self.total_steps, initial=self.global_step, unit="step",
                dynamic_ncols=True, smoothing=0.05,
            )

        for epoch in range(self.start_epoch, cfg.train.epochs):
            self.sampler.set_epoch(epoch)
            self.transformer.train()
            self.text_conditioner.train()

            t0 = time.time()
            for batch in self.loader:
                # Resolved before the step so the phase's t_range reaches `_step`. Progress is a
                # pure function of the step count, so every DDP rank picks the same phase without
                # communicating, and a resumed run lands in the right phase on its own.
                self._set_phase()
                with acc.accumulate(self.transformer):
                    loss, hf = self._step(batch)
                    acc.backward(loss)

                    if acc.sync_gradients and cfg.optimizer.max_grad_norm > 0:
                        acc.clip_grad_norm_(trainable, cfg.optimizer.max_grad_norm)

                    self._apply_lr_mul()
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                if not acc.sync_gradients:
                    continue

                self.global_step += 1
                if self.bar is not None:
                    self.bar.update(1)
                if self._eta_anchor is None:
                    self._eta_anchor = (time.time(), self.global_step)
                if self.global_step % cfg.train.log_every == 0:
                    self._log(loss, hf, epoch, t0)
                    t0 = time.time()

                if cfg.train.save_every_steps and self.global_step % cfg.train.save_every_steps == 0:
                    self.save(f"step{self.global_step:06d}")

                if self.global_step >= self.total_steps:
                    done = True
                    break

            if cfg.train.save_every_epochs and (epoch + 1) % cfg.train.save_every_epochs == 0:
                self.save(f"epoch{epoch + 1:03d}")
            if done:
                break

        if self.bar is not None:
            self.bar.close()
            self.bar = None

        if not cfg.train.skip_final_save:
            self.save("final")
        acc.end_training()

    def _log(
        self, loss: torch.Tensor, hf: torch.Tensor | None, epoch: int, t0: float
    ) -> None:
        acc = self.accelerator
        value = acc.gather(loss.detach().float().repeat(1)).mean().item()
        # gather is collective, so it must run on every rank or the non-main ranks deadlock.
        hf_value = (
            acc.gather(hf.detach().float().repeat(1)).mean().item() if hf is not None else None
        )

        # Peak memory must be gathered, not read locally: ranks are NOT symmetric. Rank 0 carries
        # NCCL buffers and does the checkpoint export, so reporting only its number both overstates
        # the typical rank and hides which rank is actually the batch-size ceiling.
        local_peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        peaks = acc.gather(
            torch.tensor([local_peak], device=acc.device, dtype=torch.float64)
        ).tolist()

        if not acc.is_main_process or self.progress_mode == "off":
            return
        # Every group's LR, not just group 0's. With `[component_lr]` set, printing `get_last_lr()[0]`
        # showed only the first component -- so a run with self_attn 4e-5 / cross_attn 1e-5 /
        # mlp 8e-6 reported a flat 4e-5 and looked exactly like a run where per-component LR had
        # silently not applied. It had; only the log was blind. Slash-joined to match the `peak
        # A/B GB` convention already used for per-rank memory, and collapsed to one number when the
        # groups agree, so the common case reads as before. The startup report names the groups in
        # this same order.
        lrs = self.scheduler.get_last_lr()
        lr = "/".join(f"{v:.2e}" for v in lrs) if len(set(lrs)) > 1 else f"{lrs[0]:.2e}"
        dt = (time.time() - t0) / max(1, self.cfg.train.log_every)
        mem = (
            "/".join(f"{p / 1e9:.1f}" for p in peaks) + "GB"
            if len(peaks) > 1
            else f"{peaks[0] / 1e9:.1f}GB"
        )
        # total = mse + hf, so the two printed numbers add up to the first.
        parts = f"loss {value:.4f}"
        if hf_value is not None:
            parts += f" (mse {value - hf_value:.4f} + hf {hf_value:.4f})"
        # Fraction of rows optimal transport actually reordered. 0.00 means it is doing nothing --
        # either the batch is 1, or the pairing happened to be optimal already.
        if "ot_moved" in getattr(self, "last_stats", {}):
            parts += f"  ot {self.last_stats['ot_moved']:.2f}"

        if self.bar is not None:
            # tqdm already draws step/total, percentage, elapsed and its own ETA, so the postfix
            # carries only what it cannot know.
            self.bar.set_description_str(f"e{epoch}")
            self.bar.set_postfix_str(f"{parts}  lr {lr}  peak {mem}", refresh=True)
            return

        eta = self._eta()
        print(
            f"e{epoch} step {self.global_step}/{self.total_steps} ({self.progress_pct():.0f}%)  "
            f"{parts}  lr {lr}  {dt:.2f}s/it  peak {mem}"
            + (f"  eta {eta}" if eta else ""),
            flush=True,
        )

    # ---------------------------------------------------------------- progress

    def progress_pct(self) -> float:
        return 100.0 * self.global_step / max(1, self.total_steps)

    def _eta(self) -> str | None:
        """Remaining wall clock, from the mean step time since the anchor.

        Deliberately a cumulative mean rather than the instantaneous `dt`: with bucketing -- and
        far more so with multi-resolution -- step cost swings with the bucket that happened to come
        up, so an ETA off the last interval would jump by several-fold between log lines. The mean
        over many steps sees the bucket mix in its true proportion. Returns None until at least one
        step has elapsed past the anchor.
        """
        if self._eta_anchor is None:
            return None
        t_anchor, s_anchor = self._eta_anchor
        done = self.global_step - s_anchor
        remaining = self.total_steps - self.global_step
        if done <= 0 or remaining <= 0:
            return None
        return _fmt_duration((time.time() - t_anchor) / done * remaining)

    def _report(self) -> None:
        if not self.accelerator.is_main_process:
            return
        cfg = self.cfg
        trainable, total = count_parameters(
            self.transformer, self.text_conditioner,
            self.text_encoder if self.train_text_encoder else None,
        )
        print(f"\ndataset  {len(self.dataset)} samples, {len(self.sampler.bucket_indices)} buckets")
        subsets = getattr(self.dataset, "subset_report", [])
        if len(subsets) > 1:
            # Worth printing even though it is configured: a subset's share of the epoch is
            # `entries x num_repeats`, not its file count, and that is the number that decides how
            # hard a regularization set actually pulls.
            total = max(1, len(self.dataset))
            for sub, n in subsets:
                flag = "" if sub.texture else "  [fullres only -- keeps its captions]"
                print(f"         {Path(sub.path).name:<24} {n:>6} entries  {100 * n / total:4.1f}% "
                      f"of the epoch  x{sub.num_repeats}{flag}")
        if self.dataset.tier_report is not None:
            summary = self.dataset.tier_report.summary()
            if summary:
                print(summary)
        print(f"steps    {self.total_steps} total, {self.steps_per_epoch}/epoch, "
              f"{self.accelerator.num_processes} process(es)")
        n_proc = self.accelerator.num_processes
        if n_proc > 1:
            # The trap this exists for: the same config on N GPUs takes N times FEWER optimizer
            # steps, because each rank consumes whole batches. The number is right there in the
            # line above and still reads as "it is running", so it is spelled out.
            print(f"         ^ {n_proc} GPUs means {n_proc}x FEWER optimizer steps than the same "
                  f"config on one ({self.total_steps * n_proc} single-GPU), at {n_proc}x the "
                  f"effective batch. Multiply `epochs` by {n_proc} to match a single-GPU run.")
            if any(p.mode == "texture" for p in cfg.curriculum.phases):
                # Only reachable with the override on -- the constructor refuses otherwise.
                print("         ^ WARNING: texture mode under DDP, override ON. This produced "
                      "progressively worsening anatomy twice. You are re-testing an unproven fix; "
                      "compare against a single-GPU control at matched steps.")
        print(f"mode     {'LoRA/' + cfg.adapter.kind if cfg.is_lora else 'full finetune'}")
        print(f"params   {trainable / 1e6:.2f}M trainable / {total / 1e6:.2f}M total")
        if self.param_report is not None:
            print(self.param_report.summary())
        if self.quant_info is not None:
            nq, ntotal, use_qmm, max_tokens, wmean = self.quant_info
            how = "auto" if cfg.quant.use_quantized_matmul == "auto" else "explicit"
            print(f"quant    {cfg.quant.mode}/{cfg.quant.weights_dtype}, {nq}/{ntotal} Linear "
                  f"quantized, skip={cfg.quant.skip_policy}")
            if cfg.quant.mode == "training":
                # The token rule does not apply here, so printing it would be misleading -- every
                # bucket in the run that exposed this sat ABOVE the crossover and qmm was still the
                # wrong call. See `resolve_quantized_matmul`.
                print(f"         quantized matmul {'ON' if use_qmm else 'OFF'} ({how}; the token "
                      f"crossover does NOT apply in training mode -- weights change every step)")
            else:
                print(f"         quantized matmul {'ON' if use_qmm else 'OFF'} ({how}; largest "
                      f"bucket {max_tokens} tokens, time-weighted mean {wmean:.0f}, "
                      f"crossover {QMM_TOKEN_CROSSOVER})")
            if cfg.quant.mode == "training" and use_qmm:
                n_buckets = len(self.sampler.bucket_indices)
                print(f"         ^ WARNING: quantized matmul is ON in training mode. Measured on a "
                      f"1.76B full FT over 21 buckets: median s/it 9.54 with it against 8.88 "
                      f"without (no gain), and mean 19.02 against 8.76 because Triton recompiles "
                      f"per bucket shape -- 79s spikes mid-run. This run has {n_buckets} bucket(s); "
                      f"expect one spike each, plus far worse on a cold ~/.triton cache "
                      f"(113-178 s/it measured). Set use_quantized_matmul = false unless you are "
                      f"deliberately A/B-ing it.")
        f = cfg.flow
        flow_bits = [f.timestep_sample_method]
        if f.timestep_sample_method == "logit_normal":
            flow_bits.append(f"sigmoid_scale {f.sigmoid_scale}")
        flow_bits.append("flux_shift" if f.flux_shift else f"shift {f.shift if f.shift else 'none'}")
        if f.use_ot:
            # Batch size 1 makes OT a no-op, and the per-bucket map can produce one silently.
            ones = [b for b in self.sampler.bucket_indices
                    if self.sampler.batch_size_for(b) < 2]
            note = f" -- NO-OP in {len(ones)} bucket(s) at batch size 1" if ones else ""
            flow_bits.append(f"optimal transport ON{note}")
        print(f"flow     {', '.join(flow_bits)}")
        if cfg.train.compile:
            print(f"compile  {cfg.train.compile} (dynamic={cfg.train.compile_dynamic}, "
                  f"regional={cfg.train.compile_regional}); first steps include compile time")
        sched = f"{cfg.schedule.kind}, warmup {cfg.schedule.warmup_steps}, floor {cfg.schedule.min_lr_ratio}"
        if cfg.schedule.kind == "rex":
            sched += f", d={cfg.schedule.d}"
        elif cfg.schedule.kind == "rerex":
            sched += (f", global_d={cfg.schedule.global_d}, local_d={cfg.schedule.local_d}, "
                      f"{cfg.schedule.num_segments} segments, weight_power={cfg.schedule.weight_power}")
        print(f"sched    {sched}")
        if self.hf_patch is not None:
            print(f"hf loss  scale {cfg.flow.hf_scale}, exponent {cfg.flow.hf_exponent}, "
                  f"patch {self.hf_patch}")
        if cfg.curriculum:
            print(cfg.curriculum.report(self.total_steps)
                  + f"\n        mapping {cfg.flow.phase_mapping}")
        if any(p.mode == "texture" for p in cfg.curriculum.phases):
            # Printed because texture mode degrades *silently*: with no slack between source and
            # canvas the crop is forced, energy selection chooses from one candidate, and the run
            # is indistinguishable from one that is actually choosing. Header-only reads, so this
            # costs milliseconds even on thousands of files.
            from PIL import Image as _Image

            from ..data.texture import describe_feasibility
            sizes = []
            # Texture-exempt subsets are never cropped, so including them would describe the crop
            # freedom of images that will never be cropped -- and a regularization set of small
            # flat squares would drag every "fits %" down and invent a "no room to choose" warning
            # about a canvas that is in fact fine for everything it will actually see.
            for p in {e.path for e in self.dataset.entries if e.texture_ok}:
                try:
                    with _Image.open(p) as im:
                        sizes.append(im.size)
                except OSError:
                    continue
            print(describe_feasibility(sizes, cfg.dataset.texture))
        est = estimate_optimizer_bytes(trainable, cfg.optimizer) / 1e9
        print(f"optim    {cfg.optimizer.kind}, state ~{est:.1f}GB "
              f"(quantized={cfg.optimizer.quantize_state}, offload={cfg.optimizer.offload_state})\n",
              flush=True)

    # -------------------------------------------------------- checkpoints

    def _unwrap(self, module):
        """Strip DDP and torch.compile wrappers before reading a state dict.

        `keep_torch_compile=False` matters: Accelerate defaults it to True, and `compile_regions`
        stamps `_orig_mod` on the root object it returns -- which, because regional compilation
        runs after the DDP wrap, is the DDP module. `extract_model_from_parallel` then sees that
        stamp on the way out and re-attaches the wrapper it had just removed.
        """
        return self.accelerator.unwrap_model(module, keep_torch_compile=False)

    def save(self, tag: str) -> None:
        acc = self.accelerator
        acc.wait_for_everyone()

        # `save_state` is collective -- every rank must call it -- so it happens before the
        # main-process-only export below, not after.
        if self.cfg.train.save_optimizer_state:
            self.save_full_state(tag)

        if not acc.is_main_process:
            return

        cfg = self.cfg
        # sd-scripts' convention: one file per checkpoint, named `<run>-stepNNNNNN.safetensors`,
        # sitting directly in the run directory. A folder per checkpoint made every LoRA in a run
        # share the same filename, so twenty of them could not live in one ComfyUI models folder
        # without being renamed by hand.
        #
        # Only exports that are genuinely ONE file get flattened. A full finetune without
        # `save_native` writes `transformer/` and `text_conditioner/` as diffusers directories, so
        # it keeps a directory -- that is the format's shape, not a naming choice.
        stem = cfg.train.run_name if tag == "final" else f"{cfg.train.run_name}-{tag}"
        single_file = cfg.is_lora or cfg.train.save_native
        dest = self.out_dir if single_file else self.out_dir / stem
        dest.mkdir(parents=True, exist_ok=True)

        # `keep_torch_compile=False` is load-bearing, not tidiness. Accelerate's default is True,
        # and under regional compilation that makes `unwrap_model` re-attach the compiled wrapper
        # it just removed -- returning keys with `module.` at the root and `_orig_mod` inside every
        # block. The export then failed with 560 unmapped keys, at the first checkpoint rather than
        # at startup. `convert.strip_wrappers` catches it too; this stops it happening at all.
        transformer = self._unwrap(self.transformer)
        conditioner = self._unwrap(self.text_conditioner)

        meta = {"step": str(self.global_step), "run": cfg.train.run_name}

        if cfg.is_lora:
            n = save_lora_checkpoint(
                str(dest / f"{stem}.safetensors"),
                transformer_lora=_peft_state_dict(transformer),
                text_conditioner_lora=(
                    _peft_state_dict(conditioner) if cfg.adapter.train_llm_adapter else None
                ),
                dtype=self.dtype,
                metadata=meta,
                # Without this the trained scale is lost: peft applies alpha/r at runtime and the
                # exported tensors carry no record of it, so a consumer that finds no `.alpha`
                # falls back to scale 1.0 and applies the LoRA at alpha/r times its trained
                # strength -- half, at the common alpha 16 / rank 8.
                alpha=cfg.adapter.alpha,
                rank=cfg.adapter.rank,
            )
            if cfg.adapter.train_text_encoder:
                # Qwen3 adapters have no place in the diffusers Anima LoRA layout, so they go to
                # their own file rather than being silently dropped from the export.
                from safetensors.torch import save_file

                te = _peft_state_dict(self._unwrap(self.text_encoder))
                save_file(
                    {f"lora_te.{k}": v.to(self.dtype).contiguous() for k, v in te.items()},
                    dest / f"{stem}_te.safetensors",
                    metadata=meta,
                )
        elif cfg.train.save_native:
            # A quantized full finetune holds SDNQTensor master weights, which safetensors cannot
            # serialize; dequantize to plain bf16 so the export is an ordinary checkpoint.
            prep = (
                (lambda sd: dequantize_state_dict(sd, self.dtype))
                if cfg.quant.mode == "training"
                else (lambda sd: sd)
            )
            n = save_native_checkpoint(
                str(dest / f"{stem}.safetensors"),
                transformer_state_dict=prep(transformer.state_dict()),
                text_conditioner_state_dict=prep(conditioner.state_dict()),
                dtype=self.dtype,
                metadata=meta,
            )
        else:
            transformer.save_pretrained(dest / "transformer")
            conditioner.save_pretrained(dest / "text_conditioner")
            n = 0

        written = dest / f"{stem}.safetensors" if single_file else dest
        # `state.json` lives beside the optimizer state, which is the only thing that reads it.
        # A flat single-file checkpoint carries its step in the safetensors metadata instead, so an
        # inference-only export is one file with nothing to keep in sync alongside it.
        if cfg.train.save_optimizer_state:
            (self.out_dir / f"{stem}-state" / "state.json").write_text(
                json.dumps({"global_step": self.global_step, "tensors": n}, indent=2)
            )
        _emit(self, f"saved {written} ({n} tensors)")
        self._prune_checkpoints()

    def _prune_checkpoints(self) -> None:
        """Keep only the newest `keep_last_n` periodic checkpoints.

        Deliberately narrow, because this deletes. A candidate must carry THIS run's name and a
        periodic tag (`-epochNNN` / `-stepNNNNNN`); `final` -- which is the bare run name, with no
        tag -- and anything else in the directory are never candidates. Ordering is by the recorded
        step rather than mtime, so a resumed run cannot delete the wrong one, and epoch-tagged and
        step-tagged checkpoints order correctly against each other.

        Everything belonging to one checkpoint is removed together: the file, its `_te` sibling,
        and its `-state` directory. Directory-form checkpoints (a full finetune without
        `save_native`, and anything written before the layout was flattened) are still recognised,
        so `keep_last_n` does not quietly stop pruning an existing run.
        """
        keep = self.cfg.train.keep_last_n
        if not keep or keep < 1:
            return

        run = re.escape(self.cfg.train.run_name)
        # `<run>-step000060` / `<run>-epoch005`, and the bare `step000060` of the old layout.
        tagged = re.compile(rf"(?:{run}-)?(epoch|step)(\d+)$")
        groups: dict[tuple[str, int], list[Path]] = {}
        for entry in self.out_dir.iterdir():
            name = entry.name
            if entry.is_file():
                if entry.suffix != ".safetensors":
                    continue
                name = entry.stem.removesuffix("_te")
            elif name.endswith("-state"):
                name = name[: -len("-state")]
            m = tagged.fullmatch(name)
            if not m:
                continue
            groups.setdefault((m.group(1), int(m.group(2))), []).append(entry)

        # An epoch tag counts epochs and a step tag counts steps, so the two are not comparable as
        # raw numbers. `state.json` carries the true global_step where it exists; otherwise the
        # safetensors metadata does; otherwise fall back to the tag number, which is correct within
        # a single tag kind and is the only case that can mix.
        def ordinal(kind: str, num: int, paths: list[Path]) -> int:
            for p in paths:
                state = (p / "state.json") if p.is_dir() else None
                if state is not None and state.exists():
                    try:
                        return int(json.loads(state.read_text())["global_step"])
                    except (ValueError, KeyError, TypeError):
                        pass
                if p.is_file():
                    try:
                        from safetensors import safe_open
                        with safe_open(p, framework="pt") as f:
                            step = (f.metadata() or {}).get("step")
                        if step is not None:
                            return int(step)
                    except Exception:
                        pass
            return num if kind == "step" else num * max(1, self.steps_per_epoch)

        ranked = sorted((ordinal(k, n, ps), (k, n), ps) for (k, n), ps in groups.items())
        for _, _, paths in ranked[:-keep]:
            for p in paths:
                shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
                _emit(self, f"pruned {p} (keep_last_n={keep})")

    def save_full_state(self, tag: str) -> None:
        """Optimizer + scheduler + weights, for resuming. Separate from `save` because the
        exported checkpoint deliberately drops optimizer state -- it is for inference -- and a
        resume that silently restarted the moments would quietly change the training dynamics.

        Sits in `<run>-<tag>-state/` beside the flat checkpoint rather than inside it: accelerate
        writes a directory, and a directory cannot live inside a `.safetensors` file.
        """
        stem = self.cfg.train.run_name if tag == "final" else f"{self.cfg.train.run_name}-{tag}"
        dest = self.out_dir / f"{stem}-state" / "accelerator"
        self.accelerator.save_state(str(dest))

    def _resume(self, path: str) -> None:
        p = Path(path)
        # Accept the checkpoint FILE as well as its state directory. Under the flat layout the
        # thing a user has in hand is `<run>-step000060.safetensors`, and pointing `resume_from` at
        # it is the obvious move -- so it resolves to the sibling `-state` directory rather than
        # erroring about a state.json that was never meant to sit next to it.
        if p.is_file() and p.suffix == ".safetensors":
            sibling = p.with_name(p.stem.removesuffix("_te") + "-state")
            if not sibling.exists():
                raise FileNotFoundError(
                    f"{p} has no optimizer state beside it (expected {sibling}). It was written "
                    f"with save_optimizer_state disabled, so it can be used for inference but not "
                    f"resumed."
                )
            p = sibling

        state_file = p / "state.json"
        if not state_file.exists():
            raise FileNotFoundError(f"no state.json in {p}; not a checkpoint directory")

        acc_dir = p / "accelerator"
        if not acc_dir.exists():
            raise FileNotFoundError(
                f"{p} has no saved optimizer state (expected {acc_dir}). It was written with "
                f"save_full_state disabled, so it can be used for inference but not resumed."
            )
        self.accelerator.load_state(str(acc_dir))

        self.global_step = json.loads(state_file.read_text())["global_step"]
        self.start_epoch = self.global_step // max(1, self.steps_per_epoch)
        self.accelerator.print(f"resumed at step {self.global_step} (epoch {self.start_epoch})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Anima")
    ap.add_argument("config", help="path to a TOML config")
    ap.add_argument("--dry-run", action="store_true",
                    help="build everything and run one step, then exit")
    ap.add_argument("--max-steps", type=int, default=None, help="override train.max_steps")
    ap.add_argument("--no-save", action="store_true", help="benchmark without writing checkpoints")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.max_steps:
        cfg.train.max_steps = args.max_steps
    if args.dry_run:
        cfg.train.max_steps = 1
        cfg.train.epochs = 1
    if args.dry_run or args.no_save:
        cfg.train.save_every_epochs = None
        cfg.train.save_every_steps = None
        cfg.train.skip_final_save = True

    Trainer(cfg, config_path=args.config).train()


if __name__ == "__main__":
    main()
