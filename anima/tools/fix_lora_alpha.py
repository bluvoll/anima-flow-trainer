"""Add the missing `.alpha` scalars to LoRAs exported before that was written.

Why this is needed: peft applies `alpha / rank` at runtime, so `lora_A`/`lora_B` carry no record of
the scale they were trained at. Exports written without an `.alpha` beside them lose it, and
ComfyUI's loader falls back to `scale = 1.0` when the key is absent
(`comfy/weight_adapter/lora.py:314`) -- so a LoRA trained at alpha 16 / rank 8 was applied at HALF
its trained strength, and alpha 8 vs alpha 64 produced byte-identical files.

The weights are correct and are not touched. This only adds the scalar that was dropped.

    python -m anima.tools.fix_lora_alpha --alpha 16 output/run/*/  *.safetensors
    python -m anima.tools.fix_lora_alpha --alpha 16 --dry-run some.safetensors

Rank is read from the tensors (`lora_A.shape[0]`), never guessed, and a file whose modules disagree
on rank is refused rather than half-written.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def collect(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        q = Path(p)
        out.extend(sorted(q.rglob("*.safetensors")) if q.is_dir() else [q])
    return [p for p in out if p.suffix == ".safetensors"]


def fix(path: Path, alpha: float, dry_run: bool, force: bool) -> str:
    with safe_open(path, framework="pt") as f:
        meta = f.metadata() or {}
        tensors = {k: f.get_tensor(k) for k in f.keys()}

    a_keys = [k for k in tensors if k.endswith(".lora_A.weight")]
    if not a_keys:
        # kohya-format files use lora_down/lora_up and already carry .alpha.
        down = [k for k in tensors if k.endswith(".lora_down.weight")]
        if down and any(k.endswith(".alpha") for k in tensors):
            return "skip: kohya format, already has .alpha"
        return "skip: no lora_A tensors"

    existing = sum(1 for k in tensors if k.endswith(".alpha"))
    if existing and not force:
        return f"skip: already has {existing} .alpha keys (use --force to overwrite)"

    ranks = {tensors[k].shape[0] for k in a_keys}
    if len(ranks) != 1:
        return f"REFUSED: modules disagree on rank {sorted(ranks)}; alpha/rank would differ per module"
    rank = ranks.pop()

    for k in a_keys:
        tensors[k[: -len(".lora_A.weight")] + ".alpha"] = torch.tensor(float(alpha))

    scale = alpha / rank
    msg = (f"{len(a_keys)} modules, rank {rank}, alpha {alpha:g} -> scale {scale:g} "
           f"(was 1.0, i.e. {scale:g}x weaker)")
    if dry_run:
        return "would fix: " + msg

    meta = {**meta, "anima_alpha": str(alpha), "anima_rank": str(rank)}
    tmp = path.with_suffix(".safetensors.tmp")
    save_file(tensors, tmp, metadata=meta)
    os.replace(tmp, path)      # atomic: a crash mid-write cannot leave a truncated LoRA
    return "fixed: " + msg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="+", help="safetensors files, or directories to search")
    ap.add_argument("--alpha", type=float, required=True, help="the alpha the LoRA was trained at")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite existing .alpha keys")
    args = ap.parse_args()

    files = collect(args.paths)
    if not files:
        print("no .safetensors found")
        return 1
    for p in files:
        print(f"{p}\n    {fix(p, args.alpha, args.dry_run, args.force)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
