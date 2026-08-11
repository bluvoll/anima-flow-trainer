"""Phase 4b gate: the loop must actually be conditioned on text and on the timestep.

A training loop can run, decrease a loss, and still be silently broken -- the classic failure is
cross-attention receiving a constant, so the model learns an unconditional prior and the caption
pipeline is decorative. That kind of bug does not raise; it just produces a model that ignores
prompts after a week of compute.

Three checks, all on the *frozen* base model so they measure wiring rather than training:

  1. Loss under the true caption must be lower than under a mismatched one. If not, the DiT is not
     using `encoder_hidden_states`.
  2. Loss must vary with the timestep. A flat curve means `timestep` is not reaching the modulation
     path (e.g. wrong scale, or the [0,1] vs [0,1000] convention broken).
  3. Loss must be far below the no-op baseline `E||target||^2`, which is what a model predicting
     zero would score.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from anima.training.config import load_config  # noqa: E402
from anima.training.flow import flow_loss, prepare_flow_batch  # noqa: E402
from anima.training.train import Trainer  # noqa: E402


@torch.no_grad()
def loss_for(trainer, batch, captions, quantile, seed=0):
    """Frozen-model loss with a fixed timestep quantile and a fixed noise draw, so the only thing
    that varies between calls is what we intend to vary."""
    device = trainer.accelerator.device
    latents = batch["latents"].to(device, torch.float32)

    torch.manual_seed(seed)
    noisy, timesteps, target = prepare_flow_batch(latents, trainer.cfg.flow, quantile=quantile)

    emb = trainer._encode(captions)
    w, h = batch["bucket"]
    pred = trainer.transformer(
        hidden_states=noisy.to(trainer.dtype),
        timestep=timesteps.to(trainer.dtype),
        encoder_hidden_states=emb.to(trainer.dtype),
        padding_mask=noisy.new_zeros(1, 1, h, w, dtype=trainer.dtype),
        return_dict=False,
    )[0]
    return flow_loss(pred, target).item(), target.pow(2).mean().item()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--batches", type=int, default=4)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.train.max_steps = 1
    trainer = Trainer(cfg)
    trainer.transformer.eval()
    trainer.text_conditioner.eval()

    loader = iter(trainer.loader)
    batches = [next(loader) for _ in range(args.batches)]

    print("\n--- 1. text conditioning ---")
    real_total = wrong_total = 0.0
    for i, b in enumerate(batches):
        wrong = ["a photograph of an empty parking lot at night"] * len(b["captions"])
        real, baseline = loss_for(trainer, b, b["captions"], quantile=0.5, seed=i)
        bad, _ = loss_for(trainer, b, wrong, quantile=0.5, seed=i)
        real_total += real
        wrong_total += bad
        print(f"  batch {i}: matched {real:.4f}   mismatched {bad:.4f}   "
              f"delta {bad - real:+.4f}")
    n = len(batches)
    conditioned = wrong_total > real_total
    print(f"  mean matched {real_total / n:.4f} vs mismatched {wrong_total / n:.4f}  "
          f"-> {'CONDITIONED' if conditioned else 'NOT CONDITIONED'}")

    print("\n--- 2. timestep sensitivity ---")
    losses = []
    for q in (0.05, 0.25, 0.5, 0.75, 0.95):
        vals = [loss_for(trainer, b, b["captions"], quantile=q, seed=i)[0]
                for i, b in enumerate(batches)]
        mean = sum(vals) / len(vals)
        losses.append(mean)
        print(f"  quantile {q:.2f}: loss {mean:.4f}")
    spread = max(losses) / max(min(losses), 1e-9)
    print(f"  max/min = {spread:.2f}x  -> {'VARIES' if spread > 1.5 else 'FLAT (suspicious)'}")

    print("\n--- 3. vs predict-zero baseline ---")
    real, baseline = loss_for(trainer, batches[0], batches[0]["captions"], quantile=0.5)
    print(f"  model {real:.4f}   predict-zero {baseline:.4f}   "
          f"ratio {real / baseline:.3f}")
    beats_baseline = real < baseline * 0.5

    ok = conditioned and spread > 1.5 and beats_baseline
    print(f"\nTRAIN SANITY {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
