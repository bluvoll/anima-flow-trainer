"""Gate for `dataset.source = "encode"` -- training without cached latents.

The property that matters is *equivalence*: an encode-on-the-fly run must be the same training run
as a cached one, differing only in when the VAE runs. If the two ever disagree -- a different crop,
a different normalisation, a latent in the wrong dtype -- nothing fails. Both paths keep producing
plausible latents and the run just trains on subtly different data than the cached run did. So the
checks here are mostly about the two paths agreeing, not about encode mode "working".

CPU-only except for `--gpu`, which adds the VAE numerical-agreement check against real caches.

    .venv/bin/python anima/parity/test_encode_mode.py
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python anima/parity/test_encode_mode.py --gpu
"""

from __future__ import annotations

import os
import argparse
import sys
import tempfile
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from anima.data.cache import cache_path, image_to_tensor, load_and_crop  # noqa: E402
from anima.data.dataset import AnimaDataset, DatasetConfig, collate  # noqa: E402
from anima.training.config import TrainConfig  # noqa: E402

PASS, FAIL = [], []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")


def make_dataset(tmp: Path, n: int = 6) -> Path:
    """A tiny image set with varied aspect ratios, so more than one bucket is exercised."""
    root = tmp / "ds"
    root.mkdir()
    sizes = [(1024, 1024), (1536, 864), (864, 1536), (1200, 900), (900, 1200), (1024, 768)]
    for i in range(n):
        w, h = sizes[i % len(sizes)]
        Image.new("RGB", (w, h), (i * 30 % 256, 90, 200)).save(root / f"img{i}.png")
        (root / f"img{i}.txt").write_text("a tag, another tag")
    return root


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true", help="also check VAE numerical agreement")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        root = make_dataset(tmp)

        # --- encode mode does not require a cache -------------------------------------------
        # The whole point: `source = "images"` hard-raises here, and that raise is what would
        # otherwise make texture training impossible.
        try:
            AnimaDataset(DatasetConfig(path=str(root), resolution=1024, source="images"))
            check("images mode still refuses to run without caches", False, "no raise")
        except RuntimeError as e:
            check("images mode still refuses to run without caches",
                  "no cached latent" in str(e), str(e)[:40])

        enc = AnimaDataset(DatasetConfig(path=str(root), resolution=1024, source="encode"))
        check("encode mode builds entries with no cache present", len(enc) == 6, f"{len(enc)}")
        check("encode mode sets encode_on_the_fly", enc.encode_on_the_fly is True)

        # --- the two paths must agree on *which* pixels -----------------------------------
        # Entry lists must match element for element, or an encode run trains on a different
        # crop of a different image than the cached run would have.
        cached_like = AnimaDataset(
            DatasetConfig(path=str(root), resolution=1024, source="images"), require_cache=False
        )
        same = ([(e.path, e.bucket, e.resolution) for e in enc.entries]
                == [(e.path, e.bucket, e.resolution) for e in cached_like.entries])
        check("encode and images modes produce identical entry lists", same)

        # --- sample shape and range ----------------------------------------------------------
        s = enc[0]
        e0 = enc.entries[0]
        img, _ = load_and_crop(e0.path, enc.bucket_managers[e0.resolution])
        w, h = e0.bucket
        check("sample yields pixels, not latents", "pixels" in s and "latents" not in s)
        check("pixel tensor is (3, 1, H, W) at the bucket size",
              tuple(s["pixels"].shape) == (3, 1, h, w), str(tuple(s["pixels"].shape)))
        check("pixels are in [-1, 1]",
              float(s["pixels"].min()) >= -1.0 and float(s["pixels"].max()) <= 1.0,
              f"[{float(s['pixels'].min()):.2f}, {float(s['pixels'].max()):.2f}]")
        # Bit-exact against the cacher's own helper: the crop must be the cacher's crop, not a
        # re-derived one, or cached and encode runs see different framing.
        check("crop is bit-identical to the cacher's load_and_crop",
              torch.equal(s["pixels"], image_to_tensor(img).squeeze(0)))

        # --- collate ------------------------------------------------------------------------
        batch = collate([enc[i] for i in range(len(enc)) if enc.entries[i].bucket == e0.bucket])
        check("collate stacks pixels and keeps the bucket", "pixels" in batch
              and batch["pixels"].shape[1:] == (3, 1, h, w) and batch["bucket"] == e0.bucket)
        check("collate still rejects mixed buckets",
              _raises(lambda: collate([enc[i] for i in range(len(enc))])))

        # --- captions still vary per epoch ----------------------------------------------------
        # Encode mode must not accidentally bypass the seeded per-sample caption RNG. Needs a
        # config that actually randomises and enough tags for a shuffle to be observable --
        # asserting variation on a static two-tag caption would pass no matter what the code did.
        shuf = DatasetConfig(path=str(root), resolution=1024, source="encode")
        shuf.caption.shuffle_tags = True
        shuf.caption.tag_dropout_percent = 0.3
        for i in range(6):
            (root / f"img{i}.txt").write_text(", ".join(f"tag{j}" for j in range(12)))
        sds = AnimaDataset(shuf)
        caps = {sds[(0, ep)]["caption"] for ep in range(16)}
        check("caption RNG still varies by epoch under encode mode", len(caps) > 1, f"{len(caps)}/16")
        check("same (index, epoch) is reproducible",
              sds[(0, 3)]["caption"] == sds[(0, 3)]["caption"])

        # --- config validation ----------------------------------------------------------------
        check("unknown source is rejected",
              _raises(lambda: AnimaDataset(DatasetConfig(path=str(root), source="nope"))))
        check("vae_encode_chunk must be >= 1",
              _raises(lambda: TrainConfig(model_path="x", vae_encode_chunk=0)))
        check("vae_encode_chunk default is 1", TrainConfig(model_path="x").vae_encode_chunk == 1)

        if args.gpu:
            gpu_checks(root)

    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} passed")
    return 1 if FAIL else 0


def gpu_checks(root: Path) -> None:
    """Encode-on-the-fly must reproduce the cache numerically, not merely plausibly."""
    from anima.data.cache import LatentCacher, load_cached_latent
    from anima.modeling.loader import load_components

    c = load_components(os.environ.get("ANIMA_MODEL", "../anima-diffusers"), dtype=torch.bfloat16,
                        load_text_encoder=False, load_vae=True, load_tokenizers=False)
    enc = LatentCacher(c.vae, device="cuda", dtype=torch.bfloat16)
    ds = AnimaDataset(DatasetConfig(path=str(root), resolution=1024, source="encode"))

    # `encode` samples the VAE posterior, so two calls on *identical* input already differ. The
    # question is whether the dataset's pixels differ by more than that -- i.e. whether the
    # difference is sampling noise or a genuinely different crop. So measure the noise floor from
    # repeated encodes of the same input and require the pixel path to sit inside it. Comparing
    # against a fixed epsilon instead would either pass on a real crop bug (on detailed images,
    # where the posterior is tiny) or fail on correct code (on flat images, where it is not).
    # Compared with *mean* abs, not max: over ~1M elements the max is an extreme order statistic
    # that swings ~2x between two draws of the same distribution, so a max-based bound is a coin
    # flip. The mean concentrates, so "same distribution" is testable.
    worst = floor = 0.0
    for i in range(len(ds)):
        e = ds.entries[i]
        img, _ = load_and_crop(e.path, ds.bucket_managers[e.resolution])
        a = enc.encode(img).cpu()
        b = enc.encode(img).cpu()                            # same input, second draw
        floor = max(floor, float((a - b).abs().mean()))
        fresh = enc.encode_tensor(ds[i]["pixels"].unsqueeze(0)).squeeze(0).cpu().float()
        worst = max(worst, float((fresh - a).abs().mean()))
    check("dataset pixels encode to the cacher's latents (within posterior noise)",
          worst <= floor * 1.25 + 1e-5, f"mean abs {worst:.2e} vs noise floor {floor:.2e}")

    # A real cache written to disk must read back as the same tensor the encode path produces.
    e = ds.entries[0]
    img, meta = load_and_crop(e.path, ds.bucket_managers[e.resolution])
    out, _ = enc.cache_image(e.path, ds.bucket_managers[e.resolution])
    on_disk = load_cached_latent(out).float()
    live = enc.encode_tensor(ds[0]["pixels"].unsqueeze(0)).squeeze(0).cpu().float()
    rel = float((live - on_disk).abs().mean() / on_disk.abs().mean())
    check("a written cache matches the live encode", rel < 5e-3, f"rel {rel:.2e}")
    assert cache_path(e.path, meta["bucket"]) == out


def _raises(fn) -> bool:
    try:
        fn()
    except Exception:
        return True
    return False


if __name__ == "__main__":
    sys.exit(main())
