"""Latent caching.

Latents are cached unconditionally: the VAE is deterministic given a bucket, so recomputing it
every epoch is pure waste. Text embeddings are *not* cached by default, because tag shuffling and
dropout have to run per step -- see caption.py. Caching them is a config switch for the case where
captions are static.

One file per (image, bucket) so a bucket-config change invalidates only what it should. safetensors
rather than npz: memory-mappable, so the dataloader pages in a single latent without decompressing
a container.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from .bucket import BucketManager

Image.MAX_IMAGE_PIXELS = None  # dataset images legitimately exceed the decompression-bomb guard

CACHE_SUFFIX = "_anima.safetensors"
CACHE_VERSION = "1"

# `<stem>_WWWWxHHHH_anima.safetensors` -- the bucket is recoverable from the name alone, which is
# what lets a dataset run from latents after its source images have been deleted.
_CACHE_RE = re.compile(rf"^(?P<stem>.+)_(?P<w>\d{{4,}})x(?P<h>\d{{4,}}){re.escape(CACHE_SUFFIX)}$")


@dataclass
class CachedLatent:
    latents: torch.Tensor          # (16, 1, H/8, W/8)
    bucket: tuple[int, int]
    original_size: tuple[int, int]
    crop_ltrb: tuple[int, int, int, int]


def cache_path(image_path: str | Path, bucket: tuple[int, int]) -> Path:
    p = Path(image_path)
    return p.with_name(f"{p.stem}_{bucket[0]:04d}x{bucket[1]:04d}{CACHE_SUFFIX}")


def load_and_crop(image_path: str | Path, bucket_mgr: BucketManager) -> tuple[Image.Image, dict]:
    """Resize to the bucket's aspect then centre-crop. Never pads."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    bucket, resized, ar_error = bucket_mgr.select_bucket(w, h)

    if bucket[0] <= 0 or bucket[1] <= 0:
        raise ValueError(f"{image_path}: degenerate bucket {bucket} for source {w}x{h}")

    img = img.resize(resized, Image.LANCZOS)
    left = (resized[0] - bucket[0]) // 2
    top = (resized[1] - bucket[1]) // 2
    img = img.crop((left, top, left + bucket[0], top + bucket[1]))

    return img, {
        "bucket": bucket,
        "original_size": (w, h),
        "crop_ltrb": (left, top, left + bucket[0], top + bucket[1]),
        "ar_error": ar_error,
    }


def image_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL -> (1, 3, 1, H, W) in [-1, 1], the 5D shape the Qwen-Image VAE expects."""
    # np.array (not asarray): PIL exposes a read-only buffer, and torch.from_numpy on a
    # non-writable array warns and yields a tensor with undefined write semantics.
    x = torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1)
    x = x.float().div_(127.5).sub_(1.0)
    return x.unsqueeze(0).unsqueeze(2)


class LatentCacher:
    """Encodes images to normalised latents and persists them.

    Normalisation is `(x - latents_mean) / latents_std` using the VAE's own config constants.
    Those constants are Anima's pretraining normalisation -- verified identical to the values
    diffusion-pipe hardcodes -- so they must not be substituted with dataset-derived statistics
    even though real data normalises to std ~0.69 rather than 1.0.
    """

    def __init__(self, vae, device: torch.device | str = "cuda", dtype: torch.dtype = torch.bfloat16):
        self.vae = vae.to(device).eval()
        self.device = torch.device(device)
        self.dtype = dtype
        z = vae.config.z_dim
        self.mean = torch.tensor(vae.config.latents_mean).view(1, z, 1, 1, 1).to(device, torch.float32)
        self.std = torch.tensor(vae.config.latents_std).view(1, z, 1, 1, 1).to(device, torch.float32)

    @torch.no_grad()
    def encode_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, 1, H, W) in [-1, 1] -> (B, 16, 1, h, w), normalised.

        The single place the VAE encode and its normalisation are written down. `encode()` below
        and the trainer's on-the-fly path (`dataset.source = "encode"`) both come through here, so
        a cached run and an encode-on-the-fly run cannot drift apart numerically -- which would
        otherwise be invisible, since both produce plausible latents.
        """
        x = x.to(self.device, self.dtype)
        latents = self.vae.encode(x).latent_dist.sample().float()
        return (latents - self.mean) / self.std

    def encode(self, img: Image.Image) -> torch.Tensor:
        return self.encode_tensor(image_to_tensor(img)).squeeze(0).to(torch.float32)  # (16, 1, h, w)

    def cache_image(
        self, image_path: str | Path, bucket_mgr: BucketManager, overwrite: bool = False
    ) -> tuple[Path, dict]:
        img, meta = load_and_crop(image_path, bucket_mgr)
        out = cache_path(image_path, meta["bucket"])

        if out.exists() and not overwrite:
            return out, meta

        latents = self.encode(img)
        save_file(
            {"latents": latents.contiguous()},
            out,
            metadata={
                "version": CACHE_VERSION,
                "bucket": f"{meta['bucket'][0]}x{meta['bucket'][1]}",
                "original_size": f"{meta['original_size'][0]}x{meta['original_size'][1]}",
                "crop_ltrb": ",".join(str(v) for v in meta["crop_ltrb"]),
                "source": hashlib.sha256(str(Path(image_path).name).encode()).hexdigest()[:16],
            },
        )
        return out, meta


def load_cached_latent(path: str | Path) -> torch.Tensor:
    return load_file(path)["latents"]


def read_cache_metadata(path: str | Path) -> dict[str, str]:
    """Header-only read -- the tensor is never touched.

    `original_size` is what makes latents-only multi-resolution possible: with the source image
    gone, it is the only way to re-derive which tier explains a given cached bucket. It has been
    written since CACHE_VERSION 1, but callers must tolerate its absence rather than assume.
    """
    from safetensors import safe_open

    with safe_open(path, framework="pt") as f:
        return f.metadata() or {}


def parse_original_size(meta: dict[str, str]) -> tuple[int, int] | None:
    raw = meta.get("original_size")
    if not raw or "x" not in raw:
        return None
    w, _, h = raw.partition("x")
    try:
        return int(w), int(h)
    except ValueError:
        return None


def is_cached(image_path: str | Path, bucket: tuple[int, int]) -> bool:
    return cache_path(image_path, bucket).exists()


def parse_cache_filename(path: str | Path) -> tuple[str, tuple[int, int]] | None:
    """`<stem>_1024x0576_anima.safetensors` -> ("<stem>", (1024, 576)). None if it doesn't match."""
    m = _CACHE_RE.match(Path(path).name)
    if not m:
        return None
    return m.group("stem"), (int(m.group("w")), int(m.group("h")))


def find_cached_latents(root: str | Path) -> dict[str, list[tuple[Path, tuple[int, int]]]]:
    """Map stem -> [(cache_path, bucket)]. A stem can have several if bucket config changed."""
    out: dict[str, list[tuple[Path, tuple[int, int]]]] = {}
    for p in sorted(Path(root).glob(f"*{CACHE_SUFFIX}")):
        parsed = parse_cache_filename(p)
        if parsed:
            stem, bucket = parsed
            out.setdefault(stem, []).append((p, bucket))
    return out


@dataclass
class CacheAudit:
    trainable: list[str]            # cache + caption both present
    missing_caption: list[str]      # cached but no .txt -- would be dropped
    uncached_images: list[str]      # image present with no cache -- would be lost on delete
    multi_bucket: dict[str, list[tuple[int, int]]]
    total_cache_bytes: int
    total_image_bytes: int

    @property
    def safe_to_delete_images(self) -> bool:
        return not self.uncached_images and not self.missing_caption


def audit_cache(root: str | Path, image_extensions: tuple[str, ...]) -> CacheAudit:
    """Check a directory before its images get deleted.

    Deleting source images is irreversible, so this reports exactly what would survive: any image
    without a cache, or any cache without a caption, is data that silently vanishes from training.
    """
    root = Path(root)
    cached = find_cached_latents(root)

    images = {
        p.stem: p
        for p in root.iterdir()
        if p.suffix.lower() in image_extensions and not p.name.endswith(CACHE_SUFFIX)
    }

    trainable, missing_caption = [], []
    for stem in cached:
        if (root / f"{stem}.txt").exists():
            trainable.append(stem)
        else:
            missing_caption.append(stem)

    uncached = sorted(set(images) - set(cached))
    multi = {s: [b for _, b in v] for s, v in cached.items() if len(v) > 1}

    cache_bytes = sum(p.stat().st_size for v in cached.values() for p, _ in v)
    image_bytes = sum(p.stat().st_size for p in images.values())

    return CacheAudit(
        trainable=sorted(trainable),
        missing_caption=sorted(missing_caption),
        uncached_images=uncached,
        multi_bucket=multi,
        total_cache_bytes=cache_bytes,
        total_image_bytes=image_bytes,
    )
