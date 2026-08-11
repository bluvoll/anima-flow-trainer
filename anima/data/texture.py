"""Texture-crop training: canvas-sized crops chosen by image content.

Ported from TrainTrain (`trainer/dataset.py`, the `texture_source` branch). The idea: instead of
training on the whole image resized down to a bucket, crop a canvas-sized region **at native pixel
scale** out of the source, pick the region by detail content, and supervise only a feathered
sub-region of it while the rest conditions the forward pass as real context.

Three things about the reference implementation are worth knowing, because two of them are
reproduced only on request.

**It black-pads.** `image.crop()` beyond the image bounds is not an error in PIL -- it fills with
zeros. The reference computes `max_sx = max(0, image.width - canvas_w)` and crops anyway, so any
image smaller than the chosen canvas yields a crop with a black band. Measured on one texture
dataset this fires for 36-43% of images at the square canvases and 83% at `1536x640`, and since the
reference fixes one canvas per epoch it happens for whole epochs at a time. `oversize` selects the
behaviour: `"pad"` reproduces it (for validating a port against an existing checkpoint), `"cover"`
scales the image up to cover the canvas before cropping, `"skip"` drops the image for that batch.

**Crop freedom depends entirely on source resolution**, and it is not a detail -- it is whether the
technique does anything at all. Measured median "tightest slack" (the smaller of the two spare
dimensions) for a 1024x1024 canvas: **1001px on a mixed set** (median 4.4MP sources) against **176px on
the texture set** (median 1.2MP). `describe_feasibility` reports this at startup so a run that cannot
crop says so instead of looking like it worked.

**Selection is a greedy hill-climb in the reference** -- up to 10 random candidates, each scored
with a full-canvas Laplacian, accepting the first above threshold. Here the energy map is computed
once per image at 1/8 scale (~64x cheaper) and crop positions are *sampled proportional to* energy,
which removes the first-candidate bias and has no rejection loop.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image

# (width, height). The reference's list, which is symmetric under transpose apart from the square
# appearing twice -- there it buys the square a 1/3 chance rather than 1/5.
CANVAS_PRESETS: list[tuple[int, int]] = [
    (640, 1536), (1536, 640),
    (832, 1216), (1216, 832),
    (1024, 1024), (1024, 1024),
]

OVERSIZE_MODES = ("cover", "pad", "skip")
PAD_MODES = ("reflect", "black")


@dataclass
class TextureConfig:
    canvases: list[tuple[int, int]] = field(default_factory=lambda: list(CANVAS_PRESETS))
    # How to handle a source smaller than the canvas. "pad" reproduces the reference exactly, black
    # bands included; "cover" scales the source up to cover the canvas; "skip" excludes the image.
    #
    # The default is "pad" because it won an A/B, which was not the expected result. "cover" is the
    # one that invents no pixels, and it does cut the average crop's black area from 10.9% to 0.25%
    # on the texture set. But it pays for that by UPSCALING: it fired on 57% of draws there at a mean
    # 1.25x (max 1.85x), so most of every texture batch arrived magnified and resampler-softened,
    # and the trained result showed it in faces. Aspect ratio is not the problem -- the resize is
    # uniform, measured at 0.04% distortion. Scale is. A black band is a consistent signal the
    # model can learn to ignore; a subtly wrong scale is not.
    oversize: str = "pad"
    # Laplacian energy is raised to this power before being used as a sampling weight. 0 makes
    # crop position uniform, which is the honest control for "does energy selection matter".
    energy_power: float = 1.0
    energy_downscale: int = 8
    # Supervised sub-region, as a fraction of the canvas, and its cosine feather in *latent* px.
    mask_ratio: tuple[float, float] = (0.5, 1.0)
    feather_latent_px: int = 2
    # Texture crops get the trigger alone, not the image's caption. This is not a convenience:
    # Anima has no size micro-conditioning (no original_size, no crop coords), so it cannot tell a
    # zoomed-in crop from a physically large subject. Captioning a 1024px crop of a 4000px image
    # with tags describing the whole image teaches that confusion into every one of those tags.
    # Empty string means unconditional, which is what the reference does when no trigger is set.
    trigger: str = ""

    # --- the oversize cascade ---------------------------------------------------------------
    # `oversize` above is the last rung. The four switches here run before it, and all four are ON
    # by default. Setting them to (False, 1.0, "black", False) is byte-identical to the trainer as
    # it stood before they existed -- verified over 200 trials across 5 canvases and random source
    # sizes, 200/200 identical crops -- and `configs/reference_pad.toml` is exactly that.
    #
    # Why: blind canvas selection pads 56% of draws on one texture set, and a black band at that rate is
    # learnable -- a padded run emitted a black bar across the bottom of roughly 1 in 50
    # generations. The cascade takes crops needing invented pixels from 58.4% to 0.7%, and black
    # pixels from 11.6% to 0.4%.
    #
    # These defaults were reverted once and then restored, which is worth recording because the
    # reason was a measurement error, not a change of opinion. The first cascade run had visibly
    # broken anatomy and was blamed on `fit_aware` below. It was actually run under DDP, while its
    # pad baseline was single-GPU -- so the comparison measured DDP, not the cascade. A DDP run
    # with the cascade OFF reproduced the same damage, and a single-GPU cascade run at the full
    # 1239 steps was better than the pad baseline (and overfits somewhat sooner, consistent with
    # the model no longer spending capacity on black bands). See the DDP warning in
    # `train.py::_report`.

    # Rung 1, and the one that does nearly all the work. Draws the canvas from those the source can
    # hold, instead of drawing blind and forcing the fit: 58.4% -> 0.7% of crops needing invented
    # pixels on that set, with no upscaling, because 99% of that dataset fits some canvas.
    # It is also the largest behavioural change of the four -- the canvas mix stops being the
    # preset's flat weighting and follows the dataset's own shapes (1024x1024 40%, 832x1216 37%,
    # 1216x832 11%, 640x1536 9%, 1536x640 2%), so the extreme aspects get rarer. That was the
    # feared downside and it did not materialise in the single-GPU A/B.
    fit_aware: bool = True
    # Rung 2. A small upscale is a lesser evil than an invented pixel, but a large one is not --
    # unrestricted `cover` lost its own A/B by softening faces at a mean 1.25x. This caps it: at
    # 1.15x about a quarter of oversize draws resolve here. 1.0 disables the rung.
    cover_max_scale: float = 1.15
    # Rung 3. What the remainder is padded WITH. "reflect" mirrors real content into the overflow;
    # "black" is PIL's zero fill, i.e. the reference's behaviour, kept for reproduction. Mirroring
    # a figure does duplicate limbs, so this is not free -- it fired on 3 of 413 crops here, which
    # is why the cost is acceptable and why it must stay behind rung 1 rather than in front of it.
    pad_mode: str = "reflect"
    # Invented pixels carry no loss: they stay in the forward pass as context, but no target tells
    # the model to produce them. The narrowest of the four -- it only ever REMOVES supervision on
    # pixels that were never in the image -- and nearly inert once rung 1 is on, by design.
    mask_padding: bool = True

    def __post_init__(self):
        self.canvases = [tuple(c) for c in self.canvases]
        self.mask_ratio = tuple(self.mask_ratio)
        if self.oversize not in OVERSIZE_MODES:
            raise ValueError(f"texture.oversize must be one of {OVERSIZE_MODES}, "
                             f"got {self.oversize!r}")
        if self.pad_mode not in PAD_MODES:
            raise ValueError(f"texture.pad_mode must be one of {PAD_MODES}, "
                             f"got {self.pad_mode!r}")
        if self.cover_max_scale < 1.0:
            raise ValueError(
                f"texture.cover_max_scale must be >= 1.0 (it is an upscale ceiling; 1.0 disables "
                f"the rung), got {self.cover_max_scale}"
            )
        if not self.canvases:
            raise ValueError("texture.canvases is empty")
        for w, h in self.canvases:
            if w <= 0 or h <= 0 or w % 16 or h % 16:
                raise ValueError(f"texture canvas {w}x{h} must be positive multiples of 16 "
                                 f"(the VAE/patch size divisor)")
        # Canvases bypass bucketing entirely, so they never met the check every bucket meets. A
        # canvas over the RoPE ceiling loads cleanly and then dies inside the attention block at
        # the first texture batch -- after model load and dataset scan. Reached for real by scaling
        # the preset list to a 1536 base: 640x1536 becomes 960x2304, and 2304 > 2048.
        from .bucket import verify_max_resolution
        verify_max_resolution(self.canvases)
        lo, hi = self.mask_ratio
        if not (0.0 < lo <= hi <= 1.0):
            raise ValueError(f"texture.mask_ratio must satisfy 0 < lo <= hi <= 1, got {(lo, hi)}")
        if self.energy_power < 0:
            raise ValueError(f"texture.energy_power must be >= 0, got {self.energy_power}")
        if self.energy_downscale < 1:
            raise ValueError(f"texture.energy_downscale must be >= 1, got {self.energy_downscale}")


def describe_feasibility(sizes: list[tuple[int, int]], cfg: TextureConfig) -> str:
    """How much crop freedom this dataset actually offers, per canvas.

    Printed at startup because texture mode degrades *silently*: with no slack the crop is forced,
    energy selection picks from one candidate, and the run looks identical to one that is choosing.
    """
    lines = ["texture  canvas feasibility"]
    for cw, ch in dict.fromkeys(cfg.canvases):
        fits = [(w - cw, h - ch) for w, h in sizes if w >= cw and h >= ch]
        pct = 100 * len(fits) / max(1, len(sizes))
        if fits:
            slack = sorted(min(a, b) for a, b in fits)
            med = slack[len(slack) // 2]
            # Relative to the canvas, not an absolute pixel count: 64px of slack is generous on a
            # 256px canvas and nothing on a 1536px one. Measured for context -- the mixed set's median slack
            # at 1024x1024 is 1001px, a texture set's is 176px.
            note = "   <- no room to choose" if med <= 0.1 * min(cw, ch) else ""
            lines.append(f"         {cw}x{ch}: fits {pct:3.0f}%  median slack {med:4d}px{note}")
        else:
            lines.append(f"         {cw}x{ch}: fits   0%  <- every source is smaller "
                         f"({cfg.oversize})")
    return "\n".join(lines)


def energy_map(img: Image.Image, downscale: int) -> np.ndarray:
    """Detail density, as a non-negative 2D array at 1/downscale scale.

    The Laplacian is taken at **full resolution** and only then box-averaged down. Doing it the
    other way -- downscale first, differentiate second -- is much cheaper and completely wrong for
    this purpose: a resize is a low-pass filter, so every frequency above the downscale factor is
    averaged away *before* it is ever measured. Measured on a test image with 4px stripes and
    `downscale = 8`, that ordering reported near-identical energy for a window containing 23
    columns of detail and one containing 119. Since fine texture is precisely what this is meant
    to find, the blindness would have been total and silent.

    Cost is one full-resolution Laplacian per image, which is still ~10x cheaper than the
    reference's up-to-ten full-canvas Laplacians per sample, and it happens once rather than per
    candidate crop.
    """
    g = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    lap = np.zeros_like(g)
    lap[1:-1, 1:-1] = np.abs(
        g[1:-1, 1:-1] * 4.0 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    )
    if downscale <= 1:
        return lap

    # Box-average |laplace| down to the coarse grid. Pad with zeros to a whole number of blocks so
    # the reshape is exact; the padding adds no energy and so cannot attract a crop.
    h, w = lap.shape
    ph, pw = (-h) % downscale, (-w) % downscale
    if ph or pw:
        lap = np.pad(lap, ((0, ph), (0, pw)))
    bh, bw = lap.shape[0] // downscale, lap.shape[1] // downscale
    return lap.reshape(bh, downscale, bw, downscale).mean(axis=(1, 3))


def choose_crop(
    size: tuple[int, int], canvas: tuple[int, int], energy: np.ndarray,
    cfg: TextureConfig, rng: random.Random,
) -> tuple[int, int]:
    """Top-left of a canvas-sized crop, sampled with probability proportional to detail^power.

    Returns (0, 0) when there is no slack, which is the honest outcome rather than an error --
    `describe_feasibility` is what tells the user the dataset cannot feed this.
    """
    w, h = size
    cw, ch = canvas
    max_x, max_y = max(0, w - cw), max(0, h - ch)
    if max_x == 0 and max_y == 0:
        return 0, 0
    if cfg.energy_power == 0:
        return rng.randint(0, max_x), rng.randint(0, max_y)

    d = cfg.energy_downscale
    eh, ew = energy.shape
    # Integral image -> summed detail inside any candidate window in O(1).
    ii = energy.cumsum(0).cumsum(1)
    ii = np.pad(ii, ((1, 0), (1, 0)))
    # Clamped to the energy map: under `oversize = "pad"` the canvas may be LARGER than the source
    # on one axis while still having slack on the other, and an unclamped window would index past
    # the end of the integral image. When the canvas overruns an axis the whole axis is the window,
    # which is correct -- there is nothing to choose along it.
    bw = min(max(1, cw // d), ew)
    bh = min(max(1, ch // d), eh)
    xs = np.linspace(0, max_x, num=min(max_x + 1, 64), dtype=np.int64)
    ys = np.linspace(0, max_y, num=min(max_y + 1, 64), dtype=np.int64)
    gx = np.clip(xs // d, 0, max(0, ew - bw))
    gy = np.clip(ys // d, 0, max(0, eh - bh))
    box = (ii[np.ix_(gy + bh, gx + bw)] - ii[np.ix_(gy, gx + bw)]
           - ii[np.ix_(gy + bh, gx)] + ii[np.ix_(gy, gx)])

    wts = np.maximum(box, 0.0).astype(np.float64) ** cfg.energy_power
    total = wts.sum()
    if not np.isfinite(total) or total <= 0:
        return rng.randint(0, max_x), rng.randint(0, max_y)
    flat = (wts / total).ravel()
    # rng, not np.random: the caller's per-sample seed is what keeps DDP ranks and dataloader
    # workers from drawing correlated crops (the same reason captions are seeded per sample).
    idx = int(np.searchsorted(flat.cumsum(), rng.random(), side="right"))
    idx = min(idx, flat.size - 1)
    return int(xs[idx % len(xs)]), int(ys[idx // len(xs)])


def fitting_canvases(
    source: tuple[int, int], canvases: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """The canvases this source can hold outright, preserving multiplicity so a canvas listed twice
    keeps its doubled weight. Empty means nothing fits and the cascade has to invent something."""
    w, h = source
    return [c for c in canvases if w >= c[0] and h >= c[1]]


def _reflect_pad(img: Image.Image, cw: int, ch: int) -> Image.Image:
    """Grow the image to at least the canvas by mirroring its own content into the overflow.

    numpy's `reflect` refuses a pad wider than the axis, which a 640px-tall source against a
    1536px canvas would hit, so this uses `symmetric` (which repeats the edge row rather than
    skipping it, and tiles for arbitrary widths).
    """
    a = np.asarray(img)
    py, px = max(0, ch - a.shape[0]), max(0, cw - a.shape[1])
    if not py and not px:
        return img
    # Split the pad across both sides so the real content stays centred rather than pinned to a
    # corner -- a crop is chosen from this afterwards, and a corner-pinned image biases where.
    pad = ((py // 2, py - py // 2), (px // 2, px - px // 2), (0, 0))
    return Image.fromarray(np.pad(a, pad, mode="symmetric"))


def crop_canvas(
    img: Image.Image, canvas: tuple[int, int], cfg: TextureConfig, rng: random.Random,
) -> tuple[Image.Image | None, np.ndarray | None]:
    """A canvas-sized crop, and which of its pixels are real.

    Returns `(crop, valid)`. `valid` is None when every pixel came from the source -- the common
    case, and the one that must stay allocation-free. Otherwise it is a float array of the canvas
    shape, 1.0 where the pixel is real and 0.0 where the cascade had to invent it, which
    `_texture_sample` folds into the loss mask so invented pixels are never a target.

    `(None, None)` means "skip this sample" under `oversize = "skip"`.
    """
    cw, ch = canvas
    w, h = img.size
    valid: Image.Image | None = None

    if w < cw or h < ch:
        scale = max(cw / w, ch / h)
        if cfg.oversize == "skip":
            return None, None
        # Rung 2: a gentle upscale invents no content and beats mirroring or black. Unconditional
        # `oversize = "cover"` keeps its old meaning -- cover at any scale, no cap.
        if cfg.oversize == "cover" or scale <= cfg.cover_max_scale:
            img = img.resize((max(cw, round(w * scale)), max(ch, round(h * scale))), Image.LANCZOS)
        else:
            # Rung 3. Track validity in SOURCE coordinates and crop it with the same box, so the
            # invented region is derived from the geometry rather than recomputed from it -- the
            # two cannot disagree.
            valid = Image.new("L", img.size, 255)
            if cfg.pad_mode == "reflect":
                img, valid = _reflect_pad(img, cw, ch), _zero_pad(valid, cw, ch)
            # "black": fall through and let PIL zero-fill the crop, reproducing the reference.
            # `valid` is smaller than the canvas and zero-fills identically.

    energy = energy_map(img, cfg.energy_downscale)
    x, y = choose_crop(img.size, canvas, energy, cfg, rng)
    box = (x, y, x + cw, y + ch)
    crop = img.crop(box)
    if valid is None:
        return crop, None
    v = np.asarray(valid.crop(box), dtype=np.float32) / 255.0
    # An exactly-covered crop is the common outcome of reflect padding plus a lucky offset; saying
    # so lets the caller skip the mask multiply entirely.
    return crop, (None if float(v.min()) == 1.0 else v)


def _zero_pad(valid: Image.Image, cw: int, ch: int) -> Image.Image:
    """The validity counterpart of `_reflect_pad`: mirrored pixels are padding, not content."""
    a = np.asarray(valid)
    py, px = max(0, ch - a.shape[0]), max(0, cw - a.shape[1])
    if not py and not px:
        return valid
    pad = ((py // 2, py - py // 2), (px // 2, px - px // 2))
    return Image.fromarray(np.pad(a, pad, mode="constant", constant_values=0))


def feather_mask(
    latent_h: int, latent_w: int, cfg: TextureConfig, rng: random.Random,
) -> torch.Tensor:
    """(1, latent_h, latent_w) loss mask: a random sub-region with cosine-tapered edges.

    Zero outside the sub-region on purpose. That area still conditions the forward pass -- it is
    real texture from the same source -- but contributes no target, so it adds context without
    adding gradient variance. Varying the sub-region's position each draw stops the model from
    tying learned content to a fixed place on the canvas.
    """
    lo, hi = cfg.mask_ratio
    ratio = rng.uniform(lo, hi)
    th = max(4, int(latent_h * ratio))
    tw = max(4, int(latent_w * ratio))
    oy = rng.randint(0, latent_h - th)
    ox = rng.randint(0, latent_w - tw)

    patch = torch.ones(th, tw)
    feather = min(cfg.feather_latent_px, th // 2, tw // 2)
    for d in range(feather):
        v = 0.5 * (1.0 - math.cos(math.pi * d / feather))
        patch[d, :] *= v
        patch[th - 1 - d, :] *= v
        patch[:, d] *= v
        patch[:, tw - 1 - d] *= v

    mask = torch.zeros(1, latent_h, latent_w)
    mask[0, oy:oy + th, ox:ox + tw] = patch
    return mask
