"""Dataset and aspect-ratio batch sampler.

Every sample in a batch must share a bucket -- latents of different shapes cannot be stacked -- so
batching is bucket-first: the sampler groups indices by bucket, then yields fixed-size batches
from within each bucket.

Per-bucket micro-batch sizes are supported because cost is not linear in resolution. Attention is
O(tokens^2), and tokens = W*H/256, so a 1536px bucket costs roughly 5x a 1024px one. A single
global batch size either wastes VRAM at low resolution or OOMs at high.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import Dataset, Sampler

from .bucket import BucketManager, verify_max_resolution
from .caption import CaptionConfig, build_caption, read_caption_files
from .texture import TextureConfig, fitting_canvases
from .cache import (
    CACHE_SUFFIX,
    cache_path,
    find_cached_latents,
    image_to_tensor,
    load_and_crop,
    load_cached_latent,
    parse_original_size,
    read_cache_metadata,
)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _mix(*values: int) -> int:
    """Fold ints into one seed. splitmix64's finalizer, so adjacent (epoch, index) pairs -- which
    is exactly what we feed it -- decorrelate instead of producing neighbouring RNG streams."""
    h = 0
    for v in values:
        h = (h * 0x9E3779B97F4A7C15 + (v & 0xFFFFFFFFFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF
        h ^= h >> 30
        h = (h * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        h ^= h >> 27
        h = (h * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        h ^= h >> 31
    return h


@dataclass
class SubsetConfig:
    """One source directory, with its own repeat count and texture eligibility.

    `texture = false` is the load-bearing field and the reason this exists. Texture crops replace
    the image's caption with `texture.trigger` (see `_texture_sample`), which is right for detail
    crops of a captioned subject and catastrophic for a regularization set: a flat colour field
    trained with an empty caption teaches the UNCONDITIONAL branch that colour, and CFG then
    subtracts it -- `pred = uncond + scale*(cond - uncond)` -- so colour anchors delivered through
    texture mode make generations less colourful, not more. Marking a subset `texture = false`
    keeps it in fullres for the whole run, captions intact, whatever the curriculum is doing.

    It is also the honest thing for flat images on their own terms: `energy_map` of a constant
    image is exactly zero, so `choose_crop` falls back to a uniform position. Texture mode exists
    to find native-scale high-frequency detail, and these have none by construction.
    """

    path: str
    num_repeats: int = 1
    texture: bool = True

    def __post_init__(self):
        if self.num_repeats < 1:
            raise ValueError(
                f"subset {self.path!r}: num_repeats must be >= 1, got {self.num_repeats}"
            )


@dataclass
class DatasetConfig:
    # Exactly one of `path` / `subsets`. `path` is the single-directory form and stays the default
    # so every existing config is untouched; `subsets` is the multi-source form.
    path: str | None = None
    subsets: list[SubsetConfig] = field(default_factory=list)
    resolution: int = 1024                 # AREA budget: max_reso = (r, r)
    # Multi-resolution: train every image at every listed area budget. Unset -> [resolution], which
    # is bit-identical to single-tier behaviour. A tier is a repeat with a different resolution, so
    # N tiers cost N x the epoch and N x the cache.
    resolutions: list[int] | None = None
    # What to do when two tiers produce the same bucket for an image. See `_scan_images`.
    tier_collapse: str = "dedup"           # "dedup" | "repeat"
    # Drop sources smaller than this fraction of the SMALLEST tier's area budget. 0.0 = keep all.
    # Anima has no size micro-conditioning, so a tiny source contributes low-token supervision the
    # model cannot attribute to the source being small -- unlike SDXL, which conditions on
    # original_size for exactly this reason.
    min_source_area: float = 0.0
    min_bucket_reso: int = 256
    # PER-SIDE cap. 1920 is Anima's declared spatial RoPE range; it runs to 2048 but extrapolates
    # past what it was configured for. Enforced in both bucketing branches, so a wide source is
    # downscaled to fit rather than producing a bucket the model cannot address.
    max_bucket_reso: int = 1920
    bucket_reso_steps: int = 64
    bucket_no_upscale: bool = True
    multires_training: bool = False
    num_repeats: int = 1
    caption: CaptionConfig = field(default_factory=CaptionConfig)

    # Train from cached latents alone, with the source images deleted. A sample is then
    # `<stem>_WWWWxHHHH_anima.safetensors` + `<stem>.txt` (+ optional `<stem>_nl.txt`); buckets
    # come from the cache filenames rather than from measuring images. "auto" uses latents-only
    # when the directory contains caches but no images, so a run keeps working after a cleanup.
    # "encode" is the odd one out: it buckets from images exactly like "images", but yields *pixels*
    # and has the trainer run the VAE every step instead of reading a cache. It exists for modes
    # where the crop is chosen per step and so cannot be precomputed (see the texture curriculum);
    # for ordinary training it is strictly worse -- a VAE forward every step, forever, plus the VAE
    # resident in VRAM, to reproduce a tensor that never changes.
    source: str = "auto"  # "images" | "latents" | "encode" | "auto"

    # Only read when a curriculum phase has `mode = "texture"`. Nested under [dataset.texture].
    texture: TextureConfig = field(default_factory=TextureConfig)

    def __post_init__(self):
        self.subsets = [
            s if isinstance(s, SubsetConfig) else SubsetConfig(**s) for s in self.subsets
        ]
        if bool(self.path) == bool(self.subsets):
            raise ValueError(
                "dataset needs exactly one of `path` (single directory) or `subsets` "
                "(a list of [[dataset.subsets]] tables), not "
                + ("both" if self.path else "neither")
            )
        seen = {}
        for s in self.subsets:
            if s.path in seen:
                raise ValueError(
                    f"dataset.subsets lists {s.path!r} twice; use num_repeats to weight it instead"
                )
            seen[s.path] = s
        if self.tier_collapse not in ("dedup", "repeat"):
            raise ValueError(
                f"dataset.tier_collapse must be 'dedup' or 'repeat', got {self.tier_collapse!r}"
            )
        if self.min_source_area < 0.0:
            raise ValueError(f"dataset.min_source_area must be >= 0, got {self.min_source_area}")
        if self.resolutions is not None:
            if not self.resolutions:
                raise ValueError("dataset.resolutions is empty; omit it for a single tier")
            for r in self.resolutions:
                if r <= 0:
                    raise ValueError(f"dataset.resolutions contains a non-positive tier: {r}")
                # A square image at tier r wants an r x r bucket, so a tier above the per-side cap
                # can never actually be reached -- it would silently behave as max_bucket_reso.
                if r > self.max_bucket_reso:
                    raise ValueError(
                        f"dataset.resolutions tier {r} exceeds max_bucket_reso "
                        f"{self.max_bucket_reso}; a square image could not reach it"
                    )

    def effective_subsets(self) -> list[SubsetConfig]:
        """Both config forms as one list, so scanning has a single code path.

        The single-`path` form is exactly a one-subset dataset carrying the top-level
        `num_repeats`, which is what keeps every existing config byte-identical in behaviour.
        """
        if self.subsets:
            return self.subsets
        return [SubsetConfig(path=self.path, num_repeats=self.num_repeats, texture=True)]

    @property
    def tiers(self) -> list[int]:
        """Area budgets to train at, ascending. Buckets are monotone non-decreasing in tier, which
        is what lets `_scan_images` dedup by keeping the first occurrence."""
        return sorted(set(self.resolutions)) if self.resolutions else [self.resolution]

    def build_bucket_manager(self, resolution: int | None = None) -> BucketManager:
        bm = BucketManager(
            no_upscale=self.bucket_no_upscale,
            max_reso=((resolution or self.resolution),) * 2,
            min_size=self.min_bucket_reso,
            max_size=self.max_bucket_reso,
            reso_steps=self.bucket_reso_steps,
            multires_training=self.multires_training,
        )
        if not self.bucket_no_upscale:
            bm.make_buckets()
            verify_max_resolution(bm.predefined_resos)
        return bm

    def build_bucket_managers(self) -> dict[int, BucketManager]:
        """One manager per tier. Not shareable: `add_if_new_reso` accumulates per-instance state."""
        return {t: self.build_bucket_manager(t) for t in self.tiers}


@dataclass
class ImageEntry:
    """One trainable sample. `path` is the source image when it exists and is informational only;
    `latent_path` is what actually gets read, so entries stay valid after images are deleted.

    `resolution` is the tier this entry came from -- reporting only, since the bucket already
    determines everything about the sample."""

    path: Path
    latent_path: Path
    bucket: tuple[int, int]
    tags: str
    nl: str | None
    resolution: int = 0
    # The ORIGINAL image size, not the bucket. Only texture mode reads it, to pick a canvas the
    # source can hold; (0, 0) means unknown (latents-only mode, where there may be no image left)
    # and disables fit-aware selection for that entry rather than guessing from the bucket.
    source_size: tuple[int, int] = (0, 0)
    # False for entries from a `texture = false` subset: they stay fullres and keep their captions
    # even while the curriculum is in a texture phase. See `SubsetConfig`.
    texture_ok: bool = True


@dataclass
class TierReport:
    """What the tier cross-product actually produced, so the cost is visible rather than implied."""

    tiers: list[int]
    entries_per_tier: dict[int, int]
    collapsed_per_tier: dict[int, int]
    dropped_small: list[str]
    n_sources: int

    def summary(self) -> str:
        if len(self.tiers) == 1 and not self.dropped_small:
            return ""
        lines = [f"tiers    {self.n_sources} sources x {len(self.tiers)} tier(s)"]
        for t in self.tiers:
            n, c = self.entries_per_tier[t], self.collapsed_per_tier[t]
            note = f"  ({c} collapsed onto a lower tier)" if c else ""
            lines.append(f"         {t:>5}: {n:>6} entries{note}")
        if self.dropped_small:
            lines.append(f"         dropped {len(self.dropped_small)} source(s) below "
                         f"min_source_area (e.g. {self.dropped_small[:3]})")
        return "\n".join(lines)


def _merge_tier_reports(reports: list[TierReport]) -> TierReport | None:
    """Fold per-subset reports into one, so the startup summary describes the run and not its
    last subset. Tiers are dataset-wide (they come from `DatasetConfig`), so the keys always
    agree and the merge is a plain sum."""
    if not reports:
        return None
    if len(reports) == 1:
        return reports[0]
    tiers = reports[0].tiers
    return TierReport(
        tiers=tiers,
        entries_per_tier={t: sum(r.entries_per_tier.get(t, 0) for r in reports) for t in tiers},
        collapsed_per_tier={t: sum(r.collapsed_per_tier.get(t, 0) for r in reports) for t in tiers},
        dropped_small=[n for r in reports for n in r.dropped_small],
        n_sources=sum(r.n_sources for r in reports),
    )


class AnimaDataset(Dataset):
    """Serves cached latents plus a freshly-built caption.

    The caption is rebuilt per __getitem__ rather than precomputed, so tag shuffling and dropout
    produce a different view of the same image on every epoch. That is the whole reason text
    embeddings are not cached by default.
    """

    def __init__(self, config: DatasetConfig, require_cache: bool = True, caption_seed: int = 0):
        self.config = config
        self.bucket_managers = config.build_bucket_managers()
        # Back-compat alias: single-tier callers (the cache CLI, older tests) want one manager.
        self.bucket_mgr = self.bucket_managers[config.tiers[0]]
        self.entries: list[ImageEntry] = []
        self.require_cache = require_cache
        self.caption_seed = caption_seed
        self.texture_cfg = config.texture
        self.tier_report: TierReport | None = None
        self._scan()

    def _scan(self) -> None:
        subsets = self.config.effective_subsets()
        resolved = [(s, self._resolve_source(Path(s.path))) for s in subsets]

        # `auto` is resolved per directory, so two subsets can legitimately disagree -- but a batch
        # carries either `pixels` or `latents`, never both (`collate` picks one key), so a mixed
        # dataset would fail at the first batch that happened to span them. Refuse up front and say
        # which subset disagreed, rather than at some random step.
        modes = {m for _, m in resolved}
        if len(modes) > 1:
            detail = ", ".join(f"{Path(s.path).name}={m}" for s, m in resolved)
            raise ValueError(
                f"dataset.subsets resolve to different source modes ({detail}). A batch is either "
                f"pixels or latents, so all subsets must agree. Set `dataset.source` explicitly "
                f"instead of leaving it 'auto'."
            )

        # "encode" reads pixels at train time, so a cache is not merely optional -- requiring one
        # would defeat the mode. Recorded on self because __getitem__ has to know which it is.
        self.encode_on_the_fly = modes.pop() == "encode"
        if self.encode_on_the_fly:
            self.require_cache = False

        reports: list[TierReport] = []
        self.subset_report: list[tuple[SubsetConfig, int]] = []
        for sub, mode in resolved:
            root = Path(sub.path)
            start = len(self.entries)
            self.tier_report = None   # so a subset's report can never be counted twice
            if mode == "latents":
                self._scan_latents(root)
            else:
                self._scan_images(root)
            fresh = self.entries[start:]
            if not sub.texture:
                for e in fresh:
                    e.texture_ok = False
            # Per-subset repeats, applied to that subset's slice only. This is the knob that
            # balances a small regularization set against a large training set.
            if sub.num_repeats > 1:
                self.entries.extend(fresh * (sub.num_repeats - 1))
            self.subset_report.append((sub, len(self.entries) - start))
            if self.tier_report is not None:
                reports.append(self.tier_report)

        verify_max_resolution({e.bucket for e in self.entries})
        self.tier_report = _merge_tier_reports(reports)

    def _resolve_source(self, root: Path) -> str:
        if not root.exists():
            raise FileNotFoundError(f"dataset path not found: {root}")
        source = self.config.source
        if source == "auto":
            has_images = any(p.suffix.lower() in IMAGE_EXTENSIONS for p in root.iterdir())
            source = "images" if has_images else "latents"
        if source not in ("images", "latents", "encode"):
            raise ValueError(f"unknown source mode: {source!r}")
        return source

    def _scan_latents(self, root: Path) -> None:
        """Build entries from cached latents; source images need not exist.

        Multiple cached buckets per stem is the *normal* case under multi-resolution, so the old
        "more than one valid bucket -> stale, delete them" rule cannot simply be relaxed or the
        stale-cache guard goes with it. Instead each cache is attributed to the tier that explains
        it: `original_size` from the cache metadata re-derives what `select_bucket` would have
        chosen at every declared tier. A file matching no tier is stale; a tier with no file is
        missing. Exact either way, and it works on caches already on disk.
        """
        cached = find_cached_latents(root)
        if not cached:
            raise ValueError(
                f"no cached latents in {root} (looked for *{CACHE_SUFFIX}). "
                f"Cache with source='images' first."
            )

        tiers = self.config.tiers
        min_px = self._min_source_pixels()
        per_tier = {t: 0 for t in tiers}
        collapsed = {t: 0 for t in tiers}
        dropped: list[str] = []
        skipped: list[str] = []
        stale: list[str] = []
        missing: list[str] = []
        unattributed = 0

        for stem, variants in sorted(cached.items()):
            caption_path = root / f"{stem}.txt"
            if not caption_path.exists():
                skipped.append(stem)
                continue

            by_bucket = {b: p for p, b in variants}
            size = None
            for p, _ in variants:
                size = parse_original_size(read_cache_metadata(p))
                if size:
                    break

            if size is None:
                # Pre-metadata cache: fall back to the area filter. Cannot attribute tiers, so
                # multi-res is refused rather than guessed at.
                unattributed += 1
                if len(tiers) > 1:
                    raise RuntimeError(
                        f"{stem}: cached latents carry no `original_size` metadata, so they cannot "
                        f"be attributed to a tier. Re-cache to use multiple resolutions."
                    )
                wanted = [(p, b) for p, b in variants if self._bucket_allowed(b)]
                if not wanted:
                    skipped.append(stem)
                    continue
                if len(wanted) > 1:
                    raise RuntimeError(
                        f"{stem}: {len(wanted)} cached buckets are all valid for this config "
                        f"({[b for _, b in wanted]}). Delete the stale ones."
                    )
                wanted_pairs = [(tiers[0], wanted[0][1])]
            else:
                w, h = size
                if min_px and w * h < min_px:
                    dropped.append(stem)
                    continue
                for t in self._collapsed_tiers(w, h):
                    collapsed[t] += 1
                wanted_pairs = self._tier_buckets(w, h)

                expected = {b for _, b in wanted_pairs}
                for b in sorted(set(by_bucket) - expected):
                    stale.append(f"{stem} @ {b[0]}x{b[1]}")
                for b in sorted(expected - set(by_bucket)):
                    missing.append(f"{stem} @ {b[0]}x{b[1]}")

            nl_path = root / f"{stem}_nl.txt"
            tags = caption_path.read_text(encoding="utf-8").strip()
            nl = nl_path.read_text(encoding="utf-8").strip() if nl_path.exists() else None
            for tier, bucket in wanted_pairs:
                p = by_bucket.get(bucket)
                if p is None:
                    continue  # reported via `missing` below
                per_tier[tier] += 1
                self.entries.append(
                    ImageEntry(
                        path=p,  # the cache file itself; there may be no image any more
                        latent_path=p, bucket=bucket, tags=tags, nl=nl, resolution=tier,
                    )
                )

        if missing:
            raise RuntimeError(
                f"{len(missing)} declared (source, tier) pair(s) have no cached latent (e.g. "
                f"{missing[:4]}). Cache every tier ({' '.join(str(t) for t in tiers)}) first."
            )
        if stale:
            # Warn, do not raise. Exact tier attribution means an unmatched cache is never *chosen*
            # -- it is simply not used -- so ignoring it is safe, and a genuinely stale cache still
            # hard-fails above via `missing`. Erroring here would also make latents mode disagree
            # with images mode, which ignores extra cache files silently, and would forbid the
            # normal workflow of keeping several tiers cached and switching between them in config.
            import warnings

            warnings.warn(
                f"{len(stale)} cached latent(s) match no declared tier and are being ignored "
                f"(e.g. {stale[:3]}). That is expected if you cached other resolutions; delete "
                f"them to reclaim the disk, or add their tier to dataset.resolutions to train on "
                f"them.",
                stacklevel=3,
            )
        if not self.entries:
            raise ValueError(f"{root}: found {len(cached)} cached latents but none had a .txt caption")
        if skipped:
            raise RuntimeError(
                f"{len(skipped)} cached latents have no usable .txt caption or bucket "
                f"(e.g. {skipped[:5]}). They would be silently dropped; fix or remove them."
            )

        self.tier_report = TierReport(
            tiers=tiers, entries_per_tier=per_tier, collapsed_per_tier=collapsed,
            dropped_small=dropped, n_sources=len(cached) - len(dropped),
        )
        if not unattributed:
            self._warn_collapse(self.tier_report)

    def _bucket_allowed(self, bucket: tuple[int, int]) -> bool:
        """Legacy filter, used only for caches predating `original_size` metadata.

        Deliberately does NOT enforce `min_bucket_reso`: `select_bucket`'s no_upscale branch never
        applies a minimum side, so requiring one here rejected caches the image path had happily
        created (a 200x1500 source yields a 192x1472 bucket). The two paths now agree.
        """
        c = self.config
        return (
            max(bucket) <= c.max_bucket_reso
            and bucket[0] % c.bucket_reso_steps == 0
            and bucket[1] % c.bucket_reso_steps == 0
            and bucket[0] * bucket[1] <= c.resolution * c.resolution
        )

    def _tier_buckets(self, w: int, h: int) -> list[tuple[int, tuple[int, int]]]:
        """(tier, bucket) for one source, with `tier_collapse` applied.

        `bucket_no_upscale` caps a bucket by the *source*, so an image already under a tier's area
        budget is left untouched and every tier above it yields the identical bucket. Buckets are
        therefore monotone non-decreasing in tier and duplicates form a contiguous run at the top,
        which is why keeping the first occurrence is the correct dedup.

        This is a per-image *ceiling*, not a floor -- `min_bucket_reso` has no bearing on it, since
        `select_bucket`'s no_upscale branch never reads `min_size`.
        """
        out: list[tuple[int, tuple[int, int]]] = []
        seen: set[tuple[int, int]] = set()
        for tier in self.config.tiers:
            bucket = self.bucket_managers[tier].select_bucket(w, h)[0]
            if self.config.tier_collapse == "dedup" and bucket in seen:
                continue
            seen.add(bucket)
            out.append((tier, bucket))
        return out

    def _collapsed_tiers(self, w: int, h: int) -> list[int]:
        """Tiers whose bucket duplicates a lower tier's -- i.e. that bought no new resolution."""
        collapsed, seen = [], set()
        for tier in self.config.tiers:
            bucket = self.bucket_managers[tier].select_bucket(w, h)[0]
            if bucket in seen:
                collapsed.append(tier)
            seen.add(bucket)
        return collapsed

    def _min_source_pixels(self) -> float:
        cfg = self.config
        return cfg.min_source_area * min(cfg.tiers) ** 2 if cfg.min_source_area else 0.0

    def _warn_collapse(self, report: TierReport) -> None:
        """Multi-res degrading into plain `num_repeats` is invisible otherwise: it costs the same
        VRAM, disk and wall-clock while producing no extra resolution. Say so."""
        if len(report.tiers) < 2 or not report.n_sources:
            return
        import warnings

        for tier in report.tiers:
            frac = report.collapsed_per_tier[tier] / report.n_sources
            if frac >= 0.01:
                warnings.warn(
                    f"tier {tier} exceeds the native area of {frac * 100:.1f}% of your sources "
                    f"({report.collapsed_per_tier[tier]}/{report.n_sources}); those images "
                    f"duplicate a lower tier's bucket rather than gaining resolution"
                    + ("  -- deduplicated, so they simply get fewer repeats."
                       if self.config.tier_collapse == "dedup"
                       else "  -- tier_collapse='repeat', so they are trained twice at the same "
                            "bucket.")
                    + " Lower the top tier, or check `cache_latents.py audit` for a ladder that "
                      "fits this dataset's size distribution.",
                    stacklevel=3,
                )

    def _scan_images(self, root: Path) -> None:
        files = sorted(
            p for p in root.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.endswith(".safetensors")
        )
        if not files:
            raise ValueError(f"no images found in {root}")

        from PIL import Image

        tiers = self.config.tiers
        min_px = self._min_source_pixels()
        per_tier = {t: 0 for t in tiers}
        collapsed = {t: 0 for t in tiers}
        dropped: list[str] = []
        missing_cache: list[tuple[str, tuple[int, int]]] = []

        for p in files:
            with Image.open(p) as im:
                w, h = im.size

            if min_px and w * h < min_px:
                dropped.append(p.name)
                continue

            for t in self._collapsed_tiers(w, h):
                collapsed[t] += 1

            tags = nl = None
            for tier, bucket in self._tier_buckets(w, h):
                if self.require_cache and not cache_path(p, bucket).exists():
                    missing_cache.append((p.name, bucket))
                    continue
                if tags is None:
                    tags, nl = read_caption_files(p)
                per_tier[tier] += 1
                self.entries.append(
                    ImageEntry(
                        path=p, latent_path=cache_path(p, bucket), bucket=bucket,
                        tags=tags, nl=nl, resolution=tier, source_size=(w, h),
                    )
                )

        if missing_cache:
            shown = ", ".join(f"{n} @ {b[0]}x{b[1]}" for n, b in missing_cache[:4])
            raise RuntimeError(
                f"{len(missing_cache)} (image, bucket) pair(s) have no cached latent (e.g. "
                f"{shown}). Run latent caching for every tier "
                f"({' '.join(str(t) for t in tiers)}) first."
            )

        self.tier_report = TierReport(
            tiers=tiers, entries_per_tier=per_tier, collapsed_per_tier=collapsed,
            dropped_small=dropped, n_sources=len(files),
        )
        self._warn_collapse(self.tier_report)

    def bucket_of(self, index: int) -> tuple[int, int]:
        return self.entries[index].bucket

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int | tuple[int, int]) -> dict:
        # The sampler yields (index, epoch) so the caption RNG can be seeded from the sample
        # itself. Deriving it from global `random` would be wrong twice over: forked DataLoader
        # workers inherit the parent's RNG state, and every DDP rank forks from the same seed --
        # so ranks would draw *identical* tag shuffles and dropouts. That silently collapses
        # caption diversity instead of failing, which is why it is seeded explicitly here.
        phase = canvas = None
        if isinstance(index, tuple):
            index, epoch, phase, canvas = (index + (None, None))[:4]
        else:
            index, epoch = index, 0
        e = self.entries[index]
        rng = random.Random(_mix(self.caption_seed, epoch, index))

        if canvas is not None:
            return self._texture_sample(e, canvas, rng)

        caption = build_caption(e.tags, e.nl, self.config.caption, rng)
        sample = {"caption": caption, "bucket": e.bucket, "path": str(e.path)}

        if self.encode_on_the_fly:
            # Crop through the same helper the cacher uses, and against this entry's *own* tier's
            # bucket manager -- otherwise a multi-tier run would re-derive one bucket for every
            # tier and silently break the sampler's bucket-homogeneity guarantee.
            img, _ = load_and_crop(e.path, self.bucket_managers[e.resolution])
            sample["pixels"] = image_to_tensor(img).squeeze(0)  # (3, 1, H, W) in [-1, 1]
        else:
            sample["latents"] = load_cached_latent(e.latent_path)
        return sample


    def _texture_sample(self, e: ImageEntry, canvas: tuple[int, int], rng: random.Random) -> dict:
        """A canvas-sized crop chosen by detail content, plus its feathered loss mask.

        Always encodes live -- the crop is chosen per step from image content, so no cache can
        express it. That is the entire reason `source = "encode"` exists.
        """
        from PIL import Image as _Image

        from .texture import crop_canvas, feather_mask

        with _Image.open(e.path) as im:
            img = im.convert("RGB")
        crop, valid = crop_canvas(img, canvas, self.texture_cfg, rng)
        if crop is None:
            # `oversize = "skip"` on an image that cannot hold the canvas. Returning a shape the
            # batch cannot stack would be worse, so fall back to covering and say so once.
            from .texture import TextureConfig
            cover = TextureConfig(**{**self.texture_cfg.__dict__, "oversize": "cover"})
            crop, valid = crop_canvas(img, canvas, cover, rng)

        cw, ch = canvas
        mask = feather_mask(ch // 8, cw // 8, self.texture_cfg, rng)
        if valid is not None and self.texture_cfg.mask_padding:
            # Invented pixels stay in the forward pass as context but carry no target. Reduced to
            # latent resolution by AREA (mean over each 8x8 block), not by sampling, so a latent
            # cell that is half padding is weighted 0.5 rather than being all-or-nothing on
            # whichever pixel a nearest-neighbour resize happened to land on.
            v = torch.from_numpy(valid).view(ch // 8, 8, cw // 8, 8).mean(dim=(1, 3))
            mask = mask * v
        return {
            # Trigger only, never the image's tags -- see TextureConfig.trigger for why this is
            # load-bearing rather than a convenience.
            "caption": self.texture_cfg.trigger,
            "bucket": canvas,
            "path": str(e.path),
            "pixels": image_to_tensor(crop).squeeze(0),
            "mask": mask,
        }


def collate(batch: list[dict]) -> dict:
    buckets = {b["bucket"] for b in batch}
    if len(buckets) != 1:
        raise RuntimeError(f"batch mixes buckets {buckets}; the sampler must group by bucket")
    key = "pixels" if "pixels" in batch[0] else "latents"
    out = {
        key: torch.stack([b[key] for b in batch]),
        "captions": [b["caption"] for b in batch],
        "bucket": batch[0]["bucket"],
        "paths": [b["path"] for b in batch],
    }
    if "mask" in batch[0]:
        out["mask"] = torch.stack([b["mask"] for b in batch])
    return out


class BucketBatchSampler(Sampler[list[int]]):
    """Yields batches whose members all share a bucket.

    `batch_size` is an int, or a map keyed by **resolution tier** -- the values in
    `dataset.resolutions` (or the single `dataset.resolution`), matched exactly:

        resolutions = [768, 1024, 1280]
        batch_size  = { 768 = 16, 1024 = 12, 1280 = 8 }

    Keyed on the tier, not on the bucket's longest side, because the tier is what predicts cost and
    the longest side is not. Bucketing targets a constant *area*, so `tier^2 / 256` is an exact
    upper bound on a tier's token count (measured on a mixed set: tiers 768/1024/1280 top out at exactly
    2304/4096/6400 tokens) while the longest side ranges overlap almost completely across tiers --
    704-1280 at tier 768 against 768-1792 at tier 1024. Sorting by longest side therefore sorts by
    something that is not the cost.

    This used to be a threshold lookup on `max(bucket)`, where each key was a floor covering
    everything up to the next one. That reads like a label and behaves like a range: `{512: 32,
    1024: 12}` silently gave a 960x960 bucket 32 images -- 2.4x the load of the 1024x1024 bucket
    next to it, and an OOM several minutes into a run. Exact tier matching cannot express that, and
    `load_config` rejects a key that is not a declared tier.

    A bucket can host more than one tier (two different sources can land on the same bucket from
    different tiers). Cost depends only on the bucket, so any of their sizes would do; the largest
    tier present wins, which is the smaller batch and therefore the safe direction.

    **This sampler is rank-agnostic on purpose.** `accelerator.prepare(dataloader)` wraps the batch
    sampler in Accelerate's `BatchSamplerShard`, which hands whole batches to each rank round-robin
    and pads with `even_batches` so ranks stay in lockstep. Because it distributes *whole* batches
    it preserves bucket homogeneity, so there is nothing for us to add. Sharding here as well would
    shard twice -- each rank would see roughly 1/world_size^2 of the data, which shows up only as a
    quietly reduced step count rather than an error.
    """

    def __init__(
        self,
        dataset: AnimaDataset,
        batch_size: int | dict[int, int],
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        self._by_tier = dict(batch_size) if isinstance(batch_size, dict) else None
        self._flat = None if isinstance(batch_size, dict) else int(batch_size)

        # `bucket_indices` stays keyed by bucket alone: it is what the trainer reads to size
        # `use_quantized_matmul` and to report the bucket table, and both want the whole dataset.
        # `_groups` is the finer split that batching uses.
        self.bucket_indices: dict[tuple[int, int], list[int]] = {}
        self._groups: dict[tuple[tuple[int, int], bool], list[int]] = {}
        self._bucket_tier: dict[tuple[int, int], int] = {}
        for i in range(len(dataset)):
            e = dataset.entries[i]
            bucket = e.bucket
            self.bucket_indices.setdefault(bucket, []).append(i)
            self._groups.setdefault((bucket, e.texture_ok), []).append(i)
            self._bucket_tier[bucket] = max(self._bucket_tier.get(bucket, 0), e.resolution)

    def batch_size_for(self, bucket: tuple[int, int]) -> int:
        if self._flat is not None:
            return self._flat
        tier = self._bucket_tier.get(bucket, 0)
        if tier in self._by_tier:
            return self._by_tier[tier]
        # Unreachable through `load_config`, which requires the keys to be exactly the declared
        # tiers. Falling back to the smallest configured size keeps a hand-built sampler safe
        # rather than silently generous.
        return min(self._by_tier.values())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        batches: list[list[int]] = []
        # Grouped by (bucket, texture eligibility), not bucket alone. A batch gets ONE canvas, so a
        # batch that mixed a texture-eligible entry with an exempt one could not honour both -- the
        # exempt entry would either be cropped anyway or force the whole batch to fullres. Splitting
        # here makes the guarantee structural: `__iter__` can read the flag off any member.
        for (bucket, _), indices in self._groups.items():
            indices = list(indices)
            if self.shuffle:
                rng.shuffle(indices)
            bs = self.batch_size_for(bucket)
            for i in range(0, len(indices), bs):
                chunk = indices[i : i + bs]
                if len(chunk) < bs and self.drop_last:
                    continue
                batches.append(chunk)
        if self.shuffle:
            rng.shuffle(batches)  # decorrelate bucket order across the epoch

        # No rank sharding and no truncation: Accelerate does both. A trailing partial
        # accumulation group is fine -- Accelerate forces a gradient sync at end_of_dataloader
        # (`Accelerator._do_sync`), so the remainder steps and no sample is dropped. This is the
        # same behaviour sd-scripts relies on.
        return batches

    def set_curriculum(self, curriculum, total_epochs: int, texture=None) -> None:
        """Let the sampler resolve the curriculum phase for each batch it emits.

        The phase has to reach `__getitem__`, and `__getitem__` runs in forked workers that
        prefetch ahead -- so setting a flag on the dataset from the training loop would arrive
        late by however deep the prefetch queue is (8 workers x prefetch 2 is ~16 batches, about 4
        optimizer steps at accum 4, against 5%-of-run phases only ~21 steps long). Resolving here
        instead is exact: the sampler runs in the main process and knows its own batch ordering,
        and `batches_consumed / total_batches` is *identically* the trainer's
        `global_step / total_steps` -- both divide out accumulation and world size. So the mode the
        dataset used and the t_range the trainer applied always come from the same phase, with no
        shared state and no cross-rank communication.
        """
        from .texture import TextureConfig
        self.curriculum = curriculum
        self.total_epochs = max(1, total_epochs)
        self.texture = texture or TextureConfig()

    def _phase_for(self, batch_ordinal: int) -> object | None:
        if not getattr(self, "curriculum", None):
            return None
        total = max(1, len(self._batches()) * self.total_epochs)
        consumed = self.epoch * len(self._batches()) + batch_ordinal
        return self.curriculum.resolve(consumed / total)

    def __iter__(self):
        # (index, epoch, phase, canvas): the dataset needs the epoch to seed its caption RNG, the
        # phase to know whether this is a texture crop, and the canvas so every sample in the
        # batch is the same shape. The canvas is drawn *here*, once per batch, for that reason --
        # drawing it per sample would produce a batch that cannot be stacked.
        rng = random.Random(self.seed + self.epoch)
        for ordinal, batch in enumerate(self._batches()):
            phase = self._phase_for(ordinal)
            canvas = None
            # `_batches` guarantees the batch is homogeneous in `texture_ok`, so member 0 speaks for
            # all of them. A `texture = false` subset therefore stays fullres -- captions intact --
            # no matter what the curriculum says.
            if (phase is not None and getattr(phase, "mode", "fullres") == "texture"
                    and self.dataset.entries[batch[0]].texture_ok):
                canvas = self._choose_canvas(batch, rng)
            yield [(i, self.epoch, phase, canvas) for i in batch]

    def _choose_canvas(self, batch: list[int], rng: random.Random) -> tuple[int, int]:
        """Pick the batch's canvas -- from those every image in it can hold, when possible.

        Drawing blind and forcing the fit is what makes padding common (56% of draws on one texture set);
        drawing from what fits makes it rare (1%), without upscaling anything. The choice is per
        BATCH, not per sample, because the batch has to stack -- so a larger batch has fewer
        canvases available to it, and at some size the intersection empties and this degrades to
        the blind draw. That is a real cost of batching in texture mode, and it degrades smoothly
        rather than failing.
        """
        canvases = self.texture.canvases
        if not getattr(self.texture, "fit_aware", True):
            return rng.choice(canvases)
        sizes = [self.dataset.entries[i].source_size for i in batch]
        if any(s == (0, 0) for s in sizes):
            return rng.choice(canvases)          # unknown source size: cannot reason about fit
        w, h = min(s[0] for s in sizes), min(s[1] for s in sizes)
        fits = fitting_canvases((w, h), canvases)
        # No canvas fits every image: fall back to the blind draw and let the crop cascade handle
        # it. Choosing the "least bad" canvas here instead would quietly bias the aspect mix toward
        # whatever the smallest image in the batch happens to be.
        return rng.choice(fits or canvases)

    def __len__(self) -> int:
        return len(self._batches())
