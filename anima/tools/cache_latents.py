"""Latent caching and cache auditing.

    python -m anima.tools.cache_latents cache /path/to/dataset --resolution 1024
    python -m anima.tools.cache_latents audit /path/to/dataset

Caching is a prerequisite for training: the dataset layer reads latents, never images. It is also
where bucket assignment is frozen, so a change to `resolution` / `bucket_reso_steps` /
`bucket_no_upscale` invalidates the cache and needs a re-run.

`audit` exists because of the latents-only workflow. Deleting source images is irreversible, and
the two ways to lose data silently are an image with no cache and a cache with no caption. It
reports both and refuses to bless the directory unless neither exists.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from ..training.config import DEFAULT_MODEL_PATH
from ..data.bucket import ROPE_DECLARED_PIXELS, verify_max_resolution
from ..data.cache import CACHE_SUFFIX, LatentCacher, audit_cache, cache_path
from ..data.dataset import IMAGE_EXTENSIONS, DatasetConfig


def _images(root: Path) -> list[Path]:
    return sorted(
        p for p in root.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.endswith(CACHE_SUFFIX)
    )


def cmd_cache(args) -> int:
    root = Path(args.path)
    if not root.is_dir():
        print(f"not a directory: {root}")
        return 1

    files = _images(root)
    if not files:
        print(f"no images in {root}")
        return 1

    tiers = sorted(set(args.resolution))
    cfg = DatasetConfig(
        path=str(root),
        resolutions=tiers,
        min_bucket_reso=args.min_bucket_reso,
        max_bucket_reso=args.max_bucket_reso,
        bucket_reso_steps=args.bucket_reso_steps,
        bucket_no_upscale=not args.upscale,
        multires_training=args.multires,
    )
    managers = cfg.build_bucket_managers()

    from PIL import Image

    # Assign every bucket first: a RoPE-illegal bucket must fail before we write 2000 files.
    # Dedup per (image, bucket) matches what the dataset will actually ask for -- a tier above an
    # image's native area yields the same bucket, hence the same file.
    plan: dict[Path, list[tuple[int, tuple[int, int]]]] = {}
    sizes: dict[Path, tuple[int, int]] = {}
    collapsed = {t: 0 for t in tiers}
    for p in files:
        with Image.open(p) as im:
            sizes[p] = im.size
        seen, keep = set(), []
        for t in tiers:
            b = managers[t].select_bucket(*sizes[p])[0]
            if b in seen:
                collapsed[t] += 1
                continue
            seen.add(b)
            keep.append((t, b))
        plan[p] = keep
    verify_max_resolution({b for v in plan.values() for _, b in v})

    print(f"\n{len(files)} images x {len(tiers)} tier(s) -> "
          f"{sum(len(v) for v in plan.values())} cache files")
    for t in tiers:
        bs = [b for v in plan.values() for tt, b in v if tt == t]
        mb = sum(16 * 4 * (b[0] * b[1] // 64) for b in bs) / 1e6
        note = f", {collapsed[t]} collapsed onto a lower tier" if collapsed[t] else ""
        print(f"  tier {t:>5}: {len(bs):>5} images, {len(set(bs)):>4} buckets, ~{mb:>7.0f} MB{note}")

    counts: dict[tuple[int, int], int] = {}
    for v in plan.values():
        for _, b in v:
            counts[b] = counts.get(b, 0) + 1
    print(f"\n{len(counts)} distinct buckets overall")
    for b, n in sorted(counts.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {b[0]:>5}x{b[1]:<5} {n:>5}")
    if len(counts) > 12:
        print(f"  ... {len(counts) - 12} more")

    missing_caption = [p.name for p in files if not p.with_suffix(".txt").exists()]
    if missing_caption and not args.allow_missing_captions:
        print(f"\n{len(missing_caption)} images have no .txt caption "
              f"(e.g. {missing_caption[:3]}). They would be unusable for training.")
        print("Re-run with --allow-missing-captions to cache them anyway.")
        return 1

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    from ..modeling.loader import load_components

    print(f"\nloading VAE from {args.model_path}")
    components = load_components(
        args.model_path, dtype=torch.bfloat16,
        load_text_encoder=False, load_vae=True, load_tokenizers=False,
    )
    cacher = LatentCacher(components.vae, device=args.device)

    total = sum(len(v) for v in plan.values())
    done = skipped = 0
    for i, p in enumerate(files, 1):
        for tier, bucket in plan[p]:
            if cache_path(p, bucket).exists() and not args.overwrite:
                skipped += 1
                continue
            cacher.cache_image(p, managers[tier], overwrite=args.overwrite)
            done += 1
        if i % 25 == 0 or i == len(files):
            print(f"  {i}/{len(files)} images  cached {done}, skipped {skipped} "
                  f"(of {total})", flush=True)

    print(f"\ncached {done}, skipped {skipped} (already present)")
    return 0


def _source_sizes(root: Path) -> list[tuple[int, int]]:
    """Source dimensions, from the images if present, else from the caches' `original_size`."""
    from PIL import Image

    files = _images(root)
    if files:
        out = []
        for p in files:
            with Image.open(p) as im:
                out.append(im.size)
        return out

    from ..data.cache import find_cached_latents, parse_original_size, read_cache_metadata

    out = []
    for variants in find_cached_latents(root).values():
        size = parse_original_size(read_cache_metadata(variants[0][0]))
        if size:
            out.append(size)
    return out


def _report_size_distribution(root: Path, steps: int) -> None:
    """Print the source-area distribution and propose a ladder.

    This is the number that decides whether multi-resolution does anything at all. With
    `bucket_no_upscale`, a tier above an image's native area yields the *same* bucket as the tier
    below it -- so a ladder sitting above the dataset's size distribution silently degrades into
    plain `num_repeats` at full VRAM, disk and wall-clock cost.
    """
    sizes = _source_sizes(root)
    if not sizes:
        return
    areas = sorted(w * h for w, h in sizes)

    def q(p: float) -> int:
        return areas[min(int(p * len(areas)), len(areas) - 1)]

    def side(area: int) -> int:
        return max(64, int(area ** 0.5) // steps * steps)

    print(f"\nsource-area distribution ({len(areas)} sources)")
    print("   pct     MP    tier at this source's own ceiling")
    for p in (0.01, 0.05, 0.10, 0.25, 0.50, 0.90):
        a = q(p)
        print(f"   p{int(p * 100):<4} {a / 1e6:>6.2f}   {side(a):>5}px")

    # Top rung at the p10 ceiling, so ~10% of sources collapse there. Lower rungs step *down*
    # geometrically rather than clustering under the top: every tier below an image's own ceiling
    # produces a genuinely different bucket, so spreading down is what buys real resolution
    # variation. Spacing is geometric because cost scales with area, not with side length.
    top = min(side(q(0.10)), ROPE_DECLARED_PIXELS)
    ladder = sorted({max(steps, side(int((top / 1.25 ** k) ** 2))) for k in range(3)})
    over = sum(1 for a in areas if a < ladder[-1] ** 2)
    print(f"\n   suggested ladder: resolutions = {ladder}")
    print(f"   -> {over}/{len(areas)} sources ({100 * over / len(areas):.1f}%) sit below the top "
          f"rung, so they collapse onto a lower tier there and simply get fewer repeats.")
    print("   Lower the top rung to shrink that; raise it to train higher at that cost.")


def cmd_audit(args) -> int:
    root = Path(args.path)
    report = audit_cache(root, IMAGE_EXTENSIONS)

    print(f"\n{root}")
    print(f"  trainable (cache + caption)  {len(report.trainable)}")
    print(f"  cached but no caption        {len(report.missing_caption)}")
    print(f"  images with no cache         {len(report.uncached_images)}")
    print(f"  cached under >1 bucket       {len(report.multi_bucket)}")
    print(f"  cache size                   {report.total_cache_bytes / 1e6:.1f} MB")
    print(f"  image size                   {report.total_image_bytes / 1e6:.1f} MB")
    if report.total_cache_bytes:
        print(f"  ratio                        "
              f"{report.total_image_bytes / report.total_cache_bytes:.1f}x smaller")

    for label, items in (("no caption", report.missing_caption),
                         ("no cache", report.uncached_images)):
        if items:
            print(f"\n  {label}: {items[:5]}{' ...' if len(items) > 5 else ''}")
    if report.multi_bucket:
        n = len(report.multi_bucket)
        widths = sorted({len(v) for v in report.multi_bucket.values()})
        print(f"\n  {n} stem(s) cached under multiple buckets ({widths} each), e.g. "
              f"{dict(list(report.multi_bucket.items())[:2])}")
        print("  Expected under multi-resolution -- each is one tier. Otherwise they are stale")
        print("  caches from an earlier config; the trainer ignores unmatched ones with a warning.")

    _report_size_distribution(root, args.bucket_reso_steps)

    if report.safe_to_delete_images:
        print("\nSAFE to delete source images: every image is cached and every cache has a caption.")
        print("Training then continues with source = \"latents\" (or \"auto\").")
        return 0

    print("\nNOT safe to delete source images -- the items listed above would be lost.")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Anima latent cache tools")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("cache", help="encode images to cached latents")
    c.add_argument("path")
    c.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    c.add_argument("--resolution", type=int, nargs="+", default=[1024],
                   help="AREA budget(s), not side lengths. Several values cache every image at "
                        "every tier, which is what dataset.resolutions then trains on.")
    c.add_argument("--min-bucket-reso", type=int, default=256)
    c.add_argument("--max-bucket-reso", type=int, default=1920,
                   help="PER-SIDE cap; 1920 = in spec, 2048 = hard RoPE limit")
    c.add_argument("--bucket-reso-steps", type=int, default=64)
    c.add_argument("--upscale", action="store_true", help="allow upscaling small images")
    c.add_argument("--multires", action="store_true", help="area tie-break between buckets")
    c.add_argument("--overwrite", action="store_true")
    c.add_argument("--allow-missing-captions", action="store_true")
    c.add_argument("--dry-run", action="store_true",
                   help="report the bucket plan and cache size, write nothing")
    c.add_argument("--device", default="cuda")
    c.set_defaults(func=cmd_cache)

    a = sub.add_parser("audit", help="check a directory before deleting its images")
    a.add_argument("path")
    a.add_argument("--bucket-reso-steps", type=int, default=64)
    a.set_defaults(func=cmd_audit)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
