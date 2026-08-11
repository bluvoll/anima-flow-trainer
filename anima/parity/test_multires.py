"""Gate for multi-resolution training.

    .venv/bin/python anima/parity/test_multires.py
    .venv/bin/python anima/parity/test_multires.py --waf     # + the 2000-image real-data check

Most of this runs against a synthetic fixture -- generated images plus hand-written cache files --
so it is fast, hermetic, and can exercise the failure paths (missing tier, stale cache, tiny
sources) that real datasets do not happen to contain. The real datasets are then used for the two
things a fixture cannot prove: that the default single-tier path is unchanged, and that the
measured collapse numbers still hold.

The load-bearing property is the **regression guard**: with `resolutions` unset, the entry list
must be identical to single-tier behaviour. Multi-res is opt-in and must stay that way.
"""

from __future__ import annotations

import os
import argparse
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from anima.data.bucket import BucketManager  # noqa: E402
from anima.data.cache import cache_path  # noqa: E402
from anima.data.dataset import AnimaDataset, BucketBatchSampler, DatasetConfig  # noqa: E402

# `Path("")` is `Path(".")`, which passes `.is_dir()` -- so an unset variable would
# have this gate scan the repo root. None means "not configured", and the checks skip.
def _dataset_env(name):
    value = os.environ.get(name, "").strip()
    return Path(value) if value else None


UNIFORM = _dataset_env("ANIMA_DATASET_UNIFORM")
MIXED = _dataset_env("ANIMA_DATASET_MIXED")


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    return bool(ok)


def make_fixture(root: Path, sizes: list[tuple[int, int]], tiers: list[int],
                 write_caches: bool = True, skip: set[tuple[int, tuple[int, int]]] = frozenset()):
    """Images + captions + fake latent caches at exactly the buckets `tiers` implies.

    The caches carry real `original_size` metadata, which is what latents-only mode uses to
    attribute each file back to a tier.
    """
    root.mkdir(parents=True, exist_ok=True)
    managers = {t: BucketManager(no_upscale=True, max_reso=(t, t), min_size=256,
                                 max_size=1920, reso_steps=64) for t in tiers}
    for i, (w, h) in enumerate(sizes):
        p = root / f"img{i:03d}.png"
        Image.new("RGB", (w, h), (i % 256, 40, 90)).save(p)     # solid colour -> tiny on disk
        p.with_suffix(".txt").write_text(f"tag{i}, shared_tag", encoding="utf-8")
        if not write_caches:
            continue
        for t in tiers:
            b = managers[t].select_bucket(w, h)[0]
            if (i, b) in skip:
                continue
            save_file(
                {"latents": torch.zeros(16, 1, b[1] // 8, b[0] // 8)},
                cache_path(p, b),
                metadata={"version": "1", "bucket": f"{b[0]}x{b[1]}",
                          "original_size": f"{w}x{h}",
                          "crop_ltrb": "0,0,0,0", "source": f"{i:016x}"},
            )


def entry_key(d: AnimaDataset):
    return sorted((e.latent_path.name, e.bucket, e.resolution) for e in d.entries)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--waf", action="store_true", help="include the 2000-image real-data check")
    args = ap.parse_args()
    r = []
    tmp = Path(tempfile.mkdtemp(prefix="anima_multires_"))

    try:
        # Sizes chosen to straddle the tier budgets: two large (no collapse), one mid (collapses
        # at the top tier only), one tiny (collapses everywhere).
        SIZES = [(1920, 1080), (2400, 1600), (900, 1100), (400, 400)]
        TIERS = [768, 1024, 1280]
        root = tmp / "fixture"
        make_fixture(root, SIZES, TIERS)

        # ---------------------------------------------------------------- cross product
        d = AnimaDataset(DatasetConfig(path=str(root), resolutions=TIERS, source="images"))
        rep = d.tier_report
        expected = sum(
            len({BucketManager(no_upscale=True, max_reso=(t, t), min_size=256, max_size=1920,
                               reso_steps=64).select_bucket(w, h)[0] for t in TIERS})
            for w, h in SIZES
        )
        r.append(check("dedup entry count == distinct (image, bucket) pairs",
                       len(d) == expected, f"{len(d)} == {expected}"))
        r.append(check("collapse is counted, not silent",
                       sum(rep.collapsed_per_tier.values()) > 0,
                       f"per tier {rep.collapsed_per_tier}"))

        d_rep = AnimaDataset(DatasetConfig(path=str(root), resolutions=TIERS,
                                           tier_collapse="repeat", source="images"))
        r.append(check("repeat mode == images x tiers exactly",
                       len(d_rep) == len(SIZES) * len(TIERS),
                       f"{len(d_rep)} == {len(SIZES)}x{len(TIERS)}"))
        r.append(check("repeat >= dedup", len(d_rep) > len(d), f"{len(d_rep)} vs {len(d)}"))

        # The tiny source must collapse onto ONE bucket at every tier -- the case that motivated
        # the whole dedup discussion.
        tiny = [e for e in d.entries if "img003" in e.latent_path.name]
        r.append(check("tiny source yields a single entry under dedup",
                       len(tiny) == 1 and tiny[0].bucket == (384, 384),
                       f"{[e.bucket for e in tiny]}"))

        # ---------------------------------------------------------------- latents parity
        d_lat = AnimaDataset(DatasetConfig(path=str(root), resolutions=TIERS, source="latents"))
        r.append(check("latents scan == images scan (tier attribution)",
                       entry_key(d_lat) == entry_key(d), f"{len(d_lat)} entries"))

        # ---------------------------------------------------------------- failure paths
        miss = tmp / "missing"
        make_fixture(miss, SIZES, TIERS, skip={(0, (1344, 768))})
        for mode in ("images", "latents"):
            try:
                AnimaDataset(DatasetConfig(path=str(miss), resolutions=TIERS, source=mode))
                r.append(check(f"missing tier raises ({mode})", False, "accepted"))
            except RuntimeError as e:
                r.append(check(f"missing tier raises ({mode})", "no cached latent" in str(e)))

        # An extra cache from an undeclared tier is IGNORED with a warning, not fatal: exact tier
        # attribution means it can never be silently chosen, and images mode ignores extras too.
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            d_sub = AnimaDataset(DatasetConfig(path=str(root), resolutions=[768],
                                               source="latents"))
            warned = any("match no declared tier" in str(w.message) for w in ws)
        r.append(check("undeclared-tier cache ignored + warned",
                       warned and len(d_sub) == len(SIZES), f"{len(d_sub)} entries"))

        # ---------------------------------------------------------------- min_source_area
        d_small = AnimaDataset(DatasetConfig(path=str(root), resolutions=TIERS,
                                             min_source_area=1.0, source="images"))
        dropped = d_small.tier_report.dropped_small
        r.append(check("min_source_area drops sub-tier sources and reports them",
                       len(dropped) == 1 and "img003" in dropped[0], f"dropped {dropped}"))

        # ---------------------------------------------------------------- collapse warning
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            AnimaDataset(DatasetConfig(path=str(root), resolutions=TIERS, source="images"))
            fired = [w for w in ws if "duplicate a lower tier" in str(w.message)]
        r.append(check("collapse warning fires on a collapsing set", bool(fired),
                       f"{len(fired)} warning(s)"))

        big = tmp / "nocollapse"
        make_fixture(big, [(3000, 2000), (2400, 1800)], TIERS)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            AnimaDataset(DatasetConfig(path=str(big), resolutions=TIERS, source="images"))
            quiet = not [w for w in ws if "duplicate a lower tier" in str(w.message)]
        r.append(check("collapse warning silent when nothing collapses", quiet))

        # ---------------------------------------------------------------- batching
        s = BucketBatchSampler(d, batch_size={0: 2}, shuffle=True, seed=0)
        homogeneous = all(len({d.bucket_of(i) for i in b}) == 1 for b in s._batches())
        served = sorted(i for b in s._batches() for i in b)
        r.append(check("batches stay bucket-homogeneous under multi-res", homogeneous))
        r.append(check("every entry served exactly once per epoch",
                       served == list(range(len(d))), f"{len(served)}/{len(d)}"))

        # ---------------------------------------------------------------- min_bucket_reso inert
        # Guards against a refactor quietly giving min_size meaning on the no_upscale path, which
        # would change bucketing for every existing cache.
        same = all(
            BucketManager(no_upscale=True, max_reso=(1024, 1024), min_size=m, max_size=1920,
                          reso_steps=64).select_bucket(w, h)[0]
            == BucketManager(no_upscale=True, max_reso=(1024, 1024), min_size=256, max_size=1920,
                             reso_steps=64).select_bucket(w, h)[0]
            for m in (64, 128, 256) for w, h in SIZES + [(200, 1500), (300, 300)]
        )
        r.append(check("min_bucket_reso inert in no_upscale mode", same))

        # ---------------------------------------------------------------- regression guard
        if UNIFORM is not None and UNIFORM.is_dir():
            base = AnimaDataset(DatasetConfig(path=str(UNIFORM), resolution=1024))
            explicit = AnimaDataset(DatasetConfig(path=str(UNIFORM), resolutions=[1024]))
            r.append(check("default path unchanged: resolution=1024",
                           len(base) == 113 and {e.bucket for e in base.entries} == {(1344, 768)},
                           f"{len(base)} entries"))
            r.append(check("resolution=R == resolutions=[R]",
                           [(e.latent_path.name, e.bucket) for e in base.entries]
                           == [(e.latent_path.name, e.bucket) for e in explicit.entries]))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m2 = AnimaDataset(DatasetConfig(path=str(UNIFORM), resolutions=[768, 1024]))
            r.append(check("the uniform set 2-tier == 226 entries, 0 collapsed",
                           len(m2) == 226 and sum(m2.tier_report.collapsed_per_tier.values()) == 0,
                           f"{len(m2)} entries"))
        else:
            print("SKIP  the uniform set regression guard (dataset not present)")

        if args.waf and MIXED is not None and MIXED.is_dir():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                w3 = AnimaDataset(DatasetConfig(path=str(MIXED), resolutions=[768, 1024, 1280],
                                                source="images"), require_cache=False)
            c = w3.tier_report.collapsed_per_tier
            r.append(check("waf 768/1024/1280 == 5803 entries (measured)", len(w3) == 5803,
                           f"{len(w3)}"))
            r.append(check("waf collapse == 33 at 1024, 164 at 1280",
                           c[1024] == 33 and c[1280] == 164, f"{c}"))
        elif args.waf:
            print("SKIP  waf check (dataset not present)")

        # ---------------------------------------------------------- batch size per tier
        # `batch_size` is keyed by resolution TIER, matched exactly. It used to be a threshold
        # ladder on the bucket's longest side, which sorts by something that is not the cost:
        # bucketing targets a constant area, so tier^2/256 bounds a tier's tokens exactly, while
        # longest sides overlap heavily across tiers. That mismatch handed a 960x960 bucket 32
        # images under `{512: 32, 1024: 12}` -- 2.4x the load of the 1024x1024 bucket beside it --
        # and OOMed minutes into a run.
        from anima.training.config import load_config

        root = tmp / "bs"
        make_fixture(root, [(2048, 1152), (1600, 1600), (900, 1500)], [768, 1024])
        cfg = DatasetConfig(path=str(root), resolutions=[768, 1024], source="images")
        ds = AnimaDataset(cfg, caption_seed=0)

        sampler = BucketBatchSampler(ds, batch_size={768: 16, 1024: 4}, shuffle=False, seed=0)
        by_tier = {}
        for bucket in sampler.bucket_indices:
            by_tier.setdefault(sampler._bucket_tier[bucket], set()).add(
                sampler.batch_size_for(bucket))
        r.append(check("batch size follows the tier, not the longest side",
                       by_tier == {768: {16}, 1024: {4}}, str(by_tier)))

        # The property that makes tier-keying correct: the tier bounds the token count exactly.
        over = {t: [b for b in sampler.bucket_indices if sampler._bucket_tier[b] == t
                    and b[0] * b[1] // 256 > t * t // 256] for t in (768, 1024)}
        r.append(check("tier^2/256 is an upper bound on every bucket in that tier",
                       not any(over.values()), str(over)))

        # And the reason the old key was wrong: longest sides do not separate the tiers.
        sides = {t: sorted({max(b) for b in sampler.bucket_indices
                            if sampler._bucket_tier[b] == t}) for t in (768, 1024)}
        r.append(check("longest sides overlap across tiers (why the old key failed)",
                       bool(set(sides[768]) & set(sides[1024]))
                       or max(sides[768]) >= min(sides[1024]), str(sides)))

        flat = BucketBatchSampler(ds, batch_size=7, shuffle=False, seed=0)
        r.append(check("a plain int applies to every tier",
                       {flat.batch_size_for(b) for b in flat.bucket_indices} == {7}))

        def load_with(batch_line: str, res_line: str = "resolutions = [768, 1024]"):
            path = tmp / "bs.toml"
            path.write_text(
                f'[train]\nmodel_path = "/x"\n{batch_line}\n'
                f'[dataset]\npath = "{root}"\n{res_line}\n[adapter]\nkind = "none"\n')
            return load_config(path)

        for line, why in [("batch_size = { 512 = 32, 1024 = 12 }", "undeclared tier 512"),
                          ("batch_size = { 768 = 16 }", "missing tier 1024")]:
            try:
                load_with(line)
                r.append(check(f"rejected at load: {why}", False, "accepted"))
            except ValueError as exc:
                r.append(check(f"rejected at load: {why}",
                               "batch_size" in str(exc), str(exc)[:60]))
        ok_cfg = load_with("batch_size = { 768 = 16, 1024 = 4 }")
        r.append(check("an exact tier map loads", ok_cfg.train.batch_size == {768: 16, 1024: 4}))
        r.append(check("a plain int loads under a ladder",
                       load_with("batch_size = 6").train.batch_size == 6))

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    n = sum(r)
    print(f"\n{n}/{len(r)} passed")
    return 0 if n == len(r) else 1


if __name__ == "__main__":
    sys.exit(main())
