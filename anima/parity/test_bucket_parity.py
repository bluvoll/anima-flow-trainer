"""Phase 3 gate: our ported bucketing must reproduce sd-scripts' decisions exactly.

The oracle is better than re-running sd-scripts: a directory cached by sd-scripts already contains its
own latent cache, and each `<name>_<W>x<H>_sd.npz` stores its latents under a `latents_{h}x{w}`
key -- i.e. the bucket sd-scripts actually chose, at 8x VAE stride. That is 2000 recorded
decisions over 1579 distinct source resolutions (AR 0.31 to 3.20), captured before we wrote a
line of this port and impossible to accidentally fit to.

Parameters were recovered from the cache: base 768x768 (max bucket area 589824 = 768^2, confirmed
by the user), reso_steps 64, and bucket_no_upscale=True -- the observed buckets include sizes
absent from the predefined list (384x1152, 448x960), which only the no_upscale branch can emit.
"""

import glob
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from anima.data.bucket import BucketManager  # noqa: E402

# sd-scripts' own latent cache is the oracle: its `latents_{h}x{w}` keys record the buckets it
# actually chose. Point ANIMA_BUCKET_ORACLE at a directory holding one.
DATASET = os.environ.get("ANIMA_BUCKET_ORACLE", "")
BASE_RESO = (768, 768)
RESO_STEPS = 64
VAE_STRIDE = 8


def load_oracle() -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Returns [((src_w, src_h), (bucket_w, bucket_h))] from sd-scripts' cache filenames."""
    pairs = []
    for path in sorted(glob.glob(os.path.join(DATASET, "*_sd.npz"))):
        stem = os.path.basename(path)[: -len("_sd.npz")]
        dims = stem.rsplit("_", 1)[-1]
        try:
            src_w, src_h = (int(v) for v in dims.split("x"))
        except ValueError:
            continue
        with np.load(path) as data:
            keys = [k for k in data.keys() if k.startswith("latents_")]
        if not keys:
            continue
        lat_h, lat_w = (int(v) for v in keys[0][len("latents_") :].split("x"))
        pairs.append(((src_w, src_h), (lat_w * VAE_STRIDE, lat_h * VAE_STRIDE)))
    return pairs


def main() -> int:
    # Distinguish "not configured" from "configured and empty". The first is a skip -- this gate
    # needs a dataset that sd-scripts has already cached, which a fresh clone will not have. The
    # second is a real failure, because it means the path was given and holds nothing usable.
    if not DATASET:
        print("SKIP  bucket parity needs an sd-scripts latent cache to compare against.\n"
              "      Point ANIMA_BUCKET_ORACLE at a directory containing *_sd.npz files:\n"
              "        ANIMA_BUCKET_ORACLE=/path/to/dataset .venv/bin/python "
              "anima/parity/test_bucket_parity.py")
        return 0

    oracle = load_oracle()
    if not oracle:
        print(f"no sd-scripts cache found in {DATASET}")
        return 1

    bm = BucketManager(
        no_upscale=True,
        max_reso=BASE_RESO,
        min_size=None,
        max_size=None,
        reso_steps=RESO_STEPS,
    )

    mismatches = []
    for (src_w, src_h), expected in oracle:
        got, _resized, _ar_err = bm.select_bucket(src_w, src_h)
        if got != expected:
            mismatches.append(((src_w, src_h), expected, got))

    n = len(oracle)
    print(f"oracle: {n} images, {len({o for _, o in oracle})} distinct buckets")
    print(f"        {len({s for s, _ in oracle})} distinct source resolutions")
    ars = [w / h for (w, h), _ in oracle]
    print(f"        aspect ratios {min(ars):.2f} .. {max(ars):.2f}")
    print(f"match:  {n - len(mismatches)}/{n}")

    if mismatches:
        print(f"\n{len(mismatches)} mismatches (first 10):")
        for src, exp, got in mismatches[:10]:
            print(f"  {src[0]}x{src[1]}  expected {exp[0]}x{exp[1]}  got {got[0]}x{got[1]}")
        by_shape = Counter((exp, got) for _, exp, got in mismatches)
        print("\nmost common (expected -> got):")
        for (exp, got), c in by_shape.most_common(5):
            print(f"  {exp} -> {got}   x{c}")
        print("\nBUCKET PARITY FAIL")
        return 1

    print("\nBUCKET PARITY PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
