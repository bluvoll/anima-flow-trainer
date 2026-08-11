"""Gate for texture-crop training.

Texture mode fails quietly by nature. A crop that ignores the energy map, a mask that supervises
everything, a canvas the source cannot fill, a caption that leaks whole-image tags onto a zoomed
crop -- none of these raise, and all of them produce a run that trains and converges to something.
So these checks assert on measured behaviour, not on whether code ran.

    .venv/bin/python anima/parity/test_texture.py
"""

from __future__ import annotations

import random
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from anima.data.dataset import (  # noqa: E402
    AnimaDataset,
    BucketBatchSampler,
    DatasetConfig,
    SubsetConfig,
    collate,
)
from anima.data.texture import (  # noqa: E402
    CANVAS_PRESETS,
    TextureConfig,
    choose_crop,
    crop_canvas,
    describe_feasibility,
    energy_map,
    feather_mask,
)
from anima.training.curriculum import Curriculum, Phase  # noqa: E402

PASS, FAIL = [], []

# Every cascade rung off -- which is now also the DEFAULT, after the cascade was reverted for
# producing broken anatomy. Kept as an explicit config rather than relying on `TextureConfig()` so
# these checks still say what they are asserting if the defaults ever move again.
REFERENCE_PAD = TextureConfig(oversize="pad", fit_aware=False, cover_max_scale=1.0,
                              pad_mode="black", mask_padding=False)

# Every rung on. The cascade is off by default but the code is still here, so it is still tested --
# an unused path that stops being exercised is one that quietly breaks before anyone re-enables it.
CASCADE = TextureConfig(oversize="pad", fit_aware=True, cover_max_scale=1.15,
                        pad_mode="reflect", mask_padding=True)


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")


def detailed_image(w=2048, h=1400, split=1100) -> Image.Image:
    """Flat on the left, high-frequency stripes on the right of `split`."""
    img = Image.new("RGB", (w, h), (128, 128, 128))
    d = ImageDraw.Draw(img)
    for x in range(split, w, 4):
        d.line([(x, 0), (x, h)], fill=(255, 255, 255) if (x // 4) % 2 else (0, 0, 0))
    return img


def main() -> int:
    cfg = TextureConfig()
    img = detailed_image()
    e = energy_map(img, cfg.energy_downscale)

    # --- energy map actually finds detail -------------------------------------------------------
    half = e.shape[1] // 2
    check("energy map is near-zero on flat regions and positive on detail",
          e[:, :half].mean() < 1e-4 < e[:, half:].mean(),
          f"{e[:, :half].mean():.5f} vs {e[:, half:].mean():.5f}")

    # --- crop selection is biased by energy, and power=0 is a real control ------------------------
    rng = random.Random(0)
    xs = [choose_crop(img.size, (1024, 1024), e, cfg, rng)[0] for _ in range(600)]
    flat = TextureConfig(energy_power=0.0)
    xs0 = [choose_crop(img.size, (1024, 1024), e, flat, rng)[0] for _ in range(600)]
    mx, mx0 = sum(xs) / len(xs), sum(xs0) / len(xs0)
    check("energy weighting shifts crops toward detail", mx > mx0 + 50, f"{mx:.0f} vs {mx0:.0f}")
    check("energy_power=0 is uniform (the honest control)", abs(mx0 - 512) < 60, f"{mx0:.0f}")
    check("higher energy_power concentrates further",
          sum(choose_crop(img.size, (1024, 1024), e, TextureConfig(energy_power=3.0), rng)[0]
              for _ in range(600)) / 600 > mx,
          "power 3 > power 1")

    # --- no slack is reported, not hidden --------------------------------------------------------
    check("a source exactly the canvas size yields the only valid crop",
          choose_crop((1024, 1024), (1024, 1024), e, cfg, rng) == (0, 0))

    # A canvas that overruns the source on ONE axis while having slack on the other. Only reachable
    # under oversize="pad", and the case that matters in practice: one real set is ~1088px tall against
    # a 1536px-tall canvas. Overrunning *both* axes takes the early return instead, which is why an
    # earlier version of this gate passed while the real run crashed on its first 640x1536 batch.
    tall = Image.new("RGB", (2048, 1088), (90, 90, 90))
    e_tall = energy_map(tall, cfg.energy_downscale)
    for canvas in ((640, 1536), (1536, 640), (1024, 1024)):
        try:
            x, y = choose_crop(tall.size, canvas, e_tall, cfg, rng)
            ok = 0 <= x <= max(0, tall.width - canvas[0]) and 0 <= y <= max(0, tall.height - canvas[1])
        except Exception as exc:
            ok, x, y = False, type(exc).__name__, exc
        check(f"canvas {canvas[0]}x{canvas[1]} overrunning one axis is in range", ok, f"{x},{y}")
    check("pad-mode crop survives a canvas taller than the source",
          crop_canvas(tall, (640, 1536), REFERENCE_PAD, random.Random(0))[0].size == (640, 1536))

    # --- oversize handling -----------------------------------------------------------------------
    small = Image.new("RGB", (800, 700), (200, 100, 50))
    padded, _ = crop_canvas(small, (1024, 1024), REFERENCE_PAD, random.Random(1))
    black = float((np.asarray(padded).sum(2) == 0).mean())
    check("oversize='pad' reproduces the reference's black band", black > 0.2, f"{black:.0%} black")
    covered, _ = crop_canvas(small, (1024, 1024), TextureConfig(oversize="cover"), random.Random(1))
    black_c = float((np.asarray(covered).sum(2) == 0).mean())
    check("oversize='cover' invents no black pixels", black_c == 0.0, f"{black_c:.0%} black")
    check("oversize='cover' still returns the canvas size", covered.size == (1024, 1024))
    check("oversize='skip' declines the sample",
          crop_canvas(small, (1024, 1024), TextureConfig(oversize="skip"),
                      random.Random(1))[0] is None)
    check("a source that fits is never upscaled by 'cover'",
          crop_canvas(img, (1024, 1024), cfg, random.Random(1))[0].size == (1024, 1024))

    # --- the oversize cascade --------------------------------------------------------------------
    # Each rung must remove a share of the invented pixels the next one would have had to make up,
    # and each must be independently switchable -- the reference's exact behaviour has to stay
    # reachable, because it is what an existing checkpoint was trained with.
    check("the DEFAULT is the full cascade, every rung on",
          (cfg.fit_aware, cfg.cover_max_scale, cfg.pad_mode, cfg.mask_padding, cfg.oversize)
          == (True, 1.15, "reflect", True, "pad"),
          f"{cfg.fit_aware}/{cfg.cover_max_scale}/{cfg.pad_mode}/{cfg.mask_padding}")

    near = Image.new("RGB", (960, 960), (200, 100, 50))          # needs 1.07x -> inside the cap
    far = Image.new("RGB", (500, 500), (200, 100, 50))           # needs 2.05x -> beyond it
    c_near, v_near = crop_canvas(near, (1024, 1024), CASCADE, random.Random(1))
    check("cascade: a near-miss is gently upscaled, not padded",
          v_near is None and float((np.asarray(c_near).sum(2) == 0).mean()) == 0.0)
    c_far, v_far = crop_canvas(far, (1024, 1024), CASCADE, random.Random(1))
    check("cascade: past the cap it pads instead of upscaling further", v_far is not None)
    check("  and its pad is mirrored content, not black",
          float((np.asarray(c_far).sum(2) == 0).mean()) == 0.0,
          f"{float((np.asarray(c_far).sum(2) == 0).mean()):.0%} black")
    check("  and it reports exactly which pixels it invented",
          v_far.shape == (1024, 1024) and 0.0 < float(v_far.mean()) < 1.0,
          f"{float(v_far.mean()):.0%} real")
    real_frac = (500 * 500) / (1024 * 1024)
    check("  validity matches the geometry, not an approximation of it",
          abs(float(v_far.mean()) - real_frac) < 0.01, f"{v_far.mean():.3f} vs {real_frac:.3f}")

    black_cfg = TextureConfig(pad_mode="black")
    c_blk, v_blk = crop_canvas(far, (1024, 1024), black_cfg, random.Random(1))
    check("pad_mode='black' still available for reproduction",
          float((np.asarray(c_blk).sum(2) == 0).mean()) > 0.5 and v_blk is not None)
    check("  and black padding reports the same invented region as reflect",
          abs(float(v_blk.mean()) - float(v_far.mean())) < 0.01)

    check("cover_max_scale=1.0 disables the gentle-upscale rung",
          crop_canvas(near, (1024, 1024), TextureConfig(cover_max_scale=1.0),
                      random.Random(1))[1] is not None)
    check("REFERENCE_PAD reproduces the reference exactly (no rung fires)",
          float((np.asarray(crop_canvas(near, (1024, 1024), REFERENCE_PAD,
                                        random.Random(1))[0]).sum(2) == 0).mean()) > 0.0)

    # A canvas the source holds outright must never allocate a validity mask -- that is the common
    # path and it has to stay free.
    check("a fitting source reports no invented pixels at all",
          crop_canvas(img, (1024, 1024), cfg, random.Random(1))[1] is None)

    # --- fit-aware canvas selection ---------------------------------------------------------------
    from anima.data.texture import fitting_canvases
    presets = list(CANVAS_PRESETS)
    check("fitting_canvases keeps multiplicity (the square's doubled weight)",
          fitting_canvases((2048, 2048), presets).count((1024, 1024)) == 2)
    check("fitting_canvases excludes what the source cannot hold",
          fitting_canvases((1100, 1100), presets) == [(1024, 1024), (1024, 1024)],
          str(fitting_canvases((1100, 1100), presets)))
    check("a source too small for anything reports nothing fits",
          fitting_canvases((600, 600), presets) == [])

    # --- mask ------------------------------------------------------------------------------------
    m = feather_mask(128, 96, cfg, random.Random(3))
    check("mask is (1, h, w) in [0,1]", tuple(m.shape) == (1, 128, 96)
          and float(m.min()) >= 0.0 and float(m.max()) <= 1.0)
    check("mask supervises a strict sub-region, not everything",
          0.1 < float((m > 0).float().mean()) < 1.0, f"{float((m > 0).float().mean()):.2f} covered")
    check("mask edges are feathered, not binary",
          float(((m > 0) & (m < 0.999)).float().sum()) > 0)
    check("mask_ratio=(1,1) covers the whole canvas",
          float((feather_mask(64, 64, TextureConfig(mask_ratio=(1.0, 1.0), feather_latent_px=0),
                              random.Random(0)) > 0).float().mean()) == 1.0)

    # --- feasibility reporting --------------------------------------------------------------------
    rep = describe_feasibility([(1088, 1280)] * 10, cfg)
    check("feasibility reports canvases the dataset cannot fill", "fits   0%" in rep)
    check("feasibility flags a canvas with no room to choose", "no room to choose" in rep)

    # --- end to end through the dataset and sampler -----------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for i in range(8):
            detailed_image(2048, 1400).save(root / f"i{i}.png")
            (root / f"i{i}.txt").write_text("tag_a, tag_b, tag_c")
        dcfg = DatasetConfig(path=str(root), resolution=1024, source="encode")
        dcfg.texture.trigger = "mytrigger"
        ds = AnimaDataset(dcfg)
        s = BucketBatchSampler(ds, 4, seed=0)
        s.set_curriculum(Curriculum([Phase(at=0.0, mode="fullres"),
                                     Phase(at=0.5, mode="texture")]), total_epochs=1, texture=dcfg.texture)

        first = collate([ds[k] for k in next(iter(s))])
        batches = list(iter(s))
        last = collate([ds[k] for k in batches[-1]])
        check("early batches are fullres (bucketed, no mask)", "mask" not in first)
        check("late batches are texture (canvas-shaped, masked)", "mask" in last)
        check("a texture batch's bucket is a canvas preset",
              tuple(last["bucket"]) in {tuple(c) for c in CANVAS_PRESETS},
              str(tuple(last["bucket"])))
        check("every sample in a texture batch shares the canvas",
              last["pixels"].shape[0] == len(batches[-1])
              and last["pixels"].shape[-1] == last["bucket"][0])
        check("texture captions are the trigger only, never the image tags",
              set(last["captions"]) == {"mytrigger"}, str(set(last["captions"]))[:40])
        check("fullres captions still carry the image tags",
              all("tag_" in c for c in first["captions"]))

        # Qwen3 tokenizes "" to ZERO tokens, so an empty caption padded to max_length yields an
        # all-zero attention mask and the conditioner attends over nothing. Three paths reach it:
        # caption dropout (returns "" by design), a texture phase with no trigger, and an empty tag
        # file. It happens not to NaN, which is exactly why it went unnoticed.
        from anima.training.train import UNCONDITIONAL, substitute_empty
        check("empty captions are substituted before encoding",
              substitute_empty(["", " ", "\n", "a, b"]) == [UNCONDITIONAL] * 3 + ["a, b"])
        check("the substitute is not itself empty", UNCONDITIONAL.strip() == ""
              and len(UNCONDITIONAL) > 0, repr(UNCONDITIONAL))
        blank = AnimaDataset(DatasetConfig(path=str(root), resolution=1024, source="encode"))
        blank.texture_cfg.trigger = ""
        check("a no-trigger texture caption survives substitution",
              substitute_empty([blank._texture_sample(blank.entries[0], (1024, 1024),
                                                      random.Random(0))["caption"]])[0].strip() == "")
        check("the mask is at latent resolution (canvas / 8)",
              tuple(last["mask"].shape[-2:]) == (last["bucket"][1] // 8, last["bucket"][0] // 8),
              str(tuple(last["mask"].shape)))

        # The cascade end to end, explicitly enabled -- it is OFF by default now, and the point of
        # these checks is that the code still works when someone turns it back on.
        #
        # These images are 2048x1400, so they hold the landscape and square canvases but not the
        # two portrait ones. Swept over 12 seeds because this dataset yields only two texture
        # batches per epoch, so one seed can miss the portrait canvases by luck and make either
        # arm look like the other.
        def canvases_over_seeds(dataset):
            out = set()
            for seed in range(12):
                sk = BucketBatchSampler(dataset, 4, seed=seed)
                sk.set_curriculum(Curriculum([Phase(at=0.0, mode="texture")]), total_epochs=1,
                                  texture=dataset.texture_cfg)
                out |= {tuple(collate([dataset[k] for k in b])["bucket"]) for b in iter(sk)}
            return out

        aware = AnimaDataset(DatasetConfig(path=str(root), resolution=1024, source="encode"))
        kept = canvases_over_seeds(aware)
        check("fit-aware selection only draws canvases the batch can hold",
              aware.texture_cfg.fit_aware is True
              and all(ch <= 1400 for _, ch in kept) and len(kept) > 1, str(sorted(kept)))

        # The control: with it explicitly off, the blind draw must be able to reach a canvas that
        # does not fit, or the check above is passing for the wrong reason.
        blind = AnimaDataset(DatasetConfig(path=str(root), resolution=1024, source="encode"))
        blind.texture_cfg.fit_aware = False
        drawn = canvases_over_seeds(blind)
        check("  and turning it off restores the blind draw",
              any(ch > 1400 for _, ch in drawn), str(sorted(drawn)))

        # Source size has to survive the scan, or fit-aware silently degrades to the blind draw.
        check("the scan records the original size, not the bucket",
              ds.entries[0].source_size == (2048, 1400), str(ds.entries[0].source_size))

        # A sample that genuinely needed padding must carry a mask with zeros in it -- this is the
        # check that the validity mask reaches the loss at all, which is the whole point.
        tiny = Path(td) / "tiny"
        tiny.mkdir()
        for i in range(4):
            detailed_image(700, 700).save(tiny / f"t{i}.png")
            (tiny / f"t{i}.txt").write_text("tag_a")
        tds = AnimaDataset(DatasetConfig(path=str(tiny), resolution=1024, source="encode"))
        tds.texture_cfg.mask_padding = True          # the default, set explicitly so the arms read
        ts = BucketBatchSampler(tds, 2, seed=0)
        ts.set_curriculum(Curriculum([Phase(at=0.0, mode="texture")]), total_epochs=1,
                          texture=tds.texture_cfg)
        tb = collate([tds[k] for k in next(iter(ts))])
        check("padding is excluded from the loss mask",
              float((tb["mask"] == 0).float().mean()) > 0.05,
              f"{float((tb['mask'] == 0).float().mean()):.0%} of the mask is zero")

        tds.texture_cfg.mask_padding = False
        tb2 = collate([tds[k] for k in next(iter(ts))])
        check("  and mask_padding=false leaves it supervised (the honest control)",
              float((tb2["mask"] == 0).float().mean()) < float((tb["mask"] == 0).float().mean()),
              f"{float((tb2['mask'] == 0).float().mean()):.0%} zero")

        # The property that would otherwise drift silently: the dataset's mode and the trainer's
        # t_range must come from the same phase. Both derive from the sampler's batch ordinal.
        modes = [("mask" in collate([ds[k] for k in b])) for b in batches]
        expected = [s._phase_for(i).mode == "texture" for i in range(len(batches))]
        check("dataset mode matches the sampler's phase for every batch", modes == expected,
              f"{sum(modes)}/{len(modes)} texture")

    # --- texture-exempt subsets -----------------------------------------------------------------
    # The whole point of `texture = false` is that a regularization subset keeps its captions. If
    # it silently got texture-cropped its caption would become `texture.trigger` (empty by
    # default), and an uncaptioned flat colour trains the UNCONDITIONAL branch -- which CFG then
    # subtracts, so colour anchors would push generations away from colour. Nothing about that
    # raises; it just quietly inverts the intent. Hence checks on behaviour, not on wiring.
    tmp = Path(tempfile.mkdtemp(prefix="anima_subset_"))
    try:
        main_dir, reg_dir = tmp / "main", tmp / "reg"
        for d in (main_dir, reg_dir):
            d.mkdir()
        rs = random.Random(0)
        for i in range(8):        # detailed sources, texture-eligible
            a = np.array([[[rs.randint(0, 255) for _ in range(3)] for _ in range(1400)]
                          for _ in range(1400)], dtype=np.uint8)
            Image.fromarray(a).save(main_dir / f"m{i}.png")
            (main_dir / f"m{i}.txt").write_text("1girl, white background")
        for i in range(4):        # flat colour anchors, texture-exempt
            Image.new("RGB", (1024, 1024), (255, 0, 0)).save(reg_dir / f"c{i}.png")
            (reg_dir / f"c{i}.txt").write_text("red background, no humans")

        cfg = DatasetConfig(
            subsets=[SubsetConfig(path=str(main_dir)),
                     SubsetConfig(path=str(reg_dir), num_repeats=3, texture=False)],
            source="encode", resolution=1024,
        )
        ds = AnimaDataset(cfg, require_cache=False)
        check("subsets scan into one dataset", len(ds) == 8 + 4 * 3, f"{len(ds)} entries")
        check("  per-subset num_repeats applies to that subset only",
              [n for _, n in ds.subset_report] == [8, 12], str([n for _, n in ds.subset_report]))
        check("  texture_ok is stamped per subset",
              (sum(e.texture_ok for e in ds.entries), sum(not e.texture_ok for e in ds.entries))
              == (8, 12))

        s = BucketBatchSampler(ds, batch_size=2, shuffle=True, seed=0)
        s.set_curriculum(Curriculum([Phase(at=0.0, t_range=(0.0, 1.0), mode="texture")]), 1,
                         cfg.texture)
        batches = list(s)
        mixed = [b for b in batches if len({ds.entries[i].texture_ok for i, *_ in b}) > 1]
        check("no batch mixes texture-eligible with exempt entries", not mixed, f"{len(mixed)}")

        got_canvas = {ds.entries[b[0][0]].texture_ok: set() for b in batches}
        for b in batches:
            got_canvas[ds.entries[b[0][0]].texture_ok].add(b[0][3] is not None)
        check("in a 100% texture phase, eligible batches all get a canvas",
              got_canvas.get(True) == {True}, str(got_canvas.get(True)))
        check("  and exempt batches never do", got_canvas.get(False) == {False},
              str(got_canvas.get(False)))

        # The consequence that actually matters, read off the samples rather than the flags.
        caps = {True: set(), False: set()}
        for b in batches:
            ok = ds.entries[b[0][0]].texture_ok
            caps[ok].add(ds[b[0]]["caption"])
        check("exempt samples keep their real captions in a texture phase",
              caps[False] == {"red background, no humans"}, str(caps[False]))
        check("  while eligible samples get the trigger (empty here)", caps[True] == {""},
              str(caps[True]))

        # Regression guard: the single-`path` form must be untouched by any of this.
        one = AnimaDataset(DatasetConfig(path=str(main_dir), source="encode", resolution=1024,
                                         num_repeats=2), require_cache=False)
        check("single-path form still works and honours top-level num_repeats",
              len(one) == 16 and all(e.texture_ok for e in one.entries), f"{len(one)}")

        for bad, why in (
            ({"path": str(main_dir), "subsets": [SubsetConfig(path=str(reg_dir))]}, "both"),
            ({}, "neither"),
            ({"subsets": [SubsetConfig(path=str(main_dir)), SubsetConfig(path=str(main_dir))]},
             "duplicate"),
        ):
            try:
                DatasetConfig(**bad)
                check(f"rejects {why} path/subsets", False, "accepted")
            except ValueError:
                check(f"rejects {why} path/subsets", True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} passed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
