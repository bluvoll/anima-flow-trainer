"""Aspect-ratio bucketing, ported verbatim from sd-scripts.

Ported, not imported: sd-scripts' `library.model_util` pulls in `original_unet` -> `library.utils`
-> `cv2`, and `library.train_util` drags the whole arg/config surface behind it. The bucketing
logic itself is ~150 lines of pure arithmetic with no dependencies, so it is copied exactly
(including the `multires_training` extension in the LoRA_Easy_Training_Scripts fork) and pinned
by a test that diffs against sd-scripts' own cached bucket decisions on 2000 real images.

Sources:
  make_bucket_resolutions  -- sd_scripts/library/model_util.py:1413
  BucketManager            -- sd_scripts/library/train_util.py:230

Why this rather than diffusion-pipe's bucketing: on a 1920x1080 source, diffusion-pipe assigned a
1616x1024 bucket (11.2% aspect distortion); this picks 1344x768 (1.6%), or an exact 1024x576 with
multires_training enabled.

Anima constraint: the DiT's RoPE has a hard ceiling at 2048px (128 patch units) and 1536px is the
practical limit on 24GB cards -- see verify_max_resolution().
"""

import math
import random
from typing import Any

import numpy as np

# diffusers' CosmosRotaryPosEmbed indexes a position table of max(max_size // patch) = 128 units.
# At patch 2 and VAE stride 8 that is 128 * 2 * 8 = 2048px. Above it, RoPE raises a shape error.
# Two different per-side ceilings, and the gap between them matters.
#
# The transformer config declares `max_size = [128, 240, 240]` (t, h, w) in pixel units, which
# `CosmosRotaryPosEmbed.__init__` divides by `patch_size` to `[128, 120, 120]` patch units. So the
# *declared* spatial capacity is 120 patch units = 1920px per side.
#
# But `forward()` builds `seq = torch.arange(max(self.max_size))` -- and that max is 128, taken
# from the TEMPORAL axis. The spatial dims then index into that longer sequence, so they reach 128
# units (2048px) before anything breaks. Verified:
#
#     1920px (120u) OK   <- declared limit
#     1984px (124u) OK   <- runs, past declared max_size
#     2048px (128u) OK   <- runs, past declared max_size
#     2112px (132u) RuntimeError: Sizes of tensors must match except in dimension 3
#
# This is exactly why diffusion-pipe asserts at 1920 while diffusers works to 2048: one follows
# the declared limit, the other the implementation artifact. Above 1920 the model is being asked
# for positions its spatial RoPE was not configured for, so we error at the true break point and
# warn through the grey band rather than pretending either number is the whole story.
ROPE_MAX_PIXELS = 2048          # hard failure above this
ROPE_DECLARED_PIXELS = 1920     # beyond this, spatial RoPE is extrapolating


def make_bucket_resolutions(
    max_reso: tuple[int, int],
    min_size: int = 256,
    max_size: int = 1024,
    divisible: int = 64,
    multires_training: bool = False,
) -> list[tuple[int, int]]:
    """Enumerate candidate bucket resolutions under a fixed max area.

    With multires_training the inner loop walks every height rather than only the tallest that
    fits, producing far more buckets (407 vs 41 at 1024/64) and therefore much closer aspect
    matches -- at the cost of smaller areas for some ratios.
    """
    max_width, max_height = max_reso
    max_area = max_width * max_height

    resos = set()

    width = int(math.sqrt(max_area) // divisible) * divisible
    resos.add((width, width))

    width = min_size
    while width <= max_size:
        max_h = min(max_size, int((max_area // width) // divisible) * divisible)

        if not multires_training:
            height = max_h
            if height >= min_size:
                resos.add((width, height))
                resos.add((height, width))
        else:
            height = min_size
            while height <= max_h:
                resos.add((width, height))
                resos.add((height, width))
                height += divisible

        width += divisible

    return sorted(resos)


class BucketManager:
    """Assigns images to buckets. Two distinct modes:

    - `no_upscale=False`: snap to the nearest predefined resolution by aspect ratio, scaling up
      or down as needed.
    - `no_upscale=True`: never enlarge. Bucket dimensions are derived per image by shrinking to
      max_area and rounding down to `reso_steps`, so buckets are *not* drawn from the predefined
      list. This is the branch the sd-scripts oracle cache was built with.
    """

    def __init__(
        self,
        no_upscale: bool,
        max_reso: tuple[int, int] | None,
        min_size: int | None,
        max_size: int | None,
        reso_steps: int,
        multires_training: bool = False,
    ) -> None:
        self.multires_training = multires_training
        if max_size is not None:
            if max_reso is not None:
                assert max_size >= max_reso[0], "max_size should be larger than the width of max_reso"
                assert max_size >= max_reso[1], "max_size should be larger than the height of max_reso"
            if min_size is not None:
                assert max_size >= min_size, "max_size should be larger than min_size"

        self.no_upscale = no_upscale
        if max_reso is None:
            self.max_reso = None
            self.max_area = None
        else:
            self.max_reso = max_reso
            self.max_area = max_reso[0] * max_reso[1]
        self.min_size = min_size
        self.max_size = max_size
        self.reso_steps = reso_steps

        self.resos: list[tuple[int, int]] = []
        self.reso_to_id: dict[tuple[int, int], int] = {}
        self.buckets: list[list[Any]] = []

    def add_image(self, reso: tuple[int, int], image_or_info: Any) -> None:
        self.buckets[self.reso_to_id[reso]].append(image_or_info)

    def shuffle(self) -> None:
        for bucket in self.buckets:
            random.shuffle(bucket)

    def sort(self) -> None:
        sorted_resos = sorted(self.resos)
        sorted_buckets = []
        sorted_reso_to_id = {}
        for i, reso in enumerate(sorted_resos):
            sorted_buckets.append(self.buckets[self.reso_to_id[reso]])
            sorted_reso_to_id[reso] = i
        self.resos = sorted_resos
        self.buckets = sorted_buckets
        self.reso_to_id = sorted_reso_to_id

    def make_buckets(self) -> None:
        resos = make_bucket_resolutions(
            self.max_reso, self.min_size, self.max_size, self.reso_steps, self.multires_training
        )
        self.set_predefined_resos(resos)

    def set_predefined_resos(self, resos: list[tuple[int, int]]) -> None:
        self.predefined_resos = resos.copy()
        self.predefined_resos_set = set(resos)
        self.predefined_aspect_ratios = np.array([w / h for w, h in resos])

    def add_if_new_reso(self, reso: tuple[int, int]) -> None:
        if reso not in self.reso_to_id:
            self.reso_to_id[reso] = len(self.resos)
            self.resos.append(reso)
            self.buckets.append([])

    def round_to_steps(self, x: float) -> int:
        x = int(x + 0.5)
        return x - x % self.reso_steps

    def select_bucket(
        self, image_width: int, image_height: int
    ) -> tuple[tuple[int, int], tuple[int, int], float]:
        """Returns (bucket_reso, resized_size, ar_error)."""
        aspect_ratio = image_width / image_height

        if not self.no_upscale:
            # Scale up or down onto a predefined resolution.
            reso = (image_width, image_height)
            if reso in self.predefined_resos_set:
                pass
            else:
                ar_errors = np.abs(self.predefined_aspect_ratios - aspect_ratio)
                if self.multires_training:
                    # Among equally-good aspect ratios, prefer the closest area. This is the
                    # fork's addition and it is what turns a 1.6% AR error into an exact match.
                    min_ar_error = ar_errors.min()
                    closest_indices = np.where(ar_errors <= min_ar_error + 1e-4)[0]
                    target_area = image_width * image_height
                    areas = np.array(
                        [self.predefined_resos[i][0] * self.predefined_resos[i][1] for i in closest_indices]
                    )
                    reso = self.predefined_resos[closest_indices[np.abs(areas - target_area).argmin()]]
                else:
                    reso = self.predefined_resos[ar_errors.argmin()]

            ar_reso = reso[0] / reso[1]
            if aspect_ratio > ar_reso:  # image is wider -> match height
                scale = reso[1] / image_height
            else:
                scale = reso[0] / image_width

            resized_size = (int(image_width * scale + 0.5), int(image_height * scale + 0.5))
        else:
            # Shrink only; the bucket is derived from the image, not chosen from a list.
            if image_width * image_height > self.max_area:
                resized_width = math.sqrt(self.max_area * aspect_ratio)
                resized_height = self.max_area / resized_width
                assert abs(resized_width / resized_height - aspect_ratio) < 1e-2, "aspect is illegal"

                # Round either the width or the height to reso_steps, whichever preserves the
                # aspect ratio better.
                b_width_rounded = self.round_to_steps(resized_width)
                b_height_in_wr = self.round_to_steps(b_width_rounded / aspect_ratio)
                ar_width_rounded = b_width_rounded / b_height_in_wr

                b_height_rounded = self.round_to_steps(resized_height)
                b_width_in_hr = self.round_to_steps(b_height_rounded * aspect_ratio)
                ar_height_rounded = b_width_in_hr / b_height_rounded

                if abs(ar_width_rounded - aspect_ratio) < abs(ar_height_rounded - aspect_ratio):
                    calc_size = (b_width_rounded, int(b_width_rounded / aspect_ratio + 0.5))
                else:
                    calc_size = (int(b_height_rounded * aspect_ratio + 0.5), b_height_rounded)
            else:
                calc_size = (image_width, image_height)  # no resize needed

            # Per-side cap -- our addition, not sd-scripts'.
            #
            # The area budget bounds w*h but neither side individually, so a wide image can sit
            # comfortably under max_area and still exceed Anima's per-side RoPE limit: at
            # resolution 1536 a 21:9 source lands at 2346px. sd-scripts has no such limit to
            # respect, so its no_upscale branch ignores max_size entirely; for Anima that produces
            # buckets the model cannot address.
            #
            # Shrink uniformly so the longest side fits. This only ever scales *down* -- a 512x512
            # image is untouched -- so the "never upscale" contract of this branch is preserved.
            if self.max_size is not None and max(calc_size) > self.max_size:
                shrink = self.max_size / max(calc_size)
                calc_size = (
                    int(calc_size[0] * shrink + 0.5),
                    int(calc_size[1] * shrink + 0.5),
                )

            # Bucket is <= the image, so the image is cropped rather than padded.
            bucket_width = calc_size[0] - calc_size[0] % self.reso_steps
            bucket_height = calc_size[1] - calc_size[1] % self.reso_steps

            reso = (bucket_width, bucket_height)

            if bucket_width > 0 and bucket_height > 0:
                scale = max(bucket_width / image_width, bucket_height / image_height)
                resized_size = (int(image_width * scale + 0.5), int(image_height * scale + 0.5))
            else:
                resized_size = calc_size

        self.add_if_new_reso(reso)

        ar_error = (reso[0] / reso[1]) - aspect_ratio
        return reso, resized_size, ar_error

    @staticmethod
    def get_crop_ltrb(
        bucket_reso: tuple[int, int], image_size: tuple[int, int]
    ) -> tuple[int, int, int, int]:
        """Centre-crop box, matching Stability AI's preprocessing.

        crop right/bottom are returned so flip augmentation can mirror the box.
        """
        bucket_ar = bucket_reso[0] / bucket_reso[1]
        image_ar = image_size[0] / image_size[1]
        if bucket_ar > image_ar:
            # bucket is wider -> match height
            resized_width = bucket_reso[1] * image_ar
            resized_height = bucket_reso[1]
        else:
            resized_width = bucket_reso[0]
            resized_height = bucket_reso[0] / image_ar
        crop_left = (bucket_reso[0] - resized_width) // 2
        crop_top = (bucket_reso[1] - resized_height) // 2
        return crop_left, crop_top, crop_left + resized_width, crop_top + resized_height


def verify_max_resolution(resos: list[tuple[int, int]] | set[tuple[int, int]]) -> None:
    """Check buckets against Anima's per-side RoPE limits at config time.

    Errors above ROPE_MAX_PIXELS (diffusers raises a shape error there, and crashing at step 4000
    is a bad way to find out) and warns between the declared and hard limits, where the model runs
    but its spatial RoPE is extrapolating past the range it was configured for.

    Both limits are PER SIDE, not area: 3072x768 has the same area as 1536x1536 and still fails.
    """
    bad = [r for r in resos if max(r) > ROPE_MAX_PIXELS]
    if bad:
        raise ValueError(
            f"{len(bad)} bucket(s) exceed Anima's RoPE limit of {ROPE_MAX_PIXELS}px per side "
            f"(e.g. {sorted(bad, key=max, reverse=True)[:3]}). Lower max_bucket_reso."
        )

    grey = [r for r in resos if max(r) > ROPE_DECLARED_PIXELS]
    if grey:
        import warnings

        warnings.warn(
            f"{len(grey)} bucket(s) exceed the declared spatial RoPE range of "
            f"{ROPE_DECLARED_PIXELS}px per side (e.g. {sorted(grey, key=max, reverse=True)[:3]}). "
            f"They run -- diffusers tolerates up to {ROPE_MAX_PIXELS}px because the position "
            f"sequence is sized by the temporal axis -- but the model was configured for "
            f"{ROPE_DECLARED_PIXELS}px and is extrapolating beyond it. diffusion-pipe refuses "
            f"these outright. Set max_bucket_reso = {ROPE_DECLARED_PIXELS} to stay in spec.",
            stacklevel=2,
        )
