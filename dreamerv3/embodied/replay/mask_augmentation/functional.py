from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Sequence
from warnings import warn

import cv2
import numpy as np
import skimage
from albucore.functions import add, add_array, add_weighted, multiply, multiply_add
from albucore.utils import (
    MAX_VALUES_BY_DTYPE,
    clip,
    clipped,
    contiguous,
    is_grayscale_image,
    is_rgb_image,
    maybe_process_in_chunks,
    preserve_channel_dim,
    get_num_channels,
)
from typing_extensions import Literal

from albumentations import random_utils
from albumentations.augmentations.utils import (
    non_rgb_warning,
)
from albumentations.core.types import (
    EIGHT,
    MONO_CHANNEL_DIMENSIONS,
    NUM_MULTI_CHANNEL_DIMENSIONS,
    ColorType,
    ImageMode,
    NumericType,
    PlanckianJitterMode,
    SizeType,
    SpatterMode,
)

DUCKIEBOTS_MASK_COLORS = np.asarray([(0, 0, 0), (251, 244, 4), (255, 255, 255), (255, 0, 255)], dtype=np.uint8)

@preserve_channel_dim
def superpixels_duckiebots_mask(
    image: np.ndarray,
    n_segments: int,
    replace_samples: Sequence[bool],
    max_size: int | None,
    interpolation: int,
) -> np.ndarray:
    if not np.any(replace_samples):
        return image

    orig_shape = image.shape
    if max_size is not None:
        size = max(image.shape[:2])
        if size > max_size:
            scale = max_size / size
            height, width = image.shape[:2]
            new_height, new_width = int(height * scale), int(width * scale)
            resize_fn = maybe_process_in_chunks(cv2.resize, dsize=(new_width, new_height), interpolation=interpolation)
            image = resize_fn(image)

    segments = skimage.segmentation.slic(
        image,
        n_segments=n_segments,
        compactness=10,
        channel_axis=-1 if image.ndim > MONO_CHANNEL_DIMENSIONS else None,
    )

    min_value = 0
    max_value = MAX_VALUES_BY_DTYPE[image.dtype]
    image = np.copy(image)
    if image.ndim == MONO_CHANNEL_DIMENSIONS:
        image = image.reshape(*image.shape, 1)
    nb_channels = image.shape[2]

    new_colors = None
    for c in range(nb_channels):
        # segments+1 here because otherwise regionprops always misses the last label
        regions = skimage.measure.regionprops(segments + 1, intensity_image=image[..., c])
        if not new_colors:
            new_colors = [random.choice(DUCKIEBOTS_MASK_COLORS) for _ in range(len(regions))]
        for ridx, region in enumerate(regions):
            # with mod here, because slic can sometimes create more superpixel than requested.
            # replace_samples then does not have enough values, so we just start over with the first one again.
            if replace_samples[ridx % len(replace_samples)]:
                # mean_intensity = region.mean_intensity
                new_color_intensity = new_colors[ridx][c]

                image_sp_c = image[..., c]

                # if image_sp_c.dtype.kind in ["i", "u", "b"]:
                #     # After rounding the value can end up slightly outside of the value_range. Hence, we need to clip.
                #     # We do clip via min(max(...)) instead of np.clip because
                #     # the latter one does not seem to keep dtypes for dtypes with large itemsizes (e.g. uint64).
                #     value: int | float
                #     value = int(np.round(new_color_intensity))
                #     value = min(max(value, min_value), max_value)
                # else:
                value = new_color_intensity

                image_sp_c[segments == ridx] = value

    if orig_shape != image.shape:
        resize_fn = maybe_process_in_chunks(
            cv2.resize,
            dsize=(orig_shape[1], orig_shape[0]),
            interpolation=interpolation,
        )
        return resize_fn(image)

    return image