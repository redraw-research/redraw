from __future__ import annotations

import math
import numbers
import random
import warnings
from types import LambdaType
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union, cast
from warnings import warn

import albucore
import cv2
import numpy as np
from albucore.functions import add_weighted, multiply, normalize, normalize_per_image
from albucore.utils import MAX_VALUES_BY_DTYPE, clip, get_num_channels, is_grayscale_image, is_rgb_image
from pydantic import AfterValidator, BaseModel, Field, ValidationInfo, field_validator, model_validator
from scipy import special
from scipy.ndimage import gaussian_filter
from typing_extensions import Annotated, Literal, Self, TypedDict

from albumentations import random_utils
from albumentations.augmentations.blur.functional import blur
from albumentations.augmentations.blur.transforms import BlurInitSchema, process_blur_limit
from albumentations.augmentations.utils import check_range
from albumentations.core.transforms_interface import (
    BaseTransformInitSchema,
    DualTransform,
    ImageOnlyTransform,
    Interpolation,
    NoOp,
)
from albumentations.core.types import (
    MAX_RAIN_ANGLE,
    MONO_CHANNEL_DIMENSIONS,
    NUM_RGB_CHANNELS,
    PAIR,
    BoxInternalType,
    ChromaticAberrationMode,
    ColorType,
    ImageCompressionType,
    ImageMode,
    KeypointInternalType,
    MorphologyMode,
    PlanckianJitterMode,
    RainMode,
    ScaleFloatType,
    ScaleIntType,
    ScaleType,
    SpatterMode,
    Targets,
)
from albumentations.core.utils import format_args, to_tuple

from albumentations.core.types import *

from . import functional as fmain

from albumentations.core.pydantic import (
    InterpolationType,
    NonNegativeFloatRangeType,
    OnePlusFloatRangeType,
    OnePlusIntRangeType,
    ProbabilityType,
    SymmetricRangeType,
    ZeroOneRangeType,
    check_0plus,
    check_01,
    check_1plus,
    nondecreasing,
)

class SuperpixelsDuckiebotsMask(ImageOnlyTransform):
    """Transform images partially/completely to their superpixel representation.
    This implementation uses skimage's version of the SLIC algorithm.

    Args:
        p_replace (float or tuple of float): Defines for any segment the probability that the pixels within that
            segment are replaced by their average color (otherwise, the pixels are not changed).

    Examples:
                * A probability of ``0.0`` would mean, that the pixels in no
                  segment are replaced by their average color (image is not
                  changed at all).
                * A probability of ``0.5`` would mean, that around half of all
                  segments are replaced by their average color.
                * A probability of ``1.0`` would mean, that all segments are
                  replaced by their average color (resulting in a Voronoi
                  image).
            Behavior based on chosen data types for this parameter:
                * If a ``float``, then that ``flat`` will always be used.
                * If ``tuple`` ``(a, b)``, then a random probability will be
                  sampled from the interval ``[a, b]`` per image.
        n_segments (tuple of int): Rough target number of how many superpixels to generate (the algorithm
            may deviate from this number). Lower value will lead to coarser superpixels.
            Higher values are computationally more intensive and will hence lead to a slowdown
            Then a value from the discrete interval ``[a..b]`` will be sampled per image.
            If input is a single integer, the range will be ``(1, n_segments)``.
            If interested in a fixed number of segments, use ``(n_segments, n_segments)``.
        max_size (int or None): Maximum image size at which the augmentation is performed.
            If the width or height of an image exceeds this value, it will be
            downscaled before the augmentation so that the longest side matches `max_size`.
            This is done to speed up the process. The final output image has the same size as the input image.
            Note that in case `p_replace` is below ``1.0``,
            the down-/upscaling will affect the not-replaced pixels too.
            Use ``None`` to apply no down-/upscaling.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    """

    class InitSchema(BaseTransformInitSchema):
        p_replace: ZeroOneRangeType = (0, 0.1)
        n_segments: OnePlusIntRangeType = (100, 100)
        max_size: int | None = Field(default=128, ge=1, description="Maximum image size for the transformation.")
        interpolation: InterpolationType = cv2.INTER_LINEAR

    def __init__(
        self,
        p_replace: ScaleFloatType = (0, 0.1),
        n_segments: ScaleIntType = (100, 100),
        max_size: int | None = 128,
        interpolation: int = cv2.INTER_LINEAR,
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.p_replace = cast(Tuple[float, float], p_replace)
        self.n_segments = cast(Tuple[int, int], n_segments)
        self.max_size = max_size
        self.interpolation = interpolation

    def get_transform_init_args_names(self) -> tuple[str, str, str, str]:
        return ("p_replace", "n_segments", "max_size", "interpolation")

    def get_params(self) -> dict[str, Any]:
        n_segments = random.randint(self.n_segments[0], self.n_segments[1])
        p = random.uniform(*self.p_replace)
        return {"replace_samples": random_utils.random(n_segments) < p, "n_segments": n_segments}

    def apply(
        self,
        img: np.ndarray,
        replace_samples: Sequence[bool],
        n_segments: int,
        **kwargs: Any,
    ) -> np.ndarray:
        return fmain.superpixels_duckiebots_mask(img, n_segments, replace_samples, self.max_size, self.interpolation)

