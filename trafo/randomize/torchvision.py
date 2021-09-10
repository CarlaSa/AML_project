import torch
from torchvision.transforms import functional as TF

from .randomize import randomize

# TODO: finalize
AFFINE_DEFAULT_ARGS = {
    "angle": 0,
    "shear": 0,
    "translate": (0, 0),
    "scale": 1
}


def affine_args_except(*keys: str) -> int:
    return {key: val for key, val in AFFINE_DEFAULT_ARGS.items()
            if key not in keys}


def gaussian_blur(image: torch.Tensor, x: float, p: float) -> torch.Tensor:
    # TODO: specify sigma?
    if x < p:
        return TF.gaussian_blur(image, kernel_size=5)
    return image


def scale(image: torch.Tensor, scale_factor: float) -> torch.Tensor:
    return TF.affine(image, scale=scale_factor, **affine_args_except("scale"))


def shear(image: torch.Tensor, shear_x: float, shear_y: float) -> torch.Tensor:
    return TF.affine(image, shear=(shear_x, shear_y),
                     **affine_args_except("shear"))


def translate(image: torch.Tensor, translate_x: float, translate_y: float) \
        -> torch.Tensor:
    return TF.affine(image, translate=(translate_x, translate_y),
                     **affine_args_except("translate"))


# non-random: Scale to 0…255
Autocontrast = randomize(TF.autocontrast, preserve_bounding_boxes=True)

# box preserving
AdjustBrightness = randomize(TF.adjust_brightness, "brightness_factor",
                             preserve_bounding_boxes=True)
AdjustSharpness = randomize(TF.adjust_sharpness, "sharpness_factor",
                            preserve_bounding_boxes=True)
GaussianBlur = randomize(gaussian_blur, "x", "p", preserve_bounding_boxes=True)

# box mutating
Rotate = randomize(TF.rotate, "angle")
Scale = randomize(scale, "scale_factor")
Shear = randomize(shear, "shear_x", "shear_y")
Translate = randomize(translate, "translate_x", "translate_y")
