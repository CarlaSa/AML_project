import torch
import random
from torchvision.transforms import functional as TF

from .randomize import randomize

# TODO: finalize


def gaussian_blur(image: torch.Tensor, p: float) -> torch.Tensor:
    # TODO: specify sigma?
    if random.random() < p:
        return TF.gaussian_blur(image, kernel_size=5)
    return image


def scale(image: torch.Tensor, scale_factor: float) -> torch.Tensor:
    return TF.affine(image, scale=scale_factor)


def shear(image: torch.Tensor, shear_x: float, shear_y: float) -> torch.Tensor:
    return TF.affine(image, shear=(shear_x, shear_y))


def translate(image: torch.Tensor, translate_x: float, translate_y: float) \
        -> torch.Tensor:
    return TF.affine(image, translate=(translate_x, translate_y))


Autocontrast = randomize(TF.autocontrast, preserve_bounding_boxes=True)

AdjustBrightness = randomize(TF.adjust_brightness, "brightness_factor",
                             preserve_bounding_boxes=True)
AdjustContrast = randomize(TF.adjust_contrast, "contrast_factor",
                           preserve_bounding_boxes=True)
AdjustSharpness = randomize(TF.adjust_sharpness, "sharpness_factor",
                            preserve_bounding_boxes=True)
GaussianBlur = randomize(TF.gaussian_blur, "p", preserve_bounding_boxes=True)
# Scale to 0…255

Rotate = randomize(TF.rotate, "angle")
Scale = randomize(scale, "scale_factor")
Shear = randomize(shear, "shear_x", "shear_y")
Translate = randomize(translate, "translate_x", "translate_y")
