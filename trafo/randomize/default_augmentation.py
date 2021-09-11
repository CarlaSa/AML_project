import random
from ..compose import Compose
from .torchvision import AdjustSharpness, AdjustBrightness, GaussianBlur
from .torchvision import Autocontrast, Rotate, Scale, Shear, Translate
from ..type_mutating import BoundingBoxesToMask, NDArrayTo3dTensor


def positive_half_gauss(mu: float, sigma: float) -> float:
    x = random.gauss(mu, sigma)
    return x if x > mu else mu


default_augmentation = Compose(
    AdjustSharpness(sharpness_factor=(1, 0.5)),
    AdjustBrightness(brightness_factor=(1, 0.05)),
    GaussianBlur(x=(0, 1), p=(0.1, 0.1), random_function=random.uniform),
    Autocontrast(),
    BoundingBoxesToMask(),
    NDArrayTo3dTensor(),
    Rotate(angle=(0, 3)),
    Shear(shear_x=(0, 3), shear_y=(0, 3)),
    Translate(translate_x=(0, 4), translate_y=(0, 4)),
    Scale(scale_factor=(1, 0.075), random_function=positive_half_gauss),
)

default_augmentation_only_values = Compose(
    AdjustSharpness(sharpness_factor=(1, 0.5)),
    AdjustBrightness(brightness_factor=(1, 0.05)),
    GaussianBlur(x=(0, 1), p=(0.1, 0.1), random_function=random.uniform),
    Autocontrast(),
    BoundingBoxesToMask(),
    NDArrayTo3dTensor(),
)

default_augmentation_only_geometric = Compose(
    BoundingBoxesToMask(),
    NDArrayTo3dTensor(),
    Rotate(angle=(0, 3)),
    Shear(shear_x=(0, 3), shear_y=(0, 3)),
    Translate(translate_x=(0, 4), translate_y=(0, 4)),
    Scale(scale_factor=(1, 0.075), random_function=positive_half_gauss),
)
