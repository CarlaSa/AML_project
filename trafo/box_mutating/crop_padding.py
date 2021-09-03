from ..trafo import Trafo, Transformable
from .crop_common import crop_image, crop_boxes

import numpy as np

# DEBUG: argument after ** must be a mapping, not a tuple

class CropPadding(Trafo):
    """
    Crop to the relevant part of the image (lung).

    Mainly based on https://www.kaggle.com/davidbroberts/cropping-chest-x-rays
    """

    def compute_parameters(self, img: np.ndarray,
                           *additional_transformands: Transformable) \
            -> dict[str, int]:
        height, width = img.shape
        row_stds = np.std(img, axis=1)
        col_stds = np.std(img, axis=0)
        thresh = 0.012 * np.max(img)  # TODO
        if np.min([row_stds.min(), col_stds.min()]) > thresh:
            return (0, width, 0, height)
        else:
            print("Removed Padding")
            left_crop = np.max([np.argmax(col_stds > thresh)-1, 0])
            right_crop = np.min(
                [width-np.argmax(col_stds[::-1] > thresh), width])
            top_crop = np.max([np.argmax(row_stds > thresh), 0])
            bottom_crop = np.min(
                [height-np.argmax(row_stds[::-1] > thresh), height])
            return {"left_crop": left_crop,
                    "right_crop": right_crop,
                    "top_crop": top_crop,
                    "bottom_crop": bottom_crop
                    }


CropPadding.transform.register(crop_image)
CropPadding.transform.register(crop_boxes)
