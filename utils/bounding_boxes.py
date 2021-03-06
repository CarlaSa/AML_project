from __future__ import annotations

import math
import json
import torch
import numpy as np
from skimage.measure import regionprops, label

from utils import singledispatchmethod

BOX_ATTRIBUTES = ["x", "y", "width", "height"]


class BoundingBoxes(np.ndarray):
    @staticmethod
    def from_array(array: np.ndarray) -> BoundingBoxes:
        needs_shape = (None, 4)
        if len(array.shape) != len(needs_shape):
            raise TypeError("Dimensions do not match", needs_shape)
        for candidate, gold in zip(array.shape, needs_shape):
            if gold is None:
                continue
            if candidate != gold:
                raise TypeError("Shape does not match", needs_shape)

        boxes = array.view(BoundingBoxes)
        boxes.sort_boxes()
        return boxes

    @staticmethod
    def from_json(meta_boxes: str, max_bounding_boxes: int) \
            -> BoundingBoxes:
        boxes = BoundingBoxes.from_array(np.zeros((max_bounding_boxes, 4)))
        if isinstance(meta_boxes, float):
            # No bounding boxes → nothing to change.
            # This if condition is reserved for possible future use.
            pass
        elif isinstance(meta_boxes, str):
            json_boxes = json.loads(meta_boxes.replace("'", '"'))
            for i, box in enumerate(json_boxes):
                # may throw an error if max_bounding_boxes is too low, which is
                # absolutely intended:
                boxes[i] = [(math.floor if i < 2 else math.ceil)
                            (box[attribute])
                            for attribute in BOX_ATTRIBUTES]
        else:
            raise TypeError("unexpected type of 'meta_boxes':",
                            type(meta_boxes))
        boxes.sort_boxes()
        return boxes

    @staticmethod
    def from_mask(mask: np.ndarray, max_bounding_boxes: int = 0,
                  enlarge_array: bool = True) -> BoundingBoxes:
        mask = mask.astype(int)
        labeled = label(mask)
        regions = regionprops(labeled)
        n_boxes = max(max_bounding_boxes, len(regions))
        boxes = BoundingBoxes.from_array(np.zeros((n_boxes, 4)))
        for i, r in enumerate(regions):
            boxes[i] = [r.bbox[1], r.bbox[0], r.bbox[3]-r.bbox[1],
                        r.bbox[2]-r.bbox[0]]
        boxes.sort_boxes()
        return boxes

    def sort_boxes(self) -> None:
        # TODO: check if .copy() is really necessary
        self[:, :] = sorted(self.copy(), key=tuple, reverse=True)

    def get_mask(self, size: tuple[int, int], dtype: np.dtype = bool) \
            -> np.ndarray:
        """
        Args:
            size (tuple): size of the image
            dtype (np.dtype): dtype of the mask
        Returns:
            np.array: binary Box Mask, where True indicates a box and 0 not
        """
        mask = np.zeros(size, dtype=dtype)
        for i in range(self.shape[0]):
            if sum(self[i]) == 0:
                break
            x, y, width, height = self[i]
            mask[math.floor(y):math.ceil(y + height),
                 math.floor(x):math.ceil(x + width)] = True
        return mask

    def get_float_mask(self, size: tuple[int, int]) -> np.ndarray:
        """
        Args:
            size (tuple): size of the image
        Returns:
            np.array: binary Box Mask, where True indicates a box and 0 not
        """
        return self.get_mask(size, float)

    @singledispatchmethod
    def mask_image(self, image) -> None:
        """
        Mask a torch.Tensor or np.ndarray image in-place.

        Sets the values to 0 where no bounding boxes are.

        Args:
            image (Union[np.ndarray, torch.Tensor]): The image to mask.
        """
        raise NotImplementedError

    @mask_image.register
    def _(self, image: np.ndarray) -> None:
        mask = self.get_mask(image.shape)
        image[~mask] = 0

    @mask_image.register
    def _(self, image: torch.Tensor) -> None:
        mask = self.get_mask(image.shape[-2:])
        mask = torch.from_numpy(mask).bool()
        dummy_dims = len(image.shape[:-2])
        image[(0,) * dummy_dims][~mask] = 0
