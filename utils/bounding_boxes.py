from __future__ import annotations

import math
import json
import numpy as np

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

        return array.view(BoundingBoxes)

    def sort_boxes(self) -> None:
        self[:, :] = sorted(self.copy(), key=tuple, reverse=True)


def bounding_boxes_mask(boxes: BoundingBoxes, size: tuple[int, int]) \
        -> np.ndarray:
    """
    Args:
        boxes (np.array): 4 x X array with x, y, width, height
        size (tuple): size of the image
    Returns:
        np.array: binary Box Mask, where 1 indicates a box and 0 not
    """
    mask = np.zeros(size)
    for i in range(boxes.shape[0]):
        if sum(boxes[i]) == 0:
            break
        x, y, width, height = boxes[i]
        #mask[math.floor(x):math.ceil(x + width),
        #     math.floor(y):math.ceil(y + height)] = 1
        mask[math.floor(y):math.ceil(y + height),
             math.floor(x):math.ceil(x + width)] = 1
    return mask


def bounding_boxes_array(meta_boxes: str, max_bounding_boxes: int) \
        -> BoundingBoxes:
    boxes = np.zeros((max_bounding_boxes, 4))
    if isinstance(meta_boxes, float):
        # No bounding boxes → nothing to change.
        # This if condition is reserved for possible future use.
        pass
    elif isinstance(meta_boxes, str):
        json_boxes = json.loads(meta_boxes.replace("'", '"'))
        for i, box in enumerate(json_boxes):
            # may throw an error if max_bounding_boxes is too low, which is
            # absolutely intended:
            array = np.array([(math.floor if i < 2 else math.ceil)
                              (box[attribute])
                              for attribute in BOX_ATTRIBUTES])
            boxes[i] = BoundingBoxes.from_array(array)
    else:
        raise TypeError("unexpected type of 'meta_boxes':", type(meta_boxes))
    return boxes
