import math
import json
import numpy as np

BOX_ATTRIBUTES = ["x", "y", "width", "height"]


def bounding_boxes_mask(boxes: np.ndarray, size: tuple[int, int]) -> np.array:
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
        mask[math.floor(x):math.ceil(x + width),
             math.floor(y):math.ceil(y + height)] = 1
    return mask


def bounding_boxes_array(meta_boxes: str, max_bounding_boxes: int) \
        -> np.ndarray:
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
            boxes[i] = np.array([(math.floor
                                  if i < 2 else math.ceil)(box[attribute])
                                 for attribute in BOX_ATTRIBUTES])
    else:
        raise TypeError("unexpected type of 'meta_boxes':", type(meta_boxes))
    return boxes
