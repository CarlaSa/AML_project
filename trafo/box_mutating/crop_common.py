import numpy as np

from utils import BoundingBoxes, CanvasTrafoRecorder


def crop_image(self, img: np.ndarray,
               left_crop, right_crop, top_crop, bottom_crop) \
        -> np.ndarray:
    return img[top_crop:bottom_crop, left_crop:right_crop]


def crop_boxes(self, boxes: BoundingBoxes,
               left_crop, right_crop, top_crop, bottom_crop) \
        -> BoundingBoxes:
    boxes[:, 0] -= left_crop  # x-left_crop
    boxes[:, 1] -= top_crop  # y-top_crop

    # Cut boxes if necessary
    # Modify width and length if the bounding boxes are cut at top or left

    boxes[:, 2] += np.minimum(boxes[:, 0], 0)
    boxes[:, 3] += np.minimum(boxes[:, 1], 0)

    # Modify x, y if bounding boxes are cut at top or left
    boxes[:, :2] = np.maximum(boxes[:, :2], 0)

    # Modify width and length if the bounding boxes are cut at bottom or right
    boxes[:, 2] = np.minimum(boxes[:, 2], right_crop - left_crop - boxes[:, 0])
    boxes[:, 3] = np.minimum(boxes[:, 3], bottom_crop - top_crop - boxes[:, 1])

    # Identify bounding boxes as zero-rows if they are not in the image
    rows_to_zero = np.array(np.prod(boxes[:, 2:] <= 0, axis=1), dtype=bool)
    boxes[rows_to_zero, :] = 0
    return boxes


def crop_recorder(self, recorder: CanvasTrafoRecorder,
                  left_crop, right_crop, top_crop, bottom_crop) \
        -> CanvasTrafoRecorder:
    recorder.crop_to(left=left_crop, right=right_crop,
                     top=top_crop, bottom=bottom_crop)
    return recorder
