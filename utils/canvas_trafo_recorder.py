import numpy as np

from . import BoundingBoxes


class CanvasTrafoRecorder:
    original_height: int
    original_width: int

    # number of pixels cropped away on the respective side, in original units
    cropped_left: float
    cropped_right: float
    cropped_top: float
    cropped_bottom: float

    # scale factor of the respective axis
    scale_height: float
    scale_width: float

    def __init__(self, original_height: int, original_width: int,
                 cropped_left: float = 0, cropped_right: float = 0,
                 cropped_top: float = 0, cropped_bottom: float = 0,
                 scale_height: float = 1, scale_width: float = 1):
        self.original_height = original_height
        self.original_width = original_width
        self.cropped_left = cropped_left
        self.cropped_right = cropped_right
        self.cropped_top = cropped_top
        self.cropped_bottom = cropped_bottom
        self.scale_height = scale_height
        self.scale_width = scale_width

    def crop_to(self, left: float = 0., right: float = None,
                top: float = 0., bottom: float = None) -> None:
        if right is None:
            right = self.original_width * self.scale_width
        if bottom is None:
            bottom = self.original_height * self.scale_height
        self.cropped_left += left/self.scale_width
        self.cropped_right += self.original_width - right/self.scale_width
        self.cropped_top += top/self.scale_height
        self.cropped_bottom += self.original_height - bottom/self.scale_height

    def scale(self, height_factor: float = 1, width_factor: float = 1) -> None:
        self.scale_height *= height_factor
        self.scale_width *= width_factor

    def reconstruct_boxes(self, boxes: BoundingBoxes) -> BoundingBoxes:
        boxes = boxes.astype("float")
        relevant_rows = boxes.sum(axis=1) != 0
        boxes[np.ix_(relevant_rows, (0, 2))] /= self.scale_width
        boxes[np.ix_(relevant_rows, (1, 3))] /= self.scale_height
        boxes[relevant_rows, 0] += self.cropped_left
        boxes[relevant_rows, 1] += self.cropped_top
        boxes[:, :2] = np.floor(boxes[:, :2])
        boxes[:, 2:] = np.ceil(boxes[:, 2:])
        return boxes.astype("int")
