"""
This module features the CustomOutput Dataset which wraps an existing dataset
to manipulate the form of its output.
Along with the class, this module defines some functions that may be used as
parameters for CustomOutput.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typeguard import typechecked
from typing import Any, Callable, Tuple, List
from warnings import warn

from utils import singledispatchmethod
from utils.bounding_boxes import BoundingBoxes
from .study_dataset import LABEL_KEYS
from .knit import Knit
from trafo import Trafo, Compose


class CustomOutput(Dataset):
    """
    Dataset featuring custom output defined by functions given at __init__().

    Args:
        dataset (Dataset): wrapped dataset
        *output (Callable[[Dataset, int, str], Any]): functions to call on dataset
            for each part of the output to be generated by __getitem__()
    """
    dataset: Knit
    output: Tuple[Callable[[Dataset, int, str], Any], ...]
    ids: List[str]
    trafo: Trafo

    def __init__(self, dataset: Knit,
                 *output: Callable[[Dataset, int], Any],
                 trafo: Trafo = Compose(accept_empty=True)) -> None:
        self.dataset = dataset
        self.output = output
        self.trafo = trafo
        if hasattr(self.dataset, "image_ids"):
            self.ids = list(self.dataset.image_ids)
        else:
            warn("Using image IDs from CSV table. This is okay if the data is "
                 + "in the same order.")
            self.ids = [id.replace("_image", "")
                        for id in self.dataset.image_table["id"]]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, _index: int) -> tuple:
        data = tuple(part(self.dataset, _index) for part in self.output)
        return self.trafo(*data)

    @singledispatchmethod
    def get(self, key) -> tuple:
        raise NotImplementedError

    get.register(__getitem__)

    @get.register
    def _(self, key: str) -> tuple:
        return self.get(self.ids.index(key))


@typechecked
def image_id(self: Dataset, _index: int) -> str:
    """Output the id of an image as string.

    Intended for use with CustomOutput.

    Args:
        self (Dataset): The wrapped dataset.
        _index (int): The index of the data point queried.

    Returns:
        str: The image id of the data point.
    """
    return self.image_ids[_index]


@typechecked
def index(self: Dataset, _index: int) -> int:
    """Output the index in the dataset.

    Intended for use with CustomOutput.

    Args:
        self (Dataset): The wrapped dataset.
        _index (int): The index of the data point queried.

    Returns:
        int: The index of the data point queried.
    """
    return _index


@typechecked
def image_csv_index(self: Dataset, _index: int) -> int:
    """Output the index of an image within the image CSV table of the dataset.

    Intended for use with CustomOutput.

    Args:
        self (Dataset): The wrapped dataset.
        _index (int): The index of the data point queried.

    Returns:
        int: The index of an image within the image CSV table of the dataset.
    """
    id = image_id(self, _index)
    mask = self.image_table["id"] == f"{id}_image"
    indices = mask.index[mask]
    assert len(indices) == 1, f"Missing or ambiguous image: {id}"
    return int(indices[0])


@typechecked
def image_csv_meta(self: Dataset, _index: int) -> pd.code.series.Series:
    """Output the row of an image from the image CSV table of the dataset.

    Intended for use with CustomOutput.

    Args:
        self (Dataset): The wrapped dataset.
        _index (int): The index of the data point queried.

    Returns:
        pd.code.series.Series: The row of an image from the image CSV table of
            the dataset.
    """
    csv_ind = image_csv_index(self, _index)
    return self.image_table.iloc[csv_ind]


@typechecked
def image_tensor(self: Dataset, _index: int) -> torch.Tensor:
    """Output the image as a torch Tensor.

    Intended for use with CustomOutput.

    Args:
        self (Dataset): The wrapped dataset.
        _index (int): The index of the data point queried.

    Returns:
        torch.Tensor: The image.
    """
    return self[_index][0]


@typechecked
def study_id(self: Dataset, _index: int) -> str:
    """Output the study id of an image from the image CSV table of the dataset.

    Intended for use with CustomOutput.

    Args:
        self (Dataset): The wrapped dataset.
        _index (int): The index of the data point queried.

    Returns:
        str: The study id of an image from the image CSV table of the dataset.
    """
    meta = image_csv_meta(self, _index)
    return meta['StudyInstanceUID']


@typechecked
def study_label(self: Dataset, _index: int) -> np.ndarray:
    """Output the study label of an image.

    Intended for use with CustomOutput.
    This requires both the image and study tables of the dataset.

    This label has the following meanings: negative, typical, indeterminate,
    atypical.

    Args:
        self (Dataset): The wrapped dataset.
        _index (int): The index of the data point queried.

    Returns:
        np.ndarray: The study label of length 4 with dtype=bool.
    """
    _study_id = study_id(self, _index)
    study_label = self.study_table.loc[self.study_table["id"]
                                       == f"{_study_id}_study"]
    assert len(study_label) == 1, \
        f"Missing or ambiguous study: {study_id}"
    return np.array(study_label.iloc[0][LABEL_KEYS], dtype=bool)


@typechecked
def study_label_5(self: Dataset, _index: int) -> np.ndarray:
    """Output the modified study label (of length 5) of an image.


    Intended for use with CustomOutput.
    This requires both the image and study tables of the dataset.

    Meanings of True value at certain indices are the following: 0 - negative,
    4 - any positive class but no bounding boxes in the image. If there are
    bounding boxes in the image: 1 - typical, 2 - indeterminate, 3 - atypical.

    Args:
        self (Dataset): The wrapped dataset.
        _index (int): The index of the data point queried.

    Returns:
        np.ndarray: The modified study label of length 5 with dtype=bool.
    """
    _bounding_boxes = bounding_boxes(self, _index)
    _study_label = study_label(self, _index)
    return np.array([*_study_label, 0]  # for negative or annotated images
                    if _study_label[0] or _bounding_boxes.sum() > 0
                    else [*[0]*4, 1],  # for positive but missing bbox
                    dtype=bool)


@typechecked
def bounding_boxes(self: Dataset, _index: int) -> BoundingBoxes:
    """Output the (transformed) bounding boxes of an image.


    Intended for use with CustomOutput.
    This requires the bounding boxes in the dataset.

    Args:
        self (Dataset): The wrapped dataset.
        _index (int): The index of the data point queried.

    Returns:
        BoundingBoxes: The corresponding bounding boxes.
    """
    boxes = getattr(self, "bounding_boxes", None)
    if boxes is not None:
        return boxes[_index]
    elif self.label_type is BoundingBoxes:
        return self[_index][1]
    raise KeyError(f"No bounding boxes found in {self.__class__.__name__}.")


@typechecked
def mask(self: Dataset, _index: int) -> np.ndarray:
    """Output the (transformed) bounding boxes mask of an image.


    Intended for use with CustomOutput.
    This requires the bounding boxes in the dataset.

    Args:
        self (Dataset): The wrapped dataset.
        _index (int): The index of the data point queried.

    Returns:
        np.ndarray: The corresponding bounding boxes mask.
    """
    shape = image_tensor(self, _index).shape[-2:]
    return bounding_boxes(self, _index).get_mask(shape)


@typechecked
def float_mask(self: Dataset, _index: int) -> np.ndarray:
    """Output the (transformed) bounding boxes float mask of an image.


    Intended for use with CustomOutput.
    This requires the bounding boxes in the dataset.

    Args:
        self (Dataset): The wrapped dataset.
        _index (int): The index of the data point queried.

    Returns:
        np.ndarray: The corresponding bounding boxes float mask.
    """
    shape = image_tensor(self, _index).shape[-2:]
    return bounding_boxes(self, _index).get_float_mask(shape)


@typechecked
def masked_image_tensor(self: Dataset, _index: int) -> torch.Tensor:
    """Output the (transformed) bounding boxes masked image.


    Intended for use with CustomOutput.
    This requires the bounding boxes in the dataset.

    Args:
        self (Dataset): The wrapped dataset.
        _index (int): The index of the data point queried.

    Returns:
        np.ndarray: The image, area outside bounding boxes valued to 0.
    """
    image = image_tensor(self, _index).clone().detach()
    bounding_boxes(self, _index).mask_image(image[0])
    return image


@typechecked
def nontransformed_bounding_boxes(self: Dataset, _index: int) -> BoundingBoxes:
    """Output the non-transformed bounding boxes of an image.


    Intended for use with CustomOutput.
    This requires the image table of the dataset.

    Args:
        self (Dataset): The wrapped dataset.
        _index (int): The index of the data point queried.

    Returns:
        BoundingBoxes: The corresponding non-transformed bounding boxes.
    """
    meta = image_csv_meta(self, _index)
    return BoundingBoxes.from_json(meta["boxes"], self.max_bounding_boxes)
