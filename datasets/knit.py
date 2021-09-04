import os
import pandas as pd
from torch.utils.data import Dataset
from typing import Optional, Sequence, Any, Union, Final
from warnings import warn
import re
from typing import Optional

from . import LoadDataset, RawImageDataset, Preprocessed
from utils.bounding_boxes import BoundingBoxes

KnitableDataset = Union[LoadDataset, RawImageDataset, Preprocessed]


class Knit(Dataset):
    """
    Enrich an existing dataset by adding related information from external
    sources, e. g. CSV files.

    Args:
        dataset (Dataset): An existing dataset.
        image_csv (str): Path to the image CSV file.
        study_csv (str): Path to the study CSV file.
    """
    dataset: KnitableDataset
    image_ids: list[str]
    image_table: pd.DataFrame
    study_table: pd.DataFrame
    bounding_boxes: list[BoundingBoxes]
    label_type: Optional[type]

    guessable: Final[list[str]] = ["image_ids", "image_table", "study_table",
                                   "bounding_boxes", "label_type"]
    id_pattern: Final[re.Pattern] = re.compile("^[a-z0-9]{12}$")

    def __init__(self, dataset: KnitableDataset,
                 image_csv: Optional[str] = None,
                 study_csv: Optional[str] = None,
                 image_ids: Optional[list[str]] = None,
                 bounding_boxes: Optional[Sequence[BoundingBoxes]] = None):
        self.dataset = dataset

        if image_csv is not None:
            self.image_table = pd.read_csv(image_csv)
        if study_csv is not None:
            self.study_table = pd.read_csv(study_csv)
        if image_ids is not None:
            self.image_ids = image_ids
        if bounding_boxes is not None:
            self.bounding_boxes = bounding_boxes

        for name in self.guessable:
            if not hasattr(self, name):
                try:
                    guess = self._try_to_guess(dataset, name)
                    setattr(self, name, guess)
                    warn(f"Guessed {name} from {dataset.__class__.__name__}")
                except Exception as e:
                    warn(f"{e.__class__.__name__}: {e}")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        return self.dataset[index]

    @classmethod
    def _try_to_guess(cls, dataset: KnitableDataset, name: str) -> Any:
        if name in cls.guessable and hasattr(dataset, name):
            return getattr(dataset, name)
        if name == "image_ids":
            filenames = getattr(dataset, "filename", None)
            if filenames is None:
                raise KeyError
            if len(filenames) != len(dataset):
                raise ValueError("Lengths differ.")
            image_ids = [fn.split(".")[0].split("_")[0].split("/")[-1]
                         for fn in filenames]
            assert all(cls.id_pattern.match(id)
                       for id in image_ids), "Image ID guess failed."
            return image_ids
        raise NotImplementedError(f"{name} not guessable from "
                                  + f"{dataset.__class__.__name__}")
