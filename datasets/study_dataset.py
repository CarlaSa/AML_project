import pandas as pd
from torch.utils.data import Dataset
import numpy as np

from typing import Tuple

LABEL_KEYS = ["Negative for Pneumonia", "Typical Appearance",
              "Indeterminate Appearance", "Atypical Appearance"]


class StudyDataset(Dataset):
    """
    Dataset featuring the labels (NumPy array) versus the study id (str) for
    each study.

    Usage:
        data = StudyDataset("data/train_study_level.csv")
    """
    study_table: pd.DataFrame

    def __init__(self, study_csv: str) -> None:
        self.study_table = pd.read_csv(study_csv)

    def __len__(self) -> int:
        return len(self.study_table)

    def __getitem__(self, index: int) -> Tuple[str, np.ndarray]:
        meta = self.study_table.iloc[index]
        study_id = meta["id"].split("_study")[0]
        assert len(study_id) == 12

        return study_id, np.array(meta[LABEL_KEYS], dtype=bool)
