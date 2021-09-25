import os
import torch
import pandas as pd
from torchvision import io
from typing import Tuple
from torch.utils.data import Dataset

from utils import CanvasTrafoRecorder


class TestData(Dataset):
    """Dataset featuring ids versus image tensor."""

    input_dir: str
    table: pd.DataFrame

    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.table = pd.read_csv(os.path.join(input_dir, "images.csv"))

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str, str, dict]:
        meta = self.table.iloc[index]
        filename = meta.filename
        path = os.path.join(self.input_dir, "images", filename)
        study_id, image_id = filename.split(".")[0].split("_")
        assert len(study_id) == 12 and len(image_id) == 12
        recorder_kwargs = {key: meta[key] for key
                           in CanvasTrafoRecorder.__annotations__.keys()}
        tensor = io.read_image(path, io.ImageReadMode.GRAY).to(float)/255
        return tensor, study_id, image_id, recorder_kwargs
