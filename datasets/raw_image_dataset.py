import os
import pydicom
import pandas as pd
from torch.utils.data import Dataset

assert pydicom.pixel_data_handlers.pylibjpeg_handler.is_available()


class RawImageDataset(Dataset):
    """
    Dataset featuring raw meta data (as Pandas Series) versus the pydicom
    FileDataset for each image.

    Usage:
        data = RawImageDataset("data/train", "data/train_image_level.csv")
    """
    image_table: pd.DataFrame
    input_dir: str

    def __init__(self, input_dir: str, image_csv: str) -> None:
        self.image_table = pd.read_csv(image_csv)
        self.input_dir = input_dir

    def __len__(self) -> int:
        return len(self.image_table)

    def __getitem__(self, index: int) -> tuple[pydicom.dataset.FileDataset,
                                               pd.core.series.Series]:
        dicom: pydicom.dataset.FileDataset
        meta: pd.core.series.Series

        meta = self.image_table.iloc[index]
        image_id = meta["id"].split("_image")[0]
        assert len(image_id) == 12, "Corrupt CSV data. Is this the right file?"
        study_id = meta["StudyInstanceUID"]
        image_path = find_dicom_path(self.input_dir, study_id, image_id)
        dicom = pydicom.read_file(image_path)
        return dicom, meta


def find_dicom_path(directory: str, study_id: str, image_id: str) -> str:
    study_path = os.realpath(os.path.join(directory, study_id))
    filename = image_id + ".dcm"
    for sub_dir in os.listdir(study_path):
        image_dir_path = os.path.join(study_path, sub_dir)
        if filename in os.listdir(image_dir_path):
            return os.path.join(image_dir_path, filename)
    raise RuntimeError("404 file not found")
