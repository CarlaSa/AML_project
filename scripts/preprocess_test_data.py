import os
import json
import pydicom
import torchvision
import pandas as pd
from tqdm import tqdm

from trafo import Compose
from trafo.type_mutating import NDArrayTo3dTensor
from trafo.type_mutating.dicom_to_ndarray import DicomToNDArray
from trafo.color import Color0ToMax, TruncateGrayValues
from trafo.box_mutating import CropToLungs, CropPadding, Scale, \
    RoundBoundingBoxes

INPUT_PATH = "_data/test"
OUTPUT_PATH = "_data/test_preprocessed256"
EXCLUDE_DICOM = {"PixelData"}
TRAFO = Compose(
   DicomToNDArray(),
   TruncateGrayValues(),
   Color0ToMax(255),
   CropToLungs(),
   CropPadding(),
   NDArrayTo3dTensor(),
   Scale((256, 256)),
   Color0ToMax(1),
   RoundBoundingBoxes()
)


def main():
    table = pd.DataFrame()
    keywords = {}
    os.makedirs(os.path.join(OUTPUT_PATH, "images"), exist_ok=True)
    for study_id in tqdm(os.listdir(INPUT_PATH)):
        study_path = os.path.join(INPUT_PATH, study_id)
        for sub_dir in os.listdir(study_path):
            image_dir_path = os.path.join(study_path, sub_dir)
            for dcm_filename in os.listdir(image_dir_path):
                image_path = os.path.join(image_dir_path, dcm_filename)
                image_id = dcm_filename.split(".dcm")[0]
                dicom = pydicom.read_file(image_path)
                assert dicom.StudyInstanceUID == study_id
                assert dicom.SeriesInstanceUID == sub_dir
                assert dicom.SOPInstanceUID == image_id
                dicom.__repr__()
                elements = list(dicom.elements())
                array = dicom.pixel_array
                png_filename = f"{study_id}_{image_id}.png"
                table = table.append({"filename": png_filename,
                                      "pixel_min": array.min(),
                                      "pixel_max": array.max(),
                                      "pixel_mean": array.mean(),
                                      "pixel_std": array.std(),
                                      **{el.keyword: el.value
                                         for el in elements
                                         if el.keyword not in EXCLUDE_DICOM}},
                                     ignore_index=True)
                keywords = {**keywords, **{el.keyword: el.name
                                           for el in elements}}
                image = TRAFO(dicom)
                torchvision.utils.save_image(image, os.path.join(OUTPUT_PATH,
                                                                 "images",
                                                                 png_filename))
    table.to_csv(os.path.join(OUTPUT_PATH, "dicom_meta.csv"))
    with open(os.path.join(OUTPUT_PATH, "dicom_keywords.json"), "w") as f:
        json.dump(keywords, f)


if __name__ == '__main__':
    main()
