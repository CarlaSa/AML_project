import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from datasets import LoadDataset, CustomOutput, Knit
from datasets.custom_output import image_tensor, image_id, study_label, \
    study_id, bounding_boxes

loaded_data = LoadDataset("_data/preprocessed256", image_dtype=float,
                          label_dtype=float)
knit_data = Knit(loaded_data,
                 image_csv="_data/train_image_level.csv",
                 study_csv="_data/train_study_level.csv")

data = CustomOutput(knit_data,
                    image_tensor, bounding_boxes, image_id,
                    study_label, study_id)

labels5 = []
for img, boxes, img_id, stud_label, stud_id in tqdm(data):
    labels5.append([boxes.sum(), *stud_label])
labels5
