import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from datasets import LoadDataset, CustomOutput, Knit
from datasets.custom_output import image_tensor, image_id, study_label, \
    study_id, bounding_boxes

loaded_data = LoadDataset("_data/preprocessed256", image_dtype=float,
                          label_dtype=float)
knit_data = Knit(loaded_data,
                 image_csv="data/train_image_level.csv",
                 study_csv="data/train_study_level.csv")

data = CustomOutput(knit_data,
                    image_tensor, bounding_boxes, image_id,
                    study_label, study_id)

labels5 = []
stud_ids = []
for img, boxes, img_id, stud_label, stud_id in tqdm(data):
    labels5.append([boxes.sum(), *stud_label])
    stud_ids.append(stud_id)
labels5 = np.array(labels5)
stud_ids = np.array(stud_ids)
labels5.shape
stud_ids.shape

# filter out images without bounding bounding_boxes
no_bb_mask = labels5[:,0] == 0
no_bb_mask.shape
imgs_no_bb = labels5[no_bb_mask]
imgs_no_bb.shape
pneumonia_mask = imgs_no_bb[:, 1] == 0
imgs_no_bb_no_pneum = imgs_no_bb[pneumonia_mask]
imgs_no_bb_no_pneum.shape

stud_no_bb_no_pneum = stud_ids[no_bb_mask]
stud_no_bb_no_pneum = stud_no_bb_no_pneum[pneumonia_mask]
stud_no_bb_no_pneum_unique = np.unique(stud_no_bb_no_pneum)
stud_no_bb_no_pneum_unique.shape

# how many of the suspected studies have indeed no bounding boxes in any of their
# images?
stud_broken = 0

for stud_id in stud_no_bb_no_pneum_unique:
    stud_mask = stud_ids == stud_id
    temp_labels = labels5[stud_mask]
    if np.sum(temp_labels[:,0]) == 0:
        stud_broken += 1

stud_broken
