import math
import numpy as np
from typing import Optional, NamedTuple


# TEMPORARY
import matplotlib.pyplot as plt


def cropping(img: np.array, bounding_boxes: Optional[np.array] = None) \
        -> tuple[int, int, int, int]:
    """
    Crop to the relevant part of the image (lung)
    Mainly based on https://www.kaggle.com/davidbroberts/cropping-chest-x-rays
    """
    class Thresholds:
        binary: float
        transversal: float
        longitudinal_top: float
        longitudinal_bottom: float

    height, width = img.shape

    class Limits(NamedTuple):
        right: float = 0.6 * width
        left: float = 0.4 * width
        top: float = 0.25 * height
        bottom: float = 0.45 * height

    limits = Limits()

    # Threshold to binarize the data in Background
    thresholds = Thresholds()
    thresholds.binary = np.mean(img)  # 150

    # TODO: maybe determine thresh via histograms...
    img[img < thresholds.binary] = 0
    img[img >= thresholds.binary] = 1

    # calculate the means of the columns and rows of the binarized image
    row_averages = np.mean(img, axis=1)
    column_averages = np.mean(img, axis=0)

    # Set Thresholds
    thresholds.transversal = 0.9 * np.mean(
        column_averages)  # 0.6*np.max(row_averages)
    # 190/255
    thresholds.longitudinal_top = 0.4*np.max(row_averages)  # 100/255
    # 0.94*np.max(row_averages)  # 240/255
    thresholds.longitudinal_bottom = np.min(
        [1.3*np.mean(row_averages), 0.94*np.max(row_averages)])

    # TEMPORARY
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(column_averages)
    ax[0].axhline(thresholds.transversal)
    ax[0].axhline(np.mean(column_averages), color="black")
    ax[0].set_title("Column Averages")
    ax[1].plot(row_averages)
    ax[1].axhline(thresholds.longitudinal_top)
    ax[1].axhline(thresholds.longitudinal_bottom)
    ax[1].axhline(np.mean(row_averages), color="black")
    ax[1].set_title("Row Averages")
    plt.show()

    # Crop left and right boundaries
    # Check condition: lower or higher threshold for each column
    col_cond = column_averages > thresholds.transversal
    # Set how many columns have to fullfill this condition subsequently
    cnt_crit_col = 3  # np.max([5, int(width*0.005)])
    # Go from left edge to the center and check whether the mean of the pixel of
    # a column is higher than the transversal_thresh. if so increase an internal
    # counter. if the counter reaches a critical value, the image will be cropped
    # there. if the condition is not fullfilled reset the counter
    merged_col_cond_left = col_cond[:-cnt_crit_col]
    for i in range(1, cnt_crit_col):
        merged_col_cond_left *= col_cond[i:-cnt_crit_col+i]
    left_crop = np.argmax(merged_col_cond_left)
    # To the same for the right boundary by goind from right to left
    merged_col_cond_right = col_cond[cnt_crit_col:]
    for i in range(1, cnt_crit_col+1):
        merged_col_cond_right *= col_cond[cnt_crit_col-i:-i]
    right_crop = width - 1 - np.argmax(merged_col_cond_right[::-1])

    # Get Cropping Edge for above boundary
    row_cond = row_averages > thresholds.longitudinal_top
    cnt_crit_row_top = 3  # np.max([5, int(height*0.005)])
    merged_row_cond_top = row_cond[:-cnt_crit_row_top]
    for i in range(1, cnt_crit_row_top):
        merged_row_cond_top *= row_cond[i:-cnt_crit_row_top+i]
    top_crop = np.argmax(merged_row_cond_top)

    # Get Cropping Edge for bottom boundary
    # Check when the row reaches the lung, i.e. whether the mean of the row is
    # lower than the threshold
    row_cond_bottom = row_averages < thresholds.longitudinal_bottom
    cnt_crit_row_bottom = np.max([10, int(height*0.005)])
    merged_row_cond_bottom = row_cond_bottom[cnt_crit_row_bottom:]
    for i in range(1, cnt_crit_row_bottom+1):
        merged_row_cond_bottom *= row_cond_bottom[cnt_crit_row_bottom-i:-i]
    bottom_crop = height - 1 - np.argmax(merged_row_cond_bottom[::-1])

    # Add some pixels at the bottom for of padding such that
    # the costophrenic angles are not cutted off
    bottom_crop = int(np.min(
        [height-1, bottom_crop + (bottom_crop-top_crop) * 0.0]))  # 0.18

    bottom_above_threshold = row_averages >= thresholds.longitudinal_bottom
    bottom_above_threshold[:math.ceil(limits.bottom)] = False
    bottom_crop = np.min([bottom_crop,
                          np.argmax(bottom_above_threshold)])

    # Consider manual crop limits
    # TODO ANSTATT KEINEN CROP ANZUWENDEN WENN LIMITS ERREICHT WERDEN,
    # KÖNNTE MAN ES AUCH NOCHMALS MIT ANDEREM TRHESHOLD PROBIEREN
    if right_crop < 0.60*width:
        right_crop = width
    if left_crop > 0.40*width:
        left_crop = 0
        print("zeroed left crop")
    if top_crop > 0.25*height:
        top_crop = 0
    if bottom_crop < 0.45*height:
        bottom_crop = height

    if bounding_boxes is not None and bounding_boxes.sum() > 0:
        bounding_boxes = bounding_boxes[bounding_boxes[:, 2] != 0]
        if left_crop > np.min(bounding_boxes[:, 0]):
            left_crop = math.floor(np.min(bounding_boxes[:, 0]))
            print("limited left crop to most left bounding box")
        if top_crop > np.min(bounding_boxes[:, 1]):
            top_crop = math.floor(np.min(bounding_boxes[:, 1]))
        if right_crop < np.max(bounding_boxes[:, 0] + bounding_boxes[:, 2]):
            right_crop = math.ceil(
                np.max(bounding_boxes[:, 0] + bounding_boxes[:, 2]))
        if bottom_crop < np.max(bounding_boxes[:, 1] + bounding_boxes[:, 3]):
            bottom_crop = math.ceil(
                np.max(bounding_boxes[:, 1] + bounding_boxes[:, 3]))

    return (left_crop, right_crop, top_crop, bottom_crop)
