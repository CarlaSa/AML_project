import numpy as np


def cropping(img: np.array) -> tuple[int, int, int, int]:
    """
    Crop the relevant part of the image (lung)
    Mainly based on https://www.kaggle.com/davidbroberts/cropping-chest-x-rays
    """
    height, width = img.shape

    thresh = 150  # Threshold to binarize the data in Background

    # TODO: maybe determine thresh via histograms...
    img[img < thresh] = 0
    img[img >= thresh] = 1

    # calculate the means of the columns and rows of the binarized image
    row_averages = np.mean(img, axis=1)
    column_averages = np.mean(img, axis=0)

    # Crop left and right boundaries
    transversal_thresh = 0.5  # 190/255
    # Check condition: lower or higher threshold for each column
    col_cond = column_averages > transversal_thresh
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
    longitudinal_thresh_top = 0.05  # 100/255
    row_cond = row_averages > longitudinal_thresh_top
    cnt_crit_row = 3  # np.max([5, int(height*0.005)])
    merged_row_cond_top = row_cond[:-cnt_crit_row]
    for i in range(1, cnt_crit_col):
        merged_row_cond_top *= row_cond[i:-cnt_crit_row+i]
    top_crop = np.argmax(merged_row_cond_top)

    # Get Cropping Edge for bottom boundary
    longitudinal_thresh_bottom = 0.6  # 240/255
    # Check when the row reaches the lung, i.e. whether the mean of the row is
    # lower than the threshold
    row_cond_bottom = row_averages < longitudinal_thresh_bottom
    merged_row_cond_bottom = row_cond_bottom[cnt_crit_row:]
    for i in range(1, cnt_crit_col+1):
        merged_row_cond_bottom *= row_cond_bottom[cnt_crit_row-i:-i]
    bottom_crop = height - 1 - np.argmax(merged_row_cond_bottom[::-1])

    # Add some pixels at the bottom for of padding such that
    # the costophrenic angles are not cutted off
    bottom_crop = int(np.min(
        [height-1, bottom_crop + (bottom_crop-top_crop) * 0.18]))

    return (left_crop, right_crop, top_crop, bottom_crop)
