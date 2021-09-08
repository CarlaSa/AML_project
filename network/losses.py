
import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_score(predictions, masks, epsilon_smooth=1):
    """
    Calculate the Dice Score for the prediction set and the target set
    """

    # Flatten predictions and masks
    y_hat = predictions.view(-1)
    y = masks.view(-1)

    # Compute 2*|X∩Y|
    numerator = 2*torch.sum(y_hat*y) + epsilon_smooth
    # Compute |X|+|Y|
    denominator = torch.sum(y_hat)+torch.sum(y) + epsilon_smooth
    return numerator/denominator


class DiceLoss(nn.Module):
    """
    Calculate a Loss based on the Dice Score
    """
    def __init__(self, epsilon_smooth=1, weight=None, size_average=True):
        super(DiceLoss, self).__init__()
        self.epsilon_smooth = epsilon_smooth

    def forward(self, (predictions, masks):
        dice_score = dice_score(predictions, masks, self.epsilon_smooth)
        return 1 - dice_score


class BCEandDiceLoss(nn.Module):
    """
    Calculate a Loss based on the Binary Crossentropy and the Dice Score
    """
    def __init__(self, epsilon_smooth=1, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.epsilon_smooth = epsilon_smooth

    def forward(self, predictions, masks):
        dice_score = dice_score(predictions, masks, self.epsilon_smooth)
        BCE = F.binary_cross_entropy(predictions, masks)
        return BCE + (1-dice_score)
