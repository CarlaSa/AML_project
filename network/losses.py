
import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_score(predictions: torch.Tensor, masks: torch.Tensor,
               epsilon_smooth: float=1., reduction: str="mean") -> torch.Tensor:
    """
    Calculate the Dice Score for the prediction set and the target set

    Args:
        predictions: predicted segmentation
                     needed shape (batch size, width, heigth) or
                                  (batch size, 1, width, height)
        masks: ground truth segmentation
               needed shape (batch size, width, height)
        epsilon_smooth: ensures no zero dividing
        reduction: specify whether the dice_scores of each instance ('none') shall
                   be returned or the mean dice score of the whole batch ('mean')

    Returns:
        torch.Tensor: dice scores for each instance within the batch: shape (batches)
    """

    if len(predictions.shape) < 3:
        print("WARNING: Please check shape of input of dice_score-function. "
              + "The calculated Dice Score could be wrong")

    # Remove unnecessary dimension in prediction:
    # (batch, 1, width, height) -> (batch, width, height)
    y_hat = predictions.squeeze()
    # Reshape: (batch, width, height) -> (batch, width*height)
    #y_hat = y_hat.reshape(y_hat.shape[0], y_hat.shape[1]*y_hat.shape[2])
    #y = masks.reshape(masks.shape[0], masks.shape[1]*masks.shape[2])
    y_hat = torch.flatten(y_hat, start_dim=1)
    y = torch.flatten(masks, start_dim=1)

    # Compute 2*|X∩Y|
    numerator = 2*torch.sum(y_hat*y, dim=1) + epsilon_smooth
    # Compute |X|+|Y|
    denominator = torch.sum(y_hat, dim=1)+torch.sum(y, dim=1) + epsilon_smooth

    if reduction == "mean":
        return torch.mean(numerator/denominator)
    elif reduction == "none":
        return numerator/denominator
    else:
        print("Error: Wrong keyword for reduction in dice_score")
        return None


class DiceLoss(nn.Module):
    """
    Calculate a Loss based on the Dice Score

    Returns:
        Mean Dice Loss of an batch
    """

    def __init__(self, epsilon_smooth=1, weight=None, size_average=True):
        super(DiceLoss, self).__init__()
        self.epsilon_smooth = epsilon_smooth

    def forward(self, predictions, masks):
        mean_dce_score = dice_score(predictions, masks, self.epsilon_smooth, 'mean')
        return 1 - mean_dce_score



class BCEandDiceLoss(nn.Module):
    """
    Calculate a Loss based on the Binary Crossentropy and the Dice Score

    Returns:
        The mean of the Dice Loss and the BCE Loss of an batch
    """
    def __init__(self, epsilon_smooth=1, weight=None, size_average=True):
        super(BCEandDiceLoss, self).__init__()
        self.epsilon_smooth = epsilon_smooth

    def forward(self, predictions, masks):
        mean_dce_score = dice_score(predictions, masks, self.epsilon_smooth, 'mean')
        BCE = F.binary_cross_entropy(predictions.squeeze(), masks)
        return BCE + (1-mean_dce_score)
