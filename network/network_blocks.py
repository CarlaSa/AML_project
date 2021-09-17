"""
In this file I tried to create a versatile template for creating models. 
Following this style guide for pytorch 
https://pythonrepo.com/repo/IgorSusmelj-pytorch-styleguide
"""


import torch.nn as nn
import torch.nn.functional as F
import torch
import warnings


class ConvBlock(nn.Module):
    """
    A Convolution Block consists of a Convolution Layer and ReLU as an 
    activation function. The keywords use_batchnorm and use_MaxPool add 
    layers of nn.BatchNorm2d and nn.MaxPool2d respectively.
    The Order of the layers is:
        Conv2d -> BatchNorm2d -> Maxpool2d -> ReLU
    """
    def __init__(self, in_channel, out_channel, kernelsize = 3, 
            padding = 1, use_batchnorm = False, use_MaxPool = True):
        super(ConvBlock, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channel, out_channel, kernelsize, 
            padding = padding))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channel))
        if use_MaxPool:
            layers.append(nn.MaxPool2d(2,2))
        layers.append(nn.ReLU())

        self.block = nn.Sequential(*layers)  
    
    def forward(self, x):
        return self.block(x)

class FCBlock(nn.Module):
    """
    Creates some fully connected layer with relu activation inbetween and 
    dropout if the parameter use_dropout is set to True. The number of 
    layers and the number of neurones is determined by the list shapes. 
    shapes[0] should be the input size, and shapes[-1] should be the 
    output size.

    """
    def __init__(self, shapes = [1024, 64, 64, 64, 4], 
                use_dropout = False):

        super(FCBlock, self).__init__()
        layers = []
        for i in range(len(shapes)-2):
            layers.append(nn.Linear(shapes[i], shapes[i+1]))
            layers.append(nn.ReLU())
            if use_dropout:
                layers.append(nn.Dropout(0.5))
        layers.append(nn.Linear(shapes[-2],shapes[-1]))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)