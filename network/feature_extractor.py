"""
We Use a Version of ResNet for Feature Extraction on the full images.
"""

import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """
    Our ResNet Blocks consist of 2 Convolutional Layers with Batch 
    normalization and ReLU as activation function; and of course a skip 
    connection.
    
    
    """

    def __init__(self, in_ch, out_ch, stri = 1, downsample = None):
        super().__init__()
        """
        parameters:

        in_ch:  number of channels of the input
        out_ch:     number of channels of the output
        downsample:     Pytorch layers that get applied in the skip connection,
                        to downsample if the dimension of x and self.Block(x) do
                        not match. This is the case if in_ch =/= out_ch.

        """

        # build main block
        layers = []
        layers.append(nn.Conv2d(in_ch, out_ch, 3,\
                padding = 1, stride = stri, bias = False))
        layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(out_ch, out_ch, 3,\
                padding = 1, stride = 1, bias = False))
        layers.append(nn.BatchNorm2d(out_ch))

        self.block = nn.Sequential(*layers)
        self.downsample = downsample
        self.relu = nn.ReLU()
    
    def forward(self, x):
        identity = x

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = identity + self.block(x)
        return self.relu(out)

class ResNet(nn.Module):
    def __init__(self, dims, out_shape = 4 , block = BasicBlock):
        """
        
        dims: list of 4 

        """
        super().__init__()
        self.current_ch = 64
        self.start = nn.Sequential(
                nn.Conv2d(1, self.current_ch, kernel_size = 7, stride = 2, padding = 3, 
                    bias = False),
                nn.BatchNorm2d(self.current_ch),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
            )
        self.layer1 = self._create_blocks(block, 64, dims[0])
        self.layer2 = self._create_blocks(block, 128, dims[1], stride = 2)
        self.layer3 = self._create_blocks(block, 256, dims[2], stride = 2)
        self.layer4 = self._create_blocks(block, 512, dims[3], stride = 2)

        self.end = nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Flatten(),
                nn.Linear(512, out_shape)
            )

    def _create_blocks(self, block, out_ch, num_blocks, stride = 1):
        """
        This function gives back num_block many layers of type block 
        If out_ch is different from self.current_ch, then in the first layer the 
        number of channels is changed and downsampling is applied. 
        """
        # If stride is set to something else than 1, the stride is used to 
        # downsample the images in a convolution layer.
        downsample = None
        if stride != 1 or self.current_ch != out_ch:
            downsample = nn.Sequential(
                    nn.Conv2d(self.current_ch, out_ch, 1, stride, bias = False),
                    nn.BatchNorm2d(out_ch)
                )
        layers = []
        # first layer changes the number of channels
        layers.append(block(self.current_ch, out_ch, stride, downsample))

        # now the current number of channels is out_ch
        self.current_ch = out_ch

        #the following number of layers do not change the number of channels.
        for _ in range(1, num_blocks):
            layers.append(
                block(self.current_ch, out_ch)
            )
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.start(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.end(x)
        return x

