import torch.nn as nn
import torch


def ConvBlock(in_channels, out_channels, kernel_size, padding, batch_norm):
    if batch_norm:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

def Up(in_channels, out_channels, upsample_conv):
    if upsample_conv:
        return nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear'),
                    nn.Conv2d(in_channels, out_channels, 1)
                   )
    else:
        return nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)


class Unet(nn.Module):
    """
    U-Net architecture (modified) according to Ronneberger
    """

    def __init__(self, upsample_conv = False, batch_norm = False):
        super().__init__()

        # First Down Block
        self.down_block1 = ConvBlock(1, 64, 3, padding='same', batch_norm=batch_norm)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Second Down Block
        self.down_block2 = ConvBlock(64, 128, 3, padding='same', batch_norm=batch_norm)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Third Down Block
        self.down_block3 = ConvBlock(128, 256, 3, padding='same', batch_norm=batch_norm)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Fourth Down Block
        self.down_block4 = ConvBlock(256, 512, 3, padding='same', batch_norm=batch_norm)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024, 3, padding='same', batch_norm=batch_norm)

        # First Up Block
        self.up1 = Up(1024, 512, upsample_conv)
        self.up_block1 = ConvBlock(512+512, 512, 3, padding='same', batch_norm=batch_norm)

        # Second Up Block
        self.up2 = Up(512, 256, upsample_conv)
        self.up_block2 = ConvBlock(256+256, 256, 3, padding='same', batch_norm=batch_norm)

        # Third Up Block
        self.up3 = Up(256, 128, upsample_conv)
        self.up_block3 = ConvBlock(128+128, 128, 3, padding='same', batch_norm=batch_norm)

        # Fourth Up Block
        self.up4 = Up(128, 64, upsample_conv)
        self.up_block4 = ConvBlock(64+64, 64, 3, padding='same', batch_norm=batch_norm)

        # Final Layer
        self.final = nn.Linear(64, 1)


    def forward(self, x):

        # -------#
        # Encoder
        #--------#

        x1 = self.down_block1(x)
        x = self.pool1(x1)

        x2 = self.down_block2(x)
        x = self.pool2(x2)

        x3 = self.down_block3(x)
        x = self.pool3(x3)

        x4 = self.down_block4(x)
        x = self.pool4(x4)

        x = self.bottleneck(x)

        # -------#
        # Decoder
        #--------#

        x = self.up1(x)
        x = torch.cat([x4, x], dim=1)  # Skip-connection
        x = self.up_block1(x)

        x = self.up2(x)
        x = torch.cat([x3, x], dim=1)  # Skip-connection
        x = self.up_block2(x)

        x = self.up3(x)
        x = torch.cat([x2, x], dim=1)  # Skip-connection
        x = self.up_block3(x)

        x = self.up4(x)
        x = torch.cat([x1, x], dim=1)  # Skip-connection
        x = self.up_block4(x)

        x = self.final(x)

        return x
