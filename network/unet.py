import torch.nn as nn
import torch


def ConvBlock(in_channels, out_channels, kernel_size, padding, batch_norm,
              p_dropout):
    if batch_norm:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )
    if not batch_norm and p_dropout>0:
        """
        based on U-Net architecture of
        https://www.frontiersin.org/articles/10.3389/fnins.2019.00097/full
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Dropout(p_dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=padding),
            nn.ReLU(inplace=True),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=padding),
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

    def __init__(self, upsample_conv=False, batch_norm=False, p_dropout=0):
        super().__init__()

        # hyperparamaters
        self.hyperparameters = {"upsample_conv": upsample_conv,
                                "batch_norm": batch_norm,
                                "p_dropout": p_dropout
                                }

        # First Down Block
        self.down_block1 = ConvBlock(
            1, 64, 3, padding='same', batch_norm=batch_norm,
            p_dropout=p_dropout)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Second Down Block
        self.down_block2 = ConvBlock(
            64, 128, 3, padding='same', batch_norm=batch_norm,
            p_dropout=p_dropout)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Third Down Block
        self.down_block3 = ConvBlock(
            128, 256, 3, padding='same', batch_norm=batch_norm,
            p_dropout=p_dropout)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Fourth Down Block
        self.down_block4 = ConvBlock(
            256, 512, 3, padding='same', batch_norm=batch_norm,
            p_dropout=p_dropout)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = ConvBlock(
            512, 1024, 3, padding='same', batch_norm=batch_norm,
            p_dropout=p_dropout)

        # First Up Block
        self.up1 = Up(1024, 512, upsample_conv)
        self.up_block1 = ConvBlock(
            512+512, 512, 3, padding='same', batch_norm=batch_norm,
            p_dropout=p_dropout)

        # Second Up Block
        self.up2 = Up(512, 256, upsample_conv)
        self.up_block2 = ConvBlock(
            256+256, 256, 3, padding='same', batch_norm=batch_norm,
            p_dropout=p_dropout)

        # Third Up Block
        self.up3 = Up(256, 128, upsample_conv)
        self.up_block3 = ConvBlock(
            128+128, 128, 3, padding='same', batch_norm=batch_norm,
            p_dropout=p_dropout)

        # Fourth Up Block
        self.up4 = Up(128, 64, upsample_conv)
        self.up_block4 = ConvBlock(
            64+64, 64, 3, padding='same', batch_norm=batch_norm,
            p_dropout=p_dropout)

        # Final Layer
        self.final = nn.Conv2d(64, 1, 1, padding="same")

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
        x = torch.sigmoid(x)

        return x.squeeze()
