import torch.nn as nn
import torch
from typing import List, Union, Dict


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
        return nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                             nn.Conv2d(in_channels, out_channels, 1))
    else:
        return nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)


class Unet(nn.Module):
    """
    U-Net architecture (modified) according to Ronneberger
    """
    hyperparameters: Dict[str, Union[bool, int]]
    down_blocks: List[nn.Sequential]
    pools: List[nn.MaxPool2d]
    bottleneck: nn.Sequential
    up_blocks: List[nn.Sequential]
    ups: List[Union[nn.Sequential, nn.ConvTranspose2d]]
    final: nn.Conv2d

    def __init__(self, upsample_conv: bool = False, batch_norm: bool = False,
                 n_blocks: int = 4, n_initial_block_channels: int = 64):
        super().__init__()

        # Reproducibility
        self.hyperparameters = {
            "upsample_conv": upsample_conv,
            "batch_norm": batch_norm,
            "n_blocks": n_blocks,
            "n_initial_block_channels": n_initial_block_channels
        }

        # Encoder
        chan_in, chan_out = 1, n_initial_block_channels
        self.down_blocks, self.pools = [], []
        for i in range(n_blocks):
            self.pools.append(nn.MaxPool2d(2, 2))
            self.down_blocks.append(ConvBlock(chan_in, chan_out, 3,
                                              padding='same',
                                              batch_norm=batch_norm))
            chan_in, chan_out = chan_out, 2 * chan_out

        self.bottleneck = ConvBlock(chan_in, chan_out, 3, padding='same',
                                    batch_norm=batch_norm)

        # Decoder
        self.up_blocks, self.ups = [], []
        for i in range(n_blocks):
            chan_in, chan_out = chan_out, chan_out / 2
            self.ups.append(Up(chan_in, chan_out, upsample_conv))
            self.up_blocks.append(ConvBlock(chan_out + chan_out, chan_out, 3,
                                            padding='same',
                                            batch_norm=batch_norm))

        chan_in, chan_out = chan_out, 1
        self.final = nn.Conv2d(chan_in, chan_out, 1, padding="same")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_con: List[torch.Tensor] = []

        # Encoder
        for down_block, pool in zip(self.down_blocks, self.down_pools):
            x = down_block(x)
            skip_con.append(x)
            x = pool(x)
        x = self.bottleneck(x)

        # Decoder
        for up_block, up in zip(self.up_blocks, self.ups):
            x = up(x)
            x = torch.cat([skip_con.pop(), x], dim=1)  # Skip-connection
            x = up_block(x)
        x = self.final(x)
        x = torch.sigmoid(x)

        return x.squeeze()
