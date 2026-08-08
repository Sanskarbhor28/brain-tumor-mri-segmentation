import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Standard Residual Block
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # First convolution
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second convolution
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # Shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):

        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

class ResidualUNet(nn.Module):
     def __init__(self):
        super().__init__()

        # ============================================================
        # Encoder
        # ============================================================

        self.enc1 = ResidualBlock(1, 64)
        self.enc2 = ResidualBlock(64, 128)
        self.enc3 = ResidualBlock(128, 256)
        self.enc4 = ResidualBlock(256, 512)

        self.pool = nn.MaxPool2d(2)

        # ============================================================
        # Bottleneck
        # ============================================================

        self.bottleneck = ResidualBlock(512, 1024)

        # ============================================================
        # Decoder
        # ============================================================

        self.up4 = nn.ConvTranspose2d(
            1024,
            512,
            kernel_size=2,
            stride=2
        )

        self.dec4 = ResidualBlock(
            1024,
            512
        )

        # ------------------------------------------------------------

        self.up3 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2
        )

        self.dec3 = ResidualBlock(
            512,
            256
        )

        # ------------------------------------------------------------

        self.up2 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.dec2 = ResidualBlock(
            256,
            128
        )

        # ------------------------------------------------------------

        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec1 = ResidualBlock(
            128,
            64
        )

        # ============================================================
        # Output Layer
        # ============================================================

        self.out = nn.Conv2d(
            64,
            1,
            kernel_size=1
        )

     def forward(self, x):

        # ============================================================
        # Encoder
        # ============================================================

        e1 = self.enc1(x)

        e2 = self.enc2(self.pool(e1))

        e3 = self.enc3(self.pool(e2))

        e4 = self.enc4(self.pool(e3))

        # ============================================================
        # Bottleneck
        # ============================================================

        b = self.bottleneck(self.pool(e4))

        # ============================================================
        # Decoder Level 4
        # ============================================================

        d4 = self.up4(b)

        d4 = torch.cat([d4, e4], dim=1)

        d4 = self.dec4(d4)

        # ============================================================
        # Decoder Level 3
        # ============================================================

        d3 = self.up3(d4)

        d3 = torch.cat([d3, e3], dim=1)

        d3 = self.dec3(d3)

        # ============================================================
        # Decoder Level 2
        # ============================================================

        d2 = self.up2(d3)

        d2 = torch.cat([d2, e2], dim=1)

        d2 = self.dec2(d2)

        # ============================================================
        # Decoder Level 1
        # ============================================================

        d1 = self.up1(d2)

        d1 = torch.cat([d1, e1], dim=1)

        d1 = self.dec1(d1)

        # ============================================================
        # Output
        # ============================================================

        return self.out(d1)