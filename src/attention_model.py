import torch
import torch.nn as nn

# Reuse the DoubleConv block from your existing model.py
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class AttentionBlock(nn.Module):
    """
    Attention Gate from:
    Attention U-Net (Oktay et al., 2018)

    g = decoder feature (gating signal)
    x = encoder skip feature
    """

    def __init__(self, gate_channels, skip_channels, inter_channels):
        super().__init__()

        # Transform decoder feature
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels)
        )

        # Transform encoder feature
        self.W_x = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels)
        )

        # Generate attention coefficients
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):

        g1 = self.W_g(g)
        x1 = self.W_x(x)

        attention = self.relu(g1 + x1)
        attention = self.psi(attention)

        # Apply attention to encoder feature
        return x * attention


class AttentionUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # =========================
        # Encoder
        # =========================

        self.enc1 = DoubleConv(1, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        # =========================
        # Bottleneck
        # =========================

        self.bottleneck = DoubleConv(512, 1024)

        # =========================
        # Decoder
        # =========================

        self.up4 = nn.ConvTranspose2d(
            1024, 512,
            kernel_size=2,
            stride=2
        )

        self.att4 = AttentionBlock(
            gate_channels=512,
            skip_channels=512,
            inter_channels=256
        )

        self.dec4 = DoubleConv(1024, 512)

        # -------------------------

        self.up3 = nn.ConvTranspose2d(
            512, 256,
            kernel_size=2,
            stride=2
        )

        self.att3 = AttentionBlock(
            gate_channels=256,
            skip_channels=256,
            inter_channels=128
        )

        self.dec3 = DoubleConv(512, 256)

        # -------------------------

        self.up2 = nn.ConvTranspose2d(
            256, 128,
            kernel_size=2,
            stride=2
        )

        self.att2 = AttentionBlock(
            gate_channels=128,
            skip_channels=128,
            inter_channels=64
        )

        self.dec2 = DoubleConv(256, 128)

        # -------------------------

        self.up1 = nn.ConvTranspose2d(
            128, 64,
            kernel_size=2,
            stride=2
        )

        self.att1 = AttentionBlock(
            gate_channels=64,
            skip_channels=64,
            inter_channels=32
        )

        self.dec1 = DoubleConv(128, 64)

        # =========================
        # Output Layer
        # =========================

        self.out = nn.Conv2d(
            64,
            1,
            kernel_size=1
        )

    def forward(self, x):

        # =========================
        # Encoder
        # =========================

        e1 = self.enc1(x)

        e2 = self.enc2(self.pool(e1))

        e3 = self.enc3(self.pool(e2))

        e4 = self.enc4(self.pool(e3))

        # =========================
        # Bottleneck
        # =========================

        b = self.bottleneck(self.pool(e4))

        # =========================
        # Decoder Level 4
        # =========================

        d4 = self.up4(b)

        e4_att = self.att4(d4, e4)

        d4 = torch.cat([d4, e4_att], dim=1)

        d4 = self.dec4(d4)

        # =========================
        # Decoder Level 3
        # =========================

        d3 = self.up3(d4)

        e3_att = self.att3(d3, e3)

        d3 = torch.cat([d3, e3_att], dim=1)

        d3 = self.dec3(d3)

        # =========================
        # Decoder Level 2
        # =========================

        d2 = self.up2(d3)

        e2_att = self.att2(d2, e2)

        d2 = torch.cat([d2, e2_att], dim=1)

        d2 = self.dec2(d2)

        # =========================
        # Decoder Level 1
        # =========================

        d1 = self.up1(d2)

        e1_att = self.att1(d1, e1)

        d1 = torch.cat([d1, e1_att], dim=1)

        d1 = self.dec1(d1)

        # =========================
        # Output
        # =========================

        return self.out(d1)