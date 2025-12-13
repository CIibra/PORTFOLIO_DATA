# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)

class UNet3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=16):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base_ch)
        self.enc2 = DoubleConv(base_ch, base_ch*2)
        self.enc3 = DoubleConv(base_ch*2, base_ch*4)
        self.pool = nn.MaxPool3d(2)
        self.up2 = nn.ConvTranspose3d(base_ch*4, base_ch*2, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose3d(base_ch*2, base_ch, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_ch*4, base_ch*2)
        self.dec1 = DoubleConv(base_ch*2, base_ch)
        self.final = nn.Conv3d(base_ch, out_ch, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)           # [B, base, D,H,W]
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.up2(e3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        out = self.final(d1)
        return out