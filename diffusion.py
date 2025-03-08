import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()

        self.enc1 = self.conv_block(in_channels, 32)  
        self.enc2 = self.conv_block(32, 64)  

        # Bottleneck
        self.bottleneck = self.conv_block(64, 128)  

        self.up1 = self.upsample(128, 64)
        self.dec1 = self.conv_block(128, 64)
        self.up2 = self.upsample(64, 32)
        self.dec2 = self.conv_block(64, 32)

        self.final = nn.Conv2d(32, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def upsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)  # Reduce artifacts
        )

    def forward(self, x):
        e1 = self.enc1(x)  
        e2 = self.enc2(e1)  

        b = self.bottleneck(e2)  

        d1 = self.up1(b)
        d1 = F.interpolate(d1, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d1 = torch.cat([d1, e2], dim=1)  
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        d2 = F.interpolate(d2, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)

        return self.final(d2)  