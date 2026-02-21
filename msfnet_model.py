import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------
# Conv Block
# ------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# ------------------------------
# Multi-Scale Attention (MSA)
# ------------------------------
class MSA(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 1)
        self.conv3 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv5 = nn.Conv2d(ch, ch, 5, padding=2)

        self.fc1 = nn.Conv2d(ch, ch // 8, 1)
        self.fc2 = nn.Conv2d(ch // 8, ch, 1)

    def forward(self, x):
        s = self.conv1(x) + self.conv3(x) + self.conv5(x)

        w = F.adaptive_avg_pool2d(s, 1)
        w = torch.sigmoid(self.fc2(F.relu(self.fc1(w))))

        return s * w


# ------------------------------
# Cross-Layer Feature Module (CLFM)
# ------------------------------
class CLFM(nn.Module):
    def __init__(self, in_ch1, in_ch2, out_ch):
        super().__init__()
        self.reduce1 = nn.Conv2d(in_ch1, out_ch, 1)
        self.reduce2 = nn.Conv2d(in_ch2, out_ch, 1)

    def forward(self, f1, f2):
        if f1.shape[-1] != f2.shape[-1]:
            f2 = F.interpolate(
                f2,
                size=f1.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        return self.reduce1(f1) * self.reduce2(f2)


# ------------------------------
# Dual Input Fusion Module (DIFM)
# ------------------------------
class DIFM(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, ch // 8, 1)
        self.fc2 = nn.Conv2d(ch // 8, ch, 1)

    def weigh(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = torch.sigmoid(self.fc2(F.relu(self.fc1(w))))
        return x * w

    def forward(self, e, d):
        return self.weigh(e) + self.weigh(d)


# ------------------------------
# MSF-Net Architecture
# ------------------------------
class MSFNet(nn.Module):
    def __init__(self, in_ch=1, base=32):
        super().__init__()

        # Encoder
        self.e1 = ConvBlock(in_ch, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.e4 = ConvBlock(base * 4, base * 8)
        self.b = ConvBlock(base * 8, base * 16)

        self.msa = MSA(base * 16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, 2)
        self.clf4 = CLFM(base * 8, base * 4, base * 8)
        self.d4 = ConvBlock(base * 8 + base * 8, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.d3 = ConvBlock(base * 4 + base * 2, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.d2 = ConvBlock(base * 2 + base, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.d1 = ConvBlock(base + base, base)

        self.dif = DIFM(base)

        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        # Encoder
        c1 = self.e1(x)
        p1 = F.max_pool2d(c1, 2)

        c2 = self.e2(p1)
        p2 = F.max_pool2d(c2, 2)

        c3 = self.e3(p2)
        p3 = F.max_pool2d(c3, 2)

        c4 = self.e4(p3)
        p4 = F.max_pool2d(c4, 2)

        b = self.msa(self.b(p4))

        # Decoder
        u4 = self.up4(b)
        f4 = self.clf4(c4, c3)
        d4 = self.d4(torch.cat([u4, f4], dim=1))

        u3 = self.up3(d4)
        c2_r = F.interpolate(c2, size=u3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.d3(torch.cat([u3, c2_r], dim=1))

        u2 = self.up2(d3)
        c1_r = F.interpolate(c1, size=u2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.d2(torch.cat([u2, c1_r], dim=1))

        u1 = self.up1(d2)
        d1 = self.d1(torch.cat([u1, c1], dim=1))

        d1 = self.dif(c1, d1)

        return self.out(d1)
