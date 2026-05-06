"""
matting/model.py — U-Net human matting model for Task 2.

Architecture: MobileNetV2 encoder (pretrained ImageNet) + lightweight decoder
              with skip connections.

Input  : RGB frame  (B, 3, H, W)  where H,W ∈ {256, 320}
Output : alpha matte (B, 1, H, W)  values ∈ [0, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


# ──────────────────────────────────────────────────────────────
# Decoder block
# ──────────────────────────────────────────────────────────────
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # Handle potential size mismatch
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                              align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ──────────────────────────────────────────────────────────────
# Main model
# ──────────────────────────────────────────────────────────────
class MattingUNet(nn.Module):
    """
    MobileNetV2-encoder U-Net for human alpha matting.
    The encoder is pretrained on ImageNet.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # ── Encoder (MobileNetV2) ──────────────────────────────
        base = tvm.mobilenet_v2(
            weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        )
        feat = base.features  # 19 sequential inverted-residual blocks

        # Extract skip-connection layers at multiple scales
        # MobileNetV2 feature map sizes for 256×256 input:
        #   feat[0]  : stride 2  →  128×128, 32-ch
        #   feat[1]  : stride 1  →  128×128, 16-ch   (first inverted block)
        #   feat[3]  : stride 2  →   64×64,  24-ch
        #   feat[6]  : stride 2  →   32×32,  32-ch
        #   feat[13] : stride 2  →   16×16,  96-ch
        #   feat[18] : stride 2  →    8×8,  320-ch   (bottleneck)
        self.enc0 = feat[0]           # → 128, 32
        self.enc1 = feat[1]           # → 128, 16
        self.enc2 = feat[2:4]         # → 64,  24
        self.enc3 = feat[4:7]         # → 32,  32
        self.enc4 = feat[7:14]        # → 16,  96
        self.enc5 = feat[14:-1]       # →  8, 320 (bottleneck before 1280 expansion)
        self.enc2 = nn.Sequential(*self.enc2)
        self.enc3 = nn.Sequential(*self.enc3)
        self.enc4 = nn.Sequential(*self.enc4)
        self.enc5 = nn.Sequential(*self.enc5)

        # ── Decoder ────────────────────────────────────────────
        self.dec4 = DecoderBlock(320, 96,  128)   #  8→16
        self.dec3 = DecoderBlock(128, 32,  64)    # 16→32
        self.dec2 = DecoderBlock(64,  24,  32)    # 32→64
        self.dec1 = DecoderBlock(32,  16,  16)    # 64→128
        self.dec0 = DecoderBlock(16,  32,  16)    # 128→256

        # ── Output ─────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Encoder
        s0 = self.enc0(x)    # (B,32,H/2,W/2)
        s1 = self.enc1(s0)   # (B,16,H/2,W/2)
        s2 = self.enc2(s1)   # (B,24,H/4,W/4)
        s3 = self.enc3(s2)   # (B,32,H/8,W/8)
        s4 = self.enc4(s3)   # (B,96,H/16,W/16)
        b  = self.enc5(s4)   # (B,320,H/32,W/32)

        # Decoder
        x = self.dec4(b,  s4)
        x = self.dec3(x,  s3)
        x = self.dec2(x,  s2)
        x = self.dec1(x,  s1)
        x = self.dec0(x,  s0)

        # Upsample to original size
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.head(x)


# ──────────────────────────────────────────────────────────────
# Smoke-test
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = MattingUNet(pretrained=False)
    dummy = torch.randn(2, 3, 256, 256)
    out   = model(dummy)
    print(f"Output shape : {out.shape}")   # (2, 1, 256, 256)
    assert out.shape == (2, 1, 256, 256)
    print("MattingUNet OK")
