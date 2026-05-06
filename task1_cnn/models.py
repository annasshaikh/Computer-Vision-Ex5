"""
models.py — CNN architectures for the Seeds classification task.

Model A — BaselineCNN
    3 conv blocks (Conv2d → BN → ReLU → MaxPool)
    filter progression 32→64→128
    Global Average Pooling → Dense → Softmax
    ≤1.5M parameters

Model B — DeepRegCNN
    4+ conv blocks with dropout (0.2–0.5)
    L2 weight-decay is passed to the optimizer (not here)
    Optional residual connection on the 3rd / 4th block
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────
# Shared helper
# ──────────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    """Conv2d → BatchNorm2d → ReLU → MaxPool2d."""

    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size: int = 3, pool: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# ──────────────────────────────────────────────────────────────
# Residual wrapper (used in Model B)
# ──────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    """
    Two conv layers with a skip connection.
    Spatial size stays the same (no pooling inside the block).
    """

    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.drop  = nn.Dropout2d(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return F.relu(out + x, inplace=True)


# ──────────────────────────────────────────────────────────────
# Model A — Baseline CNN
# ──────────────────────────────────────────────────────────────
class BaselineCNN(nn.Module):
    """
    3 conv blocks, filter progression 32→64→128.
    Global Average Pooling → Linear(128, num_classes).
    """

    def __init__(self, num_outputs: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3,   32),   # 128→64
            ConvBlock(32,  64),   # 64→32
            ConvBlock(64, 128),   # 32→16
        )
        self.gap       = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, num_outputs)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────────
# Model B — Deeper / Regularised CNN
# ──────────────────────────────────────────────────────────────
class DeepRegCNN(nn.Module):
    """
    4 conv blocks with dropout, optional residual, L2 via optimizer.
    filter progression 32→64→128→256.
    """

    def __init__(self, num_outputs: int = 1,
                 dropout: float = 0.3,
                 residual: bool = True):
        super().__init__()
        self.residual = residual

        # Block 1 & 2
        self.block1 = ConvBlock(3,   32,  dropout=0.0)   # 128→64
        self.block2 = ConvBlock(32,  64,  dropout=dropout * 0.5)  # 64→32

        # Optional residual at 64-ch
        self.res64  = ResBlock(64, dropout=dropout * 0.5) if residual else nn.Identity()

        # Block 3 & 4
        self.block3 = ConvBlock(64,  128, dropout=dropout)  # 32→16
        self.block4 = ConvBlock(128, 256, dropout=dropout)  # 16→8

        # Optional residual at 256-ch
        self.res256 = ResBlock(256, dropout=dropout) if residual else nn.Identity()

        self.gap        = nn.AdaptiveAvgPool2d(1)
        self.dropout_fc = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(256, num_outputs)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.res64(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.res256(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dropout_fc(x)
        return self.classifier(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────
def build_model(cfg: dict, model_key: str = "model_a",
                 num_outputs: int = 1) -> nn.Module:
    mcfg = cfg[model_key]
    name = mcfg["name"]
    if name == "BaselineCNN":
        return BaselineCNN(num_outputs)
    elif name == "DeepRegCNN":
        return DeepRegCNN(num_outputs,
                          dropout=mcfg.get("dropout", 0.3),
                          residual=mcfg.get("residual", True))
    else:
        raise ValueError(f"Unknown model name: {name}")


# ──────────────────────────────────────────────────────────────
# Smoke-test
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for Model, label in [(BaselineCNN, "Model A"), (DeepRegCNN, "Model B")]:
        m = Model(num_classes=150)
        dummy = torch.randn(2, 3, 128, 128)
        out   = m(dummy)
        params = m.count_parameters()
        print(f"{label:10s} | output {out.shape} | params {params:,}")
        if label == "Model A":
            assert params <= 1_500_000, f"Model A exceeds 1.5M params! ({params:,})"
