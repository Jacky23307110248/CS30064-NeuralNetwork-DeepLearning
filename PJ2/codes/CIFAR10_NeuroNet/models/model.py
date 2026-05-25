"""CIFAR ResNet with BatchNorm and residual blocks."""
from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn

from config import resolve_channels

DEFAULT_CHANNELS = (64, 128, 256, 512)


def _get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.1, inplace=True)
    raise ValueError(f"Unknown activation: {name}")


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int, activation: str):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act = _get_activation(activation)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.act(out)
        return out


class CifarResNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        channels: Sequence[int] = DEFAULT_CHANNELS,
        blocks_per_stage: Sequence[int] = (2, 2, 2, 2),
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_planes = channels[0]
        self.activation_name = activation
        self.dropout_p = dropout

        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.act = _get_activation(activation)

        self.layer1 = self._make_layer(channels[0], blocks_per_stage[0], stride=1)
        self.layer2 = self._make_layer(channels[1], blocks_per_stage[1], stride=2)
        self.layer3 = self._make_layer(channels[2], blocks_per_stage[2], stride=2)
        self.layer4 = self._make_layer(channels[3], blocks_per_stage[3], stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(channels[3], num_classes)

        self._init_weights()

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s, self.activation_name))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def build_model(cfg: dict) -> CifarResNet:
    channels = resolve_channels(cfg.get("channels"), cfg.get("width_mult"))
    blocks = cfg.get("blocks_per_stage", (2, 2, 2, 2))
    if isinstance(blocks, list):
        blocks = tuple(blocks)
    model = CifarResNet(
        num_classes=10,
        channels=channels,
        blocks_per_stage=blocks,
        activation=cfg.get("activation", "relu"),
        dropout=float(cfg.get("dropout", 0.0)),
    )
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
