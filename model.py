"""
Backbone model builder.

`timm` 의 다양한 backbone 을 불러오고,
마지막 classifier 를 HAM10000 의 7개 클래스에 맞게 교체한다.
"""
from __future__ import annotations

import timm
import torch.nn as nn

SUPPORTED_BACKBONES: dict[str, str] = {
    "efficientnet_b3":       "efficientnet_b3",
    "resnet50":              "resnet50",
    "resnet101":             "resnet101",
    "resnet152":             "resnet152",
    "densenet121":           "densenet121",
    "mobilenetv3_large_100": "mobilenetv3_large_100",
}


def build_model(
    backbone: str = "efficientnet_b3",
    num_classes: int = 7,
    pretrained: bool = True,
) -> nn.Module:
    """timm 기반 backbone 빌더. ImageNet-1k pretrained 가중치 사용.

    Args:
        backbone   : SUPPORTED_BACKBONES 의 키 중 하나.
        num_classes: 출력 클래스 수 (HAM10000: 7)
        pretrained : ImageNet 사전학습 가중치 로딩 여부
    """
    if backbone not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"지원하지 않는 backbone: '{backbone}'. "
            f"가능한 옵션: {list(SUPPORTED_BACKBONES)}"
        )
    return timm.create_model(
        SUPPORTED_BACKBONES[backbone],
        pretrained=pretrained,
        num_classes=num_classes,
    )


def build_efficientnet_b3(num_classes: int = 7, pretrained: bool = True) -> nn.Module:
    """하위 호환성 유지용 래퍼."""
    return build_model("efficientnet_b3", num_classes=num_classes, pretrained=pretrained)
