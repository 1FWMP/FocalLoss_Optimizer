"""
Backbone model builder.

`timm` 의 EfficientNet-B3 (ImageNet 사전학습) 를 불러오고,
마지막 classifier 를 HAM10000 의 7개 클래스에 맞게 교체한다.
"""
from __future__ import annotations

import timm
import torch.nn as nn


def build_efficientnet_b3(num_classes: int = 7, pretrained: bool = True) -> nn.Module:
    """timm.create_model 의 num_classes 인자가 자동으로 분류기 헤드를 교체해 주므로
    별도의 `model.classifier = ...` 작업이 필요 없다.

    Args:
        num_classes : 출력 클래스 수 (HAM10000: 7)
        pretrained  : ImageNet 사전학습 가중치 로딩 여부
    """
    model = timm.create_model(
        "efficientnet_b3",
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model
