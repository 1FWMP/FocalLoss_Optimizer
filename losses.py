"""
Loss functions.

- CrossEntropyLoss 는 nn.CrossEntropyLoss 를 그대로 사용한다.
- CBFocalLoss : Cui et al., "Class-Balanced Loss Based on Effective Number of Samples", CVPR 2019.
    - alpha (per-class weight) = 1 / E_n   where  E_n = (1 - beta^n_c) / (1 - beta)
    - 안정성을 위해 weight 는 sum=num_classes 로 정규화 (논문 official impl 과 동일)
    - softmax + (1 - p_t)^gamma * log(p_t) 형태의 multi-class focal loss 사용
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CBFocalLoss(nn.Module):
    """Class-Balanced Focal Loss.

    Args:
        samples_per_cls : 학습 데이터의 각 클래스별 샘플 수. 길이 = num_classes.
        num_classes     : 클래스 수.
        beta            : Effective number 의 hyper-parameter (0 ~ 1). 1에 가까울수록
                          가중치가 클래스 빈도에 더 민감해진다. (논문 권장 0.999)
        gamma           : Focal Loss 의 focusing parameter. 0이면 weighted CE 와 동일.
        eps             : log(0) 방지를 위한 작은 값.
    """

    def __init__(
        self,
        samples_per_cls: Sequence[int],
        num_classes: int,
        beta: float = 0.999,
        gamma: float = 2.0,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        if len(samples_per_cls) != num_classes:
            raise ValueError(
                f"samples_per_cls 길이({len(samples_per_cls)}) != num_classes({num_classes})"
            )

        self.num_classes = num_classes
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.eps = float(eps)

        # ----- Effective Number of Samples 기반 클래스 가중치 -----
        # 빈 클래스(0개) 처리를 위해 최소 1로 클램프.
        spc = np.asarray(samples_per_cls, dtype=np.float64)
        spc = np.maximum(spc, 1.0)

        effective_num = 1.0 - np.power(self.beta, spc)
        weights = (1.0 - self.beta) / effective_num          # = 1 / E_n  (Eq.4)
        # 가중치 합이 num_classes 가 되도록 정규화 (official impl 컨벤션)
        weights = weights / weights.sum() * num_classes

        # buffer 로 등록하면 .to(device) / state_dict 에 자연스럽게 따라감.
        self.register_buffer(
            "class_weights",
            torch.tensor(weights, dtype=torch.float32),
        )

    # --------------------------------------------------------------
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : [B, C] (raw, softmax 적용 전)
            targets : [B]    (long, 0..C-1)
        """
        if logits.dim() != 2:
            raise ValueError(f"logits 는 [B, C] 여야 합니다. got {tuple(logits.shape)}")
        batch_size, num_classes = logits.shape
        if num_classes != self.num_classes:
            raise ValueError(
                f"logits 의 클래스 수({num_classes}) != 설정된 num_classes({self.num_classes})"
            )

        # one-hot 라벨
        targets_onehot = F.one_hot(targets, num_classes=self.num_classes).float()

        # 수치 안정성을 위해 log_softmax 와 softmax 를 함께 사용:
        #   log p = log_softmax(logits)
        #   p     = exp(log p)         (이미 [eps, 1] 범위로 안전하게 만들기 위해 clamp)
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp().clamp(min=self.eps, max=1.0 - self.eps)

        # Focal modulation: (1 - p_t)^gamma  (정답 클래스에서만 의미)
        focal_factor = torch.pow(1.0 - probs, self.gamma)

        # per-class alpha 를 [1, C] 로 broadcast.
        alpha = self.class_weights.unsqueeze(0).to(logits.device)  # [1, C]

        # CB-Focal Loss (정답 클래스 외에는 one-hot 으로 0 처리)
        # loss_{b,c} = - alpha_c * one_hot * (1 - p)^gamma * log p
        loss = -alpha * targets_onehot * focal_factor * log_probs

        # 클래스 차원 합 -> 샘플별 loss, 배치 평균.
        loss = loss.sum(dim=1).mean()
        return loss


def build_loss(
    loss_type: str,
    samples_per_cls: Sequence[int],
    num_classes: int,
    beta: float,
    gamma: float,
) -> nn.Module:
    """CLI 인자로 loss 를 선택하여 생성하는 팩토리."""
    if loss_type == "ce":
        return nn.CrossEntropyLoss()
    if loss_type == "cb_focal":
        return CBFocalLoss(
            samples_per_cls=samples_per_cls,
            num_classes=num_classes,
            beta=beta,
            gamma=gamma,
        )
    raise ValueError(f"지원하지 않는 loss_type: {loss_type}")
