"""
Augmentation transforms (Optuna 탐색용).

이 모듈은 main.py 의 기존 플래그 기반 build_transforms(args) 와는 별개로,
Optuna objective 함수가 trial 마다 넘겨주는 **연속/이산 파라미터 딕셔너리**
(`aug_params`) 를 받아 동적으로 transform 파이프라인을 조립한다.

지원하는 8가지 증강 기법
  1. Crop         : RandomResizedCrop (scale 하한을 파라미터로)
  2. Cutout       : RandomErasing (적용 확률 + 영역 크기)
  3. ColorJitter  : brightness/contrast/saturation/hue (strength 파라미터)
  4. Sobel        : 커스텀 엣지 추출 필터 (적용 확률)              ← Custom
  5. Noise        : 커스텀 Gaussian Noise (std 범위)              ← Custom
  6. Blur         : torchvision GaussianBlur (적용 확률 + sigma)
  7. Rotate       : RandomRotation (각도 파라미터)
  8. Average Blur : 커스텀 평균(박스) 블러 (kernel_size 동적)      ← Custom

커스텀 transform 3종(Sobel / GaussianNoise / AverageBlur)은 입력이 PIL Image 든
Tensor 든 모두 동작하도록 내부에서 적절히 변환하고, **입력과 동일한 타입으로 반환**한다.
"""
from __future__ import annotations

import random
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F_nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# ImageNet 정규화 상수 (timm pretrained 기준). main.py 와 공유.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# 기본 입력 해상도 (기존 파이프라인과 동일: train crop 300, val resize 320→crop 300)
DEFAULT_IMAGE_SIZE = 300

# 통제 실험(단독 튜닝 / 조합 히트맵)에서 사용하는 8가지 증강 기법의 정식 이름.
# 히트맵 축 순서이기도 하다. (어디서도 순서를 바꾸지 말 것)
AUG_NAMES = (
    "crop",         # RandomResizedCrop (scale 하한)
    "rotate",       # RandomRotation (각도)
    "colorjitter",  # ColorJitter (강도)
    "blur",         # GaussianBlur (sigma)
    "avgblur",      # Average(box) Blur (kernel)
    "sobel",        # Sobel 엣지 (확률)
    "noise",        # Gaussian Noise (std)
    "cutout",       # RandomErasing (영역 크기)
)


# ===========================================================================
# 커스텀 Transform 3종
# ===========================================================================
def _to_tensor_keep_type(x: Any) -> Tuple[torch.Tensor, bool]:
    """입력을 (C,H,W) float tensor 로 변환하고, 원래 PIL 이었는지 여부를 함께 반환."""
    if torch.is_tensor(x):
        return x, False
    return TF.to_tensor(x), True


def _restore_type(t: torch.Tensor, was_pil: bool) -> Any:
    """텐서를 (필요하면) PIL 로 되돌린다. 값 범위는 [0,1] 로 clamp."""
    t = t.clamp(0.0, 1.0)
    if was_pil:
        return TF.to_pil_image(t)
    return t


class SobelFilter:
    """Sobel 엣지 추출 필터.

    확률 `p` 로 적용되며, 적용 시 이미지를 그레이스케일 엣지 강도 맵(3채널 복제)으로
    치환한다. 색/텍스처 정보를 잃으므로 augmentation 으로 쓸 때는 확률을 낮게 두는 것을
    권장한다.

    입력: PIL Image 또는 (C,H,W) float Tensor. 출력: 입력과 동일 타입.
    """

    def __init__(self, p: float = 0.5) -> None:
        self.p = float(p)

    def __call__(self, x: Any) -> Any:
        if self.p <= 0.0 or random.random() >= self.p:
            return x

        t, was_pil = _to_tensor_keep_type(x)
        # 그레이스케일 (1,H,W)
        gray = t.mean(dim=0, keepdim=True)
        gray = gray.unsqueeze(0)  # (1,1,H,W)

        kx = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=t.dtype,
        ).view(1, 1, 3, 3)
        ky = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=t.dtype,
        ).view(1, 1, 3, 3)

        gx = F_nn.conv2d(gray, kx, padding=1)
        gy = F_nn.conv2d(gray, ky, padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-12).squeeze(0)  # (1,H,W)
        mag = mag / (mag.amax() + 1e-8)  # 0~1 정규화

        edge = mag.repeat(3, 1, 1)  # (3,H,W)
        return _restore_type(edge, was_pil)


class GaussianNoise:
    """지정된 표준편차(std) 의 정규분포 노이즈를 더한다.

    std 가 0 이하이면 no-op. 확률 `p` 로 적용.
    입력: PIL Image 또는 (C,H,W) float Tensor. 출력: 입력과 동일 타입.
    """

    def __init__(self, std: float = 0.05, p: float = 1.0) -> None:
        self.std = float(std)
        self.p = float(p)

    def __call__(self, x: Any) -> Any:
        if self.std <= 0.0 or self.p <= 0.0 or random.random() >= self.p:
            return x

        t, was_pil = _to_tensor_keep_type(x)
        noise = torch.randn_like(t) * self.std
        return _restore_type(t + noise, was_pil)


class AverageBlur:
    """커널 크기를 동적으로 조절하는 평균(박스) 블러.

    kernel_size 는 홀수로 강제되며, 1 이하이면 no-op. 확률 `p` 로 적용.
    depthwise conv2d (groups=C) 로 채널별 평균 블러를 적용한다.
    입력: PIL Image 또는 (C,H,W) float Tensor. 출력: 입력과 동일 타입.
    """

    def __init__(self, kernel_size: int = 3, p: float = 0.5) -> None:
        k = max(1, int(kernel_size))
        if k % 2 == 0:  # 홀수 보장
            k += 1
        self.kernel_size = k
        self.p = float(p)

    def __call__(self, x: Any) -> Any:
        if self.kernel_size <= 1 or self.p <= 0.0 or random.random() >= self.p:
            return x

        t, was_pil = _to_tensor_keep_type(x)
        c, _, _ = t.shape
        k = self.kernel_size
        weight = torch.full(
            (c, 1, k, k), 1.0 / float(k * k), dtype=t.dtype
        )
        out = F_nn.conv2d(t.unsqueeze(0), weight, padding=k // 2, groups=c).squeeze(0)
        return _restore_type(out, was_pil)


# ===========================================================================
# 동적 파이프라인 빌더
# ===========================================================================
def build_transforms(
    aug_params: Dict[str, Any],
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> Tuple[T.Compose, T.Compose]:
    """`aug_params` 딕셔너리로 (train_tf, val_tf) 를 동적으로 조립.

    누락된 키는 안전한 기본값(거의 no-op)으로 처리하므로, 일부 파라미터만 넘겨도 된다.

    인식하는 키
        crop_scale_min       (float) RandomResizedCrop scale 하한          [0,1]
        rotate_deg           (float) RandomRotation 각도 (±deg)
        colorjitter_strength (float) ColorJitter b/c/s 강도 (hue=strength*0.5)
        sobel_prob           (float) Sobel 적용 확률
        noise_std            (float) Gaussian Noise 표준편차
        noise_prob           (float) Gaussian Noise 적용 확률
        blur_prob            (float) GaussianBlur 적용 확률
        blur_sigma           (float) GaussianBlur sigma
        avgblur_prob         (float) Average Blur 적용 확률
        avgblur_kernel       (int)   Average Blur 커널 크기 (홀수로 강제)
        cutout_prob          (float) RandomErasing 적용 확률
        cutout_scale         (float) RandomErasing 영역 상한 (area 비율)

    파이프라인 순서
        [PIL] RandomResizedCrop → H/V Flip → Rotate → ColorJitter → GaussianBlur
        [ToTensor]
        [Tensor] AverageBlur → Sobel → GaussianNoise
        [Normalize]
        [Tensor] Cutout(RandomErasing)
    """
    p = aug_params

    crop_scale_min = float(p.get("crop_scale_min", 0.8))
    crop_scale_min = min(max(crop_scale_min, 0.05), 1.0)

    rotate_deg = float(p.get("rotate_deg", 0.0))
    cj = float(p.get("colorjitter_strength", 0.0))

    sobel_prob = float(p.get("sobel_prob", 0.0))

    noise_std = float(p.get("noise_std", 0.0))
    noise_prob = float(p.get("noise_prob", 1.0))

    blur_prob = float(p.get("blur_prob", 0.0))
    blur_sigma = float(p.get("blur_sigma", 1.0))

    avgblur_prob = float(p.get("avgblur_prob", 0.0))
    avgblur_kernel = int(p.get("avgblur_kernel", 3))

    cutout_prob = float(p.get("cutout_prob", 0.0))
    cutout_scale = float(p.get("cutout_scale", 0.1))

    # ---- PIL 단계 ----------------------------------------------------------
    pil_stage = [
        T.RandomResizedCrop(size=image_size, scale=(crop_scale_min, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
    ]
    if rotate_deg > 0.0:
        pil_stage.append(T.RandomRotation(degrees=rotate_deg))
    if cj > 0.0:
        pil_stage.append(
            T.ColorJitter(
                brightness=cj,
                contrast=cj,
                saturation=cj,
                hue=min(cj * 0.5, 0.5),
            )
        )
    if blur_prob > 0.0:
        sigma = max(blur_sigma, 0.1)
        pil_stage.append(
            T.RandomApply(
                [T.GaussianBlur(kernel_size=5, sigma=(sigma, sigma))],
                p=min(blur_prob, 1.0),
            )
        )

    # ---- Tensor 단계 (Normalize 이전) --------------------------------------
    tensor_stage = []
    if avgblur_prob > 0.0:
        tensor_stage.append(AverageBlur(kernel_size=avgblur_kernel, p=avgblur_prob))
    if sobel_prob > 0.0:
        tensor_stage.append(SobelFilter(p=sobel_prob))
    if noise_std > 0.0:
        tensor_stage.append(GaussianNoise(std=noise_std, p=noise_prob))

    # ---- Cutout (Normalize 이후) -------------------------------------------
    erasing_stage = []
    if cutout_prob > 0.0:
        erasing_stage.append(
            T.RandomErasing(
                p=min(cutout_prob, 1.0),
                scale=(0.02, max(cutout_scale, 0.02)),
                value=0,
            )
        )

    train_tf = T.Compose(
        [
            *pil_stage,
            T.ToTensor(),
            *tensor_stage,
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            *erasing_stage,
        ]
    )

    val_tf = T.Compose(
        [
            T.Resize(int(round(image_size * 320 / 300))),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_tf, val_tf


def _val_transform(image_size: int = DEFAULT_IMAGE_SIZE) -> T.Compose:
    """검증용 deterministic transform (Resize→CenterCrop→ToTensor→Normalize)."""
    return T.Compose(
        [
            T.Resize(int(round(image_size * 320 / 300))),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_controlled_transforms(
    active_augs,
    aug_params: Dict[str, Any],
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> Tuple[T.Compose, T.Compose]:
    """통제 실험(단독 튜닝 / 조합 히트맵)용 transform 빌더.

    `active_augs` 에 명시된 기법만 적용한다. baseline(아무 기법도 없음) 은
    `Resize→CenterCrop` (무증강, flip 도 제외) 이며, 여기에 active 기법을 얹는다.
    각 기법은 **항상 적용(prob=1.0)** 하고 강도/파라미터는 `aug_params` 에서 읽는다.
    예외) sobel 은 강도 개념이 없어 적용 확률(`sobel_prob`) 자체가 파라미터다.

    Args:
        active_augs: 적용할 기법 이름들의 컬렉션 (AUG_NAMES 의 부분집합).
                     비어 있으면 baseline(무증강).
        aug_params : 기법별 파라미터 dict. 인식 키는 build_transforms 와 동일.
        image_size : 입력 해상도.

    Returns:
        (train_tf, val_tf)
    """
    active = {str(a).strip() for a in active_augs if str(a).strip()}
    unknown = active - set(AUG_NAMES)
    if unknown:
        raise ValueError(f"알 수 없는 augmentation: {sorted(unknown)}. 가능: {list(AUG_NAMES)}")

    p = aug_params or {}

    # ---- PIL 단계 ----------------------------------------------------------
    # crop 이 active 면 RandomResizedCrop, 아니면 deterministic Resize+CenterCrop.
    if "crop" in active:
        crop_scale_min = float(p.get("crop_scale_min", 0.8))
        crop_scale_min = min(max(crop_scale_min, 0.05), 1.0)
        pil_stage = [T.RandomResizedCrop(size=image_size, scale=(crop_scale_min, 1.0))]
    else:
        pil_stage = [
            T.Resize(int(round(image_size * 320 / 300))),
            T.CenterCrop(image_size),
        ]

    if "rotate" in active:
        rotate_deg = float(p.get("rotate_deg", 0.0))
        if rotate_deg > 0.0:
            pil_stage.append(T.RandomRotation(degrees=rotate_deg))

    if "colorjitter" in active:
        cj = float(p.get("colorjitter_strength", 0.0))
        if cj > 0.0:
            pil_stage.append(
                T.ColorJitter(
                    brightness=cj, contrast=cj, saturation=cj, hue=min(cj * 0.5, 0.5)
                )
            )

    if "blur" in active:
        sigma = max(float(p.get("blur_sigma", 1.0)), 0.1)
        pil_stage.append(T.GaussianBlur(kernel_size=5, sigma=(sigma, sigma)))  # 항상 적용

    # ---- Tensor 단계 (Normalize 이전) --------------------------------------
    tensor_stage = []
    if "avgblur" in active:
        tensor_stage.append(
            AverageBlur(kernel_size=int(p.get("avgblur_kernel", 3)), p=1.0)
        )
    if "sobel" in active:
        sp = float(p.get("sobel_prob", 0.0))
        if sp > 0.0:
            tensor_stage.append(SobelFilter(p=sp))  # sobel 은 확률이 곧 파라미터
    if "noise" in active:
        std = float(p.get("noise_std", 0.0))
        if std > 0.0:
            tensor_stage.append(GaussianNoise(std=std, p=1.0))

    # ---- Cutout (Normalize 이후) -------------------------------------------
    erasing_stage = []
    if "cutout" in active:
        cs = max(float(p.get("cutout_scale", 0.1)), 0.02)
        erasing_stage.append(T.RandomErasing(p=1.0, scale=(0.02, cs), value=0))

    train_tf = T.Compose(
        [
            *pil_stage,
            T.ToTensor(),
            *tensor_stage,
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            *erasing_stage,
        ]
    )
    return train_tf, _val_transform(image_size)
