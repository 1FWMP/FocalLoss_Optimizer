"""
HAM10000 Custom Dataset.

- HAM10000_metadata.csv 의 (image_id, dx) 매핑을 사용하여 PyTorch Dataset 을 구성.
- 7개 클래스(dx)는 알파벳 순서로 정렬해 라벨 인코딩(0~6) 한다.
    akiec, bcc, bkl, df, mel, nv, vasc
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

# 공식적으로 사용되는 7개 진단 클래스 (정렬 고정).
# 모델/실험 간 일관성을 위해 코드 전반에서 이 순서를 사용한다.
HAM10000_CLASSES: Tuple[str, ...] = (
    "akiec",  # Actinic keratoses / Bowen's disease
    "bcc",    # Basal cell carcinoma
    "bkl",    # Benign keratosis-like lesions
    "df",     # Dermatofibroma
    "mel",    # Melanoma
    "nv",     # Melanocytic nevi (다수 클래스)
    "vasc",   # Vascular lesions
)
NUM_CLASSES: int = len(HAM10000_CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(HAM10000_CLASSES)}


def _resolve_image_path(
    image_id: str, image_roots: Sequence[Path]
) -> Optional[Path]:
    """주어진 image_id 에 해당하는 .jpg 파일을 여러 후보 디렉토리에서 탐색한다."""
    for root in image_roots:
        candidate = root / f"{image_id}.jpg"
        if candidate.exists():
            return candidate
    return None


class HAM10000Dataset(Dataset):
    """
    Args:
        df: 사전에 train/valid 로 분할된 metadata DataFrame.
        image_roots: 이미지가 들어 있는 폴더 후보 리스트.
            (download_data.py 가 만든 images/ 단일 폴더이거나
             원본 HAM10000_images_part_1/2 폴더들이 모두 가능)
        transform: torchvision transform.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_roots: Sequence[Path],
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.transform = transform
        self.image_roots = [Path(r) for r in image_roots]

        # 이미지 경로를 미리 캐싱하여 학습 중 디스크 탐색 비용을 줄인다.
        records: List[Tuple[Path, int]] = []
        missing: List[str] = []
        for _, row in df.iterrows():
            image_id = str(row["image_id"])
            label = CLASS_TO_IDX[str(row["dx"])]
            path = _resolve_image_path(image_id, self.image_roots)
            if path is None:
                missing.append(image_id)
                continue
            records.append((path, label))

        if missing:
            print(
                f"[WARN] 이미지 파일 누락 {len(missing)}건 (예: {missing[:3]}). "
                f"다운로드 또는 image_roots 경로를 확인하세요."
            )

        if not records:
            raise RuntimeError(
                "유효한 이미지가 하나도 없습니다. data_dir 구조를 확인하세요."
            )

        self.samples: List[Tuple[Path, int]] = records
        self.labels: np.ndarray = np.array([lbl for _, lbl in records], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    # ------------------------------------------------------------------
    # 보조 메서드
    # ------------------------------------------------------------------
    def class_counts(self) -> np.ndarray:
        """클래스별 샘플 수 (Class-Balanced Loss 계산에 사용)."""
        counts = np.zeros(NUM_CLASSES, dtype=np.int64)
        for label in self.labels:
            counts[label] += 1
        return counts


def discover_image_roots(data_dir: Path) -> List[Path]:
    """data_dir 하위에서 가능한 이미지 폴더를 모두 수집해서 반환."""
    candidates = [
        data_dir / "images",
        data_dir / "HAM10000_images_part_1",
        data_dir / "HAM10000_images_part_2",
        data_dir / "ham10000_images",
        data_dir,  # 메타와 같은 위치에 jpg가 있는 경우
    ]
    roots = [p for p in candidates if p.exists() and p.is_dir()]
    # 중복 제거 (data_dir 자체가 들어가는 경우 등)
    seen, unique = set(), []
    for r in roots:
        rp = r.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(r)
    return unique


def load_metadata(data_dir: Path) -> pd.DataFrame:
    """HAM10000_metadata.csv 로드 + 필요한 컬럼만 반환."""
    csv_path = data_dir / "HAM10000_metadata.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"HAM10000_metadata.csv 를 찾지 못했습니다: {csv_path}\n"
            f"먼저 `python download_data.py --data_dir {data_dir}` 를 실행하세요."
        )
    df = pd.read_csv(csv_path)
    needed = {"image_id", "dx"}
    if not needed.issubset(df.columns):
        raise ValueError(f"메타데이터에 필요한 컬럼이 없습니다: {needed}")

    # 같은 lesion_id를 공유하는 이미지가 train/val 양쪽에 들어가면 leakage 가 생긴다.
    # main.py 의 split 단계에서 lesion 단위 stratified split 을 수행한다.
    return df[["lesion_id", "image_id", "dx"]].copy() if "lesion_id" in df.columns \
        else df[["image_id", "dx"]].copy()
