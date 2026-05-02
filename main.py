"""
HAM10000 7-class Skin Lesion Classification.

- Backbone : EfficientNet-B3 (timm, ImageNet pretrained)
- Loss     : Cross-Entropy  vs  Class-Balanced Focal Loss (Cui et al., 2019)
- Metric   : Macro F1-score (스 클래스 불균형이 심하므로 accuracy 대신 사용)

실행 예:
    # 1) 데이터 다운로드
    python download_data.py --data_dir ./data

    # 2) Cross-Entropy 베이스라인
    python main.py --data_dir ./data --loss_type ce

    # 3) Class-Balanced Focal Loss
    python main.py --data_dir ./data --loss_type cb_focal --beta 0.999 --gamma 2.0
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import (
    HAM10000_CLASSES,
    HAM10000Dataset,
    NUM_CLASSES,
    discover_image_roots,
    load_metadata,
)
from losses import build_loss
from model import SUPPORTED_BACKBONES, build_model


# ---------------------------------------------------------------------------
# 1. CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HAM10000 - Focal Loss vs Cross-Entropy 비교 실험"
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="HAM10000_metadata.csv 와 이미지가 있는 루트 디렉토리")
    parser.add_argument("--loss_type", type=str, default="ce",
                        choices=["ce", "cb_focal"],
                        help="사용할 손실 함수")
    parser.add_argument("--backbone", type=str, default="efficientnet_b3",
                        choices=list(SUPPORTED_BACKBONES),
                        help="사용할 backbone 모델")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.999,
                        help="Class-Balanced Loss 의 beta")
    parser.add_argument("--gamma", type=float, default=2.0,
                        help="Focal Loss 의 gamma")

    # 학습 환경 보조 인자 (선택)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--save_name", type=str, default=None,
                        help="저장할 모델 파일명 (미지정 시 best_model_{backbone}_{loss_type}.pth)")
    parser.add_argument("--no_pretrained", action="store_true",
                        help="ImageNet 사전학습 가중치 미사용 (디버깅용)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 2. Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN 결정성은 속도와의 trade-off. 실험 비교에서는 deterministic 우선.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# 3. Transforms / Dataloader
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """학습/검증용 augmentation 정의."""
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(size=300),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(320),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_tf, val_tf


def split_train_val(
    df: pd.DataFrame, val_size: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """클래스 비율을 유지(stratify) 하면서 train/val 로 분할.

    HAM10000 은 같은 lesion 의 여러 장 사진이 존재하므로, 가능하다면 lesion_id
    단위로 분할해 leakage 를 방지한다.
    """
    if "lesion_id" in df.columns:
        # lesion 단위로 묶고, 그 lesion 의 대표 클래스(최빈)를 기준으로 stratified split.
        lesion_df = (
            df.groupby("lesion_id")["dx"]
              .agg(lambda s: s.value_counts().index[0])
              .reset_index()
        )
        train_lesions, val_lesions = train_test_split(
            lesion_df["lesion_id"].values,
            test_size=val_size,
            stratify=lesion_df["dx"].values,
            random_state=seed,
        )
        train_df = df[df["lesion_id"].isin(train_lesions)].reset_index(drop=True)
        val_df = df[df["lesion_id"].isin(val_lesions)].reset_index(drop=True)
    else:
        train_df, val_df = train_test_split(
            df, test_size=val_size, stratify=df["dx"].values, random_state=seed
        )
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

    return train_df, val_df


# ---------------------------------------------------------------------------
# 4. Train / Validate
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss, total_n = 0.0, 0

    pbar = tqdm(loader, desc=f"[Train E{epoch}]", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_n += bs
        pbar.set_postfix(loss=f"{total_loss / total_n:.4f}")

    return total_loss / max(total_n, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float, str]:
    """반환: (avg_loss, macro_f1, classification_report_str)."""
    model.eval()
    total_loss, total_n = 0.0, 0
    all_preds, all_targets = [], []

    pbar = tqdm(loader, desc=f"[Valid E{epoch}]", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_n += bs

        preds = logits.argmax(dim=1)
        all_preds.append(preds.detach().cpu().numpy())
        all_targets.append(labels.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    macro_f1 = f1_score(
        all_targets, all_preds,
        labels=list(range(NUM_CLASSES)),
        average="macro",
        zero_division=0,
    )
    report = classification_report(
        all_targets, all_preds,
        labels=list(range(NUM_CLASSES)),
        target_names=list(HAM10000_CLASSES),
        digits=4,
        zero_division=0,
    )
    return total_loss / max(total_n, 1), float(macro_f1), report


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device                : {device}")
    print(f"[INFO] backbone              : {args.backbone}")
    print(f"[INFO] loss_type             : {args.loss_type}")
    print(f"[INFO] epochs / bs / lr      : {args.epochs} / {args.batch_size} / {args.lr}")
    if args.loss_type == "cb_focal":
        print(f"[INFO] beta / gamma          : {args.beta} / {args.gamma}")

    # ---- 5.1 Metadata & split -------------------------------------------------
    df = load_metadata(data_dir)
    train_df, val_df = split_train_val(df, args.val_size, args.seed)
    print(f"[INFO] train / val samples   : {len(train_df)} / {len(val_df)}")

    image_roots = discover_image_roots(data_dir)
    if not image_roots:
        raise FileNotFoundError(
            f"이미지 폴더를 찾지 못했습니다: {data_dir}\n"
            f"먼저 download_data.py 로 데이터를 받아주세요."
        )
    print(f"[INFO] image roots           : {[str(p) for p in image_roots]}")

    # ---- 5.2 Datasets / Loaders ----------------------------------------------
    train_tf, val_tf = build_transforms()
    train_set = HAM10000Dataset(train_df, image_roots, transform=train_tf)
    val_set = HAM10000Dataset(val_df, image_roots, transform=val_tf)

    counts = train_set.class_counts()
    print("[INFO] train class distribution:")
    for cls, n in zip(HAM10000_CLASSES, counts):
        print(f"    {cls:>6s} : {n}")

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    # ---- 5.3 Model / Loss / Optimizer ----------------------------------------
    model = build_model(
        backbone=args.backbone, num_classes=NUM_CLASSES, pretrained=not args.no_pretrained
    ).to(device)

    criterion = build_loss(
        loss_type=args.loss_type,
        samples_per_cls=counts.tolist(),
        num_classes=NUM_CLASSES,
        beta=args.beta,
        gamma=args.gamma,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # ---- 5.4 Training loop ----------------------------------------------------
    best_f1 = -1.0
    save_name = args.save_name or f"best_model_{args.backbone}_{args.loss_type}.pth"
    best_path = output_dir / save_name
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_f1, report = validate(
            model, val_loader, criterion, device, epoch
        )
        scheduler.step()
        dt = time.time() - t0

        print(
            f"\n[Epoch {epoch:03d}/{args.epochs}] "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_macroF1={val_f1:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}  "
            f"time={dt:.1f}s"
        )
        print(report)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_f1": val_f1,
            "lr": optimizer.param_groups[0]["lr"],
        })

        # Best 모델 저장 (Macro F1 기준)
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_macro_f1": val_f1,
                    "args": vars(args),
                    "classes": list(HAM10000_CLASSES),
                },
                best_path,
            )
            print(f"  >> Best updated! Macro F1 = {best_f1:.4f}  saved -> {best_path}")

    # ---- 5.5 학습 로그 저장 ---------------------------------------------------
    log_path = output_dir / f"history_{args.backbone}_{args.loss_type}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            {"best_macro_f1": best_f1, "history": history, "args": vars(args)},
            f, indent=2, ensure_ascii=False,
        )
    print(f"\n[DONE] Best Macro F1 = {best_f1:.4f}")
    print(f"[DONE] Best weights : {best_path}")
    print(f"[DONE] Train history: {log_path}")


if __name__ == "__main__":
    main()
