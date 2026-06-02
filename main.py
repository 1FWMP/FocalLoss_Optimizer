"""
HAM10000 7-class Skin Lesion Classification.

- Backbone : timm pretrained (기본 resnet50)
- Loss     : Cross-Entropy  vs  Class-Balanced Focal Loss (Cui et al., 2019)
- Metric   : Macro F1-score (클래스 불균형이 심하므로 accuracy 대신 사용)

두 가지 실행 모드
  1) 단일 학습 (기존 방식)
       python main.py --data_dir ./data --backbone resnet50 --loss_type ce
       python main.py --data_dir ./data --loss_type cb_focal --beta 0.999 --gamma 2.0
       # --use_cutmix / --use_elastic / --use_colorjitter 플래그로 증강 on/off

  2) Optuna 증강 탐색 (신규)
       python main.py --optuna --data_dir ./data --backbone resnet50 --loss_type ce \
           --n_trials 30 --storage sqlite:///optuna_study.db \
           --study_name resnet50_ce_aug_search --epochs 15
       # 8가지 증강 기법의 연속 파라미터 공간을 베이지안 최적화(TPE)+MedianPruner 로 탐색.
       # SQLite storage 를 쓰므로 중단 후 재실행하면 완료된 trial 은 건너뛰고 이어서 진행.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import List, Optional, Tuple

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
from transforms import (
    AUG_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_controlled_transforms,
    build_transforms as build_optuna_transforms,
)


# ---------------------------------------------------------------------------
# 1. CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HAM10000 - Focal Loss vs Cross-Entropy / Optuna 증강 탐색"
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="HAM10000_metadata.csv 와 이미지가 있는 루트 디렉토리")
    parser.add_argument("--loss_type", type=str, default="ce",
                        choices=["ce", "cb_focal"],
                        help="사용할 손실 함수")
    parser.add_argument("--backbone", type=str, default="resnet50",
                        choices=list(SUPPORTED_BACKBONES),
                        help="사용할 backbone 모델 (Optuna 탐색은 resnet50 고정 권장)")
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
                        help="저장할 모델 파일명 (미지정 시 best_model_{run_name}.pth)")
    parser.add_argument("--no_pretrained", action="store_true",
                        help="ImageNet 사전학습 가중치 미사용 (디버깅용)")
    parser.add_argument("--run_name", type=str, default=None,
                        help="실험 고유 이름 (미지정 시 {backbone}_{loss_type}). 모델/히스토리 파일 prefix 로 사용됨")

    # 기존 플래그 기반 증강 (단일 학습 모드 전용)
    parser.add_argument("--use_cutmix", action="store_true", help="CutMix augmentation 적용")
    parser.add_argument("--use_elastic", action="store_true", help="Elastic Transform 적용")
    parser.add_argument("--use_colorjitter", action="store_true", help="Color Jitter 적용")

    # Optuna 증강 탐색 모드 전용
    parser.add_argument("--optuna", action="store_true",
                        help="Optuna 증강 파라미터 탐색 모드로 실행")
    parser.add_argument("--n_trials", type=int, default=30,
                        help="Optuna 탐색 총 trial 횟수 (완료분 제외, 남은 만큼만 진행)")
    parser.add_argument("--storage", type=str, default="sqlite:///optuna_study.db",
                        help="Optuna storage (SQLite). 중단점 재개에 사용")
    parser.add_argument("--study_name", type=str, default="resnet50_ce_aug_search",
                        help="Optuna study 식별 이름 (joint 탐색용)")

    # 단독 튜닝 모드 (stage 1): 한 기법만 켜고 그 강도 파라미터만 Optuna 탐색
    parser.add_argument("--search_aug", type=str, default=None,
                        choices=list(AUG_NAMES),
                        help="지정 시 해당 augmentation 1개만 단독 Optuna 튜닝 (study=singleaug_{aug})")

    # 조합 실험 모드 (stage 2/3): 지정 기법들을 고정 파라미터로 켜고 1회 학습
    parser.add_argument("--combo_augs", type=str, default=None,
                        help="콤마구분 기법 목록(예: 'crop,rotate'). 지정 시 조합 학습 1회 실행. "
                             "빈 문자열/'base' 면 무증강 baseline")
    parser.add_argument("--aug_params_json", type=str, default=None,
                        help="조합 학습 시 사용할 기법별 고정 파라미터 JSON "
                             "(미지정 시 {output_dir}/optuna_best_per_aug.json)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 단독 튜닝 search space (stage 1): 기법별로 강도 파라미터 하나만 탐색.
#   각 기법은 항상 적용(prob=1.0)되며 여기서 정의한 파라미터만 변수.
#   예외) sobel 은 강도가 없어 적용 확률 자체를 탐색.
# ---------------------------------------------------------------------------
def suggest_aug_params(trial: "optuna.trial.Trial", aug: str) -> dict:  # noqa: F821
    if aug == "crop":
        return {"crop_scale_min": trial.suggest_float("crop_scale_min", 0.5, 1.0)}
    if aug == "rotate":
        return {"rotate_deg": trial.suggest_float("rotate_deg", 0.0, 180.0)}
    if aug == "colorjitter":
        return {"colorjitter_strength": trial.suggest_float("colorjitter_strength", 0.05, 0.5)}
    if aug == "blur":
        return {"blur_sigma": trial.suggest_float("blur_sigma", 0.1, 3.0)}
    if aug == "avgblur":
        return {"avgblur_kernel": trial.suggest_int("avgblur_kernel", 3, 9, step=2)}
    if aug == "sobel":
        return {"sobel_prob": trial.suggest_float("sobel_prob", 0.05, 0.5)}
    if aug == "noise":
        return {"noise_std": trial.suggest_float("noise_std", 1e-3, 0.1, log=True)}
    if aug == "cutout":
        return {"cutout_scale": trial.suggest_float("cutout_scale", 0.05, 0.4)}
    raise ValueError(f"알 수 없는 augmentation: {aug}")


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
# 3. CutMix helper
# ---------------------------------------------------------------------------
def cutmix_batch(
    images: torch.Tensor, labels: torch.Tensor, alpha: float = 1.0
) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
    lam = float(np.random.beta(alpha, alpha))
    rand_idx = torch.randperm(images.size(0), device=images.device)
    _, _, H, W = images.shape
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h, cut_w = int(H * cut_ratio), int(W * cut_ratio)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1 = max(cx - cut_w // 2, 0);  x2 = min(cx + cut_w // 2, W)
    y1 = max(cy - cut_h // 2, 0);  y2 = min(cy + cut_h // 2, H)
    images = images.clone()
    images[:, :, y1:y2, x1:x2] = images[rand_idx, :, y1:y2, x1:x2]
    lam = 1.0 - (y2 - y1) * (x2 - x1) / (H * W)  # 실제 mix 비율 재계산
    return lam, images, labels, labels[rand_idx]


# ---------------------------------------------------------------------------
# 4. Transforms (단일 학습 모드 — 플래그 기반)
# ---------------------------------------------------------------------------
def build_transforms(args: argparse.Namespace) -> Tuple[transforms.Compose, transforms.Compose]:
    """단일 학습 모드용 augmentation 정의 (기존 방식 유지).

    Baseline: RandomResizedCrop + HorizontalFlip + VerticalFlip
    옵션 (args 플래그로 on/off):
      --use_colorjitter : Color Jitter (brightness/contrast/saturation/hue)
      --use_elastic     : Elastic Transform
      --use_cutmix      : CutMix (batch-level, train loop 에서 적용)

    Optuna 탐색 모드에서는 transforms.py 의 build_transforms(aug_params) 를 사용한다.
    """
    pil_augs = [
        transforms.RandomResizedCrop(size=300),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
    ]
    if args.use_colorjitter:
        pil_augs.append(
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        )
    if args.use_elastic:
        pil_augs.append(transforms.ElasticTransform(alpha=50.0, sigma=5.0))

    train_tf = transforms.Compose([
        *pil_augs,
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
# 5. Train / Validate
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_cutmix: bool = False,
) -> float:
    model.train()
    total_loss, total_n = 0.0, 0

    pbar = tqdm(loader, desc=f"[Train E{epoch}]", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_cutmix and np.random.random() > 0.5:
            lam, images, labels_a, labels_b = cutmix_batch(images, labels)
            logits = model(images)
            loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
        else:
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
# 6. 공용 학습 루틴 (단일 학습 / Optuna trial 양쪽에서 사용)
# ---------------------------------------------------------------------------
def train_model(
    args: argparse.Namespace,
    train_loader: DataLoader,
    val_loader: DataLoader,
    counts: np.ndarray,
    device: torch.device,
    run_name: str,
    output_dir: Path,
    trial: Optional["optuna.trial.Trial"] = None,  # noqa: F821
) -> float:
    """모델/Loss/Optimizer 를 만들고 학습 루프를 돌린 뒤 Best Macro F1 을 반환.

    trial 이 주어지면 매 epoch 마다 val Macro F1 을 Optuna 에 report 하고,
    pruner 판단에 따라 optuna.TrialPruned 를 raise 한다.
    """
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

    # CB-Focal 사용 시 클래스별 alpha (= criterion.class_weights) 값을 한 번 출력.
    if args.loss_type == "cb_focal":
        alpha = criterion.class_weights.detach().cpu().numpy()
        alpha_str = ", ".join(
            f"{cls}={a:.4f}" for cls, a in zip(HAM10000_CLASSES, alpha)
        )
        print(f"[INFO] CB-Focal alpha         : {alpha_str}")
        print(f"[INFO] alpha sum              : {float(alpha.sum()):.4f}  (== num_classes={NUM_CLASSES})")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = -1.0
    save_name = args.save_name or f"best_model_{run_name}.pth"
    best_path = output_dir / save_name
    history: List[dict] = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            use_cutmix=args.use_cutmix,
        )
        val_loss, val_f1, report = validate(model, val_loader, criterion, device, epoch)
        scheduler.step()
        dt = time.time() - t0

        print(
            f"\n[{run_name}] [Epoch {epoch:03d}/{args.epochs}] "
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

        # ---- Optuna pruning ----
        if trial is not None:
            import optuna  # 지역 import: optuna 미설치 환경(단일 학습)에서도 동작
            trial.report(val_f1, step=epoch)
            if trial.should_prune():
                print(f"  >> [PRUNED] trial {trial.number} @ epoch {epoch} (val_macroF1={val_f1:.4f})")
                # 중도 종료 trial 도 진행 기록은 남긴다.
                _dump_history(output_dir, run_name, best_f1, history, args, pruned_at=epoch)
                raise optuna.TrialPruned()

    _dump_history(output_dir, run_name, best_f1, history, args)
    print(f"\n[DONE] {run_name}  Best Macro F1 = {best_f1:.4f}")
    print(f"[DONE] Best weights : {best_path}")
    return best_f1


def _dump_history(
    output_dir: Path,
    run_name: str,
    best_f1: float,
    history: List[dict],
    args: argparse.Namespace,
    pruned_at: Optional[int] = None,
) -> None:
    log_path = output_dir / f"history_{run_name}.json"
    payload = {
        "best_macro_f1": best_f1,
        "history": history,
        "args": vars(args),
    }
    if pruned_at is not None:
        payload["pruned_at_epoch"] = pruned_at
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 7. 데이터 준비 (공용)
# ---------------------------------------------------------------------------
def prepare_data(args: argparse.Namespace):
    """metadata 로드 → split → image roots 탐색. (train_df, val_df, image_roots) 반환."""
    data_dir = Path(args.data_dir).expanduser().resolve()
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
    return train_df, val_df, image_roots


# ---------------------------------------------------------------------------
# 8. 단일 학습 모드 (기존 방식)
# ---------------------------------------------------------------------------
def run_single(args: argparse.Namespace, device: torch.device, output_dir: Path) -> None:
    run_name = args.run_name or f"{args.backbone}_{args.loss_type}"
    if args.run_name is None:
        args.run_name = run_name

    aug_flags = [
        f for f, on in [
            ("cutmix", args.use_cutmix),
            ("elastic", args.use_elastic),
            ("colorjitter", args.use_colorjitter),
        ] if on
    ]
    print(f"[INFO] mode                  : single training")
    print(f"[INFO] run_name              : {run_name}")
    print(f"[INFO] backbone              : {args.backbone}")
    print(f"[INFO] loss_type             : {args.loss_type}")
    print(f"[INFO] augmentation (extra)  : {aug_flags if aug_flags else 'baseline only'}")
    print(f"[INFO] epochs / bs / lr      : {args.epochs} / {args.batch_size} / {args.lr}")
    if args.loss_type == "cb_focal":
        print(f"[INFO] beta / gamma          : {args.beta} / {args.gamma}")

    train_df, val_df, image_roots = prepare_data(args)

    train_tf, val_tf = build_transforms(args)
    train_set = HAM10000Dataset(train_df, image_roots, transform=train_tf)
    val_set = HAM10000Dataset(val_df, image_roots, transform=val_tf)

    counts = train_set.class_counts()
    print("[INFO] train class distribution:")
    for cls, n in zip(HAM10000_CLASSES, counts):
        print(f"    {cls:>6s} : {n}")

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory,
    )

    best_f1 = train_model(
        args, train_loader, val_loader, counts, device, run_name, output_dir,
    )
    print(f"[DONE] Train history: {output_dir / f'history_{run_name}.json'}")
    print(f"[RESULT] Best Macro F1 = {best_f1:.4f}")


# ---------------------------------------------------------------------------
# 9. Optuna 증강 탐색 모드
# ---------------------------------------------------------------------------
def run_optuna(args: argparse.Namespace, device: torch.device, output_dir: Path) -> None:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

    print(f"[INFO] mode                  : Optuna augmentation search")
    print(f"[INFO] backbone              : {args.backbone}")
    print(f"[INFO] loss_type             : {args.loss_type}")
    print(f"[INFO] study_name            : {args.study_name}")
    print(f"[INFO] storage               : {args.storage}")
    print(f"[INFO] n_trials (target)     : {args.n_trials}")
    print(f"[INFO] epochs / bs / lr      : {args.epochs} / {args.batch_size} / {args.lr}")

    train_df, val_df, image_roots = prepare_data(args)

    # 데이터셋/로더는 한 번만 만든다. 증강(train_set.transform)만 trial 마다 교체.
    train_set = HAM10000Dataset(train_df, image_roots, transform=None)
    counts = train_set.class_counts()
    print("[INFO] train class distribution:")
    for cls, n in zip(HAM10000_CLASSES, counts):
        print(f"    {cls:>6s} : {n}")

    # val transform 은 고정. build_optuna_transforms 의 val 파이프라인을 재사용.
    _, val_tf = build_optuna_transforms({})
    val_set = HAM10000Dataset(val_df, image_roots, transform=val_tf)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory,
    )

    # ---- objective -------------------------------------------------------
    def objective(trial: "optuna.trial.Trial") -> float:
        # 매 trial 동일 초기 상태 → 오직 augmentation 만 변수가 되도록 고정.
        set_seed(args.seed)

        aug_params = {
            "crop_scale_min":       trial.suggest_float("crop_scale_min", 0.5, 1.0),
            "rotate_deg":           trial.suggest_float("rotate_deg", 0.0, 180.0),
            "colorjitter_strength": trial.suggest_float("colorjitter_strength", 0.0, 0.5),
            "sobel_prob":           trial.suggest_float("sobel_prob", 0.0, 0.5),
            "noise_std":            trial.suggest_float("noise_std", 1e-3, 0.1, log=True),
            "noise_prob":           trial.suggest_float("noise_prob", 0.0, 1.0),
            "blur_prob":            trial.suggest_float("blur_prob", 0.0, 1.0),
            "blur_sigma":           trial.suggest_float("blur_sigma", 0.1, 2.0),
            "avgblur_prob":         trial.suggest_float("avgblur_prob", 0.0, 1.0),
            "avgblur_kernel":       trial.suggest_int("avgblur_kernel", 3, 7, step=2),
            "cutout_prob":          trial.suggest_float("cutout_prob", 0.0, 1.0),
            "cutout_scale":         trial.suggest_float("cutout_scale", 0.05, 0.3),
        }

        train_tf, _ = build_optuna_transforms(aug_params)
        train_set.transform = train_tf  # 데이터 재스캔 없이 증강만 교체

        run_name = f"{args.backbone}_{args.loss_type}_trial_{trial.number}"
        print("\n" + "=" * 60)
        print(f"  Trial {trial.number}  ({run_name})")
        for k, v in aug_params.items():
            print(f"    {k:>20s} : {v}")
        print("=" * 60)

        best_f1 = train_model(
            args, train_loader, val_loader, counts, device, run_name, output_dir,
            trial=trial,
        )
        return best_f1

    # ---- study (resume 지원) --------------------------------------------
    sampler = TPESampler(seed=args.seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,   # ← 중단점 재개: 동일 study 가 있으면 이어서 진행
        sampler=sampler,
        pruner=pruner,
    )

    # 이미 끝난(COMPLETE) trial 수를 빼서 "남은 횟수"만 실행 → 재실행해도 누적 폭주 방지.
    completed = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    remaining = max(0, args.n_trials - len(completed))
    print(f"[INFO] completed trials      : {len(completed)} / {args.n_trials}")
    print(f"[INFO] remaining trials      : {remaining}")

    if remaining > 0:
        study.optimize(objective, n_trials=remaining)
    else:
        print("[INFO] 목표 trial 수에 이미 도달했습니다. (재실행 시 추가 탐색 없음)")

    # ---- 결과 요약 -------------------------------------------------------
    print("\n" + "#" * 60)
    print("#  Optuna 탐색 완료")
    print("#" * 60)
    print(f"[BEST] value (Macro F1)      : {study.best_value:.4f}")
    print(f"[BEST] trial number          : {study.best_trial.number}")
    print(f"[BEST] run_name              : {args.backbone}_{args.loss_type}_trial_{study.best_trial.number}")
    print("[BEST] params:")
    for k, v in study.best_params.items():
        print(f"    {k:>20s} : {v}")

    # best 파라미터를 별도 JSON 으로 저장.
    best_path = output_dir / f"optuna_best_{args.study_name}.json"
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "study_name": args.study_name,
                "best_value": study.best_value,
                "best_trial_number": study.best_trial.number,
                "best_params": study.best_params,
                "n_completed": len(completed) + remaining,
            },
            f, indent=2, ensure_ascii=False,
        )
    print(f"[DONE] best params 저장      : {best_path}")


# ---------------------------------------------------------------------------
# 9b. 단독 튜닝 모드 (stage 1) — 한 기법만 Optuna 탐색
# ---------------------------------------------------------------------------
def run_single_aug_search(args: argparse.Namespace, device: torch.device, output_dir: Path) -> None:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

    aug = args.search_aug
    study_name = f"singleaug_{aug}"  # 기법별 독립 study (resume 도 기법 단위)

    print(f"[INFO] mode                  : single-aug tuning")
    print(f"[INFO] target augmentation   : {aug}")
    print(f"[INFO] backbone / loss        : {args.backbone} / {args.loss_type}")
    print(f"[INFO] study_name            : {study_name}")
    print(f"[INFO] storage               : {args.storage}")
    print(f"[INFO] n_trials (target)     : {args.n_trials}")
    print(f"[INFO] epochs / bs / lr      : {args.epochs} / {args.batch_size} / {args.lr}")

    train_df, val_df, image_roots = prepare_data(args)

    train_set = HAM10000Dataset(train_df, image_roots, transform=None)
    counts = train_set.class_counts()
    val_set = HAM10000Dataset(val_df, image_roots, transform=_only_val_transform())

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory,
    )

    def objective(trial: "optuna.trial.Trial") -> float:
        set_seed(args.seed)  # 매 trial 동일 초기화 → 강도만 변수
        params = suggest_aug_params(trial, aug)
        train_tf, _ = build_controlled_transforms([aug], params)
        train_set.transform = train_tf

        run_name = f"{args.backbone}_{args.loss_type}_singleaug_{aug}_trial_{trial.number}"
        print("\n" + "=" * 60)
        print(f"  [single-aug: {aug}] Trial {trial.number}  params={params}")
        print("=" * 60)
        return train_model(
            args, train_loader, val_loader, counts, device, run_name, output_dir, trial=trial,
        )

    sampler = TPESampler(seed=args.seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(
        direction="maximize", study_name=study_name, storage=args.storage,
        load_if_exists=True, sampler=sampler, pruner=pruner,
    )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    remaining = max(0, args.n_trials - len(completed))
    print(f"[INFO] completed / remaining : {len(completed)} / {remaining}")
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)

    print("\n" + "#" * 60)
    print(f"#  [single-aug: {aug}] 탐색 완료")
    print("#" * 60)
    print(f"[BEST] value (Macro F1)      : {study.best_value:.4f}")
    print(f"[BEST] params                : {study.best_params}")

    # 기법별 best 를 공용 JSON 에 누적 저장 (8개 모이면 조합 실험 입력이 됨).
    combined_path = output_dir / "optuna_best_per_aug.json"
    combined = {}
    if combined_path.exists():
        combined = json.loads(combined_path.read_text(encoding="utf-8"))
    combined[aug] = {"best_value": float(study.best_value), "params": study.best_params}
    combined_path.write_text(
        json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[DONE] per-aug best 갱신     : {combined_path}")


# ---------------------------------------------------------------------------
# 9c. 조합 실험 모드 (stage 2/3) — 고정 파라미터로 지정 기법 조합 1회 학습
# ---------------------------------------------------------------------------
def _load_aug_params(args: argparse.Namespace, output_dir: Path) -> dict:
    """per-aug best JSON 을 읽어 {param: value} 평탄화 dict 로 반환."""
    path = Path(args.aug_params_json) if args.aug_params_json else output_dir / "optuna_best_per_aug.json"
    if not path.exists():
        print(f"[WARN] 파라미터 JSON 이 없어 기본값으로 진행: {path}")
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    flat: dict = {}
    for _aug, entry in raw.items():
        params = entry.get("params", entry) if isinstance(entry, dict) else {}
        if isinstance(params, dict):
            flat.update(params)
    return flat


def run_combo(args: argparse.Namespace, device: torch.device, output_dir: Path) -> None:
    raw = (args.combo_augs or "").strip().lower()
    if raw in ("", "base", "none"):
        active = []
    else:
        active = [a.strip() for a in raw.split(",") if a.strip()]
    # 대칭성 보장: 정렬해 canonical run_name 생성 (crop_rotate == rotate_crop).
    active_sorted = sorted(active, key=lambda a: AUG_NAMES.index(a) if a in AUG_NAMES else 999)
    tag = "_".join(active_sorted) if active_sorted else "base"
    run_name = args.run_name or f"combo_{tag}"

    aug_params = _load_aug_params(args, output_dir)

    print(f"[INFO] mode                  : combination run")
    print(f"[INFO] run_name              : {run_name}")
    print(f"[INFO] active augmentations  : {active_sorted if active_sorted else 'baseline (none)'}")
    print(f"[INFO] backbone / loss        : {args.backbone} / {args.loss_type}")
    print(f"[INFO] epochs / bs / lr      : {args.epochs} / {args.batch_size} / {args.lr}")

    train_df, val_df, image_roots = prepare_data(args)

    train_tf, val_tf = build_controlled_transforms(active_sorted, aug_params)
    train_set = HAM10000Dataset(train_df, image_roots, transform=train_tf)
    val_set = HAM10000Dataset(val_df, image_roots, transform=val_tf)
    counts = train_set.class_counts()

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory,
    )

    best_f1 = train_model(args, train_loader, val_loader, counts, device, run_name, output_dir)
    print(f"[RESULT] {run_name}  Best Macro F1 = {best_f1:.4f}")


def _only_val_transform():
    """검증 transform 한 개만 필요할 때 (build_controlled_transforms 의 val 재사용)."""
    _, val_tf = build_controlled_transforms([], {})
    return val_tf


# ---------------------------------------------------------------------------
# 10. Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device                : {device}")

    if args.search_aug:                 # stage 1: 단독 튜닝
        run_single_aug_search(args, device, output_dir)
    elif args.combo_augs is not None:   # stage 2/3: 조합 학습 1회
        run_combo(args, device, output_dir)
    elif args.optuna:                   # joint 탐색 (검증 baseline)
        run_optuna(args, device, output_dir)
    else:                               # 단일 학습 (기존 플래그 방식)
        run_single(args, device, output_dir)


if __name__ == "__main__":
    main()
