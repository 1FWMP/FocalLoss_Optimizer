"""
저장된 .pth 모델들의 val accuracy / macro F1 / precision / recall 을 일괄 출력.
학습 시와 동일한 seed / val_size 로 split 을 재현한다.

출력 섹션:
  1. 전체 요약   — Accuracy, Macro F1, Macro Precision, Macro Recall, 저장 F1, Best Ep
  2. 클래스별 Accuracy
  3. 클래스별 F1 Score
  4. 클래스별 Precision
  5. 클래스별 Recall
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import HAM10000_CLASSES, HAM10000Dataset, NUM_CLASSES, discover_image_roots, load_metadata
from model import build_model
from main import split_train_val

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

VAL_TF = transforms.Compose([
    transforms.Resize(320),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


@torch.no_grad()
def evaluate(ckpt_path: Path, device: torch.device) -> dict:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]

    data_dir = Path(args["data_dir"]).expanduser().resolve()
    df = load_metadata(data_dir)
    _, val_df = split_train_val(df, val_size=args["val_size"], seed=args["seed"])

    image_roots = discover_image_roots(data_dir)
    val_set = HAM10000Dataset(val_df, image_roots, transform=VAL_TF)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=0)

    model = build_model(
        backbone=args["backbone"], num_classes=NUM_CLASSES, pretrained=False
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_preds, all_targets = [], []
    for images, labels in val_loader:
        images = images.to(device)
        preds = model(images).argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(labels.numpy())

    all_preds   = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    acc      = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    per_cls_p, per_cls_r, per_cls_f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, labels=list(range(len(HAM10000_CLASSES))),
        average=None, zero_division=0
    )
    macro_precision = float(per_cls_p.mean())
    macro_recall    = float(per_cls_r.mean())

    per_class_acc, per_class_f1, per_class_precision, per_class_recall = {}, {}, {}, {}
    for i, cls in enumerate(HAM10000_CLASSES):
        mask = all_targets == i
        per_class_acc[cls]       = accuracy_score(all_targets[mask], all_preds[mask]) if mask.sum() > 0 else float("nan")
        per_class_f1[cls]        = float(per_cls_f1[i])
        per_class_precision[cls] = float(per_cls_p[i])
        per_class_recall[cls]    = float(per_cls_r[i])

    run_name = args.get("run_name") or f"{args['backbone']}_{args['loss_type']}"
    return {
        "run_name":          run_name,
        "backbone":          args["backbone"],
        "loss_type":         args["loss_type"],
        "accuracy":          acc,
        "macro_f1":          macro_f1,
        "macro_precision":   macro_precision,
        "macro_recall":      macro_recall,
        "per_class_acc":       per_class_acc,
        "per_class_f1":        per_class_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall":    per_class_recall,
        "saved_f1":          ckpt.get("val_macro_f1", float("nan")),
        "best_epoch":        ckpt.get("epoch", -1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="저장된 best_model_*.pth 일괄 평가 (Accuracy / Macro F1 / Precision / Recall + 클래스별)"
    )
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="best_model_*.pth 가 있는 디렉토리 (기본 ./outputs)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir).expanduser().resolve()
    pth_files = sorted(output_dir.glob("best_model_*.pth"))

    if not pth_files:
        print("[ERROR] .pth 파일이 없습니다.")
        return

    results = []
    for pth in pth_files:
        print(f"  평가 중: {pth.name} ...", flush=True)
        r = evaluate(pth, device)
        results.append(r)

    # 정렬: macro_f1 내림차순
    results.sort(key=lambda x: x["macro_f1"], reverse=True)

    SEP_W = 90
    header = f"{'실험':<35} {'Accuracy':>9} {'Macro F1':>9} {'Macro P':>9} {'Macro R':>9} {'저장 F1':>9} {'Best Ep':>7}"
    print("\n" + "=" * SEP_W)
    print("전체 요약")
    print("=" * SEP_W)
    print(header)
    print("-" * SEP_W)
    for r in results:
        exp = r["run_name"]
        print(
            f"{exp:<35} {r['accuracy']:>9.4f} {r['macro_f1']:>9.4f}"
            f" {r['macro_precision']:>9.4f} {r['macro_recall']:>9.4f}"
            f" {r['saved_f1']:>9.4f} {r['best_epoch']:>7}"
        )

    for metric_label, key in [
        ("클래스별 Accuracy",   "per_class_acc"),
        ("클래스별 F1 Score",   "per_class_f1"),
        ("클래스별 Precision",  "per_class_precision"),
        ("클래스별 Recall",     "per_class_recall"),
    ]:
        print(f"\n--- {metric_label} ---")
        cls_header = f"{'실험':<35}" + "".join(f"{c:>8}" for c in HAM10000_CLASSES)
        print(cls_header)
        print("-" * len(cls_header))
        for r in results:
            row = f"{r['run_name']:<35}" + "".join(f"{r[key][c]:>8.3f}" for c in HAM10000_CLASSES)
            print(row)


if __name__ == "__main__":
    main()
