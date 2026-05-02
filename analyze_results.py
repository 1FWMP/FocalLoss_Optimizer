"""
실험 결과 분석 스크립트.

outputs/ 폴더의 history_{backbone}_{loss_type}.json 파일을 읽어
summary.csv 와 시각화 이미지를 생성한다.

사용법:
    python analyze_results.py --output_dir ./outputs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


LOSS_TYPES = ["cb_focal", "ce"]  # 긴 것을 먼저 매칭해야 오파싱 방지


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./outputs")
    return parser.parse_args()


def _parse_stem(stem: str) -> tuple[str, str] | None:
    """history_{backbone}_{loss_type} → (backbone, loss_type)."""
    prefix = "history_"
    if not stem.startswith(prefix):
        return None
    body = stem[len(prefix):]
    for lt in LOSS_TYPES:
        if body.endswith(f"_{lt}"):
            backbone = body[: -(len(lt) + 1)]
            return backbone, lt
    return None


def load_histories(output_dir: Path) -> dict[str, dict]:
    histories: dict[str, dict] = {}
    for path in sorted(output_dir.glob("history_*.json")):
        parsed = _parse_stem(path.stem)
        if parsed is None:
            print(f"[WARN] 파싱 실패, 건너뜀: {path.name}")
            continue
        backbone, loss_type = parsed
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        key = f"{backbone}_{loss_type}"
        histories[key] = {
            "backbone":      backbone,
            "loss_type":     loss_type,
            "best_macro_f1": data["best_macro_f1"],
            "history":       data["history"],
        }
    return histories


def save_summary_csv(histories: dict, output_dir: Path) -> pd.DataFrame:
    rows = []
    for key, h in histories.items():
        last = h["history"][-1]
        rows.append({
            "experiment":      key,
            "backbone":        h["backbone"],
            "loss_type":       h["loss_type"],
            "best_macro_f1":   round(h["best_macro_f1"], 4),
            "final_train_loss": round(last["train_loss"], 4),
            "final_val_loss":  round(last["val_loss"], 4),
            "epochs":          len(h["history"]),
        })
    df = pd.DataFrame(rows).sort_values("best_macro_f1", ascending=False).reset_index(drop=True)
    csv_path = output_dir / "summary.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[DONE] summary.csv -> {csv_path}")
    print(df.to_string(index=False))
    return df


def plot_f1_bar(histories: dict, plot_dir: Path) -> None:
    items = sorted(histories.items(), key=lambda x: x[1]["best_macro_f1"], reverse=True)
    labels = [k for k, _ in items]
    values = [v["best_macro_f1"] for _, v in items]
    colors = ["#FF5722" if v["loss_type"] == "cb_focal" else "#2196F3" for _, v in items]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=colors)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_ylabel("Best Macro F1")
    ax.set_title("Best Macro F1 by Experiment")
    ax.set_ylim(0, min(1.0, max(values) + 0.06))

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#2196F3", label="CE"),
        Patch(color="#FF5722", label="CB-Focal"),
    ])
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    path = plot_dir / "f1_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[DONE] f1_comparison.png -> {path}")


def plot_learning_curves(histories: dict, plot_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for key, h in histories.items():
        epochs   = [r["epoch"]        for r in h["history"]]
        val_f1   = [r["val_macro_f1"] for r in h["history"]]
        val_loss = [r["val_loss"]      for r in h["history"]]
        linestyle = "--" if h["loss_type"] == "cb_focal" else "-"
        axes[0].plot(epochs, val_f1,   label=key, linestyle=linestyle)
        axes[1].plot(epochs, val_loss, label=key, linestyle=linestyle)

    for ax, title, ylabel in zip(
        axes,
        ["Val Macro F1", "Val Loss"],
        ["Macro F1",     "Loss"],
    ):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = plot_dir / "learning_curves.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[DONE] learning_curves.png -> {path}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    histories = load_histories(output_dir)
    if not histories:
        print("[ERROR] history_*.json 파일을 찾지 못했습니다. 먼저 run_experiments.sh 를 실행하세요.")
        return

    save_summary_csv(histories, output_dir)
    plot_f1_bar(histories, plot_dir)
    plot_learning_curves(histories, plot_dir)


if __name__ == "__main__":
    main()
