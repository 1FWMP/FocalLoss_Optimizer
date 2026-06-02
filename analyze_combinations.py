"""
조합 실험(Stage 2) 결과 분석.

outputs/history_combo_*.json 들을 읽어
  - 8x8 Macro F1 히트맵 (대각선=단일 기법, 비대각=두 기법 조합, 대칭)
  - combination_summary.csv
를 생성한다.

run_name(파일명) 규칙: history_combo_{tag}.json
    tag = "base"            → baseline(무증강)
    tag = "{aug}"           → 단일 기법 (대각선)
    tag = "{augA}_{augB}"   → 두 기법 조합 (상삼각). aug 이름엔 '_' 가 없음.

사용법:
    python analyze_combinations.py --output_dir ./outputs
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # 디스플레이 없는 서버에서도 저장 가능
import matplotlib.pyplot as plt
import numpy as np

from transforms import AUG_NAMES


def _parse_tag(tag: str) -> Optional[List[str]]:
    """파일 tag → 활성 기법 리스트. baseline 은 [] 반환, 인식 실패 시 None."""
    if tag == "base":
        return []
    parts = tag.split("_")
    if all(p in AUG_NAMES for p in parts):
        return parts
    return None


def load_results(output_dir: Path) -> Dict[Tuple[str, ...], float]:
    """{ (정렬된 active augs tuple) : best_macro_f1 } 매핑 반환."""
    results: Dict[Tuple[str, ...], float] = {}
    for path in sorted(output_dir.glob("history_combo_*.json")):
        tag = path.stem[len("history_combo_"):]
        augs = _parse_tag(tag)
        if augs is None:
            print(f"[WARN] tag 파싱 실패, 건너뜀: {path.name}")
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            f1 = float(data["best_macro_f1"])
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"[WARN] 읽기 실패 {path.name}: {e}")
            continue
        key = tuple(sorted(augs, key=lambda a: AUG_NAMES.index(a)))
        results[key] = f1
    return results


def build_matrix(results: Dict[Tuple[str, ...], float]) -> np.ndarray:
    """8x8 행렬. [i,i]=단일 i, [i,j]=[j,i]=pair. 값 없으면 NaN."""
    n = len(AUG_NAMES)
    mat = np.full((n, n), np.nan, dtype=np.float64)
    idx = {a: i for i, a in enumerate(AUG_NAMES)}
    for key, f1 in results.items():
        if len(key) == 1:
            i = idx[key[0]]
            mat[i, i] = f1
        elif len(key) == 2:
            i, j = idx[key[0]], idx[key[1]]
            mat[i, j] = f1
            mat[j, i] = f1
    return mat


def plot_heatmap(mat: np.ndarray, baseline: Optional[float], out_path: Path) -> None:
    n = len(AUG_NAMES)
    fig, ax = plt.subplots(figsize=(9, 7.5))

    finite = mat[np.isfinite(mat)]
    vmin = float(finite.min()) if finite.size else 0.0
    vmax = float(finite.max()) if finite.size else 1.0
    im = ax.imshow(mat, cmap="viridis", vmin=vmin, vmax=vmax)

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(AUG_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(AUG_NAMES)

    # 셀 값 주석 (대각선은 굵게)
    for i in range(n):
        for j in range(n):
            v = mat[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "-", ha="center", va="center", color="gray", fontsize=8)
                continue
            # 밝기에 따라 글자색 자동
            color = "white" if (v - vmin) / max(vmax - vmin, 1e-9) < 0.5 else "black"
            ax.text(
                j, i, f"{v:.3f}", ha="center", va="center",
                color=color, fontsize=8, fontweight="bold" if i == j else "normal",
            )

    # 그림 내 텍스트는 폰트 의존성을 피하려 영어로 (한글 폰트 없는 환경 대비)
    title = "Augmentation Combination - Best Macro F1\n(diagonal: single aug, off-diagonal: pair)"
    if baseline is not None:
        title += f"\nbaseline (no aug) = {baseline:.4f}"
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Best Macro F1")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[DONE] 히트맵 저장 : {out_path}")


def write_summary(
    results: Dict[Tuple[str, ...], float],
    baseline: Optional[float],
    out_path: Path,
) -> None:
    rows = []
    for key, f1 in results.items():
        delta = (f1 - baseline) if baseline is not None else float("nan")
        rows.append({
            "combo": "+".join(key) if key else "base",
            "n_aug": len(key),
            "best_macro_f1": round(f1, 6),
            "delta_vs_baseline": round(delta, 6),
        })
    rows.sort(key=lambda r: r["best_macro_f1"], reverse=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["combo", "n_aug", "best_macro_f1", "delta_vs_baseline"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[DONE] 요약 CSV 저장: {out_path}")

    # 콘솔 Top 10
    print("\n[TOP 10 조합] (Best Macro F1 기준)")
    for r in rows[:10]:
        print(f"  {r['best_macro_f1']:.4f}  (Δ{r['delta_vs_baseline']:+.4f})  {r['combo']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Augmentation 조합 히트맵/요약 생성")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    results = load_results(output_dir)
    if not results:
        print(f"[ERROR] history_combo_*.json 을 찾지 못했습니다: {output_dir}")
        return

    baseline_path = output_dir / "history_combo_base.json"
    baseline = None
    if baseline_path.exists():
        try:
            baseline = float(json.loads(baseline_path.read_text(encoding="utf-8"))["best_macro_f1"])
        except (json.JSONDecodeError, KeyError, ValueError):
            baseline = None

    mat = build_matrix(results)
    plot_heatmap(mat, baseline, output_dir / "plots" / "combination_heatmap.png")
    write_summary(results, baseline, output_dir / "combination_summary.csv")


if __name__ == "__main__":
    main()
