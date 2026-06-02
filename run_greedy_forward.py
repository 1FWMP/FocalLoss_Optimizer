"""
Stage 3 : Greedy forward selection.

조합 히트맵(Stage 2)에서 가장 좋은 두 기법 조합(pair)을 시작점으로 잡고,
남은 기법을 하나씩 추가하면서 Best Macro F1 이 margin 이상 개선될 때만 채택한다.
더 이상 개선되지 않으면 중단한다.

  - 각 후보 조합은 main.py --combo_augs 로 1회 학습 (history 있으면 재사용 → resume).
  - 시작 pair 는 기존 history_combo_*.json 결과에서 자동 선택하거나 --start 로 지정.
  - 진행 경로는 outputs/greedy_path.json 에 기록.

사용법:
    python run_greedy_forward.py --data_dir ./data --output_dir ./outputs \
        --params_json ./outputs/optuna_best_per_aug.json --margin 0.005
    # 시작 pair 직접 지정:
    python run_greedy_forward.py ... --start crop,rotate
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from transforms import AUG_NAMES
from analyze_combinations import load_results


def _canonical(active: List[str]) -> List[str]:
    return sorted(set(active), key=lambda a: AUG_NAMES.index(a))


def _tag(active: List[str]) -> str:
    return "_".join(_canonical(active)) if active else "base"


def run_or_load(active: List[str], args: argparse.Namespace) -> float:
    """조합 active 를 학습(또는 기존 history 재사용)하고 Best Macro F1 반환."""
    active = _canonical(active)
    tag = _tag(active)
    output_dir = Path(args.output_dir)
    log = output_dir / f"history_combo_{tag}.json"

    if not log.exists():
        cmd = [
            sys.executable, "main.py",
            "--combo_augs", ",".join(active),
            "--aug_params_json", args.params_json,
            "--data_dir", args.data_dir,
            "--backbone", args.backbone,
            "--loss_type", args.loss_type,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--lr", str(args.lr),
            "--num_workers", str(args.num_workers),
            "--output_dir", args.output_dir,
        ]
        print(f"\n[RUN] combo: {tag}")
        subprocess.run(cmd, check=True)
    else:
        print(f"[REUSE] combo: {tag}  ({log.name})")

    data = json.loads(log.read_text(encoding="utf-8"))
    return float(data["best_macro_f1"])


def pick_start_pair(output_dir: Path) -> Optional[List[str]]:
    """기존 결과에서 Best Macro F1 이 가장 높은 두 기법 조합을 반환."""
    results = load_results(output_dir)
    pairs = {k: v for k, v in results.items() if len(k) == 2}
    if not pairs:
        return None
    best = max(pairs, key=pairs.get)
    return list(best)


def main() -> None:
    parser = argparse.ArgumentParser(description="Greedy forward augmentation selection")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--params_json", type=str, default="./outputs/optuna_best_per_aug.json")
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--loss_type", type=str, default="ce")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--margin", type=float, default=0.005,
                        help="이 값 이상 개선될 때만 기법 추가 채택")
    parser.add_argument("--start", type=str, default=None,
                        help="시작 pair (예: 'crop,rotate'). 미지정 시 히트맵 최고 pair 자동 선택")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()

    # ---- 시작 조합 결정 ----
    if args.start:
        current = _canonical([a.strip() for a in args.start.split(",") if a.strip()])
    else:
        current = pick_start_pair(output_dir)
        if current is None:
            print("[ERROR] 시작 pair 를 찾지 못했습니다. 먼저 조합 히트맵 실험을 돌리거나 --start 를 지정하세요.")
            sys.exit(1)

    current_f1 = run_or_load(current, args)
    print(f"\n[START] {_tag(current)}  Macro F1 = {current_f1:.4f}")

    path_log = [{"step": 0, "combo": _canonical(current), "macro_f1": current_f1, "added": None}]
    step = 0

    while True:
        step += 1
        remaining = [a for a in AUG_NAMES if a not in current]
        if not remaining:
            print("[STOP] 모든 기법을 포함했습니다.")
            break

        print(f"\n===== step {step}: 후보 추가 시도 (현재 {_tag(current)} = {current_f1:.4f}) =====")
        best_cand, best_cand_f1 = None, current_f1
        for cand in remaining:
            f1 = run_or_load(current + [cand], args)
            print(f"  + {cand:>11s} -> {f1:.4f}  (Δ{f1 - current_f1:+.4f})")
            if f1 > best_cand_f1:
                best_cand, best_cand_f1 = cand, f1

        if best_cand is not None and (best_cand_f1 - current_f1) >= args.margin:
            current = _canonical(current + [best_cand])
            print(f"[ADD] +{best_cand}  ->  {_tag(current)}  Macro F1 = {best_cand_f1:.4f}")
            path_log.append({
                "step": step, "combo": current, "macro_f1": best_cand_f1, "added": best_cand,
            })
            current_f1 = best_cand_f1
        else:
            gain = best_cand_f1 - current_f1
            print(f"[STOP] 최선 후보 개선폭 {gain:+.4f} < margin {args.margin}. 중단.")
            break

    # ---- 결과 저장 ----
    out = output_dir / "greedy_path.json"
    out.write_text(
        json.dumps(
            {"margin": args.margin, "final_combo": current, "final_macro_f1": current_f1, "path": path_log},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n[DONE] 최종 조합: {_tag(current)}  Macro F1 = {current_f1:.4f}")
    print(f"[DONE] 경로 기록: {out}")

    # joint 검증 baseline 과 비교 (있으면)
    joint_candidates = sorted(output_dir.glob("optuna_best_*.json"))
    joint_candidates = [p for p in joint_candidates if p.name != "optuna_best_per_aug.json"]
    if joint_candidates:
        try:
            jb = json.loads(joint_candidates[0].read_text(encoding="utf-8")).get("best_value")
            if jb is not None:
                print(f"[COMPARE] joint Optuna best = {float(jb):.4f}  vs  greedy = {current_f1:.4f}")
        except (json.JSONDecodeError, ValueError):
            pass


if __name__ == "__main__":
    main()
