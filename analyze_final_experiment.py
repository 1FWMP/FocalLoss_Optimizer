"""
최종 실험(run_final_experiment.sh) 결과 집계·분석.

대상: outputs_final/ 의
    history_{run_name}.json   (best_macro_f1 + args)
    best_model_{run_name}.pth (per-class F1 평가용, --no_per_class 로 생략 가능)

run_name 규칙(드라이버가 생성):
    {backbone}_ce_{noaug|rotblur}_s{seed}
    {backbone}_cbf_g{gamma}_{noaug|rotblur}_s{seed}
다만 조건/시드는 파일명이 아니라 **history 내부 args**(loss_type/gamma/combo_augs/seed)
에서 직접 읽어 안정적으로 분류한다.

생성물:
    outputs_final/final_summary.csv          # (증강 × loss설정) 별 Macro F1 mean±std(n=3)
    outputs_final/final_contrasts.csv        # 증강효과 / loss효과 페어드 대조
    outputs_final/final_per_class_f1.csv     # (증강 × loss설정) 별 클래스별 F1 mean±std
    outputs_final/plots/gamma_sensitivity.png        # γ→Macro F1 (CE 기준선 오버레이)
    outputs_final/plots/minority_f1_vs_gamma.png     # γ→소수클래스 F1

사용법:
    python analyze_final_experiment.py --output_dir ./outputs_final
    python analyze_final_experiment.py --output_dir ./outputs_final --no_per_class
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # 디스플레이 없는 서버 대비
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    # 단일 소스(transforms.AUG_NAMES)를 우선 사용.
    from transforms import AUG_NAMES
except Exception:  # noqa: BLE001
    # torch 미설치 환경에서 --no_per_class 집계만 할 때를 위한 폴백.
    # (transforms.py:AUG_NAMES 와 순서 동일하게 유지할 것)
    AUG_NAMES = ["crop", "rotate", "colorjitter", "blur", "avgblur", "sobel", "noise", "cutout"]

# HAM10000 의 소수(희귀) 클래스 — CB-Focal/ focusing 의 핵심 관전 포인트
MINORITY_CLASSES = ["akiec", "df", "vasc"]


# ---------------------------------------------------------------------------
# 조건 분류 헬퍼
# ---------------------------------------------------------------------------
def _fmt_gamma(g: float) -> str:
    return str(int(g)) if float(g).is_integer() else f"{float(g):g}"


def _aug_tag(combo_augs: Optional[str]) -> str:
    """history args.combo_augs → 증강 arm 라벨 ('noaug' 또는 'rotate+blur' 등)."""
    raw = (combo_augs or "").strip().lower()
    if raw in ("", "base", "none"):
        return "noaug"
    active = [a.strip() for a in raw.split(",") if a.strip()]
    active = sorted(active, key=lambda a: AUG_NAMES.index(a) if a in AUG_NAMES else 999)
    return "+".join(active) if active else "noaug"


def _loss_setting(loss_type: str, gamma: float) -> str:
    """loss 설정 라벨. CE / CBF(γ=k)."""
    if loss_type == "cb_focal":
        return f"CBF(γ={_fmt_gamma(gamma)})"
    return "CE"


# ---------------------------------------------------------------------------
# 로딩
# ---------------------------------------------------------------------------
def load_runs(output_dir: Path) -> pd.DataFrame:
    rows: List[dict] = []
    for path in sorted(output_dir.glob("history_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            args = data.get("args", {})
            f1 = float(data["best_macro_f1"])
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"[WARN] 읽기 실패, 건너뜀 {path.name}: {e}")
            continue
        loss_type = args.get("loss_type", "ce")
        gamma = float(args.get("gamma", 2.0))
        rows.append({
            "run_name": args.get("run_name") or path.stem[len("history_"):],
            "aug": _aug_tag(args.get("combo_augs")),
            "loss_setting": _loss_setting(loss_type, gamma),
            "loss_type": loss_type,
            "gamma": gamma if loss_type == "cb_focal" else np.nan,
            "seed": int(args.get("seed", -1)),
            "best_macro_f1": f1,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1) 헤드라인 표 : (증강 × loss설정) mean±std
# ---------------------------------------------------------------------------
def summarize(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    g = (
        df.groupby(["aug", "loss_setting"])["best_macro_f1"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "macro_f1_mean", "std": "macro_f1_std", "count": "n_seeds"})
    )
    g["macro_f1_std"] = g["macro_f1_std"].fillna(0.0)
    g = g.sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)

    out = output_dir / "final_summary.csv"
    g.round(6).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[DONE] 헤드라인 요약 → {out}")

    print("\n[최종 요약] (증강 × loss설정) Best Macro F1  mean ± std (n=seed)")
    for _, r in g.iterrows():
        print(f"  {r['macro_f1_mean']:.4f} ± {r['macro_f1_std']:.4f}  "
              f"(n={int(r['n_seeds'])})  {r['aug']:>12s} × {r['loss_setting']}")
    best = g.iloc[0]
    print(f"\n  >> 시드 평균 최고: {best['aug']} × {best['loss_setting']} "
          f"= {best['macro_f1_mean']:.4f}")
    return g


# ---------------------------------------------------------------------------
# 2) gamma 민감도 곡선 (CE 를 수평 기준선으로)
# ---------------------------------------------------------------------------
def plot_gamma_sensitivity(df: pd.DataFrame, plot_dir: Path) -> None:
    augs = sorted(df["aug"].unique())
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(augs), 1)))

    for aug, color in zip(augs, colors):
        sub = df[df["aug"] == aug]
        cbf = sub[sub["loss_type"] == "cb_focal"]
        if not cbf.empty:
            grp = cbf.groupby("gamma")["best_macro_f1"].agg(["mean", "std"]).reset_index()
            grp["std"] = grp["std"].fillna(0.0)
            ax.errorbar(grp["gamma"], grp["mean"], yerr=grp["std"], marker="o",
                        capsize=4, color=color, label=f"{aug} · CB-Focal")
        ce = sub[sub["loss_type"] == "ce"]
        if not ce.empty:
            ax.axhline(ce["best_macro_f1"].mean(), linestyle="--", color=color, alpha=0.7,
                       label=f"{aug} · CE (ref)")

    ax.set_xlabel("Focal gamma (focusing)")
    ax.set_ylabel("Best Macro F1 (mean ± std over seeds)")
    ax.set_title("CB-Focal gamma sensitivity vs CE baseline")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    plot_dir.mkdir(parents=True, exist_ok=True)
    path = plot_dir / "gamma_sensitivity.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[DONE] gamma 곡선 → {path}")


# ---------------------------------------------------------------------------
# 3) 페어드 대조 (같은 seed 끼리 Δ)
# ---------------------------------------------------------------------------
def _paired_delta(df: pd.DataFrame, a_mask: pd.Series, b_mask: pd.Series) -> Optional[dict]:
    """a − b 를 seed 로 페어링해 Δ 통계 반환. (a, b 각각 seed→f1)"""
    a = df[a_mask].set_index("seed")["best_macro_f1"]
    b = df[b_mask].set_index("seed")["best_macro_f1"]
    common = sorted(set(a.index) & set(b.index))
    if not common:
        return None
    deltas = np.array([a[s] - b[s] for s in common], dtype=float)
    res = {
        "n_pairs": len(common),
        "delta_mean": float(deltas.mean()),
        "delta_std": float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0,
    }
    # 유의성: scipy 있으면 paired t-test, 없으면 효과크기로 대체
    if len(deltas) > 1:
        try:
            from scipy import stats
            t, p = stats.ttest_rel(
                [a[s] for s in common], [b[s] for s in common]
            )
            res["t_stat"] = float(t)
            res["p_value"] = float(p)
        except Exception:
            res["t_stat"] = np.nan
            res["p_value"] = np.nan
            res["cohen_dz"] = res["delta_mean"] / res["delta_std"] if res["delta_std"] else np.nan
    return res


def contrasts(df: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    rows: List[dict] = []

    # (A) 증강 효과: rotate+blur − noaug  (각 loss설정별)
    aug_on = [a for a in df["aug"].unique() if a != "noaug"]
    for setting in sorted(df["loss_setting"].unique()):
        for aon in aug_on:
            r = _paired_delta(
                df,
                (df["aug"] == aon) & (df["loss_setting"] == setting),
                (df["aug"] == "noaug") & (df["loss_setting"] == setting),
            )
            if r:
                rows.append({"contrast": f"[증강] {aon} − noaug",
                             "condition": setting, **r})

    # (B) loss 효과: 각 증강에서 best-gamma CB-Focal − CE
    for aug in sorted(df["aug"].unique()):
        sub = summary[(summary["aug"] == aug)]
        cbf = sub[sub["loss_setting"].str.startswith("CBF")]
        if cbf.empty or "CE" not in set(sub["loss_setting"]):
            continue
        best_setting = cbf.sort_values("macro_f1_mean", ascending=False).iloc[0]["loss_setting"]
        r = _paired_delta(
            df,
            (df["aug"] == aug) & (df["loss_setting"] == best_setting),
            (df["aug"] == aug) & (df["loss_setting"] == "CE"),
        )
        if r:
            rows.append({"contrast": f"[loss] {best_setting} − CE",
                         "condition": aug, **r})

    if not rows:
        print("[WARN] 대조 분석할 페어가 없습니다.")
        return

    cdf = pd.DataFrame(rows)
    out = output_dir / "final_contrasts.csv"
    cdf.round(6).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[DONE] 대조 분석 → {out}")

    print("\n[페어드 대조] (같은 seed 끼리 Δ, +면 앞 조건이 우세)")
    for _, r in cdf.iterrows():
        sig = ""
        if "p_value" in r and pd.notna(r.get("p_value")):
            sig = f"  p={r['p_value']:.3f}"
        elif "cohen_dz" in r and pd.notna(r.get("cohen_dz")):
            sig = f"  dz={r['cohen_dz']:.2f}"
        print(f"  {r['delta_mean']:+.4f} ± {r['delta_std']:.4f}  "
              f"({r['contrast']} | {r['condition']}){sig}")


# ---------------------------------------------------------------------------
# 4) 소수 클래스 per-class F1 (best_model_*.pth 평가)
# ---------------------------------------------------------------------------
def per_class_analysis(df: pd.DataFrame, output_dir: Path, plot_dir: Path) -> None:
    try:
        import torch
        from eval_accuracy import evaluate
        from dataset import HAM10000_CLASSES
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] per-class 평가 의존성 로드 실패 → 생략: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: List[dict] = []
    for _, run in df.iterrows():
        ckpt = output_dir / f"best_model_{run['run_name']}.pth"
        if not ckpt.exists():
            print(f"[WARN] 체크포인트 없음, 건너뜀: {ckpt.name}")
            continue
        print(f"  per-class 평가: {ckpt.name} ...", flush=True)
        try:
            r = evaluate(ckpt, device)
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] 평가 실패 {ckpt.name}: {e}")
            continue
        row = {"aug": run["aug"], "loss_setting": run["loss_setting"],
               "loss_type": run["loss_type"], "gamma": run["gamma"], "seed": run["seed"]}
        for cls in HAM10000_CLASSES:
            row[f"f1_{cls}"] = r["per_class_f1"][cls]
        rows.append(row)

    if not rows:
        print("[WARN] per-class 평가 결과가 없습니다 (체크포인트 미존재?).")
        return

    pdf = pd.DataFrame(rows)
    f1_cols = [c for c in pdf.columns if c.startswith("f1_")]

    # (증강 × loss설정) 별 클래스별 mean±std
    agg = pdf.groupby(["aug", "loss_setting"])[f1_cols].agg(["mean", "std"])
    agg.columns = [f"{c}_{stat}" for c, stat in agg.columns]
    agg = agg.reset_index()
    out = output_dir / "final_per_class_f1.csv"
    agg.round(6).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[DONE] per-class F1 → {out}")

    # 소수 클래스 평균 F1 을 gamma 함수로 (증강 arm별), CE 기준선
    minority_cols = [f"f1_{c}" for c in MINORITY_CLASSES if f"f1_{c}" in pdf.columns]
    pdf["minority_mean_f1"] = pdf[minority_cols].mean(axis=1)
    augs = sorted(pdf["aug"].unique())
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(augs), 1)))
    for aug, color in zip(augs, colors):
        sub = pdf[pdf["aug"] == aug]
        cbf = sub[sub["loss_type"] == "cb_focal"]
        if not cbf.empty:
            grp = cbf.groupby("gamma")["minority_mean_f1"].agg(["mean", "std"]).reset_index()
            grp["std"] = grp["std"].fillna(0.0)
            ax.errorbar(grp["gamma"], grp["mean"], yerr=grp["std"], marker="o",
                        capsize=4, color=color, label=f"{aug} · CB-Focal")
        ce = sub[sub["loss_type"] == "ce"]
        if not ce.empty:
            ax.axhline(ce["minority_mean_f1"].mean(), linestyle="--", color=color, alpha=0.7,
                       label=f"{aug} · CE (ref)")
    ax.set_xlabel("Focal gamma (focusing)")
    ax.set_ylabel(f"Minority-class mean F1 ({'/'.join(MINORITY_CLASSES)})")
    ax.set_title("Minority-class F1 vs gamma (CB-Focal vs CE)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = plot_dir / "minority_f1_vs_gamma.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[DONE] 소수클래스 F1 곡선 → {path}")


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="최종 실험 집계·분석")
    parser.add_argument("--output_dir", type=str, default="./outputs_final")
    parser.add_argument("--no_per_class", action="store_true",
                        help="best_model_*.pth per-class F1 평가를 생략(집계만)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    plot_dir = output_dir / "plots"

    df = load_runs(output_dir)
    if df.empty:
        print(f"[ERROR] history_*.json 을 찾지 못했습니다: {output_dir}")
        return
    print(f"[INFO] 로드한 런 수: {len(df)}  "
          f"(증강 {df['aug'].nunique()}종 × loss설정 {df['loss_setting'].nunique()}종)")

    summary = summarize(df, output_dir)
    plot_gamma_sensitivity(df, plot_dir)
    contrasts(df, summary, output_dir)
    if not args.no_per_class:
        per_class_analysis(df, output_dir, plot_dir)


if __name__ == "__main__":
    main()
