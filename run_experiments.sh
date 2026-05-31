#!/usr/bin/env bash
# Step 1 : ResNet-101 vs ResNet-152 (baseline aug + CE) → 더 높은 F1 backbone 자동 선택
# Step 2 : 선택된 backbone × 2³=8 augmentation 조합 비교
#          (baseline / CutMix / Elastic / ColorJitter 각 on|off)
#
# history_{run_name}.json 이 이미 존재하는 실험은 자동으로 건너뜀.
#
# 사용법:
#   bash run_experiments.sh
#   DATA_DIR=/path/to/data bash run_experiments.sh
set -euo pipefail

DATA_DIR="${DATA_DIR:-./data}"
EPOCHS=30
BATCH_SIZE=32
LR=1e-4
NUM_WORKERS=4
OUTPUT_DIR="./outputs"

mkdir -p "$OUTPUT_DIR"

# -----------------------------------------------------------------------
# run_exp <run_name> [extra main.py args ...]
#   history_{run_name}.json 이 있으면 SKIP
# -----------------------------------------------------------------------
run_exp() {
    local run_name=$1
    shift
    local log_file="${OUTPUT_DIR}/history_${run_name}.json"

    echo ""
    echo "=========================================="
    echo "  run : $run_name"
    echo "=========================================="

    if [ -f "$log_file" ]; then
        echo "  [SKIP] 이미 완료된 실험: $log_file"
        return
    fi

    python main.py \
        --data_dir    "$DATA_DIR" \
        --epochs      "$EPOCHS" \
        --batch_size  "$BATCH_SIZE" \
        --lr          "$LR" \
        --num_workers "$NUM_WORKERS" \
        --output_dir  "$OUTPUT_DIR" \
        --run_name    "$run_name" \
        "$@"
}

# ========================================================================
# Step 1 : ResNet depth 비교 (baseline augmentation + CE)
# ========================================================================
echo ""
echo "####################################################"
echo "#  Step 1 : ResNet-101 vs ResNet-152 (baseline CE) #"
echo "####################################################"

run_exp resnet101_ce --backbone resnet101 --loss_type ce
run_exp resnet152_ce --backbone resnet152 --loss_type ce

# --- Step 1 결과에서 Macro F1 이 더 높은 backbone 자동 선택 ---
WINNER_BACKBONE=$(python3 - <<PYEOF
import json, pathlib, sys
output_dir = pathlib.Path("${OUTPUT_DIR}")
candidates = {
    "resnet101": output_dir / "history_resnet101_ce.json",
    "resnet152": output_dir / "history_resnet152_ce.json",
}
scores = {}
for bb, path in candidates.items():
    if path.exists():
        scores[bb] = json.loads(path.read_text(encoding="utf-8"))["best_macro_f1"]
if not scores:
    print("[ERROR] Step 1 history 파일이 없습니다.", file=sys.stderr)
    sys.exit(1)
winner = max(scores, key=scores.get)
for bb, f1 in sorted(scores.items()):
    flag = " <-- WINNER" if bb == winner else ""
    print(f"  {bb} : Macro F1 = {f1:.4f}{flag}", file=sys.stderr)
print(winner)
PYEOF
)

echo ""
echo "  >> Step 1 Winner backbone : $WINNER_BACKBONE"

# ========================================================================
# Step 2 : Augmentation 2^3 = 8 조합 비교 (winner backbone + CE)
#   Baseline : RandomResizedCrop + HorizontalFlip + VerticalFlip
#   변수     : CutMix (--use_cutmix) / Elastic (--use_elastic) / ColorJitter (--use_colorjitter)
# ========================================================================
echo ""
echo "####################################################"
echo "#  Step 2 : Augmentation 2^3 = 8 조합              #"
echo "#  backbone : ${WINNER_BACKBONE}                   #"
echo "####################################################"

B="$WINNER_BACKBONE"

# run_aug <tag> [extra flags ...]
run_aug() {
    local tag=$1; shift
    run_exp "${B}_ce_${tag}" --backbone "$B" --loss_type ce "$@"
}

run_aug base
run_aug cm          --use_cutmix
run_aug el          --use_elastic
run_aug cj          --use_colorjitter
run_aug cm_el       --use_cutmix  --use_elastic
run_aug cm_cj       --use_cutmix  --use_colorjitter
run_aug el_cj       --use_elastic --use_colorjitter
run_aug cm_el_cj    --use_cutmix  --use_elastic --use_colorjitter

# ========================================================================
# 결과 분석
# ========================================================================
echo ""
echo "=========================================="
echo "  모든 실험 완료. 결과 분석 중..."
echo "=========================================="
python analyze_results.py --output_dir "$OUTPUT_DIR"

echo ""
python eval_accuracy.py

echo ""
echo "[DONE] $OUTPUT_DIR 폴더를 확인하세요."
