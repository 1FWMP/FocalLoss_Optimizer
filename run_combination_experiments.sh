#!/usr/bin/env bash
# ===========================================================================
# Stage 2 : Augmentation 조합 히트맵 실험 (resnet50 × CE 고정)
#
#   Stage 1(run_single_aug_search.sh)에서 얻은 기법별 best 파라미터를 고정값으로
#   사용하여, 다음을 각각 1회씩 학습한다.
#     - baseline (무증강)                          : 1 런
#     - 단일 기법 (대각선)                          : 8 런
#     - 두 기법 조합 (상삼각, A+B == B+A 이므로 28쌍): 28 런
#   → 총 37 런. 각 런의 Best Macro F1 을 모아 8x8 히트맵을 만든다.
#
#   history_combo_{tag}.json 이 이미 있으면 SKIP → 중단점 재개 가능.
#   공정 비교를 위해 모든 런은 동일 epoch, pruning 없이 끝까지 학습한다.
#
#   완료 후 analyze_combinations.py 로 히트맵/요약 CSV 를 생성하고,
#   (선택) run_greedy_forward.py 로 최고 pair 부터 greedy forward selection 을 수행.
#
# 사용법:
#   bash run_combination_experiments.sh
#   DATA_DIR=/path/to/data EPOCHS=15 RUN_GREEDY=1 bash run_combination_experiments.sh
# ===========================================================================
set -euo pipefail

DATA_DIR="${DATA_DIR:-./data}"
EPOCHS="${EPOCHS:-15}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
LOSS_TYPE="${LOSS_TYPE:-ce}"
BACKBONE="${BACKBONE:-resnet50}"
PARAMS_JSON="${PARAMS_JSON:-${OUTPUT_DIR}/optuna_best_per_aug.json}"
RUN_GREEDY="${RUN_GREEDY:-1}"          # 1이면 히트맵 후 greedy 수행
GREEDY_MARGIN="${GREEDY_MARGIN:-0.005}"

AUGS=(crop rotate colorjitter blur avgblur sobel noise cutout)

mkdir -p "$OUTPUT_DIR"

if [ ! -f "$PARAMS_JSON" ]; then
    echo "[ERROR] 기법별 파라미터 JSON 이 없습니다: $PARAMS_JSON"
    echo "        먼저 'bash run_single_aug_search.sh' 를 실행하세요."
    exit 1
fi

echo ""
echo "############################################################"
echo "#  Stage 2 : 조합 히트맵 실험"
echo "#    backbone : ${BACKBONE} / ${LOSS_TYPE}"
echo "#    params   : ${PARAMS_JSON}"
echo "#    epochs   : ${EPOCHS} (pruning off)"
echo "############################################################"

# run_combo <tag> <comma-list>  : history 있으면 SKIP
run_combo() {
    local tag=$1; shift
    local combo=$1
    local log="${OUTPUT_DIR}/history_combo_${tag}.json"
    echo ""
    echo "----- combo: ${tag} -----"
    if [ -f "$log" ]; then
        echo "[SKIP] $log"
        return
    fi
    python main.py \
        --combo_augs      "$combo" \
        --aug_params_json "$PARAMS_JSON" \
        --data_dir        "$DATA_DIR" \
        --backbone        "$BACKBONE" \
        --loss_type       "$LOSS_TYPE" \
        --epochs          "$EPOCHS" \
        --batch_size      "$BATCH_SIZE" \
        --lr              "$LR" \
        --num_workers     "$NUM_WORKERS" \
        --output_dir      "$OUTPUT_DIR"
}

# --- baseline (무증강) ---
run_combo base ""

# --- 단일 기법 (대각선) ---
for a in "${AUGS[@]}"; do
    run_combo "$a" "$a"
done

# --- 두 기법 조합 (상삼각) ---
n=${#AUGS[@]}
for ((i = 0; i < n; i++)); do
    for ((j = i + 1; j < n; j++)); do
        a="${AUGS[i]}"; b="${AUGS[j]}"
        run_combo "${a}_${b}" "${a},${b}"
    done
done

# ========================================================================
# 분석 (히트맵 + 요약)
# ========================================================================
echo ""
echo "=========================================="
echo "  히트맵/요약 생성"
echo "=========================================="
python analyze_combinations.py --output_dir "$OUTPUT_DIR"

# ========================================================================
# (선택) Stage 3 : greedy forward selection
# ========================================================================
if [ "$RUN_GREEDY" = "1" ]; then
    echo ""
    echo "=========================================="
    echo "  Stage 3 : greedy forward selection"
    echo "=========================================="
    python run_greedy_forward.py \
        --data_dir        "$DATA_DIR" \
        --output_dir      "$OUTPUT_DIR" \
        --params_json     "$PARAMS_JSON" \
        --backbone        "$BACKBONE" \
        --loss_type       "$LOSS_TYPE" \
        --epochs          "$EPOCHS" \
        --batch_size      "$BATCH_SIZE" \
        --lr              "$LR" \
        --num_workers     "$NUM_WORKERS" \
        --margin          "$GREEDY_MARGIN"
fi

echo ""
echo "[DONE] 결과: ${OUTPUT_DIR}/plots/combination_heatmap.png , ${OUTPUT_DIR}/combination_summary.csv"
