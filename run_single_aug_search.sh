#!/usr/bin/env bash
# ===========================================================================
# Stage 1 : 단독 augmentation 튜닝 (resnet50 × CE 고정)
#
#   8가지 증강 기법을 "하나씩" 단독으로 켜고(나머지는 OFF, 매 이미지 적용 prob=1.0),
#   그 기법의 강도 파라미터만 Optuna(TPE)+MedianPruner 로 탐색한다.
#   → 기법별 "고유 최적 강도" 를 얻는다.
#
#   각 기법은 독립 study(`singleaug_{aug}`)를 가지므로 중단되어도 재실행하면
#   완료된 기법/trial 은 건너뛰고 이어서 진행한다. (SQLite storage 공유)
#
#   결과: outputs/optuna_best_per_aug.json  (기법별 best 파라미터 누적)
#         → 이후 run_combination_experiments.sh 의 입력이 된다.
#
# 사용법:
#   bash run_single_aug_search.sh
#   DATA_DIR=/path/to/data N_TRIALS=12 EPOCHS=15 bash run_single_aug_search.sh
# ===========================================================================
set -euo pipefail

DATA_DIR="${DATA_DIR:-./data}"
N_TRIALS="${N_TRIALS:-12}"
EPOCHS="${EPOCHS:-15}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
LOSS_TYPE="${LOSS_TYPE:-ce}"
BACKBONE="${BACKBONE:-resnet50}"
STORAGE="${STORAGE:-sqlite:///optuna_singleaug.db}"

# 히트맵 축 순서와 동일 (transforms.AUG_NAMES)
AUGS=(crop rotate colorjitter blur avgblur sobel noise cutout)

mkdir -p "$OUTPUT_DIR"

echo ""
echo "############################################################"
echo "#  Stage 1 : 단독 augmentation 튜닝"
echo "#    backbone  : ${BACKBONE} / ${LOSS_TYPE}"
echo "#    augs      : ${AUGS[*]}"
echo "#    n_trials  : ${N_TRIALS} / aug,  epochs ${EPOCHS}"
echo "#    storage   : ${STORAGE}"
echo "############################################################"

for aug in "${AUGS[@]}"; do
    echo ""
    echo "=========================================="
    echo "  단독 튜닝: ${aug}"
    echo "=========================================="
    python main.py \
        --search_aug  "$aug" \
        --data_dir    "$DATA_DIR" \
        --backbone    "$BACKBONE" \
        --loss_type   "$LOSS_TYPE" \
        --n_trials    "$N_TRIALS" \
        --epochs      "$EPOCHS" \
        --batch_size  "$BATCH_SIZE" \
        --lr          "$LR" \
        --num_workers "$NUM_WORKERS" \
        --storage     "$STORAGE" \
        --output_dir  "$OUTPUT_DIR"
done

echo ""
echo "[DONE] Stage 1 완료. 기법별 best: ${OUTPUT_DIR}/optuna_best_per_aug.json"
echo "[NEXT] bash run_combination_experiments.sh"
