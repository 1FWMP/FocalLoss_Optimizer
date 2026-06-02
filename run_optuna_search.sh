#!/usr/bin/env bash
# ===========================================================================
# Optuna 데이터 증강 탐색 (resnet50 고정)
#
#   8가지 증강 기법(Crop / Cutout / ColorJitter / Sobel / Noise / Blur /
#   Rotate / Average Blur)의 연속·이산 파라미터 공간을 TPE 베이지안 최적화로 탐색.
#   각 trial 은 resnet50 을 학습하고 Best Macro F1 을 목적함수로 반환한다.
#
# [Resume 동작]
#   - Optuna SQLite storage(기본 sqlite:///optuna_study.db) 를 사용한다.
#   - main.py 가 study 를 load_if_exists=True 로 열고, 이미 COMPLETE 인 trial 수를
#     n_trials 에서 빼서 "남은 횟수"만 실행한다.
#   - 따라서 학습이 도중에 끊겨도 이 스크립트를 다시 실행하면 완료된 trial 은
#     자동으로 건너뛰고(SKIP) 남은 trial 만 이어서 진행한다. (재실행해도 누적 폭주 X)
#   - 처음부터 다시 하려면 optuna_study.db 파일을 지우거나 STUDY_NAME 을 바꾼다.
#
# 사용법:
#   bash run_optuna_search.sh
#   DATA_DIR=/path/to/data N_TRIALS=50 EPOCHS=15 bash run_optuna_search.sh
# ===========================================================================
set -euo pipefail

DATA_DIR="${DATA_DIR:-./data}"
N_TRIALS="${N_TRIALS:-30}"
EPOCHS="${EPOCHS:-15}"          # trial 당 에폭 (pruner 가 부진한 trial 을 조기 종료)
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
LOSS_TYPE="${LOSS_TYPE:-ce}"
BACKBONE="${BACKBONE:-resnet50}"
STORAGE="${STORAGE:-sqlite:///optuna_study.db}"
STUDY_NAME="${STUDY_NAME:-resnet50_ce_aug_search}"

mkdir -p "$OUTPUT_DIR"

echo ""
echo "############################################################"
echo "#  Optuna Augmentation Search"
echo "#    backbone   : ${BACKBONE}"
echo "#    loss_type  : ${LOSS_TYPE}"
echo "#    study_name : ${STUDY_NAME}"
echo "#    storage    : ${STORAGE}"
echo "#    n_trials   : ${N_TRIALS}  (완료분 제외 후 남은 만큼만 실행)"
echo "#    epochs     : ${EPOCHS} / trial"
echo "############################################################"
echo ""

python main.py \
    --optuna \
    --data_dir    "$DATA_DIR" \
    --backbone    "$BACKBONE" \
    --loss_type   "$LOSS_TYPE" \
    --n_trials    "$N_TRIALS" \
    --storage     "$STORAGE" \
    --study_name  "$STUDY_NAME" \
    --epochs      "$EPOCHS" \
    --batch_size  "$BATCH_SIZE" \
    --lr          "$LR" \
    --num_workers "$NUM_WORKERS" \
    --output_dir  "$OUTPUT_DIR"

echo ""
echo "[DONE] 탐색 완료. best params: ${OUTPUT_DIR}/optuna_best_${STUDY_NAME}.json"
echo "[TIP]  중단되었다면 같은 명령으로 재실행하면 남은 trial 부터 이어집니다."
