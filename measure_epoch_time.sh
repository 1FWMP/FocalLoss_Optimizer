#!/usr/bin/env bash
# ===========================================================================
# resnet50 epoch당 시간 측정 헬퍼
#
#   실험 총 소요시간을 추정하기 위해 1~2 epoch 만 돌려 "time=...s" 를 확인한다.
#   Optuna 탐색 모드(--optuna)로 돌리므로 transforms.py 의 커스텀 증강이 모두
#   활성화된 **가장 무거운 워크로드(상한)** 를 측정한다.
#   (단독 튜닝/조합 실험은 활성 증강이 1~2개뿐이라 보통 이보다 빠르다.)
#
#   출력 로그에서 두 번째 epoch 의 `time=XX.Xs` 가 정상상태(steady-state)
#   epoch당 학습+검증 시간이다. (첫 epoch 은 worker spin-up 등 오버헤드 포함)
#
# 총 시간 추정:
#   총 시간(h) ≈ (epoch당 초) × (플랜별 총 epoch 수) / 3600
#       1번 joint Optuna      : 약 270 epoch
#       2번 단독→조합          : 약 1330 epoch
#       하이브리드(1+2)        : 약 1600 epoch
#
# 사용법:
#   bash measure_epoch_time.sh
#   DATA_DIR=/path/to/data BATCH_SIZE=32 bash measure_epoch_time.sh
# ===========================================================================
set -euo pipefail

DATA_DIR="${DATA_DIR:-./data}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCHS="${EPOCHS:-2}"   # 2 epoch: 두 번째 값이 steady-state

TMP_DIR="./outputs/_timing"
TMP_DB="timing_tmp.db"

mkdir -p "$TMP_DIR"

echo "############################################################"
echo "#  resnet50 epoch 타이밍 측정 (전체 증강 활성 = 상한)"
echo "#    batch_size  : ${BATCH_SIZE}"
echo "#    num_workers : ${NUM_WORKERS}"
echo "#    epochs      : ${EPOCHS}  (두 번째 epoch 의 time= 값을 보세요)"
echo "############################################################"
echo ""

python main.py \
    --optuna \
    --data_dir    "$DATA_DIR" \
    --backbone    resnet50 \
    --loss_type   ce \
    --n_trials    1 \
    --epochs      "$EPOCHS" \
    --batch_size  "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --storage     "sqlite:///${TMP_DB}" \
    --study_name  timing_test \
    --output_dir  "$TMP_DIR"

echo ""
echo "[CLEANUP] 임시 파일 삭제 (측정용 db / 체크포인트)"
rm -rf "$TMP_DIR"
rm -f "$TMP_DB"

echo ""
echo "[DONE] 위 로그의 두 번째 epoch 'time=XX.Xs' 가 epoch당 시간입니다."
echo "       총 시간(h) ≈ (epoch당 초) × (270=1번 / 1330=2번 / 1600=하이브리드) / 3600"
