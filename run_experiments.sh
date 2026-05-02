#!/usr/bin/env bash
# 6번의 실험을 순차 실행한 뒤 결과를 분석한다.
# history_*.json 이 이미 존재하는 실험은 건너뛴다.
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

run_exp() {
    local backbone=$1
    local loss_type=$2
    shift 2
    local log_file="$OUTPUT_DIR/history_${backbone}_${loss_type}.json"

    echo ""
    echo "=========================================="
    echo "  backbone  : $backbone"
    echo "  loss_type : $loss_type"
    echo "=========================================="

    if [ -f "$log_file" ]; then
        echo "  [SKIP] 이미 완료된 실험입니다: $log_file"
        return
    fi

    python main.py \
        --data_dir    "$DATA_DIR" \
        --backbone    "$backbone" \
        --loss_type   "$loss_type" \
        --epochs      "$EPOCHS" \
        --batch_size  "$BATCH_SIZE" \
        --lr          "$LR" \
        --num_workers "$NUM_WORKERS" \
        --output_dir  "$OUTPUT_DIR" \
        "$@"
}

# ------------------------------------------------------------------
# Phase 1: Backbone 비교 (CE 고정, 변수 분리)
# ------------------------------------------------------------------
run_exp efficientnet_b3       ce
run_exp resnet50               ce
run_exp densenet121            ce
run_exp mobilenetv3_large_100  ce

# ------------------------------------------------------------------
# Phase 2: 메인 실험 (최적 backbone × CE vs CB-Focal)
# ------------------------------------------------------------------
run_exp efficientnet_b3 cb_focal --beta 0.999 --gamma 2.0
run_exp densenet121     cb_focal --beta 0.999 --gamma 2.0

# ------------------------------------------------------------------
# 결과 분석
# ------------------------------------------------------------------
echo ""
echo "=========================================="
echo "  모든 실험 완료. 결과 분석 중..."
echo "=========================================="
python analyze_results.py --output_dir "$OUTPUT_DIR"

echo ""
echo "[DONE] $OUTPUT_DIR 폴더를 BackendAI 파일 매니저에서 다운로드하세요."
