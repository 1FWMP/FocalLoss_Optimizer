#!/usr/bin/env bash
# ===========================================================================
# 최종 실험 : ResNet-50 × (무증강 vs rotate+blur) × (CE vs CB-Focal γ-sweep)
#
#   4단계 augmentation 탐색에서 고른 best 증강(rotate+blur, README 7.9)을 정식
#   예산(30 epoch)에서 재검증하면서, 프로젝트 핵심 가설(CE vs CB-Focal)과
#   focal focusing(gamma) 민감도를 함께 본다.
#
#   설계 (2 × 5 × 3 = 30런):
#     - Augmentation : base(무증강) / rotate+blur
#     - Loss 설정    : CE  +  CB-Focal(beta=0.999) × gamma∈{0,1,2,5}
#     - Seed         : 42 / 43 / 44   (에러바용 반복)
#
#   * 증강은 transforms.build_controlled_transforms(=combo 모드)로 통제하여
#     base 와 rotate+blur 가 동일 파이프라인에서 증강만 차이나게 한다.
#   * 증강 강도는 stage1 결과(outputs/optuna_best_per_aug.json)에서 자동 로드.
#   * history_{run_name}.json 이 있으면 SKIP → 중단되어도 재실행하면 이어서 진행.
#   * 결과는 기존 outputs/ 와 섞이지 않도록 별도 outputs_final/ 에 저장.
#
# 사용법:
#   bash run_final_experiment.sh
#   DATA_DIR=/path/to/data bash run_final_experiment.sh
#   # 장시간(30런 × 30 epoch) → 백그라운드 권장 (README 7.8):
#   DATA_DIR=/path nohup bash run_final_experiment.sh > run_final.log 2>&1 &
#   tail -f run_final.log
# ===========================================================================
set -euo pipefail

DATA_DIR="${DATA_DIR:-./data}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs_final}"
BACKBONE="${BACKBONE:-resnet50}"
BETA="${BETA:-0.999}"
PARAMS_JSON="${PARAMS_JSON:-./outputs/optuna_best_per_aug.json}"
SEEDS="${SEEDS:-42 43 44}"
GAMMAS="${GAMMAS:-0 1 2 5}"
RUN_ANALYSIS="${RUN_ANALYSIS:-1}"   # 1이면 30런 종료 후 집계 스크립트 자동 실행

mkdir -p "$OUTPUT_DIR"

if [ ! -f "$PARAMS_JSON" ]; then
    echo "[ERROR] 증강 강도 파라미터 JSON 이 없습니다: $PARAMS_JSON"
    echo "        먼저 stage1(run_single_aug_search.sh)을 돌려 optuna_best_per_aug.json 을 만드세요."
    exit 1
fi

echo "############################################################"
echo "#  최종 실험 : ${BACKBONE} × {base, rotate+blur} × {CE, CBF γ∈{${GAMMAS}}}"
echo "#    seeds    = ${SEEDS}"
echo "#    epochs   = ${EPOCHS} / run,  총 $(echo $SEEDS | wc -w) × 2 × (1+$(echo $GAMMAS | wc -w)) 런"
echo "#    params   = ${PARAMS_JSON}"
echo "#    output   = ${OUTPUT_DIR}"
echo "############################################################"

# run_one <run_name> <combo_augs> [extra main.py args ...]   history 있으면 SKIP
run_one() {
    local run_name=$1
    local combo=$2
    shift 2
    local log="${OUTPUT_DIR}/history_${run_name}.json"
    echo ""
    echo "==================== ${run_name} ===================="
    if [ -f "$log" ]; then
        echo "[SKIP] 이미 완료: $log"
        return
    fi
    python main.py \
        --combo_augs      "$combo" \
        --aug_params_json "$PARAMS_JSON" \
        --backbone        "$BACKBONE" \
        --data_dir        "$DATA_DIR" \
        --epochs          "$EPOCHS" \
        --batch_size      "$BATCH_SIZE" \
        --lr              "$LR" \
        --num_workers     "$NUM_WORKERS" \
        --output_dir      "$OUTPUT_DIR" \
        --run_name        "$run_name" \
        "$@"
}

for seed in $SEEDS; do
    for aug_tag in noaug rotblur; do
        if [ "$aug_tag" = "noaug" ]; then
            combo="base"
        else
            combo="rotate,blur"
        fi

        # --- CE 기준선 (gamma 무관) ---
        run_one "${BACKBONE}_ce_${aug_tag}_s${seed}" "$combo" \
            --loss_type ce --seed "$seed"

        # --- CB-Focal : beta 고정, gamma 스윕 ---
        for g in $GAMMAS; do
            run_one "${BACKBONE}_cbf_g${g}_${aug_tag}_s${seed}" "$combo" \
                --loss_type cb_focal --beta "$BETA" --gamma "$g" --seed "$seed"
        done
    done
done

echo ""
echo "[DONE] 30런 완료. history/best_model → ${OUTPUT_DIR}"

if [ "$RUN_ANALYSIS" = "1" ]; then
    echo ""
    echo "=========================================="
    echo "  결과 집계 (analyze_final_experiment.py)"
    echo "=========================================="
    python analyze_final_experiment.py --output_dir "$OUTPUT_DIR"
fi
