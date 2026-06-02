#!/usr/bin/env bash
# ===========================================================================
# Augmentation 방법론 4단계를 한 번에 순차 실행하는 래퍼.
#
#   stage 1 : run_single_aug_search.sh        (8기법 단독 튜닝)
#   stage 2 : run_combination_experiments.sh  (조합 히트맵 + greedy)
#   stage 4 : run_optuna_search.sh            (joint 검증 baseline)
#
#   각 스크립트는 history/SQLite 기반으로 완료분을 SKIP 하므로, 중간에 끊겨도
#   이 래퍼를 다시 실행하면 이어서 진행한다.
#
#   환경변수(DATA_DIR, EPOCHS, N_TRIALS, BATCH_SIZE 등)는 prefix 로 한 번만 주면
#   하위 세 스크립트에 모두 그대로 전달된다.
#
# 사용법:
#   bash run_all_aug.sh
#   DATA_DIR=/path/to/data EPOCHS=15 N_TRIALS=12 bash run_all_aug.sh
#
#   장시간(예상 ~20h+) 작업이므로 백그라운드 실행 권장:
#   DATA_DIR=/path/to/data nohup bash run_all_aug.sh > run_all_aug.log 2>&1 &
#   tail -f run_all_aug.log
# ===========================================================================
set -euo pipefail

echo "############################################################"
echo "#  Augmentation 4단계 일괄 실행"
echo "#    DATA_DIR  = ${DATA_DIR:-./data}"
echo "#    EPOCHS    = ${EPOCHS:-(스크립트 기본값)}"
echo "#    N_TRIALS  = ${N_TRIALS:-(스크립트 기본값)}"
echo "############################################################"

echo ""; echo ">>> [1/3] stage 1 : 단독 튜닝"
bash run_single_aug_search.sh

echo ""; echo ">>> [2/3] stage 2+3 : 조합 히트맵 + greedy"
bash run_combination_experiments.sh

echo ""; echo ">>> [3/3] stage 4 : joint 검증 baseline"
bash run_optuna_search.sh

echo ""
echo "############################################################"
echo "#  전체 완료"
echo "#    히트맵   : outputs/plots/combination_heatmap.png"
echo "#    조합요약 : outputs/combination_summary.csv"
echo "#    greedy   : outputs/greedy_path.json"
echo "#    기법별   : outputs/optuna_best_per_aug.json"
echo "#    joint    : outputs/optuna_best_resnet50_ce_aug_search.json"
echo "############################################################"
