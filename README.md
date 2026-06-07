# FocalLoss_Optimizer

HAM10000 (피부 병변, 7-class) 미세 분류 실험.
**Cross-Entropy** vs **Class-Balanced Focal Loss (Cui et al., 2019)** 의 성능을
다양한 backbone 과 augmentation 조합 위에서 비교한다.

## 1. 환경 세팅

### Conda (권장 — 팀 공용)
```bash
conda env create -f environment.yml
conda activate focal-ham10000
```

> **참고** — `environment.yml` 은 **conda 로는 파이썬만 깔고 PyTorch 는 pip
> wheel** 로 설치하도록 구성돼 있다. conda 채널의 `pytorch-cuda` 가 워낙 커서
> 한국 네트워크에서 `IncompleteRead` 로 끊기는 문제를 피하기 위함.
>
> 다른 CUDA 버전이 필요하면 `--index-url` 을 바꿔주면 된다.
> - CUDA 11.8: `https://download.pytorch.org/whl/cu118`
> - CPU only : 인덱스 줄을 모두 지우고 `torch==2.2.2 / torchvision==0.17.2` 만 남기면 PyPI CPU 빌드가 받아진다.

### pip만 사용할 경우
```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 다운로드가 자주 끊길 때 (Windows / 사내망)
```bash
conda config --set remote_read_timeout_secs 600
conda config --set remote_connect_timeout_secs 60
# 끊긴 env 잔재가 남아있으면 지우고 다시 생성
conda env remove -n focal-ham10000
conda env create -f environment.yml
```

> 경로에 한글이 들어 있어 `cp949 codec can't decode...` 경고가 보일 수 있는데,
> conda 의 explicit-spec 플러그인이 파일명을 cp949 로 읽으려다 실패한 것일 뿐
> 실제 YAML 파싱은 정상 진행되므로 **무시해도 된다**.

## 2. 데이터 다운로드

Kaggle 미러(`kmader/skin-cancer-mnist-ham10000`) 를 사용한다.
최초 1회는 Kaggle 인증이 필요할 수 있으니, `~/.kaggle/kaggle.json` 또는
`kagglehub login` 을 먼저 수행해 주세요.

```bash
python download_data.py --data_dir ./data
```

완료 후 디렉토리 구조:
```
data/
  HAM10000_metadata.csv
  images/
    ISIC_0024306.jpg
    ...
```

## 3. 학습 실행

### 전체 실험 자동 실행 (권장)
```bash
# Step 1(ResNet depth 비교) → Step 2(aug 8 조합) → 결과 분석 자동 수행
bash run_experiments.sh
# 데이터 경로가 다를 경우:
DATA_DIR=/path/to/data bash run_experiments.sh
```

### 개별 실행 예시

#### Cross-Entropy 베이스라인
```bash
python main.py --data_dir ./data --backbone resnet101 --loss_type ce \
    --epochs 30 --batch_size 32 --lr 1e-4
```

#### Augmentation 조합 (CutMix + Elastic + ColorJitter)
```bash
python main.py --data_dir ./data --backbone resnet101 --loss_type ce \
    --use_cutmix --use_elastic --use_colorjitter \
    --run_name resnet101_ce_cm_el_cj \
    --epochs 30 --batch_size 32 --lr 1e-4
```

#### Class-Balanced Focal Loss
```bash
python main.py --data_dir ./data --backbone densenet121 --loss_type cb_focal \
    --beta 0.999 --gamma 2.0 \
    --epochs 30 --batch_size 32 --lr 1e-4
```

### Optuna 증강 탐색 (resnet50 × CE 고정)

8가지 증강 기법(Crop / Cutout / ColorJitter / Sobel / Noise / Blur / Rotate /
Average Blur)의 **연속·이산 파라미터 공간**을 TPE 베이지안 최적화로 탐색한다.
`MedianPruner` 로 부진한 trial 을 조기 종료하고, SQLite storage 로 중단점 재개를 지원한다.

```bash
# 권장: 쉘 스크립트 (trial 당 15 에폭 기본)
bash run_optuna_search.sh
DATA_DIR=/path/to/data N_TRIALS=50 EPOCHS=15 bash run_optuna_search.sh

# 직접 호출
python main.py --optuna --data_dir ./data --backbone resnet50 --loss_type ce \
    --n_trials 30 --epochs 15 \
    --storage sqlite:///optuna_study.db --study_name resnet50_ce_aug_search
```

> **Resume** — 학습이 도중에 끊겨도 같은 명령을 다시 실행하면 이미 완료된 trial 은
> 자동으로 건너뛰고 남은 횟수만큼 이어서 탐색한다(`load_if_exists=True` +
> `remaining = n_trials − 완료 trial 수`). 처음부터 다시 하려면 `optuna_study.db`
> 를 삭제하거나 `--study_name` 을 바꾼다.
>
> 탐색 종료 후 best 파라미터는 `outputs/optuna_best_{study_name}.json` 에 저장된다.

매 에포크마다 다음을 출력한다.
- `train_loss`, `val_loss`
- 클래스별 F1 + **Macro F1** (`sklearn.classification_report`)

Best Macro F1 갱신 시 `outputs/best_model_{run_name}.pth` 로 저장되며,
학습 이력은 `outputs/history_{run_name}.json` 에 기록된다.

## 4. CLI 인자 요약

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--data_dir` | (필수) | 메타데이터/이미지 루트 디렉토리 |
| `--backbone` | `resnet50` | 사용할 backbone (아래 목록 참고) |
| `--loss_type` | `ce` | `ce` 또는 `cb_focal` |
| `--epochs` | 30 | 학습 에포크 수 |
| `--batch_size` | 32 | 배치 사이즈 |
| `--lr` | 1e-4 | 학습률 (AdamW) |
| `--beta` | 0.999 | CB Loss 의 beta |
| `--gamma` | 2.0 | Focal Loss 의 gamma |
| `--run_name` | `{backbone}_{loss_type}` | 실험 고유 이름 (파일 저장 prefix) |
| `--use_cutmix` | False | CutMix augmentation 적용 (batch-level) |
| `--use_elastic` | False | Elastic Transform 적용 |
| `--use_colorjitter` | False | Color Jitter 적용 |
| `--optuna` | False | Optuna 증강 파라미터 탐색 모드로 실행 |
| `--n_trials` | 30 | Optuna 총 trial 횟수 (완료분 제외, 남은 만큼만 진행) |
| `--storage` | `sqlite:///optuna_study.db` | Optuna storage (중단점 재개용) |
| `--study_name` | `resnet50_ce_aug_search` | Optuna study 식별 이름 |
| `--val_size` | 0.2 | 검증 비율 (lesion 단위 stratified split) |
| `--seed` | 42 | 재현성 시드 |
| `--num_workers` | 4 | DataLoader 워커 수 |
| `--output_dir` | `./outputs` | 모델/로그 저장 위치 |

## 5. 프로젝트 구조

```
FocalLoss_Optimizer/
├── main.py            # 학습/검증 엔트리포인트 (단일 학습 + --optuna 탐색 모드)
├── dataset.py         # HAM10000Dataset, 이미지 경로 탐색, 메타 로딩
├── transforms.py      # 커스텀 증강(Sobel/Noise/AverageBlur) + build_transforms(aug_params)
├── losses.py          # CBFocalLoss + build_loss factory
├── model.py           # SUPPORTED_BACKBONES + build_model factory (timm)
├── eval_accuracy.py   # 저장된 .pth 일괄 평가 (Accuracy / F1 / Precision / Recall)
├── analyze_results.py # summary.csv + 학습 곡선 시각화
├── run_experiments.sh # Step1(ResNet depth) → Step2(aug 8조합) 자동 실행
├── run_optuna_search.sh # joint Optuna 탐색 (8기법 동시, stage4 검증 baseline)
├── run_single_aug_search.sh    # stage1: 8기법 단독 튜닝
├── run_combination_experiments.sh # stage2: 조합 히트맵 + stage3 greedy
├── run_greedy_forward.py       # stage3: greedy forward selection
├── run_all_aug.sh              # stage1→2(+3)→4 일괄 실행 래퍼
├── analyze_combinations.py     # 8x8 히트맵 + combination_summary.csv
├── measure_epoch_time.sh       # epoch당 시간 측정 헬퍼
├── download_data.py   # Kaggle 미러에서 HAM10000 받기
├── environment.yml    # conda 환경
├── requirements.txt   # pip 의존성
└── outputs/           # 학습 결과
    ├── best_model_{run_name}.pth
    ├── history_{run_name}.json
    ├── summary.csv
    └── plots/
```

### 지원 backbone

| 키 | 비고 |
| --- | --- |
| `efficientnet_b3` | Phase 1·2 실험 대상 |
| `resnet50` | Phase 1 실험 대상 |
| `resnet101` | Step 1 실험 대상 |
| `resnet152` | Step 1 실험 대상 |
| `densenet121` | Phase 1·2 최고 성능 (Macro F1 0.7817) |
| `mobilenetv3_large_100` | Phase 1 실험 대상 |

## 6. 구현 노트

- **Class-Balanced Focal Loss**: `losses.py:CBFocalLoss`
  - Effective Number of Samples: `E_n = (1 - β^n_c) / (1 - β)`
  - 클래스 가중치: `α_c = 1/E_n` 후 합 = `num_classes` 가 되도록 정규화
  - `log_softmax + clamp(eps=1e-7)` 으로 수치 안정성 확보
- **CutMix**: `main.py:cutmix_batch`
  - Beta(1,1) 분포에서 λ 샘플링 → 랜덤 패치 교환 → 실제 픽셀 비율로 λ 재계산
  - loss = λ·criterion(logits, labels_a) + (1−λ)·criterion(logits, labels_b) 로 CE/CB-Focal 모두 호환
  - 50% 확률로 적용 (`--use_cutmix` 플래그로 활성화)
- **Augmentation 조합 실험**: `run_experiments.sh` Step 2
  - Baseline(Resize+Crop+Flip) 위에 CutMix / ElasticTransform / ColorJitter 각각 on/off → 2³=8 조합
  - ElasticTransform: `torchvision.transforms.ElasticTransform(alpha=50.0, sigma=5.0)`
  - ColorJitter: brightness/contrast/saturation ±0.2, hue ±0.1
- **데이터 누수 방지**: 동일 `lesion_id` 의 이미지는 train/val 한쪽에만 들어가도록
  lesion 단위 stratified split 을 수행 (`main.py:split_train_val`).
- **평가 지표**: 데이터 불균형이 심해 정확도는 다수 클래스(`nv`) 에 끌려가므로
  **Macro F1** 을 best 모델 선정 기준으로 사용한다.
- **평가 스크립트** (`eval_accuracy.py`): 전체 Accuracy / Macro F1 / Macro Precision / Macro Recall
  + 클래스별(akiec~vasc) Accuracy / F1 / Precision / Recall 을 테이블 형태로 출력.

## 7. Augmentation 실험 방법론 (단독 튜닝 → 조합 히트맵 → greedy)

> 이 섹션은 "어떤 데이터 증강을, 어떤 세기로, 어떻게 조합해야 Macro F1 이 가장
> 좋아지는가" 를 **체계적으로** 규명하기 위한 4단계 실험의 설계·논리·해석 방법을
> 한곳에 정리한 것이다. 나중에 보고서를 쓰거나 결과를 다시 읽을 때 이 섹션만 보면
> 전체 그림과 각 선택의 근거를 이해할 수 있도록 작성했다.

### 7.0 핵심 아이디어 한 줄 요약

각 증강 기법의 **고유 최적 세기**를 먼저 따로 찾고(stage 1), 그 값을 고정한 채
**기법들을 짝지어 상호작용**을 본 뒤(stage 2, 8×8 히트맵), 좋은 조합에서 출발해
**한 개씩 더해보며 누적**(stage 3, greedy)한다. 마지막으로 8개를 한꺼번에 최적화한
**joint Optuna(stage 4)** 와 비교해 우리 구조적 결론이 black-box 전역 최적화와
얼마나 부합하는지 검증한다.

### 7.1 다루는 8가지 증강 기법

| 이름(키) | 구현 | 단독 튜닝 파라미터 | 비고 |
| --- | --- | --- | --- |
| `crop` | `RandomResizedCrop(scale=(s,1.0))` | `crop_scale_min` ∈ [0.5, 1.0] | off 면 `Resize+CenterCrop` |
| `rotate` | `RandomRotation(±deg)` | `rotate_deg` ∈ [0, 180] | |
| `colorjitter` | `ColorJitter(b/c/s, hue=½·s)` | `colorjitter_strength` ∈ [0.05, 0.5] | |
| `blur` | `GaussianBlur(k=5, sigma)` | `blur_sigma` ∈ [0.1, 3.0] | torchvision 내장 |
| `avgblur` | **커스텀** 평균(박스) 블러 | `avgblur_kernel` ∈ {3,5,7,9} | depthwise conv2d |
| `sobel` | **커스텀** Sobel 엣지맵 치환 | `sobel_prob` ∈ [0.05, 0.5] | 강도 개념이 없어 *확률* 이 파라미터 |
| `noise` | **커스텀** Gaussian Noise | `noise_std` ∈ [1e-3, 0.1] (log) | |
| `cutout` | `RandomErasing(scale=(0.02,c))` | `cutout_scale` ∈ [0.05, 0.4] | |

커스텀 3종(`sobel/noise/avgblur`)은 `transforms.py` 에 구현되어 있고 입력이 PIL/Tensor
어느 쪽이든 동작한다. 파이프라인 순서는 항상
`PIL(Crop→Rotate→ColorJitter→GaussianBlur) → ToTensor → Tensor(AvgBlur→Sobel→Noise)
→ Normalize → Cutout(RandomErasing)` 로 **고정**한다. 순서를 고정하므로 "A+B" 와 "B+A"
는 동일한 파이프라인이 되어 조합이 **대칭**이 된다(히트맵 상삼각만 계산하면 됨).

### 7.2 평가 지표 — 왜 Macro F1 인가

HAM10000 은 `nv`(양성 모반) 가 전체의 ~67% 를 차지하는 **극심한 불균형** 데이터다.
정확도(accuracy) 는 다수 클래스만 잘 맞혀도 높게 나와 소수 클래스(예: `df`, `vasc`)
성능을 가린다. 따라서 **모든 클래스의 F1 을 동일 가중으로 평균한 Macro F1** 을
모든 단계의 단일 비교 지표(= 셀 값, objective return, best 모델 기준) 로 사용한다.

### 7.3 Stage 1 — 단독 튜닝 (`run_single_aug_search.sh`)

- **목적**: 각 기법 *하나*의 고유 최적 세기를 찾는다.
- **방법**: 기법을 하나만 켜고(나머지 7개 OFF), baseline = `Resize+CenterCrop` 위에
  그 기법만 얹어 **항상 적용(prob=1.0)** 한 뒤 **세기 파라미터만** Optuna(TPE)+MedianPruner
  로 탐색한다. 8개 기법 각각 독립 study(`singleaug_{aug}`)를 가진다.
- **prob=1.0 으로 고정하는 이유**: "얼마나 자주(prob)" 와 "얼마나 세게(magnitude)" 가
  섞이면, 탐색이 prob→0 (사실상 미사용) 으로 도망쳐 세기 효과를 못 본다. 매번 적용으로
  고정해 *세기* 만 본다. (단 `sobel` 은 엣지맵으로 통째로 치환하는 파괴적 연산이라 매번
  적용하면 학습이 망가지므로, 예외적으로 *적용 확률* 자체를 파라미터로 둔다.)
  `prob=1.0` 이라도 `RandomRotation/ColorJitter` 등은 내부적으로 매번 랜덤값을 뽑으므로
  augmentation 효과(다양성)는 그대로 유지된다.
- **산출물**: `outputs/optuna_best_per_aug.json` — 기법별 best 파라미터가 누적 저장된다
  (다음 단계의 입력). 중단되어도 재실행하면 완료된 기법/trial 은 건너뛴다(SQLite resume).

### 7.4 Stage 2 — 조합 히트맵 (`run_combination_experiments.sh` → `analyze_combinations.py`)

- **목적**: 기법 간 **상호작용**(시너지/상충) 을 본다.
- **방법**: stage 1 의 기법별 best 파라미터를 **고정**한 채 아래를 각각 1회씩 학습한다.
  - `baseline`(무증강) 1런, **단일 기법** 8런(대각선), **두 기법 조합** 28런(상삼각).
  - 총 37런. 공정 비교를 위해 모두 **동일 epoch, pruning 없이** 끝까지 학습한다.
- **baseline 정의**: `Resize+CenterCrop` 만(flip 도 제외한 순수 무증강). 그래서
  - 대각선 셀 = "그 기법 1개 vs 무증강" 의 마진,
  - 비대각 셀 = "두 기법 동시" 의 성능.
- **결과물**: `outputs/plots/combination_heatmap.png` (8×8, 값 주석·대칭),
  `outputs/combination_summary.csv` (조합별 Best Macro F1 + baseline 대비 Δ, 내림차순).
- **해석법**: 대각선보다 그 행/열의 비대각이 더 높으면 **시너지**, 두 단일보다 조합이
  낮으면 **상충**(과도한 정규화 등). baseline 대비 Δ 로 각 조합의 순수 기여를 읽는다.

### 7.5 Stage 3 — Greedy forward selection (`run_greedy_forward.py`)

- **목적**: 히트맵에서 가장 좋은 pair 를 시작점으로, 기법을 **누적**하면 더 좋아지는지.
- **방법**: 최고 pair 에서 시작 → 남은 기법을 하나씩 추가해보고, **Macro F1 이
  `--margin`(기본 0.005) 이상 개선될 때만** 채택 → 개선 없으면 중단.
- **margin 을 두는 이유**: seed 변동폭 수준의 미세한 향상에 휩쓸려 불필요한 기법을
  쌓는 것을 막는다(강한 기법을 여럿 쌓으면 과정규화로 오히려 하락 가능).
- **산출물**: `outputs/greedy_path.json` (추가 경로와 단계별 F1).

### 7.6 Stage 4 — Joint Optuna 검증 baseline (`run_optuna_search.sh`)

- **목적**: 8개 파라미터를 **한 study 에서 동시에** TPE 로 최적화한 "전역 최적 정책" 과
  stage 1~3 의 구조적 결과를 **맞비교** 한다.
- **해석**: greedy ≈ joint 면 "해석 가능한 단계적 파이프라인이 black-box 최적화와
  맞먹는다"는 강한 결론. joint > greedy 면 "단독최적→조합(greedy 근사)이 놓친 상호작용
  이득" 을 정량화한 것 — 어느 쪽이든 버리는 결과가 아니라 **상호 보완적 발견**이다.
  (단독최적을 합치는 것은 정의상 상호작용을 무시하는 근사이므로 차이는 *예상된 현상*.)

### 7.7 실행 순서 요약

```bash
# (0) epoch당 시간 측정 → 총 소요시간 가늠
bash measure_epoch_time.sh

# (1)~(4) 한 번에: stage1 → stage2(+3) → stage4
bash run_all_aug.sh

# 단계별로 따로 돌리려면:
bash run_single_aug_search.sh        # (1) 단독 튜닝 → outputs/optuna_best_per_aug.json
bash run_combination_experiments.sh  # (2)+(3) 조합 히트맵 + greedy (RUN_GREEDY=1 기본)
bash run_optuna_search.sh            # (4) joint 검증 baseline
```

데이터 경로/예산은 환경변수를 prefix 로 한 번만 주면 하위 스크립트에 모두 전달된다.
장시간(~20h+) 작업이므로 백그라운드 실행을 권장한다.

```bash
DATA_DIR=/path/to/data EPOCHS=15 N_TRIALS=12 bash run_all_aug.sh
```

### 7.8 nohup 으로 백그라운드 실행 (장시간 작업 권장)

터미널/SSH 가 끊겨도 작업이 죽지 않도록 `nohup` + `&` 로 백그라운드 실행한다.

```bash
cd FocalLoss_Optimizer

# 실행 — 환경변수는 nohup 앞에 prefix 로 한 번만 주면 하위 3개 스크립트에 모두 전달된다.
DATA_DIR=/path/to/data EPOCHS=15 N_TRIALS=12 \
    nohup bash run_all_aug.sh > run_all_aug.log 2>&1 &

# 진행 상황 실시간 확인 (Ctrl+C 는 보기만 종료, 작업은 계속 돈다)
tail -f run_all_aug.log

# 돌고 있는지 확인 / PID 찾기
jobs -l                       # 현재 셸에서 띄운 경우
ps -ef | grep run_all_aug     # 재접속 후 등 다른 셸에서

# 중단해야 할 때 (위에서 찾은 PID)
kill <PID>
```

- `nohup` — 터미널 종료(HUP) 신호를 무시해 SSH 접속을 끊고 나가도 계속 돈다.
- `> run_all_aug.log 2>&1` — stdout 과 stderr 를 **모두** 로그 파일로 보낸다(`2>&1` 이 있어야 에러도 같이 기록).
- 끝의 `&` — 백그라운드로 띄운다.
- 중간에 끊겨도 같은 명령을 다시 실행하면 history/SQLite 기반으로 완료분은 SKIP 하고 이어서 진행한다.

### 7.9 실험 결과 — 최종 증강 조합 선정 (resnet50 × CE × 15 epoch)

> `outputs/` 는 `.gitignore` 대상이라 결과 파일이 저장소에 남지 않으므로, 4단계
> 실험의 **실측 수치와 그로부터 도출한 최종 결론**을 여기에 박제해 둔다.
> (재현하려면 `bash run_all_aug.sh` 후 `outputs/combination_summary.csv`,
> `outputs/greedy_path.json`, `outputs/optuna_best_*.json` 을 다시 확인하면 된다.)

#### 결론 한 줄

**최종 증강 = `rotate + blur` (회전 + 가우시안 블러)**
`rotate_deg ≈ 108.2°`, `blur_sigma ≈ 0.268` (GaussianBlur k=5),
**Macro F1 = 0.6881** — 무증강 baseline 0.5929 대비 **+0.0953 (약 +16%)** 로 전체 37런 중 1위.

```bash
# 최종 학습 명령 (강도는 optuna_best_per_aug.json 에서 자동 로드)
python main.py --combo_augs "rotate,blur" \
    --aug_params_json ./outputs/optuna_best_per_aug.json \
    --data_dir ./data --backbone resnet50 --loss_type ce \
    --epochs 30 --batch_size 32 --lr 1e-4
```

#### 선정 근거 — 4단계 교차검증

모든 비교 지표는 **Macro F1**. 기준점은 `history_combo_base.json` = **무증강 0.5929**.

**① Stage 1 단독 기법 순위** (`optuna_best_per_aug.json`)

| 기법 | 단독 Macro F1 | Δ vs baseline | 판정 |
|---|---|---|---|
| **rotate** | 0.6704 | **+0.078** | 압도적 1위 |
| cutout | 0.6443 | +0.051 | 유효 |
| sobel | 0.6351 | +0.042 | 유효 |
| crop | 0.6097 | +0.017 | 약하게 유효 |
| noise | 0.6072 | +0.014 | 약하게 유효 |
| blur | 0.5929 | +0.000 | 단독으론 무의미 (=baseline) |
| colorjitter | 0.5908 | −0.002 | 무해하나 무익 |
| avgblur | 0.4741 | **−0.119** | **유해 (치명적)** → 제외 |

**② Stage 2 조합 히트맵** (`combination_summary.csv`) — 상위가 전부 rotate 포함

| 순위 | 조합 | Macro F1 | Δ |
|---|---|---|---|
| **1** | **rotate+blur** | **0.6881** | **+0.095** |
| 2 | crop+rotate | 0.6801 | +0.087 |
| 3 | rotate (단독) | 0.6704 | +0.078 |
| 4 | rotate+noise | 0.6653 | +0.072 |

- **blur 시너지**: blur 는 단독이면 baseline 과 동일(Δ0)인데 rotate 와 합치면 rotate
  단독(0.6704)보다 **+0.0177** 더 오른다. 혼자선 무익하지만 rotate 의 기하학적
  다양성 위에서 **약한 정규화**로 작동하는 전형적 상호작용 → 히트맵 rotate 행의
  blur 셀이 대각선보다 밝은 것이 증거.

**③ Stage 3 greedy** (`greedy_path.json`) — rotate+blur 에서 시작, margin 0.005 개선
없어 즉시 중단. 3중 조합은 무엇을 더해도 전부 하락:

```
rotate+blur (기준)        0.68813
  + crop   → 0.67978  (−0.008)   + colorjitter → 0.64924  (−0.039)
  + noise  → 0.63749  (−0.051)   + sobel       → 0.63517  (−0.053)
  + cutout → 0.61220  (−0.076)   + avgblur     → 0.57690  (−0.111)
```
→ 강한 증강을 더 쌓으면 과정규화로 손해. **2개에서 멈추는 것이 최적.**

**④ Stage 4 joint Optuna 검증** (`optuna_best_resnet50_ce_aug_search.json`)
joint best = **0.6108** 로 greedy(0.6881)보다 낮다. 단 `n_completed=12` —
**12차원 공간을 12 trial 로** 탐색해 TPE 가 수렴 못 했고, best_params 가
`avgblur_prob=0.94`(가장 유해한 기법을 켬)·`blur_sigma≈1.0`(과한 블러) 등 나쁜
영역에 갇혔다. 즉 구조적 결론이 틀린 게 아니라 **joint 의 예산 부족**(README 7.6 의
"greedy > joint" 케이스)일 뿐 — 오히려 단계적 탐색이 더 적은 예산으로 더 나은 해를
찾았다는 근거.

#### 종합 논리

1. **단독 최강(rotate) + 검증된 시너지(blur)** — 4개 분석 모두 rotate 를 최상위로 지목.
2. rotate+blur 는 **전체 37런의 전역 최댓값**이자 greedy 종착점 (두 방법이 독립적으로 일치).
3. **3중 조합 전부 하락** → "2개가 최적" 을 데이터로 확정.
4. 유해 기법(avgblur −0.12, colorjitter −0.002)은 명확히 배제.

#### ⚠️ 한계 (보고서에 명시)

본 탐색은 **resnet50 × CE × 15 epoch** 조건이다. 최종 backbone 이 Phase 1 우승자
**DenseNet-121** 이라면 증강의 *상대 순위* 는 대체로 전이되나 절댓값은 달라질 수 있다.
여유가 되면 최종 backbone 에서 `rotate+blur` vs `rotate 단독` vs `무증강` 만 30 epoch
로 재확인하면 결론이 완결된다.

### 7.10 최종 실험 — 증강 ablation × loss × gamma (ResNet-50, 30 epoch)

> 7.9 에서 고른 best 증강(`rotate+blur`)을 **정식 예산(30 epoch)** 에서 재검증하면서,
> 프로젝트 핵심 가설(**CE vs CB-Focal**)과 **focal focusing(gamma) 민감도** 를 한 번에
> 본다. backbone 은 ResNet 으로 고정하되, 증강이 튜닝된 바로 그 **resnet50** 으로 고정해
> 전이 가정(confound)을 제거한다.

#### 설계 (2 × 5 × 3 = 30런)

| 요인 | 수준 |
|---|---|
| **Augmentation** | `base`(무증강) / `rotate+blur` |
| **Loss 설정** | `CE` + `CB-Focal`(β=0.999) × `gamma ∈ {0, 1, 2, 5}` |
| **Seed** | 42 / 43 / 44 (에러바용 반복) |

- CB-Focal 의 클래스별 가중치(=원조 Focal 의 alpha 역할)는 `beta` 로부터 생성되므로
  **alpha(=beta)=0.999 고정**, **gamma 만 스윕**한다. `gamma=0` 은 focusing OFF
  (= class-balanced weighted CE), `gamma=2` 는 논문 기본값.
- 모든 런은 `build_controlled_transforms`(combo 모드)로 통제 → `base` 와 `rotate+blur`
  가 동일 파이프라인에서 증강만 차이. 증강 강도는 `outputs/optuna_best_per_aug.json`
  (`rotate_deg≈108.2`, `blur_sigma≈0.268`) 자동 로드.
- 결과는 기존 outputs/ 와 섞이지 않도록 **별도 `outputs_final/`** 에 저장.

#### 실행

```bash
# 30런 일괄 (history_{run_name}.json 있으면 SKIP → 중단점 재개)
DATA_DIR=/path/to/data bash run_final_experiment.sh
# 장시간(30런 × 30 epoch) → 백그라운드 권장 (7.8 nohup 패턴)
DATA_DIR=/path nohup bash run_final_experiment.sh > run_final.log 2>&1 &
tail -f run_final.log

# (옵션) 환경변수로 범위 조절
SEEDS="42 43 44" GAMMAS="0 1 2 5" EPOCHS=30 bash run_final_experiment.sh
```

run_name 규칙: `resnet50_ce_{noaug|rotblur}_s{seed}`,
`resnet50_cbf_g{gamma}_{noaug|rotblur}_s{seed}` → 파일 충돌 없이 조건/시드 식별.

#### 분석 (`analyze_final_experiment.py`, 드라이버가 자동 호출)

```bash
python analyze_final_experiment.py --output_dir ./outputs_final
# torch/데이터 없이 집계만:  --no_per_class
```

생성물 (모두 `outputs_final/`):
- `final_summary.csv` — (증강 × loss설정) 별 Best Macro F1 **mean ± std(n=3)**, 내림차순.
- `plots/gamma_sensitivity.png` — γ→Macro F1 곡선(에러바) + CE 수평 기준선.
- `final_contrasts.csv` — **페어드 대조**(같은 seed Δ): ①증강효과(rotate+blur−noaug),
  ②loss효과(best-γ CB-Focal−CE). `scipy` 있으면 paired t-test p값, 없으면 효과크기(dz).
- `final_per_class_f1.csv` + `plots/minority_f1_vs_gamma.png` — 소수 클래스
  (akiec/df/vasc) F1 을 γ 함수로(=CB-Focal·focusing 의 존재 이유 검증).

#### 의사결정 규칙
1. **H1(증강 전이)**: 같은 loss·seed 에서 `rotate+blur − noaug` Δ 가 seed std 를 넘어
   양수면 증강 이득이 30 epoch 에서도 유효.
2. **H2(loss)**: best-γ CB-Focal − CE, 특히 소수 클래스 F1 향상 여부.
3. **H3(gamma)**: γ 곡선에서 최적 focusing 식별(과하면 하락하는지).
4. **최종 채택**: 시드 평균 Macro F1 최댓값 조건. CB-Focal(최적 γ) ≈ CE(차이가
   seed std 이내)이면 단순한 **CE 기본** + 소수 클래스 F1 로 근거 보강.

> 결과 수치가 나오면 이 절에 7.9 처럼 표로 박제할 것(`outputs_final/` 는 gitignore 대상).

#### 실험 결과 (실측, resnet50 × 30 epoch)

> 30런 중 **26런은 정상(30 epoch) 완료**. 단 **seed 42 의 4개 조건이 1 epoch 만 돌고
> 중단**되어, `analyze_final_experiment.py` 가 생성한 `final_summary.csv` /
> `final_contrasts.csv` 의 해당 행이 오염돼 있다. 아래 표는 **깨진 런을 제외하고
> 다시 집계한 값**이며, 원본 CSV 를 그대로 인용하면 안 된다.

##### ⚠️ 데이터 정합성 — 깨진 4개 런

`history.args.epochs == 1`, history 길이 1 인 미완료 런(전부 seed 42):

| 조건 | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| `ce` × `noaug` | **1 ep · 0.164** | 30 ep · 0.642 | 30 ep · 0.713 |
| `ce` × `rotblur` | **1 ep · 0.170** | 30 ep · 0.739 | 30 ep · 0.714 |
| `cbf γ=2` × `noaug` | **1 ep · 0.197** | 30 ep · 0.671 | 30 ep · 0.690 |
| `cbf γ=2` × `rotblur` | **1 ep · 0.221** | 30 ep · 0.705 | 30 ep · 0.712 |

이 4개 조건만 `final_summary.csv` 에서 std ≈ 0.28~0.32 로 튄다 — **학습 붕괴가 아니라
1-epoch 모델이 평균에 섞인 집계 오염**이다. 재실행하려면 해당 4개
`history_*.json`(+`best_model_*.pth`)을 지우고 `run_final_experiment.sh` 를 다시
돌리면 SKIP 로직이 나머지는 건너뛰고 이 4개만 30 epoch 으로 채운다.

##### 헤드라인 — Best Macro F1 (깨진 런 제외, mean ± std)

| 증강 | loss 설정 | Macro F1 | n(seed) |
|---|---|---|---|
| `rotblur` | **CE** | **0.7263 ± 0.0175** | 2* |
| `rotblur` | CBF γ=1 | 0.7127 ± 0.0091 | 3 |
| `rotblur` | CBF γ=5 | 0.7097 ± 0.0090 | 3 |
| `rotblur` | CBF γ=0 | 0.7096 ± 0.0055 | 3 |
| `rotblur` | CBF γ=2 | 0.7088 ± 0.0048 | 2* |
| `noaug` | CBF γ=0 | 0.7051 ± 0.0089 | 3 |
| `noaug` | CBF γ=2 | 0.6805 ± 0.0140 | 2* |
| `noaug` | CBF γ=1 | 0.6807 ± 0.0113 | 3 |
| `noaug` | **CE** | 0.6775 ± 0.0503 | 2* |
| `noaug` | CBF γ=5 | 0.6722 ± 0.0257 | 3 |

`*` = seed 42 누락으로 2 seed 집계. **시드 평균 최고 = `rotblur` × CE = 0.7263**
(단 2 seed 라 seed 42 재실행 전까지는 잠정).

##### 가설 판정

- **H1 (증강 전이) — ✅ 성립.** 같은 loss·seed 에서 `rotblur − noaug` Δ 가 모든
  loss 설정에서 양수. paired t-test 로 유의: CBF γ=1 Δ=+0.0321 (p=0.049),
  γ=2 Δ=+0.0283 (p=0.021). 30 epoch 에서도 7.9 의 증강 이득이 유지된다
  (단 +0.03~0.05 수준으로, 15 epoch 의 +0.095 보다는 작다 — 예산이 늘면 무증강도
  따라잡아 격차가 줄어드는 전형적 패턴).
- **H2 (loss: CB-Focal vs CE) — ❌ 이득 없음.** 깨진 런을 빼면 같은 증강에서
  CE 와 CB-Focal(최적 γ) 차이가 seed std 이내다. `rotblur` 에서는 오히려 CE 가
  best-γ CBF(γ=1) 보다 +0.019 높고(2 seed), `noaug` 에서도 CBF γ=0 우세폭이
  +0.024 ± 0.041 로 비유의. **이 데이터에서는 CB-Focal 이 CE 를 이긴다는 증거가
  없다.** (원본 `final_contrasts.csv` 의 Δ≈0.17~0.20 은 1-epoch CE 와 비교한 허수.)
- **H3 (gamma 민감도) — 평탄, 과하면 하락.** `rotblur` 에서 γ∈{0,1,2,5} 가
  0.709~0.713 으로 사실상 무차별 → focusing 의 추가 이득 없음. `noaug` 에서는
  γ↑ 시 하락(γ=0: 0.705 → γ=5: 0.672)해 **focusing 이 오히려 해로움**.

##### 소수 클래스 (akiec / df / vasc) — `final_per_class_f1.csv`, 정상 런 한정

CB-Focal 의 존재 이유인 소수 클래스 F1 도 γ 로 끌어올리지 못한다. 소수 3클래스 평균 F1:

| 증강 | γ=0 | γ=1 | γ=5 |
|---|---|---|---|
| `rotblur` | 0.665 | 0.674 | 0.682 |
| `noaug` | 0.677 | 0.645 | 0.633 |

`rotblur` 에서 γ 증가에 따라 미미하게(+0.017) 오르지만 `noaug` 에서는 반대로 내려가
방향이 일관되지 않다. 다수 클래스 `nv` 는 전 조건 0.92 로 포화. **focusing 의
소수클래스 구제 효과는 관측되지 않음** → H2 결론 보강.

#### 최종 결론

1. **증강(rotate+blur)은 30 epoch 에서도 유효**(H1, 통계 유의) — 채택.
2. **CB-Focal 은 CE 대비 이득 없음**(H2·H3·소수클래스 모두 음성) → 최종 모델은
   **CE 기본** 으로 가고, CB-Focal 은 "시도했으나 이 데이터·backbone 에선 개선 없음"
   으로 보고하는 것이 타당.
3. **단, `rotblur × CE` 가 1위라는 결론은 seed 42(ce·γ=2 4런) 재실행 후 확정**할 것.
   현재 CE 조건은 2 seed 뿐이라 1위 마진(+0.014)이 표본 부족에 취약하다.
