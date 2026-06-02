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
DATA_DIR=/path/to/data nohup bash run_all_aug.sh > run_all_aug.log 2>&1 &
tail -f run_all_aug.log
```
