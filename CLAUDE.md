# CLAUDE.md

이 파일은 Claude Code가 본 저장소에서 작업할 때 참고하는 가이드입니다.

> **팀 프로젝트 주의사항** — 이 파일은 팀과 공유됩니다.
> 작업이 끝난 뒤 **`git push` 하기 전에 항상 이 CLAUDE.md를 최신 상태로 업데이트**해 주세요.
> (새 모듈 추가, CLI 인자 변경, 학습 파이프라인 수정 등 반영)

---

## 1. 프로젝트 개요

**FocalLoss_Optimizer** — HAM10000 (피부 병변, 7-class) 미세 분류 실험.
**Cross-Entropy** vs **Class-Balanced Focal Loss (Cui et al., CVPR 2019)** 의 성능을
다양한 backbone 과 augmentation 조합 위에서 비교한다.

- 평가 지표: **Macro F1** (클래스 불균형이 심해 accuracy 대신 사용)
- 데이터 누수 방지: 동일 `lesion_id` 가 train/val 양쪽에 들어가지 않도록 lesion 단위 stratified split

### 실험 구조 (4단계)

| Phase | 목적 | 실험 내용 |
|---|---|---|
| 1 | Backbone 비교 (완료) | ResNet-50 / DenseNet-121 / MobileNetV3-Large / EfficientNet-B3 × CE |
| 2 | Loss 비교 (완료) | EfficientNet-B3 × CE vs CB-Focal, DenseNet-121 × CE vs CB-Focal |
| 3 | ResNet depth 비교 | ResNet-101 vs ResNet-152 (baseline aug + CE) → 더 높은 F1 backbone 선택 |
| 4 | Augmentation 비교 | 선택된 ResNet × CE × 2³=8 조합 (CutMix / Elastic / ColorJitter on/off) |
| 5 | Augmentation 탐색 (Optuna) | resnet50 × CE 고정, 8가지 증강 기법의 연속/이산 파라미터 공간을 TPE 베이지안 최적화 + MedianPruner 로 탐색 |

- Phase 1 결과: DenseNet-121이 Best Macro F1 0.7817로 최고 backbone 확인
- Phase 3·4 는 `run_experiments.sh` 가 순서대로 자동 실행하며 Step 1→2 winner 선택도 자동화
- Phase 5 는 `run_optuna_search.sh` 가 실행. SQLite storage 로 중단점 재개(resume) 지원 — 끊겨도 다시 실행하면 완료된 trial 건너뛰고 이어서 진행

## 2. 디렉토리 구조

```
FocalLoss_Optimizer/
├── main.py               # 학습/검증 엔트리포인트 (단일 학습 + --optuna 탐색 모드)
├── dataset.py            # HAM10000Dataset, discover_image_roots, load_metadata
├── transforms.py         # 커스텀 증강(Sobel/GaussianNoise/AverageBlur) + build_transforms(aug_params)
├── losses.py             # CBFocalLoss + build_loss factory
├── model.py              # SUPPORTED_BACKBONES + build_model factory (timm)
├── download_data.py      # Kaggle 미러에서 HAM10000 받기
├── run_experiments.sh    # Step1(ResNet depth) → Step2(8 aug 조합) 자동 순차 실행
├── run_optuna_search.sh  # Optuna 증강 파라미터 탐색 (resnet50 고정, resume 지원)
├── measure_epoch_time.sh # resnet50 epoch당 시간 측정 헬퍼 (실험 총 소요시간 추정용)
├── analyze_results.py    # summary.csv + 시각화 이미지 생성 (run_name 기반)
├── environment.yml       # conda 환경 (파이썬만 conda, 나머지는 pip)
├── requirements.txt      # pip 의존성
├── README.md             # 사용자/팀원용 문서
├── CLAUDE.md             # ← 이 파일 (Claude Code 가이드)
└── outputs/              # 학습 결과
    ├── best_model_{run_name}.pth     # run_name = --run_name 또는 {backbone}_{loss_type}
    ├── history_{run_name}.json
    ├── summary.csv
    └── plots/
        ├── f1_comparison.png
        └── learning_curves.png
```

## 3. 자주 쓰는 명령어

```bash
# 환경 (conda 권장)
conda env create -f environment.yml
conda activate focal-ham10000

# 데이터 다운로드 (Kaggle 인증 필요할 수 있음)
python download_data.py --data_dir ./data

# 전체 실험 자동 실행 (Step1: ResNet depth → Step2: 8 aug 조합 → 결과 분석)
# 이미 history_{run_name}.json 이 존재하는 실험은 자동으로 건너뜀
bash run_experiments.sh
# 데이터 경로가 다를 경우:
DATA_DIR=/path/to/data bash run_experiments.sh

# Optuna 증강 탐색 (resnet50 × CE 고정, 8 augmentation 파라미터 TPE 탐색)
# SQLite storage 사용 → 중단되어도 재실행하면 남은 trial 만 이어서 진행
bash run_optuna_search.sh
DATA_DIR=/path/to/data N_TRIALS=50 EPOCHS=15 bash run_optuna_search.sh
# 직접 호출:
python main.py --optuna --data_dir ./data --backbone resnet50 --loss_type ce \
    --n_trials 30 --epochs 15 --storage sqlite:///optuna_study.db \
    --study_name resnet50_ce_aug_search

# epoch당 시간 측정 (실험 총 소요시간 추정용, 두 번째 epoch 의 time= 값 확인)
bash measure_epoch_time.sh

# 개별 실행 예시
# ResNet-101 baseline (Step 1)
python main.py --data_dir ./data --backbone resnet101 --loss_type ce --epochs 30 --batch_size 32 --lr 1e-4

# ResNet-101 + CutMix + ColorJitter (Step 2 조합 예시)
python main.py --data_dir ./data --backbone resnet101 --loss_type ce \
    --use_cutmix --use_colorjitter \
    --run_name resnet101_ce_cm_cj \
    --epochs 30 --batch_size 32 --lr 1e-4

# Class-Balanced Focal Loss (EfficientNet-B3)
python main.py --data_dir ./data --backbone efficientnet_b3 --loss_type cb_focal \
    --beta 0.999 --gamma 2.0 --epochs 30 --batch_size 32 --lr 1e-4

# 결과 평가 (Accuracy / Macro F1 / Precision / Recall + 클래스별 전체)
python eval_accuracy.py

# 결과 분석만 별도 실행
python analyze_results.py --output_dir ./outputs
```

### 지원 backbone (`model.py:SUPPORTED_BACKBONES`)

| 키 | timm 모델명 |
|---|---|
| `efficientnet_b3` | efficientnet_b3 |
| `resnet50` | resnet50 |
| `resnet101` | resnet101 |
| `resnet152` | resnet152 |
| `densenet121` | densenet121 |
| `mobilenetv3_large_100` | mobilenetv3_large_100 |

모두 ImageNet-1k pretrained 가중치 사용.

## 4. 코드 컨벤션 / 작업 시 주의

- **클래스 순서 고정**: `dataset.py:HAM10000_CLASSES` (akiec, bcc, bkl, df, mel, nv, vasc) — 모델/실험 일관성을 위해 어디서도 순서를 바꾸지 말 것.
- **CB-Focal 가중치 정규화**: `losses.py:CBFocalLoss` 에서 `weights / weights.sum() * num_classes` (논문 official impl 컨벤션). 임의로 빼지 말 것.
- **수치 안정성**: focal loss 계산은 `log_softmax + clamp(eps=1e-7)` 조합 유지.
- **재현성**: `main.py:set_seed` 에서 `cudnn.deterministic=True, benchmark=False`. 속도 우선이 필요하면 별도 플래그로 분리할 것 (현재 코드를 직접 바꾸지 말 것).
- **Best 모델 기준**: Macro F1 (accuracy 아님). `outputs/best_model_{run_name}.pth`.
- **run_name 규칙**: `--run_name` 미지정 시 `{backbone}_{loss_type}` 로 자동 생성. 모델·히스토리 파일명과 eval 출력 레이블 모두 이 값 사용. `run_experiments.sh` 의 aug 실험은 `{backbone}_ce_{aug_tag}` 형식 (예: `resnet101_ce_cm_el`).
- **CutMix**: `main.py:cutmix_batch` 에서 구현. batch-level 적용이므로 transforms 가 아닌 train loop 내부에서 실행. loss = λ·L(a) + (1−λ)·L(b) 형태로 CE·CB-Focal 모두 호환.
- **Elastic Transform**: torchvision `>=0.12` 의 `transforms.ElasticTransform(alpha=50.0, sigma=5.0)` 사용. PIL 단계에서 적용 (`ToTensor` 앞).
- **두 가지 증강 경로 (혼동 주의)**:
  - 단일 학습 모드는 `main.py:build_transforms(args)` — `--use_cutmix/--use_elastic/--use_colorjitter` 플래그 기반 (기존 방식, 그대로 유지).
  - Optuna 탐색 모드는 `transforms.py:build_transforms(aug_params)` — trial 이 넘기는 파라미터 dict 로 동적 조립. main.py 에서는 `build_optuna_transforms` 별칭으로 import.
- **커스텀 증강 3종** (`transforms.py`): `SobelFilter`(엣지 추출, 확률), `GaussianNoise`(std·확률), `AverageBlur`(홀수 커널·확률). 입력이 PIL/Tensor 모두여도 내부 변환 후 **입력과 동일 타입으로 반환**. 파이프라인 순서: PIL(Crop→Flip→Rotate→ColorJitter→GaussianBlur) → ToTensor → Tensor(AverageBlur→Sobel→Noise) → Normalize → Cutout(RandomErasing).
- **Optuna 탐색 규칙** (`main.py:run_optuna`):
  - 데이터셋/로더는 1회만 생성하고 trial 마다 `train_set.transform` 만 교체 (디스크 재스캔 회피). val transform 은 고정.
  - 각 trial 시작 시 `set_seed(args.seed)` 로 모델 초기화·증강 RNG 를 동일하게 맞춰 **augmentation 만 변수**가 되도록 통제.
  - 매 epoch `trial.report` + `MedianPruner(n_startup_trials=5, n_warmup_steps=5)` 로 부진 trial 조기 종료.
  - **Resume**: `create_study(load_if_exists=True)` + `remaining = n_trials - (COMPLETE trial 수)` 로 남은 만큼만 실행. 처음부터 다시 하려면 `optuna_study.db` 삭제 또는 `--study_name` 변경.
  - run_name 은 `{backbone}_{loss_type}_trial_{trial.number}`. trial 마다 best 모델/히스토리가 `outputs/` 에 쌓이므로(=resnet50 ckpt 다수) 디스크 사용량 주의.
- **이미지 루트 탐색**: `dataset.py:discover_image_roots` 가 `images/`, `HAM10000_images_part_{1,2}/`, `ham10000_images/`, `data_dir` 자체를 모두 시도. 새 폴더 구조가 생기면 여기에 추가.
- **PyTorch 설치**: `environment.yml`은 conda 채널의 거대한 `pytorch-cuda` 를 피하고 PyTorch 공식 pip wheel 을 사용. 한국 네트워크에서 끊김 문제 때문이니 conda 채널로 되돌리지 말 것.

## 5. Git 커밋/푸쉬 규칙 (팀 합의)

> **❗ 매우 중요 — Claude Code 사용 시**
>
> - 커밋 메시지에 **`Co-Authored-By: Claude ...` 줄(co-author 트레일러)을 절대 포함하지 말 것.**
>   기본 동작에서 추가되는 `Co-Authored-By: Claude Opus ...` 같은 라인은 모두 **제거하고** 커밋한다.
> - 마찬가지로 `🤖 Generated with [Claude Code]` 같은 자동 서명 푸터도 넣지 않는다.
> - 즉, 커밋 메시지는 **사람이 직접 쓴 것처럼 본문(제목 + 설명)만** 남긴다.

### 커밋 예시 (OK)
```
feat(losses): CB-Focal 가중치 정규화 방식 변경

weight.sum() == num_classes 가 되도록 정규화하여
official impl 과 동치가 되게 수정.
```

### 커밋 예시 (NG — 이렇게 쓰지 말 것)
```
feat(losses): CB-Focal 가중치 정규화 방식 변경

...

🤖 Generated with [Claude Code](https://claude.com/claude-code)
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

### 푸쉬 전 체크리스트
1. 변경된 코드/명령/파이프라인을 **CLAUDE.md (이 파일) 와 README.md** 에 반영했는가?
2. 커밋 메시지에 `Co-Authored-By:` 또는 Claude 자동 서명이 들어가 있지 않은가?
3. `outputs/`, `data/`, `*.pth`, 큰 캐시 파일 등을 실수로 커밋하지 않았는가? (필요하면 `.gitignore` 확인)
4. (가능하면) 학습 한 epoch 라도 돌려서 import / dataloader / loss 가 깨지지 않는지 확인.

## 6. 자주 발생하는 이슈

- **`cp949 codec can't decode...`** (conda env 생성 시): 경로 한글 때문에 conda explicit-spec 플러그인이 cp949 로 읽다 실패하는 것. **무시해도 됨** — YAML 파싱은 정상.
- **Kaggle 다운로드 실패**: `~/.kaggle/kaggle.json` 또는 `kagglehub login` 먼저 수행. 사내망이면 `conda config --set remote_read_timeout_secs 600`.
- **이미지 누락 경고** (`[WARN] 이미지 파일 누락 N건`): `dataset.py:HAM10000Dataset` 가 출력. `discover_image_roots` 가 잡지 못한 경로가 있는지 확인.
