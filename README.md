# FocalLoss_Optimizer

HAM10000 (피부 병변, 7-class) 미세 분류 실험.
**Cross-Entropy** vs **Class-Balanced Focal Loss (Cui et al., 2019)** 의 성능을
EfficientNet-B3 (timm) 위에서 비교한다.

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

### Cross-Entropy 베이스라인
```bash
python main.py --data_dir ./data --loss_type ce --epochs 30 --batch_size 32 --lr 1e-4
```

### Class-Balanced Focal Loss
```bash
python main.py --data_dir ./data --loss_type cb_focal \
    --beta 0.999 --gamma 2.0 \
    --epochs 30 --batch_size 32 --lr 1e-4
```

매 에포크마다 다음을 출력한다.
- `train_loss`, `val_loss`
- 클래스별 F1 + **Macro F1** (`sklearn.classification_report`)

Best Macro F1 갱신 시 `outputs/best_model.pth` 로 저장되며,
학습 이력은 `outputs/history_<loss_type>.json` 에 기록된다.

## 4. CLI 인자 요약

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--data_dir` | (필수) | 메타데이터/이미지 루트 디렉토리 |
| `--loss_type` | `ce` | `ce` 또는 `cb_focal` |
| `--epochs` | 30 | 학습 에포크 수 |
| `--batch_size` | 32 | 배치 사이즈 |
| `--lr` | 1e-4 | 학습률 (AdamW) |
| `--beta` | 0.999 | CB Loss 의 beta |
| `--gamma` | 2.0 | Focal Loss 의 gamma |
| `--val_size` | 0.2 | 검증 비율 (lesion 단위 stratified split) |
| `--seed` | 42 | 재현성 시드 |
| `--num_workers` | 4 | DataLoader 워커 수 |
| `--output_dir` | `./outputs` | 모델/로그 저장 위치 |

## 5. 프로젝트 구조

```
FocalLoss_Optimizer/
├── main.py            # 학습/검증 엔트리포인트
├── dataset.py         # HAM10000Dataset, 이미지 경로 탐색, 메타 로딩
├── losses.py          # CBFocalLoss + build_loss factory
├── model.py           # EfficientNet-B3 (timm) 빌더
├── download_data.py   # Kaggle 미러에서 HAM10000 받기
├── environment.yml    # conda 환경
├── requirements.txt   # pip 의존성
└── outputs/           # 학습 결과 (best_model.pth, history JSON)
```

## 6. 구현 노트

- **Class-Balanced Focal Loss**: `losses.py:CBFocalLoss`
  - Effective Number of Samples: `E_n = (1 - β^n_c) / (1 - β)`
  - 클래스 가중치: `α_c = 1/E_n` 후 합 = `num_classes` 가 되도록 정규화
  - `log_softmax + clamp(eps=1e-7)` 으로 수치 안정성 확보
- **데이터 누수 방지**: 동일 `lesion_id` 의 이미지는 train/val 한쪽에만 들어가도록
  lesion 단위 stratified split 을 수행 (`main.py:split_train_val`).
- **평가 지표**: 데이터 불균형이 심해 정확도는 다수 클래스(`nv`) 에 끌려가므로
  **Macro F1** 을 best 모델 선정 기준으로 사용한다.
