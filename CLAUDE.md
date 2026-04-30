# CLAUDE.md

이 파일은 Claude Code가 본 저장소에서 작업할 때 참고하는 가이드입니다.

> **팀 프로젝트 주의사항** — 이 파일은 팀과 공유됩니다.
> 작업이 끝난 뒤 **`git push` 하기 전에 항상 이 CLAUDE.md를 최신 상태로 업데이트**해 주세요.
> (새 모듈 추가, CLI 인자 변경, 학습 파이프라인 수정 등 반영)

---

## 1. 프로젝트 개요

**FocalLoss_Optimizer** — HAM10000 (피부 병변, 7-class) 미세 분류 실험.
**Cross-Entropy** vs **Class-Balanced Focal Loss (Cui et al., CVPR 2019)** 의 성능을
EfficientNet-B3 (timm) 위에서 비교한다.

- 평가 지표: **Macro F1** (클래스 불균형이 심해 accuracy 대신 사용)
- 데이터 누수 방지: 동일 `lesion_id` 가 train/val 양쪽에 들어가지 않도록 lesion 단위 stratified split

## 2. 디렉토리 구조

```
FocalLoss_Optimizer/
├── main.py            # 학습/검증 엔트리포인트 (parse_args → train loop)
├── dataset.py         # HAM10000Dataset, discover_image_roots, load_metadata
├── losses.py          # CBFocalLoss + build_loss factory
├── model.py           # EfficientNet-B3 (timm) 빌더
├── download_data.py   # Kaggle 미러에서 HAM10000 받기
├── environment.yml    # conda 환경 (파이썬만 conda, 나머지는 pip)
├── requirements.txt   # pip 의존성
├── README.md          # 사용자/팀원용 문서
├── CLAUDE.md          # ← 이 파일 (Claude Code 가이드)
└── outputs/           # 학습 결과 (best_model.pth, history_<loss>.json)
```

## 3. 자주 쓰는 명령어

```bash
# 환경 (conda 권장)
conda env create -f environment.yml
conda activate focal-ham10000

# 데이터 다운로드 (Kaggle 인증 필요할 수 있음)
python download_data.py --data_dir ./data

# Cross-Entropy 베이스라인
python main.py --data_dir ./data --loss_type ce --epochs 30 --batch_size 32 --lr 1e-4

# Class-Balanced Focal Loss
python main.py --data_dir ./data --loss_type cb_focal \
    --beta 0.999 --gamma 2.0 --epochs 30 --batch_size 32 --lr 1e-4
```

## 4. 코드 컨벤션 / 작업 시 주의

- **클래스 순서 고정**: `dataset.py:HAM10000_CLASSES` (akiec, bcc, bkl, df, mel, nv, vasc) — 모델/실험 일관성을 위해 어디서도 순서를 바꾸지 말 것.
- **CB-Focal 가중치 정규화**: `losses.py:CBFocalLoss` 에서 `weights / weights.sum() * num_classes` (논문 official impl 컨벤션). 임의로 빼지 말 것.
- **수치 안정성**: focal loss 계산은 `log_softmax + clamp(eps=1e-7)` 조합 유지.
- **재현성**: `main.py:set_seed` 에서 `cudnn.deterministic=True, benchmark=False`. 속도 우선이 필요하면 별도 플래그로 분리할 것 (현재 코드를 직접 바꾸지 말 것).
- **Best 모델 기준**: Macro F1 (accuracy 아님). `outputs/best_model.pth`.
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
