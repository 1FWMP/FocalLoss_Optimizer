"""
HAM10000 데이터셋 다운로드 스크립트.

Kaggle 공개 미러("kmader/skin-cancer-mnist-ham10000")에서 데이터를 받아온 뒤,
프로젝트 학습 파이프라인이 기대하는 구조로 정리한다.

최종 구조 (예시):
    <data_dir>/
        HAM10000_metadata.csv
        images/
            ISIC_0024306.jpg
            ISIC_0024307.jpg
            ...

사용 예:
    python download_data.py --data_dir ./data
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HAM10000 dataset downloader")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="HAM10000을 저장할 루트 디렉토리",
    )
    parser.add_argument(
        "--keep_split_folders",
        action="store_true",
        help="HAM10000_images_part_1/2 폴더 구조를 유지 (기본은 images/ 한 폴더로 합침)",
    )
    return parser.parse_args()


def _import_kagglehub():
    try:
        import kagglehub  # noqa: F401
        return kagglehub
    except ImportError as e:
        raise SystemExit(
            "[ERROR] kagglehub 패키지가 필요합니다. `pip install kagglehub` 또는 environment.yml로 설치하세요."
        ) from e


def main() -> None:
    args = parse_args()
    kagglehub = _import_kagglehub()

    data_dir = Path(args.data_dir).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] 데이터 저장 경로: {data_dir}")

    # 1) Kaggle Hub로 다운로드 (캐시 디렉토리에 저장됨)
    print("[INFO] Kaggle에서 HAM10000 다운로드 중... (최초 1회는 시간이 걸립니다)")
    cache_path = Path(
        kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
    )
    print(f"[INFO] 캐시 디렉토리: {cache_path}")

    # 2) 메타데이터 복사
    meta_src = cache_path / "HAM10000_metadata.csv"
    if not meta_src.exists():
        # 일부 미러는 .tab 형식으로 제공
        alt = cache_path / "HAM10000_metadata.tab"
        if alt.exists():
            meta_src = alt
        else:
            raise FileNotFoundError(
                f"HAM10000_metadata.csv 를 찾지 못했습니다: {cache_path}"
            )
    shutil.copy2(meta_src, data_dir / "HAM10000_metadata.csv")
    print(f"[INFO] 메타데이터 복사 완료 -> {data_dir / 'HAM10000_metadata.csv'}")

    # 3) 이미지 복사
    part_dirs = [
        cache_path / "HAM10000_images_part_1",
        cache_path / "HAM10000_images_part_2",
    ]
    part_dirs = [p for p in part_dirs if p.exists()]
    if not part_dirs:
        # 일부 버전은 ham10000_images/ 단일 폴더 형태
        single = cache_path / "ham10000_images"
        if single.exists():
            part_dirs = [single]

    if not part_dirs:
        raise FileNotFoundError(f"이미지 폴더를 찾지 못했습니다: {cache_path}")

    if args.keep_split_folders:
        for src in part_dirs:
            dst = data_dir / src.name
            print(f"[INFO] 복사 {src} -> {dst}")
            shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        merged = data_dir / "images"
        merged.mkdir(exist_ok=True)
        total = 0
        for src in part_dirs:
            for fp in src.iterdir():
                if fp.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    shutil.copy2(fp, merged / fp.name)
                    total += 1
        print(f"[INFO] 총 {total}장의 이미지를 {merged}로 통합했습니다.")

    print("[DONE] HAM10000 준비 완료.")
    print(
        f"학습 실행 예시: python main.py --data_dir {data_dir} --loss_type cb_focal"
    )


if __name__ == "__main__":
    main()
