# -*- coding: utf-8 -*-
"""
Train Script Runner
train 폴더 안의 모든 학습 스크립트를 순차적으로 실행합니다.
"""

import os
import sys
import subprocess
from pathlib import Path


def run_training_script(script_path: Path):
    """개별 학습 스크립트 실행"""
    print("\n" + "=" * 80)
    print(f"🚀 Running: {script_path.name}")
    print("=" * 80 + "\n")

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            cwd=script_path.parent.parent  # 프로젝트 루트에서 실행
        )
        print(f"\n✅ {script_path.name} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {script_path.name} failed with error code {e.returncode}")
        return False


def main():
    # train 폴더 경로
    train_dir = Path(__file__).parent / "train"

    if not train_dir.exists():
        print(f"❌ Error: {train_dir} directory not found!")
        sys.exit(1)

    # train 폴더 내 모든 .py 파일 찾기 (정렬된 순서대로)
    training_scripts = sorted(train_dir.glob("*.py"))

    if not training_scripts:
        print(f"❌ No training scripts found in {train_dir}")
        sys.exit(1)

    print("=" * 80)
    print("📚 Training Scripts Found:")
    print("=" * 80)
    for i, script in enumerate(training_scripts, 1):
        print(f"  {i}. {script.name}")
    print("=" * 80)

    # 각 스크립트 순차 실행
    results = {}
    for script in training_scripts:
        success = run_training_script(script)
        results[script.name] = success

        if not success:
            print(f"\n⚠️  {script.name} failed. Continuing with next script...")

    # 최종 결과 요약
    print("\n" + "=" * 80)
    print("📊 Training Summary:")
    print("=" * 80)
    for script_name, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"  {script_name}: {status}")
    print("=" * 80)

    # 실패한 스크립트가 있으면 종료 코드 1 반환
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
