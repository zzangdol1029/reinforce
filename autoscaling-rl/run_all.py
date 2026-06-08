"""전체 실험 파이프라인 — 한 번에 실행.

    python run_all.py                # thread 환경 전체 (v1 + v2)
    python run_all.py --quick        # v1 thread만 (빠른 확인, 약 3분)
    python run_all.py --v2-only      # v2 thread만
    python run_all.py --with-container  # 인스턴스 환경까지 포함 (확장 실험)

순서:
  1) 학습      train.py  (환경 x 알고리즘, 검증 best 체크포인트 자동 저장)
  2) 평가      evaluate.py  (best 가중치로 비교 그래프·summary.csv 생성)
  3) 설명 그림  make_figures.py  (트래픽 샘플, 비단조 용량 곡선)
  4) 안내      PPT 재생성·시각화 데모 명령 출력

모든 산출물은 results/ 에 생성된다. 중간에 끊겨도 train.py의
체크포인트(--resume) 덕분에 다시 실행하면 이어서 진행된다.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

PY = sys.executable

# 주력: thread 환경 (인스턴스 고정 운영 환경 반영)
V1 = [("threadpool", "dqn", 500), ("threadpool", "ppo", 600)]
V2 = [("ev-threadpool", "dqn", 200), ("ev-threadpool", "ppo", 300)]
# 확장 실험용: 인스턴스 오토스케일링 (--with-container)
CONTAINER_JOBS = [("container", "dqn", 300), ("container", "ppo", 300),
                  ("ev-container", "dqn", 200), ("ev-container", "ppo", 300)]


def run(cmd: list[str]):
    print(f"\n$ {' '.join(cmd)}")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"실패: {' '.join(cmd)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="v1 thread만 실행")
    ap.add_argument("--v2-only", action="store_true", help="v2 thread만 실행")
    ap.add_argument("--with-container", action="store_true",
                    help="인스턴스 환경 포함 (확장 실험)")
    args = ap.parse_args()

    jobs = V1 if args.quick else V2 if args.v2_only else V1 + V2
    if args.with_container:
        jobs = jobs + CONTAINER_JOBS
    envs = sorted({e for e, _, _ in jobs})

    t0 = time.time()
    print(f"=== 1/3 학습 ({len(jobs)}개 작업) ===")
    for env, algo, eps in jobs:
        run([PY, "train.py", "--env", env, "--algo", algo,
             "--episodes", str(eps), "--resume"])

    print("\n=== 2/3 평가 + 그래프 ===")
    for env in envs:
        run([PY, "evaluate.py", "--env", env])

    print("\n=== 3/3 설명용 그림 ===")
    run([PY, "make_figures.py"])

    mins = (time.time() - t0) / 60
    print(f"""
=== 완료 ({mins:.1f}분) — 다음 단계 ===
  결과 수치 :  results/<env>_summary.csv  ->  PPT 결과 표에 기입
  결과 그림 :  results/<env>_learning_curves.png / _behavior.png / _comparison.png
  PPT 재생성:  npm install pptxgenjs && node build_deck.js
  라이브 데모:  python simulate_live.py --env threadpool --algo dqn
  HTML 데모 :  python export_policy.py --algo dqn  ->  viz_simulation_dqn.html
  GIF (PPT용):  python make_gif.py --env threadpool --algo dqn
""")


if __name__ == "__main__":
    main()
