#!/usr/bin/env bash
# week10 실습 실행 (conda activate week10-dezero 후)
#   bash run_week10.sh
#   bash run_week10.sh q1

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

if [[ "${CONDA_DEFAULT_ENV:-}" != "week10-dezero" ]]; then
  echo "먼저: conda activate week10-dezero  (현재: ${CONDA_DEFAULT_ENV:-없음})" >&2
  exit 1
fi

ONLY="${1:-all}"

run() {
  echo
  echo "========== $1 =========="
  python "$1"
}

case "$ONLY" in
  all)
    run dezero3.py
    run dezero4.py
    run q_learning_nn.py
    run quiz_q1_optimizer_compare.py
    run quiz_q2_sin_4pi.py
    run quiz_q3_gridworld.py
    ;;
  dezero3)  run dezero3.py ;;
  dezero4)  run dezero4.py ;;
  qlearning) run q_learning_nn.py ;;
  q1) run quiz_q1_optimizer_compare.py ;;
  q2) run quiz_q2_sin_4pi.py ;;
  q3) run quiz_q3_gridworld.py ;;
  *)
    echo "사용: bash run_week10.sh [all|dezero3|dezero4|qlearning|q1|q2|q3]" >&2
    exit 1
    ;;
esac

echo
echo "완료."
