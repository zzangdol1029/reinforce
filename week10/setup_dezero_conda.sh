#!/usr/bin/env bash
# week10 Conda 환경 설치 (최초 1회 또는 갱신)
#
# 사용법:
#   cd week10 && bash setup_dezero_conda.sh
#   conda activate week10-dezero
#   python dezero3.py
#   ...
#
# 참고: conda가 없으면 스크립트가 종료됩니다.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

command -v conda >/dev/null || { echo "conda 없음"; exit 1; }
# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '!/^(\s*)(#|$)/ && NF>=2 {print $1}' | grep -Fxq 'week10-dezero'; then
  echo "[1/2] conda env update -f environment.yml --prune"
  conda env update -f environment.yml --prune
else
  echo "[1/2] conda env create -f environment.yml"
  conda env create -f environment.yml
fi

echo
echo "[2/2] import 확인"
conda run -n week10-dezero python -c "import dezero, numpy as np; print('dezero:', dezero.__file__); print('numpy:', np.__version__)"

cat << EOF

완료. 이제:

  conda activate week10-dezero
  cd $HERE
  python dezero3.py
  python dezero4.py
  python q_learning_nn.py
  python quiz_q1_optimizer_compare.py
  python quiz_q2_sin_4pi.py
  python quiz_q3_gridworld.py

한 번에: bash run_week10.sh

EOF
