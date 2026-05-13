#!/usr/bin/env bash
# ===================================================================
#  DeZero 실습 환경 설정 스크립트
#  - week10 폴더에서 한 번만 실행하면 됩니다.
#  - 시스템 Python(또는 conda Python)에 영향을 주지 않도록
#    .venv-dezero 라는 별도의 virtualenv 를 생성합니다.
#  - numpy >= 1.24 와 호환되지 않는 dezero 의 `np.int` 참조를
#    venv 내부에서만 자동으로 패치합니다.
# ===================================================================
set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="${PYTHON:-python3}"
VENV_DIR=".venv-dezero"

echo "==> Python:        $($PY --version)"
echo "==> Project root:  $HERE"
echo "==> Virtualenv:    $HERE/$VENV_DIR"
echo

if [ ! -d "$VENV_DIR" ]; then
    echo "[1/4] virtualenv 생성"
    "$PY" -m venv "$VENV_DIR"
else
    echo "[1/4] virtualenv 이미 존재 — 건너뜀"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

echo
echo "[2/4] pip 업그레이드"
python -m pip install --upgrade pip wheel setuptools

echo
echo "[3/4] 의존성 설치 (numpy, matplotlib, dezero)"
pip install -r requirements.txt

echo
echo "[4/4] DeZero numpy 호환 패치 적용"
# pip show 로 위치 조회 (import 가 실패하더라도 동작)
DEZERO_DIR="$(python - <<'PY'
import sys, importlib.util
spec = importlib.util.find_spec("dezero")
if spec is None or spec.submodule_search_locations is None:
    sys.exit("dezero 위치를 찾을 수 없습니다")
print(list(spec.submodule_search_locations)[0])
PY
)"
echo "    -> $DEZERO_DIR"

# np.int (NumPy 1.20+ 에서 deprecated) -> int 로 치환
for f in "transforms.py" "datasets.py"; do
    target="$DEZERO_DIR/$f"
    if [ -f "$target" ]; then
        # macOS / Linux 모두 동작하는 sed in-place
        python - "$target" <<'PY'
import sys, re, pathlib
p = pathlib.Path(sys.argv[1])
src = p.read_text()
new = re.sub(r'\bnp\.int\b', 'int', src)
if new != src:
    p.write_text(new)
    print(f"    patched: {p.name}")
else:
    print(f"    no-change: {p.name}")
PY
    fi
done

echo
echo "[verify] import dezero / matplotlib 백엔드 확인"
python - <<'PY'
import dezero, numpy, matplotlib
print('  dezero  :', dezero.__file__)
print('  numpy   :', numpy.__version__)
print('  mpl ver :', matplotlib.__version__)
print('  backend :', matplotlib.get_backend())
# GUI 가능 여부 빠른 체크 (창은 열지 않음)
import platform
sys_name = platform.system()
gui_be = 'MacOSX' if sys_name == 'Darwin' else 'TkAgg'
try:
    matplotlib.use(gui_be, force=True)
    print(f'  GUI test: {gui_be} 백엔드 활성화 OK')
except Exception as e:
    print(f'  GUI test: {gui_be} 활성화 실패 -- {type(e).__name__}: {e}')
PY

echo
echo "============================================================"
echo "  설정 완료. 다음과 같이 사용하세요."
echo "============================================================"
echo "  source $VENV_DIR/bin/activate"
echo "  python dezero3.py          # 실습 #1 : 선형 회귀"
echo "  python dezero4.py          # 실습 #2 : 비선형 회귀 (MLP)"
echo "  python q_learning_nn.py    # 실습 #3 : Q-Network  (그래프 창 3개)"
echo
echo "  💡 그래프 창은 plt.show() 가 blocking 모드이므로,"
echo "      창을 닫아야 다음 그래프가 뜹니다."
echo "      q_learning_nn.py 실행 순서:"
echo "        1) 학습 (약 5~15초)  ->  2) loss curve   (창 닫기)"
echo "        ->  3) Q heatmap     (창 닫기)"
echo "        ->  4) greedy policy (창 닫기)"
echo
