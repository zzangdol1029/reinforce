# Q-Network 실습 — 설치 및 실행 안내

> 📄 강의 노트: [`08-Q-Network.md`](./08-Q-Network.md)  
> 📄 원본 슬라이드: [`08-Q-Network.pdf`](./08-Q-Network.pdf)

이 폴더의 실습 코드는 *밑바닥부터 시작하는 딥러닝 4* (사이토 고키, 한빛미디어) 의 표준 DeZero 예제를 PDF 원본 그대로 옮겨 둔 것입니다.

```
week10/
├─ dezero3.py            # 실습 #1 — 선형 회귀  y = 2x + 5         (그래프 1개)
├─ dezero4.py            # 실습 #2 — 비선형 회귀 MLP  y = sin(2πx)  (그래프 1개)
├─ q_learning_nn.py      # 실습 #3 — Q-Network on 3x4 Grid World  (그래프 3개)
│
├─ quiz_q1_optimizer_compare.py   # PDF p.16 Quiz Q1 — MomentumSGD / AdaGrad / Adam 비교 (+SGD)
├─ quiz_q2_sin_4pi.py             # PDF p.16 Quiz Q2 — y = sin(4πx) 회귀
│
├─ common/
│   └─ gridworld.py      # 책 저장소 호환 GridWorld 환경
├─ matplotlibrc          # 백엔드 = MacOSX (이 폴더에서 실행 시 자동 적용)
├─ requirements.txt      # numpy / matplotlib / dezero
└─ setup_dezero.sh       # 가상환경 + 의존성 + 호환 패치 자동화
```

---

## 1. 빠른 설치 (권장)

`setup_dezero.sh` 한 줄이면 됩니다.

```bash
cd week10
bash setup_dezero.sh
```

스크립트가 자동으로 수행하는 작업:

1. `week10/.venv-dezero/` 에 가상환경 생성 (시스템 Python 영향 ❌)
2. `requirements.txt` 의 패키지 설치 (`numpy`, `matplotlib`, `dezero`)
3. DeZero 의 `np.int` 참조를 `int` 로 자동 패치
   - DeZero 0.0.13 은 NumPy 1.20 에서 deprecate 된 `np.int` 를 아직 사용
   - 시스템이 NumPy ≥ 1.24 여도 가상환경 내부에서만 패치되므로 안전
4. `import dezero` 성공 여부 확인 출력

설치가 끝나면 가상환경을 활성화하고 실습을 실행합니다.

```bash
source .venv-dezero/bin/activate

python dezero3.py        # 실습 #1
python dezero4.py        # 실습 #2
python q_learning_nn.py  # 실습 #3

python quiz_q1_optimizer_compare.py   # PDF p.16 Q1 — 옵티마이저 비교
python quiz_q2_sin_4pi.py           # PDF p.16 Q2 — sin(4πx)

python quiz_q1_optimizer_compare.py   # PDF p.16 Q1 — 옵티마이저 비교
python quiz_q2_sin_4pi.py           # PDF p.16 Q2 — sin(4πx)
```

---

## 2. 수동 설치

이미 가지고 있는 conda/venv 환경에서 진행하려면:

```bash
pip install -r requirements.txt
python - <<'PY'
import importlib.util, pathlib, re
loc = pathlib.Path(importlib.util.find_spec("dezero").submodule_search_locations[0])
for fn in ("transforms.py", "datasets.py"):
    p = loc / fn
    if p.exists():
        p.write_text(re.sub(r'\bnp\.int\b', 'int', p.read_text()))
        print("patched:", p)
PY
```

또는 PDF 안내(2 페이지) 대로 NumPy 1.23 으로 다운그레이드해도 됩니다. 단, NumPy 1.23.x 는 Python 3.10 까지만 지원하므로 가상환경의 Python 버전을 맞춰야 합니다.

```bash
conda create -n dezero python=3.10 -y
conda activate dezero
pip install numpy==1.23.5 matplotlib dezero
```

---

## 3. 예상 실행 결과

### 실습 #1 — `dezero3.py`

10 회마다 loss 출력 후 W, b 출력 → **데이터 산점도 + 학습된 직선 그래프 1 개**

```
... (10 회마다 loss.data)
====
W = [[2.11807369]]
b = [5.46608905]
```

(목표 W ≈ 2, b ≈ 5 + 노이즈 평균 0.5 → b ≈ 5.5)

### 실습 #2 — `dezero4.py`

1000 회마다 loss 출력 → **데이터 산점도 + MLP 적합 곡선 그래프 1 개**

```
variable(0.8165178492839196)
variable(0.24990280802148895)
...
variable(0.07618764131185567)
```

### 실습 #3 — `q_learning_nn.py`

실행 흐름 (총 약 5–15 초):

```
[학습 1000 에피소드]
        │
        ▼
[ 창 1 : Loss curve ]            ← 닫으면 ▶
        │
        ▼
[ 창 2 : Q 함수 heatmap ]        ← 닫으면 ▶
        │
        ▼
[ 창 3 : greedy 정책 ↑ ↓ ← → ]
```

> 💡 macOS GUI 백엔드의 `plt.show()` 는 **blocking** 입니다.  
> 한 그래프 창을 **마우스로 닫아야** 다음 그래프 창이 뜹니다.  
> (`week10/matplotlibrc` 가 `backend: MacOSX, interactive: False` 로 설정해 둡니다.)

> 💡 SSH 등 GUI 가 없는 환경에서 plot 만 건너뛰려면 `MPLBACKEND=Agg` 환경변수와 함께 실행하세요.
> ```bash
> MPLBACKEND=Agg python q_learning_nn.py
> ```

---

## 4. 자주 발생하는 문제

| 증상 | 원인 | 해결 |
|------|------|------|
| `AttributeError: module 'numpy' has no attribute 'int'` | DeZero 가 NumPy ≥ 1.20 와 비호환 | `bash setup_dezero.sh` 재실행 또는 수동 패치 |
| `ModuleNotFoundError: No module named 'common.gridworld'` | 실습을 `week10/` 가 아닌 다른 디렉토리에서 실행 | `cd week10` 후 실행 |
| `RuntimeError: matplotlib is currently using a non-GUI backend` | GUI 없는 환경에서 `plt.show()` 호출 | `MPLBACKEND=Agg` 환경변수 사용 |
| **그래프 창이 안 뜸 / 즉시 닫힘** | matplotlib 백엔드가 `Agg`(비대화형) 로 잡힘 | 아래 *그래프 백엔드 확인* 참고 |

### 그래프 백엔드 확인

```bash
source .venv-dezero/bin/activate
cd week10           # ← matplotlibrc 가 있는 폴더에서 실행
python -c "
import matplotlib, matplotlib.pyplot as plt
print('backend     :', matplotlib.get_backend())   # MacOSX 가 정상
print('interactive :', matplotlib.is_interactive()) # False 가 정상
"
```

- 결과가 `MacOSX` 가 아닌 `Agg` 라면 `week10/matplotlibrc` 가 무시된 것입니다.
  반드시 `week10` 폴더 안에서 `python` 을 실행하거나, 환경변수로 강제하세요.
  ```bash
  MPLBACKEND=MacOSX python q_learning_nn.py
  ```
- 작은 테스트로 한 줄에 확인:
  ```bash
  python -c "import matplotlib.pyplot as plt; plt.plot([0,1,4,9]); plt.show()"
  ```
  → 창이 뜨고, 닫으면 명령이 종료되어야 정상.
