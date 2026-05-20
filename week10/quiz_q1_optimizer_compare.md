# quiz_q1_optimizer_compare.py — 변수 의미·비교 표

PDF 16페이지 **Quiz Q1**: `dezero4.py` 와 **동일한 데이터·모델**에서 Optimizer 4종(+SGD 기준선)의 **loss 곡선**을 비교합니다.

## 비교 변수 표 (슬라이드·보고서용)

| 변수 | 의미 | 비교 시 설정 (기본값) |
| :--- | :--- | :--- |
| **optimizer** | 가중치를 갱신하는 알고리즘 | **4종을 한 번에 실행** (동일 조건): SGD(기준선), MomentumSGD, AdaGrad, Adam |
| **lr (학습률)** | 1 step 당 파라미터 갱신 크기 | 옵티마이저마다 **서로 다른 기본값** (스케일이 달라 공정 비교용으로 튜닝됨): `--lr-sgd 0.2`, `--lr-momentum 0.2`, `--lr-adagrad 0.05`, `--lr-adam 0.02` |
| **iters** | 학습 반복(iteration) 횟수 | **모두 동일** — 기본 `10000` (`--iters`) |
| **hidden** | 2층 MLP 은닉층 뉴런 수 | **모두 동일** — 기본 `10` (`dezero4.py` 와 동일, `--hidden`) |
| **seed** | 난수 시드 (데이터·초기화 재현) | **모두 동일** — 기본 `0` (`--seed`) |
| **초기 가중치** | 학습 시작 시 W, b | **4회 모두 동일** — 템플릿 모델 스냅샷 후 `restore_params` 로 복원 |
| **데이터** | 비선형 회귀 과제 | **모두 동일** — 표본 100개, \(x \sim U(0,1)\), \(y = \sin(2\pi x) + \text{noise}\) (`dezero4.py` 와 동일) |
| **log-every** | loss 를 기록하는 step 간격 | **동일** — 기본 `10` (`--log-every`) |
| **loss (지표)** | 학습이 잘 되는지 보는 값 | **MSE** (평균 제곱 오차); 그래프 y축은 log scale |

## Optimizer 별 학습률 (이 스크립트에서만 바뀌는 항목)

| Optimizer | CLI 옵션 | 기본 lr | 비고 |
| :--- | :--- | ---: | :--- |
| SGD | `--lr-sgd` | 0.2 | 비교 기준선 |
| MomentumSGD | `--lr-momentum` | 0.2 | 관성(momentum) 반영 |
| AdaGrad | `--lr-adagrad` | 0.05 | 누적 제곱 기울기로 lr 자동 축소 |
| Adam | `--lr-adam` | 0.02 | 1·2차 모멘트 적응형 |

## 실행 예

```bash
conda activate week10-dezero
cd week10
python quiz_q1_optimizer_compare.py
python quiz_q1_optimizer_compare.py --no-plot --iters 8000
```

실행 후 `results_quiz_q1_optimizer_compare/parameter_summary.txt` 에 위 표 요약과 최종 MSE 가 저장됩니다.
