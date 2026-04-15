# paper_repro 결과 리포트

## 설정

- preset: `paper_repro`
- episodes: `500`
- alpha/gamma: `0.1` / `0.9`
- epsilon: `0.3` (every `50` ep x `0.95`)
- seeds: `(7, 42, 101, 202, 777, 909, 1301, 2024, 3031, 4049)`
- evaluation_pairs: `((0, 5),)`

## 핵심 수치 (실험 평균)

| metric | Q-Learning(mean+-std) | Baseline | 개선율(대비 Baseline) |
|---|---:|---:|---:|
| path_reward | 1.8250 +- 0.0909 | 1.7083 | 6.83% |
| total_bandwidth | 25.3000 +- 0.9000 | 24.0000 | 5.42% |
| total_delay | 17.5000 +- 1.2845 | 19.0000 | -7.89% |
| hops | 3.0000 +- 0.0000 | 3.0000 | 0.00% |

## 생성 이미지

- `paper_repro_learning_curve.png`
- `paper_repro_comparison_bars.png`
- `paper_repro_seed_boxplot.png`

## 해석 가이드

1. `comparison_bars`: 논문표 참조값과 현재 실험값의 거리 확인
2. `learning_curve`: 학습 수렴 형태와 변동성 확인
3. `seed_boxplot`: 재현성(시드 민감도) 확인
