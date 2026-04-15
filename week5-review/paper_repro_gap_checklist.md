# 논문 재현 갭 분석 체크리스트

대상 논문: **A Q-Learning based Routing Optimization Model in a Software Defined Network (2021)**  
대상 코드: `week5-review/sdn_qlearning_implementation.py`

> 주의: 현재 저장소에서 PDF 본문의 텍스트 추출이 제한되어, 일부 항목은 `detailed_explanation.md`/`usage_guide.md`와 기존 코드 기준으로 점검했습니다. 원문 표/그림 번호 기준 수치 검증은 추가 확인이 필요합니다.

---

## 1) 체크리스트 + 갭 분석표

| 구분 | 논문 재현에 필요한 항목 | 현재 코드 상태 | 갭 수준 | 보완 내용 |
|---|---|---|---|---|
| 토폴로지 | 노드/링크 구조가 논문과 동일해야 함 | 6노드 고정 토폴로지 사용 | 중간 | 토폴로지를 설정 파일(JSON/YAML)로 분리하여 논문 토폴로지 반영 필요 |
| 트래픽 | 트래픽 패턴(고정/동적, 플로우 수, 부하 수준) 명시 | `(0,5),(1,5),(2,5)` 평가 쌍으로 단순화 | 큼 | 시간대별 트래픽 매트릭스, 부하 레벨별 시나리오 추가 |
| 보상함수 | 논문 식/가중치와 동일한 reward 설계 | BW·Delay 0.5/0.5 정규화 + 도착 보너스 | 중간 | 논문의 정확한 가중치/패널티/정규화 범위로 맞춤 필요 |
| 학습 설정 | α, γ, ε 초기값/감쇠/에피소드 수 일치 | 기본값(α0.1, γ0.9, ε0.3, 500ep) 설정화 완료 | 중간 | 논문 하이퍼파라미터 표 기준 값으로 교체 |
| 반복 횟수 | 다중 실행(여러 seed) + 통계(평균/분산) | 5개 seed 반복, mean/std 계산 | 낮음 | seed 개수를 논문 수준으로 확장(예: 10/30회) |
| 비교군 | 논문 비교 알고리즘과 동일해야 함 | BFS 최소홉 baseline 1개 | 큼 | 논문 비교군(예: 기존 QoS 라우팅/ECMP 등) 추가 필요 |
| 지표 | 논문 지표(지연, 처리량, PDR, 보상 등) 측정 | success/hops/delay/bandwidth/path_reward 기록 | 중간 | PDR, jitter, link utilization, control overhead 등 추가 |
| 결과 보고 | 표/그래프 + 재현 가능한 산출물 | CSV/PNG 자동 저장 | 낮음 | 논문 표 형식으로 후처리 스크립트 추가 권장 |

---

## 2) 이번 개편으로 개선된 점

- **재현형 구조화**: `ExperimentConfig`로 학습/평가/반복 설정 분리
- **반복 실험**: 다중 seed 반복 실행 후 평균/표준편차 산출
- **산출물 자동화**:
  - `results_repro/qlearning_metrics_by_seed.csv`
  - `results_repro/summary_metrics.csv`
  - `results_repro/learning_curve_mean_std.png`
- **기존 가이드 호환**: 루트에 `learning_curve.png`도 함께 저장

---

## 3) “논문 재현 완료”로 보려면 추가할 것

1. 논문 원문에서 **토폴로지/트래픽/하이퍼파라미터/비교군/지표**를 표로 추출
2. `ExperimentConfig`를 논문 수치로 고정한 preset 추가
3. 비교군을 논문과 동일한 알고리즘으로 확장
4. 각 실험을 N회 반복해 신뢰구간(또는 표준편차)까지 보고
5. 논문 결과표와 같은 형식으로 자동 리포트 생성
