# week11 — Deep Q-Network (DQN)

> 강의 노트: [`09-Deep-Q-Network.md`](./09-Deep-Q-Network.md)  
> 원본 PDF: [`09-Deep Q-Network.pdf`](./09-Deep%20Q-Network.pdf)

## 파일

| 파일 | 설명 |
|------|------|
| `Replay_buffer.py` | 실습 #1 — Experience Replay (`deque` 버퍼 + CartPole 수집 예제) |
| `dqn2.py` | 실습 #2 — CartPole DQN (Target Network + 학습·그래프·시연) |
| `quiz_q1_mountain_car_dqn.py` | 퀴즈 Q1 — Mountain Car DQN (하이퍼파라미터 튜닝·보상 그래프 저장) |

## 환경 (중요)

| 스크립트 | 필요 패키지 | `(base)` 에서 |
|----------|-------------|----------------|
| `Replay_buffer.py` | gymnasium 만 | ✅ 가능 (`pip install gymnasium`) |
| `dqn2.py`, 퀴즈 | **dezero + NumPy 1.x** | ❌ `np.int` 오류 (NumPy 2) |

**`dqn2.py` 는 `(base)` 가 아니라 `week10-dezero` 환경에서 실행하세요.**

```powershell
cd week11
.\setup_conda.ps1          # 최초 1회 (week10-dezero + gymnasium + dezero 패치)

conda activate week10-dezero
python Replay_buffer.py
python dqn2.py
```

이미 week10 을 설치했다면:

```powershell
conda activate week10-dezero
pip install "gymnasium[classic-control]"
cd week11
python dqn2.py
```

프롬프트가 `(week10-dezero)` 인지 확인하세요. `(base)` 이면 Anaconda base 의 dezero(NumPy 2)가 로드됩니다.

## 실행

```powershell
# 실습 #1
python Replay_buffer.py

# 실습 #2 (학습 + 그래프 + CartPole 시연 창)
python dqn2.py

# 퀴즈 Q1 (권장 기본값 = epsilon decay + warmup 내장, 약 10분)
python quiz_q1_mountain_car_dqn.py
python quiz_q1_mountain_car_dqn.py --play
```

### 퀴즈 Q1 권장 Hyper-parameter (검증: train max **-116**, shaping+seed)

| 변수 | 값 | 비고 |
|------|-----|------|
| `seed` | 0 | 재현성 |
| `gamma` | 0.99 | |
| `lr` | 0.0005 | |
| `eps_start` → `eps_end` | 1.0 → 0.02 | `eps_decay_ratio=0.95` |
| `warmup_steps` | 15000 | |
| `updates_per_step` | 4 | |
| `reward shaping` | on (`scale=10`) | Mountain Car 수렴 가속 |
| `episodes` | 1500 | 800은 편차 큼 |
| `batch_size` | 64 | |

**점수 해석:** `-200`=실패, `-153`=깃발 도달(느림), `-110~-130`=양호. 로그에 `-200`만 많아 보여도 `최대 total reward`/`SUCCESS`를 확인.

교재 슬라이드 기본값: `--no-epsilon-decay --no-warmup --no-reward-shaping --episodes 300`

퀴즈 결과: `results_quiz_q1_mountain_car/episode_total_reward.png`, `hyperparameters.txt`
