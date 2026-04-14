# Q-Learning 기반 SDN 라우팅 최적화 - 실행 가이드

## 📌 빠른 시작 (Quick Start)

### 1. 환경 설정

```bash
# 필수 패키지 설치
pip install numpy matplotlib

# 또는 conda 사용
conda install numpy matplotlib
```

### 2. 코드 실행

```bash
# 시뮬레이션 실행
python sdn_qlearning_implementation.py

# 출력:
# - 학습 곡선: learning_curve.png
# - 콘솔 결과: 상세 성능 분석
```

### 3. 결과 확인

```
Q-Learning 성능 개선율: 45.83%
  - 기존 방법 (Shortest Path): 2.0000 보상
  - Q-Learning: 2.9167 보상
```

---

## 🏗️ 프로젝트 구조

```
sdn_qlearning/
├── sdn_qlearning_presentation.html      # 발표 슬라이드
├── sdn_qlearning_implementation.py      # 메인 구현 코드
├── detailed_explanation.md              # 상세 설명서
├── usage_guide.md                       # 실행 가이드 (이 파일)
├── learning_curve.png                   # 학습 곡선 그래프
└── README.md                            # 프로젝트 설명
```

---

## 📚 파일별 설명

### 1. sdn_qlearning_presentation.html

**목적**: 학술 발표용 슬라이드

**사용 방법**:
```bash
# 브라우저에서 열기
open sdn_qlearning_presentation.html
# 또는
firefox sdn_qlearning_presentation.html
```

**기능**:
- 9개 슬라이드로 구성
- 화살표 키로 네비게이션
- 이전/다음 버튼 제공
- 반응형 디자인

**슬라이드 내용**:
1. 제목 슬라이드
2. 강화학습 배경
3. SDN 개요
4. 문제 정의
5. 강화학습 모델 요소
6. Q-Learning 알고리즘
7. Policy Evaluation & Control
8. 보상 함수 설계
9. 결론 및 성과

### 2. sdn_qlearning_implementation.py

**목적**: Q-Learning 알고리즘 구현 및 시뮬레이션

**주요 클래스**:

#### NetworkEnvironment
```python
# 네트워크 환경 모델링
env = NetworkEnvironment(num_nodes=6)

# 주요 메서드:
env.reset()                          # 에피소드 초기화
env.get_available_actions(node)      # 가능한 다음 노드
env.step(action)                     # 행동 실행
```

**토폴로지**:
```
노드: 0, 1, 2, 3, 4, 5 (6개 라우터)

링크 (출발지 → 도착지):
  0 → 1: BW=10, Delay=5ms
  0 → 2: BW=8,  Delay=8ms
  1 → 2: BW=12, Delay=2ms
  1 → 3: BW=6,  Delay=10ms
  1 → 4: BW=7,  Delay=7ms
  2 → 3: BW=9,  Delay=6ms
  2 → 4: BW=5,  Delay=12ms
  3 → 5: BW=8,  Delay=4ms
  4 → 5: BW=10, Delay=3ms
```

#### QLearningAgent
```python
# Q-Learning 에이전트
agent = QLearningAgent(
    num_nodes=6,
    alpha=0.1,      # 학습률
    gamma=0.9,      # 할인율
    epsilon=0.3     # 탐험률
)

# 주요 메서드:
rewards = agent.train(env, num_episodes=500)    # 학습
path, reward = agent.get_optimal_path(env)      # 경로 계산
```

#### BaselineRouter
```python
# 기준선 알고리즘 (Shortest Path)
baseline = BaselineRouter(env)
path, reward = baseline.find_shortest_path(0, 5)
```

---

## 🔧 커스터마이징 가이드

### 1. 네트워크 토폴로지 변경

```python
class NetworkEnvironment:
    def __init__(self, num_nodes: int = 8):  # 노드 수 변경
        self.num_nodes = num_nodes
        
        # 링크 정보 수정
        self.links = {
            (0, 1): {'bandwidth': 15, 'delay': 3},
            (0, 2): {'bandwidth': 10, 'delay': 5},
            # ... 추가
        }
```

### 2. 하이퍼파라미터 조정

```python
# 학습률 변경 (더 빠른 학습)
agent = QLearningAgent(alpha=0.2)

# 할인율 변경 (미래 보상 중요도)
agent = QLearningAgent(gamma=0.95)

# 탐험률 변경 (탐험 정도)
agent = QLearningAgent(epsilon=0.5)

# 에피소드 수 증가 (더 좋은 수렴)
rewards = agent.train(env, num_episodes=1000)
```

### 3. 보상 함수 수정

```python
def _calculate_reward(self, src: int, dst: int) -> float:
    # 가중치 조정
    w1 = 0.7  # 대역폭 중요도 증가
    w2 = 0.3  # 지연시간 중요도 감소
    
    # 또는 다른 메트릭 추가
    # w3 = 0.3  # 에너지 효율성
    # w4 = 0.2  # 신뢰성
```

### 4. 에이전트 수 증가 (다중 에이전트)

```python
# 여러 출발지-목적지 쌍에 대해 학습
sources_dests = [(0, 5), (1, 5), (2, 5)]

for src, dst in sources_dests:
    agent = QLearningAgent()
    env.reset(src, dst)
    agent.train(env, num_episodes=500)
```

---

## 📊 결과 분석

### 1. 콘솔 출력 해석

```
✓ 500 에피소드 학습 완료
  - 초기 평균 보상: 11.8908    ← 초기 성능
  - 최종 평균 보상: 12.5500    ← 최종 성능

✓ Q-Learning 성능 개선율: 45.83%
  ↑ 기존 방법 대비 개선도
```

### 2. 경로 분석

```
경로: 0 → 1 → 2 → 3 → 5
  0→1: BW=10, Delay=5ms,  보상=0.7083
  1→2: BW=12, Delay=2ms,  보상=0.9167
  2→3: BW=9,  Delay=6ms,  보상=0.6250
  3→5: BW=8,  Delay=4ms,  보상=0.6667
  ─────────────────────────────────
  합계: BW=39, Delay=17ms, 보상=2.9167
```

**해석**:
- 총 대역폭: 39 Mbps (높음)
- 총 지연: 17ms (중간)
- 종합 점수: 2.9167/4.0

### 3. Q-table 해석

```
상태 0: 최적 다음 노드 = 1, Q값 = 9.8113
  ↑ 노드 0에서 최선의 선택은 노드 1로의 이동
  ↑ 이 경로의 기대 누적 보상은 9.81
```

### 4. 학습 곡선 분석

```
학습 곡선 그래프에서:
  - 파란색: 개별 에피소드 보상 (변동성 있음)
  - 빨간색: 이동 평균 (추세)

특징:
  1. 초기: 높은 변동성 (탐험 단계)
  2. 중기: 안정화 (학습 단계)
  3. 후기: 수렴 (활용 단계)
```

---

## 🧪 확장 실험

### 실험 1: 하이퍼파라미터 민감도 분석

```python
import matplotlib.pyplot as plt

alphas = [0.01, 0.05, 0.1, 0.2, 0.5]
results = []

for alpha in alphas:
    agent = QLearningAgent(alpha=alpha)
    rewards = agent.train(env, num_episodes=500)
    results.append(np.mean(rewards[-50:]))

plt.plot(alphas, results)
plt.xlabel('Learning Rate (alpha)')
plt.ylabel('Average Final Reward')
plt.show()
```

### 실험 2: 에피소드 수 영향

```python
episodes = [100, 200, 500, 1000, 2000]
results = []

for ep in episodes:
    agent = QLearningAgent()
    rewards = agent.train(env, num_episodes=ep)
    results.append(np.mean(rewards[-50:]))

plt.plot(episodes, results)
plt.xlabel('Number of Episodes')
plt.ylabel('Average Final Reward')
plt.xscale('log')
plt.show()
```

### 실험 3: 네트워크 규모 확장성

```python
sizes = [5, 10, 15, 20]
training_times = []

for size in sizes:
    env = NetworkEnvironment(num_nodes=size)
    agent = QLearningAgent(num_nodes=size)
    
    start = time.time()
    agent.train(env, num_episodes=500)
    elapsed = time.time() - start
    
    training_times.append(elapsed)

plt.plot(sizes, training_times, marker='o')
plt.xlabel('Network Size (nodes)')
plt.ylabel('Training Time (seconds)')
plt.show()
```

---

## 🐛 문제 해결 (Troubleshooting)

### 1. 수렴이 안 됨

**증상**: 학습 곡선이 평탄하거나 진동

**해결책**:
```python
# 1. 학습률 감소
agent.epsilon *= 0.99  # 탐험률 감소

# 2. 에피소드 수 증가
agent.train(env, num_episodes=1000)

# 3. 보상 함수 재검토
# 보상이 의미 있는 신호를 전달하는지 확인
```

### 2. 메모리 부족

**증상**: MemoryError, Q-table 크기 초과

**해결책**:
```python
# 1. Function approximation 사용
# (신경망으로 Q-value 근사)

# 2. State aggregation
# (유사한 상태를 병합)

# 3. Linear Q-Learning
# (Q-table 대신 선형 함수)
```

### 3. Q-Learning이 발산

**증상**: Q값이 계속 증가

**해결책**:
```python
# 1. 할인율 확인
gamma = 0.9  # 너무 크면 발산

# 2. 보상 정규화
reward = reward / reward_scale

# 3. 학습률 감소 스케줄
alpha = 1.0 / (1 + episode)
```

---

## 📈 성능 최적화 팁

### 1. 학습 속도 향상

```python
# 경험 재현 (Experience Replay) 추가
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
```

### 2. 탐험 전략 개선

```python
# UCB (Upper Confidence Bound) 전략
def select_action_ucb(self, state, available_actions):
    n_visits = {}
    ucb_values = []
    
    for a in available_actions:
        n = n_visits.get(a, 0)
        q = self.q_table[state][a]
        ucb = q + c * np.sqrt(np.log(n+1) / (n+1))
        ucb_values.append(ucb)
    
    return available_actions[np.argmax(ucb_values)]
```

### 3. 다중 에이전트 협력

```python
# 여러 에이전트가 같은 Q-table 공유
shared_q_table = defaultdict(lambda: defaultdict(float))

agent1 = QLearningAgent()
agent1.q_table = shared_q_table

agent2 = QLearningAgent()
agent2.q_table = shared_q_table

# 두 에이전트 모두 같은 Q값 업데이트
```

---

## 📖 추가 학습 자료

### 온라인 리소스

1. **Reinforcement Learning Textbook**
   - URL: http://incompleteideas.net/book/the-book-2nd.html
   - Q-Learning 이론의 고전

2. **OpenAI Gym**
   - URL: https://gym.openai.com/
   - 다양한 RL 환경 제공

3. **DeepMind**
   - URL: https://deepmind.com/blog
   - 최신 강화학습 연구

### 권장 논문

1. Watkins & Dayan (1992) - Q-Learning 원본 논문
2. IEEE Access (2020) - Deep Q-learning for routing schemes
3. Springer (2024) - Improved Exploration Strategy for Q-Learning

---

## 💻 시스템 요구사항

```
최소 요구사항:
  - Python 3.7+
  - 메모리: 512MB
  - 디스크: 50MB
  
권장 사양:
  - Python 3.9+
  - 메모리: 4GB
  - 디스크: 100MB
  
선택 패키지:
  - Jupyter Notebook (상호작용)
  - TensorFlow (신경망 기반 확장)
  - Mininet (실제 네트워크 시뮬레이션)
```

---

## 📞 지원 및 피드백

### 코드 개선 사항

- [ ] 다중 에이전트 지원
- [ ] GPU 가속
- [ ] 실시간 시각화
- [ ] 상태 정규화
- [ ] 자동 하이퍼파라미터 조정

### 알려진 제한사항

1. **확장성**: 상태 공간이 커지면 Q-table 크기 증가
2. **수렴 속도**: 초기 수렴이 느림
3. **탐험**: 무작위 탐험은 비효율적

---

## 📝 라이선스

이 코드는 교육 목적으로 자유롭게 사용할 수 있습니다.

---

## 🎯 최종 체크리스트

발표 준비물:
- [x] 슬라이드 (HTML)
- [x] 실행 코드
- [x] 시뮬레이션 결과
- [x] 학습 곡선 그래프
- [x] 상세 설명서
- [x] 실행 가이드

학습 확인:
- [ ] Q-Learning 기본 이해
- [ ] MDP 정형화 이해
- [ ] Policy Evaluation/Control 이해
- [ ] 코드 실행 및 분석

발표 준비:
- [ ] 슬라이드 검토
- [ ] 시뮬레이션 결과 해석
- [ ] 질문 예상 및 답변 준비
- [ ] 타이밍 연습

---

**마지막 업데이트**: 2024년  
**버전**: 1.0  
**난이도**: 중상
