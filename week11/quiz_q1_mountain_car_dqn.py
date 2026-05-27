"""
Quiz Q1 (PDF p.24) — Mountain Car + DQN
======================================
과제: Hyper-parameter 를 변경하여 최대 total reward policy 를 찾을 것.

제출 (PPT):
  1) 프로그램 소스
  2) 최적 hyperparameter  → 아래 OPTIMAL_* 상수 참고
  3) Episode 별 total reward graph
  4) 최대 total reward 및 policy 동영상

실행:
  source ../.venv/bin/activate   # 또는 conda activate week10-dezero
  python quiz_q1_mountain_car_dqn.py
  python quiz_q1_mountain_car_dqn.py --play
"""
from __future__ import annotations

import argparse
import copy
from collections import deque
from pathlib import Path
import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L

OUT_DIR = Path(__file__).resolve().parent / "results_quiz_q1_mountain_car"

# =============================================================================
# 최적 Hyper-parameter (MountainCar-v0)
# =============================================================================
# 보상 해석 (매 스텝 환경 보상 -1, 깃발 도달 시 terminated=True 로 종료):
#   -200.0 : 200스텝 truncation → 깃발 미도달 (실패)
#   -153.0 : 153스텝에 깃발 도달 (성공, 다만 느림)
#   -110~-125 : 잘 학습된 정책 (빠르게 깃발 도달)
# → 숫자가 0에 가까울수록(덜 음수) 좋음. 로그에 -200만 보여도 최고 기록은 따로 집계됨.
#
# DQN 은 시드·탐험에 따라 편차가 큼 → SEED 고정 + best 모델 저장 + shaping 권장.
# =============================================================================

SEED = 0                  # 재현성 (None 이면 시드 고정 안 함)

# --- DQN 핵심 ---
GAMMA = 0.99              # 할인율
LR = 0.0005               # 0.001 은 불안정할 때 있음 → 0.0005 로 안정화
BUFFER_SIZE = 100_000
BATCH_SIZE = 64           # 32 → 64 (배치 노이즈 감소)
EPISODES = 1500           # 800은 편차 큼, 1500 권장
SYNC_INTERVAL = 100       # target 동기화를 덜 자주 (Q 목표 흔들림 완화)

# --- 신경망 ---
HIDDEN1 = 256
HIDDEN2 = 256

# --- ε-greedy (탐험을 오래 유지 — Mountain Car 핵심) ---
EPS_START = 1.0
EPS_END = 0.02            # 후반에도 약간의 탐험 유지
EPS_DECAY_RATIO = 0.95    # 95% episode 까지 감쇠 (0.7 은 탐험 종료가 너무 일찍)
USE_EPSILON_DECAY = True

# --- 학습 보조 ---
WARMUP_STEPS = 15_000     # 랜덤 워밍업 (왕복 경험 확보)
USE_WARMUP = True
UPDATES_PER_STEP = 4      # 스텝당 학습 횟수 증가
LOG_INTERVAL = 50         # 로그 간격 (너무 잦으면 -200 만 보이는 착시)
GREEDY_EVAL_INTERVAL = 100  # N episode 마다 greedy 소평가 후 best 저장
GREEDY_EVAL_EPISODES = 5    # 위 평가 시 에피소드 수
MIN_EPISODES_BEFORE_GREEDY_EVAL = 200  # 초반은 greedy 평가 생략

# --- Reward shaping (강의 노트: Mountain Car 에 적극 고려) ---
# 목표 방향(오른쪽)으로 위치가 늘면 소량 보너스 → 희소 -1 보상만으로는 학습이 매우 느림
USE_REWARD_SHAPING = True
SHAPING_SCALE = 10.0      # bonus = SHAPING_SCALE * (next_pos - pos), pos 증가 시만

# --- 실패 기준 (로그용) ---
FAILURE_REWARD = -200.0   # truncation 시 total reward


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)
        state = np.stack([x[0] for x in data])
        action = np.array([x[1] for x in data])
        reward = np.array([x[2] for x in data])
        next_state = np.stack([x[3] for x in data])
        done = np.array([x[4] for x in data]).astype(np.int32)
        return state, action, reward, next_state, done


class QNet(Model):
    def __init__(self, action_size, hidden1: int = HIDDEN1, hidden2: int = HIDDEN2):
        super().__init__()
        self.l1 = L.Linear(hidden1)
        self.l2 = L.Linear(hidden2)
        self.l3 = L.Linear(action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class DQNAgent:
    """dqn2.py 와 동일 구조. Mountain Car: state=[위치, 속도], action=3 (좌/정지/우)."""

    def __init__(
        self,
        *,
        state_dim: int,
        action_size: int,
        gamma: float,
        lr: float,
        epsilon: float,
        buffer_size: int,
        batch_size: int,
        hidden1: int,
        hidden2: int,
    ):
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.action_size = action_size
        self.state_dim = state_dim

        self.replay_buffer = ReplayBuffer(buffer_size, batch_size)
        self.qnet = QNet(action_size, hidden1, hidden2)
        self.qnet_target = QNet(action_size, hidden1, hidden2)
        self.optimizer = optimizers.Adam(lr)
        self.optimizer.setup(self.qnet)

    def sync_qnet(self):
        self.qnet_target = copy.deepcopy(self.qnet)

    def get_action(self, state: np.ndarray) -> int:
        # ε-greedy: 확률 epsilon 으로 무작위 행동 (탐험)
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(self.action_size))
        state = state[np.newaxis, :].astype(np.float32)
        qs = self.qnet(state)
        return int(qs.data.argmax())

    def store(self, state, action, reward, next_state, done) -> None:
        self.replay_buffer.add(state, action, reward, next_state, done)

    def learn(self) -> None:
        """Replay buffer 에서 미니배치를 뽑아 TD target 으로 Q-network 1회 학습."""
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.get_batch()
        qs = self.qnet(state)
        q = qs[np.arange(self.batch_size), action]  # Q(s, a) — 실제 취한 행동

        # Target network 로 max_a Q(s', a) 계산 (DQN target 고정)
        next_qs = self.qnet_target(next_state)
        next_q = next_qs.max(axis=1)
        next_q.unchain()
        target = reward + (1 - done) * self.gamma * next_q  # R + γ max Q(s',·)

        loss = F.mean_squared_error(q, target)
        self.qnet.cleargrads()
        loss.backward()
        self.optimizer.update()

    def update(self, state, action, reward, next_state, done, *, n_updates: int = 1) -> None:
        """한 환경 스텝의 경험을 저장하고, n_updates 회 학습."""
        self.store(state, action, reward, next_state, done)
        for _ in range(n_updates):
            self.learn()


def set_global_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def shape_reward(
    state: np.ndarray,
    next_state: np.ndarray,
    reward: float,
    *,
    enabled: bool,
    scale: float,
) -> float:
    """목표(오른쪽) 방향 진행 시 작은 보너스 — Mountain Car DQN 수렴 가속."""
    if not enabled:
        return reward
    delta_pos = float(next_state[0] - state[0])
    if delta_pos > 0:
        return reward + scale * delta_pos
    return reward


def is_success(total_reward: float, terminated: bool) -> bool:
    """깃발 도달(terminated) 이고 truncation(-200) 이 아닐 때 성공."""
    return terminated and total_reward > FAILURE_REWARD + 0.5


def linear_epsilon(
    episode: int,
    episodes: int,
    eps_start: float,
    eps_end: float,
    decay_ratio: float,
) -> float:
    """에피소드 진행에 따라 epsilon을 선형 감쇠. decay_ratio<=0 이면 eps_end 고정."""
    if decay_ratio <= 0:
        return eps_end
    decay_episodes = max(1, int(episodes * decay_ratio))
    progress = min(1.0, episode / decay_episodes)
    return eps_end + (eps_start - eps_end) * (1.0 - progress)


def warmup_replay_buffer(
    agent: DQNAgent,
    env: gym.Env,
    steps: int,
    *,
    seed: int | None,
    use_shaping: bool,
    shaping_scale: float,
) -> None:
    """학습 전 랜덤 행동으로 replay buffer를 채움 (Mountain Car 탐험 강화)."""
    state, _ = env.reset(seed=seed)
    state = np.asarray(state, dtype=np.float32)
    for step in range(steps):
        action = int(env.action_space.sample())
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        ns = np.asarray(next_state, dtype=np.float32)
        r = shape_reward(state, ns, float(reward), enabled=use_shaping, scale=shaping_scale)
        agent.store(state, action, r, ns, done)
        state = ns
        if done:
            state, _ = env.reset(seed=None if seed is None else seed + step + 1)
            state = np.asarray(state, dtype=np.float32)
    print(f"warmup: replay buffer size = {len(agent.replay_buffer)}")


def train(
    agent: DQNAgent,
    env: gym.Env,
    episodes: int,
    sync_interval: int,
    *,
    seed: int | None,
    eps_start: float,
    eps_end: float,
    eps_decay_ratio: float,
    updates_per_step: int,
    use_shaping: bool,
    shaping_scale: float,
    log_interval: int = LOG_INTERVAL,
) -> tuple[list[float], list[bool], DQNAgent, int, DQNAgent]:
    """
    학습 루프.
    반환: (보상 기록, 성공 여부, greedy 최적 에이전트, greedy 최적 ep, 학습 최고 보상 에이전트).
    """
    reward_history: list[float] = []
    success_history: list[bool] = []
    best_train_reward = float("-inf")
    best_train_episode = -1
    best_train_agent: DQNAgent | None = None

    best_greedy_mean = float("-inf")
    best_greedy_episode = -1
    best_greedy_agent: DQNAgent | None = None
    success_count = 0

    for episode in range(episodes):
        agent.epsilon = linear_epsilon(
            episode, episodes, eps_start, eps_end, eps_decay_ratio
        )
        ep_seed = None if seed is None else seed + episode
        state, _ = env.reset(seed=ep_seed)
        state = np.asarray(state, dtype=np.float32)
        done = False
        total_reward = 0.0
        terminated_ep = False

        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            terminated_ep = terminated_ep or terminated
            ns = np.asarray(next_state, dtype=np.float32)
            r = shape_reward(state, ns, float(reward), enabled=use_shaping, scale=shaping_scale)
            agent.update(state, action, r, ns, done, n_updates=updates_per_step)
            state = ns
            total_reward += float(reward)  # 로그/평가는 환경 원 보상(-1/step) 기준

        if episode % sync_interval == 0:
            agent.sync_qnet()

        success = is_success(total_reward, terminated_ep)
        success_history.append(success)
        if success:
            success_count += 1

        reward_history.append(total_reward)
        if total_reward > best_train_reward:
            best_train_reward = total_reward
            best_train_episode = episode
            best_train_agent = copy.deepcopy(agent)

        # 제출/시연용: greedy(ε=0) 성능이 가장 좋았던 시점의 가중치 저장
        if (
            episode > 0
            and episode % GREEDY_EVAL_INTERVAL == 0
            and episode >= MIN_EPISODES_BEFORE_GREEDY_EVAL
        ):
            greedy_mean, _ = evaluate_greedy(
                agent, env, GREEDY_EVAL_EPISODES, seed=None if seed is None else seed + 50_000 + episode
            )
            if greedy_mean > best_greedy_mean:
                best_greedy_mean = greedy_mean
                best_greedy_episode = episode
                best_greedy_agent = copy.deepcopy(agent)

        if episode % log_interval == 0:
            rate = 100.0 * success_count / (episode + 1)
            tag = "SUCCESS" if success else "fail"
            print(
                f"episode :{episode}, total reward : {total_reward} ({tag}), "
                f"epsilon : {agent.epsilon:.3f}, success_rate : {rate:.1f}%"
            )

    assert best_train_agent is not None
    if best_greedy_agent is None:
        best_greedy_agent = best_train_agent
        best_greedy_episode = best_train_episode
    return reward_history, success_history, best_greedy_agent, best_greedy_episode, best_train_agent


def evaluate_greedy(
    agent: DQNAgent,
    env: gym.Env,
    n_episodes: int,
    *,
    seed: int | None,
) -> tuple[float, float]:
    """ε=0 으로 n회 평가 → (평균 total reward, 성공률%)."""
    agent.epsilon = 0.0
    rewards: list[float] = []
    successes = 0
    for i in range(n_episodes):
        state, _ = env.reset(seed=None if seed is None else seed + 10_000 + i)
        state = np.asarray(state, dtype=np.float32)
        done = False
        total = 0.0
        terminated = False
        while not done:
            action = agent.get_action(state)
            ns, reward, term, trunc, _ = env.step(action)
            done = bool(term or trunc)
            terminated = terminated or term
            state = np.asarray(ns, dtype=np.float32)
            total += float(reward)
        rewards.append(total)
        if is_success(total, terminated):
            successes += 1
    return float(np.mean(rewards)), 100.0 * successes / n_episodes


def play_greedy(agent: DQNAgent, env: gym.Env) -> float:
    """학습된 Q-network 로만 행동 (ε=0, 탐험 없음). --play 시 사용."""
    agent.epsilon = 0.0
    state, _ = env.reset()
    state = np.asarray(state, dtype=np.float32)
    done = False
    total_reward = 0.0
    while not done:
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        state = np.asarray(next_state, dtype=np.float32)
        total_reward += float(reward)
        env.render()
    return total_reward


def save_reward_plot(reward_history: list[float], path: Path) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.xlabel("episode")
    plt.ylabel("total reward")
    plt.plot(range(len(reward_history)), reward_history)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"저장: {path}")


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI 기본값 = 파일 상단 OPTIMAL_* 상수 (튜닝 시 --flag 로 덮어쓰기)."""
    ap = argparse.ArgumentParser(description="Quiz Q1: Mountain Car DQN")
    ap.add_argument("--gamma", type=float, default=GAMMA)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument(
        "--epsilon",
        type=float,
        default=EPS_END,
        help="고정 epsilon (--no-epsilon-decay 일 때만 사용)",
    )
    ap.add_argument("--eps-start", type=float, default=EPS_START)
    ap.add_argument("--eps-end", type=float, default=EPS_END)
    ap.add_argument("--eps-decay-ratio", type=float, default=EPS_DECAY_RATIO)
    ap.add_argument(
        "--no-epsilon-decay",
        action="store_true",
        default=not USE_EPSILON_DECAY,
        help="epsilon 고정 (--epsilon 값만 사용)",
    )
    ap.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    ap.add_argument("--no-warmup", action="store_true", default=not USE_WARMUP)
    ap.add_argument("--updates-per-step", type=int, default=UPDATES_PER_STEP)
    ap.add_argument("--buffer-size", type=int, default=BUFFER_SIZE)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--episodes", type=int, default=EPISODES)
    ap.add_argument("--sync-interval", type=int, default=SYNC_INTERVAL)
    ap.add_argument("--hidden1", type=int, default=HIDDEN1)
    ap.add_argument("--hidden2", type=int, default=HIDDEN2)
    ap.add_argument("--log-interval", type=int, default=LOG_INTERVAL)
    ap.add_argument("--seed", type=int, default=SEED, help="재현용 시드 (None: python -c ... -1)")
    ap.add_argument(
        "--no-reward-shaping",
        action="store_true",
        default=not USE_REWARD_SHAPING,
        help="환경 원보상(-1/step)만 사용",
    )
    ap.add_argument("--shaping-scale", type=float, default=SHAPING_SCALE)
    ap.add_argument("--eval-episodes", type=int, default=20, help="학습 후 greedy 평가 횟수")
    ap.add_argument("--play", action="store_true", help="학습 후 greedy 시연 (render)")
    ap.add_argument("--no-plot", action="store_true")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    seed = None if args.seed < 0 else args.seed
    use_shaping = not args.no_reward_shaping

    set_global_seed(seed)
    env = gym.make("MountainCar-v0", render_mode="rgb_array")
    state_dim = int(env.observation_space.shape[0])
    action_size = int(env.action_space.n)

    # epsilon decay 사용 시: 1.0 → 0.05 선형 감쇠 / 미사용 시 EPS_END 고정
    eps_start = args.epsilon if args.no_epsilon_decay else args.eps_start
    eps_end = args.epsilon if args.no_epsilon_decay else args.eps_end

    agent = DQNAgent(
        state_dim=state_dim,
        action_size=action_size,
        gamma=args.gamma,
        lr=args.lr,
        epsilon=eps_start,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
    )

    print("=== 최적 Hyper-parameter (OPTIMAL_* 상수 기준) ===")
    print(f"  gamma={args.gamma}, lr={args.lr}, episodes={args.episodes}, seed={seed}")
    print(f"  eps: {eps_start} → {eps_end} (decay_ratio={args.eps_decay_ratio})")
    print(f"  warmup={args.warmup_steps}, updates/step={args.updates_per_step}")
    print(f"  reward_shaping={use_shaping} (scale={args.shaping_scale})")
    print(f"  해석: -200=실패, -110~-130=성공(빠름), 로그는 {args.log_interval} ep 마다 샘플")
    print("Hyper-parameters (CLI):", vars(args))

    if not args.no_warmup and args.warmup_steps > 0:
        warmup_replay_buffer(
            agent,
            env,
            args.warmup_steps,
            seed=seed,
            use_shaping=use_shaping,
            shaping_scale=args.shaping_scale,
        )
        agent.sync_qnet()

    reward_history, success_history, best_greedy_agent, best_greedy_ep, best_train_agent = train(
        agent,
        env,
        args.episodes,
        args.sync_interval,
        seed=seed,
        eps_start=eps_start,
        eps_end=eps_end,
        eps_decay_ratio=0.0 if args.no_epsilon_decay else args.eps_decay_ratio,
        updates_per_step=max(1, args.updates_per_step),
        use_shaping=use_shaping,
        shaping_scale=args.shaping_scale,
        log_interval=max(1, args.log_interval),
    )
    agent = best_greedy_agent  # 시연/최종 평가는 greedy 성능 최적인 가중치

    best = max(reward_history)
    n_success = sum(success_history)
    print(f"\n학습 요약:")
    print(f"  학습 중 최대 total reward: {best} (탐험 포함, episode {int(np.argmax(reward_history))})")
    print(f"  greedy best snapshot ep: {best_greedy_ep}")
    print(f"  깃발 도달 성공: {n_success}/{len(success_history)} ep ({100*n_success/len(success_history):.1f}%)")
    if best <= FAILURE_REWARD + 0.5:
        print("  ⚠ 아직 -200(실패)만 나왔다면 episodes 를 늘리거나 shaping 을 켜 두세요.")

    mean_eval, eval_success = evaluate_greedy(
        agent, env, args.eval_episodes, seed=seed
    )
    print(f"  최종 greedy 평가({args.eval_episodes}회): mean={mean_eval:.1f}, success={eval_success:.1f}%  ← 제출·시연용")

    plot_path = OUT_DIR / "episode_total_reward.png"
    if not args.no_plot:
        save_reward_plot(reward_history, plot_path)

    hp_path = OUT_DIR / "hyperparameters.txt"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hp_path.write_text(
        "\n".join(
            [
                "Quiz Q1 — Mountain Car DQN (최적 Hyper-parameter)",
                "# -200=실패, -110~-130=성공(빠른 깃발 도달)",
                "",
                f"seed: {SEED}",
                f"gamma: {GAMMA}",
                f"lr: {LR}",
                f"buffer_size: {BUFFER_SIZE}",
                f"batch_size: {BATCH_SIZE}",
                f"episodes: {EPISODES}",
                f"sync_interval: {SYNC_INTERVAL}",
                f"hidden1: {HIDDEN1}",
                f"hidden2: {HIDDEN2}",
                f"eps_start: {EPS_START}",
                f"eps_end: {EPS_END}",
                f"eps_decay_ratio: {EPS_DECAY_RATIO}",
                f"warmup_steps: {WARMUP_STEPS}",
                f"updates_per_step: {UPDATES_PER_STEP}",
                f"use_reward_shaping: {USE_REWARD_SHAPING}",
                f"shaping_scale: {SHAPING_SCALE}",
                "",
                "--- 실제 실행 (CLI) ---",
                *[f"{k}: {v}" for k, v in vars(args).items()],
                "",
                f"max_total_reward_train: {best}",
                f"greedy_best_episode: {best_greedy_ep}",
                f"train_success_count: {n_success}",
                f"final_greedy_eval_mean: {mean_eval}",
                f"final_greedy_eval_success_pct: {eval_success}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"저장: {hp_path}")

    if args.play:
        env2 = gym.make("MountainCar-v0", render_mode="human")
        tr = play_greedy(agent, env2)
        print("Total Reward:", tr)
        env2.close()

    env.close()


if __name__ == "__main__":
    main()
