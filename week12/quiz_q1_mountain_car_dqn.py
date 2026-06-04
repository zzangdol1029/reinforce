"""
Quiz Q1 (PDF slide 28) — Mountain Car + DQN
===========================================
(Q1) DQN 을 Mountain Car 에 적용하고 Hyper-parameter 를 변경하여
     최대 total reward policy 를 찾을 것.

제출 (PPT):
  1) 프로그램 소스
  2) 최적 hyperparameter
  3) Episode 별 total reward graph
  4) 최대 total reward 값 및 해당 policy 적용 시의 동영상

Mountain Car (PDF slide 27):
  - Reward: -1 for each time step
  - total reward 가 0에 가까울수록(덜 음수) 좋음. -200=실패, -110~-130=성공

실행:
  conda activate week10-dezero
  python quiz_q1_mountain_car_dqn.py                    # 20개 HP 스윕 (1500 ep, shaping+ε-decay)
  python quiz_q1_mountain_car_dqn.py --no-per-run-plot
  python quiz_q1_mountain_car_dqn.py --play
"""
from __future__ import annotations

import argparse
import copy
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L

from mountain_car_sweep_report import publish_sweep_report, stats_from_history

OUT_DIR = Path(__file__).resolve().parent / "results_quiz_q1_mountain_car_dqn"
SEED = 0
N_HP_RUNS = 20
DEFAULT_EPISODES = 1500
FAILURE_REWARD = -200.0
BUFFER_SIZE = 100_000
USE_REWARD_SHAPING = True
EPS_START = 1.0
EPS_DECAY_RATIO = 0.95
WARMUP_STEPS = 15_000
LOG_INTERVAL = 50

# 스윕 1위(run18: max -86) 주변 + week11 검증값 (gamma, lr, eps_end, batch, sync, hidden, shaping, upd/step)
HP_PRESET_SPECS: list[tuple[float, float, float, int, int, int, float, int]] = [
    (0.99, 0.001, 0.02, 32, 10, 256, 10.0, 4),
    (0.99, 0.001, 0.02, 32, 20, 256, 10.0, 4),
    (0.99, 0.001, 0.02, 64, 10, 256, 10.0, 4),
    (0.99, 0.0005, 0.02, 32, 10, 256, 10.0, 4),
    (0.99, 0.0005, 0.05, 32, 10, 256, 10.0, 4),
    (0.99, 0.0005, 0.02, 64, 20, 256, 10.0, 2),
    (0.98, 0.0005, 0.05, 32, 10, 256, 10.0, 4),
    (0.98, 0.0005, 0.05, 32, 10, 128, 10.0, 4),
    (0.98, 0.001, 0.02, 64, 20, 256, 10.0, 4),
    (0.98, 0.0002, 0.15, 32, 10, 256, 10.0, 4),
    (0.99, 0.001, 0.10, 32, 10, 128, 10.0, 4),
    (0.99, 0.0002, 0.02, 64, 10, 128, 10.0, 4),
    (0.99, 0.001, 0.02, 32, 50, 256, 15.0, 4),
    (0.98, 0.001, 0.02, 32, 10, 256, 15.0, 4),
    (0.99, 0.0005, 0.02, 32, 10, 256, 20.0, 4),
    (0.98, 0.0005, 0.10, 64, 20, 128, 10.0, 2),
    (0.99, 0.001, 0.05, 32, 20, 256, 10.0, 4),
    (0.98, 0.0002, 0.05, 32, 20, 256, 10.0, 4),
    (0.99, 0.0005, 0.15, 32, 10, 256, 10.0, 4),
    (0.99, 0.001, 0.02, 64, 10, 256, 10.0, 2),
]


def build_hp_runs(n: int = N_HP_RUNS) -> list[dict]:
    specs = HP_PRESET_SPECS[:n]
    runs: list[dict] = []
    for i, (gamma, lr, eps_end, batch, sync, hidden, shaping, ups) in enumerate(specs, 1):
        runs.append(
            {
                "name": (
                    f"run{i:02d}_g{gamma:.2f}_lr{lr:g}_ee{eps_end:.2f}_"
                    f"b{batch}_s{sync}_h{hidden}_sh{int(shaping)}"
                ),
                "gamma": gamma,
                "lr": lr,
                "epsilon": eps_end,
                "eps_start": EPS_START,
                "eps_end": eps_end,
                "eps_decay_ratio": EPS_DECAY_RATIO,
                "buffer_size": BUFFER_SIZE,
                "batch_size": batch,
                "sync_interval": sync,
                "hidden1": hidden,
                "hidden2": hidden,
                "shaping_scale": shaping,
                "updates_per_step": ups,
                "warmup_steps": WARMUP_STEPS,
                "use_shaping": USE_REWARD_SHAPING,
            }
        )
    return runs


def shape_reward(
    state: np.ndarray,
    next_state: np.ndarray,
    reward: float,
    *,
    enabled: bool,
    scale: float,
) -> float:
    if not enabled:
        return float(reward)
    delta_pos = float(next_state[0] - state[0])
    if delta_pos > 0:
        return float(reward) + scale * delta_pos
    return float(reward)


def is_success(total_reward: float, terminated: bool) -> bool:
    return terminated and total_reward > FAILURE_REWARD + 0.5


def linear_epsilon(
    episode: int,
    episodes: int,
    eps_start: float,
    eps_end: float,
    decay_ratio: float,
) -> float:
    if decay_ratio <= 0:
        return eps_end
    decay_episodes = max(1, int(episodes * decay_ratio))
    progress = min(1.0, episode / decay_episodes)
    return eps_end + (eps_start - eps_end) * (1.0 - progress)


def warmup_replay_buffer(
    agent: "DQNAgent",
    env: gym.Env,
    steps: int,
    *,
    seed: int | None,
    use_shaping: bool,
    shaping_scale: float,
) -> None:
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
    print(f"  warmup: buffer={len(agent.replay_buffer)}")


@dataclass
class RunResult:
    name: str
    gamma: float
    lr: float
    epsilon: float
    buffer_size: int
    batch_size: int
    sync_interval: int
    hidden1: int
    hidden2: int
    episodes: int
    max_total_reward: float
    max_reward_episode: int
    mean_last_50: float
    min_total_reward: float
    mean_all: float
    std_all: float
    success_count: int
    success_rate_pct: float
    reward_history: list[float]
    run_dir: Path


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
    def __init__(self, action_size, hidden1: int = 128, hidden2: int = 128):
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
    def __init__(
        self,
        *,
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
        self.batch_size = batch_size
        self.action_size = action_size

        self.replay_buffer = ReplayBuffer(buffer_size, batch_size)
        self.qnet = QNet(action_size, hidden1, hidden2)
        self.qnet_target = QNet(action_size, hidden1, hidden2)
        self.optimizer = optimizers.Adam(lr)
        self.optimizer.setup(self.qnet)

    def sync_qnet(self):
        self.qnet_target = copy.deepcopy(self.qnet)

    def get_action(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(self.action_size))
        state = state[np.newaxis, :].astype(np.float32)
        qs = self.qnet(state)
        return int(qs.data.argmax())

    def store(self, state, action, reward, next_state, done) -> None:
        self.replay_buffer.add(state, action, reward, next_state, done)

    def learn(self) -> None:
        if len(self.replay_buffer) < self.batch_size:
            return
        state, action, reward, next_state, done = self.replay_buffer.get_batch()
        qs = self.qnet(state)
        q = qs[np.arange(self.batch_size), action]
        next_qs = self.qnet_target(next_state)
        next_q = next_qs.max(axis=1)
        next_q.unchain()
        target = reward + (1 - done) * self.gamma * next_q
        loss = F.mean_squared_error(q, target)
        self.qnet.cleargrads()
        loss.backward()
        self.optimizer.update()

    def update(self, state, action, reward, next_state, done, *, n_updates: int = 1) -> None:
        self.store(state, action, reward, next_state, done)
        for _ in range(max(1, n_updates)):
            self.learn()


def set_global_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


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
) -> tuple[list[float], int]:
    reward_history: list[float] = []
    success_count = 0
    for episode in range(episodes):
        agent.epsilon = linear_epsilon(episode, episodes, eps_start, eps_end, eps_decay_ratio)
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
            r_train = shape_reward(state, ns, float(reward), enabled=use_shaping, scale=shaping_scale)
            agent.update(state, action, r_train, ns, done, n_updates=updates_per_step)
            state = ns
            total_reward += float(reward)

        if episode % sync_interval == 0:
            agent.sync_qnet()

        if is_success(total_reward, terminated_ep):
            success_count += 1
        reward_history.append(total_reward)
        if episode % log_interval == 0:
            rate = 100.0 * success_count / (episode + 1)
            tag = "OK" if is_success(total_reward, terminated_ep) else "fail"
            print(
                f"  episode {episode:4d}, total reward: {total_reward:.1f} ({tag}), "
                f"eps={agent.epsilon:.3f}, success_rate={rate:.1f}%"
            )
    return reward_history, success_count


def play_greedy(agent, env) -> float:
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


def save_reward_plot(reward_history: list[float], path: Path, *, show: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.xlabel("episode")
    plt.ylabel("total reward")
    plt.plot(range(len(reward_history)), reward_history)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"  그래프 저장: {path}")
    if show:
        plt.show()
    plt.close()


def run_experiment(
    hp: dict,
    *,
    episodes: int,
    seed: int | None,
    show_plot: bool,
    save_individual_plot: bool,
) -> RunResult:
    name = hp["name"]
    run_dir = OUT_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(seed)
    env = gym.make("MountainCar-v0", render_mode="rgb_array")
    action_size = int(env.action_space.n)

    use_shaping = hp.get("use_shaping", USE_REWARD_SHAPING)
    shaping_scale = hp["shaping_scale"]
    eps_start = hp.get("eps_start", EPS_START)
    eps_end = hp.get("eps_end", hp["epsilon"])
    print(f"\n{'=' * 60}")
    print(
        f"[{name}] gamma={hp['gamma']}, lr={hp['lr']}, eps {eps_start}→{eps_end}, "
        f"batch={hp['batch_size']}, sync={hp['sync_interval']}, hidden={hp['hidden1']}, "
        f"shaping={use_shaping}×{shaping_scale}, upd/step={hp.get('updates_per_step', 4)}"
    )
    print(f"  episodes={episodes}, seed={seed}")

    agent = DQNAgent(
        action_size=action_size,
        gamma=hp["gamma"],
        lr=hp["lr"],
        epsilon=eps_start,
        buffer_size=hp["buffer_size"],
        batch_size=hp["batch_size"],
        hidden1=hp["hidden1"],
        hidden2=hp["hidden2"],
    )
    if hp.get("warmup_steps", 0) > 0:
        warmup_replay_buffer(
            agent,
            env,
            hp["warmup_steps"],
            seed=seed,
            use_shaping=use_shaping,
            shaping_scale=shaping_scale,
        )
        agent.sync_qnet()

    reward_history, success_count = train(
        agent,
        env,
        episodes,
        hp["sync_interval"],
        seed=seed,
        eps_start=eps_start,
        eps_end=eps_end,
        eps_decay_ratio=hp.get("eps_decay_ratio", EPS_DECAY_RATIO),
        updates_per_step=hp.get("updates_per_step", 4),
        use_shaping=use_shaping,
        shaping_scale=shaping_scale,
        log_interval=LOG_INTERVAL,
    )
    env.close()

    st = stats_from_history(reward_history, success_count=success_count)

    if save_individual_plot:
        save_reward_plot(reward_history, run_dir / "episode_total_reward.png", show=show_plot)

    (run_dir / "hyperparameters.txt").write_text(
        "\n".join(
            [
                "Quiz Q1 — Mountain Car DQN",
                f"name: {name}",
                f"gamma: {hp['gamma']}",
                f"lr: {hp['lr']}",
                f"epsilon_end: {eps_end}",
                f"eps_start: {eps_start}",
                f"shaping_scale: {shaping_scale}",
                f"updates_per_step: {hp.get('updates_per_step', 4)}",
                f"warmup_steps: {hp.get('warmup_steps', 0)}",
                f"buffer_size: {hp['buffer_size']}",
                f"batch_size: {hp['batch_size']}",
                f"sync_interval: {hp['sync_interval']}",
                f"hidden1: {hp['hidden1']}",
                f"hidden2: {hp['hidden2']}",
                f"episodes: {episodes}",
                f"seed: {seed}",
                "",
                f"max_total_reward: {st.max_total_reward}",
                f"max_reward_episode: {st.max_reward_episode}",
                f"mean_last_50: {st.mean_last_50}",
                f"min_total_reward: {st.min_total_reward}",
                f"mean_all: {st.mean_all}",
                f"std_all: {st.std_all}",
                f"success_count: {st.success_count}",
                f"success_rate_pct: {st.success_rate_pct}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"  → max={st.max_total_reward:.1f}, mean50={st.mean_last_50:.1f}, "
        f"success={st.success_count}/{episodes} ({st.success_rate_pct:.1f}%)"
    )

    return RunResult(
        name=name,
        gamma=hp["gamma"],
        lr=hp["lr"],
        epsilon=hp["epsilon"],
        buffer_size=hp["buffer_size"],
        batch_size=hp["batch_size"],
        sync_interval=hp["sync_interval"],
        hidden1=hp["hidden1"],
        hidden2=hp["hidden2"],
        episodes=episodes,
        max_total_reward=st.max_total_reward,
        max_reward_episode=st.max_reward_episode,
        mean_last_50=st.mean_last_50,
        min_total_reward=st.min_total_reward,
        mean_all=st.mean_all,
        std_all=st.std_all,
        success_count=st.success_count,
        success_rate_pct=st.success_rate_pct,
        reward_history=reward_history,
        run_dir=run_dir,
    )


def print_console_ranking(results: list[RunResult], rank_key: str) -> None:
    ranked = sorted(results, key=lambda r: getattr(r, rank_key), reverse=True)
    print(f"\n{'=' * 92}")
    print(f"전체 {len(ranked)}개 HP 결과 — 정렬: {rank_key} (높을수록 좋음)")
    print(
        f"{'#':<3} {'이름':<30} {'max':>7} {'mean50':>7} {'succ%':>6} "
        f"{'min':>7}  g    lr     eps   bat  sync h"
    )
    print("-" * 92)
    for i, r in enumerate(ranked, 1):
        print(
            f"{i:<3} {r.name:<30} {r.max_total_reward:7.1f} {r.mean_last_50:7.1f} "
            f"{r.success_rate_pct:5.1f}% {r.min_total_reward:7.1f}  "
            f"{r.gamma:.2f} {r.lr:g} {r.epsilon:.2f} {r.batch_size:3d} {r.sync_interval:3d} {r.hidden1}"
        )


def run_sweep(args: argparse.Namespace) -> RunResult:
    seed = None if args.seed < 0 else args.seed
    hp_runs = build_hp_runs(args.n_hp)
    results: list[RunResult] = []
    for i, hp in enumerate(hp_runs, 1):
        print(f"\n>>> HP {i}/{len(hp_runs)}: {hp['name']}")
        results.append(
            run_experiment(
                hp,
                episodes=args.episodes,
                seed=seed,
                show_plot=args.show_plot,
                save_individual_plot=not args.no_per_run_plot,
            )
        )

    print_console_ranking(results, args.rank_by)

    def row_csv(r: RunResult, rank: int) -> dict:
        return {
            "rank": rank,
            "name": r.name,
            "max_total_reward": f"{r.max_total_reward:.4f}",
            "mean_last_50": f"{r.mean_last_50:.4f}",
            "min_total_reward": f"{r.min_total_reward:.4f}",
            "mean_all": f"{r.mean_all:.4f}",
            "std_all": f"{r.std_all:.4f}",
            "success_count": r.success_count,
            "success_rate_pct": f"{r.success_rate_pct:.2f}",
            "gamma": r.gamma,
            "lr": r.lr,
            "epsilon": r.epsilon,
            "batch_size": r.batch_size,
            "sync_interval": r.sync_interval,
            "hidden": r.hidden1,
        }

    csv_fields = [
        "rank", "name", "max_total_reward", "mean_last_50", "min_total_reward",
        "mean_all", "std_all", "success_count", "success_rate_pct",
        "gamma", "lr", "epsilon", "batch_size", "sync_interval", "hidden",
    ]

    best = publish_sweep_report(
        out_dir=OUT_DIR,
        title="Mountain Car DQN",
        results=results,
        rank_key=args.rank_by,
        csv_fieldnames=csv_fields,
        row_to_csv=row_csv,
        table_headers=["#", "name", "max", "mean50", "succ%", "gamma", "lr", "eps", "batch", "sync"],
        row_to_table=lambda r, i: [
            str(i),
            r.name[:24],
            f"{r.max_total_reward:.1f}",
            f"{r.mean_last_50:.1f}",
            f"{r.success_rate_pct:.1f}",
            f"{r.gamma:.2f}",
            f"{r.lr:g}",
            f"{r.epsilon:.2f}",
            str(r.batch_size),
            str(r.sync_interval),
        ],
        hp_dict_for_best=lambda r: {
            "name": r.name,
            "gamma": r.gamma,
            "lr": r.lr,
            "epsilon": r.epsilon,
            "buffer_size": r.buffer_size,
            "batch_size": r.batch_size,
            "sync_interval": r.sync_interval,
            "hidden1": r.hidden1,
            "hidden2": r.hidden2,
        },
        hp_lines_for_best=lambda r: [
            f"gamma = {r.gamma}",
            f"lr = {r.lr}",
            f"epsilon_end = {r.epsilon}",
            f"batch_size = {r.batch_size}",
            f"sync_interval = {r.sync_interval}",
            f"hidden = {r.hidden1}",
            f"(see run hyperparameters.txt for shaping/warmup)",
        ],
        histories_for_plot=lambda rs: [
            (r.name, r.reward_history, r.max_total_reward) for r in rs
        ],
        episodes=args.episodes,
        seed=seed,
        show_plot=args.show_plot,
        top_k=min(5, len(results)),
    )

    best_hp = next(h for h in hp_runs if h["name"] == best.name)

    if args.play:
        print(f"\n최적 HP({best.name}) 재학습 후 시연...")
        set_global_seed(seed)
        env2 = gym.make("MountainCar-v0", render_mode="human")
        agent = DQNAgent(
            action_size=int(env2.action_space.n),
            gamma=best_hp["gamma"],
            lr=best_hp["lr"],
            epsilon=best_hp.get("eps_start", EPS_START),
            buffer_size=best_hp["buffer_size"],
            batch_size=best_hp["batch_size"],
            hidden1=best_hp["hidden1"],
            hidden2=best_hp["hidden2"],
        )
        if best_hp.get("warmup_steps", 0) > 0:
            warmup_replay_buffer(
                agent,
                env2,
                best_hp["warmup_steps"],
                seed=seed,
                use_shaping=best_hp.get("use_shaping", True),
                shaping_scale=best_hp["shaping_scale"],
            )
            agent.sync_qnet()
        train(
            agent,
            env2,
            args.episodes,
            best_hp["sync_interval"],
            seed=seed,
            eps_start=best_hp.get("eps_start", EPS_START),
            eps_end=best_hp.get("eps_end", best_hp["epsilon"]),
            eps_decay_ratio=best_hp.get("eps_decay_ratio", EPS_DECAY_RATIO),
            updates_per_step=best_hp.get("updates_per_step", 4),
            use_shaping=best_hp.get("use_shaping", True),
            shaping_scale=best_hp["shaping_scale"],
            log_interval=50,
        )
        tr = play_greedy(agent, env2)
        print(f"시연 total reward: {tr:.1f}")
        env2.close()

    return best


def run_single(args: argparse.Namespace) -> RunResult:
    hp = {
        "name": "single_cli",
        "gamma": args.gamma,
        "lr": args.lr,
        "epsilon": args.eps_end,
        "eps_start": args.eps_start,
        "eps_end": args.eps_end,
        "eps_decay_ratio": args.eps_decay_ratio,
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size,
        "sync_interval": args.sync_interval,
        "hidden1": args.hidden1,
        "hidden2": args.hidden2,
        "shaping_scale": args.shaping_scale,
        "updates_per_step": args.updates_per_step,
        "warmup_steps": args.warmup_steps,
        "use_shaping": not args.no_shaping,
    }
    seed = None if args.seed < 0 else args.seed
    result = run_experiment(
        hp,
        episodes=args.episodes,
        seed=seed,
        show_plot=args.show_plot,
        save_individual_plot=True,
    )
    print(f"\n최대 total reward: {result.max_total_reward:.1f} (episode {result.max_reward_episode})")

    if args.play:
        set_global_seed(seed)
        env2 = gym.make("MountainCar-v0", render_mode="human")
        agent = DQNAgent(
            action_size=int(env2.action_space.n),
            gamma=args.gamma,
            lr=args.lr,
            epsilon=args.eps_start,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            hidden1=args.hidden1,
            hidden2=args.hidden2,
        )
        if args.warmup_steps > 0:
            warmup_replay_buffer(
                agent,
                env2,
                args.warmup_steps,
                seed=seed,
                use_shaping=not args.no_shaping,
                shaping_scale=args.shaping_scale,
            )
            agent.sync_qnet()
        train(
            agent,
            env2,
            args.episodes,
            args.sync_interval,
            seed=seed,
            eps_start=args.eps_start,
            eps_end=args.eps_end,
            eps_decay_ratio=args.eps_decay_ratio,
            updates_per_step=args.updates_per_step,
            use_shaping=not args.no_shaping,
            shaping_scale=args.shaping_scale,
            log_interval=50,
        )
        tr = play_greedy(agent, env2)
        print(f"시연 total reward: {tr:.1f}")
        env2.close()

    return result


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Quiz Q1: Mountain Car DQN")
    ap.add_argument("--single", action="store_true", help="HP 스윕 대신 CLI 한 설정만 학습")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--eps-start", type=float, default=EPS_START)
    ap.add_argument("--eps-end", type=float, default=0.02)
    ap.add_argument("--eps-decay-ratio", type=float, default=EPS_DECAY_RATIO)
    ap.add_argument("--shaping-scale", type=float, default=10.0)
    ap.add_argument("--updates-per-step", type=int, default=4)
    ap.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    ap.add_argument("--no-shaping", action="store_true")
    ap.add_argument("--buffer-size", type=int, default=BUFFER_SIZE)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    ap.add_argument("--n-hp", type=int, default=N_HP_RUNS, help=f"스윕 HP 개수 (기본 {N_HP_RUNS})")
    ap.add_argument(
        "--rank-by",
        choices=["max_total_reward", "mean_last_50"],
        default="max_total_reward",
        help="최적 HP 선정 기준",
    )
    ap.add_argument("--sync-interval", type=int, default=10)
    ap.add_argument("--hidden1", type=int, default=256)
    ap.add_argument("--hidden2", type=int, default=256)
    ap.add_argument("--seed", type=int, default=SEED, help="재현용 시드 (-1: 고정 안 함)")
    ap.add_argument("--play", action="store_true", help="(스윕) 최적 HP 재학습 후 greedy 시연")
    ap.add_argument("--show-plot", action="store_true", help="matplotlib 창 표시")
    ap.add_argument("--no-per-run-plot", action="store_true", help="run별 episode 그래프 생략")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if args.single:
        run_single(args)
    else:
        print(
            f"Hyper-parameter 스윕: {args.n_hp}개 × {args.episodes} ep "
            f"(shaping+ε-decay+warmup, 기준: {args.rank_by})"
        )
        run_sweep(args)


if __name__ == "__main__":
    main()
