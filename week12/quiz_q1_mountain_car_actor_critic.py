"""
Quiz Q1 (PDF slide 28) — Mountain Car + Actor-Critic
=====================================================
(Q1) Actor-Critic 을 Mountain Car 문제에 적용하고 Hyper-parameter 를
     변경하여 최대 total reward policy 를 결정하라.

제출 (PPT):
  1) 프로그램 소스
  2) 최적 hyperparameter
  3) Episode 별 total reward graph
  4) 최대 total reward 값 및 해당 policy 적용 시의 동영상

Mountain Car (PDF slide 27):
  - Reward: -1 for each time step
  - Starting position: uniform in [-0.6, -0.4]
  - Episode ends: position >= 0.5 or length >= 200
  - total reward 가 0에 가까울수록(덜 음수) 좋음. -200=200스텝 실패, -110~-130=성공
  - 학습: 상태 정규화 + 에피소드 MC Actor-Critic + shaping + ε-greedy
  - 로그/제출 total reward 는 환경 원보상(-1/step) 기준

실행:
  conda activate week10-dezero
  python quiz_q1_mountain_car_actor_critic.py                    # 20개 HP 스윕 + 통계/그래프
  python quiz_q1_mountain_car_actor_critic.py --no-per-run-plot  # run별 그래프 생략(빠름)
  python quiz_q1_mountain_car_actor_critic.py --play             # 최적 HP 시연
"""
from __future__ import annotations

import argparse
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

OUT_DIR = Path(__file__).resolve().parent / "results_quiz_q1_mountain_car_ac"
SEED = 0
N_HP_RUNS = 20
DEFAULT_EPISODES = 1500
FAILURE_REWARD = -200.0
USE_REWARD_SHAPING = True
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_RATIO = 0.9
MAX_GRAD_NORM = 1.0
TRAIN_REWARD_CLIP = 5.0

# 스윕 1위(run02: max -85) 및 run10(-114, succ 28%) 주변만 탐색
HP_PRESET_SPECS: list[tuple[float, float, float, int, float, float]] = [
    (0.99, 0.0005, 0.001, 256, 10.0, 0.01),
    (0.99, 0.0005, 0.001, 256, 15.0, 0.01),
    (0.99, 0.0005, 0.001, 256, 20.0, 0.01),
    (0.99, 0.0005, 0.001, 256, 25.0, 0.02),
    (0.99, 0.0004, 0.0008, 256, 10.0, 0.01),
    (0.99, 0.0003, 0.001, 256, 10.0, 0.02),
    (0.99, 0.0003, 0.0005, 256, 20.0, 0.02),
    (0.99, 0.0005, 0.001, 128, 10.0, 0.01),
    (0.98, 0.0005, 0.001, 256, 20.0, 0.01),
    (0.98, 0.0005, 0.001, 256, 15.0, 0.05),
    (0.98, 0.0005, 0.001, 256, 10.0, 0.02),
    (0.98, 0.0005, 0.0012, 256, 18.0, 0.01),
    (0.995, 0.0003, 0.001, 256, 30.0, 0.02),
    (0.995, 0.0005, 0.001, 256, 15.0, 0.05),
    (0.995, 0.0005, 0.0008, 256, 10.0, 0.01),
    (0.99, 0.0006, 0.0012, 256, 12.0, 0.01),
    (0.98, 0.0004, 0.0009, 256, 14.0, 0.02),
    (0.99, 0.0005, 0.001, 256, 20.0, 0.05),
    (0.98, 0.0005, 0.001, 128, 20.0, 0.01),
    (0.995, 0.0004, 0.001, 256, 20.0, 0.03),
]


def build_hp_runs(n: int = N_HP_RUNS) -> list[dict]:
    specs = HP_PRESET_SPECS[:n]
    runs: list[dict] = []
    for i, (gamma, lr_pi, lr_v, hidden, shaping_scale, entropy_coef) in enumerate(specs, 1):
        runs.append(
            {
                "name": (
                    f"run{i:02d}_g{gamma:.3f}_pi{lr_pi:g}_v{lr_v:g}_"
                    f"h{hidden}_sh{int(shaping_scale)}_ent{entropy_coef:g}"
                ),
                "gamma": gamma,
                "lr_pi": lr_pi,
                "lr_v": lr_v,
                "hidden": hidden,
                "shaping_scale": shaping_scale,
                "entropy_coef": entropy_coef,
                "eps_start": EPS_START,
                "eps_end": EPS_END,
                "eps_decay_ratio": EPS_DECAY_RATIO,
            }
        )
    return runs


def normalize_state(state: np.ndarray) -> np.ndarray:
    """MountainCar obs 스케일 차이 완화 (pos, vel)."""
    s = np.asarray(state, dtype=np.float32)
    pos = (s[0] + 0.6) / 1.8
    vel = float(np.clip(s[1] / 0.07, -1.0, 1.0))
    return np.array([pos, vel], dtype=np.float32)


def shape_reward(
    state: np.ndarray,
    next_state: np.ndarray,
    reward: float,
    *,
    enabled: bool,
    scale: float,
) -> float:
    """위치 증가 shaping (학습용). 스텝 보상 클리핑으로 MC return 폭주 방지."""
    if not enabled:
        return float(reward)
    r = float(reward)
    delta_pos = float(next_state[0] - state[0])
    if delta_pos > 0:
        r += scale * delta_pos
    return float(np.clip(r, -TRAIN_REWARD_CLIP, TRAIN_REWARD_CLIP))


def safe_prob_vector(probs: np.ndarray, action_size: int) -> np.ndarray:
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    if p.size != action_size or not np.isfinite(p).all() or float(p.sum()) <= 0:
        return np.full(action_size, 1.0 / action_size, dtype=np.float64)
    p = np.clip(p, 1e-8, 1.0)
    return p / p.sum()


def clip_model_grads(model: Model, max_norm: float) -> None:
    sq = 0.0
    for param in model.params():
        g = param.grad
        if g is None:
            continue
        sq += float(np.sum(g.data.astype(np.float64) ** 2))
    if sq <= 0:
        return
    scale = min(1.0, max_norm / (sq**0.5 + 1e-8))
    if scale >= 1.0:
        return
    for param in model.params():
        g = param.grad
        if g is not None:
            g.data *= scale


def normalize_returns(returns: np.ndarray) -> np.ndarray:
    if len(returns) < 2:
        return returns
    std = float(returns.std())
    if std < 1e-8:
        return returns - float(returns.mean())
    return (returns - float(returns.mean())) / std


def linear_epsilon(
    episode: int,
    episodes: int,
    eps_start: float,
    eps_end: float,
    decay_ratio: float,
) -> float:
    decay_episodes = max(1, int(episodes * decay_ratio))
    progress = min(1.0, episode / decay_episodes)
    return eps_end + (eps_start - eps_end) * (1.0 - progress)


@dataclass
class RunResult:
    name: str
    gamma: float
    lr_pi: float
    lr_v: float
    hidden: int
    shaping_scale: float
    entropy_coef: float
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


class PolicyNet(Model):
    """Actor — policy network π_θ(a|s)."""

    def __init__(self, action_size: int, hidden: int = 128):
        super().__init__()
        self.l1 = L.Linear(hidden)
        self.l2 = L.Linear(action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x))
        return x


class ValueNet(Model):
    """Critic — value network V_w(s)."""

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.l1 = L.Linear(hidden)
        self.l2 = L.Linear(1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class Agent:
    def __init__(
        self,
        action_size: int,
        *,
        gamma: float,
        lr_pi: float,
        lr_v: float,
        hidden: int,
        entropy_coef: float = 0.01,
    ):
        self.gamma = gamma
        self.action_size = action_size
        self.entropy_coef = entropy_coef

        self.pi = PolicyNet(action_size, hidden)
        self.v = ValueNet(hidden)
        self.optimizer_pi = optimizers.Adam(lr_pi)
        self.optimizer_v = optimizers.Adam(lr_v)
        self.optimizer_pi.setup(self.pi)
        self.optimizer_v.setup(self.v)

    def get_action(self, state: np.ndarray, *, epsilon: float = 0.0) -> tuple[int, object]:
        if epsilon > 0.0 and np.random.rand() < epsilon:
            action = int(np.random.randint(self.action_size))
            state_b = normalize_state(state)[np.newaxis, :]
            probs = self.pi(state_b)[0]
            p = safe_prob_vector(probs.data, self.action_size)
            return action, probs[action]

        state_b = normalize_state(state)[np.newaxis, :]
        probs = self.pi(state_b)[0]
        p = safe_prob_vector(probs.data, self.action_size)
        action = int(np.random.choice(self.action_size, p=p))
        return action, probs[action]

    def get_action_greedy(self, state: np.ndarray) -> int:
        state_b = normalize_state(state)[np.newaxis, :]
        probs = self.pi(state_b)[0]
        p = safe_prob_vector(probs.data, self.action_size)
        return int(np.argmax(p))

    def update_episode(self, trajectory: list[tuple[np.ndarray, object, float]]) -> None:
        """에피소드 MC return + return 정규화, 에피소드당 1회 backward (NaN 방지)."""
        n = len(trajectory)
        if n == 0:
            return

        returns = np.zeros(n, dtype=np.float32)
        g = 0.0
        for t in range(n - 1, -1, -1):
            g = float(trajectory[t][2]) + self.gamma * g
            returns[t] = g
        returns = normalize_returns(returns).astype(np.float32)

        self.v.cleargrads()
        self.pi.cleargrads()
        loss_v_acc = 0
        loss_pi_acc = 0

        for t, (state, action_prob, _) in enumerate(trajectory):
            state_b = normalize_state(state)[np.newaxis, :]
            g_t = float(returns[t])
            target = np.array([[g_t]], dtype=np.float32)

            v = self.v(state_b)
            loss_v_acc = loss_v_acc + F.mean_squared_error(v, target)

            delta = target - v
            delta.unchain()
            loss_pi_acc = loss_pi_acc + (-F.log(action_prob) * delta)

            if self.entropy_coef > 0:
                probs = self.pi(state_b)[0]
                entropy = -F.sum(probs * F.log(probs + 1e-8))
                loss_pi_acc = loss_pi_acc - self.entropy_coef * entropy

        loss_v_acc.backward()
        loss_pi_acc.backward()
        clip_model_grads(self.v, MAX_GRAD_NORM)
        clip_model_grads(self.pi, MAX_GRAD_NORM)
        self.optimizer_v.update()
        self.optimizer_pi.update()


def set_global_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def train(
    agent,
    env,
    episodes,
    *,
    use_shaping: bool,
    shaping_scale: float,
    eps_start: float,
    eps_end: float,
    eps_decay_ratio: float,
    log_interval: int = 50,
) -> tuple[list[float], int]:
    reward_history: list[float] = []
    success_count = 0
    for episode in range(episodes):
        epsilon = linear_epsilon(episode, episodes, eps_start, eps_end, eps_decay_ratio)
        state, _ = env.reset()
        state = np.asarray(state, dtype=np.float32)
        done = False
        total_reward = 0.0
        terminated_ep = False
        trajectory: list[tuple[np.ndarray, object, float]] = []

        while not done:
            action, prob = agent.get_action(state, epsilon=epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            terminated_ep = terminated_ep or terminated
            ns = np.asarray(next_state, dtype=np.float32)
            r_train = shape_reward(
                state, ns, float(reward), enabled=use_shaping, scale=shaping_scale
            )
            trajectory.append((state.copy(), prob, r_train))
            state = ns
            total_reward += float(reward)

        if trajectory:
            agent.update_episode(trajectory)

        if terminated_ep and total_reward > FAILURE_REWARD + 0.5:
            success_count += 1
        reward_history.append(total_reward)
        if episode % log_interval == 0:
            rate = 100.0 * success_count / (episode + 1)
            tag = "OK" if total_reward > FAILURE_REWARD + 0.5 else "fail"
            print(
                f"  episode {episode:4d}, total reward: {total_reward:.1f} ({tag}), "
                f"eps={epsilon:.3f}, success_rate={rate:.1f}%"
            )
    return reward_history, success_count


def play_greedy(agent, env) -> float:
    state, _ = env.reset()
    state = np.asarray(state, dtype=np.float32)
    done = False
    total_reward = 0.0
    while not done:
        action = agent.get_action_greedy(state)
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

    print(f"\n{'=' * 60}")
    use_shaping = hp.get("use_shaping", USE_REWARD_SHAPING)
    shaping_scale = hp["shaping_scale"]
    entropy_coef = hp.get("entropy_coef", 0.01)
    eps_start = hp.get("eps_start", EPS_START)
    eps_end = hp.get("eps_end", EPS_END)
    eps_decay = hp.get("eps_decay_ratio", EPS_DECAY_RATIO)
    print(
        f"[{name}] gamma={hp['gamma']}, lr_pi={hp['lr_pi']}, lr_v={hp['lr_v']}, "
        f"hidden={hp['hidden']}, shaping={use_shaping}×{shaping_scale}, ent={entropy_coef}"
    )
    print(
        f"  episodes={episodes}, seed={seed}, MC-AC, eps {eps_start}→{eps_end} "
        f"(decay {eps_decay})"
    )

    agent = Agent(
        action_size,
        gamma=hp["gamma"],
        lr_pi=hp["lr_pi"],
        lr_v=hp["lr_v"],
        hidden=hp["hidden"],
        entropy_coef=entropy_coef,
    )
    reward_history, success_count = train(
        agent,
        env,
        episodes,
        use_shaping=use_shaping,
        shaping_scale=shaping_scale,
        eps_start=eps_start,
        eps_end=eps_end,
        eps_decay_ratio=eps_decay,
    )
    env.close()

    st = stats_from_history(reward_history, success_count=success_count)

    if save_individual_plot:
        save_reward_plot(reward_history, run_dir / "episode_total_reward.png", show=show_plot)

    (run_dir / "hyperparameters.txt").write_text(
        "\n".join(
            [
                "Quiz Q1 — Mountain Car Actor-Critic",
                f"name: {name}",
                f"gamma: {hp['gamma']}",
                f"lr_pi: {hp['lr_pi']}",
                f"lr_v: {hp['lr_v']}",
                f"hidden: {hp['hidden']}",
                f"use_reward_shaping: {use_shaping}",
                f"shaping_scale: {shaping_scale}",
                f"entropy_coef: {entropy_coef}",
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
        lr_pi=hp["lr_pi"],
        lr_v=hp["lr_v"],
        hidden=hp["hidden"],
        shaping_scale=shaping_scale,
        entropy_coef=entropy_coef,
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
    print(f"\n{'=' * 88}")
    print(f"전체 {len(ranked)}개 HP 결과 — 정렬: {rank_key} (높을수록 좋음)")
    print(
        f"{'#':<3} {'이름':<28} {'max':>7} {'mean50':>7} {'succ%':>6} "
        f"{'min':>7}  gamma  lr_pi    lr_v   hid  sh"
    )
    print("-" * 88)
    for i, r in enumerate(ranked, 1):
        print(
            f"{i:<3} {r.name:<28} {r.max_total_reward:7.1f} {r.mean_last_50:7.1f} "
            f"{r.success_rate_pct:5.1f}% {r.min_total_reward:7.1f}  "
            f"{r.gamma:.2f}  {r.lr_pi:g}  {r.lr_v:g}  {r.hidden:3d} {r.shaping_scale:4.0f}"
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
            "lr_pi": r.lr_pi,
            "lr_v": r.lr_v,
            "hidden": r.hidden,
            "shaping_scale": r.shaping_scale,
            "entropy_coef": r.entropy_coef,
        }

    csv_fields = [
        "rank", "name", "max_total_reward", "mean_last_50", "min_total_reward",
        "mean_all", "std_all", "success_count", "success_rate_pct",
        "gamma", "lr_pi", "lr_v", "hidden", "shaping_scale", "entropy_coef",
    ]

    best = publish_sweep_report(
        out_dir=OUT_DIR,
        title="Mountain Car Actor-Critic",
        results=results,
        rank_key=args.rank_by,
        csv_fieldnames=csv_fields,
        row_to_csv=row_csv,
        table_headers=["#", "name", "max", "mean50", "succ%", "gamma", "lr_pi", "lr_v", "hid", "sh"],
        row_to_table=lambda r, i: [
            str(i),
            r.name[:22],
            f"{r.max_total_reward:.1f}",
            f"{r.mean_last_50:.1f}",
            f"{r.success_rate_pct:.1f}",
            f"{r.gamma:.2f}",
            f"{r.lr_pi:g}",
            f"{r.lr_v:g}",
            str(r.hidden),
            f"{r.shaping_scale:.0f}",
        ],
        hp_dict_for_best=lambda r: {
            "name": r.name,
            "gamma": r.gamma,
            "lr_pi": r.lr_pi,
            "lr_v": r.lr_v,
            "hidden": r.hidden,
            "shaping_scale": r.shaping_scale,
            "entropy_coef": r.entropy_coef,
        },
        hp_lines_for_best=lambda r: [
            f"gamma = {r.gamma}",
            f"lr_pi = {r.lr_pi}",
            f"lr_v = {r.lr_v}",
            f"hidden = {r.hidden}",
            f"shaping_scale = {r.shaping_scale}",
            f"entropy_coef = {r.entropy_coef}",
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
        print(f"\n최적 HP({best.name})로 재학습 후 시연...")
        set_global_seed(seed)
        env2 = gym.make("MountainCar-v0", render_mode="human")
        agent = Agent(
            int(env2.action_space.n),
            gamma=best_hp["gamma"],
            lr_pi=best_hp["lr_pi"],
            lr_v=best_hp["lr_v"],
            hidden=best_hp["hidden"],
            entropy_coef=best_hp.get("entropy_coef", 0.01),
        )
        train(
            agent,
            env2,
            args.episodes,
            use_shaping=best_hp.get("use_shaping", USE_REWARD_SHAPING),
            shaping_scale=best_hp["shaping_scale"],
            eps_start=best_hp.get("eps_start", EPS_START),
            eps_end=best_hp.get("eps_end", EPS_END),
            eps_decay_ratio=best_hp.get("eps_decay_ratio", EPS_DECAY_RATIO),
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
        "lr_pi": args.lr_pi,
        "lr_v": args.lr_v,
        "hidden": args.hidden,
        "shaping_scale": args.shaping_scale,
        "entropy_coef": args.entropy_coef,
        "use_shaping": not args.no_shaping,
        "eps_start": EPS_START,
        "eps_end": EPS_END,
        "eps_decay_ratio": EPS_DECAY_RATIO,
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
        agent = Agent(
            int(env2.action_space.n),
            gamma=args.gamma,
            lr_pi=args.lr_pi,
            lr_v=args.lr_v,
            hidden=args.hidden,
            entropy_coef=args.entropy_coef,
        )
        train(
            agent,
            env2,
            args.episodes,
            use_shaping=not args.no_shaping,
            shaping_scale=args.shaping_scale,
            eps_start=EPS_START,
            eps_end=EPS_END,
            eps_decay_ratio=EPS_DECAY_RATIO,
            log_interval=50,
        )
        tr = play_greedy(agent, env2)
        print(f"시연 total reward: {tr:.1f}")
        env2.close()

    return result


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Quiz Q1: Mountain Car Actor-Critic")
    ap.add_argument(
        "--single",
        action="store_true",
        help="HP 스윕 대신 CLI 인자 한 설정만 학습",
    )
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr_pi", type=float, default=0.0005)
    ap.add_argument("--lr_v", type=float, default=0.001)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--shaping-scale", type=float, default=10.0)
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument(
        "--no-shaping",
        action="store_true",
        help="reward shaping 끔 (Mountain Car 에서는 거의 -200만 나옴)",
    )
    ap.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    ap.add_argument("--n-hp", type=int, default=N_HP_RUNS, help=f"스윕 HP 개수 (기본 {N_HP_RUNS})")
    ap.add_argument(
        "--rank-by",
        choices=["max_total_reward", "mean_last_50"],
        default="max_total_reward",
        help="최적 HP 선정 기준",
    )
    ap.add_argument("--seed", type=int, default=SEED, help="재현용 시드 (-1: 고정 안 함)")
    ap.add_argument("--play", action="store_true", help="(스윕) 최적 HP 재학습 후 greedy 시연")
    ap.add_argument("--show-plot", action="store_true", help="matplotlib 창 표시")
    ap.add_argument(
        "--no-per-run-plot",
        action="store_true",
        help="run별 episode 그래프 생략 (20회 스윕 시 속도↑)",
    )
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if args.single:
        run_single(args)
    else:
        n = args.n_hp
        print(
            f"Hyper-parameter 스윕: {n}개 × {args.episodes} ep "
            f"(MC-AC + shaping + eps-greedy, 기준: {args.rank_by})"
        )
        run_sweep(args)


if __name__ == "__main__":
    main()
