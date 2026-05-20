"""
Quiz Q3 (08-Q-Network.pdf) — 5×5 Grid World + Q-Network
======================================================
슬라이드 환경:
  - 시작 (4, 0), 벽 (2,1)(2,2), 사과 +1 (0,4), 폭탄 -1 (0,3)(3,4)

요구사항:
  - Q-Network 로 Q(s,a) 학습 후 **Q 테이블(추정)** 과 **greedy policy** 도출
  - 신경망 최적화 하이퍼파라미터는 아래 `HyperConfig` 또는 CLI 로 설정

실행 예:
  python quiz_q3_gridworld.py
  python quiz_q3_gridworld.py --episodes 8000 --lr 0.005 --hidden 128 --no-show
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

# matplotlib 백엔드는 pyplot / mpl_korean_font 로드 **이전**에 설정 (week10/matplotlibrc 대응)
import matplotlib

if "--no-show" in sys.argv:
    matplotlib.use("Agg", force=True)
elif sys.platform == "darwin":
    matplotlib.use("MacOSX", force=True)
else:
    matplotlib.use("TkAgg", force=True)

# ── 경로: week3 환경, week4 렌더러 ─────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "week3"))
from gridworld5x5 import GridWorld5x5  # noqa: E402

_week4_render = _ROOT / "week4" / "common" / "gridworld_render.py"
_spec = importlib.util.spec_from_file_location("week4_gridworld_render", _week4_render)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load Renderer from {_week4_render}")
_gw4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gw4)
Renderer = _gw4.Renderer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mpl_korean_font import configure_korean_font  # noqa: E402

import matplotlib.pyplot as plt

_DEZERO_OK = False
_DEZERO_ERR: str | None = None
try:
    from dezero import Model
    from dezero import optimizers
    import dezero.functions as F
    import dezero.layers as L

    _DEZERO_OK = True
except Exception as e:
    _DEZERO_ERR = repr(e)


OUT_DIR = Path(__file__).resolve().parent / "results_quiz_q3_gridworld"


class SupportsQAgent(Protocol):
    def get_action(self, state_vec: np.ndarray) -> int: ...
    def update(self, state_vec: np.ndarray, action: int, reward: float, next_vec: np.ndarray, done: bool) -> float: ...
    def q_values(self, state_vec: np.ndarray) -> np.ndarray: ...


@dataclass
class HyperConfig:
    """신경망·Q-learning 최적화 파라미터 (슬라이드 과제: 여기/CLI에서 조정)."""

    gamma: float = 0.9
    lr: float = 0.01
    epsilon: float = 0.15
    hidden: int = 100
    episodes: int = 6000
    optimizer: str = "adam"  # sgd | momentum | adagrad | adam
    seed: int = 42


def one_hot_state(state: tuple[int, int], height: int, width: int) -> np.ndarray:
    y, x = state
    vec = np.zeros(height * width, dtype=np.float32)
    vec[width * y + x] = 1.0
    return vec[np.newaxis, :]


def make_optimizer(name: str, lr: float):
    if not _DEZERO_OK:
        raise RuntimeError("internal")
    name = name.lower().strip()
    match name:
        case "sgd":
            return optimizers.SGD(lr)
        case "momentum":
            return optimizers.MomentumSGD(lr, momentum=0.9)
        case "adagrad":
            return optimizers.AdaGrad(lr)
        case "adam":
            return optimizers.Adam(lr)
        case _:
            raise ValueError(f"unknown optimizer: {name}")


# ── DeZero Q-Network ───────────────────────────────────────────────────────
if _DEZERO_OK:

    class QNet(Model):
        def __init__(self, state_dim: int, hidden: int, action_size: int = 4):
            super().__init__()
            self.l1 = L.Linear(hidden)
            self.l2 = L.Linear(action_size)

        def forward(self, x):
            x = F.relu(self.l1(x))
            return self.l2(x)

    class DezeroQNetworkAgent:
        backend = "dezero"

        def __init__(self, env: GridWorld5x5, cfg: HyperConfig):
            self.env = env
            self.cfg = cfg
            self.action_size = 4
            d = env.height * env.width
            self.qnet = QNet(d, cfg.hidden, self.action_size)
            self.optimizer = make_optimizer(cfg.optimizer, cfg.lr)
            self.optimizer.setup(self.qnet)

        def get_action(self, state_vec: np.ndarray) -> int:
            if np.random.rand() < self.cfg.epsilon:
                return int(np.random.randint(0, self.action_size))
            qs = self.qnet(state_vec)
            return int(qs.data.argmax())

        def update(self, state_vec, action: int, reward: float, next_vec, done: bool) -> float:
            if done:
                next_q = np.zeros(1, dtype=np.float32)
            else:
                next_qs = self.qnet(next_vec)
                next_q = next_qs.max(axis=1)
                next_q.unchain()

            target = self.cfg.gamma * next_q + reward
            qs = self.qnet(state_vec)
            q = qs[:, action]
            loss = F.mean_squared_error(target, q)

            self.qnet.cleargrads()
            loss.backward()
            self.optimizer.update()
            return float(loss.data)

        def q_values(self, state_vec: np.ndarray) -> np.ndarray:
            qs = self.qnet(state_vec)
            return np.asarray(qs.data, dtype=np.float32)


# ── NumPy Q-Network (DeZero 미사용/호환 실패 시 동일 TD+MSE 업데이트) ─────
class NumpyQNet:
    """2층 ReLU + 선형 출력 (행동별 Q). 손실 L = ½(Q(s,a) − target)²."""

    def __init__(self, state_dim: int, hidden: int, action_size: int = 4, lr: float = 0.01):
        self.lr = lr
        rng = np.random.default_rng(seed=42)
        s1 = np.sqrt(2.0 / state_dim)
        s2 = np.sqrt(2.0 / hidden)
        self.W1 = (rng.standard_normal((state_dim, hidden)) * s1).astype(np.float32)
        self.b1 = np.zeros((1, hidden), dtype=np.float32)
        self.W2 = (rng.standard_normal((hidden, action_size)) * s2).astype(np.float32)
        self.b2 = np.zeros((1, action_size), dtype=np.float32)
        self.zero_grad_buffers()

    def zero_grad_buffers(self) -> None:
        self._dW1 = np.zeros_like(self.W1)
        self._db1 = np.zeros_like(self.b1)
        self._dW2 = np.zeros_like(self.W2)
        self._db2 = np.zeros_like(self.b2)

    def forward(self, s: np.ndarray) -> tuple[np.ndarray, dict]:
        z1 = s @ self.W1 + self.b1
        h = np.maximum(z1, 0.0)
        q = h @ self.W2 + self.b2
        return q, {"s": s, "z1": z1, "h": h, "q": q}

    def accumulate_td_grad(self, cache: dict, action: int, dLdqa: float) -> None:
        """∂L/∂q_a = dLdqa, 나머지 행동은 0."""
        grad_out = np.zeros_like(cache["q"], dtype=np.float32)
        grad_out[0, action] = np.float32(dLdqa)
        s = cache["s"]
        z1 = cache["z1"]
        h = cache["h"]
        grad_h = grad_out @ self.W2.T
        dz1 = np.where(z1 > 0, grad_h, 0.0).astype(np.float32)
        self._dW2 += h.T @ grad_out
        self._db2 += grad_out.sum(axis=0, keepdims=True)
        self._dW1 += s.T @ dz1
        self._db1 += dz1.sum(axis=0, keepdims=True)

    def step(self) -> None:
        self.W1 -= self.lr * self._dW1
        self.b1 -= self.lr * self._db1
        self.W2 -= self.lr * self._dW2
        self.b2 -= self.lr * self._db2
        self.zero_grad_buffers()


class NumpyQNetworkAgent:
    backend = "numpy"

    def __init__(self, env: GridWorld5x5, cfg: HyperConfig):
        self.cfg = cfg
        self.action_size = 4
        d = env.height * env.width
        self.net = NumpyQNet(d, cfg.hidden, self.action_size, lr=cfg.lr)

    def get_action(self, state_vec: np.ndarray) -> int:
        if np.random.rand() < self.cfg.epsilon:
            return int(np.random.randint(0, self.action_size))
        q, _ = self.net.forward(state_vec)
        return int(np.argmax(q[0]))

    def update(self, state_vec, action: int, reward: float, next_vec, done: bool) -> float:
        if done:
            target = float(reward)
        else:
            qn, _ = self.net.forward(next_vec)
            target = float(reward + self.cfg.gamma * np.max(qn[0]))
        q, cache = self.net.forward(state_vec)
        qa = float(q[0, action])
        d_ld_qa = qa - target
        loss = 0.5 * d_ld_qa**2
        self.net.zero_grad_buffers()
        self.net.accumulate_td_grad(cache, action, d_ld_qa)
        self.net.step()
        return float(loss)

    def q_values(self, state_vec: np.ndarray) -> np.ndarray:
        q, _ = self.net.forward(state_vec)
        return q.astype(np.float32)


def build_agent(env: GridWorld5x5, cfg: HyperConfig) -> SupportsQAgent:
    if _DEZERO_OK:
        return DezeroQNetworkAgent(env, cfg)
    print("[info] DeZero 사용 불가 → NumPy Q-Network(SGD, cfg.optimizer 무시):", _DEZERO_ERR)
    return NumpyQNetworkAgent(env, cfg)


def extract_q_table(agent: SupportsQAgent, env: GridWorld5x5) -> dict:
    """신경망에서 모든 (상태, 행동) 쌍의 Q 추정값을 딕셔너리로 수집."""
    Q: dict = {}
    h, w = env.height, env.width
    for state in env.states():
        if state in env.wall_states:
            continue
        sv = one_hot_state(state, h, w)
        qs = agent.q_values(sv)
        for a in range(4):
            Q[state, a] = float(qs[0, a])
    return Q


def greedy_policy_from_q(Q: dict, env: GridWorld5x5) -> dict:
    pi = {}
    for state in env.states():
        if state in env.wall_states or state in env.terminal_states:
            continue
        qs = [Q[state, a] for a in range(4)]
        best = int(np.argmax(qs))
        pi[state] = {a: (1.0 if a == best else 0.0) for a in range(4)}
    return pi


def print_q_summary(Q: dict, env: GridWorld5x5) -> None:
    """셀별 max_a Q 와 greedy 화살표 (텍스트, Windows 콘솔 호환)."""
    arrows = ["^", "v", "<", ">"]
    print("\n--- Q 테이블 요약: 각 칸 max_a Q(s,a) 및 greedy 행동 ---")
    for y in range(env.height):
        row_q, row_a = [], []
        for x in range(env.width):
            s = (y, x)
            if s in env.wall_states:
                row_q.append("  wall ")
                row_a.append("  ##  ")
                continue
            qs = [Q[s, a] for a in range(4)]
            mx = float(np.max(qs))
            row_q.append(f"{mx:+7.3f}")
            row_a.append(f"  {arrows[int(np.argmax(qs))]}  ")
        print(" | ".join(row_q))
        print(" | ".join(row_a))


def train(
    env: GridWorld5x5,
    agent: SupportsQAgent,
    episodes: int,
    max_steps_per_episode: int = 500,
) -> list[float]:
    losses = []
    for ep in range(episodes):
        state = env.reset()
        sv = one_hot_state(state, env.height, env.width)
        ep_loss, n = 0.0, 0
        done = False
        steps = 0
        while not done and steps < max_steps_per_episode:
            action = agent.get_action(sv)
            next_state, reward, done = env.step(action)
            nv = one_hot_state(next_state, env.height, env.width)
            ep_loss += agent.update(sv, action, float(reward), nv, done)
            n += 1
            sv = nv
            steps += 1
        losses.append(ep_loss / max(n, 1))
        if (ep + 1) % max(1, episodes // 10) == 0:
            print(f"  episode {ep + 1}/{episodes}  mean_loss(last)={losses[-1]:.6f}")
    return losses


def evaluate_greedy(
    env: GridWorld5x5,
    agent: SupportsQAgent,
    trials: int = 200,
    max_steps: int = 500,
) -> tuple[float, float]:
    """ε=0 으로 목표 도달 비율 / 폭탄 비율."""
    goals, traps = 0, 0
    h, w = env.height, env.width
    for _ in range(trials):
        s = env.reset()
        done = False
        steps = 0
        while not done and steps < max_steps:
            sv = one_hot_state(s, h, w)
            qs = agent.q_values(sv)
            a = int(np.argmax(qs[0]))
            s, r, done = env.step(a)
            steps += 1
        if s == env.goal_state:
            goals += 1
        elif s in env.trap_states:
            traps += 1
    return goals / trials, traps / trials


def save_figures(
    env: GridWorld5x5,
    Q: dict,
    pi: dict,
    loss_hist: list[float],
    cfg: HyperConfig,
    *,
    backend: str,
    show: bool,
) -> None:
    configure_korean_font()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    renderer = Renderer(
        env.reward_map,
        env.goal_state,
        env.wall_states,
        trap_states=env.trap_states,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    renderer._draw_q_diamond(Q, axes[0])
    opt_label = cfg.optimizer if backend == "dezero" else "numpy-SGD"
    axes[0].set_title(
        f"Q(s,a) — hidden={cfg.hidden}, lr={cfg.lr}, opt={opt_label}, ε={cfg.epsilon}",
        fontsize=10,
    )
    renderer._draw_greedy_policy(Q, axes[1])
    axes[1].set_title("Greedy policy (학습 후)", fontsize=10)
    fig.suptitle("Quiz Q3 — 5×5 Grid World + Q-Network", fontsize=12)
    fig.tight_layout()
    p1 = OUT_DIR / "quiz_q3_q_and_policy.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    print(f"저장: {p1}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    fig2, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(loss_hist, lw=0.8, alpha=0.85)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean TD MSE loss / step")
    ax.set_title("에피소드별 평균 손실")
    ax.grid(True, alpha=0.35)
    fig2.tight_layout()
    p2 = OUT_DIR / "quiz_q3_loss.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    print(f"저장: {p2}")
    if show:
        plt.show()
    else:
        plt.close(fig2)


def save_hyper_txt(cfg: HyperConfig, goal_rate: float, trap_rate: float, backend: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "Quiz Q3 — Hyperparameters & eval",
        "",
        f"backend: {backend}",
        "",
        *[f"{k}: {v}" for k, v in asdict(cfg).items()],
        "",
        f"greedy_eval_goal_rate: {goal_rate:.4f}",
        f"greedy_eval_trap_rate: {trap_rate:.4f}",
    ]
    p = OUT_DIR / "hyperparameters.txt"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"저장: {p}")


def parse_args() -> tuple[HyperConfig, bool]:
    p = argparse.ArgumentParser(description="Q3: Q-Network on 5x5 Grid World")
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--epsilon", type=float, default=None)
    p.add_argument("--hidden", type=int, default=None)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--optimizer", type=str, default=None, choices=("sgd", "momentum", "adagrad", "adam"))
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-show", action="store_true", help="matplotlib 창 없이 PNG만 저장")
    args = p.parse_args()

    cfg = HyperConfig()
    if args.gamma is not None:
        cfg.gamma = args.gamma
    if args.lr is not None:
        cfg.lr = args.lr
    if args.epsilon is not None:
        cfg.epsilon = args.epsilon
    if args.hidden is not None:
        cfg.hidden = args.hidden
    if args.episodes is not None:
        cfg.episodes = args.episodes
    if args.optimizer is not None:
        cfg.optimizer = args.optimizer
    if args.seed is not None:
        cfg.seed = args.seed

    return cfg, not args.no_show


def main() -> None:
    cfg, show = parse_args()
    np.random.seed(cfg.seed)

    print("HyperConfig:", asdict(cfg))
    env = GridWorld5x5()
    agent = build_agent(env, cfg)
    backend = getattr(agent, "backend", "unknown")
    print(f"백엔드: {backend}")

    print("\n학습 시작 …")
    loss_hist = train(env, agent, cfg.episodes)

    Q = extract_q_table(agent, env)
    pi = greedy_policy_from_q(Q, env)

    print_q_summary(Q, env)

    g_rate, t_rate = evaluate_greedy(env, agent, trials=300)
    print(f"\n[Greedy 평가, 300회] 목표 도달={g_rate:.3f}, 폭탄={t_rate:.3f}")

    save_hyper_txt(cfg, g_rate, t_rate, backend)
    save_figures(env, Q, pi, loss_hist, cfg, backend=backend, show=show)

    print(f"\n완료. 결과 폴더: {OUT_DIR}")


if __name__ == "__main__":
    main()
