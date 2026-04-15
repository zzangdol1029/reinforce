"""
Q-Learning 기반 SDN 라우팅 최적화 (논문 재현 모드 지원)

사용 예시:
    python sdn_qlearning_implementation.py --preset paper_repro
    python sdn_qlearning_implementation.py --preset paper_repro --show
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, List, Tuple
import argparse
import csv
import random

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class LinkMetric:
    bandwidth: float
    delay: float


@dataclass
class ExperimentConfig:
    name: str = "paper_repro"
    num_episodes: int = 500
    max_steps_per_episode: int = 20
    alpha: float = 0.1
    gamma: float = 0.9
    epsilon_start: float = 0.3
    epsilon_decay_every: int = 50
    epsilon_decay: float = 0.95
    epsilon_min: float = 0.01
    seeds: Tuple[int, ...] = (7, 42, 101, 202, 777, 909, 1301, 2024, 3031, 4049)
    evaluation_pairs: Tuple[Tuple[int, int], ...] = ((0, 5),)


# detailed_explanation.md의 표(7.1) 기준 참조값
PAPER_REFERENCE = {
    "path_reward": 2.9167,
    "total_bandwidth": 39.0,
    "total_delay": 17.0,
    "hops": 4.0,
    "baseline_path_reward": 2.0,
    "baseline_total_bandwidth": 27.0,
    "baseline_total_delay": 15.0,
    "baseline_hops": 3.0,
}


class NetworkEnvironment:
    def __init__(self, links: Dict[Tuple[int, int], LinkMetric], num_nodes: int = 6):
        self.links = links
        self.num_nodes = num_nodes
        self.current_node = 0
        self.destination = num_nodes - 1

        self.adjacency = defaultdict(list)
        for (src, dst) in self.links:
            self.adjacency[src].append(dst)

        self.max_bandwidth = max(v.bandwidth for v in links.values())
        self.max_delay = max(v.delay for v in links.values())

    @classmethod
    def default(cls) -> "NetworkEnvironment":
        links = {
            (0, 1): LinkMetric(10, 5),
            (0, 2): LinkMetric(8, 8),
            (1, 3): LinkMetric(6, 10),
            (1, 4): LinkMetric(7, 7),
            (2, 3): LinkMetric(9, 6),
            (2, 4): LinkMetric(5, 12),
            (3, 5): LinkMetric(8, 4),
            (4, 5): LinkMetric(10, 3),
            (1, 2): LinkMetric(12, 2),
        }
        return cls(links=links, num_nodes=6)

    def reset(self, source: int = 0, dest: int | None = None) -> int:
        self.current_node = source
        self.destination = self.num_nodes - 1 if dest is None else dest
        return self.current_node

    def get_available_actions(self, node: int) -> List[int]:
        return self.adjacency[node]

    def reward(self, src: int, dst: int) -> float:
        if (src, dst) not in self.links:
            return -10.0
        info = self.links[(src, dst)]
        bw_score = info.bandwidth / self.max_bandwidth
        delay_score = 1.0 - (info.delay / self.max_delay)
        return 0.5 * bw_score + 0.5 * delay_score

    def step(self, action: int) -> Tuple[int, float, bool]:
        if action not in self.get_available_actions(self.current_node):
            return self.current_node, -10.0, False
        r = self.reward(self.current_node, action)
        self.current_node = action
        done = self.current_node == self.destination
        if done:
            r += 10.0
        return self.current_node, r, done


class QLearningAgent:
    def __init__(self, num_nodes: int, alpha: float, gamma: float, epsilon: float):
        self.num_nodes = num_nodes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: defaultdict(float))

    def select_action(self, state: int, actions: List[int]) -> int:
        if not actions:
            return state
        if random.random() < self.epsilon:
            return random.choice(actions)
        qvals = [self.q_table[state][a] for a in actions]
        max_q = max(qvals)
        best = [a for a, q in zip(actions, qvals) if q == max_q]
        return random.choice(best)

    def update(self, state: int, action: int, reward: float, next_state: int, next_actions: List[int]):
        max_next_q = max([self.q_table[next_state][a] for a in next_actions], default=0.0)
        current = self.q_table[state][action]
        self.q_table[state][action] = current + self.alpha * (reward + self.gamma * max_next_q - current)

    def train(self, env: NetworkEnvironment, cfg: ExperimentConfig) -> List[float]:
        rewards = []
        for ep in range(cfg.num_episodes):
            state = env.reset()
            ep_reward = 0.0
            for _ in range(cfg.max_steps_per_episode):
                acts = env.get_available_actions(state)
                if not acts:
                    break
                act = self.select_action(state, acts)
                next_state, r, done = env.step(act)
                next_acts = env.get_available_actions(next_state)
                self.update(state, act, r, next_state, next_acts)
                ep_reward += r
                state = next_state
                if done:
                    break
            rewards.append(ep_reward)
            if ep > 0 and ep % cfg.epsilon_decay_every == 0:
                self.epsilon = max(cfg.epsilon_min, self.epsilon * cfg.epsilon_decay)
        return rewards

    def greedy_path(self, env: NetworkEnvironment, source: int, dest: int, max_steps: int = 20) -> List[int]:
        path = [source]
        state = source
        step = 0
        while state != dest and step < max_steps:
            acts = env.get_available_actions(state)
            if not acts:
                break
            qvals = [self.q_table[state][a] for a in acts]
            best_action = acts[int(np.argmax(qvals))]
            state = best_action
            path.append(state)
            step += 1
        return path


class BaselineRouter:
    def __init__(self, env: NetworkEnvironment):
        self.env = env

    def shortest_hop_path(self, source: int, dest: int) -> List[int]:
        q = deque([(source, [source])])
        visited = {source}
        while q:
            cur, path = q.popleft()
            if cur == dest:
                return path
            for nxt in self.env.get_available_actions(cur):
                if nxt not in visited:
                    visited.add(nxt)
                    q.append((nxt, path + [nxt]))
        return [source, dest]


def path_metrics(env: NetworkEnvironment, path: List[int]) -> Dict[str, float]:
    total_bw = 0.0
    total_delay = 0.0
    total_reward = 0.0
    for s, d in zip(path[:-1], path[1:]):
        if (s, d) not in env.links:
            continue
        link = env.links[(s, d)]
        total_bw += link.bandwidth
        total_delay += link.delay
        total_reward += env.reward(s, d)
    return {
        "hops": float(max(0, len(path) - 1)),
        "total_bandwidth": float(total_bw),
        "total_delay": float(total_delay),
        "path_reward": float(total_reward),
    }


def aggregate(metric_list: List[Dict[str, float]]) -> Dict[str, float]:
    keys = metric_list[0].keys()
    out = {}
    for k in keys:
        vals = [m[k] for m in metric_list]
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals))
    return out


def save_csv(path: Path, rows: List[Dict[str, float]]):
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def plot_learning_curve(all_rewards: List[List[float]], out_path: Path, show: bool):
    arr = np.array(all_rewards)
    mean_curve = arr.mean(axis=0)
    std_curve = arr.std(axis=0)
    x = np.arange(len(mean_curve))

    plt.figure(figsize=(10, 5))
    plt.plot(x, mean_curve, color="royalblue", label="mean reward")
    plt.fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                     color="royalblue", alpha=0.2, label="+-1 std")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Paper Repro: Learning Curve (mean +- std across seeds)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    if show:
        plt.show()
    else:
        plt.close()


def plot_paper_comparison(q_agg: Dict[str, float], base: Dict[str, float], out_path: Path, show: bool):
    labels = ["path_reward", "total_bandwidth", "total_delay", "hops"]
    q_vals = [q_agg[f"{k}_mean"] for k in labels]
    b_vals = [base[k] for k in labels]
    p_q_vals = [PAPER_REFERENCE[k] for k in labels]
    p_b_vals = [PAPER_REFERENCE[f"baseline_{k}"] for k in labels]

    x = np.arange(len(labels))
    w = 0.18
    plt.figure(figsize=(11, 5.3))
    plt.bar(x - 1.5 * w, q_vals, width=w, label="Q-Learning (exp)", color="#1f77b4")
    plt.bar(x - 0.5 * w, b_vals, width=w, label="Baseline (exp)", color="#ff7f0e")
    plt.bar(x + 0.5 * w, p_q_vals, width=w, label="Q-Learning (paper)", color="#2ca02c", alpha=0.75)
    plt.bar(x + 1.5 * w, p_b_vals, width=w, label="Baseline (paper)", color="#d62728", alpha=0.75)

    plt.xticks(x, ["Reward", "Bandwidth", "Delay", "Hops"])
    plt.title("Paper Repro Comparison: Experiment vs Paper Reference")
    plt.ylabel("Metric Value")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    if show:
        plt.show()
    else:
        plt.close()


def plot_seed_boxplot(seed_rows: List[Dict[str, float]], out_path: Path, show: bool):
    metrics = ["path_reward", "total_bandwidth", "total_delay", "hops"]
    data = [[row[m] for row in seed_rows] for m in metrics]

    plt.figure(figsize=(10, 4.8))
    plt.boxplot(data, tick_labels=["Reward", "Bandwidth", "Delay", "Hops"], showmeans=True)
    plt.title("Seed Variability (Q-Learning, paper_repro preset)")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    if show:
        plt.show()
    else:
        plt.close()


def write_report_md(path: Path, cfg: ExperimentConfig, q_agg: Dict[str, float], base: Dict[str, float]):
    def imp(q, b):
        if b == 0:
            return 0.0
        return (q - b) / abs(b) * 100.0

    txt = f"""# paper_repro 결과 리포트

## 설정

- preset: `{cfg.name}`
- episodes: `{cfg.num_episodes}`
- alpha/gamma: `{cfg.alpha}` / `{cfg.gamma}`
- epsilon: `{cfg.epsilon_start}` (every `{cfg.epsilon_decay_every}` ep x `{cfg.epsilon_decay}`)
- seeds: `{cfg.seeds}`
- evaluation_pairs: `{cfg.evaluation_pairs}`

## 핵심 수치 (실험 평균)

| metric | Q-Learning(mean+-std) | Baseline | 개선율(대비 Baseline) |
|---|---:|---:|---:|
| path_reward | {q_agg['path_reward_mean']:.4f} +- {q_agg['path_reward_std']:.4f} | {base['path_reward']:.4f} | {imp(q_agg['path_reward_mean'], base['path_reward']):.2f}% |
| total_bandwidth | {q_agg['total_bandwidth_mean']:.4f} +- {q_agg['total_bandwidth_std']:.4f} | {base['total_bandwidth']:.4f} | {imp(q_agg['total_bandwidth_mean'], base['total_bandwidth']):.2f}% |
| total_delay | {q_agg['total_delay_mean']:.4f} +- {q_agg['total_delay_std']:.4f} | {base['total_delay']:.4f} | {imp(q_agg['total_delay_mean'], base['total_delay']):.2f}% |
| hops | {q_agg['hops_mean']:.4f} +- {q_agg['hops_std']:.4f} | {base['hops']:.4f} | {imp(q_agg['hops_mean'], base['hops']):.2f}% |

## 생성 이미지

- `paper_repro_learning_curve.png`
- `paper_repro_comparison_bars.png`
- `paper_repro_seed_boxplot.png`

## 해석 가이드

1. `comparison_bars`: 논문표 참조값과 현재 실험값의 거리 확인
2. `learning_curve`: 학습 수렴 형태와 변동성 확인
3. `seed_boxplot`: 재현성(시드 민감도) 확인
"""
    path.write_text(txt, encoding="utf-8")


def build_preset(name: str) -> ExperimentConfig:
    if name == "paper_repro":
        return ExperimentConfig(name="paper_repro")
    if name == "fast_check":
        return ExperimentConfig(name="fast_check", num_episodes=200, seeds=(7, 42, 101))
    raise ValueError(f"Unknown preset: {name}")


def run(cfg: ExperimentConfig, show: bool = False):
    out_dir = Path(__file__).resolve().parent / "results_repro" / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)

    env = NetworkEnvironment.default()
    baseline = BaselineRouter(env)

    all_rewards: List[List[float]] = []
    seed_rows: List[Dict[str, float]] = []

    for seed in cfg.seeds:
        random.seed(seed)
        np.random.seed(seed)
        agent = QLearningAgent(env.num_nodes, cfg.alpha, cfg.gamma, cfg.epsilon_start)
        rewards = agent.train(env, cfg)
        all_rewards.append(rewards)

        for src, dst in cfg.evaluation_pairs:
            path = agent.greedy_path(env, src, dst)
            m = path_metrics(env, path)
            m["seed"] = float(seed)
            m["src"] = float(src)
            m["dst"] = float(dst)
            seed_rows.append(m)

    b_rows = []
    for src, dst in cfg.evaluation_pairs:
        p = baseline.shortest_hop_path(src, dst)
        b_rows.append(path_metrics(env, p))
    base = {
        "path_reward": float(np.mean([r["path_reward"] for r in b_rows])),
        "total_bandwidth": float(np.mean([r["total_bandwidth"] for r in b_rows])),
        "total_delay": float(np.mean([r["total_delay"] for r in b_rows])),
        "hops": float(np.mean([r["hops"] for r in b_rows])),
    }

    q_agg = aggregate([{k: row[k] for k in ("path_reward", "total_bandwidth", "total_delay", "hops")} for row in seed_rows])

    save_csv(out_dir / "paper_repro_metrics_by_seed.csv", seed_rows)
    save_csv(out_dir / "paper_repro_summary.csv", [{
        "method": "Q-Learning",
        **q_agg,
    }, {
        "method": "Baseline",
        **{f"{k}_mean": v for k, v in base.items()},
        **{f"{k}_std": 0.0 for k in base.keys()},
    }])

    plot_learning_curve(all_rewards, out_dir / "paper_repro_learning_curve.png", show=show)
    plot_paper_comparison(q_agg, base, out_dir / "paper_repro_comparison_bars.png", show=show)
    plot_seed_boxplot(seed_rows, out_dir / "paper_repro_seed_boxplot.png", show=show)
    write_report_md(out_dir / "paper_repro_report.md", cfg, q_agg, base)

    print("=" * 88)
    print(f"완전 재현 모드 실행 완료: {cfg.name}")
    print("=" * 88)
    print("설정:", asdict(cfg))
    print("\n[Q-Learning mean +- std]")
    for key in ("path_reward", "total_bandwidth", "total_delay", "hops"):
        print(f"- {key:>16}: {q_agg[key + '_mean']:.4f} +- {q_agg[key + '_std']:.4f}")
    print("\n[Baseline mean]")
    for key, val in base.items():
        print(f"- {key:>16}: {val:.4f}")
    print(f"\n결과 폴더: {out_dir}")
    print("- paper_repro_learning_curve.png")
    print("- paper_repro_comparison_bars.png")
    print("- paper_repro_seed_boxplot.png")
    print("- paper_repro_report.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SDN Q-Learning paper reproduction runner")
    parser.add_argument("--preset", default="paper_repro", choices=["paper_repro", "fast_check"])
    parser.add_argument("--show", action="store_true", help="그래프 창 표시")
    args = parser.parse_args()

    cfg = build_preset(args.preset)
    run(cfg, show=args.show)
