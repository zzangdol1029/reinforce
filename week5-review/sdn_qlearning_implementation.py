"""
Q-Learning 기반 SDN 라우팅 최적화 (논문 재현 모드 지원)

구성 요약:
    - NetworkEnvironment: 노드·링크 그래프, 보상, step/reset (강화학습 MDP 역할)
    - QLearningAgent: 표 형태 Q-learning, ε-greedy + ε 감쇠
    - BaselineRouter: 최단 홉(BFS) 경로 — Q-learning 결과와 비교용
    - run(): 단일 설정으로 학습·지표·그래프·리포트 저장
    - run_epsilon_sweep(): ε만 바꿔 여러 번 실험 후 CSV/PNG 비교

사용 예시:
    python sdn_qlearning_implementation.py --preset paper_repro
    python sdn_qlearning_implementation.py --preset paper_repro --show
    python sdn_qlearning_implementation.py --preset paper_repro --sweep-epsilon
    python sdn_qlearning_implementation.py --preset fast_check --sweep-epsilon --epsilons 0.1,0.3,0.5
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, List, Tuple
import argparse
import csv
import random

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 데이터 구조: 링크 속성 및 실험 하이퍼파라미터 묶음
# ---------------------------------------------------------------------------


@dataclass
class LinkMetric:
    """단방향 링크 (src -> dst) 의 정적 특성. 보상 계산·경로 지표에 사용."""

    bandwidth: float  # 대역폭 (합산 지표용, 보상 정규화에도 사용)
    delay: float  # 지연 (합산 시 경로 total_delay, 보상에서는 낮을수록 유리하게 반영)


@dataclass
class ExperimentConfig:
    """
    한 번의 실험(또는 스윕 시 한 설정)에 쓰이는 모든 하이퍼파라미터.

    - 강화학습: alpha(TD 학습률), gamma(할인율), epsilon_* (탐색률 스케줄)
    - 통계: seeds(재현·분산 추정), evaluation_pairs(출발-도착 쌍, 경로 평가)
    """

    name: str = "paper_repro"  # 결과 폴더 이름·로그용 식별자
    num_episodes: int = 500  # 학습 에피소드 수 (늘리면 수렴 가능성↑, 시간↑)
    max_steps_per_episode: int = 20  # 한 에피소드 최대 전이 수 (그래프 직경 상한)
    alpha: float = 0.1  # Q 갱신 시 TD 오차에 곱하는 학습률
    gamma: float = 0.9  # 미래 보상 할인 계수
    epsilon_start: float = 0.3  # 초기 ε-greedy 탐색 확률 (스윕 시 이 값만 바꿔 비교 가능)
    epsilon_decay_every: int = 50  # 이 간격(에피소드)마다 ε를 곱해 감소
    epsilon_decay: float = 0.95  # ε <- max(epsilon_min, ε * epsilon_decay)
    epsilon_min: float = 0.01  # 탐색률 하한 (완전 탐욕으로 가기 전 최소 노이즈)
    seeds: Tuple[int, ...] = (7, 42, 101, 202, 777, 909, 1301, 2024, 3031, 4049)
    # 서로 다른 난수 시드로 학습을 반복해 평균·표준편차를 냄 (논문 재현·안정성)
    evaluation_pairs: Tuple[Tuple[int, int], ...] = ((0, 5),)
    # 학습 후 greedy 경로를 뽑아 평가할 (source, destination) 쌍; 복수 쌍이면 지표는 평균


# detailed_explanation.md의 표(7.1) 기준 참조값 — 막대 그래프에서 실험값과 나란히 표시
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


# ---------------------------------------------------------------------------
# 환경: 상태=현재 노드, 행동=인접 노드로 이동
# ---------------------------------------------------------------------------


class NetworkEnvironment:
    """
    간단한 네트워크 그래프 환경.

    상태: 현재 스위치(노드) id
    행동: 인접 노드 중 하나 (유효하지 않은 행동은 step에서 페널티)
    종료: current_node == destination
    """

    def __init__(self, links: Dict[Tuple[int, int], LinkMetric], num_nodes: int = 6):
        self.links = links  # (src, dst) -> 링크 메타 (양방향이 아니면 역방향 키 없음)
        self.num_nodes = num_nodes
        self.current_node = 0
        self.destination = num_nodes - 1

        # 인접 리스트: 행동 공간 = adjacency[node]
        self.adjacency = defaultdict(list)
        for (src, dst) in self.links:
            self.adjacency[src].append(dst)

        # reward()에서 대역폭·지연을 [0,1] 근처로 정규화할 때 쓰는 분모
        self.max_bandwidth = max(v.bandwidth for v in links.values())
        self.max_delay = max(v.delay for v in links.values())

    @classmethod
    def default(cls) -> "NetworkEnvironment":
        """논문 재현용 6노드 예시 토폴로지. 링크 (대역폭, 지연) 튜플."""
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
        """에피소드 시작: 출발·도착 설정 후 초기 상태(노드) 반환."""
        self.current_node = source
        self.destination = self.num_nodes - 1 if dest is None else dest
        return self.current_node

    def get_available_actions(self, node: int) -> List[int]:
        """주어진 노드에서 이동 가능한 이웃 노드 목록 = 행동 집합."""
        return self.adjacency[node]

    def reward(self, src: int, dst: int) -> float:
        """
        한 홉 (src -> dst) 에 대한 즉시 보상.

        대역폭은 클수록, 지연은 작을수록 좋게 설계.
        존재하지 않는 링크는 강한 음의 보상.
        """
        if (src, dst) not in self.links:
            return -10.0
        info = self.links[(src, dst)]
        bw_score = info.bandwidth / self.max_bandwidth
        delay_score = 1.0 - (info.delay / self.max_delay)
        return 0.5 * bw_score + 0.5 * delay_score

    def step(self, action: int) -> Tuple[int, float, bool]:
        """
        행동 실행 후 (다음 상태, 보상, 종료 여부).

        종료 시 목표 도달 보너스 +10 을 같은 스텝 보상에 더함 → 목표 도달을 강하게 유도.
        """
        if action not in self.get_available_actions(self.current_node):
            return self.current_node, -10.0, False
        r = self.reward(self.current_node, action)
        self.current_node = action
        done = self.current_node == self.destination
        if done:
            r += 10.0
        return self.current_node, r, done


# ---------------------------------------------------------------------------
# 에이전트: 테이블 Q-learning (Off-policy, max over next actions)
# ---------------------------------------------------------------------------


class QLearningAgent:
    """
    이산 상태·행동 공간에서의 Q-learning.

    Q(s,a) <- Q(s,a) + alpha * ( r + gamma * max_a' Q(s',a') - Q(s,a) )
    행동 선택: epsilon 확률로 무작위, 아니면 Q가 최대인 행동(동률이면 무작위 타이브레이크).
    """

    def __init__(self, num_nodes: int, alpha: float, gamma: float, epsilon: float):
        self.num_nodes = num_nodes  # (현재 코드 경로에서는 주로 호환용; Q는 defaultdict)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # q_table[state][action] -> float; 미방문은 0.0
        self.q_table = defaultdict(lambda: defaultdict(float))

    def select_action(self, state: int, actions: List[int]) -> int:
        """ε-greedy 행동 선택."""
        if not actions:
            return state
        if random.random() < self.epsilon:
            return random.choice(actions)
        qvals = [self.q_table[state][a] for a in actions]
        max_q = max(qvals)
        best = [a for a, q in zip(actions, qvals) if q == max_q]
        return random.choice(best)

    def update(self, state: int, action: int, reward: float, next_state: int, next_actions: List[int]):
        """한 번의 전이에 대한 Q-learning TD 갱신."""
        max_next_q = max([self.q_table[next_state][a] for a in next_actions], default=0.0)
        current = self.q_table[state][action]
        self.q_table[state][action] = current + self.alpha * (reward + self.gamma * max_next_q - current)

    def train(self, env: NetworkEnvironment, cfg: ExperimentConfig) -> List[float]:
        """
        cfg.num_episodes 만큼 에피소드 학습.

        반환: 에피소드별 누적 보상 리스트 — 학습 곡선 플롯에 사용.
        주의: 에이전트 내부 self.epsilon은 학습 중 감쇠되므로, 같은 에이전트로 재학습하지 않는 한
        매 실행은 새 QLearningAgent로 시작하는 구조가 안전함.
        """
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
            # 탐색을 서서히 줄여 후반에는 활용 비중 증가
            if ep > 0 and ep % cfg.epsilon_decay_every == 0:
                self.epsilon = max(cfg.epsilon_min, self.epsilon * cfg.epsilon_decay)
        return rewards

    def greedy_path(self, env: NetworkEnvironment, source: int, dest: int, max_steps: int = 20) -> List[int]:
        """
        학습된 Q에 대해 탐욕적으로 다음 행동만 고르는 경로 (평가용).

        ε와 무관하게 pure greedy — 논문/보고서에서 '학습 후 정책' 시각화·지표에 사용.
        """
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


# ---------------------------------------------------------------------------
# 베이스라인: 대역폭·지연 무시하고 홉 수만 최소화하는 최단 경로 (BFS)
# ---------------------------------------------------------------------------


class BaselineRouter:
    """Q-learning과 동일 그래프에서 '최단 홉' 경로만 계산해 비교 기준선으로 쓴다."""

    def __init__(self, env: NetworkEnvironment):
        self.env = env

    def shortest_hop_path(self, source: int, dest: int) -> List[int]:
        """가중치 없는 그래프에서 BFS로 최소 홉 경로. 도달 실패 시 [source, dest] 폴백."""
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


# ---------------------------------------------------------------------------
# 지표·저장·플롯
# ---------------------------------------------------------------------------


def path_metrics(env: NetworkEnvironment, path: List[int]) -> Dict[str, float]:
    """
    경로(노드 나열)에 대해 링크별 합산 지표 계산.

    - hops: 홉 수
    - total_bandwidth / total_delay: 경로에 포함된 링크 속성의 단순 합 (논문 지표와 맞춤)
    - path_reward: 각 홉의 env.reward 합 (목표 보너스는 경로에 포함되지 않음 — greedy_path는 홉 보상만)
    """
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
    """여러 시드(또는 샘플)의 동일 키에 대해 mean/std를 붙여 반환."""
    keys = metric_list[0].keys()
    out = {}
    for k in keys:
        vals = [m[k] for m in metric_list]
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals))
    return out


def save_csv(path: Path, rows: List[Dict[str, float]]):
    """Dict 리스트를 UTF-8 CSV로 저장. 첫 행의 키 순서가 헤더가 됨."""
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def plot_learning_curve(all_rewards: List[List[float]], out_path: Path, show: bool):
    """시드별 에피소드 보상 곡선을 쌓아 평균·표준편차 밴드로 표시."""
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
    """
    실험 Q-learning vs 실험 Baseline vs 논문 표 참조값 4묶음 막대 그래프.

    한 축에 네 종류가 있어 논문 재현 오차를 눈으로 비교하기 위함.
    """
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


def collect_experiment_results(
    cfg: ExperimentConfig,
) -> Tuple[Dict[str, float], Dict[str, float], List[Dict[str, float]], List[List[float]]]:
    """
    단일 ExperimentConfig에 대해 전체 파이프라인 실행.

    반환:
        q_agg: 시드·평가쌍에 걸친 Q 경로 지표의 평균/표준편차
        base: 동일 evaluation_pairs에 대한 최단환 경로 지표 평균
        seed_rows: 시드마다 한 행(경로 지표 + seed, src, dst)
        all_rewards: 시드마다 에피소드 보상 시계열 — 학습 곡선용
    """
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

    q_agg = aggregate(
        [{k: row[k] for k in ("path_reward", "total_bandwidth", "total_delay", "hops")} for row in seed_rows]
    )

    return q_agg, base, seed_rows, all_rewards


def _gain_lower_is_better(q: float, b: float) -> float:
    """지연 등 '작을수록 좋은' 지표: baseline 대비 감소율(%). 양수면 Q 쪽이 더 작음(유리)."""
    if b == 0:
        return 0.0
    return (b - q) / abs(b) * 100.0


def _gain_higher_is_better(q: float, b: float) -> float:
    """보상 등 '클수록 좋은' 지표: baseline 대비 증가율(%)."""
    if b == 0:
        return 0.0
    return (q - b) / abs(b) * 100.0


def plot_epsilon_sweep_bars(
    sweep_rows: List[Dict[str, float]],
    base: Dict[str, float],
    out_path: Path,
    show: bool,
) -> None:
    """ε별 평균 total_delay 막대 + baseline 수평선."""
    labels = [f"ε={r['epsilon']:.2f}" for r in sweep_rows]
    delays = [r["total_delay_mean"] for r in sweep_rows]
    x = np.arange(len(labels))
    plt.figure(figsize=(max(8.0, 1.4 * len(labels)), 5.0))
    plt.bar(x, delays, color="steelblue", label="Q-Learning (mean delay)")
    plt.axhline(base["total_delay"], color="darkorange", linestyle="--", linewidth=2, label="Baseline (shortest-hop)")
    plt.xticks(x, labels)
    plt.ylabel("Total delay (path sum)")
    plt.title("Epsilon sweep: mean total delay vs baseline")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    if show:
        plt.show()
    else:
        plt.close()


def plot_epsilon_sweep_table(
    sweep_rows: List[Dict[str, float]],
    base: Dict[str, float],
    out_path: Path,
    show: bool,
) -> None:
    """
    ε 스윕 결과를 표 이미지로 저장.

    Gain % 열은 지연 감소율(_gain_lower_is_better). baseline 행은 비교 기준.
    """
    col_labels = [
        "Method",
        "Avg delay",
        "Gain %",
        "Med delay",
        "Gain %",
        "Max delay",
        "Gain %",
    ]
    table_rows: List[List[str]] = []
    b_d = base["total_delay"]
    for r in sweep_rows:
        gm = _gain_lower_is_better(r["total_delay_mean"], b_d)
        gmed = _gain_lower_is_better(r["total_delay_median"], b_d)
        gmax = _gain_lower_is_better(r["total_delay_max"], b_d)
        table_rows.append(
            [
                f"Q-Learn ε={r['epsilon']:.2f}",
                f"{r['total_delay_mean']:.2f}",
                f"{gm:.2f}",
                f"{r['total_delay_median']:.2f}",
                f"{gmed:.2f}",
                f"{r['total_delay_max']:.2f}",
                f"{gmax:.2f}",
            ]
        )
    table_rows.append(
        [
            "Baseline (shortest-hop)",
            f"{b_d:.2f}",
            "—",
            f"{b_d:.2f}",
            "—",
            f"{b_d:.2f}",
            "—",
        ]
    )

    fig, ax = plt.subplots(figsize=(12, 0.55 * (len(table_rows) + 2)))
    ax.axis("off")
    tbl = ax.table(
        cellText=table_rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.scale(1.0, 1.4)
    fig.suptitle(
        "Hyperparameter sweep: delay vs baseline (Gain % = delay reduction vs baseline)",
        fontsize=11,
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def run_epsilon_sweep(
    base_cfg: ExperimentConfig,
    epsilon_values: List[float],
    *,
    show: bool = False,
) -> None:
    """
    epsilon_start만 바꾼 여러 실험을 연속 실행.

    각 ε에 대해 collect_experiment_results 호출 → CSV, 막대, 표 PNG 저장.
    마지막 반복의 base는 동일 토폴로지에서 baseline이므로 모든 행과 동일 기준선.
    """
    out_dir = Path(__file__).resolve().parent / "results_repro" / f"{base_cfg.name}_epsilon_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep_rows: List[Dict[str, float]] = []

    for eps in epsilon_values:
        cfg = replace(base_cfg, epsilon_start=eps)
        q_agg, base, seed_rows, _all_rewards = collect_experiment_results(cfg)
        delays = [row["total_delay"] for row in seed_rows]
        sweep_rows.append(
            {
                "epsilon": float(eps),
                "path_reward_mean": q_agg["path_reward_mean"],
                "total_bandwidth_mean": q_agg["total_bandwidth_mean"],
                "total_delay_mean": q_agg["total_delay_mean"],
                "total_delay_std": q_agg["total_delay_std"],
                "total_delay_median": float(np.median(delays)),
                "total_delay_max": float(np.max(delays)),
                "hops_mean": q_agg["hops_mean"],
                "gain_delay_mean_pct": _gain_lower_is_better(q_agg["total_delay_mean"], base["total_delay"]),
                "gain_path_reward_pct": _gain_higher_is_better(q_agg["path_reward_mean"], base["path_reward"]),
            }
        )

    save_csv(out_dir / "sweep_metrics.csv", sweep_rows)
    plot_epsilon_sweep_bars(sweep_rows, base, out_dir / "sweep_delay_comparison.png", show)
    plot_epsilon_sweep_table(sweep_rows, base, out_dir / "sweep_results_table.png", show)

    print("=" * 88)
    print(f"Epsilon sweep 완료 (preset base: {base_cfg.name})")
    print("=" * 88)
    print(f"저장 폴더: {out_dir}")
    print("- sweep_metrics.csv")
    print("- sweep_delay_comparison.png")
    print("- sweep_results_table.png")
    for r in sweep_rows:
        print(
            f"  ε={r['epsilon']:.2f}  delay_mean={r['total_delay_mean']:.3f}  "
            f"gain_vs_baseline={r['gain_delay_mean_pct']:.2f}%"
        )
    print(f"  Baseline delay (mean): {base['total_delay']:.3f}")


def plot_seed_boxplot(seed_rows: List[Dict[str, float]], out_path: Path, show: bool):
    """시드에 따른 경로 지표 분포를 박스플롯으로 — 민감도·분산 시각화."""
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
    """Markdown 리포트: 설정 요약, 표 형식 수치, 생성 파일 목록, 해석 가이드."""

    def imp(q, b):
        """개선율: (q-b)/|b|*100 — 지연처럼 작을수록 좋은 지표는 해석 시 부호에 주의."""
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
    """CLI --preset 과 대응되는 기본 ExperimentConfig 생성."""
    if name == "paper_repro":
        return ExperimentConfig(name="paper_repro")
    if name == "fast_check":
        return ExperimentConfig(name="fast_check", num_episodes=200, seeds=(7, 42, 101))
    raise ValueError(f"Unknown preset: {name}")


def run(cfg: ExperimentConfig, show: bool = False):
    """
    단일 설정 전체 실행: 학습·요약 CSV·학습곡선·비교 막대·박스플롯·리포트 MD.

    결과 디렉터리: results_repro/<cfg.name>/
    """
    out_dir = Path(__file__).resolve().parent / "results_repro" / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)

    q_agg, base, seed_rows, all_rewards = collect_experiment_results(cfg)

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
    parser.add_argument(
        "--sweep-epsilon",
        action="store_true",
        help="epsilon_start를 여러 값으로 바꿔 실행하고 CSV·PNG 표·막대 그래프 저장",
    )
    parser.add_argument(
        "--epsilons",
        default="0.2,0.3,0.5",
        help="--sweep-epsilon 시 사용할 값 목록 (쉼표 구분, 예: 0.2,0.3,0.5)",
    )
    args = parser.parse_args()

    cfg = build_preset(args.preset)
    if args.sweep_epsilon:
        eps_list = [float(x.strip()) for x in args.epsilons.split(",") if x.strip()]
        run_epsilon_sweep(cfg, eps_list, show=args.show)
    else:
        run(cfg, show=args.show)
