"""
5x5 Grid World — Q-Learning (하이퍼파라미터 비교 + 결과 이미지·지표 저장)

환경: week3/gridworld5x5.py
결과 폴더: week5/results_q_learning_5x5/
- `combined_all_q_policy.png`: 모든 설정의 Q·policy를 **한 화면**(가로 열 배치, 각 열 위=Q·아래=policy)
- `summary_metrics.png`: 지표 **막대 그래프만** 별도
- `metrics.csv`, `README_ANALYSIS.md`, `PARAMETER_REFLECTION.md` (파라미터 고찰)
실행: 기본은 창 2개(위 순서). 창 없이 저장만: `python q_learning_5x5.py --no-show`
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "week3"))

import matplotlib

if sys.platform == "darwin":
    matplotlib.use("macosx", force=True)
else:
    for _name in ("TkAgg", "Qt5Agg", "QtAgg"):
        try:
            matplotlib.use(_name, force=True)
            break
        except Exception:
            continue

import matplotlib.pyplot as plt

from collections import defaultdict

import numpy as np

from gridworld5x5 import GridWorld5x5

_week4_render = _root / "week4" / "common" / "gridworld_render.py"
_spec = importlib.util.spec_from_file_location("week4_gridworld_render", _week4_render)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load Renderer from {_week4_render}")
_gw4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gw4)
Renderer = _gw4.Renderer

OUT_DIR = Path(__file__).resolve().parent / "results_q_learning_5x5"


def greedy_probs(Q, state, epsilon=0.0, action_size=4):
    qs = [Q[(state, action)] for action in range(action_size)]
    max_action = int(np.argmax(qs))
    base_prob = epsilon / action_size
    action_probs = {action: base_prob for action in range(action_size)}
    action_probs[max_action] += 1.0 - epsilon
    return action_probs


def greedy_policy_deterministic(Q, env, action_size=4):
    pi = {}
    for state in env.states():
        if state in env.wall_states:
            continue
        qs = [Q[(state, a)] for a in range(action_size)]
        best = int(np.argmax(qs))
        pi[state] = {a: (1.0 if a == best else 0.0) for a in range(action_size)}
    return pi


class QLearningAgent:
    def __init__(self, gamma=0.9, alpha=0.8, epsilon=0.1, action_size=4):
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.action_size = action_size
        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.b = defaultdict(lambda: random_actions.copy())
        self.Q = defaultdict(lambda: 0)

    def get_action(self, state):
        ap = self.b[state]
        return int(np.random.choice(list(ap.keys()), p=list(ap.values())))

    def update(self, state, action, reward, next_state, done):
        if done:
            next_q_max = 0.0
        else:
            next_q_max = max(self.Q[next_state, a] for a in range(self.action_size))
        target = reward + self.gamma * next_q_max
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha
        self.b[state] = greedy_probs(self.Q, state, self.epsilon, self.action_size)


@dataclass
class RunConfig:
    name: str
    gamma: float
    alpha: float
    epsilon: float
    episodes: int


def train_one(
    env: GridWorld5x5,
    cfg: RunConfig,
    seed: int | None = 42,
) -> tuple[QLearningAgent, dict]:
    if seed is not None:
        np.random.seed(seed)

    agent = QLearningAgent(
        gamma=cfg.gamma,
        alpha=cfg.alpha,
        epsilon=cfg.epsilon,
    )

    goals = 0
    traps = 0
    steps_per_ep = []

    for _ in range(cfg.episodes):
        state = env.reset()
        steps = 0
        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            steps += 1
            if done:
                if next_state == env.goal_state:
                    goals += 1
                elif next_state in env.trap_states:
                    traps += 1
                break
            state = next_state
        steps_per_ep.append(steps)

    v_start = max(agent.Q[env.start_state, a] for a in range(4))
    v_vals = []
    for s in env.states():
        if s in env.wall_states:
            continue
        v_vals.append(max(agent.Q[s, a] for a in range(4)))
    v_mean = float(np.mean(v_vals)) if v_vals else 0.0

    stats = {
        "goal_rate": goals / cfg.episodes,
        "trap_rate": traps / cfg.episodes,
        "mean_episode_steps": float(np.mean(steps_per_ep)),
        "v_at_start": float(v_start),
        "v_mean_free_states": float(v_mean),
    }
    return agent, stats


def save_combined_q_policy_figure(
    agents: list[QLearningAgent],
    env: GridWorld5x5,
    renderer: Renderer,
    configs: list[RunConfig],
    path: Path,
    *,
    show: bool,
) -> None:
    """모든 설정을 가로(열)로 배치한다. 각 열에서 위=Q, 아래=greedy policy."""
    _gw4._module._configure_matplotlib_korean_font()
    n = len(configs)
    col_w = 2.85
    fig_h = 7.0
    fig, axes = plt.subplots(2, n, figsize=(max(10.0, col_w * n), fig_h))
    axes = np.asarray(axes)
    if n == 1:
        axes = axes.reshape(2, 1)

    for j, (agent, cfg) in enumerate(zip(agents, configs)):
        renderer._draw_q_diamond(agent.Q, axes[0, j])
        renderer._draw_greedy_policy(agent.Q, axes[1, j])
        axes[0, j].set_title(
            f"[{cfg.name}] Q  |  γ={cfg.gamma}  α={cfg.alpha}  ε={cfg.epsilon}",
            fontsize=10,
        )
        axes[1, j].set_title(f"[{cfg.name}] greedy policy", fontsize=10)

    fig.suptitle(
        "하이퍼파라미터별 Q-Learning 결과 비교 (동일 seed, 동일 에피소드 수)",
        fontsize=12,
        y=1.002,
    )
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def save_metrics_bar(
    configs: list[RunConfig],
    stats_list: list[dict],
    path: Path,
    *,
    show: bool = False,
) -> None:
    names = [c.name for c in configs]
    v_starts = [s["v_at_start"] for s in stats_list]
    goal_rates = [s["goal_rate"] for s in stats_list]

    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(x, v_starts, color="steelblue")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=15, ha="right")
    axes[0].set_ylabel("max_a Q(start, a)")
    axes[0].set_title("시작 상태에서 추정 가치 (학습 후)")

    axes[1].bar(x, goal_rates, color="seagreen")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=15, ha="right")
    axes[1].set_ylabel("비율")
    axes[1].set_title("에피소드당 목표 도달 비율")
    axes[1].set_ylim(0, 1.05)

    fig.suptitle("지표 요약 — 설정 간 비교 (막대 그래프만)", fontsize=12, y=1.03)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def write_csv(path: Path, configs: list[RunConfig], stats_list: list[dict]) -> None:
    lines = [
        "name,gamma,alpha,epsilon,episodes,goal_rate,trap_rate,mean_steps,v_start,v_mean_free",
    ]
    for c, st in zip(configs, stats_list):
        lines.append(
            f"{c.name},{c.gamma},{c.alpha},{c.epsilon},{c.episodes},"
            f"{st['goal_rate']:.6f},{st['trap_rate']:.6f},{st['mean_episode_steps']:.4f},"
            f"{st['v_at_start']:.6f},{st['v_mean_free_states']:.6f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_analysis_md(path: Path, configs: list[RunConfig], stats_list: list[dict]) -> None:
    body = r"""# Q-Learning 5×5 하이퍼파라미터 실험 — 결과 해석

이 문서는 `q_learning_5x5.py`가 여러 설정으로 학습한 뒤 저장한 지표(`metrics.csv`)와 그림(`*.png`)을 바탕으로 **어떤 경향이 나오는지**만 짧게 정리합니다.

## 지표 의미

| 항목 | 의미 |
|------|------|
| **goal_rate** | 전체 에피소드 중 **목표(사과)** 에 도달한 비율 |
| **trap_rate** | **함정(폭탄)** 에 도달한 비율 |
| **v_at_start** | 학습 후 시작 칸 (4,0)에서 max_a Q(s,a) — 출발지에서 본 **누적 보상의 추정** |
| **v_mean_free** | 벽이 아닌 칸들에 대해 max_a Q(s,a) 평균 — 전체 격자의 **가치 추정 수준** |

## 설정별로 일반적으로 기대되는 경향

1. **α (alpha, 학습률)**  
   - **너무 크면**: 업데이트가 크게 흔들려 **발산·진동**할 수 있음. 목표 도달 비율이 들쭉날쭉해질 수 있음.  
   - **너무 작으면**: 수렴이 **느리고**, 같은 에피소드 수에서는 `v_at_start`가 더 낮게 남을 수 있음.

2. **ε (epsilon, ε-greedy 탐색)**  
   - **크면**: 무작위 행동 비율이 높아 **탐색은 많지만** 최적 행동을 자주 안 해서, 수렴이 느리거나 목표 비율이 낮게 보일 수 있음.  
   - **작으면**: 빨리 **착취(exploitation)** 쪽으로 가지만, **지역 최적**에 갇일 위험.

3. **γ (gamma, 할인율)**  
   - **크면**: 먼 미래 보상까지 반영해 **장기적으로** 목표를 향하는 가치 추정.  
   - **작으면**: **근시안적**이라 함정 근처에서 벗어나는 패턴이 덜 학습될 수 있음.

4. **에피소드 수**  
   - 부족하면 지표와 그림이 **아직 수렴 전** 상태일 수 있음. 늘리면 `goal_rate`와 `v_at_start`가 안정되는 경우가 많음.

## 본 실험에서 확인할 것

- 같은 `seed`로 돌리면 **재현**됩니다. 설정을 바꿔 `metrics.csv`와 `summary_metrics.png`를 비교해 보세요.  
- `combined_all_q_policy.png`에서 **모든 설정을 한 화면**에 가로로 열을 배치하고, 각 열에서 위=Q·아래=policy로 비교합니다.
- 파라미터에 대한 상세 **고찰**은 `PARAMETER_REFLECTION.md`를 참고하세요.

---
*자동 생성: `q_learning_5x5.py`*

## 이번 실행 요약 (자동 기록)

| 설정 | goal_rate | trap_rate | mean_steps | v_start | v_mean_free |
|------|-----------|-----------|------------|---------|-------------|
"""
    rows = []
    for c, st in zip(configs, stats_list):
        rows.append(
            f"| {c.name} | {st['goal_rate']:.4f} | {st['trap_rate']:.4f} | "
            f"{st['mean_episode_steps']:.2f} | {st['v_at_start']:.4f} | {st['v_mean_free_states']:.4f} |"
        )
    path.write_text(body + "\n".join(rows) + "\n", encoding="utf-8")


def write_parameter_reflection_md(path: Path) -> None:
    """α, ε, γ를 바꿨을 때의 이론적·실험적 고찰 (정적 설명)."""
    text = r"""# 하이퍼파라미터 변경에 대한 고찰 (Q-Learning 5×5 실험)

본 문서는 `q_learning_5x5.py`의 기본 비교 설정(`baseline`, `high_alpha`, `low_alpha`, `high_epsilon`, `low_gamma`)을 전제로, **각 기호가 알고리즘과 결과 지표에 주는 의미**를 정리합니다. 수치는 실행마다 달라질 수 있으나, **방향성**은 아래와 같이 해석하는 경우가 많습니다.

---

## 1. α (학습률, step-size)

**역할:** TD 오차에 곱해지는 계수로, 한 스텝에서 \(Q\)를 얼마나 크게 고칠지 결정합니다.

| 변형 | 기대 효과 | 고찰 |
|------|-----------|------|
| **high_alpha (예: 0.95)** | 업데이트 폭이 커짐 | 수렴이 빨라질 수 있으나, 노이즈가 큰 환경에서는 **진동**하거나 한쪽으로 과하게 치우칠 수 있음. |
| **low_alpha (예: 0.3)** | 업데이트 폭이 작음 | 같은 에피소드 수에서는 **수렴이 더뎌** 보일 수 있으나, 단계별 변화는 부드러울 수 있음. |
| **baseline (예: 0.8)** | 중간 | 실습에서 자주 쓰는 스케일; 충분한 에피소드면 high/low와 **최종 정책이 유사**해지는 경우가 많음. |

**지표와의 연결:** `v_at_start`(출발지에서의 \(\max_a Q\))는 **미래 보상의 할인합**에 대한 추정이므로, α만 바꾸고 에피소드가 매우 길면 **비슷한 최적해 근처**로 수렴해 수치가 근접할 수 있습니다. 반면 **초기 수렴 속도**·**중간 진동**은 α에 더 민감합니다.

---

## 2. ε (ε-greedy 탐색률)

**역할:** 행동 선택 시 **무작위 행동**에 할당되는 확률(구조에 따라 \( \varepsilon/|A| \) 등). 나머지는 현재 \(Q\) 기준 탐욕 행동에 몰아줍니다.

| 변형 | 기대 효과 | 고찰 |
|------|-----------|------|
| **high_epsilon (예: 0.35)** | 탐색 비중 증가 | **목표까지의 경로가 길어지고**(`mean_episode_steps` 증가), 같은 스텝 수 안에 **최적 행동을 덜 따르는** 비율이 올라가 **goal_rate**가 떨어지기 쉬움. |
| **baseline (예: 0.1)** | 소폭 탐색 | 탐색과 활용의 균형이 무난한 편. |
| **낮은 ε** | 거의 활용 | 정책이 이미 좋을 때는 유리하나, 초기에는 **국소 최적**에 빠질 위험. |

**지표와의 연결:** ε가 크면 **행동 데이터의 분포**가 “최적 경로”에서 멀어지므로, **목표 도달 비율**과 **평균 스텝 수**에 가장 직접적으로 나타나는 경우가 많습니다. 반면 학습이 충분히 끝난 뒤의 \(Q\) **추정값 자체**는 다른 설정과 비슷해 보일 수 있습니다(같은 seed·같은 업데이트 식이면 \(Q\)는 경로에 따라 달라지지만, 장기 평균으로는 유사할 수 있음).

---

## 3. γ (할인율, discount factor)

**역할:** 미래 보상에 곱해지는 \( \gamma^k \)로, **멀리 있는 보상**을 현재 가치에 얼마나 반영할지 결정합니다.

| 변형 | 기대 효과 | 고찰 |
|------|-----------|------|
| **low_gamma (예: 0.6)** | 단기 보상 위주 | 목표까지 여러 스텝이 필요한 과제에서 **누적 가치의 스케일**이 줄어들어, `v_at_start` 등 **\(V \approx \max_a Q\)** 가 작게 나옴. |
| **baseline (예: 0.9)** | 장기 보상 반영 | 출발점에서 목표까지의 경로 가치가 **크게** 추정되기 쉬움. |
| **γ → 1** | (에피소드 과제에서) 장기적 | 단, 수렴·안정성 조건과 함께 고려해야 함. |

**지표와의 연결:** γ를 낮추면 **같은 정책이라도** Bellman 식에서 미래 항이 작아져 **\(Q\)의 절댓값 전반이 줄어듦**. 따라서 **goal_rate는 높아도** `v_at_start`는 작게 찍힐 수 있으며, 이는 “못 찾아서”가 아니라 **가치의 단위(시간 범위)** 가 달라진 결과로 해석하는 것이 타당합니다.

---

## 4. 종합: 그림·표를 읽는 순서

1. **`combined_all_q_policy.png`:** 행마다 같은 환경에서 **Q의 패턴**과 **greedy 화살표**가 어떻게 달라지는지 비교합니다. ε·γ 변화는 **함정·목표 주변 행동 분포**에서 차이가 드러나기 쉽습니다.  
2. **`summary_metrics.png`:** `v_at_start`와 `goal_rate`를 한눈에 비교합니다.  
3. **`metrics.csv`:** 동일 seed 재현 및 보고서용 수치 확인.

---

*자동 생성: `q_learning_5x5.py` — `PARAMETER_REFLECTION.md`*
"""
    path.write_text(text, encoding="utf-8")


# 비교용 기본 설정 (이름은 파일명·그래프에 사용)
DEFAULT_CONFIGS: list[RunConfig] = [
    RunConfig("baseline", gamma=0.9, alpha=0.8, epsilon=0.1, episodes=25_000),
    RunConfig("high_alpha", gamma=0.9, alpha=0.95, epsilon=0.1, episodes=25_000),
    RunConfig("low_alpha", gamma=0.9, alpha=0.3, epsilon=0.1, episodes=25_000),
    RunConfig("high_epsilon", gamma=0.9, alpha=0.8, epsilon=0.35, episodes=25_000),
    RunConfig("low_gamma", gamma=0.6, alpha=0.8, epsilon=0.1, episodes=25_000),
]


def main(
    configs: list[RunConfig] | None = None,
    *,
    show_plots: bool = True,
    seed: int = 42,
) -> None:
    configs = configs or DEFAULT_CONFIGS
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    env = GridWorld5x5()
    renderer = Renderer(
        env.reward_map,
        env.goal_state,
        env.wall_states,
        trap_states=env.trap_states,
    )

    all_stats: list[dict] = []
    agents: list[QLearningAgent] = []

    if show_plots:
        print(
            "matplotlib 창 2개: (1) 모든 설정의 Q·policy 한 화면 → 닫으면 (2) 요약 막대 그래프.",
            flush=True,
        )

    for cfg in configs:
        agent, stats = train_one(env, cfg, seed=seed)
        agents.append(agent)
        all_stats.append(stats)

    save_combined_q_policy_figure(
        agents,
        env,
        renderer,
        configs,
        OUT_DIR / "combined_all_q_policy.png",
        show=show_plots,
    )

    write_csv(OUT_DIR / "metrics.csv", configs, all_stats)
    save_metrics_bar(
        configs, all_stats, OUT_DIR / "summary_metrics.png", show=show_plots
    )
    write_analysis_md(OUT_DIR / "README_ANALYSIS.md", configs, all_stats)
    write_parameter_reflection_md(OUT_DIR / "PARAMETER_REFLECTION.md")

    print(f"저장 위치: {OUT_DIR}")
    print(
        "생성 파일: combined_all_q_policy.png, summary_metrics.png, metrics.csv, "
        "README_ANALYSIS.md, PARAMETER_REFLECTION.md"
    )
    for c, st in zip(configs, all_stats):
        print(
            f"  [{c.name}] goal_rate={st['goal_rate']:.3f}  v_start={st['v_at_start']:.4f}  "
            f"mean_steps={st['mean_episode_steps']:.1f}"
        )


if __name__ == "__main__":
    # 기본: 창 1 = 모든 설정 Q·policy 한 화면, 창 2 = 요약 막대 그래프만.
    import argparse

    p = argparse.ArgumentParser(description="5x5 Q-Learning 하이퍼파라미터 비교")
    p.add_argument(
        "--no-show",
        action="store_true",
        help="matplotlib 창을 띄우지 않고 PNG/CSV만 저장",
    )
    args = p.parse_args()
    main(show_plots=not args.no_show)
