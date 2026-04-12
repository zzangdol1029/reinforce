"""
Quiz (p.42) — 5x5 Grid World
  - Goal(사과): (0, 4), 보상 +1
  - 함정(폭탄): (0, 3), (3, 4), 보상 -1  — (하단좌표 기준 문제문) 목표 왼쪽 (3,4), 우측 (4,1) → (h,w)=(0,3),(3,4)
  - 벽: (2, 1), (2, 2)
  - 시작: (4, 0)
정책/가치 반복 등과 연동할 수 있도록 week3/gridworld.py 와 동일한 API.

이 파일은 **환경(Environment)** 만 정의합니다.
- 에이전트/정책/가치함수 계산은 `policy_iter5x5.py`, `value_iter5x5.py`에서 수행합니다.
- 좌표는 (h, w) = (row, col) 형식입니다. 예) (0,4)는 첫 행, 다섯 번째 열.
- `reward_map`에서
  - `None` 은 벽(이동 불가)을 의미합니다.
  - 숫자(예: +1, -1)는 해당 칸으로 **도착했을 때 받는 보상**입니다.
"""

import numpy as np
import common.gridworld_render as render_helper


class GridWorld5x5:
    def __init__(self):
        # ====== 행동(action) 정의 ======
        # 0,1,2,3을 각각 상/하/좌/우로 사용합니다.
        self.action_space = [0, 1, 2, 3]
        self.action_meaning = {
            0: "UP",
            1: "DOWN",
            2: "LEFT",
            3: "RIGHT",
        }

        # ====== 보상(reward) 맵 ======
        # 기본은 0 보상 (빈 칸). dtype=object로 두는 이유는 벽을 None으로 표시하기 위함입니다.
        self.reward_map = np.zeros((5, 5), dtype=object)

        # 함정(폭탄): 도착 보상 -1
        self.reward_map[0, 3] = -1.0
        self.reward_map[3, 4] = -1.0

        # 목표(사과): 도착 보상 +1
        self.reward_map[0, 4] = 1.0

        # 벽: 이동 불가 (보상값 자체가 중요한 게 아니라 "통과 불가" 표시로 씁니다)
        self.reward_map[2, 1] = None
        self.reward_map[2, 2] = None

        # ====== 주요 상태들 ======
        self.goal_state = (0, 4)
        self.wall_states = frozenset(((2, 1), (2, 2)))
        self.trap_states = frozenset(((0, 3), (3, 4)))
        self.start_state = (4, 0)

        # 에이전트의 현재 상태 (step/reset에서 사용)
        self.agent_state = self.start_state

    @property
    def terminal_states(self):
        """목표·함정 — 반복 정책/가치 평가 시 V=0 으로 두는 상태."""
        return frozenset({self.goal_state}) | self.trap_states

    @property
    def height(self):
        return int(self.reward_map.shape[0])

    @property
    def width(self):
        return int(self.reward_map.shape[1])

    @property
    def shape(self):
        return self.reward_map.shape

    def actions(self):
        """가능한 행동 리스트 반환 (0,1,2,3)."""
        return self.action_space

    def states(self):
        """
        모든 격자 좌표를 순회하는 제너레이터.
        - (0,0)부터 (4,4)까지 순서대로 yield
        - 벽/목표/함정도 "상태"로는 포함됩니다. (알고리즘에서 제외/고정 처리)
        """
        for h in range(self.height):
            for w in range(self.width):
                yield (h, w)

    def next_state(self, state, action):
        """
        상태 전이 함수: s' = f(s, a)
        - 행동에 따라 다음 좌표를 계산
        - 경계 밖으로 나가거나 벽이면 제자리(state 유지)
        """
        action_move_map = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        move = action_move_map[action]
        next_state = (state[0] + move[0], state[1] + move[1])
        ny, nx = next_state

        # 격자 밖이면 이동 무효
        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
            next_state = state
        # 벽이면 이동 무효
        elif next_state in self.wall_states:
            next_state = state

        return next_state

    def reward(self, state, action, next_state):
        """
        보상 함수: r(s, a, s')
        - 이 환경은 '도착한 칸(next_state)'의 reward_map 값을 보상으로 사용합니다.
        - 벽(None)은 실제로는 next_state가 될 수 없지만, 안전하게 0.0 처리합니다.
        """
        r = self.reward_map[next_state]
        return 0.0 if r is None else float(r)

    def is_terminal(self, state):
        """종료 상태 여부 (goal 또는 trap)."""
        return state in self.terminal_states

    def reset(self):
        """
        에피소드 시작 상태로 리셋.
        Returns:
            start_state
        """
        self.agent_state = self.start_state
        return self.agent_state

    def step(self, action):
        """
        환경을 1 스텝 진행.
        Args:
            action: 0~3 (UP/DOWN/LEFT/RIGHT)
        Returns:
            next_state, reward, done
        """
        state = self.agent_state
        next_state = self.next_state(state, action)
        reward = self.reward(state, action, next_state)
        done = self.is_terminal(next_state)

        self.agent_state = next_state
        return next_state, reward, done

    def render_v(self, v=None, policy=None, print_value=True, use_matplotlib=True, **kwargs):
        """
        5x5에서도 3x4와 동일한 render API를 제공.
        - use_matplotlib=True이면 matplotlib 기반(색+화살표) 렌더링 시도
        - matplotlib이 없으면 텍스트 출력으로 fallback
        """
        renderer = render_helper.Renderer(
            self.reward_map, self.goal_state, self.wall_states, trap_states=self.trap_states
        )
        renderer.render_v(v, policy, print_value, use_matplotlib=use_matplotlib, **kwargs)

    def render_q(self, q=None, print_value=True):
        renderer = render_helper.Renderer(
            self.reward_map, self.goal_state, self.wall_states, trap_states=self.trap_states
        )
        renderer.render_q(q, print_value)

    # (이전 텍스트 렌더링용 _v_lookup은 common renderer가 대체)


if __name__ == "__main__":
    from collections import defaultdict

    import matplotlib.pyplot as plt

    from value_iter5x5 import greedy_policy, value_iter_onestep

    env = GridWorld5x5()
    gamma = 0.9
    print("start:", env.reset())
    s, r, d = env.step(3)
    print("RIGHT:", s, r, d)
    print("shape:", env.shape, "goal", env.goal_state, "walls", env.wall_states)

    # Value Iteration: Bellman optimal backup 5 steps — one window
    print("\n--- Value Iteration demo (5 steps) ---")
    V = defaultdict(float)
    for state in env.states():
        V[state] = 0.0

    snapshots = []
    for k in range(1, 6):
        V = value_iter_onestep(V, env, gamma)
        pi_k = greedy_policy(V, env, gamma)
        snapshots.append((f"Value Iter step {k}", V.copy(), pi_k))
    print("Demo: 5 steps shown. Run value_iter5x5.py for full convergence.")

    n = len(snapshots)
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.6 * ncols, 5.0 * nrows),
        constrained_layout=True,
    )
    if nrows * ncols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, (name, V_snap, pi_snap) in enumerate(snapshots):
        ax = axes[i]
        env.render_v(
            V_snap,
            pi_snap,
            use_matplotlib=True,
            ax=ax,
            show=False,
            title=name,
            draw_colorbar=False,
        )

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.show()
