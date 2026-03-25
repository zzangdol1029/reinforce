"""
Quiz (p.42) — 5x5 Grid World
  - Goal(사과): (0, 4), 보상 +1
  - 함정(폭탄): (0, 3), (3, 3), 보상 -1
  - 벽: (2, 1), (2, 2)
  - 시작: (4, 0)
정책/가치 반복 등과 연동할 수 있도록 week3/gridworld.py 와 동일한 API.
"""

import numpy as np


class GridWorld5x5:
    def __init__(self):
        self.action_space = [0, 1, 2, 3]
        self.action_meaning = {
            0: "UP",
            1: "DOWN",
            2: "LEFT",
            3: "RIGHT",
        }

        self.reward_map = np.zeros((5, 5), dtype=object)
        self.reward_map[0, 3] = -1.0
        self.reward_map[0, 4] = 1.0
        self.reward_map[2, 1] = None
        self.reward_map[2, 2] = None
        self.reward_map[3, 3] = -1.0

        self.goal_state = (0, 4)
        self.wall_states = frozenset(((2, 1), (2, 2)))
        self.trap_states = frozenset(((0, 3), (3, 3)))
        self.start_state = (4, 0)
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
        return self.action_space

    def states(self):
        for h in range(self.height):
            for w in range(self.width):
                yield (h, w)

    def next_state(self, state, action):
        action_move_map = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        move = action_move_map[action]
        next_state = (state[0] + move[0], state[1] + move[1])
        ny, nx = next_state

        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
            next_state = state
        elif next_state in self.wall_states:
            next_state = state

        return next_state

    def reward(self, state, action, next_state):
        r = self.reward_map[next_state]
        return 0.0 if r is None else float(r)

    def is_terminal(self, state):
        return state in self.terminal_states

    def reset(self):
        self.agent_state = self.start_state
        return self.agent_state

    def step(self, action):
        state = self.agent_state
        next_state = self.next_state(state, action)
        reward = self.reward(state, action, next_state)
        done = self.is_terminal(next_state)

        self.agent_state = next_state
        return next_state, reward, done

    def render_v(self, v=None, policy=None, print_value=True):
        for h in range(self.height):
            row_parts = []
            for w in range(self.width):
                cell = (h, w)
                if cell in self.wall_states:
                    row_parts.append("  ###  ")
                elif cell == self.goal_state:
                    val = self._v_lookup(v, h, w) if print_value else 0.0
                    row_parts.append(f" G{val:5.2f}" if print_value else "  G   ")
                elif cell in self.trap_states:
                    val = self._v_lookup(v, h, w) if print_value else 0.0
                    row_parts.append(f" T{val:5.2f}" if print_value else "  T   ")
                else:
                    val = self._v_lookup(v, h, w)
                    row_parts.append(f"{val:7.2f}" if print_value else "  .   ")
            print("  " + "".join(row_parts))
        print()

    def render_q(self, q=None, print_value=True):
        for h in range(self.height):
            for w in range(self.width):
                cell = (h, w)
                if cell in self.wall_states:
                    print(f"  ({h},{w}) WALL")
                    continue
                parts = []
                for a in range(4):
                    if isinstance(q, dict):
                        qa = float(q.get((cell, a), 0.0))
                    elif isinstance(q, np.ndarray) and q.ndim == 3:
                        qa = float(q[h, w, a])
                    else:
                        qa = 0.0
                    if print_value:
                        parts.append(f"a{a}={qa:.2f}")
                print(f"  ({h},{w}) " + " ".join(parts))
            print()

    @staticmethod
    def _v_lookup(v, h, w):
        if v is None:
            return 0.0
        if isinstance(v, dict):
            return float(v.get((h, w), 0.0))
        if isinstance(v, np.ndarray):
            return float(v[h, w])
        return 0.0


if __name__ == "__main__":
    env = GridWorld5x5()
    print("start:", env.reset())
    s, r, d = env.step(3)
    print("RIGHT:", s, r, d)
    print("shape:", env.shape, "goal", env.goal_state, "walls", env.wall_states)
