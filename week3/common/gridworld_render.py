"""GridWorld 시각화 (render_v, render_q)"""

import numpy as np


class Renderer:
    def __init__(self, reward_map, goal_state, wall_state):
        self.reward_map = reward_map
        self.goal_state = goal_state
        self.wall_state = wall_state
        self.height = len(reward_map)
        self.width = len(reward_map[0])

    def _v_at(self, v, h, w):
        if v is None:
            return 0.0
        if isinstance(v, dict):
            return float(v.get((h, w), 0.0))
        if isinstance(v, np.ndarray):
            return float(v[h, w])
        return 0.0

    def render_v(self, v=None, policy=None, print_value=True):
        for h in range(self.height):
            row_parts = []
            for w in range(self.width):
                if (h, w) == self.wall_state:
                    row_parts.append("  ###  ")
                elif (h, w) == self.goal_state:
                    val = self._v_at(v, h, w) if print_value else 0.0
                    row_parts.append(f" G{val:5.2f}" if print_value else "  G   ")
                else:
                    val = self._v_at(v, h, w)
                    row_parts.append(f"{val:7.2f}" if print_value else "  .   ")
            print("  " + "".join(row_parts))
        print()

    def render_q(self, q=None, print_value=True):
        if q is None:
            q = {}
        for h in range(self.height):
            for w in range(self.width):
                cell = (h, w)
                if cell == self.wall_state:
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
