"""
실습 #1 gridworld.py
"""

import numpy as np
import common.gridworld_render as render_helper


class GridWorld:
    def __init__(self):
        self.action_space = [0, 1, 2, 3]  # 행동 공간(가능한 행동들)
        self.action_meaning = {  # 행동의 의미
            0: "UP",
            1: "DOWN",
            2: "LEFT",
            3: "RIGHT",
        }

        self.reward_map = np.array(  # 보상 맵(각 좌표의 보상 값)
            [[0, 0, 0, 1.0],
             [0, None, 0, -1.0],
             [0, 0, 0, 0]]
        )

        self.goal_state = (0, 3)      # 목표 상태(좌표)
        self.wall_state = (1, 1)      # 벽 상태(좌표)
        self.start_state = (2, 0)     # 시작 상태(좌표)
        self.agent_state = self.start_state  # 에이전트 초기 상태(좌표)

        self.terminal_states = set()
        for h in range(len(self.reward_map)):
            for w in range(len(self.reward_map[0])):
                r = self.reward_map[h, w]
                if r is not None and r != 0:
                    self.terminal_states.add((h, w))

    @property
    def height(self):
        return len(self.reward_map)

    @property
    def width(self):
        return len(self.reward_map[0])

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
        elif next_state == self.wall_state:
            next_state = state

        return next_state

    def reward(self, state, action, next_state):
        r = self.reward_map[next_state]
        if r is None:
            return 0.0
        return float(r)

    def reset(self):
        self.agent_state = self.start_state
        return self.agent_state

    def step(self, action):
        state = self.agent_state
        next_state = self.next_state(state, action)
        reward = self.reward(state, action, next_state)
        done = next_state in self.terminal_states

        self.agent_state = next_state
        return next_state, reward, done

    def render_v(self, v=None, policy=None, print_value=True, use_matplotlib=True):
        renderer = render_helper.Renderer(
            self.reward_map, self.goal_state, self.wall_state
        )
        renderer.render_v(v, policy, print_value, use_matplotlib=use_matplotlib)

    def render_q(self, q=None, print_value=True):
        renderer = render_helper.Renderer(
            self.reward_map, self.goal_state, self.wall_state
        )
        renderer.render_q(q, print_value)


if __name__ == "__main__":
    env = GridWorld()

    print(env.height)
    print(env.width)
    print(env.shape)

    for action in env.actions():
        print(action)

    print("===")

    for state in env.states():
        print(state)

    print("\n--- step / render_v ---")
    print("goal", env.goal_state, "| wall", env.wall_state, "| start", env.start_state)
    print("reset ->", env.reset())
    s, r, d = env.step(3)
    print("step RIGHT(3) ->", s, "r=", r, "done=", d)

    print("빈 가치로 그리드(0 / G 목표 / ### 벽):")
    env.render_v(use_matplotlib=True)

    V = {}
    for state in env.states():
        V[state] = np.random.randn()
    env.render_v(V, use_matplotlib=True)
