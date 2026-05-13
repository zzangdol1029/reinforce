"""
CartPole 환경: gymnasium 우선, 없으면 레거시 gym 사용.
반환 규격을 (obs, reward, terminated, truncated, info) 로 통일.
"""
from __future__ import annotations

from typing import Any


def make_cartpole() -> tuple[Any, str]:
    try:
        import gymnasium as gym

        env = gym.make("CartPole-v1")
        return env, "gymnasium"
    except Exception:
        import gym

        try:
            env = gym.make("CartPole-v1")
        except Exception:
            env = gym.make("CartPole-v0")
        return env, "gym_legacy"


class CartPoleStepAdapter:
    def __init__(self, raw_env, backend: str):
        self.raw = raw_env
        self.backend = backend

    def reset(self) -> tuple[Any, dict]:
        if self.backend == "gymnasium":
            obs, info = self.raw.reset()
            return obs, info
        obs = self.raw.reset()
        return obs, {}

    def step(self, action: int) -> tuple[Any, float, bool, bool]:
        """
        반환:
          obs, reward, done, dead
          - done: 에피소드 종료(종료 + 트렁케이션)
          - dead: 상태 종료(막대 넘어짐 등); True이면 다음 가치를 0으로 부트스트랩하지 않음
                  truncated(시간 제한만)에서는 dead=False 로 부트스트랩 유지
        """
        if self.backend == "gymnasium":
            obs, reward, terminated, truncated, info = self.raw.step(action)
            done = bool(terminated or truncated)
            dead = bool(terminated)  # truncation만 있는 경우에는 dead=False
            return obs, float(reward), done, dead
        obs, reward, done, info = self.raw.step(action)
        truncated = False
        if isinstance(info, dict):
            truncated = bool(info.get("TimeLimit.truncated", False))
        dead = bool(done and not truncated)
        return obs, float(reward), done, dead
