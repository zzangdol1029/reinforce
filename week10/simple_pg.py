"""
실습 #1 — Simple Policy Gradient (CartPole)
- 에피소드 전체에서 동일 가중치 G(τ) 로 모든 timestep 의 log π 를 갱신

실행:
  python simple_pg.py
"""
from __future__ import annotations

import argparse

import numpy as np

from cartpole_env import CartPoleStepAdapter, make_cartpole
from pg_numpy_core import SoftmaxPolicy


def moving_average(xs: list[float], w: int) -> np.ndarray:
    if not xs:
        return np.array([])
    a = np.array(xs, dtype=np.float64)
    out = np.empty_like(a)
    cum = np.cumsum(np.insert(a, 0, 0.0))
    for i in range(len(a)):
        j = max(0, i - w + 1)
        out[i] = (cum[i + 1] - cum[j]) / (i - j + 1)
    return out


class AgentSimplePG:
    def __init__(self, pi: SoftmaxPolicy, gamma: float = 0.98):
        self.pi = pi
        self.gamma = gamma
        self.memory: list[tuple[float, dict, int]] = []

    def get_action(self, state: np.ndarray) -> tuple[int, dict]:
        s = np.asarray(state, dtype=np.float32)[np.newaxis, :]
        probs, cache = self.pi.predict(s)
        probs1 = probs[0]
        a = int(np.random.choice(len(probs1), p=np.asarray(probs1)))
        return a, cache

    def add(self, reward: float, cache_t: dict, action: int) -> None:
        self.memory.append((reward, cache_t, action))

    def update(self) -> None:
        if not self.memory:
            return
        # 에피소드 끝 return G(τ): 역방향 누적의 최종값
        g_accum = 0.0
        for r, _, _ in reversed(self.memory):
            g_accum = float(r + self.gamma * g_accum)
        big_g = g_accum

        self.pi.zero_grad_buffers()
        for _, cache, act in self.memory:
            self.pi.accumulate_pg_grad(cache, act, coeff=big_g)
        self.pi.step()
        self.memory.clear()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--plot", action="store_true", help="저장 후 matplotlib 창 또는 파일")
    args = ap.parse_args()

    raw, backend = make_cartpole()
    env = CartPoleStepAdapter(raw, backend)

    pi = SoftmaxPolicy(state_dim=4, action_size=2, hidden=args.hidden, lr=args.lr)
    agent = AgentSimplePG(pi, gamma=args.gamma)

    rewards_hist: list[float] = []

    try:
        for ep in range(args.episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0.0

            while not done:
                action, cache = agent.get_action(obs)
                next_obs, reward, done, _dead = env.step(action)
                agent.add(float(reward), cache, action)
                obs = next_obs
                total_reward += reward

            agent.update()
            rewards_hist.append(total_reward)

            if (ep + 1) % 100 == 0:
                window = rewards_hist[-100:]
                avg = sum(window) / len(window)
                print(f"[{ep + 1:5d}/{args.episodes}] last100_avg={avg:6.1f} last={total_reward:4.1f}")

    finally:
        raw.close()

    if args.plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            axes[0].plot(rewards_hist, alpha=0.45, label="raw")
            axes[0].plot(moving_average(rewards_hist, 100), color="orange", lw=2, label="MA-100")
            axes[0].set_title("Simple PG — episode reward")
            axes[0].set_xlabel("Episode")
            axes[0].legend()
            axes[0].grid(True, alpha=0.35)

            axes[1].plot(moving_average(rewards_hist, 100))
            axes[1].set_title("(100 회 이동 평균)")
            axes[1].grid(True, alpha=0.35)

            fig.tight_layout()
            outp = __file__.replace(".py", "_reward.png")
            fig.savefig(outp, dpi=120)
            print(f"플롯 저장: {outp}")
            plt.show()
        except Exception as exc:
            print(f"matplotlib 사용 불가, 스킵: {exc}")

    avg_last = sum(rewards_hist[-100:]) / min(100, len(rewards_hist))
    print(f"완료. 마지막 100에피 평균 보상: {avg_last:.2f}")


if __name__ == "__main__":
    main()
