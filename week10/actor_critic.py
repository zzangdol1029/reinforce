"""
실습 #3 — Actor-Critic (1-step TD, CartPole)
- Critic: ½ δ² 방식 회귀 (δ = tgt − V_w(s), 여기선 V 기준 역전파는 ∂L/∂V = V − tgt)
- Actor: −δ · log π_θ(a|s) 형태와 동등한 PG 누적 (δ를 상수처럼 취급)

실행:
  python actor_critic.py
"""
from __future__ import annotations

import argparse

import numpy as np

from cartpole_env import CartPoleStepAdapter, make_cartpole
from pg_numpy_core import SoftmaxPolicy, ValueNet


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


class AgentActorCritic:
    """매 스텝마다 학습."""

    def __init__(self, pi: SoftmaxPolicy, vw: ValueNet, gamma: float = 0.98):
        self.pi = pi
        self.vw = vw
        self.gamma = gamma

    def get_action(self, state: np.ndarray) -> tuple[int, dict]:
        s = np.asarray(state, dtype=np.float32)[np.newaxis, :]
        probs, cache_pi = self.pi.predict(s)
        probs1 = probs[0]
        a = int(np.random.choice(len(probs1), p=np.asarray(probs1)))
        return a, cache_pi

    def update(
        self,
        reward: float,
        dead: bool,
        cache_pi: dict,
        action: int,
        state: np.ndarray,
        next_state: np.ndarray,
    ) -> None:
        s = np.asarray(state, dtype=np.float32)[np.newaxis, :]
        ns = np.asarray(next_state, dtype=np.float32)[np.newaxis, :]

        v_curr, cache_v_curr = self.vw.forward(s)
        v_next, _ = self.vw.forward(ns)

        if dead:
            v_next_np = 0.0
        else:
            v_next_np = float(v_next[0, 0])


        target = np.float32(reward + self.gamma * v_next_np)
        vc = float(v_curr[0, 0])
        delta_td = float(target - vc)  # TD error (+ 쪽은 예측보다 좋았음)

        # ── Critic: L_v = ½(V − tgt)²  →  ∂L/∂V = V − tgt
        dv = np.array([[np.float32(vc - target)]], dtype=np.float32)
        self.vw.zero_grad_buffers()
        self.vw.accumulate_value_grad(cache_v_curr, dv)
        self.vw.step()

        # ── Actor: L_π = −δ · log π  (δ 분리)
        self.pi.zero_grad_buffers()
        self.pi.accumulate_pg_grad(cache_pi, action, coeff=delta_td)
        self.pi.step()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--lr_pi", type=float, default=2e-4)
    ap.add_argument("--lr_v", type=float, default=5e-4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    raw, backend = make_cartpole()
    env = CartPoleStepAdapter(raw, backend)

    pi = SoftmaxPolicy(state_dim=4, action_size=2, hidden=args.hidden, lr=args.lr_pi)
    vw = ValueNet(state_dim=4, hidden=args.hidden, lr=args.lr_v)
    agent = AgentActorCritic(pi, vw, gamma=args.gamma)

    rewards_hist: list[float] = []

    try:
        for ep in range(args.episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0.0

            while not done:
                action, cache_pi = agent.get_action(obs)
                next_obs, reward, done, dead = env.step(action)
                agent.update(float(reward), dead, cache_pi, action, obs, next_obs)
                obs = next_obs
                total_reward += reward

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
            axes[0].set_title("Actor-Critic — episode reward")
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
