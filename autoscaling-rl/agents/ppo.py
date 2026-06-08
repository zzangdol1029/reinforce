"""PPO — Clipped Surrogate Objective (DeZero 직접 구현).

구성 요소
---------
- Actor / Critic   : 분리된 MLP (state_dim -> 64 -> 64), tanh 활성화
- GAE(λ)           : advantage 추정의 분산 감소
- Clipped objective: ratio를 [1-ε, 1+ε]로 잘라 과도한 정책 갱신 방지
- Entropy bonus    : 조기 수렴(탐험 부족) 방지

DeZero에는 element-wise min이 없어 min(surr1, surr2)는
선택 마스크(상수)를 곱하는 방식으로 구현했다 — min의 기울기는
"더 작은 쪽으로만 흐른다"와 동일한 결과.
"""
from __future__ import annotations

import numpy as np

from .common import MLP, F, clip_grads
from dezero import optimizers


class PPOAgent:
    def __init__(self, state_dim: int, n_actions: int, cfg: dict, seed: int = 0):
        np.random.seed(seed)
        self.cfg = cfg
        self.n_actions = n_actions

        self.actor = MLP(n_actions, cfg["hidden"], activation="tanh")
        self.critic = MLP(1, cfg["hidden"], activation="tanh")
        dummy = np.zeros((1, state_dim), dtype=np.float32)
        self.actor(dummy)
        self.critic(dummy)
        self.opt_a = optimizers.Adam(cfg["lr"]).setup(self.actor)
        self.opt_c = optimizers.Adam(cfg["lr"]).setup(self.critic)
        self.reset_rollout()

    # ------------------------------------------------------------------
    def reset_rollout(self):
        self.states, self.actions, self.rewards = [], [], []
        self.dones, self.logps, self.values = [], [], []

    def act(self, state: np.ndarray, greedy: bool = False):
        logits = self.actor(state[None].astype(np.float32)).data[0]
        # 수치 안정 softmax
        z = logits - logits.max()
        probs = np.exp(z) / np.exp(z).sum()
        if greedy:
            return int(probs.argmax()), 0.0, 0.0
        a = int(np.random.choice(self.n_actions, p=probs))
        v = float(self.critic(state[None].astype(np.float32)).data[0, 0])
        return a, float(np.log(probs[a] + 1e-8)), v

    def remember(self, s, a, r, done, logp, value):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
        self.dones.append(float(done))
        self.logps.append(logp)
        self.values.append(value)

    # ------------------------------------------------------------------
    def update(self, last_state: np.ndarray) -> dict:
        c = self.cfg
        last_v = float(self.critic(last_state[None].astype(np.float32)).data[0, 0])

        # ---- GAE(λ) ------------------------------------------------------
        rewards = np.asarray(self.rewards, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        values = np.asarray(self.values + [last_v], dtype=np.float32)
        adv = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + c["gamma"] * values[t + 1] * nonterminal - values[t]
            gae = delta + c["gamma"] * c["gae_lambda"] * nonterminal * gae
            adv[t] = gae
        returns = adv + values[:-1]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        S = np.asarray(self.states, dtype=np.float32)
        A = np.asarray(self.actions, dtype=np.int64)
        OLD = np.asarray(self.logps, dtype=np.float32)

        n = len(rewards)
        idx = np.arange(n)
        stats = dict(pi_loss=0.0, v_loss=0.0, entropy=0.0)
        for _ in range(c["epochs"]):
            np.random.shuffle(idx)
            for start in range(0, n, c["minibatch"]):
                mb = idx[start : start + c["minibatch"]]
                m = len(mb)

                logits = self.actor(S[mb])
                probs = F.softmax(logits)
                logp_all = F.log(F.clip(probs, 1e-8, 1.0))
                logp = F.get_item(logp_all, (np.arange(m), A[mb]))
                ratio = F.exp(logp - OLD[mb])

                surr1 = ratio * adv[mb]
                surr2 = F.clip(ratio, 1.0 - c["clip_eps"], 1.0 + c["clip_eps"]) * adv[mb]
                # element-wise min: 작은 쪽만 선택하는 상수 마스크
                take1 = (surr1.data <= surr2.data).astype(np.float32)
                pi_loss = -F.sum(surr1 * take1 + surr2 * (1.0 - take1)) / m

                v = F.reshape(self.critic(S[mb]), (m,))
                v_loss = F.sum((v - returns[mb]) ** 2) / m
                entropy = -F.sum(probs * logp_all) / m

                loss = pi_loss + c["vf_coef"] * v_loss - c["ent_coef"] * entropy
                self.actor.cleargrads()
                self.critic.cleargrads()
                loss.backward()
                clip_grads([self.actor, self.critic], c["max_grad_norm"])
                self.opt_a.update()
                self.opt_c.update()

                stats["pi_loss"] += float(pi_loss.data)
                stats["v_loss"] += float(v_loss.data)
                stats["entropy"] += float(entropy.data)

        self.reset_rollout()
        return stats

    # ------------------------------------------------------------------ I/O
    def save(self, path: str):
        self.actor.save_weights(path + ".actor.npz")
        self.critic.save_weights(path + ".critic.npz")

    def load(self, path: str):
        self.actor.load_weights(path + ".actor.npz")
        self.critic.load_weights(path + ".critic.npz")
