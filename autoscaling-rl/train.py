"""학습 스크립트.

사용법:
    python train.py --env container    --algo dqn          # v1 (공식 기반)
    python train.py --env ev-container --algo ppo          # v2 (이산사건)
    python train.py --env ev-threadpool --algo dqn --episodes 200

옵션:
    --resume          이전 체크포인트에서 이어서 학습
    --max-seconds N   N초 학습 후 체크포인트 저장하고 종료 (분할 학습용)

과적합 방지 장치
----------------
1. 시드 분리 — 학습/검증/평가 트래픽이 절대 겹치지 않음 (config.py 참고)
2. 검증 기반 best 체크포인트 — VAL_INTERVAL 에피소드마다 검증 트래픽으로
   greedy 평가하고, 최고 성능 시점의 가중치를 *_best 로 따로 저장.
   학습 후반의 정책 붕괴가 최종 모델을 오염시키지 않는다.
   evaluate.py는 _best 를 우선 사용.
3. 도메인 랜덤화 — 트래픽 패턴·heavy-tail 강도가 에피소드마다 변경 (env 쪽)

산출물 (results/):
    <env>_<algo>_w.npz        마지막 가중치 (PPO는 .actor/.critic 2개)
    <env>_<algo>_best.npz     검증 최고 가중치  <- 평가에 사용
    <env>_<algo>_curve.npy    학습 곡선 (에피소드 보상)
    <env>_<algo>_val.npy      검증 곡선 [(episode, val_reward), ...]
    <env>_<algo>_state.pkl    분할 학습 재개용 상태
"""
from __future__ import annotations

import argparse
import os
import pickle
import time

import numpy as np

import config as C
from envs import (ContainerAutoscaleEnv, ThreadPoolEnv,
                  EventContainerEnv, EventThreadPoolEnv)
from agents import DQNAgent, PPOAgent

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

ENV_CHOICES = ["container", "threadpool", "ev-container", "ev-threadpool"]


def make_env(name: str, seed: int | None = None):
    """환경 이름 -> 인스턴스. 설정은 전부 config.py에서 가져온다."""
    if name == "container":
        p = C.CONTAINER
        return ContainerAutoscaleEnv(
            c_min=p["c_min"], c_max=p["c_max"], c_init=p["c_init"], mu=p["mu"],
            cold_start_steps=p["cold_start_steps"], lam_base=p["lam_base"],
            lam_max=p["lam_max"], sla=C.SLA_LATENCY, episode_steps=C.EPISODE_STEPS,
            w_sla=p["w_sla"], w_cost=p["w_cost"], w_thrash=p["w_thrash"],
            c_base=p.get("c_base"), w_under=p.get("w_under", 0.0), seed=seed,
        )
    if name == "threadpool":
        p = C.THREADPOOL
        return ThreadPoolEnv(
            c_min=p["c_min"], c_max=p["c_max"], c_init=p["c_init"], c_step=p["c_step"],
            mu_thread=p["mu_thread"], c_knee=p["c_knee"], alpha_cs=p["alpha_cs"],
            lam_base=p["lam_base"], lam_max=p["lam_max"], sla=C.SLA_LATENCY,
            episode_steps=C.EPISODE_STEPS, w_sla=p["w_sla"], w_cost=p["w_cost"],
            w_thrash=p["w_thrash"],
            c_base=p.get("c_base"), w_under=p.get("w_under", 0.0), seed=seed,
        )
    if name == "ev-container":
        return EventContainerEnv(episode_steps=C.EPISODE_STEPS, seed=seed,
                                 **C.EVENT_CONTAINER)
    if name == "ev-threadpool":
        return EventThreadPoolEnv(episode_steps=C.EPISODE_STEPS, seed=seed,
                                  **C.EVENT_THREADPOOL)
    raise ValueError(name)


def paths(env_name: str, algo: str):
    base = os.path.join(RESULTS, f"{env_name}_{algo}")
    return base + "_w.npz", base + "_curve.npy", base + "_state.pkl", base + "_best.npz"


# ------------------------------------------------------------------ 검증
def validate(env_name: str, agent) -> float:
    """검증 트래픽(VAL_SEED, 학습과 분리)에서 greedy 정책의 평균 보상."""
    env = make_env(env_name)
    total = 0.0
    for i in range(C.VAL_EPISODES):
        s, _ = env.reset(seed=C.VAL_SEED + i)
        done = False
        while not done:
            a = agent.act(s, greedy=True)
            if isinstance(a, tuple):     # PPO.act는 (a, logp, v)
                a = a[0]
            s, r, term, trunc, _ = env.step(a)
            done = term or trunc
            total += r
    return total / C.VAL_EPISODES


# ------------------------------------------------------------------ DQN
def train_dqn(env_name: str, episodes: int, seed: int,
              resume: bool, max_seconds: float | None) -> np.ndarray:
    env = make_env(env_name, seed)
    total_steps = episodes * C.EPISODE_STEPS
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n,
                     total_steps, C.DQN, seed=seed)
    w_path, curve_path, state_path, best_path = paths(env_name, "dqn")

    curve, val_log, best_val, ep_start = [], [], -np.inf, 0
    if resume and os.path.exists(state_path):
        agent.load(w_path)
        with open(state_path, "rb") as f:
            st = pickle.load(f)
        from collections import deque
        agent.buffer.buf = deque(st["buffer"], maxlen=C.DQN["buffer_size"])
        agent.step_count = st["step_count"]
        curve, val_log, best_val = list(st["curve"]), list(st["val_log"]), st["best_val"]
        ep_start = st["ep_done"]
        print(f"[DQN/{env_name}] resume from ep {ep_start} (best val={best_val:.2f})")

    t0 = time.time()
    for ep in range(ep_start, episodes):
        s, _ = env.reset(seed=seed * 10_000 + ep)    # 학습 시드 영역
        ep_r, done = 0.0, False
        while not done:
            a = agent.act(s)
            s2, r, term, trunc, _ = env.step(a)
            done = term or trunc
            agent.remember(s, a, r, s2, done)
            if agent.step_count % C.DQN.get("train_freq", 1) == 0:
                agent.learn()
            s = s2
            ep_r += r
        curve.append(ep_r)

        # ---- 검증 기반 best 체크포인트 (과적합/후반 붕괴 방지) ----
        if (ep + 1) % C.VAL_INTERVAL == 0:
            v = validate(env_name, agent)
            val_log.append((ep + 1, v))
            if v > best_val:
                best_val = v
                agent.save(best_path)
            print(f"[DQN/{env_name}] ep {ep+1:4d}/{episodes}  "
                  f"R(10)={np.mean(curve[-10:]):8.2f}  val={v:8.2f}  "
                  f"best={best_val:8.2f}  eps={agent.epsilon:.3f}  "
                  f"({time.time()-t0:.0f}s)")
        if max_seconds and time.time() - t0 > max_seconds:
            break

    agent.save(w_path)
    with open(state_path, "wb") as f:
        pickle.dump({"buffer": list(agent.buffer.buf), "step_count": agent.step_count,
                     "curve": curve, "val_log": val_log, "best_val": best_val,
                     "ep_done": len(curve)}, f)
    np.save(curve_path, np.asarray(curve))
    np.save(curve_path.replace("_curve", "_val"), np.asarray(val_log))
    return np.asarray(curve)


# ------------------------------------------------------------------ PPO
def train_ppo(env_name: str, episodes: int, seed: int,
              resume: bool, max_seconds: float | None) -> np.ndarray:
    env = make_env(env_name, seed)
    agent = PPOAgent(env.observation_space.shape[0], env.action_space.n,
                     C.PPO, seed=seed)
    w_path, curve_path, state_path, best_path = paths(env_name, "ppo")

    curve, val_log, best_val, ep_start = [], [], -np.inf, 0
    if resume and os.path.exists(state_path):
        agent.load(w_path)
        with open(state_path, "rb") as f:
            st = pickle.load(f)
        curve, val_log, best_val = list(st["curve"]), list(st["val_log"]), st["best_val"]
        ep_start = st["ep_done"]
        print(f"[PPO/{env_name}] resume from ep {ep_start} (best val={best_val:.2f})")

    rollout = 0
    t0 = time.time()
    for ep in range(ep_start, episodes):
        s, _ = env.reset(seed=seed * 10_000 + ep)
        ep_r, done = 0.0, False
        while not done:
            a, logp, v = agent.act(s)
            s2, r, term, trunc, _ = env.step(a)
            done = term or trunc
            agent.remember(s, a, r, done, logp, v)
            rollout += 1
            if rollout >= C.PPO["rollout_steps"]:
                agent.update(s2)
                rollout = 0
            s = s2
            ep_r += r
        curve.append(ep_r)

        if (ep + 1) % C.VAL_INTERVAL == 0:
            v = validate(env_name, agent)
            val_log.append((ep + 1, v))
            if v > best_val:
                best_val = v
                agent.save(best_path)
            print(f"[PPO/{env_name}] ep {ep+1:4d}/{episodes}  "
                  f"R(10)={np.mean(curve[-10:]):8.2f}  val={v:8.2f}  "
                  f"best={best_val:8.2f}  ({time.time()-t0:.0f}s)")
        if max_seconds and time.time() - t0 > max_seconds:
            break

    agent.save(w_path)
    with open(state_path, "wb") as f:
        pickle.dump({"curve": curve, "val_log": val_log, "best_val": best_val,
                     "ep_done": len(curve)}, f)
    np.save(curve_path, np.asarray(curve))
    np.save(curve_path.replace("_curve", "_val"), np.asarray(val_log))
    return np.asarray(curve)


# ----------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env", choices=ENV_CHOICES, required=True)
    ap.add_argument("--algo", choices=["dqn", "ppo"], required=True)
    ap.add_argument("--episodes", type=int, default=C.TRAIN_EPISODES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=None)
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    fn = train_dqn if args.algo == "dqn" else train_ppo
    curve = fn(args.env, args.episodes, args.seed, args.resume, args.max_seconds)
    print(f"done: {len(curve)} episodes, final R(10)={np.mean(curve[-10:]):.2f}")
