"""학습된 정책을 내장한 시각화 HTML 생성기 (threadpool 환경).

    python export_policy.py                      # results/의 threadpool DQN 사용
    python export_policy.py --algo ppo
    python export_policy.py --out viz_demo.html

생성된 HTML은 의존성 없는 단일 파일 — 더블클릭으로 브라우저에서 열면
학습된 DQN/PPO 정책이 실시간으로 worker thread 수를 조절하는 모습을
애니메이션으로 볼 수 있다 (v1 threadpool 환경의 충실한 JS 포팅).
발표 데모용: RL / Rule-based / Static / 수동 정책을 즉석에서 전환·비교.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import config as C

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load_weights(algo: str) -> dict:
    """results/threadpool_<algo>_{best|w}.npz 에서 MLP 가중치를 JSON 직렬화."""
    for tag in ("best", "w"):
        if algo == "dqn":
            p = os.path.join(RESULTS, f"threadpool_dqn_{tag}.npz")
            if os.path.exists(p):
                z = np.load(p)
                return {k.replace("/", "_"): z[k].tolist() for k in z.files}
        else:  # ppo: actor만 있으면 행동 결정 가능
            p = os.path.join(RESULTS, f"threadpool_ppo_{tag}.npz.actor.npz")
            if os.path.exists(p):
                z = np.load(p)
                return {k.replace("/", "_"): z[k].tolist() for k in z.files}
    raise FileNotFoundError(
        f"results/에 threadpool_{algo} 가중치가 없습니다.\n"
        f"먼저:  python train.py --env threadpool --algo {algo}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--algo", choices=["dqn", "ppo"], default="dqn")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    weights = load_weights(args.algo)
    p = C.THREADPOOL
    env_cfg = dict(c_min=p["c_min"], c_max=p["c_max"], c_init=p["c_init"],
                   c_step=p["c_step"], mu_thread=p["mu_thread"],
                   c_knee=p["c_knee"], alpha_cs=p["alpha_cs"],
                   lam_base=p["lam_base"], lam_max=p["lam_max"],
                   sla=C.SLA_LATENCY, w_sla=p["w_sla"], w_cost=p["w_cost"],
                   w_thrash=p["w_thrash"], c_base=p.get("c_base"),
                   w_under=p.get("w_under", 0.0), steps=C.EPISODE_STEPS)

    tpl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "viz_template.html")
    html = open(tpl_path, encoding="utf-8").read()
    html = html.replace("/*__POLICY__*/null", json.dumps(weights))
    html = html.replace("/*__ENV__*/null", json.dumps(env_cfg))
    html = html.replace("__ALGO__", args.algo.upper())

    out = args.out or f"viz_simulation_{args.algo}.html"
    open(out, "w", encoding="utf-8").write(html)
    print(f"생성 완료: {out}  (브라우저로 열기)")


if __name__ == "__main__":
    main()
