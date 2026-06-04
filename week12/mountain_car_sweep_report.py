"""
Mountain Car HP 스윕 공통 — 통계·CSV·비교 그래프·최적 설정 요약
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class SweepStats:
    max_total_reward: float
    max_reward_episode: int
    mean_last_50: float
    min_total_reward: float
    mean_all: float
    std_all: float
    success_count: int
    success_rate_pct: float


def stats_from_history(
    reward_history: list[float],
    *,
    success_count: int,
    tail_n: int = 50,
) -> SweepStats:
    arr = np.asarray(reward_history, dtype=np.float64)
    n = len(arr)
    tail = arr[-min(tail_n, n) :] if n else arr
    return SweepStats(
        max_total_reward=float(arr.max()) if n else float("nan"),
        max_reward_episode=int(arr.argmax()) if n else -1,
        mean_last_50=float(tail.mean()) if len(tail) else float("nan"),
        min_total_reward=float(arr.min()) if n else float("nan"),
        mean_all=float(arr.mean()) if n else float("nan"),
        std_all=float(arr.std()) if n > 1 else 0.0,
        success_count=success_count,
        success_rate_pct=100.0 * success_count / n if n else 0.0,
    )


def pick_evenly_spaced(grid: Sequence[tuple], n: int) -> list[tuple]:
    if len(grid) <= n:
        return list(grid)
    idx = np.linspace(0, len(grid) - 1, n, dtype=int)
    return [grid[i] for i in idx]


def print_best_banner(
    *,
    title: str,
    best_name: str,
    best_max: float,
    best_mean50: float,
    best_success_rate: float,
    hp_lines: list[str],
    out_dir: Path,
    rank_by: str,
) -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  ★★★ 최적 Hyper-parameter ({title}) ★★★")
    print(sep)
    print(f"  선택 기준: {rank_by}")
    print(f"  설정 이름: {best_name}")
    print(f"  max total reward : {best_max:.2f}  (0에 가까울수록 좋음, -200=실패)")
    print(f"  mean(last 50)    : {best_mean50:.2f}")
    print(f"  success rate     : {best_success_rate:.1f}%")
    print("  Hyper-parameters:")
    for line in hp_lines:
        print(f"    {line}")
    print(f"  결과 폴더: {out_dir / 'best'}")
    print(sep)


def save_best_hyperparameters(
    path: Path,
    *,
    title: str,
    best_name: str,
    rank_by: str,
    hp_dict: dict[str, Any],
    stats: SweepStats,
    episodes: int,
    seed: int | None,
    n_runs: int,
) -> None:
    lines = [
        title,
        "",
        "========== ★ 최적 Hyper-parameter ==========",
        f"best_name: {best_name}",
        f"selection_criterion: {rank_by}",
        f"n_hp_runs: {n_runs}",
        "",
        "--- Hyper-parameters ---",
        *[f"{k}: {v}" for k, v in hp_dict.items() if k != "name"],
        "",
        "--- 성능 통계 ---",
        f"max_total_reward: {stats.max_total_reward:.4f}",
        f"max_reward_episode: {stats.max_reward_episode}",
        f"mean_last_50: {stats.mean_last_50:.4f}",
        f"min_total_reward: {stats.min_total_reward:.4f}",
        f"mean_all_episodes: {stats.mean_all:.4f}",
        f"std_all_episodes: {stats.std_all:.4f}",
        f"success_count: {stats.success_count}",
        f"success_rate_pct: {stats.success_rate_pct:.2f}",
        f"episodes: {episodes}",
        f"seed: {seed}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_sweep_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"CSV 저장: {path}")


def save_statistics_table_png(
    path: Path,
    *,
    title: str,
    headers: list[str],
    table_rows: list[list[str]],
    highlight_row: int = 0,
    show: bool,
) -> None:
    fig_h = max(6, 0.35 * len(table_rows) + 1.5)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.axis("off")
    tbl = ax.table(
        cellText=table_rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1.0, 1.35)
    for j in range(len(headers)):
        tbl[(0, j)].set_facecolor("#4472C4")
        tbl[(0, j)].set_text_props(color="white", weight="bold")
    hr = highlight_row + 1
    for j in range(len(headers)):
        tbl[(hr, j)].set_facecolor("#FFF2CC")
    ax.set_title(title, fontsize=11, pad=12)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"통계 표 그래프 저장: {path}")
    if show:
        plt.show()
    plt.close()


def save_metric_bar_chart(
    path: Path,
    *,
    names: list[str],
    values: list[float],
    ylabel: str,
    title: str,
    highlight_index: int = 0,
    show: bool,
) -> None:
    n = len(names)
    fig_h = max(8, 0.32 * n)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    colors = ["#2E7D32" if i == highlight_index else "#5C9FD4" for i in range(n)]
    y_pos = np.arange(n)
    ax.barh(y_pos, values, color=colors, alpha=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    for i, v in enumerate(values):
        ax.text(v, i, f" {v:.1f}", va="center", fontsize=6)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    print(f"막대 그래프 저장: {path}")
    if show:
        plt.show()
    plt.close()


def save_episode_reward_plot(
    reward_history: list[float],
    path: Path,
    *,
    title: str = "Episode total reward",
    show: bool,
) -> None:
    """제출용: 한 HP run 의 episode 별 total reward 곡선."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.xlabel("episode")
    plt.ylabel("total reward")
    plt.plot(range(len(reward_history)), reward_history)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"Episode reward 그래프 저장: {path}")
    if show:
        plt.show()
    plt.close()


def save_top_k_overlay(
    path: Path,
    histories: list[tuple[str, list[float], float]],
    *,
    title: str,
    show: bool,
) -> None:
    plt.figure(figsize=(10, 5))
    for name, hist, mx in histories:
        plt.plot(hist, alpha=0.85, linewidth=1.5, label=f"{name} (max={mx:.0f})")
    plt.xlabel("episode")
    plt.ylabel("total reward")
    plt.title(title)
    plt.legend(fontsize=7, loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    print(f"Top-K 곡선 저장: {path}")
    if show:
        plt.show()
    plt.close()


def save_scatter_max_vs_mean50(
    path: Path,
    *,
    names: list[str],
    max_rewards: list[float],
    mean50: list[float],
    best_idx: int,
    title: str,
    show: bool,
) -> None:
    plt.figure(figsize=(8, 6))
    for i, name in enumerate(names):
        c = "#2E7D32" if i == best_idx else "#5C9FD4"
        s = 120 if i == best_idx else 50
        plt.scatter(mean50[i], max_rewards[i], c=c, s=s, alpha=0.85, edgecolors="k", linewidths=0.3)
        if i == best_idx or i % 4 == 0:
            plt.annotate(name, (mean50[i], max_rewards[i]), fontsize=6, xytext=(4, 4), textcoords="offset points")
    plt.xlabel("mean reward (last 50 episodes)")
    plt.ylabel("max total reward")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    print(f"산점도 저장: {path}")
    if show:
        plt.show()
    plt.close()


def publish_sweep_report(
    *,
    out_dir: Path,
    title: str,
    results: Sequence[Any],
    rank_key: str,
    csv_fieldnames: list[str],
    row_to_csv: Callable[[Any, int], dict[str, Any]],
    table_headers: list[str],
    row_to_table: Callable[[Any, int], list[str]],
    hp_dict_for_best: Callable[[Any], dict[str, Any]],
    hp_lines_for_best: Callable[[Any], list[str]],
    histories_for_plot: Callable[[Sequence[Any]], list[tuple[str, list[float], float]]],
    episodes: int,
    seed: int | None,
    show_plot: bool,
    top_k: int = 5,
) -> Any:
    """results 를 정렬·저장·그래프 생성 후 1위 반환."""
    ranked = sorted(results, key=lambda r: getattr(r, rank_key), reverse=True)
    best = ranked[0]
    best_idx = 0

    rows_csv = [row_to_csv(r, i) for i, r in enumerate(ranked, 1)]
    save_sweep_csv(out_dir / "sweep_results.csv", rows_csv, csv_fieldnames)

    summary_lines = [
        title,
        f"총 HP 실험 수: {len(ranked)}",
        f"episodes: {episodes}",
        f"seed: {seed}",
        f"순위 기준: {rank_key} (높을수록 좋음)",
        "",
        f"BEST_NAME: {best.name}",
        f"BEST_{rank_key}: {getattr(best, rank_key):.4f}",
        "",
        "전체 순위는 sweep_results.csv / sweep_statistics_table.png 참고",
    ]
    (out_dir / "sweep_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"요약 저장: {out_dir / 'sweep_summary.txt'}")

    table_rows = [row_to_table(r, i) for i, r in enumerate(ranked, 1)]
    save_statistics_table_png(
        out_dir / "sweep_statistics_table.png",
        title=f"{title} — HP sweep stats (n={len(ranked)}, row1=best)",
        headers=table_headers,
        table_rows=table_rows,
        highlight_row=0,
        show=show_plot,
    )

    names = [r.name for r in ranked]
    max_vals = [r.max_total_reward for r in ranked]
    mean50_vals = [r.mean_last_50 for r in ranked]

    save_metric_bar_chart(
        out_dir / "sweep_max_reward_ranking.png",
        names=names,
        values=max_vals,
        ylabel="max total reward",
        title=f"{title} — max total reward ranking (n={len(ranked)})",
        highlight_index=best_idx,
        show=show_plot,
    )
    save_metric_bar_chart(
        out_dir / "sweep_mean_last50_ranking.png",
        names=names,
        values=mean50_vals,
        ylabel="mean(last 50)",
        title=f"{title} — mean(last 50) ranking",
        highlight_index=best_idx,
        show=show_plot,
    )
    save_scatter_max_vs_mean50(
        out_dir / "sweep_scatter_max_vs_mean50.png",
        names=names,
        max_rewards=max_vals,
        mean50=mean50_vals,
        best_idx=best_idx,
        title=f"{title} — max vs mean(last 50)",
        show=show_plot,
    )

    top_hist = histories_for_plot(ranked[:top_k])
    save_top_k_overlay(
        out_dir / f"sweep_top{top_k}_episode_rewards.png",
        top_hist,
        title=f"{title} — top {top_k} learning curves",
        show=show_plot,
    )

    stats = SweepStats(
        max_total_reward=best.max_total_reward,
        max_reward_episode=best.max_reward_episode,
        mean_last_50=best.mean_last_50,
        min_total_reward=best.min_total_reward,
        mean_all=best.mean_all,
        std_all=best.std_all,
        success_count=best.success_count,
        success_rate_pct=best.success_rate_pct,
    )
    hp_dict = hp_dict_for_best(best)
    save_best_hyperparameters(
        out_dir / "best" / "BEST_hyperparameters.txt",
        title=title,
        best_name=best.name,
        rank_by=rank_key,
        hp_dict=hp_dict,
        stats=stats,
        episodes=episodes,
        seed=seed,
        n_runs=len(ranked),
    )

    best_dir = out_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    best_curve = best_dir / "episode_total_reward.png"
    hist = getattr(best, "reward_history", None)
    if hist:
        save_episode_reward_plot(
            hist,
            best_curve,
            title=f"Best: {best.name} — episode total reward",
            show=show_plot,
        )
    else:
        best_run_dir = Path(best.run_dir)
        src = best_run_dir / "episode_total_reward.png"
        if src.exists():
            best_curve.write_bytes(src.read_bytes())
            print(f"Episode reward 그래프 복사: {best_curve}")

    graph_files = [
        out_dir / "sweep_statistics_table.png",
        out_dir / "sweep_max_reward_ranking.png",
        out_dir / "sweep_mean_last50_ranking.png",
        out_dir / "sweep_scatter_max_vs_mean50.png",
        out_dir / f"sweep_top{top_k}_episode_rewards.png",
        best_dir / "episode_total_reward.png",
    ]
    print("\n--- 생성된 그래프 (PNG) ---")
    for p in graph_files:
        mark = "OK" if p.exists() else "MISSING"
        print(f"  [{mark}] {p}")

    print_best_banner(
        title=title,
        best_name=best.name,
        best_max=best.max_total_reward,
        best_mean50=best.mean_last_50,
        best_success_rate=best.success_rate_pct,
        hp_lines=hp_lines_for_best(best),
        out_dir=out_dir,
        rank_by=rank_key,
    )

    return best
