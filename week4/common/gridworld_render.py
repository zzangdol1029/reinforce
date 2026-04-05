"""
week4에서 week3의 gridworld_render(Renderer)를 그대로 재사용하기 위한 래퍼.

week3/gridworld.py는 `import common.gridworld_render as render_helper`를 사용합니다.
week4에서 실행할 때 `common` 패키지는 week4/common 이 우선 인식되므로,
여기에 gridworld_render를 제공해 week3 렌더러를 그대로 연결합니다.

week3 소스는 수정하지 않습니다.

추가: Q(s,a)에 대해 render_v와 같은 스타일의 matplotlib 히트맵(행동별 2×2 패널).
창 표시는 스크립트에서 `import common.mpl_window`를 GridWorld보다 먼저 호출.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_WEEK3_COMMON_DIR = Path(__file__).resolve().parent.parent.parent / "week3" / "common"
_WEEK3_RENDER_PATH = _WEEK3_COMMON_DIR / "gridworld_render.py"

_spec = importlib.util.spec_from_file_location("week3_common_gridworld_render", _WEEK3_RENDER_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load week3 gridworld_render module from: {_WEEK3_RENDER_PATH}")

_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

_BaseRenderer = _module.Renderer

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import FancyArrowPatch, Rectangle

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
    plt = None  # type: ignore
    TwoSlopeNorm = None  # type: ignore
    Rectangle = None  # type: ignore

_ACTION_LABELS = ("UP (a=0)", "DOWN (a=1)", "LEFT (a=2)", "RIGHT (a=3)")


class Renderer(_BaseRenderer):
    """week3 Renderer + Q(s,a) matplotlib, 실습 슬라이드 스타일 V(s)."""

    def _render_v_matplotlib(
        self,
        v,
        policy,
        print_value,
        *,
        ax=None,
        show=True,
        title=None,
        draw_colorbar=False,
    ):
        """
        실습 자료 슬라이드와 동일한 레이아웃:
        - V 숫자: 각 칸 우상단
        - R / GOAL 라벨: 보상이 있는 칸 우하단(칸 안)
        - 얇은 회색 격자선
        (week3 기본은 숫자 중앙 + 격자 오른쪽에 R 라벨)
        """
        _module._configure_matplotlib_korean_font()

        values = np.zeros((self.height, self.width))
        wall_mask = np.zeros((self.height, self.width), dtype=bool)

        for h in range(self.height):
            for w in range(self.width):
                if (h, w) in self.wall_states:
                    wall_mask[h, w] = True
                    values[h, w] = np.nan
                else:
                    values[h, w] = self._v_at(v, h, w) if print_value else 0.0

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 4.5))
        else:
            fig = ax.figure
        masked = np.ma.masked_where(wall_mask | np.isnan(values), values)

        vmax = max(np.nanmax(np.abs(values)), 0.01)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        cmap = plt.cm.RdYlGn
        im = ax.imshow(masked, cmap=cmap, norm=norm, aspect="equal", origin="upper", zorder=0)

        ax.set_xticks(np.arange(-0.5, self.width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.height, 1), minor=True)
        ax.grid(which="minor", color="#cccccc", linewidth=0.8, zorder=2)
        ax.set_axisbelow(True)

        for wh, ww in self.wall_states:
            ax.add_patch(
                Rectangle(
                    (ww - 0.5, wh - 0.5),
                    1,
                    1,
                    facecolor="#555555",
                    edgecolor="black",
                    linewidth=1,
                    zorder=10,
                )
            )

        # V값: 칸 우상단 (실습 슬라이드)
        for h in range(self.height):
            for w in range(self.width):
                if (h, w) in self.wall_states:
                    continue
                if print_value:
                    ax.text(
                        w + 0.45,
                        h - 0.45,
                        f"{values[h, w]:.2f}",
                        ha="right",
                        va="top",
                        color="black",
                        fontsize=10,
                        zorder=5,
                    )

        if policy is not None:
            dirs = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            max_p = 0.01
            for h in range(self.height):
                for w in range(self.width):
                    if (h, w) in self.wall_states or (h, w) == self.goal_state or (h, w) in self.trap_states:
                        continue
                    state = (h, w)
                    for a in range(4):
                        p = self._pi_prob(policy, state, a)
                        max_p = max(max_p, p)
            for h in range(self.height):
                for w in range(self.width):
                    if (h, w) in self.wall_states or (h, w) == self.goal_state or (h, w) in self.trap_states:
                        continue
                    state = (h, w)
                    for a in range(4):
                        p = self._pi_prob(policy, state, a)
                        if p <= 0:
                            continue
                        dx, dy = dirs[a]
                        scale = 0.22 * (p / max_p) ** 0.5
                        x0, y0 = float(w), float(h)
                        ax.add_patch(
                            FancyArrowPatch(
                                (x0, y0),
                                (x0 + dx * scale, y0 + dy * scale),
                                arrowstyle="-|>",
                                mutation_scale=8,
                                color="navy",
                                linewidth=0.8,
                                zorder=6,
                            )
                        )

        ax.set_xticks(np.arange(self.width))
        ax.set_yticks(np.arange(self.height))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(self.height - 0.5, -0.5)

        # R 라벨: 보상 칸 우하단 (실습 슬라이드). 격자 밖이 아닌 칸 안.
        for h in range(self.height):
            for w in range(self.width):
                r = self.reward_map[h, w]
                if r is None or r == 0:
                    continue
                if (h, w) == self.goal_state and r > 0:
                    ax.text(
                        w + 0.45,
                        h + 0.45,
                        f"R {r:g} (GOAL)",
                        ha="right",
                        va="bottom",
                        fontsize=8,
                        color="darkgreen",
                        zorder=12,
                    )
                elif r < 0:
                    ax.text(
                        w + 0.45,
                        h + 0.45,
                        f"R {r:g}",
                        ha="right",
                        va="bottom",
                        fontsize=8,
                        color="darkred",
                        zorder=12,
                    )

        if draw_colorbar:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="V(s)")

        if title is None:
            title = "V(s) and policy" if policy is not None else "V(s)"
        ax.set_title(title)

        if ax is None:
            plt.tight_layout()

        if show:
            plt.show()

    def _q_at(self, q, h, w, a):
        if q is None:
            return 0.0
        if isinstance(q, dict):
            return float(q.get(((h, w), a), 0.0))
        if isinstance(q, np.ndarray) and q.ndim == 3:
            return float(q[h, w, a])
        return 0.0

    def render_q(
        self,
        q=None,
        print_value=True,
        *,
        use_matplotlib=True,
        show=True,
        savefig=None,
    ):
        """
        Q(s,a) 출력.
        - use_matplotlib=True이고 matplotlib 사용 가능: 행동별 2×2 히트맵.
        - 그렇지 않으면 week3와 동일하게 텍스트만 출력.
        """
        if use_matplotlib and _HAS_MPL:
            self._render_q_matplotlib(q, print_value, show=show, savefig=savefig)
        else:
            super().render_q(q, print_value)

    def _render_q_matplotlib(self, q, print_value, *, show=True, savefig=None):
        _module._configure_matplotlib_korean_font()

        # 전체 패널에서 동일한 색 스케일(|Q| 최댓값)
        abs_vals = []
        for h in range(self.height):
            for w in range(self.width):
                if (h, w) in self.wall_states:
                    continue
                for a in range(4):
                    abs_vals.append(abs(self._q_at(q, h, w, a)))
        vmax = max(max(abs_vals) if abs_vals else 0.0, 0.01)

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes_flat = axes.flatten()

        for a in range(4):
            ax = axes_flat[a]
            values = np.zeros((self.height, self.width))
            wall_mask = np.zeros((self.height, self.width), dtype=bool)

            for h in range(self.height):
                for w in range(self.width):
                    if (h, w) in self.wall_states:
                        wall_mask[h, w] = True
                        values[h, w] = np.nan
                    else:
                        values[h, w] = self._q_at(q, h, w, a) if print_value else 0.0

            masked = np.ma.masked_where(wall_mask | np.isnan(values), values)
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
            cmap = plt.cm.RdYlGn
            ax.imshow(masked, cmap=cmap, norm=norm, aspect="equal", origin="upper")

            for wh, ww in self.wall_states:
                ax.add_patch(
                    Rectangle(
                        (ww - 0.5, wh - 0.5),
                        1,
                        1,
                        facecolor="#555555",
                        edgecolor="black",
                        linewidth=1,
                        zorder=10,
                    )
                )

            if print_value:
                for h in range(self.height):
                    for w in range(self.width):
                        if (h, w) in self.wall_states:
                            continue
                        qa = self._q_at(q, h, w, a)
                        ax.text(
                            w,
                            h,
                            f"{qa:.2f}",
                            ha="center",
                            va="center",
                            color="black",
                            fontsize=9,
                            zorder=5,
                        )

            ax.set_xticks(np.arange(self.width))
            ax.set_yticks(np.arange(self.height))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xlim(-0.5, self.width - 0.5)
            ax.set_ylim(self.height - 0.5, -0.5)
            ax.set_title(_ACTION_LABELS[a])

        fig.suptitle("Q(s, a)", fontsize=12)
        plt.tight_layout()

        if savefig:
            fig.savefig(savefig, dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)


    def render_q_and_policy(self, q, *, show=True, savefig=None):
        """
        한 figure에 두 패널:
        - 왼쪽: Q(s,a) 마름모 시각화 (셀을 4등분 삼각형, 슬라이드 스타일)
        - 오른쪽: greedy policy 화살표 (↑↓←→)
        """
        if not _HAS_MPL:
            super().render_q(q)
            return
        _module._configure_matplotlib_korean_font()
        fig, (ax_q, ax_pi) = plt.subplots(1, 2, figsize=(13, 4.5))
        self._draw_q_diamond(q, ax_q)
        self._draw_greedy_policy(q, ax_pi)
        ax_q.set_title("Q 함수 시각화", fontsize=11)
        ax_pi.set_title("Q 함수에 대한 greedy policy", fontsize=11)
        plt.tight_layout()
        if savefig:
            fig.savefig(savefig, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)

    def _draw_q_diamond(self, q, ax):
        """각 셀을 4등분 삼각형으로 Q(s,a)를 색상·숫자로 표시."""
        from matplotlib.patches import Polygon as MplPolygon

        abs_vals = [
            abs(self._q_at(q, h, w, a))
            for h in range(self.height)
            for w in range(self.width)
            if (h, w) not in self.wall_states
            for a in range(4)
        ]
        vmax = max(max(abs_vals) if abs_vals else 0.0, 0.01)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        cmap = plt.cm.RdYlGn

        for h in range(self.height):
            for w in range(self.width):
                if (h, w) in self.wall_states:
                    ax.add_patch(Rectangle(
                        (w - 0.5, h - 0.5), 1, 1,
                        facecolor="#666666", edgecolor="#aaaaaa", lw=0.8, zorder=2,
                    ))
                    continue

                cx, cy = float(w), float(h)
                # 삼각형 꼭짓점: 0=UP(위), 1=DOWN(아래), 2=LEFT(왼쪽), 3=RIGHT(오른쪽)
                tri_verts = [
                    [(cx - 0.5, cy - 0.5), (cx + 0.5, cy - 0.5), (cx, cy)],
                    [(cx - 0.5, cy + 0.5), (cx + 0.5, cy + 0.5), (cx, cy)],
                    [(cx - 0.5, cy - 0.5), (cx - 0.5, cy + 0.5), (cx, cy)],
                    [(cx + 0.5, cy - 0.5), (cx + 0.5, cy + 0.5), (cx, cy)],
                ]
                txt_pos = [
                    (cx,        cy - 0.28),
                    (cx,        cy + 0.28),
                    (cx - 0.28, cy       ),
                    (cx + 0.28, cy       ),
                ]

                for a, (verts, (tx, ty)) in enumerate(zip(tri_verts, txt_pos)):
                    qa = self._q_at(q, h, w, a)
                    poly = MplPolygon(
                        verts, facecolor=cmap(norm(qa)),
                        edgecolor="white", lw=0.5, zorder=3,
                    )
                    ax.add_patch(poly)
                    ax.text(tx, ty, f"{qa:.2f}",
                            ha="center", va="center", fontsize=7, zorder=5)

        # R 라벨 (격자 오른쪽)
        for h in range(self.height):
            for w in range(self.width):
                r = self.reward_map[h, w]
                if r is None or r == 0:
                    continue
                x_lbl = self.width - 0.5 + 0.1
                if (h, w) == self.goal_state and r > 0:
                    ax.text(x_lbl, h, f"R {r:g} (GOAL)", ha="left", va="center", fontsize=8)
                elif r < 0:
                    ax.text(x_lbl, h, f"R {r:g}", ha="left", va="center", fontsize=8)

        ax.set_xlim(-0.5, self.width - 0.5 + 1.6)
        ax.set_ylim(self.height - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.axis("off")

    def _draw_greedy_policy(self, q, ax):
        """greedy policy (max_a Q(s,a)) 화살표 그리기."""
        for h in range(self.height):
            for w in range(self.width):
                fc = "#666666" if (h, w) in self.wall_states else "white"
                ax.add_patch(Rectangle(
                    (w - 0.5, h - 0.5), 1, 1,
                    facecolor=fc, edgecolor="#cccccc", lw=1,
                ))

        # 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        arrow_delta = [(0, -0.32), (0, 0.32), (-0.32, 0), (0.32, 0)]

        for h in range(self.height):
            for w in range(self.width):
                if (h, w) in self.wall_states:
                    continue
                if (h, w) == self.goal_state or (h, w) in self.trap_states:
                    continue
                best_a = max(range(4), key=lambda a: self._q_at(q, h, w, a))
                dx, dy = arrow_delta[best_a]
                ax.annotate(
                    "",
                    xy=(w + dx, h + dy),
                    xytext=(w - dx, h - dy),
                    arrowprops=dict(
                        arrowstyle="-|>", color="black", lw=1.5, mutation_scale=15,
                    ),
                    zorder=5,
                )

        # R 라벨 (격자 오른쪽)
        for h in range(self.height):
            for w in range(self.width):
                r = self.reward_map[h, w]
                if r is None or r == 0:
                    continue
                x_lbl = self.width - 0.5 + 0.1
                if (h, w) == self.goal_state and r > 0:
                    ax.text(x_lbl, h, f"R {r:g} (GOAL)", ha="left", va="center", fontsize=8)
                elif r < 0:
                    ax.text(x_lbl, h, f"R {r:g}", ha="left", va="center", fontsize=8)

        ax.set_xlim(-0.5, self.width - 0.5 + 1.6)
        ax.set_ylim(self.height - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.axis("off")


__all__ = ["Renderer"]
