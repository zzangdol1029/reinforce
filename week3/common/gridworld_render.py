"""
GridWorld 시각화(Rendering) 모듈

이 파일의 목적
--------------
GridWorld 환경에서 계산한
- 가치함수 V(s)
- 정책 π(a|s)
를 사람이 보기 쉬운 형태로 출력하는 도구를 제공합니다.

지원 출력
---------
1) 텍스트 출력(터미널)
   - 숫자 격자 형태로 V(s)를 출력
   - 벽(###), 목표(G), 함정(T) 표시

2) matplotlib 출력(그림)
   - V(s)를 색으로 표현 (양수: 초록, 음수: 빨강, 0: 노랑 근처)
   - 벽은 회색으로 표시
   - 정책 π가 있으면 행동 확률에 비례한 화살표를 그림
   - reward_map에 있는 비영(非0) 보상은 오른쪽에 R 라벨로 표시

주의
----
matplotlib이 설치되어 있지 않거나 GUI backend가 없는 환경에서는
텍스트 출력으로 자동 fallback 됩니다.
"""

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import FancyArrowPatch, Rectangle

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
    Rectangle = None


class Renderer:
    def __init__(self, reward_map, goal_state, wall_states, trap_states=None):
        """
        Args:
            reward_map: 보상 맵 (numpy array, dtype=object 가능; None은 벽에 사용 가능)
            goal_state: 목표 상태 (h, w)
            wall_states: 벽 상태. (h, w) 단일 튜플 또는 set/frozenset/list 등 iterable
            trap_states: 함정(종료) 상태들의 집합 (없으면 None)
        """
        self.reward_map = reward_map
        self.goal_state = goal_state

        # 3x4는 단일 wall_state를 넘기고, 5x5는 여러 벽(set)을 넘기므로 둘 다 지원
        if isinstance(wall_states, tuple) and len(wall_states) == 2:
            self.wall_states = frozenset([wall_states])
        else:
            self.wall_states = frozenset(wall_states)

        self.trap_states = frozenset(trap_states or [])
        self.height = len(reward_map)
        self.width = len(reward_map[0])

    def _v_at(self, v, h, w):
        """
        V(s) 조회 헬퍼.
        - v가 None이면 0으로 간주
        - v가 dict이면 key=(h,w)로 조회
        - v가 numpy array이면 v[h,w]로 조회
        """
        if v is None:
            return 0.0
        if isinstance(v, dict):
            return float(v.get((h, w), 0.0))
        if isinstance(v, np.ndarray):
            return float(v[h, w])
        return 0.0

    def _pi_prob(self, policy, state, action):
        """
        π(a|s) 조회 헬퍼.
        - policy[state]가 dict 형태(예: {0:0.25,1:0.25,...})라고 가정
        - 없거나 에러면 0으로 간주
        """
        if policy is None:
            return 0.0
        try:
            d = policy[state]
        except (KeyError, TypeError):
            return 0.0
        if isinstance(d, dict):
            return float(d.get(action, 0.0))
        return 0.0

    def render_v(
        self,
        v=None,
        policy=None,
        print_value=True,
        use_matplotlib=True,
        *,
        ax=None,
        show=None,
        title=None,
        draw_colorbar=False,
    ):
        """
        V(s) 시각화. matplotlib 사용 가능하고 use_matplotlib=True이면
        슬라이드처럼 색상(양수 녹/음수 적), 벽 회색, R 라벨, 정책 pi가 있으면 화살표.
        """
        # show 기본값: ax가 주어졌으면 외부에서 show()하므로 False, 아니면 True
        if show is None:
            show = ax is None

        # use_matplotlib=True 이고 matplotlib import에 성공했을 때만 그림으로 렌더링
        if use_matplotlib and _HAS_MPL:
            self._render_v_matplotlib(
                v,
                policy,
                print_value,
                ax=ax,
                show=show,
                title=title,
                draw_colorbar=draw_colorbar,
            )
        else:
            self._render_v_text(v, policy, print_value)

    def _render_v_text(self, v=None, policy=None, print_value=True):
        """
        텍스트(터미널) 렌더링.
        - 벽: ###, 목표: G, 함정: T
        - print_value=True면 각 칸에 V값을 함께 표시
        """
        for h in range(self.height):
            row_parts = []
            for w in range(self.width):
                cell = (h, w)
                if cell in self.wall_states:
                    row_parts.append("  ###  ")
                elif cell == self.goal_state:
                    val = self._v_at(v, h, w) if print_value else 0.0
                    row_parts.append(f" G{val:5.2f}" if print_value else "  G   ")
                elif cell in self.trap_states:
                    val = self._v_at(v, h, w) if print_value else 0.0
                    row_parts.append(f" T{val:5.2f}" if print_value else "  T   ")
                else:
                    val = self._v_at(v, h, w)
                    row_parts.append(f"{val:7.2f}" if print_value else "  .   ")
            print("  " + "".join(row_parts))
        print()

    def _render_v_matplotlib(self, v, policy, print_value, *, ax=None, show=True, title=None, draw_colorbar=False):
        """
        matplotlib 렌더링.
        - values: 각 칸의 V(s) 값(벽은 NaN)
        - TwoSlopeNorm: 0을 중심으로 음/양을 같은 스케일로 표현
        - RdYlGn: 음수(빨강) → 0(노랑) → 양수(초록)
        - policy가 있으면 화살표를 그림(확률에 비례해 길이 스케일)
        """
        values = np.zeros((self.height, self.width))
        wall_mask = np.zeros((self.height, self.width), dtype=bool)

        for h in range(self.height):
            for w in range(self.width):
                if (h, w) in self.wall_states:
                    wall_mask[h, w] = True
                    values[h, w] = np.nan
                else:
                    values[h, w] = self._v_at(v, h, w) if print_value else 0.0

        # ax가 주어지면 해당 subplot에 그립니다. 없으면 새 figure/ax 생성.
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 4.5))
        else:
            fig = ax.figure
        masked = np.ma.masked_where(wall_mask | np.isnan(values), values)

        # 색상 스케일을 |V|의 최댓값에 맞춤 (0으로 나누기 방지로 최소값 0.01)
        vmax = max(np.nanmax(np.abs(values)), 0.01)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        cmap = plt.cm.RdYlGn
        im = ax.imshow(masked, cmap=cmap, norm=norm, aspect="equal", origin="upper")

        # 벽 셀(들) 회색 사각형으로 덧그림 (마스킹만으로는 경계가 흐릴 수 있어 강조)
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

        # 각 칸 중앙에 숫자(V값) 표시
        for h in range(self.height):
            for w in range(self.width):
                if (h, w) in self.wall_states:
                    continue
                if print_value:
                    ax.text(
                        w,
                        h,
                        f"{values[h, w]:.2f}",
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=10,
                        zorder=5,
                    )

        # 정책 화살표
        # - 행동 인코딩: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        # - dirs는 "그림 좌표" 기준 방향 벡터 (x는 열, y는 행)
        # - 확률이 클수록 화살표를 길게 그림
        if policy is not None:
            dirs = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            # 전체 상태 중 최대 확률(max_p)을 먼저 구해서, 각 화살표 길이를 상대적으로 정규화
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
                        # sqrt로 완만하게 스케일 (확률 차이가 너무 과장되지 않도록)
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
        ax.set_xlim(-0.5, self.width - 0.5 + 1.5)
        ax.set_ylim(self.height - 0.5, -0.5)

        # reward_map에 있는 비영(非0) 보상은 오른쪽에 라벨로 표시
        for h in range(self.height):
            for w in range(self.width):
                r = self.reward_map[h, w]
                if r is None or r == 0:
                    continue
                x_text = self.width + 0.05
                if (h, w) == self.goal_state and r > 0:
                    ax.text(x_text, h, f"R {r} (GOAL)", va="center", fontsize=9)
                elif r < 0:
                    ax.text(x_text, h, f"R {r}", va="center", fontsize=9)

        if draw_colorbar:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="V(s)")

        if title is None:
            title = "V(s) and policy" if policy is not None else "V(s)"
        ax.set_title(title)

        # subplot에 그리는 경우 tight_layout은 외부에서 호출하는 게 안전하지만,
        # 단독 figure로 그릴 때는 여기서 한 번 정리해줌.
        if ax is None:
            plt.tight_layout()

        if show:
            plt.show()

    def render_q(self, q=None, print_value=True):
        """
        Q(s,a) 텍스트 출력.
        - q를 dict로 주면 key=((h,w), action) 형태를 기대
        - q를 ndarray로 주면 q[h,w,a]를 기대
        """
        if q is None:
            q = {}
        for h in range(self.height):
            for w in range(self.width):
                cell = (h, w)
                if cell in self.wall_states:
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
