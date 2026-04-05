"""
matplotlib.pyplot 가 로드되기 **전**에 import 하면, 그래프가 PNG 대신 창으로 뜹니다.

mc_eval.py / mc_control.py 맨 위에서 다른 common.* 보다 먼저:
    import common.mpl_window  # noqa: F401
"""

import sys

import matplotlib

# macOS: 네이티브 창. 그 외: TkAgg(대부분 pip 설치본에 포함).
_order = ("macosx", "TkAgg") if sys.platform == "darwin" else ("TkAgg", "Qt5Agg", "QtAgg")

for _name in _order:
    try:
        matplotlib.use(_name, force=True)
        break
    except Exception:
        continue
