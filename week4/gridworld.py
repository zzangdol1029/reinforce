"""
week4에서 week3의 GridWorld를 그대로 재사용하기 위한 래퍼.

슬라이드 코드가 `from common.gridworld import GridWorld` / `from gridworld import GridWorld`
형태를 쓰기 때문에, week4에도 같은 모듈 이름을 제공하되 실제 구현은 week3를 가져옵니다.

week3 소스는 수정하지 않습니다.
"""

import sys
from pathlib import Path
import importlib.util

_WEEK4_DIR = Path(__file__).resolve().parent
_WEEK3_DIR = _WEEK4_DIR.parent / "week3"

# week3를 경로에 넣어 week3_gridworld 모듈을 로드할 때 week3 소스를 찾게 함.
sys.path.insert(0, str(_WEEK3_DIR))
# week3가 path[0]이면 `import common.gridworld_render`가 week3/common만 가리켜
# week4의 matplotlib Q Renderer가 무시됨 → week4를 그보다 앞에 둠.
sys.path.insert(0, str(_WEEK4_DIR))

# 주의: 이 파일 이름도 gridworld.py 이므로,
# 여기서 `import gridworld`를 하면 "자기 자신"을 다시 import하려고 해서 순환 import가 발생합니다.
# 따라서 week3의 gridworld.py를 "다른 모듈 이름"으로 로드합니다.
_WEEK3_GRIDWORLD = _WEEK3_DIR / "gridworld.py"
_spec = importlib.util.spec_from_file_location("week3_gridworld", _WEEK3_GRIDWORLD)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load week3 gridworld module from: {_WEEK3_GRIDWORLD}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

GridWorld = _module.GridWorld

__all__ = ["GridWorld"]

