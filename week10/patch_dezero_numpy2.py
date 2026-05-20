"""
DeZero 패키지에 대한 NumPy 2.x 호환 패치(np.int 제거).
import dezero 없이 site-packages 경로만 찾아 수정합니다.

같은 Python 으로 실행:
  python patch_dezero_numpy2.py
"""
from __future__ import annotations

import pathlib
import re
import sys
import importlib.util


def find_dezero_root() -> pathlib.Path | None:
    spec = importlib.util.find_spec("dezero")
    if spec is None:
        return None
    if spec.submodule_search_locations:
        return pathlib.Path(list(spec.submodule_search_locations)[0]).resolve()
    if spec.origin:
        return pathlib.Path(spec.origin).resolve().parent
    return None


def patch_text(path: pathlib.Path) -> bool:
    old = path.read_text(encoding="utf-8")
    new = re.sub(r"\bnp\.int\b", "int", old)
    if new == old:
        return False
    path.write_text(new, encoding="utf-8")
    return True


def main() -> int:
    root = find_dezero_root()
    if root is None or not root.is_dir():
        print(
            "패키지 경로를 찾을 수 없습니다. 먼저 `pip install dezero` 하였는지 확인하세요.",
            file=sys.stderr,
        )
        return 1

    print(f"dezero 경로: {root}")
    changed = 0
    for path in sorted(root.rglob("*.py")):
        if patch_text(path):
            print(f"패치: {path.relative_to(root)}")
            changed += 1

    if changed == 0:
        print("패치할 np.int 가 없거나 이미 처리되었습니다.")

    # 이제 import 검증
    try:
        import dezero  # noqa: F401
    except Exception as exc:
        print("경고: 패치 후에도 import 실패:", exc, file=sys.stderr)
        return 1

    print("import dezero OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
