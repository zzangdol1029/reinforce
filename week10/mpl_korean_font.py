"""
matplotlib 한글 표시 (네모□ 깨짐 방지)
====================================
기본 폰트(DejaVu Sans 등)에는 한글 글리프가 없어 제목·범례의 한글이 □ 로 보입니다.
시스템에 설치된 한글 폰트 중 첫 번째로 사용 가능한 것을 고릅니다.

macOS: Apple SD Gothic Neo, AppleGothic
Windows: Malgun Gothic
Linux: NanumGothic, Noto Sans CJK KR 등 (설치되어 있어야 함)
"""
from __future__ import annotations

from typing import Optional


def configure_korean_font() -> Optional[str]:
    """
    matplotlib 전역(rcParams)에 한글 폰트를 설정합니다.
    plt.figure 등 그리기 전에 한 번 호출하세요.

    Returns:
        선택된 폰트 패밀리 이름. 없으면 sans-serif 폴백만 적용하고 None.
    """
    import matplotlib
    import matplotlib.font_manager as fm

    # 마이너스 기호가 깨지는 경우 방지 (일부 한글 폰트)
    matplotlib.rcParams["axes.unicode_minus"] = False

    candidates = [
        "Apple SD Gothic Neo",
        "AppleGothic",
        "Malgun Gothic",
        "NanumGothic",
        "Nanum Gothic",
        "Noto Sans CJK KR",
        "Noto Sans KR",
    ]

    installed = {font.name for font in fm.fontManager.ttflist}
    for family in candidates:
        if family in installed:
            matplotlib.rcParams["font.family"] = family
            return family

    # 정확한 패밀리명이 안 맞을 때: 파일명으로 Apple Gothic 계열 검색 (macOS)
    for font in fm.fontManager.ttflist:
        name_l = font.name.lower()
        if "gothic" in name_l and ("apple" in name_l or "sd " in name_l):
            matplotlib.rcParams["font.family"] = font.name
            return font.name

    matplotlib.rcParams["font.sans-serif"] = candidates + list(
        matplotlib.rcParams["font.sans-serif"]
    )
    matplotlib.rcParams["font.family"] = "sans-serif"
    return None
