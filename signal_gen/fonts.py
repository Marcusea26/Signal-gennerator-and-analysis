"""
matplotlib 中文显示配置。

matplotlib 默认字体 DejaVu Sans 不含中文字形，会把中文渲染成方框并刷屏
``Glyph xxx missing from font(s)`` 警告。这里统一探测系统中可用的中文字体
并设为默认，从而让 CLI 与 GUI 的图表标签都能正常显示中文。
"""

from __future__ import annotations

from typing import List, Optional

__all__ = ["configure_chinese_font", "available_cjk_fonts", "CHINESE_FONT_CANDIDATES"]

#: 按优先级排列的中文字体候选
CHINESE_FONT_CANDIDATES: List[str] = [
    "Microsoft YaHei UI",
    "Microsoft YaHei",
    "SimHei",
    "DengXian",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "WenQuanYi Micro Hei",
    "PingFang SC",
    "Heiti SC",
    "Arial Unicode MS",
]


def available_cjk_fonts() -> List[str]:
    """返回系统中实际可用的中文字体名列表。"""
    try:
        import matplotlib.font_manager as fm
    except ImportError:
        return []
    installed = {f.name for f in fm.fontManager.ttflist}
    return [name for name in CHINESE_FONT_CANDIDATES if name in installed]


def configure_chinese_font(force: Optional[str] = None) -> Optional[str]:
    """把 matplotlib 默认字体设为中文字体。

    参数
    ----
    force
        指定字体名；为 ``None`` 时自动选择首个可用候选。

    返回
    ----
    实际生效的字体名；找不到任何中文字体时返回 ``None``。
    """
    import matplotlib
    import matplotlib.pyplot as plt

    chosen = force
    if chosen is None:
        candidates = available_cjk_fonts()
        chosen = candidates[0] if candidates else None

    if chosen is None:
        # 退而求其次：允许中文以方框显示，但不刷警告
        matplotlib.rcParams["axes.unicode_minus"] = False
        return None

    matplotlib.rcParams["font.sans-serif"] = [chosen] + matplotlib.rcParams["font.sans-serif"]
    matplotlib.rcParams["font.family"] = "sans-serif"
    # 中文字体通常不含 U+2212 负号，需关闭 unicode minus
    matplotlib.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.unicode_minus"] = False
    return chosen
