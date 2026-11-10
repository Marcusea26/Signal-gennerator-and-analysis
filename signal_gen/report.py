"""
外部音频文件的分析与报告图。

这个模块来自一份早期脚本（读取信号文件 → 分析 → 画图）。旧脚本用的是
``soundfile`` + **滑动窗算术平均**的 1/6 倍频程，两者都有问题：

* 滑动窗算术平均的读数比标准口径低 **36 ~ 38 dB** —— 正确口径是
  **功率积分**（IEC 61260 / ANSI S1.11），即频带总功率；
* 逐频点做 Python 循环 + ``searchsorted``，在 300 s / 48 kHz 的文件上
  实测 **23.5 s**（720 万次循环），而标准实现只要毫秒级。

所以这里不是"搬运旧代码"，而是把它的**工作流**（读文件 → 统计 → 频谱 →
1/6 倍频程 → 出图）用现有代码里已经正确的部件重新搭一遍，并且：

* 幅度标度统一为 **dBFS**（满幅正弦 = 0 dBFS），与生成信号的读数可比；
* 时域用 **min/max 包络**绘制，长文件不会因为几百万个样点糊成一片；
* 报告图是**双栏**布局（时域 + 频域），可直接贴进报告。

对外接口
--------

* :func:`analyze_file` —— 读文件并返回全部分析结果（纯数据，无绘图）；
* :func:`build_report_figure` —— 生成双栏报告图；
* :func:`report_file` —— 一步到位：读文件 → 分析 → 出图 → 存盘。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .audio_io import NON_WAV_SUFFIXES, audio_info, load_audio, signal_stats
from .spectrum import (
    DEFAULT_WINDOW,
    band_levels_dbfs,
    one_sided_spectrum_dbfs_power,
)

__all__ = [
    "analyze_file",
    "build_report_figure",
    "report_file",
    "format_band_table",
    "minmax_envelope",
    "NON_WAV_SUFFIXES",
]

#: 报告图默认的分析频段 (Hz)
DEFAULT_F_MIN = 20.0
DEFAULT_F_MAX = 24000.0

#: 报告图默认的倍频程阶数（6 = 1/6 倍频程）
DEFAULT_ORDER = 6

#: 报告图时域栏的像素列数（与 GUI 一致）
TIME_COLUMNS = 1600


def minmax_envelope(signal: np.ndarray, fs: float, ncols: int = TIME_COLUMNS):
    """把信号压成"每列 min/max"的包络。

    与 :func:`signal_gen.gui.minmax_envelope` 同源 —— 时域图有上万个像素列，
    直接画几十万个样点既慢又糊，而**隔点抽稀**在高频段必然混叠。
    这里返回 ``(t, lo, hi)``，样点数不足 ``ncols`` 时原样返回全部样点。
    """
    x = np.asarray(signal, dtype=np.float64)
    n = x.size
    if n == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    if n <= ncols:
        t = np.arange(n) / float(fs)
        return t, x, x
    pad = (-n) % ncols
    padded = np.concatenate([x, np.full(pad, np.nan)])
    cols = padded.reshape(ncols, -1)
    lo = np.nanmin(cols, axis=1)
    hi = np.nanmax(cols, axis=1)
    # 每列覆盖 n/ncols 个样点，时间轴步长必须是 (n/ncols)/fs
    t = (np.arange(ncols) + 0.5) * (n / ncols) / float(fs)
    return t, lo, hi


def analyze_file(
    path,
    window: str = DEFAULT_WINDOW,
    order: int = DEFAULT_ORDER,
    f_min: float = DEFAULT_F_MIN,
    f_max: float = DEFAULT_F_MAX,
) -> Dict[str, Any]:
    """读取一个音频文件并完成全部分析，返回纯数据（不绘图）。

    返回字典字段：

    * ``path`` / ``name``   文件路径与文件名；
    * ``signal`` / ``fs``   波形与采样率；
    * ``n_samples``         样本数；
    * ``duration``          时长 (s)；
    * ``info``              文件格式信息（采样率 / 声道 / 位深 / 编码）；
    * ``stats``             读数（峰值 / RMS / 峰值因子 / 直流偏移 / 削波样点）；
    * ``freqs`` / ``db``    单边 dBFS 幅度谱；
    * ``centers`` / ``bands`` 1/6 倍频程频带中心与频带功率 (dBFS)；
    * ``window`` / ``order`` 实际使用的窗函数与倍频程阶数。

    分析口径与生成信号**完全一致** —— 同一套窗函数、同样的 dBFS 标度、
    同一条 1/6 倍频程曲线，因此外部文件与内部生成的信号可以直接对比。
    """
    p = Path(path)
    signal, fs = load_audio(p)
    if signal.size == 0:
        raise ValueError(f"文件里没有采样数据: {p}")

    info = audio_info(p)
    stats = signal_stats(signal)

    freqs, db, power = one_sided_spectrum_dbfs_power(signal, fs, window=window)
    hi = min(float(f_max), fs / 2.0 * 0.99)
    # band_levels_dbfs 已经返回 dB 值，不要再取一次 log10
    centers, levels = band_levels_dbfs(
        freqs, power, f_min=float(f_min), f_max=hi, order=int(order)
    )

    return {
        "path": p,
        "name": p.name,
        "signal": signal,
        "fs": int(fs),
        "n_samples": int(signal.size),
        "duration": float(signal.size) / float(fs),
        "info": info,
        "stats": stats,
        "freqs": freqs,
        "db": db,
        "centers": centers,
        "bands": levels,
        "window": window,
        "order": int(order),
    }


def format_band_table(centers: np.ndarray, bands: np.ndarray) -> str:
    """把倍频程频带结果格式化成对齐的文本表格。"""
    if centers.size == 0:
        return "（分析频段内没有完整的频带）"
    lines = []
    for c, level in zip(centers, bands):
        lines.append(f"  {c:10.2f} Hz  {level:+8.2f} dBFS")
    return "\n".join(lines)


def build_report_figure(
    result: Dict[str, Any],
    figsize: Tuple[float, float] = (12.0, 8.0),
    title: Optional[str] = None,
):
    """生成双栏分析报告图（时域 + 频域），返回 ``matplotlib.figure.Figure``。

    上栏是时域波形（min/max 包络），下栏是频谱并叠加 1/6 倍频程频带功率。
    两栏共用同一套 dBFS 标度，可直接用于报告或存档。

    调用方负责 ``fig.savefig`` 与 ``plt.close``。
    """
    import matplotlib
    if matplotlib.get_backend().lower() not in ("agg", "module://matplotlib_inline.backend_inline"):
        pass  # 后端由调用方决定，这里不强行切换

    import matplotlib.pyplot as plt

    from .fonts import configure_chinese_font

    configure_chinese_font()

    signal = result["signal"]
    fs = float(result["fs"])
    freqs = result["freqs"]
    db = result["db"]
    centers = result["centers"]
    bands = result["bands"]
    stats = result["stats"]

    fig, axes = plt.subplots(2, 1, figsize=figsize)

    # ---- 上栏：时域 ----
    ax = axes[0]
    t, lo, hi = minmax_envelope(signal, fs, TIME_COLUMNS)
    if t.size < signal.size:
        ax.fill_between(t, lo, hi, color="#e07b39", lw=0.0, alpha=0.9)
    else:
        ax.plot(t, lo, lw=0.7, color="#e07b39")
    ax.set_xlim(0, result["duration"])
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("幅度")
    ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.6)

    # ---- 下栏：频域 ----
    ax = axes[1]
    ax.plot(freqs, db, lw=0.5, alpha=0.55, color="#4a7ec7", label="原始频谱")
    ax.plot(centers, bands, lw=1.6, color="#d62728", marker="o", ms=2.5,
            label=f"1/{result['order']} 倍频程（频带功率）")
    ax.set_xscale("log")
    ax.set_ylim(-140, 5)
    ax.set_xlim(DEFAULT_F_MIN, min(20000.0, fs / 2.0))
    ax.set_xlabel("频率 (Hz)")
    ax.set_ylabel("电平 (dBFS)")
    ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.6)
    ax.legend(loc="upper right", fontsize=8)

    heading = title or result.get("name", "信号分析")
    info = result["info"]
    fig.suptitle(
        f"{heading} — {info['sample_rate']} Hz / {info['channels']} ch / "
        f"{info['bits']} bit {info['encoding']} / {result['duration']:.2f} s",
        fontsize=11,
    )
    fig.text(
        0.01, 0.01,
        f"峰值 {stats['peak_dbfs']:.2f} dBFS   RMS {stats['rms_dbfs']:.2f} dBFS   "
        f"峰值因子 {stats['crest_db']:.2f} dB   直流 {stats['dc']:+.2e}",
        fontsize=8, color="#555555",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    return fig


def report_file(
    path,
    out_path,
    window: str = DEFAULT_WINDOW,
    order: int = DEFAULT_ORDER,
    f_min: float = DEFAULT_F_MIN,
    f_max: float = DEFAULT_F_MAX,
    dpi: int = 140,
) -> Tuple[Path, Dict[str, Any]]:
    """一步到位：读文件 → 分析 → 出双栏报告图 → 存盘。

    返回 ``(保存路径, 分析结果)``。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = analyze_file(path, window=window, order=order, f_min=f_min, f_max=f_max)

    out = Path(out_path)
    if not out.is_absolute():
        out = Path.cwd() / out
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = build_report_figure(result)
    try:
        fig.savefig(out, dpi=dpi)
    finally:
        plt.close(fig)
    return out, result
