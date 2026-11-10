"""
标准测试信号的频谱规格与整形工具。

本模块只放**标准数据**和**纯函数**，不依赖 :mod:`signal_gen.core`，
方便被 core / 测试 / GUI 共同引用。

标准测试信号的激励信号
----------------------

**EIA RS-426-B**（ANSI/CEA-426-B，1998；亦写作 ANSI/CTA-426-B）
    标题 "Loudspeakers, Optimum Amplifier Power"，规定的是"最佳功放功率"，
    而不是"功率承载能力"。其加速寿命测试信号（Keele 测试 CD 的 Track 2）
    是**频谱加权的高斯噪声**，标准原文的措辞（见 D. B. Keele,
    *Development of Test Signals for EIA-426-B*, AES 111th Convention, 2001
    的逐条转述）是：

    * *spectral-weighted Gaussian noise, soft clipped to a crest factor of
      6 dB and lasting for 30 minutes*；
    * 加权曲线**以恒定百分比带宽分析仪读数为准**：在 **40 Hz – 1 kHz 之间
      读数平坦**，**1 kHz – 10 kHz 之间按 3 dB/octave（10 dB/decade）滚降**；
    * 限带由 **40 Hz 高通（至少 6 阶）** 与 **10 kHz 低通（至少 4 阶）**
      两个 maximally-flat Butterworth 滤波器完成；
    * 噪声的 RMS 电平比 1 kHz 校准音低 **3 dB**；因峰值因子为 6 dB，
      噪声峰值与校准音峰值相同（均为 −6 dBFS）。

    **注意：这不是"带通粉噪"**。粉噪（功率谱 ∝ 1/f）只是 40 Hz – 1 kHz
    这一段的形状；1 kHz 以上还要**再叠加一段 1/f 的功率谱倾斜**，
    总斜率变成 1/f²（读数按 3 dB/octave 下降）。详见 :func:`eia426b_spectrum`。

    标准还包含一项用 **40 Hz – 10 kHz 变速扫频**做的功率压缩测试、
    一组失真测试纯音，以及 8 小时加速寿命测试；扫频与噪声是分开的两项
    测试，本模块只实现寿命测试的噪声激励信号。

**IEC 60268-1:1985 Clause 7** —— **模拟节目信号**（simulated programme signal）：
一种**平稳的加权高斯噪声**，其 1/3 倍频程功率谱按标准 **Table II** 的
**相对电平表**分布（20 Hz – 20 kHz 共 31 个频带），并且**不做幅度限制**。
Clause 7 原文：

    *A signal, whose mean power spectral density closely resembles the average
    of the mean power spectral densities of a wide range of programme
    material, including both speech and music of several kinds, is stationary
    weighted Gaussian noise without amplitude limiting, the weighted power
    spectrum being in accordance with Table II and Figure 1, page 22, when
    measured with third-octave filters in accordance with IEC Publication 225.*

    Table II 给的是**相对**频谱：表里的 "0 dB" 不是信号必须达到的电平，
    而是各频带相互参照的基准；每一行还印着容差（±0.5 ~ ±3 dB）。因此判据
    不是"某个频带必须等于某个绝对值"，而是"**存在一个统一电平，能把全部
    频带都放进各自的容差窗**"（见 :func:`check_iec268_program`）。

    Clause 7 还给出**一种实现途径**：*Such a signal may be obtained from a
    pink-noise source by means of the filter circuit shown in Figure 2.* ——
    一条无源 RC 梯形网络（粉噪经 430 Ω + 2.2 µF → 3.3 kΩ‖91 nF →
    330 Ω + 2.2 µF → 3.3 kΩ‖68 nF → 0.47 µF → 10 kΩ）。
    **Figure 2 是"一种实现"，Table II 才是定义。** 本项目两种都提供：
    ``iec268``（Table II 逐带精确）与 ``iec268_fig2``（Figure 2 电路）。

**IEC 60268-1 附录 C / IEC 60065** 的**带限粉噪滤波器**：在 22.4 Hz – 22.4 kHz
    之间基本平坦，典型实现为 **12 dB/oct 高通（-3 dB @ 22.7 Hz）** 加上
    **18 dB/oct 低通（-3 dB @ 22.4 kHz）**。

    注意：这**不是** Clause 7 的节目信号 —— 它只规定滤波器的频响，不规定
    频谱形状，更不规定绝对电平。两者常被混为一谈。

    **IEC 60268-5**（旧称 IEC 268-5）规定用该滤波后的粉噪、峰值因子 2:1
    来做扬声器的功率承载测试（即所谓"clipped pink noise"）。
    本项目里对应 ``iec268_pink``。

    频带边界按 ANSI S1.11-2004 / IEC 61260-1 的**标准 1/3 倍频程滤波带**
    划分（精确中心 × 10^(±0.05)），相邻频带严丝合缝。

峰值因子（Crest Factor）
---------------------
    **峰值因子 = 峰值与 RMS 之比的 dB 值**（``20·log10(peak / rms)``），
    衡量波形"多尖"的程度。高斯噪声在长时间记录上的峰值因子高达 12 ~ 16 dB，
    **必须靠削波压下来** —— 这也是业界"clipped pink noise"生成器的做法。

    两个标准都要求激励信号具有确定的峰值因子（AES2 用 6 dB，IEC 60268-5 用 2:1）。

    但 **IEC 60268-1 Clause 7 恰好相反**：它明确要求节目信号
    *"without amplitude limiting"*（**不做幅度限制**），所以
    :data:`IEC268_CREST_DB` = **0**（不限幅）。需要限幅的场合
    （例如 IEC 60268-7:2010 8.3.2 规定耳机的节目信号 *"with additional
    clipping"*、peak-to-r.m.s. 1.8 ~ 2.2 ≈ 峰值因子 5.1 ~ 6.9 dB）
    由用户通过 ``crest_factor_db`` 参数指定。

    * :func:`limit_crest_factor` 用二分法求出**硬**削波阈值，使实际信号的
      峰值/RMS 精确等于目标值（IEC 60268-5 用这一种）；
    * :func:`soft_clip_to_crest` 用 Keele 论文 Eq.(1) 的**软**削波函数
      （模拟二极管全波整流的圆拐点）把峰值因子压到 6 dB —— 这是
      EIA RS-426-B 明确要求的做法（*soft clipped to a crest factor of 6 dB*）。
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "IEC268_HP_FC",
    "IEC268_HP_ORDER",
    "IEC268_HP_ROLLOFF_DB_OCT",
    "IEC268_LP_FC",
    "IEC268_LP_ORDER",
    "IEC268_LP_ROLLOFF_DB_OCT",
    "IEC268_FLAT_LO",
    "IEC268_FLAT_HI",
    "IEC268_PROGRAM_TABLE",
    "IEC268_PROGRAM_CENTERS",
    "IEC268_PROGRAM_LEVELS",
    "IEC268_PROGRAM_TOLERANCES",
    "IEC268_PROGRAM_EXACT_CENTERS",
    "IEC268_PROGRAM_TOTAL_POWER_DB",
    "IEC268ProgramCheck",
    "NYQUIST_MARGIN",
    "EIA426B_HP_FC",
    "EIA426B_HP_ORDER",
    "EIA426B_HP_ROLLOFF_DB_OCT",
    "EIA426B_LP_FC",
    "EIA426B_LP_ORDER",
    "EIA426B_LP_ROLLOFF_DB_OCT",
    "EIA426B_CAL_TONE_PEAK_DBFS",
    "EIA426B_RMS_DBFS",
    "EIA426B_WEIGHT_KNEE_HZ",
    "EIA426B_SOFT_CLIP_K",
    "EIA426B_CREST_DB",
    "IEC268_CREST_DB",
    "IEC268_PINK_CREST_DB",
    "STANDARD_CREST_DB",
    "butterworth_magnitude",
    "third_octave_exact_center",
    "third_octave_edges",
    "third_octave_band_power",
    "check_iec268_program",
    "eia426b_spectrum",
    "iec268_figure2_response",
    "iec268_lp_corner",
    "iec268_program_density_db",
    "iec268_program_nodes_db",
    "iec268_program_psd",
    "limit_crest_factor",
    "soft_clip",
    "soft_clip_to_crest",
    "crest_factor_db",
]


# ---------------------------------------------------------------------------
# IEC 60268-1 附录 C / IEC 60065 —— 带限粉噪滤波器
#
# 注意：这是**功率承载测试**用的滤波器（"clipped pink noise"），
# **不是** Clause 7 的模拟节目信号 —— 后者见下面的 Table II 数据块。
# ---------------------------------------------------------------------------

#: 高通 -3 dB 频率 (Hz)。标准要求滤波器在 22.4 Hz 以上基本平坦。
IEC268_FLAT_LO = 22.4
#: 低通 -3 dB 频率 (Hz)。标准要求滤波器在 22.4 kHz 以下基本平坦。
IEC268_FLAT_HI = 22400.0

#: 高通截止频率 (Hz)
IEC268_HP_FC = 22.7

#: 高通滚降斜率 (dB/oct)：IEC 60268-1 规定为 12 dB/oct。
IEC268_HP_ROLLOFF_DB_OCT = 12.0
#: 低通滚降斜率 (dB/oct)：IEC 60268-1 规定为 18 dB/oct。
IEC268_LP_ROLLOFF_DB_OCT = 18.0

#: Butterworth 阶数 N（= 极点数）。
#:
#: 注意别被"阶数"这个词绕进去：:func:`butterworth_magnitude` 用的是
#: ``|H| = 1 / sqrt(1 + (f/fc)^(2N))``，这个 N 是**极点数**，其渐近滚降为
#: ``6·N dB/oct``（N=2 → 12 dB/oct，N=3 → 18 dB/oct）。
#:
#: 早前的实现误把"12 dB/oct"当成"6 阶"，于是高通实际是 36 dB/oct、
#: 低通实际是 54 dB/oct —— 比标准陡 3 倍，频带外的残留能量远低于标准，
#: 生成的信号自然"不对"。这里改成由斜率直接推导阶数，避免再写错。
IEC268_HP_ORDER = int(round(IEC268_HP_ROLLOFF_DB_OCT / 6.0))
#: 低通截止频率 (Hz)。
#:
#: IEC 60268-1（源自 IEC 60065 Annex C）的带通滤波器要求通带覆盖
#: **22.4 Hz – 22.4 kHz**，高通 12 dB/oct、低通 18 dB/oct。
#:
#: 关键在于低通的 -3 dB 点必须落在通带**上沿** 22.4 kHz，而不是更低：
#: 早前把 -3 dB 点放在 19.6 kHz，22.4 kHz 处就已经衰减 10.9 dB，
#: 通带根本到不了 22.4 kHz —— 注释写着"22.4 kHz 以下平坦"而实际拐点
#: 却是 19.6 kHz。
#:
#: 业界公开的工程实现（Rod Elliott / ESP 的 IEC 60065 噪声发生器）取
#: 24.1 kHz，理由是"比规范所指示的略高，但这样整条响应才落在标准给出
#: 的容差窗内"（ESP 原话：a little higher than the specification
#: indicates, but it fits inside the 'window' shown）。本项目取标准的
#: 22.4 kHz 作为名义值，是更保守（通带更窄）的一端。
IEC268_LP_FC = IEC268_FLAT_HI
IEC268_LP_ORDER = int(round(IEC268_LP_ROLLOFF_DB_OCT / 6.0))

#: 奈奎斯特频率的保留比例：整形内容最高只做到 ``0.99·fs/2``，
#: 留一点余量避免混叠。与 :mod:`signal_gen.core` 里的 ``_NYQUIST_MARGIN`` 一致。
NYQUIST_MARGIN = 0.99


# ---------------------------------------------------------------------------
# IEC 60268-1:1985 Clause 7 —— 模拟节目信号（simulated programme signal）
# ---------------------------------------------------------------------------

#: Table II（IEC 60268-1:1985，PDF 第 23 页 / 印刷第 21 页）
#: "Power spectrum of simulated programme signal"：
#: **名义** 1/3 倍频程中心频率 (Hz) → (相对电平 dB, 容差 dB)。
#:
#: 表里的 "+/−" 是分列印的，但**每一行的上下容差相等**，所以这里只存一个值。
#:
#: 这是一张**相对**频谱表：0 dB 不是信号必须达到的电平，而是各频带相互
#: 参照的基准；判据是"存在一个统一电平，能把全部频带都放进各自的容差窗"。
IEC268_PROGRAM_TABLE: Dict[float, Tuple[float, float]] = {
    20.0: (-13.5, 3.0),
    25.0: (-10.2, 2.0),
    31.5: (-7.4, 1.0),
    40.0: (-5.2, 1.0),
    50.0: (-3.5, 1.0),
    63.0: (-2.3, 1.0),
    80.0: (-1.4, 1.0),
    100.0: (-0.9, 0.8),
    125.0: (-0.5, 0.6),
    160.0: (-0.2, 0.5),
    200.0: (-0.1, 0.5),
    250.0: (0.0, 0.5),
    315.0: (0.0, 0.5),
    400.0: (0.0, 0.5),
    500.0: (0.0, 0.5),
    630.0: (0.0, 0.5),
    800.0: (0.0, 0.5),
    1000.0: (-0.1, 0.6),
    1250.0: (-0.3, 0.7),
    1600.0: (-0.6, 0.8),
    2000.0: (-1.0, 1.0),
    2500.0: (-1.6, 1.0),
    3150.0: (-2.5, 1.0),
    4000.0: (-3.7, 1.0),
    5000.0: (-5.1, 1.0),
    6300.0: (-7.0, 1.0),
    8000.0: (-9.4, 1.0),
    10000.0: (-11.9, 1.0),
    12500.0: (-14.8, 1.5),
    16000.0: (-18.2, 2.0),
    20000.0: (-21.6, 3.0),
}

#: Table II 的名义 1/3 倍频程中心频率 (Hz)，升序。
IEC268_PROGRAM_CENTERS: Tuple[float, ...] = tuple(IEC268_PROGRAM_TABLE.keys())
#: Table II 的相对电平 (dB)。
IEC268_PROGRAM_LEVELS: Tuple[float, ...] = tuple(
    level for level, _tol in IEC268_PROGRAM_TABLE.values()
)
#: Table II 的容差 (dB)（上下相等）。
IEC268_PROGRAM_TOLERANCES: Tuple[float, ...] = tuple(
    tol for _level, tol in IEC268_PROGRAM_TABLE.values()
)

#: Table II 各频带的**精确** 1/3 倍频程中心频率 (Hz)：``1000·10^(n/10)``。
#:
#: 名义值（20、25、31.5 …）是表上的标签；真正定义滤波带的是这组精确中心
#: （ANSI S1.11-2004 / IEC 61260-1），频带边界 = 精确中心 × 10^(±0.05)。
#: 两者一一对应、顺序一致（20 Hz 名义 ↔ 19.95 Hz 精确）。
IEC268_PROGRAM_EXACT_CENTERS: Tuple[float, ...] = tuple(
    1000.0 * 10.0 ** (n / 10.0) for n in range(-17, 14)
)

#: 31 个相对电平的**功率和** (dB)：``10·log10(Σ 10^(T_i/10))`` = 12.56 dB。
#:
#: Clause 7 的注（Note）说："整个信号的功率约比一个频带的 0 dB 高 12.5 dB"，
#: 正是这个数 —— 说明 Table II 是**相对**频谱，绝对电平由使用场合定。
IEC268_PROGRAM_TOTAL_POWER_DB = float(
    10.0 * np.log10(sum(10.0 ** (lvl / 10.0) for lvl in IEC268_PROGRAM_LEVELS))
)


# ---------------------------------------------------------------------------
# EIA RS-426-B —— 加速寿命测试噪声（Keele 测试 CD 的 Track 2）
# ---------------------------------------------------------------------------

#: 高通 -3 dB 频率 (Hz)。标准规定噪声限带到 **40 Hz**。
EIA426B_HP_FC = 40.0
#: 低通 -3 dB 频率 (Hz)。标准规定噪声限带到 **10 kHz**。
EIA426B_LP_FC = 10000.0

#: 高通阶数（极点数）：标准要求 **至少 6 阶** maximally-flat Butterworth。
#:
#: 注意与 AES2（"12 dB/oct 的 Butterworth，高低通各一个"）的说法不同 ——
#: 那是老版 AES2 对"最佳功放功率"测试的概括；EIA RS-426-B 自己的措辞是
#: *a 40-Hz high-pass filter of at least 6th-order minimum and a 10-kHz
#: low-pass filter of at least 4th-order minimum*，Keele 的实现两者都取 6 阶。
EIA426B_HP_ORDER = 6
#: 低通阶数（极点数）：标准要求**至少 4 阶**；Keele 的实现用 6 阶。
EIA426B_LP_ORDER = 6

#: 高通 / 低通的渐近滚降斜率 (dB/oct)：由极点数决定（6·N dB/oct）。
EIA426B_HP_ROLLOFF_DB_OCT = 6.0 * EIA426B_HP_ORDER
EIA426B_LP_ROLLOFF_DB_OCT = 6.0 * EIA426B_LP_ORDER

#: 加权谱的**拐点** (Hz)：1 kHz 以上在粉噪斜率之上再叠加一段 1/f 倾斜。
#:
#: 标准措辞：*is flat between 40 Hz and 1 kHz, and then rolls off at
#: 3 dB/octave (10 dB/decade) between 1 kHz and 10 kHz*（读数取自恒定
#: 百分比带宽分析仪）。平坦段 ⇒ 功率谱密度 ∝ 1/f；滚降段 ⇒ ∝ 1/f²。
EIA426B_WEIGHT_KNEE_HZ = 1000.0

#: 软削波的 **rounding 参数 k**（Keele Eq.(1)）。
#:
#: 论文原文：*k = 1 quite rounded, k = 4 typical rounding, k = 100 hard clip*，
#: Keele 生成 CD 用的就是 **k = 4**。
EIA426B_SOFT_CLIP_K = 4.0

#: 标准要求的峰值因子 (dB)：*soft clipped to a crest factor of 6 dB*。
EIA426B_CREST_DB = 6.0

#: 1 kHz 校准音的**峰值**电平 (dBFS)：−6 dBFS（对应 16 bit 满刻度 ±16384）。
#:
#: 标准规定寿命测试噪声的 RMS 比校准音低 3 dB；又因噪声峰值因子为 6 dB，
#: 所以**噪声峰值与校准音峰值相同**（都是 −6 dBFS），噪声 RMS 为 −12 dBFS。
EIA426B_CAL_TONE_PEAK_DBFS = -6.0
#: 寿命测试噪声的 RMS 电平 (dBFS)：比校准音低 3 dB。
EIA426B_RMS_DBFS = EIA426B_CAL_TONE_PEAK_DBFS - 3.0

#: IEC 60268-1 Clause 7 模拟节目信号的峰值因子 (dB)：**0 = 不限幅**。
#:
#: Clause 7 明确要求 *"stationary weighted Gaussian noise without amplitude
#: limiting"* —— 平稳加权高斯噪声、**不做幅度限制**。所以标准信号本身
#: 不规定峰值因子；长记录上的高斯噪声天然就有 12 ~ 15 dB。
#:
#: 需要限幅时（例如 IEC 60268-7:2010 8.3.2 规定耳机的节目信号
#: *"with additional clipping"*、peak-to-r.m.s. 1.8 ~ 2.2 ≈ 峰值因子
#: 5.1 ~ 6.9 dB），由用户通过 ``crest_factor_db`` 参数指定。
IEC268_CREST_DB = 0.0

#: IEC 60268-5 / IEC 60065 附录 C "clipped pink noise" 功率测试要求的
#: 峰值因子 (dB)：2:1
IEC268_PINK_CREST_DB = 6.0

#: 各标准信号**默认**的峰值因子 (dB)；0 表示不限幅。
#:
#: 这张表是"标准峰值因子"的唯一数据源 —— :data:`signal_gen.core.DEFAULT_PARAMS`
#: 里的 ``crest_factor_db`` 直接引用这里的常量，不会各写一份而漂移。
STANDARD_CREST_DB: Dict[str, float] = {
    "eia426b": EIA426B_CREST_DB,
    "iec268": IEC268_CREST_DB,
    "iec268_fig2": IEC268_CREST_DB,
    "iec268_pink": IEC268_PINK_CREST_DB,
}


# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------

def third_octave_exact_center(nominal: float) -> float:
    """名义 1/3 倍频程中心频率 → **精确**中心频率 ``1000·10^(n/10)``。

    标准表上写的是"名义"值（20、25、31.5、40 …，按 ×1.25 取整），
    而 IEC 61260 / ANSI S1.11 另给了一组"精确"中心频率（按 ×10^0.1 =
    1.25893 递增）。两者相差最多 **1 %**（例如名义 160 Hz 对应精确
    158.49 Hz）。

    关键：1/3 倍频程频带的**边界**就是按这组精确中心定义的
    （见 :func:`third_octave_edges`），因此标准表上的名义值只用来
    "指认"是哪个频带，真正决定滤波器通带的是精确中心。
    """
    n = round(math.log10(nominal / 1000.0) * 10.0)
    return 1000.0 * 10.0 ** (n / 10.0)


# ---------------------------------------------------------------------------
# 频域整形工具
# ---------------------------------------------------------------------------

def butterworth_magnitude(
    freqs: np.ndarray,
    fc: float,
    order: int,
    btype: str = "low",
) -> np.ndarray:
    """Butterworth 滤波器的幅度响应（解析式，**不产生任何滤波过程**）。

    直接在频域整形的好处：没有滤波器的启动瞬态，响应严格等于解析式，
    而且只做一次 FFT，比"生成噪声 → 卷积滤波"快得多。

    幅频特性::

        |H_LP(f)| = 1 / sqrt(1 + (f/fc)^(2N))
        |H_HP(f)| = (f/fc)^N / sqrt(1 + (f/fc)^(2N))

    参数
    ----
    freqs
        频率轴 (Hz)，非负。
    fc
        -3 dB 截止频率 (Hz)，必须为正。
    order
        极点数 N。这个解析式下 N 同时决定渐近滚降斜率 **6·N dB/oct**：

        ======  ==============  ==========================
        N       滚降 (dB/oct)   用途
        ======  ==============  ==========================
        1        6               一阶 RC
        2       12               IEC 60268-1 高通 / AES2 高低通
        3       18               IEC 60268-1 低通
        ======  ==============  ==========================

        换算关系：``N = round(rolloff_db_per_oct / 6)``。
        常见误解是把"12 dB/oct"当成"6 阶"—— 那样实际会得到 36 dB/oct。
    btype
        ``"low"`` 或 ``"high"``。
    """
    f = np.asarray(freqs, dtype=np.float64)
    if fc <= 0.0:
        raise ValueError(f"截止频率必须为正，得到 {fc!r}")
    if order < 1:
        raise ValueError(f"滤波器阶数必须 ≥ 1，得到 {order!r}")

    x = f / float(fc)
    denom = np.sqrt(1.0 + x ** (2 * order))
    if btype == "low":
        return 1.0 / denom
    if btype == "high":
        return (x ** order) / denom
    raise ValueError(f"未知的滤波器类型: {btype!r}（可选 'low' / 'high'）")


def iec268_lp_corner(fs: float) -> float:
    """IEC 60268-1 低通滤波器的**实际** -3 dB 拐点 (Hz)。

    标准给出的拐点是 22.4 kHz（通带上沿）；但采样率不足时拐点必须跟着
    奈奎斯特频率走，否则整形目标落在可还原频率范围之外，低通等于没做
    （且会混叠）。

    * fs = 48 kHz → 22.4 kHz（标准值）
    * fs = 16 kHz → 7.92 kHz（``0.99·fs/2``）
    """
    return min(IEC268_LP_FC, NYQUIST_MARGIN * float(fs) / 2.0)


# ---------------------------------------------------------------------------
# IEC 60268-1 Clause 7 —— 模拟节目信号的频谱整形
# ---------------------------------------------------------------------------

#: dB → 指数的换算因子 ``ln10 / 10``（即 10^(dB/10) = e^(k·dB)）。
_DB_TO_EXPONENT = math.log(10.0) / 10.0

#: 小于该指数时改用 ``(e^z - 1)/z`` 的级数展开，避免除零。
_SMALL_EXPONENT = 1e-12

#: 节点拟合的最大迭代次数与收敛容差 (dB)。
_FIT_ITERATIONS = 100
_FIT_TOLERANCE_DB = 1e-10

#: 一个精确频带中心与名义标称值之间允许的最大偏差（比值）：
#: 半个 1/12 倍频程 —— 相邻的 1/3 倍频程中心离得远得多，不会认错。
_BAND_MATCH_RATIO = 2.0 ** (1.0 / 24.0)


def _relative_expm1(z: np.ndarray) -> np.ndarray:
    """``(e^z - 1) / z``，在 ``z → 0`` 时趋于 1（避免 0/0）。"""
    small = np.abs(z) < _SMALL_EXPONENT
    safe = np.where(small, 1.0, z)
    return np.asarray(np.where(small, 1.0, np.expm1(safe) / safe), dtype=np.float64)


def _iec268_band_levels_of_nodes(nodes_db: np.ndarray) -> np.ndarray:
    """"密度在 lg f 上分段线性"时，各 1/3 倍频程频带的**闭式积分**电平。

    节点落在各频带的中心（相邻节点相隔 0.1 个十倍频程），每个频带跨过
    节点两侧各 0.05 个十倍频程。于是频带平均功率是指数函数在两段上的
    闭式积分：

    * 左半段 ``∫_{-h}^{0} e^{k·m·x} dx = h·(1 - e^{-k·m·h}) / (k·m)``
    * 右半段 ``∫_{0}^{h} e^{k·m·x} dx = h·(e^{k·m·h} - 1) / (k·m)``

    再除以频带宽度 ``2h`` 取平均，即得频带电平（dB）。
    """
    half = 0.05
    slopes = np.diff(nodes_db) / (2.0 * half)
    left = np.concatenate(([slopes[0]], slopes))
    right = np.concatenate((slopes, [slopes[-1]]))
    falling = np.exp(-_DB_TO_EXPONENT * left * half) * _relative_expm1(
        _DB_TO_EXPONENT * left * half
    )
    rising = _relative_expm1(_DB_TO_EXPONENT * right * half)
    return np.asarray(
        nodes_db + 10.0 * np.log10(0.5 * (falling + rising)), dtype=np.float64
    )


@functools.lru_cache(maxsize=1)
def iec268_program_nodes_db() -> Tuple[float, ...]:
    """Table II 的**节点电平** (dB/decade)：让每个频带的闭式积分正好等于表值。

    节点是"功率谱密度在 lg f 上的 dB 值"（即每十倍频程的功率），落各
    频带的精确中心。表里印的是**频带电平**，两者差一个由两侧斜率决定的
    平均因子；这个差值随斜率变化，所以用不动点迭代求解：

        ``nodes ← nodes + (表值 - 由 nodes 算出的频带电平)``

    迭代到残差 < 1e-10 dB 即收敛（实际约 10 次迭代）。
    """
    target = np.asarray(IEC268_PROGRAM_LEVELS, dtype=np.float64)
    nodes = target.copy()
    for _ in range(_FIT_ITERATIONS):
        error = target - _iec268_band_levels_of_nodes(nodes)
        nodes = nodes + error
        if float(np.max(np.abs(error))) < _FIT_TOLERANCE_DB:
            break
    return tuple(float(value) for value in nodes)


def iec268_program_density_db(freqs: np.ndarray) -> np.ndarray:
    """Clause 7 节目信号的**功率谱密度**（dB/decade，即"每十倍频程的功率"）。

    密度在 lg f 上是分段直线：节点落在 31 个频带的精确中心上，
    20 Hz 以下与 20 kHz 以上按**端段斜率**继续延伸（标准 Figure 1 的形状）。

    参数
    ----
    freqs
        频率轴 (Hz)，非负。

    返回
    ----
    与输入等长的密度 (dB/decade)。``f = 0`` 处按端段斜率外延（有限值）。

    说明：密度 0 dB/decade 意味着"每十倍频程的功率相同"，即功率谱密度
    ∝ 1/f —— 正是 1/3 倍频程读数平坦的粉噪。Table II 中段（250 Hz – 800 Hz）
    的 0 dB 就是这个意思。
    """
    f = np.asarray(freqs, dtype=np.float64)
    nodes = np.asarray(iec268_program_nodes_db(), dtype=np.float64)
    centres = np.log10(np.asarray(IEC268_PROGRAM_EXACT_CENTERS, dtype=np.float64))
    u = np.log10(np.maximum(f, 1e-30))
    inner = np.interp(u, centres, nodes)
    low_slope = (nodes[1] - nodes[0]) / 0.1
    high_slope = (nodes[-1] - nodes[-2]) / 0.1
    below = nodes[0] + low_slope * (u - centres[0])
    above = nodes[-1] + high_slope * (u - centres[-1])
    return np.asarray(
        np.where(u < centres[0], below, np.where(u > centres[-1], above, inner)),
        dtype=np.float64,
    )


#: 频带功率 (dB) 与"每十倍频程功率密度"之间的换算常数 (dB)。
#:
#: 1/3 倍频程频带在 lg f 上的宽度恰为 **0.1 个十倍频程**（IEC 61260 的
#: 精确中心 × 10^(±0.05)），于是
#:
#:     P_band = ∫ S(f) df = ln10 · 0.1 · 10^(D/10) · ⟨10^(m·u/10)⟩
#:
#: 即 ``10·log10(P_band) = D + 10·log10(⟨…⟩) - 6.378``。把这个常数补上，
#: 整形出来的频谱**积分后的频带功率 (dB) 就正好等于 Table II 的表值**。
_BAND_POWER_OFFSET_DB = 10.0 * math.log10(math.log(10.0) * 0.1)


def iec268_program_psd(freqs: np.ndarray) -> np.ndarray:
    """Clause 7 节目信号的**功率谱密度**（总功率/Hz）。

    ``S(f) = 10^(D(f)/10) / f``，其中 ``D(f)`` 是 :func:`iec268_program_density_db`
    给出的每十倍频程功率 (dB)。除以 ``f`` 是"每十倍频程的功率"换成"每 Hz 的
    功率"；再乘上 :data:`_BAND_POWER_OFFSET_DB` 的修正，使**积分后的频带
    功率 (dB) 直接等于 Table II 的表值**（而不是差一个常数）。

    ``f = 0`` 处密度为 0（去直流）。

    说明：这是**频谱形状**的绝对基准。生成器按 ``sqrt(S(f))`` 给定频域幅度
    （与本项目其它标准信号一致），所以合成信号的实际频谱与本函数只差一个
    与频率无关的常数因子 —— 绝对电平由随后的归一化步骤决定，**频带之间的
    相对关系**（也就是标准要考核的东西）完全一致。
    """
    f = np.asarray(freqs, dtype=np.float64)
    psd = np.zeros_like(f)
    nz = f > 0.0
    psd[nz] = 10.0 ** (
        (iec268_program_density_db(f[nz]) - _BAND_POWER_OFFSET_DB) / 10.0
    ) / f[nz]
    return psd


#: Figure 2 无源 RC 梯形网络的元件值（IEC 60268-1:1985，PDF 第 24 页 /
#: 印刷第 22 页）。信号依次经过这些元件：
#: 源阻抗 430 Ω → 串联 2.2 µF → 节点 A（3.3 kΩ 并联 91 nF）→
#: 串联 330 Ω + 2.2 µF → 节点 B（3.3 kΩ 并联 68 nF）→
#: 串联 0.47 µF → 10 kΩ 输出电阻（负载 ≥100 kΩ，10 kΩ 即整个分流）。
_FIG2_R_SOURCE_OHM = 430.0
_FIG2_C_FIRST_F = 2.2e-6
_FIG2_R_SHUNT_A_OHM = 3300.0
_FIG2_C_SHUNT_A_F = 91e-9
_FIG2_R_SERIES_B_OHM = 330.0
_FIG2_C_SECOND_F = 2.2e-6
_FIG2_R_SHUNT_B_OHM = 3300.0
_FIG2_C_SHUNT_B_F = 68e-9
_FIG2_C_OUTPUT_F = 0.47e-6
_FIG2_R_OUTPUT_OHM = 10e3


def iec268_figure2_response(freqs: np.ndarray) -> np.ndarray:
    """Figure 2 无源 RC 梯形网络的**复数传递函数**（10 kΩ 上的电压 / 源电动势）。

    由三个节点的节点方程闭式解出（消去 A、B 两点）：

    * 节点 A：``(g1 + y_a + g2)·v_a - g2·v_b = g1·e``
    * 节点 B：``-g2·v_a + (g2 + y_b + g3)·v_b - g3·v_c = 0``
    * 节点 C：``-g3·v_b + (g3 + y_c)·v_c = 0``

    其中 ``g = 1/Z`` 是串联支路的导纳、``y = 1/R + sC`` 是并联支路的导纳，
    ``e`` 是源的电动势（源自身的输出阻抗算在第一只 430 Ω 里）。

    两级 2.2 µF 电容与后面的电阻构成低频滚降，91 nF / 68 nF 并联电容构成
    高频滚降，**通带损耗约 4 dB**。

    参数
    ----
    freqs
        频率轴 (Hz)，**必须为正**（0 Hz 处电路是直流通路，传递函数无定义）。

    返回
    ----
    与输入等长的复数响应（无量纲）。
    """
    f = np.asarray(freqs, dtype=np.float64)
    if f.size and not bool(np.all(np.isfinite(f) & (f > 0.0))):
        raise ValueError("Figure 2 的传递函数只在正频率处有定义")
    s = 2j * np.pi * f
    z_first = _FIG2_R_SOURCE_OHM + 1.0 / (s * _FIG2_C_FIRST_F)
    z_second = _FIG2_R_SERIES_B_OHM + 1.0 / (s * _FIG2_C_SECOND_F)
    z_output = 1.0 / (s * _FIG2_C_OUTPUT_F)
    y_a = 1.0 / _FIG2_R_SHUNT_A_OHM + s * _FIG2_C_SHUNT_A_F
    y_b = 1.0 / _FIG2_R_SHUNT_B_OHM + s * _FIG2_C_SHUNT_B_F
    y_c = 1.0 / _FIG2_R_OUTPUT_OHM
    g1 = 1.0 / z_first
    g2 = 1.0 / z_second
    g3 = 1.0 / z_output
    a11 = g1 + y_a + g2
    a22 = g2 + y_b + g3
    a33 = g3 + y_c
    determinant = a11 * (a22 * a33 - g3 * g3) - g2 * g2 * a33
    return np.asarray(g1 * g2 * g3 / determinant, dtype=np.complex128)


@dataclass(frozen=True)
class IEC268ProgramCheck:
    """按 Table II 判定一个 1/3 倍频程频谱是否符合 Clause 7。

    Table II 是**相对**频谱，所以判定不是"某个频带必须等于某个绝对电平"，
    而是"**存在一个统一电平，能把全部频带都放进各自的容差窗**"：

    * 每个频带允许的电平窗是 ``[T_i - t_i - L, T_i + t_i - L]``，
      其中 ``T_i`` 是表值、``t_i`` 是容差、``L`` 是待求的统一偏移；
    * 所有窗**有公共交集** ⇒ 符合；
    * 报告的偏移取交集的中点（最小裕量最大的位置）。

    属性
    ----
    frequencies
        参与判定的频带中心 (Hz)，已是 Table II 的**名义**值，升序。
    band_levels
        对应的测量电平 (dB)，任意绝对基准。
    relative_levels
        Table II 的相对电平 (dB)。
    lower_tolerance_db / upper_tolerance_db
        Table II 的下 / 上容差 (dB)（本表上下相等）。
    """

    frequencies: np.ndarray
    band_levels: np.ndarray
    relative_levels: np.ndarray
    lower_tolerance_db: np.ndarray
    upper_tolerance_db: np.ndarray

    @property
    def offset_db(self) -> float:
        """把测量电平对到 Table II 上所需的统一偏移 (dB)。"""
        table = self.relative_levels
        levels = self.band_levels
        lowest = float(np.max(table - self.lower_tolerance_db - levels))
        highest = float(np.min(table + self.upper_tolerance_db - levels))
        return 0.5 * (lowest + highest)

    @property
    def deviations_db(self) -> np.ndarray:
        """每个频带对到 Table II 之后的偏差 (dB)。"""
        return np.asarray(
            self.band_levels + self.offset_db - self.relative_levels, dtype=np.float64
        )

    @property
    def margin_db(self) -> float:
        """最小裕量 (dB)：每个频带到自己较近的那条容差限的距离。

        为负表示**没有**任何统一电平能把所有频带都装进容差窗 —— 此时其
        绝对值就是最差频带的越界量。
        """
        deviation = self.deviations_db
        return float(np.min(np.minimum(
            self.upper_tolerance_db - deviation,
            self.lower_tolerance_db + deviation,
        )))

    @property
    def binding_bands_hz(self) -> Tuple[float, ...]:
        """裕量最小的频带 (Hz) —— 决定判定结果的那几个。"""
        deviation = self.deviations_db
        margins = np.minimum(
            self.upper_tolerance_db - deviation,
            self.lower_tolerance_db + deviation,
        )
        binding = margins <= float(np.min(margins)) + 1e-9
        return tuple(float(value) for value in self.frequencies[binding])

    @property
    def passes(self) -> bool:
        """是否符合 Table II 的容差。"""
        return self.margin_db >= -1e-9


def _iec268_nominal_band(frequency: float) -> float:
    """把（名义或精确的）频带中心映射到 Table II 的**名义**频带。"""
    for nominal in IEC268_PROGRAM_TABLE:
        ratio = frequency / nominal
        if 1.0 / _BAND_MATCH_RATIO < ratio < _BAND_MATCH_RATIO:
            return nominal
    raise ValueError(
        f"{frequency:g} Hz 不是 IEC 60268-1 Table II 的 1/3 倍频程频带"
        f"（20 Hz – 20 kHz）"
    )


def check_iec268_program(
    frequencies: Sequence[float],
    band_levels: Sequence[float],
) -> IEC268ProgramCheck:
    """按 IEC 60268-1:1985 Table II 判定一个 1/3 倍频程频谱。

    Clause 7 要求加权功率谱 *"in accordance with Table II [...] when measured
    with third-octave filters"*。Table II 是相对频谱，所以判据是
    "**存在一个统一电平，能把全部频带都放进各自的容差窗**"。

    只判定**给出的**频带 —— 采样率不足以容纳某个频带时（例如 44.1 kHz
    装不下 20 kHz 带）那个频带就不参与判定。

    参数
    ----
    frequencies
        频带中心 (Hz)，可以用名义值（20、25、31.5 …）或精确值
        （19.95、25.12 …），不能重复。
    band_levels
        对应的频带电平 (dB)，任意绝对基准都行（判定会自动求出统一偏移）。

    返回
    ----
    :class:`IEC268ProgramCheck`。
    """
    f = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    lv = np.atleast_1d(np.asarray(band_levels, dtype=np.float64))
    if f.ndim != 1 or lv.shape != f.shape:
        raise ValueError("'frequencies' 与 'band_levels' 必须是等长的一维数组")
    if f.size < 2:
        raise ValueError("判定一张相对频谱至少需要两个频带")
    if not bool(np.all(np.isfinite(f) & (f > 0.0))):
        raise ValueError("'frequencies' 必须是正的有限数")
    if not bool(np.all(np.isfinite(lv))):
        raise ValueError("'band_levels' 必须是有限数")

    nominal = np.array([_iec268_nominal_band(float(value)) for value in f])
    if np.unique(nominal).size != nominal.size:
        raise ValueError("'frequencies' 里出现了重复的 Table II 频带")
    order = np.argsort(nominal)
    nominal = nominal[order]
    levels = lv[order]

    rows = [IEC268_PROGRAM_TABLE[float(value)] for value in nominal]
    return IEC268ProgramCheck(
        frequencies=np.asarray(nominal, dtype=np.float64),
        band_levels=np.asarray(levels, dtype=np.float64),
        relative_levels=np.array([level for level, _t in rows], dtype=np.float64),
        lower_tolerance_db=np.array([tol for _l, tol in rows], dtype=np.float64),
        upper_tolerance_db=np.array([tol for _l, tol in rows], dtype=np.float64),
    )


def third_octave_edges(
    centers: Sequence[float],
    standard: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """1/3 倍频程频带边界（ANSI S1.11-2004 / IEC 61260-1）。

    标准滤波带的定义（这也是真实分析仪所使用的频带）：

    1. 先把名义中心换算成**精确**中心 ``1000·10^(n/10)``；
    2. 下边界 = 精确中心 × ``10^(-0.05)``，上边界 = 精确中心 × ``10^(+0.05)``。

    因为精确中心按 10^0.1 等比排列，第 k 频带上边界恰好等于第 k+1 频带
    下边界 —— 相邻频带**严丝合缝**，既不重叠也不留洞。举例：1000 Hz 频带
    是 891.3 Hz – 1122 Hz，20 kHz 频带是 17783 Hz – 22387 Hz，
    与 ANSI S1.11-2004 的表格一致。

    参数
    ----
    centers
        1/3 倍频程中心频率（标准表上的名义值即可）。
    standard
        ``True``（默认）给出**标准**频带（精确中心 × 10^(±0.05)）。

        ``False`` 给出历史上的近似带"名义中心 × 2^(±1/6)"：相邻频带
        **重叠 0.79 %** 并且留下最大 **1.57 %** 的空洞（例如 12.5 kHz 与
        16 kHz 之间有 223.6 Hz 的死区）。这不是标准频带，真实分析仪测出来
        的分带功率与标准表对不上 —— 仅保留用于对照。
    """
    fc = np.asarray(centers, dtype=np.float64)
    if standard:
        fc = np.array([third_octave_exact_center(float(c)) for c in fc])
        return fc * 10.0 ** -0.05, fc * 10.0 ** 0.05
    ratio = 2.0 ** (1.0 / 6.0)
    return fc / ratio, fc * ratio


def third_octave_band_power(
    freqs: np.ndarray,
    power: np.ndarray,
    centers: Sequence[float],
    standard: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """按 1/3 倍频程对功率谱做**积分**（求和），得到每个频带的总功率。

    与 :func:`signal_gen.spectrum.one_sixth_octave_smoothing` 口径一致 ——
    两者都是"频带总功率"，只是阶数不同（1/3 与 1/6）。
    传入的 ``power`` 应来自
    :func:`signal_gen.spectrum.one_sided_spectrum_dbfs_power`，
    其单位是 dBFS 标度的每 bin 功率。

    频带边界默认取**标准**频带（见 :func:`third_octave_edges`），
    与真实分析仪一致 —— 这样测出来的分带功率才能直接和 IEC 60268-1
    Table II 的标准表对比。

    返回 ``(中心频率, 各频带功率)``；中心频率按传入的 ``centers`` 原样返回。
    """
    f = np.asarray(freqs, dtype=np.float64)
    p = np.asarray(power, dtype=np.float64)
    if f.size == 0:
        return np.zeros(0), np.zeros(0)

    if f.size > 1 and not bool(np.all(np.diff(f) >= 0.0)):
        idx = np.argsort(f, kind="stable")
        f = f[idx]
        p = p[idx]

    lo, hi = third_octave_edges(centers, standard=standard)
    out_c: List[float] = []
    out_p: List[float] = []
    for c, a, b in zip(centers, lo, hi):
        i0 = int(np.searchsorted(f, a, side="left"))
        i1 = int(np.searchsorted(f, b, side="left"))
        if i1 > i0:
            out_c.append(float(c))
            out_p.append(float(np.sum(p[i0:i1])))
    return np.asarray(out_c), np.asarray(out_p)


def eia426b_spectrum(freqs: np.ndarray) -> np.ndarray:
    """EIA RS-426-B 寿命测试噪声的**频域幅度整形因子**（未归一化）。

    标准原文（经 Keele, AES 2001 逐条转述）：

        *The specified weighting (as measured by a constant percentage
        bandwidth analyzer), bandlimits the noise to 40 Hz and 10 kHz,
        is flat between 40 Hz and 1 kHz, and then rolls off at
        3 dB/octave (10 dB/decade) between 1 kHz and 10 kHz.*

    关键在于"**以恒定百分比带宽分析仪读数为准**"这句话：

    * 40 Hz – 1 kHz 读数**平坦** ⇒ 每个 1/3 倍频程频带功率相同
      ⇒ 功率谱密度 ∝ **1/f**（粉噪）；
    * 1 kHz – 10 kHz 读数按 **3 dB/octave（10 dB/decade）** 下降
      ⇒ 功率谱密度 ∝ **1/f²**。

    所以返回的是**幅度谱**（功率谱开根号）::

        |A(f)| = f^-0.5                      , f ≤ 1 kHz
        |A(f)| = f^-0.5 · (1 kHz / f)^0.5    , f > 1 kHz

    **这不是"粉噪 + 带通"**：粉噪斜率只是 40 Hz – 1 kHz 这一段的形状，
    1 kHz 以上还要**再叠加一段 1/f 的功率谱倾斜**，总斜率变成 1/f²。
    把它当成带通粉噪生成，1 kHz 以上就会比标准高出一整个倍频程的斜率
    （−3 dB/oct），高频能量明显偏多。

    参数
    ----
    freqs
        频率轴 (Hz)，非负。

    返回
    ----
    与输入等长的幅度整形因子；``f = 0`` 处为 0（去直流）。
    """
    f = np.asarray(freqs, dtype=np.float64)
    amp = np.zeros_like(f)
    nz = f > 0.0
    # 功率谱 ∝ 1/f  →  幅度 ∝ f^-0.5
    amp[nz] = f[nz] ** -0.5
    # 1 kHz 以上再叠加一段 1/f 的功率谱倾斜 → 幅度再乘 f^-0.5
    hi = f > EIA426B_WEIGHT_KNEE_HZ
    amp[hi] *= (EIA426B_WEIGHT_KNEE_HZ / f[hi]) ** 0.5
    return amp


# ---------------------------------------------------------------------------
# 软削波（Keele Eq. 1）
# ---------------------------------------------------------------------------

def soft_clip(x: np.ndarray, k: float = 4.0) -> np.ndarray:
    """Keele 论文 Eq.(1) 的**软削波**输入输出函数。

    原文给出的分段形式（``x`` 为输入、``y`` 为输出、``k`` 为 rounding 参数）::

        y = [ x^k / (x^k + 1) ]^(1/k)      , x ≥ 0
        y = −[ (−x)^k / ((−x)^k + 1) ]^(1/k) , x < 0

    等价于紧凑写法 ``y = sign(x)·|x| / (1 + |x|^k)^(1/k)``，本实现用后者
    （数值上用对数域计算，避免 ``x**k`` 溢出）。

    性质：

    * 小信号（|x| ≲ 0.1）斜率为 1，即**线性无失真**；
    * 输出渐近饱和于 ±1，**永远不会超过 ±1**；
    * ``k`` 越小拐点越圆（k=1 "quite rounded"），越大越接近硬削波
      （k=100 "hard clip"）；EIA RS-426-B 的测试 CD 用的是 **k = 4**
      （"typical rounding"）。

    参数
    ----
    x
        输入信号（不会被修改）。
    k
        rounding 参数，必须 > 0。
    """
    if k <= 0.0:
        raise ValueError(f"rounding 参数 k 必须为正，得到 {k!r}")
    x = np.asarray(x, dtype=np.float64)
    a = np.abs(x)
    out = np.zeros_like(a)
    nz = a > 0.0
    if np.any(nz):
        log_a = np.log(a[nz])
        # log(1 + a^k) = logaddexp(0, k·log a)
        log_den = np.logaddexp(0.0, k * log_a) / k
        out[nz] = np.exp(log_a - log_den)
    return np.sign(x) * out


def soft_clip_to_crest(
    signal: np.ndarray,
    crest_db: float,
    k: float = 4.0,
    iterations: int = 48,
) -> np.ndarray:
    """用 :func:`soft_clip` 把信号的峰值因子压到指定值。

    EIA RS-426-B 的寿命测试噪声要求 *soft clipped to a crest factor of
    6 dB*：先用一个输入增益把噪声推进软削波函数的拐点，使峰值被"磨圆"，
    再整体缩放到要求的峰值电平。Keele 的实现正是如此（论文步骤 5：
    *the input data must be scaled up or down to appropriately clip the
    data at the proper level and then scaled back*）。

    做法：二分输入增益 ``A``。峰值因子 ``crest(A)`` 随 ``A`` 单调递减
    （增益越大 → 削波越深 → 峰值被压得越狠 → 峰值因子越小），因此可以
    唯一地解出 ``crest(A) = 目标峰值因子``。

    与硬削波（:func:`limit_crest_factor`）相比，软削波得到的是**圆拐点**
    的削波，峰值附近的概率密度连续、不会像硬削波那样在阈值处堆积，
    这正是标准点名 "soft clipped" 的原因。

    参数
    ----
    signal
        输入信号（不会被修改）。
    crest_db
        目标峰值因子 (dB)。``<= 0`` 表示不限幅，直接原样返回。
    k
        软削波的 rounding 参数，见 :func:`soft_clip`。
    iterations
        二分次数。48 次足够把增益定到 1e-7 量级。

    返回
    ----
    软削波后的信号（**未**做峰值归一化）。
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.size == 0 or crest_db is None or crest_db <= 0.0:
        return x

    target = 10.0 ** (float(crest_db) / 20.0)
    rms0 = float(np.sqrt(np.mean(np.square(x))))
    if rms0 <= 0.0:
        return x
    # 归一到单位 RMS，二分区间才与信号本身的无量纲增益解耦
    x = x / rms0

    def _crest(gain: float) -> float:
        y = soft_clip(gain * x, k)
        r = float(np.sqrt(np.mean(np.square(y))))
        if r <= 0.0:
            return float("inf")
        return float(np.max(np.abs(y))) / r

    lo, hi = 1e-3, 1e3
    for _ in range(80):
        if _crest(hi) <= target:
            break
        hi *= 2.0
    else:  # pragma: no cover - 理论上不会发生
        raise ValueError("无法把峰值因子压到目标值，请检查目标是否合理")

    for _ in range(max(1, int(iterations))):
        mid = math.sqrt(lo * hi)
        if _crest(mid) > target:
            lo = mid
        else:
            hi = mid

    gain = math.sqrt(lo * hi)
    return soft_clip(gain * x, k)


# ---------------------------------------------------------------------------
# 峰值因子
# ---------------------------------------------------------------------------

def crest_factor_db(signal: np.ndarray) -> float:
    """返回信号的峰值因子 (dB)：``20·log10(peak / rms)``。"""
    x = np.asarray(signal, dtype=np.float64)
    if x.size == 0:
        return 0.0
    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(np.square(x))))
    if rms <= 0.0:
        return 0.0
    return 20.0 * math.log10(peak / rms)


def limit_crest_factor(
    signal: np.ndarray,
    crest_db: float,
    iterations: int = 28,
) -> np.ndarray:
    """把信号的峰值因子压到指定值（硬削波）。

    为什么需要削波：高斯噪声在长记录上的峰值因子高达 14 ~ 16 dB，
    而标准要求的是确定的 6 dB（2:1）。业界通行的做法就是"clipped pink noise"。

    实现：用二分法求削波阈值 ``T``。削波后信号的峰值恰为 ``T``，
    而 ``h(T) = T / rms(clip(x, T))`` 随 ``T`` 单调递增，
    因此二分可以唯一地解出 ``h(T) = 目标峰值因子``。

    参数
    ----
    signal
        输入信号（不会被修改）。
    crest_db
        目标峰值因子 (dB)。``<= 0`` 表示不限幅，直接原样返回。
    iterations
        二分次数。28 次即可把阈值定到 1e-8 量级，远超需要的精度。

    返回
    ----
    削波后的信号。若原信号本来就满足要求，则原样返回。
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.size == 0 or crest_db is None or crest_db <= 0.0:
        return x

    target = 10.0 ** (float(crest_db) / 20.0)
    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(np.square(x))))
    if peak <= 0.0 or rms <= 0.0:
        return x
    if peak / rms <= target:
        return x  # 本来就满足，不需要削波

    lo = 0.0
    hi = peak
    x2 = np.square(x)
    for _ in range(max(1, int(iterations))):
        mid = 0.5 * (lo + hi)
        # rms(clip(x, ±mid))² = mean(min(x², mid²)) —— 一次遍历即可算出
        r = float(np.sqrt(np.minimum(x2, mid * mid).mean()))
        if r <= 0.0:
            lo = mid
            continue
        if mid / r < target:
            lo = mid   # 阈值偏小 → 峰值因子偏小 → 需要更大的阈值
        else:
            hi = mid

    threshold = 0.5 * (lo + hi)
    return np.clip(x, -threshold, threshold)
