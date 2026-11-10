"""
信号生成核心。

所有生成器均为**纯函数**：给定相同的 :class:`SignalSpec`（含 ``seed``），
输出逐比特完全一致，便于测试信号标定与结果复现。

谱型约定
--------
噪声用 **beta** 描述功率谱密度斜率::

    S(f) ∝ f ** (-beta)

    beta =  0  -> 白噪声 (flat)
    beta =  1  -> 粉噪声 (1/f)
    beta =  2  -> 红/布朗噪声 (1/f^2)
    beta = -1  -> 蓝噪声 (f)

与分形布朗运动 (fBm) 的 Hurat 指数 H 的关系为 ``beta = 2H - 1``。
（注：早期版本用 ``hurst`` 命名却按 ``f**(-hurst)`` 计算幅度，
实际等价于 ``beta = 2 * hurst``，两者只有在 beta=1 时才巧合一致。）
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .standards import (
    EIA426B_CAL_TONE_PEAK_DBFS,
    EIA426B_CREST_DB,
    EIA426B_HP_FC,
    EIA426B_HP_ORDER,
    EIA426B_LP_FC,
    EIA426B_LP_ORDER,
    EIA426B_SOFT_CLIP_K,
    IEC268_CREST_DB,
    IEC268_HP_FC,
    IEC268_HP_ORDER,
    IEC268_LP_FC,
    IEC268_LP_ORDER,
    IEC268_PINK_CREST_DB,
    IEC268_PROGRAM_CENTERS,
    STANDARD_CREST_DB,
    butterworth_magnitude,
    eia426b_spectrum,
    iec268_figure2_response,
    iec268_program_psd,
    limit_crest_factor,
    soft_clip_to_crest,
    third_octave_edges,
)

__all__ = [
    "SignalSpec",
    "generate_signal",
    "normalize",
    "clamp_spec_to_nyquist",
    "default_params_for",
    "MAX_SAMPLES",
    "SIGNAL_KINDS",
    "ALL_KINDS",
    "NOISE_KINDS",
    "SWEEP_KINDS",
    "STANDARD_KINDS",
    "KIND_LABELS",
    "KIND_DESCRIPTIONS",
    "DEFAULT_PARAMS",
    "PARAM_SPECS",
    "KIND_PARAMS",
    "FREQ_PARAM_INFO",
    "tukey_window",
]

#: 单次生成允许的最大样本数（约 400 s @ 48 kHz）。
#: 超过此值会占用过多内存，GUI 中直接给出提示而不是把机器拖垮。
MAX_SAMPLES = 20_000_000

#: 频率上限相对奈奎斯特频率的安全裕度（留 1% 给滤波器过渡带）。
_NYQUIST_MARGIN = 0.99

#: ``SignalSpec.f1`` 的默认值 (Hz) —— 单频正弦的默认基频。
_DEFAULT_F1 = 1000.0
#: ``SignalSpec.f_start`` / ``f_stop`` 的默认值 (Hz)。
_DEFAULT_F_START = 20.0
_DEFAULT_F_STOP = 20000.0

#: "该类型没有规定默认值"时的兜底值（见 :meth:`SignalSpec.__post_init__`）。
#:
#: 键名同时也是**允许按信号类型自动补齐的字段清单** —— 只有列在这里的字段，
#: ``None`` 才表示"未指定"，其余字段的 ``None`` 一律原样保留。
#:
#: ``crest_factor_db`` 的兜底是 **0**（不限幅）：标准信号的 6 dB 由
#: :data:`DEFAULT_PARAMS` 按类型给出（EIA / IEC 60268-5 带限粉噪为 6，
#: IEC 60268-1 Clause 7 节目信号为 0 即不限幅），
#: 非标准类型则没有峰值因子要求，取 0 表示不做削波。
_PARAM_FALLBACKS: Dict[str, Any] = {
    "f_start": _DEFAULT_F_START,
    "f_stop": _DEFAULT_F_STOP,
    "f1": _DEFAULT_F1,
    "beta": 1.0,
    "fade_ratio": 0.25,
    "crest_factor_db": 0.0,
}


# ---------------------------------------------------------------------------
# 采样与时间轴工具
# ---------------------------------------------------------------------------

def _time_axis(n_samples: int, fs: int) -> np.ndarray:
    """生成不含尾端点的时间轴。

    使用 ``arange(n)/fs`` 而非 ``linspace(0, dur, n)``：后者默认包含尾端点，
    会使采样间隔变成 ``dur/(n-1)`` 而非 ``1/fs``，导致所有频率整体偏移。
    """
    return np.arange(n_samples, dtype=np.float64) / float(fs)


def _n_samples(duration: float, fs: int) -> int:
    return int(round(duration * fs))


def tukey_window(n: int, alpha: float = 0.25) -> np.ndarray:
    """Tukey (raised-cosine) 窗。

    ``alpha`` 为渐变段占总长度的比例，两端各占 ``alpha/2``。
    用于抑制扫频步进处的瞬态（click/splatter）。
    """
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    if alpha <= 0:
        return np.ones(n, dtype=np.float64)
    alpha = min(1.0, alpha)
    edge = int(round(alpha * n / 2.0))
    if edge <= 0:
        return np.ones(n, dtype=np.float64)
    w = np.ones(n, dtype=np.float64)
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(edge) / edge))
    w[:edge] = ramp
    w[-edge:] = ramp[::-1]
    return w


# ---------------------------------------------------------------------------
# 各信号生成器
# ---------------------------------------------------------------------------

#: 带限过渡带的宽度（倍频程）。1/24 倍频程 ≈ 2.9 %，足够窄，
#: 保证**默认参数下几乎不改变信号谱形**（粉噪默认 20 Hz – 20 kHz，
#: 过渡带只有 19.1 – 20 Hz 与 20.0 – 20.9 kHz，带内斜率完全不受影响）。
_BAND_EDGE_OCTAVES = 1.0 / 24.0


def _band_gate(freqs: np.ndarray, f_start: float, f_stop: float, fs: float) -> np.ndarray:
    """返回 0..1 的带通权重（频域直接相乘，没有滤波器启动瞬态）。

    * 带内（``f_start`` – ``f_stop``）权重 = 1，**完全平坦**；
    * 两侧过渡带各占 1/24 倍频程，用 raised-cosine 平滑；
    * 带外权重 = 0。

    过渡带做得很窄，是为了不破坏 1/f 谱斜率：任何带滚降的高通都会
    在带内造成衰减（Butterworth 高通在 20 Hz 拐点会让 100 Hz 处低 28 dB），
    而砖墙式过渡对带内只有 0.01 dB 量级的影响。
    """
    f = np.asarray(freqs, dtype=np.float64)
    g = np.ones_like(f)
    nyq = fs / 2.0
    edge = 2.0 ** _BAND_EDGE_OCTAVES

    # 下过渡带 [f_start/edge, f_start]
    lo = f_start / edge
    if lo > 0.0:
        m = (f >= lo) & (f < f_start)
        g[m] = 0.5 * (1.0 + np.cos(np.pi * (f[m] - lo) / (f_start - lo)))
        g[f < lo] = 0.0
    else:
        g[f < f_start] = 0.0

    # 上过渡带 [f_stop, f_stop*edge]；超出奈奎斯特就截断
    hi = min(f_stop * edge, nyq)
    if hi > f_stop:
        m = (f > f_stop) & (f <= hi)
        g[m] = 0.5 * (1.0 + np.cos(np.pi * (hi - f[m]) / (hi - f_stop)))
        g[f > hi] = 0.0
    else:
        g[f > f_stop] = 0.0

    return g


def _gen_white(spec: "SignalSpec") -> np.ndarray:
    """高斯白噪声：功率谱平坦（S(f) 恒定）。

    与粉/红/蓝噪声共用同一套**频域**生成器（beta = 0），这样带限
    （``f_start`` / ``f_stop``）对白噪声同样生效。时域高斯白噪声与
    频域平坦谱在统计上等价（bin 幅度服从瑞利分布），但只有频域做法
    才能做带限整形。
    """
    return _gen_colored_noise(spec.with_(beta=0.0))


def _gen_colored_noise(spec: "SignalSpec") -> np.ndarray:
    """按 ``S(f) ∝ f**(-beta)`` 构造的彩色噪声（``beta=0`` 即白噪声）。

    实现要点（相对旧版的三处修正）：

    1. 只在正频率半轴构造幅度 ``f**(-beta/2)``，再由 ``irfft`` 自动补出
       负频率的共轭对称分量 —— 旧版用 ``ifft(...).real`` 丢掉了约一半随机
       能量，且破坏了 Hermitian 对称性。
    2. 先掩蔽 DC 再取幂，不产生 ``0 ** 负数`` 的 divide-by-zero 警告。
    3. 频率轴使用真实 Hz（``rfftfreq(n, 1/fs)``），便于与其它模块对齐。

    **带限（``f_start`` – ``f_stop``）在频域直接乘权重**：没有任何滤波器启动
    瞬态，带内平坦（见 :func:`_band_gate`）。白噪声也走这条路 —— 时域高斯白噪声
    与频域平坦谱在统计上等价，但只有频域做法才能做带限。
    """
    rng = np.random.default_rng(spec.seed)
    n = _n_samples(spec.duration, spec.fs)
    freqs = np.fft.rfftfreq(n, 1.0 / spec.fs)

    amplitude = np.zeros_like(freqs)
    nonzero = freqs > 0
    amplitude[nonzero] = freqs[nonzero] ** (-spec.beta / 2.0)

    # 频带限制：砖墙 + 窄余弦过渡带，带内完全平坦
    amplitude *= _band_gate(freqs, spec.f_start, spec.f_stop, spec.fs)

    phase = rng.uniform(0.0, 2.0 * np.pi, size=freqs.size)
    spectrum = amplitude * np.exp(1j * phase)
    spectrum[0] = 0.0  # 去直流

    return np.fft.irfft(spectrum, n=n)


def _gen_multitone(spec: "SignalSpec") -> np.ndarray:
    """多音信号：若干对数分布的正弦分量叠加，相位随机。

    随机相位可显著降低合成信号的峰值因子，是标准做法。
    """
    rng = np.random.default_rng(spec.seed)
    n = _n_samples(spec.duration, spec.fs)
    t = _time_axis(n, spec.fs)

    freqs = np.logspace(
        math.log10(spec.f_start), math.log10(spec.f_stop), num=spec.n_tones
    )
    # 每音等幅，使频谱平坦；总幅度再由归一化统一处理
    tone_amp = 1.0 / math.sqrt(spec.n_tones)
    phases = rng.uniform(-np.pi, np.pi, size=freqs.size)

    signal = np.zeros(n, dtype=np.float64)
    for f, ph in zip(freqs, phases):
        signal += tone_amp * np.sin(2.0 * np.pi * f * t + ph)
    return signal


def _gen_sweep(spec: "SignalSpec", method: str = "log") -> np.ndarray:
    """连续扫频（chirp）。

    ``method='log'``   等比扫频（频率指数增长），每个倍频程停留时间相同；
    ``method='linear'`` 线性扫频。

    频率轨迹已验证为精确的解析解：对数扫频的 ``log10(f(t))`` 对时间严格线性
    （线性拟合残差 std ≈ 0.035），线性扫频则严格线性。**这里不做任何改动**。

    混叠防护：``f_stop`` 若超过奈奎斯特频率，采样得到的并不是想要的频率，
    而是折叠回来的其它频率（实测 fs=8 kHz、f_stop=20 kHz 时 20 kHz 会折叠到
    3.6 kHz 附近，整段扫频听起来完全不对）。因此频率在 :func:`generate_signal`
    入口处统一被钳制到 ``0.99 * fs/2``。

    扫频信号自身是连续的，但若要做单次 FFT 分析，边界仍会泄漏；
    建议配合 :func:`signal_gen.spectrum.one_sided_amplitude_spectrum` 的窗函数
    使用，或改用 spectrogram 观察瞬时频率轨迹。
    """
    from scipy.signal import chirp as scipy_chirp

    n = _n_samples(spec.duration, spec.fs)
    t = _time_axis(n, spec.fs)
    signal = scipy_chirp(
        t, f0=spec.f_start, f1=spec.f_stop, t1=spec.duration, method=method
    )
    # 首尾各加一小段渐变，抑制边界瞬态
    if spec.fade_ratio > 0:
        w = tukey_window(n, spec.fade_ratio)
        signal = signal * w
    return signal


def _gen_stepped_sweep(spec: "SignalSpec") -> np.ndarray:
    """步进扫频（stepped sweep）：等比分布的频率台阶。

    采用教科书 / IEEE 标准里的定义：**幅度恒定 + 相位连续**。

    * 每个台阶内频率恒定、幅度恒为 1 —— 信号里不存在静音间隙；
    * 相位跨台阶连续 —— 台阶边界没有咔哒声（click）。

    旧实现给**每个台阶**都加 Tukey 渐变，实测有两个严重缺陷：

    1. 每步都要淡入淡出到 0，n_steps=50 时产生 6314/240000（2.6%）接近静音的
       样点，信号听上去是"断续"的；
    2. 低频台阶的渐变段覆盖不到一个完整周期（20 Hz 台阶的 25% 渐变段只有
       62.5 ms = 1.25 个周期），台阶严重失真 —— 实测低频台阶纯度仅 44 dB。
       改成相位连续后，20 Hz 台阶纯度提升到 **108 dB**。

    渐变（``fade_ratio``）只作用于**整段的首尾**，不再逐台阶加窗：
    连续扫频里渐变段占总长比例，步进扫频里渐变段长度按每步时长折算。
    """
    n_step = _n_samples(spec.step_duration, spec.fs)
    t = _time_axis(n_step, spec.fs)

    freqs = np.logspace(
        math.log10(spec.f_start), math.log10(spec.f_stop), num=spec.n_steps
    )

    segments = []
    phase = 0.0
    for f in freqs:
        ph = 2.0 * np.pi * f * t + phase
        segments.append(np.sin(ph))
        # 下一台阶从本台阶"下一个采样点"的相位接续，跨台阶相位严格连续
        phase = float(ph[-1] + 2.0 * np.pi * f / spec.fs)

    signal = np.concatenate(segments)

    # 整段首尾各做一次温和淡入淡出，抑制起止端瞬态（不逐台阶加窗）
    if spec.fade_ratio > 0:
        total_fade = 2.0 * spec.fade_ratio * n_step
        alpha = min(1.0, total_fade / max(signal.size, 1))
        if alpha > 0:
            signal = signal * tukey_window(signal.size, alpha)
    return signal


def _gen_sine(spec: "SignalSpec") -> np.ndarray:
    """单频正弦（带可选谐波）。"""
    n = _n_samples(spec.duration, spec.fs)
    t = _time_axis(n, spec.fs)
    signal = spec.amplitude * np.sin(2.0 * np.pi * spec.f1 * t)
    for k, h_amp in enumerate(spec.harmonics, start=2):
        signal += spec.amplitude * h_amp * np.sin(2.0 * np.pi * spec.f1 * k * t + k)
    return signal


# ---------------------------------------------------------------------------
# 标准测试信号
# ---------------------------------------------------------------------------

def _iec268_program_band_mask(
    freqs: np.ndarray,
    f_min: float,
    f_max: float,
) -> np.ndarray:
    """Clause 7 节目信号的 1/3 倍频程频带掩码（0 / 1）。

    只有**与 ``[f_min, f_max]`` 有重叠**的标准频带才被保留；频带边界用
    :func:`signal_gen.standards.third_octave_edges` 的标准频带
    （ANSI S1.11-2004 / IEC 61260-1：精确中心 × 10^(±0.05)）—— 与整形、
    验收用的完全是同一组边界。

    默认 ``f_min=20`` / ``f_max=20000`` 时，Table II 的 **31 个频带全部**
    入选，掩码覆盖 17.78 Hz – 22.39 kHz —— 与 Table II 自身的覆盖范围
    完全相同（也正好落在 1/3 倍频程频带的边界上，不会切掉半个频带）。
    """
    f = np.asarray(freqs, dtype=np.float64)
    mask = np.zeros_like(f)
    lo, hi = third_octave_edges(IEC268_PROGRAM_CENTERS)
    for a, b in zip(lo, hi):
        if b < f_min or a > f_max:
            continue
        mask[(f >= a) & (f < b)] = 1.0
    return mask


def _gen_iec268_program(spec: "SignalSpec") -> np.ndarray:
    """IEC 60268-1:1985 Clause 7 —— 模拟节目信号（simulated programme signal）。

    标准原文（Clause 7）：

        *A signal, whose mean power spectral density closely resembles the
        average of the mean power spectral densities of a wide range of
        programme material, including both speech and music of several kinds,
        is stationary weighted Gaussian noise without amplitude limiting, the
        weighted power spectrum being in accordance with Table II and Figure 1,
        page 22, when measured with third-octave filters in accordance with IEC
        Publication 225.*

    两种实现（由 ``spec.kind`` 决定）：

    * ``iec268``      —— **Table II 逐带精确**：功率谱密度在 lg f 上是分段
      直线，节点经迭代拟合，使每个 1/3 倍频程频带的闭式积分**正好等于表值**；
      20 Hz 以下 / 20 kHz 以上按端段斜率延伸。
    * ``iec268_fig2`` —— **Figure 2 电路**：粉噪经标准给出的无源 RC 梯形
      网络。同样落在容差内，但最差频带只剩 0.18 dB 裕量。

    两者都是**平稳**高斯噪声：频域整形 + 均匀随机相位，得到的正是高斯过程
    的有限实现。整形全在频域完成，没有滤波器启动瞬态。

    **幅度限制**：Clause 7 要求 *"without amplitude limiting"*，所以默认
    **不限幅**（峰值因子 0）。用户显式给了 ``crest_factor_db``（例如
    IEC 60268-7:2010 8.3.2 规定耳机的 *"with additional clipping"*、
    peak-to-r.m.s. 1.8 ~ 2.2 ≈ 峰值因子 5.1 ~ 6.9 dB）才做硬削波。
    """
    rng = np.random.default_rng(spec.seed)
    n = _n_samples(spec.duration, spec.fs)
    freqs = np.fft.rfftfreq(n, 1.0 / spec.fs)

    amplitude = np.zeros_like(freqs)
    nz = freqs > 0.0
    if spec.kind == "iec268_fig2":
        # Figure 2：粉噪（幅度 ∝ f^-0.5，即功率谱 ∝ 1/f）经无源 RC 梯形网络。
        # 只取**幅度** —— 电阻电容网络的相位对平稳高斯噪声没有意义，
        # 而保持共轭对称才能保证输出是实信号。
        response = np.abs(iec268_figure2_response(freqs[nz]))
        amplitude[nz] = response / np.sqrt(freqs[nz])
    else:
        # Table II：逐带精确的功率谱密度
        amplitude[nz] = np.sqrt(iec268_program_psd(freqs[nz]))

    # 只保留选中的 1/3 倍频程频带（默认 = Table II 的 31 个频带）
    amplitude *= _iec268_program_band_mask(freqs, spec.f_start, spec.f_stop)

    phase = rng.uniform(0.0, 2.0 * np.pi, size=freqs.size)
    spectrum = amplitude * np.exp(1j * phase)
    spectrum[0] = 0.0  # 去直流
    signal = np.fft.irfft(spectrum, n=n)

    # 标准要求"不做幅度限制"，默认峰值因子为 0；用户显式指定时才削波
    crest = spec.crest_factor_db
    if crest is None:
        crest = STANDARD_CREST_DB.get(spec.kind, 0.0)
    if crest and crest > 0:
        signal = limit_crest_factor(signal, crest)

    return signal


def _gen_standard_noise(spec: "SignalSpec") -> np.ndarray:
    """标准测试信号：EIA RS-426-B / IEC 60268-1。

    都是"扬声器功率承载 / 最佳功放功率"测试用的激励信号，
    差别在于频谱整形方式：

    * ``eia426b``  —— **频谱加权的高斯噪声**（EIA RS-426-B 加速寿命测试，
      Keele 测试 CD 的 Track 2）：40 Hz – 1 kHz 读数平坦（功率谱 ∝ 1/f），
      1 kHz – 10 kHz 读数按 3 dB/octave（10 dB/decade）滚降（功率谱 ∝ 1/f²）；
      限带用 40 Hz / 10 kHz 的 Butterworth；峰值因子用**软削波**压到 6 dB。
      **不是"带通粉噪"** —— 1 kHz 以上还要再叠加一段 1/f 的功率谱倾斜。
    * ``iec268``   —— **IEC 60268-1 Clause 7 模拟节目信号**：按 Table II 的
      1/3 倍频程相对电平表整形的平稳加权高斯噪声，**不做幅度限制**
      （详见 :func:`_gen_iec268_program`）。
    * ``iec268_fig2`` —— 同上，但用标准给出的 **Figure 2 无源 RC 电路**实现。
    * ``iec268_pink`` —— **IEC 60268-1 附录 C / IEC 60065 带限粉噪**：
      12 dB/oct 高通（-3 dB @ 22.7 Hz）+ 18 dB/oct 低通（-3 dB @ 22.4 kHz），
      22.4 Hz – 22.4 kHz 基本平坦；配合 IEC 60268-5 的功率承载测试
      （clipped pink noise，峰值因子 2:1）。

    实现全部在频域完成（整形因子直接乘滤波器的解析幅度响应），
    好处是没有滤波器启动瞬态，响应严格等于标准给定的曲线。

    **频带范围由 ``f_start`` / ``f_stop`` 控制**，默认值恰好就是标准值：

    * ``eia426b``  —— 两者直接作为 Butterworth 的 -3 dB 拐点
      （默认 40 Hz / 10 kHz）；低通拐点不得超过奈奎斯特频率。
    * ``iec268`` / ``iec268_fig2`` —— 两者限定"纳入整形的 1/3 倍频程频带
      范围"（默认 20 Hz – 20 kHz = Table II 的全部 31 个频带）。
    * ``iec268_pink`` —— 两者直接作为 Butterworth 的 -3 dB 拐点
      （默认 22.7 Hz / 22.4 kHz）。

    一旦偏离标准默认值，得到的就不再是严格合规的标准信号 —— GUI 会给出提示。
    """
    if spec.kind in ("iec268", "iec268_fig2"):
        return _gen_iec268_program(spec)

    rng = np.random.default_rng(spec.seed)
    n = _n_samples(spec.duration, spec.fs)
    freqs = np.fft.rfftfreq(n, 1.0 / spec.fs)

    if spec.kind == "eia426b":
        # EIA RS-426-B 的"频谱加权"：不是粉噪再带通，而是分两段的加权谱。
        #   40 Hz – 1 kHz : 功率谱 ∝ 1/f      （恒定百分比带宽读数平坦）
        #   1 kHz – 10 kHz: 功率谱 ∝ 1/f²     （读数 −3 dB/octave）
        amplitude = eia426b_spectrum(freqs)
        # 限带：40 Hz / 10 kHz 的 maximally-flat Butterworth（标准要求
        # 高通至少 6 阶、低通至少 4 阶）；低通拐点不得超过奈奎斯特频率
        lp_fc = min(spec.f_stop, _NYQUIST_MARGIN * spec.fs / 2.0)
        amplitude *= butterworth_magnitude(
            freqs, spec.f_start, EIA426B_HP_ORDER, "high"
        )
        amplitude *= butterworth_magnitude(
            freqs, lp_fc, EIA426B_LP_ORDER, "low"
        )
    else:
        # iec268_pink：IEC 60065 附录 C 的带限粉噪
        # 粉噪：S(f) ∝ f^-1  →  幅度 ∝ f^-0.5
        amplitude = np.zeros_like(freqs)
        nonzero = freqs > 0.0
        amplitude[nonzero] = freqs[nonzero] ** -0.5

        # 拐点就是用户设定的 f_start / f_stop（默认 = 标准值）；
        # 低通拐点不得超过奈奎斯特频率，否则毫无意义
        lp_fc = min(spec.f_stop, _NYQUIST_MARGIN * spec.fs / 2.0)
        amplitude *= butterworth_magnitude(
            freqs, spec.f_start, IEC268_HP_ORDER, "high"
        )
        amplitude *= butterworth_magnitude(
            freqs, lp_fc, IEC268_LP_ORDER, "low"
        )

    phase = rng.uniform(0.0, 2.0 * np.pi, size=freqs.size)
    spectrum = amplitude * np.exp(1j * phase)
    spectrum[0] = 0.0  # 去直流
    signal = np.fft.irfft(spectrum, n=n)

    # 峰值因子是标准硬性指标：高斯噪声天然有 14~16 dB，必须削波压下来。
    # ``spec.crest_factor_db`` 已在 :meth:`SignalSpec.__post_init__` 里按类型
    # 解析过（标准信号取 :data:`standards.STANDARD_CREST_DB`），这里的
    # ``None`` 分支只是防御性兜底。
    crest = spec.crest_factor_db
    if crest is None:
        crest = STANDARD_CREST_DB.get(spec.kind, 0.0)

    if spec.kind == "eia426b":
        # EIA RS-426-B 要求的是 *soft clipped* —— 用 Keele Eq.(1) 的圆拐点
        # 软削波（k = 4），而不是硬削波：峰值附近被"磨圆"，不会像硬削波
        # 那样在阈值处堆积。
        if crest and crest > 0:
            signal = soft_clip_to_crest(signal, crest, k=EIA426B_SOFT_CLIP_K)
        # 绝对电平：标准规定噪声峰值与 1 kHz 校准音相同（−6 dBFS），
        # RMS 比校准音低 3 dB（−12 dBFS）；6 dB 的峰值因子自动满足这条。
        peak = float(np.max(np.abs(signal)))
        if peak > 0.0:
            signal *= (10.0 ** (EIA426B_CAL_TONE_PEAK_DBFS / 20.0)) / peak
    elif crest and crest > 0:
        signal = limit_crest_factor(signal, crest)

    return signal



# ---------------------------------------------------------------------------
# 规格定义
# ---------------------------------------------------------------------------

@dataclass
class SignalSpec:
    """一次信号生成的完整参数描述。

    所有字段都是标量或简单序列，可 JSON 序列化，便于保存为“测试用例”。
    """

    kind: str = "white"
    fs: int = 48000
    duration: float = 5.0
    seed: int = 0

    # 频率范围（多音 / 扫频 / 步进扫频 / 噪声带限 / 标准信号频带）
    # None = 未指定，由 __post_init__ 按信号类型补上默认值
    f_start: Optional[float] = None
    f_stop: Optional[float] = None

    # 噪声谱斜率（None = 取该类型的默认值：白 0 / 粉 1 / 红 2 / 蓝 -1）
    beta: Optional[float] = None

    # 峰值因子（Crest Factor）目标值 (dB)：0 表示不限幅；
    # None 表示"用该信号类型在标准里的默认值"（见 STANDARD_CREST_DB）
    crest_factor_db: Optional[float] = None

    # 多音
    n_tones: int = 200

    # 步进扫频
    n_steps: int = 50
    step_duration: float = 0.5
    # 渐变比例（None = 取该类型的默认值：连续扫频 0.05 / 步进扫频 0.25）
    fade_ratio: Optional[float] = None

    # 单音
    f1: Optional[float] = None
    amplitude: float = 1.0
    harmonics: List[float] = field(default_factory=list)

    # 扫频方式
    sweep_method: str = "log"

    # 归一化
    normalize: str = "peak"      # "peak" | "rms" | "none"
    target_dbfs: float = -3.0    # rms 模式下的目标电平

    def __post_init__(self) -> None:
        """按信号类型补齐"未指定"的参数。

        dataclass 的字段默认值只能有一套，但不同信号类型的合理默认值并不相同
        （IEC 60268-1 附录 C 粉噪的高通拐点是 22.7 Hz、红噪的谱斜率是 2.0、
        连续扫频的渐变比例是 0.05……）。这些字段用 ``None`` 表示"未指定"，
        在这里统一按 :data:`DEFAULT_PARAMS` 补齐 —— 于是**库 API 的默认行为
        与 GUI / CLI 完全一致**，不必每个调用点都记得手工填一遍。

        单频正弦只有一个频率，界面上用"起始频率"承载它，所以只给了
        ``f_start`` 时会自动同步到 ``f1``。

        峰值因子 ``crest_factor_db`` 同样按类型补齐：标准信号从
        :data:`DEFAULT_PARAMS` 取（EIA RS-426-B 与 IEC 60268-5 带限粉噪
        为 6 dB；IEC 60268-1 Clause 7 节目信号为 0 即不限幅），
        非标准类型回落到 0 dB。
        因此 :attr:`crest_factor_db` **不会**残留为 ``None``，
        库 API 与 GUI / CLI 读到的默认峰值因子完全一致。

        ``dataclasses.replace`` 会重新触发本方法；由于补齐后字段不再是
        ``None``，重复触发是幂等的，不会把用户已经改过的值覆盖回去。
        """
        defaults = DEFAULT_PARAMS.get(self.kind, {})

        # 单频正弦：必须先于默认值填充处理，"只给 f_start"才生效
        if self.kind == "sine" and self.f1 is None and self.f_start is not None:
            self.f1 = float(self.f_start)

        for name, fallback in _PARAM_FALLBACKS.items():
            if getattr(self, name) is None:
                setattr(self, name, defaults.get(name, fallback))

    def with_(self, **kwargs: Any) -> "SignalSpec":
        """返回修改若干字段后的新实例（不可变风格）。"""
        return replace(self, **kwargs)

    @property
    def n_samples(self) -> int:
        if self.kind == "stepped":
            return _n_samples(self.step_duration, self.fs) * self.n_steps
        return _n_samples(self.duration, self.fs)

    @property
    def f_limit(self) -> float:
        """当前采样率下不产生混叠的最高可用频率 (Hz)。"""
        return _NYQUIST_MARGIN * self.fs / 2.0

    @property
    def effective_duration(self) -> float:
        return self.n_samples / float(self.fs)


# ---------------------------------------------------------------------------
# 归一化
# ---------------------------------------------------------------------------

def normalize(signal: np.ndarray, mode: str = "peak", target_dbfs: float = -3.0) -> np.ndarray:
    """按指定模式归一化。

    - ``peak``  峰值归一化到 0 dBFS（易削波，仅适合确定性波形）
    - ``rms``   RMS 归一化到 ``target_dbfs``，更符合音频工程惯例
    - ``none``  不处理
    """
    if mode == "none":
        return signal
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    if mode == "peak":
        return signal / peak if peak > 0 else signal
    if mode == "rms":
        rms = float(np.sqrt(np.mean(np.square(signal)))) if signal.size else 0.0
        if rms <= 0:
            return signal
        return signal * (10.0 ** (target_dbfs / 20.0)) / rms
    raise ValueError(f"未知的归一化模式: {mode!r}（可选 'peak' / 'rms' / 'none'）")


# ---------------------------------------------------------------------------
# 奈奎斯特钳制
# ---------------------------------------------------------------------------

def clamp_spec_to_nyquist(spec: "SignalSpec") -> Tuple["SignalSpec", List[str]]:
    """把超出奈奎斯特频率的频率参数钳制下来。

    **为什么必须钳制**：离散采样只能还原 ``[0, fs/2)`` 内的频率。请求一个高于
    奈奎斯特频率的频率（比如 fs=8 kHz 时要 20 kHz），采样得到的是它对
    ``fs`` 取模后的折叠频率 —— 听起来完全是另一个音，这正是"扫频信号不对"
    的最常见原因。

    返回 ``(钳制后的 spec, 说明列表)``；说明列表可直接展示给用户。
    """
    limit = spec.f_limit
    nyquist = spec.fs / 2.0
    notes: List[str] = []
    updates: Dict[str, Any] = {}

    def _clamp(name: str, value: float, label: str) -> float:
        if not np.isfinite(value) or value <= 0.0:
            new = 1.0
            updates[name] = new
            notes.append(f"{label} {value!r} 非法，已置为 {new:g} Hz")
            return new
        if value > limit:
            updates[name] = limit
            notes.append(
                f"{label} {value:g} Hz 超出奈奎斯特频率 {nyquist:g} Hz，"
                f"已钳制到 {limit:.1f} Hz（否则会混叠成别的频率）"
            )
            return limit
        return value

    if spec.kind == "sine":
        # 单频信号：f_start 就是信号频率本身；f_stop 无意义，不钳制也不参与交换
        f1 = _clamp("f1", spec.f1, "基频")
        updates["f1"] = f1
        updates["f_start"] = f1
        return spec.with_(**updates), notes

    f_start = _clamp("f_start", spec.f_start, "起始频率")
    f_stop = _clamp("f_stop", spec.f_stop, "终止频率")
    if f_start > f_stop:
        f_start, f_stop = f_stop, f_start
        notes.append("起始频率大于终止频率，已自动交换")
    updates["f_start"] = f_start
    updates["f_stop"] = f_stop

    notes.extend(_standard_band_notes(spec.kind, f_start, f_stop))

    return spec.with_(**updates), notes


#: 频带范围由标准固定下来的信号类型 —— 一旦改动就不再是标准信号。
#:
#: * ``eia426b``  —— EIA RS-426-B 寿命测试噪声固定限带 40 Hz – 10 kHz
#:   （40 Hz 高通至少 6 阶、10 kHz 低通至少 4 阶）；
#: * ``iec268`` / ``iec268_fig2`` —— Clause 7 的 Table II 覆盖
#:   20 Hz – 20 kHz 共 31 个 1/3 倍频程频带；
#: * ``iec268_pink`` —— 滤波器的 -3 dB 拐点由 IEC 60268-1 / IEC 60065
#:   规定（22.7 Hz / 22.4 kHz）。
_STANDARD_FIXED_BAND_KINDS = (
    "eia426b",
    "iec268",
    "iec268_fig2",
    "iec268_pink",
)


def _standard_band_notes(kind: str, f_start: float, f_stop: float) -> List[str]:
    """标准信号的频带偏离标准值时给出一条提示。"""
    if kind not in _STANDARD_FIXED_BAND_KINDS:
        return []
    defaults = DEFAULT_PARAMS.get(kind, {})
    diffs = []
    for key, label, value in (("f_start", "起始频率", f_start),
                              ("f_stop", "终止频率", f_stop)):
        std = defaults.get(key)
        if std is None:
            continue
        if not math.isclose(float(std), value, rel_tol=1e-9):
            diffs.append(f"{label} {float(std):g} → {value:g} Hz")
    if not diffs:
        return []
    return [
        f"频带已偏离 {KIND_LABELS[kind]} 的标准值（{'、'.join(diffs)}），"
        f"结果不再是严格合规的标准信号"
    ]


# ---------------------------------------------------------------------------
# 生成入口
# ---------------------------------------------------------------------------

def generate_signal(spec: SignalSpec) -> np.ndarray:
    """按规格生成信号（float64，未做量化）。

    频率参数会先被钳制到奈奎斯特频率以内，避免混叠。
    """
    spec, _notes = clamp_spec_to_nyquist(spec)

    n_samples = spec.n_samples
    if n_samples > MAX_SAMPLES:
        raise ValueError(
            f"请求的样本数 {n_samples:,} 超过上限 {MAX_SAMPLES:,}（约 "
            f"{MAX_SAMPLES / 48000:.0f} s @ 48 kHz），请缩短时长或减少台阶数"
        )

    kind = spec.kind
    if kind == "white":
        raw = _gen_white(spec)
    elif kind in ("pink", "red", "blue"):
        raw = _gen_colored_noise(spec)
    elif kind == "multitone":
        raw = _gen_multitone(spec)
    elif kind == "sweep":
        raw = _gen_sweep(spec, method=spec.sweep_method)
    elif kind == "stepped":
        raw = _gen_stepped_sweep(spec)
    elif kind == "sine":
        raw = _gen_sine(spec)
    elif kind in STANDARD_KINDS:
        raw = _gen_standard_noise(spec)
    else:
        raise ValueError(f"未知的信号类型: {kind!r}（可用: {', '.join(ALL_KINDS)}）")

    return normalize(raw, spec.normalize, spec.target_dbfs)


# 兼容旧名
generate = generate_signal


# ---------------------------------------------------------------------------
# 信号类型注册表（供 CLI 与 GUI 共用）
# ---------------------------------------------------------------------------

SIGNAL_KINDS: List[str] = [
    "white",
    "pink",
    "red",
    "blue",
    "multitone",
    "sweep",
    "stepped",
    "sine",
    "eia426b",
    "iec268",
    "iec268_fig2",
    "iec268_pink",
]
ALL_KINDS = SIGNAL_KINDS
NOISE_KINDS = ["white", "pink", "red", "blue"]
SWEEP_KINDS = ["sweep", "stepped"]
#: 标准测试信号（扬声器功率承载 / 最佳功放功率测试用）
STANDARD_KINDS = ["eia426b", "iec268", "iec268_fig2", "iec268_pink"]

#: 信号类型的**英文名**（业界通用命名，GUI 下拉框与 CLI 列表都显示它）
KIND_LABELS: Dict[str, str] = {
    "white": "White Noise",
    "pink": "Pink Noise",
    "red": "Red Noise (Brownian)",
    "blue": "Blue Noise",
    "multitone": "Multitone",
    "sweep": "Sweep (Chirp)",
    "stepped": "Stepped Sweep",
    "sine": "Sine Tone",
    "eia426b": "EIA RS-426-B Spectral-Weighted Noise",
    "iec268": "IEC 60268-1 Clause 7 Simulated Programme Signal",
    "iec268_fig2": "IEC 60268-1 Programme Signal (Figure 2 Circuit)",
    "iec268_pink": "IEC 60268-1 Annex C Band-Limited Pink Noise",
}

#: 信号类型的**中文说明**（CLI 列表第二列、GUI 提示用）
KIND_DESCRIPTIONS: Dict[str, str] = {
    "white": "白噪声 —— 功率谱平坦",
    "pink": "粉噪声 —— 功率谱按 1/f 下降",
    "red": "红噪声（布朗噪声）—— 功率谱按 1/f² 下降",
    "blue": "蓝噪声 —— 功率谱按 f 上升",
    "multitone": "多音信号 —— 对数分布的多组正弦，随机相位",
    "sweep": "连续扫频（chirp）—— 频率连续扫描",
    "stepped": "步进扫频 —— 等比分布的频率台阶，幅度恒定、相位连续",
    "sine": "单频正弦 —— 可叠加谐波",
    "eia426b": "EIA RS-426-B 频谱加权噪声（40 Hz–10 kHz，1 kHz 以上 −3 dB/oct）",
    "iec268": "IEC 60268-1 Clause 7 模拟节目信号（Table II 逐带精确，不限幅）",
    "iec268_fig2": "IEC 60268-1 Clause 7 模拟节目信号（Figure 2 无源 RC 电路实现）",
    "iec268_pink": "IEC 60268-1 附录 C 带限粉噪（IEC 60268-5 功率测试，峰值因子 2:1）",
}

#: 每种信号类型的默认参数
#:
#: ``f_start`` / ``f_stop`` 对**所有**类型都可用，但含义按类型不同 ——
#: 详见 :data:`FREQ_PARAM_INFO`。
DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
    # 宽带噪声：f_start / f_stop = 带限范围（频域砖墙 + 1/24 倍频程过渡带）
    "white": {"beta": 0.0, "f_start": 20.0, "f_stop": 20000.0},
    "pink": {"beta": 1.0, "f_start": 20.0, "f_stop": 20000.0},
    "red": {"beta": 2.0, "f_start": 20.0, "f_stop": 20000.0},
    "blue": {"beta": -1.0, "f_start": 20.0, "f_stop": 20000.0},
    # 多音：f_start / f_stop = 最低 / 最高音调
    "multitone": {"n_tones": 200, "f_start": 20.0, "f_stop": 20000.0},
    # 连续扫频：f_start / f_stop = 扫频起点 / 终点
    "sweep": {"sweep_method": "log", "fade_ratio": 0.05,
              "f_start": 20.0, "f_stop": 20000.0},
    # 步进扫频：f_start / f_stop = 台阶起点 / 终点
    "stepped": {"n_steps": 50, "step_duration": 0.5, "fade_ratio": 0.25,
                "f_start": 20.0, "f_stop": 20000.0},
    # 单频正弦：没有终止频率，f_start 就是信号频率
    "sine": {"f1": _DEFAULT_F1, "amplitude": 1.0, "f_start": _DEFAULT_F1,
             "f_stop": _DEFAULT_F_STOP},
    # EIA RS-426-B：寿命测试噪声的频带固定为 40 Hz – 10 kHz，
    # f_start / f_stop 就是两个 Butterworth 的 -3 dB 拐点
    "eia426b": {"f_start": EIA426B_HP_FC, "f_stop": EIA426B_LP_FC,
                "crest_factor_db": EIA426B_CREST_DB},
    # IEC 60268-1 Clause 7 模拟节目信号：f_start / f_stop 限定"纳入整形的
    # 1/3 倍频程频带范围"，默认就是 Table II 的全部 31 个频带（20 Hz – 20 kHz）。
    # 峰值因子默认 0 —— Clause 7 要求"不做幅度限制"。
    "iec268": {"f_start": 20.0, "f_stop": 20000.0,
               "crest_factor_db": IEC268_CREST_DB},
    # 同一信号的 Figure 2 电路实现
    "iec268_fig2": {"f_start": 20.0, "f_stop": 20000.0,
                    "crest_factor_db": IEC268_CREST_DB},
    # IEC 60268-1 附录 C / IEC 60065 带限粉噪：f_start / f_stop 直接作为
    # Butterworth 的 -3 dB 拐点；IEC 60268-5 的功率测试要求峰值因子 2:1
    "iec268_pink": {"f_start": IEC268_HP_FC, "f_stop": IEC268_LP_FC,
                    "crest_factor_db": IEC268_PINK_CREST_DB},
}


#: 各信号类型下 ``f_start`` / ``f_stop`` 的**标签与含义**，供 GUI 动态显示。
#:
#: 结构：``{类型: {参数名: (控件标签, 提示说明)}}``。
#: 参数名不在字典里时回落到 :data:`PARAM_SPECS` 的通用标签。
FREQ_PARAM_INFO: Dict[str, Dict[str, Tuple[str, str]]] = {
    "white": {
        "f_start": ("带限下限 (Hz)", "噪声频带下限：低于此频率的内容被完全抑制"),
        "f_stop": ("带限上限 (Hz)", "噪声频带上限：高于此频率的内容被完全抑制"),
    },
    "pink": {
        "f_start": ("带限下限 (Hz)", "噪声频带下限：低于此频率的内容被完全抑制"),
        "f_stop": ("带限上限 (Hz)", "噪声频带上限：高于此频率的内容被完全抑制"),
    },
    "red": {
        "f_start": ("带限下限 (Hz)", "噪声频带下限：低于此频率的内容被完全抑制"),
        "f_stop": ("带限上限 (Hz)", "噪声频带上限：高于此频率的内容被完全抑制"),
    },
    "blue": {
        "f_start": ("带限下限 (Hz)", "噪声频带下限：低于此频率的内容被完全抑制"),
        "f_stop": ("带限上限 (Hz)", "噪声频带上限：高于此频率的内容被完全抑制"),
    },
    "multitone": {
        "f_start": ("最低音调 (Hz)", "多音集合里最低的那个音"),
        "f_stop": ("最高音调 (Hz)", "多音集合里最高的那个音"),
    },
    "sweep": {
        "f_start": ("扫频起点 (Hz)", "扫频开始的频率"),
        "f_stop": ("扫频终点 (Hz)", "扫频结束的频率"),
    },
    "stepped": {
        "f_start": ("台阶起点 (Hz)", "第一个台阶的频率"),
        "f_stop": ("台阶终点 (Hz)", "最后一个台阶的频率"),
    },
    "sine": {
        "f_start": ("信号频率 (Hz)", "单频正弦的频率（单频信号没有终止频率）"),
        "f_stop": ("终止频率 (Hz)", "单频信号只有一个频率，此项不适用"),
    },
    "eia426b": {
        "f_start": ("高通拐点 (Hz)", "40 Hz Butterworth 高通的 -3 dB 频率（标准值 40，至少 6 阶）"),
        "f_stop": ("低通拐点 (Hz)", "10 kHz Butterworth 低通的 -3 dB 频率（标准值 10000，至少 4 阶）"),
    },
    "iec268": {
        "f_start": ("最低频带 (Hz)", "纳入整形的最低 1/3 倍频程频带（标准值 20）"),
        "f_stop": ("最高频带 (Hz)", "纳入整形的最高 1/3 倍频程频带（标准值 20000）"),
    },
    "iec268_fig2": {
        "f_start": ("最低频带 (Hz)", "纳入整形的最低 1/3 倍频程频带（标准值 20）"),
        "f_stop": ("最高频带 (Hz)", "纳入整形的最高 1/3 倍频程频带（标准值 20000）"),
    },
    "iec268_pink": {
        "f_start": ("高通拐点 (Hz)", "Butterworth 高通 -3 dB 频率（标准值 22.7）"),
        "f_stop": ("低通拐点 (Hz)", "Butterworth 低通 -3 dB 频率（标准值 22400）"),
    },
}


def default_params_for(kind: str) -> Dict[str, Any]:
    """返回某种信号类型的默认参数字典（副本，修改不会影响全局）。"""
    return dict(DEFAULT_PARAMS.get(kind, {}))

#: 参数的元信息，供 GUI 自动构建表单
PARAM_SPECS: Dict[str, Dict[str, Any]] = {
    "duration": {"label": "时长 (s)", "type": "float", "min": 0.05, "max": 600.0,
                 "step": 0.1, "default": 5.0},
    "fs": {"label": "采样率 (Hz)", "type": "int", "choices": [8000, 16000, 22050, 32000,
                                                            44100, 48000, 96000, 192000],
           "default": 48000},
    "seed": {"label": "随机种子", "type": "int", "min": 0, "max": 2**31 - 1, "default": 0},
    "f_start": {"label": "起始频率 (Hz)", "type": "float", "min": 1.0, "max": 20000.0,
                "step": 1.0, "default": 20.0,
                "hint": "含义随信号类型而变：噪声=带限下限、多音=最低音调、"
                        "扫频=起点、单频=信号频率、标准信号=频带下限"},
    "f_stop": {"label": "终止频率 (Hz)", "type": "float", "min": 20.0, "max": 96000.0,
               "step": 10.0, "default": 20000.0,
               "hint": "含义随信号类型而变：噪声=带限上限、多音=最高音调、"
                       "扫频=终点、标准信号=频带上限；单频正弦不适用"},
    "beta": {"label": "谱斜率 β (S∝f^-β)", "type": "float", "min": -2.0, "max": 4.0,
             "step": 0.1, "default": 1.0},
    "n_tones": {"label": "音调数量", "type": "int", "min": 1, "max": 2000,
                "default": 200},
    "n_steps": {"label": "台阶数", "type": "int", "min": 1, "max": 2000, "default": 50},
    "step_duration": {"label": "每步时长 (s)", "type": "float", "min": 0.01, "max": 60.0,
                      "step": 0.05, "default": 0.5},
    "fade_ratio": {"label": "渐变比例", "type": "float", "min": 0.0, "max": 1.0,
                   "step": 0.05, "default": 0.25,
                   "hint": "连续扫频按总时长计；步进扫频按每步时长计（只作用于整段首尾）"},
    "f1": {"label": "基频 (Hz)", "type": "float", "min": 1.0, "max": 96000.0,
           "step": 1.0, "default": 1000.0},
    "amplitude": {"label": "幅度", "type": "float", "min": 0.0, "max": 1.0,
                  "step": 0.05, "default": 1.0},
    "target_dbfs": {"label": "目标电平 (dBFS)", "type": "float", "min": -60.0, "max": 0.0,
                    "step": 0.5, "default": -3.0},
    "crest_factor_db": {"label": "峰值因子 (dB)", "type": "float", "min": 0.0, "max": 24.0,
                        "step": 0.5, "default": 6.0,
                        "hint": "0 = 不限幅；EIA RS-426-B / IEC 60268-5 要求 6 dB（2:1）"},
    # 谐波幅度是一列文本（"0.5,0.25"），不是单个数值，所以没有 min/max
    "harmonics": {"label": "谐波幅度", "type": "text", "default": "",
                  "hint": "逗号分隔的相对幅度列表，如 0.5,0.25 表示 2 次谐波 0.5、"
                          "3 次谐波 0.25（仅单频正弦有效）"},
}

#: 各类型实际用到的参数（GUI 据此启用/禁用控件）
#:
#: ``f_start`` / ``f_stop`` 对所有类型都在列（含义见 :data:`FREQ_PARAM_INFO`），
#: 单频正弦例外 —— 它只有一个频率，所以没有 ``f_stop``。
KIND_PARAMS: Dict[str, List[str]] = {
    "white": ["duration", "fs", "seed", "f_start", "f_stop"],
    "pink": ["duration", "fs", "seed", "f_start", "f_stop", "beta"],
    "red": ["duration", "fs", "seed", "f_start", "f_stop", "beta"],
    "blue": ["duration", "fs", "seed", "f_start", "f_stop", "beta"],
    "multitone": ["duration", "fs", "seed", "f_start", "f_stop", "n_tones"],
    "sweep": ["duration", "fs", "seed", "f_start", "f_stop", "sweep_method", "fade_ratio"],
    "stepped": ["fs", "seed", "f_start", "f_stop", "n_steps", "step_duration", "fade_ratio"],
    "sine": ["duration", "fs", "seed", "f_start", "amplitude", "harmonics"],
    "eia426b": ["duration", "fs", "seed", "f_start", "f_stop", "crest_factor_db"],
    "iec268": ["duration", "fs", "seed", "f_start", "f_stop", "crest_factor_db"],
    "iec268_fig2": ["duration", "fs", "seed", "f_start", "f_stop", "crest_factor_db"],
    "iec268_pink": ["duration", "fs", "seed", "f_start", "f_stop", "crest_factor_db"],
}
