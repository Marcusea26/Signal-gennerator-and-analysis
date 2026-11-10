"""
步进扫频信号生成（薄封装）。

核心实现已移至 :mod:`signal_gen.core`，本文件仅为兼容旧用法保留。
推荐直接使用::

    python -m signal_gen generate stepped --n-steps 50 --step-duration 0.5 --seed 42
"""

from __future__ import annotations

import sys
from pathlib import Path

# 归档脚本：把项目根目录加入 sys.path，保证直接运行本文件也能 import 到 signal_gen
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from signal_gen.core import SignalSpec, generate_signal
from signal_gen.spectrum import band_levels_dbfs, one_sided_spectrum_dbfs_power


def generate_stepped_chirp(
    f_start: float,
    f_stop: float,
    points: int,
    duration: float,
    sampling_rate: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """生成步进扫频信号（保留旧函数签名）。

    关键修正：改为 **幅度恒定 + 相位连续** 的教科书式 stepped sweep。

    旧版给每个台阶加 Tukey 渐变，实测有两个严重缺陷：

    1. 每步都淡入淡出到 0，产生 2.6% 接近静音的样点，信号听起来是"断续"的；
    2. 低频台阶的渐变段覆盖不到一个完整周期（20 Hz 台阶只有 1.25 个周期），
       台阶严重失真 —— 低频台阶纯度只有 44 dB；改为相位连续后提升到 108 dB。

    另外，频率超出奈奎斯特频率时会被钳制，避免混叠成别的频率。
    """
    spec = SignalSpec(
        kind="stepped",
        f_start=f_start,
        f_stop=f_stop,
        n_steps=points,
        step_duration=duration,
        fs=sampling_rate,
        normalize="none",
    )
    signal = generate_signal(spec)
    t = np.arange(signal.size, dtype=np.float64) / float(sampling_rate)
    return t, signal, signal.size / float(sampling_rate)


def main() -> None:
    import matplotlib.pyplot as plt

    spec = SignalSpec(
        kind="stepped", n_steps=50, step_duration=0.5, fs=48000, seed=0
    )
    signal = generate_signal(spec)

    # 一次 FFT 同时拿到幅度谱与功率谱
    freqs, db, power = one_sided_spectrum_dbfs_power(
        signal, spec.fs, window="blackmanharris"
    )
    centers, bands = band_levels_dbfs(freqs, power)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    t = np.arange(signal.size) / spec.fs
    axes[0].plot(t, signal, lw=0.4, color="orange")
    axes[0].set_title("Stepped Sweep — Time Domain")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, which="both", ls="--", lw=0.5)

    axes[1].plot(freqs, db, lw=0.5, alpha=0.6, label="original spectrum")
    axes[1].plot(centers, bands, lw=1.4, color="red", label="1/6 octave (power mean)")
    axes[1].set_xscale("log")
    axes[1].set_xlim(20, 20000)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Level (dBFS)")
    axes[1].set_title("Stepped Sweep — Frequency Domain")
    axes[1].grid(True, which="both", ls="--", lw=0.5)
    axes[1].legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
