"""
扫频信号生成（薄封装）。

核心实现已移至 :mod:`signal_gen.core`，本文件仅为兼容旧用法保留。
推荐直接使用::

    python -m signal_gen generate sweep --sweep-method log --duration 2 --seed 42
    python -m signal_gen generate stepped --n-steps 50 --step-duration 0.5
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

    修正：旧版每步用 ``np.arange(0, duration, 1/fs)``，浮点误差会导致样本数
    多出 1 个（实测 100 个时长中有 20 个中招），总时长随之漂移。
    现改为 ``arange(n)/fs`` 精确取样。

    另：频率超出奈奎斯特频率时会被钳制，避免混叠成别的频率。
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
        kind="sweep", sweep_method="log", duration=2.0, fs=48000, seed=0
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
    axes[0].set_title("Log Sweep — Time Domain")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, which="both", ls="--", lw=0.5)

    axes[1].plot(freqs, db, lw=0.5, alpha=0.6, label="original spectrum")
    axes[1].plot(centers, bands, lw=1.4, color="red", label="1/6 octave (power mean)")
    axes[1].set_xscale("log")
    axes[1].set_xlim(20, 20000)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Level (dBFS)")
    axes[1].set_title("Log Sweep — Frequency Domain")
    axes[1].grid(True, which="both", ls="--", lw=0.5)
    axes[1].legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
