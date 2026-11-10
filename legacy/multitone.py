"""
多音信号生成（薄封装）。

核心实现已移至 :mod:`signal_gen.core`，本文件仅为兼容旧用法保留。
推荐直接使用::

    python -m signal_gen generate multitone --n-tones 200 --seed 42
"""

from __future__ import annotations

import sys
from pathlib import Path

# 归档脚本：把项目根目录加入 sys.path，保证直接运行本文件也能 import 到 signal_gen
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from signal_gen.core import SignalSpec, generate_signal
from signal_gen.spectrum import band_levels_dbfs, one_sided_power_spectrum, spectrum_dbfs


def generate_multitone_signal(
    frequencies, sampling_rate: int, duration: float
) -> tuple[np.ndarray, np.ndarray]:
    """生成多音调信号（保留旧函数签名）。

    修正：旧版时间轴用 ``linspace(0, duration, n)``（含尾端点），采样间隔
    变成 ``duration/(n-1)`` 而非 ``1/fs``，导致所有音调频率整体偏移
    （20 kHz 处实测 +0.083 Hz）。现统一走 :func:`signal_gen.core._time_axis`。
    """
    n = int(round(sampling_rate * duration))
    t = np.arange(n, dtype=np.float64) / float(sampling_rate)
    signal = np.zeros(n, dtype=np.float64)
    rng = np.random.default_rng(0)
    for f in frequencies:
        signal += np.sin(2 * np.pi * f * t + rng.uniform(-np.pi, np.pi))
    return t, signal


def main() -> None:
    import matplotlib.pyplot as plt

    spec = SignalSpec(kind="multitone", n_tones=200, duration=5.0, fs=48000, seed=0)
    signal = generate_signal(spec)

    freqs, db = spectrum_dbfs(signal, spec.fs, window="blackmanharris")
    _, power = one_sided_power_spectrum(signal, spec.fs, window="blackmanharris")
    centers, bands = band_levels_dbfs(freqs, power)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    t = np.arange(signal.size) / spec.fs
    axes[0].plot(t, signal, lw=0.4, color="orange")
    axes[0].set_title("Multitone — Time Domain")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, which="both", ls="--", lw=0.5)

    axes[1].plot(freqs, db, lw=0.5, alpha=0.6, label="original spectrum")
    axes[1].plot(centers, bands, lw=1.4, color="red", label="1/6 octave (power mean)")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Level (dBFS)")
    axes[1].set_title("Multitone — Frequency Domain")
    axes[1].grid(True, which="both", ls="--", lw=0.5)
    axes[1].legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
