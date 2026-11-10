"""
粉噪声生成（薄封装）。

核心实现已移至 :mod:`signal_gen.core`，本文件仅为兼容旧用法保留。
推荐直接使用::

    python -m signal_gen generate pink --duration 5 --seed 42 --normalize rms --target-dbfs -18
"""

from __future__ import annotations

import sys
from pathlib import Path

# 归档脚本：把项目根目录加入 sys.path，保证直接运行本文件也能 import 到 signal_gen
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from signal_gen.core import SignalSpec, generate_signal
from signal_gen.spectrum import band_levels_dbfs, one_sided_power_spectrum, spectrum_dbfs


def generate_pink_noise(length: int, beta: float = 1.0) -> np.ndarray:
    """生成粉噪声（1/f 功率谱）。

    参数
    ----
    length
        样本数。
    beta
        谱斜率，S(f) ∝ f^-beta。beta=1 为粉噪声，beta=2 为红噪声。

    注：旧版签名为 ``hurst``，语义实为 ``f**(-hurst)`` 的幅度指数，
    等价于 beta = 2*hurst。为避免混淆，本版统一使用 beta。
    """
    spec = SignalSpec(
        kind="pink", duration=length / 48000.0, beta=beta, normalize="none"
    )
    return generate_signal(spec)


def main() -> None:
    import matplotlib.pyplot as plt

    spec = SignalSpec(kind="pink", duration=5.0, fs=48000, seed=0)
    signal = generate_signal(spec)

    freqs, db = spectrum_dbfs(signal, spec.fs, window="blackmanharris")
    _, power = one_sided_power_spectrum(signal, spec.fs, window="blackmanharris")
    centers, bands = band_levels_dbfs(freqs, power)

    # 拟合 -3 dB/oct 斜率
    mask = (centers >= 100) & (centers <= 10000)
    slope = np.polyfit(np.log2(centers[mask]), bands[mask], 1)[0]
    print(f"Pink noise slope = {slope:+.3f} dB/octave (理论 -3.010)")

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    t = np.arange(signal.size) / spec.fs
    axes[0].plot(t, signal, lw=0.4, color="orange")
    axes[0].set_title("Pink Noise — Time Domain")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, which="both", ls="--", lw=0.5)

    axes[1].plot(freqs, db, lw=0.5, alpha=0.6, label="original spectrum")
    axes[1].plot(centers, bands, lw=1.4, color="red", label="1/6 octave (power mean)")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Level (dBFS)")
    axes[1].set_title("Pink Noise — Frequency Domain")
    axes[1].grid(True, which="both", ls="--", lw=0.5)
    axes[1].legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
