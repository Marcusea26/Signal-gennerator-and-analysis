"""
归档：早期"读取信号文件 → 分析 → 画图"的独立脚本。

本脚本的能力已**并入 :mod:`signal_gen.report`**，请改用::

    # 命令行
    python -m signal_gen analyze 某文件.wav --plot report.png

    # 库 API
    from signal_gen import analyze_file, build_report_figure

并入时修掉了旧实现的三处问题：

1. **1/6 倍频程口径不对** —— 旧版是"以每个 FFT 频点为中心、对**幅度谱**做
   算术平均"的滑动窗，读数比标准口径（IEC 61260 的**功率积分**）低
   36 ~ 38 dB；现版用 :func:`signal_gen.spectrum.one_sixth_octave_smoothing`，
   满幅正弦读 0 dBFS、粉噪曲线平直。
2. **性能差 500 倍** —— 旧版对每个频点做一次 Python 循环 + ``searchsorted``，
   在 300 s / 48 kHz 的文件上实测 23.5 s（720 万次循环）；现版毫秒级。
3. **依赖 soundfile 才能跑** —— 旧版必须装 soundfile；现版 WAV 走自研的
   RIFF 解析器（零依赖，还支持 IEEE 浮点 WAV），其它格式才可选 soundfile。

另外旧版是**硬编码文件路径**且会把波形先归一化到峰值 1.0（丢掉了绝对刻度），
现版保持 dBFS 绝对刻度，读数与生成信号可直接比较。

本文件仅作历史参考保留，不参与测试。
"""

import sys
from pathlib import Path

# 归档脚本：把项目根目录加入 sys.path，保证直接运行本文件也能 import 到 signal_gen
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import soundfile as sf  # noqa: E402


def load_audio(file_path):
    signal, sr = sf.read(file_path)

    # 如果是多通道音频，转为单通道
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)

    return signal, sr


def one_sixth_octave_smoothing_fast(frequencies, spectrum):
    """
    加速版 1/6 倍频程平滑：
    - 频点数保持不变
    - 频率轴保持不变
    - 计算逻辑不变：对每个频点，以其为中心做 1/6 倍频程窗口平均
    - 通过前缀和 + 搜索边界加速

    参数:
    frequencies : 递增的一维 numpy 数组
    spectrum    : 对应幅值数组

    返回:
    smoothed_spectrum : 与输入长度相同
    """
    frequencies = np.asarray(frequencies)
    spectrum = np.asarray(spectrum)

    n = len(frequencies)
    smoothed_spectrum = np.empty(n, dtype=np.float64)

    # 前缀和，prefix_sum[i] 表示 spectrum[:i] 的和
    prefix_sum = np.zeros(n + 1, dtype=np.float64)
    prefix_sum[1:] = np.cumsum(spectrum, dtype=np.float64)

    # 对每个频点进行平滑
    # 利用 np.searchsorted 快速找到左右边界索引
    factor_low = 2 ** (1 / 12)
    factor_high = 2 ** (1 / 12)

    for i, fc in enumerate(frequencies):
        f_low = fc / factor_low
        f_high = fc * factor_high

        left = np.searchsorted(frequencies, f_low, side='left')
        right = np.searchsorted(frequencies, f_high, side='right') - 1

        if left <= right:
            total = prefix_sum[right + 1] - prefix_sum[left]
            count = right - left + 1
            smoothed_spectrum[i] = total / count
        else:
            smoothed_spectrum[i] = spectrum[i]

    return smoothed_spectrum


def main():
    audio_file_path = r'E:\Projects\D242\EIA-426B\EIA-426B 40-200Hz CF 6dB.wav'

    # 加载音频
    signal, sr = load_audio(audio_file_path)
    print("采样率:", sr)

    signal = signal.astype(np.float64)

    # 归一化
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val

    duration = len(signal) / sr
    t = np.linspace(0, duration, len(signal), endpoint=False)

    # 峰值因子
    rms = np.sqrt(np.mean(signal ** 2))
    peak = np.max(np.abs(signal))
    CF = 20 * np.log10(peak / rms) if rms > 0 else np.inf
    print("CF =", CF)

    # FFT
    fft_signal = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(len(signal), 1 / sr)

    # 只保留正频率部分
    positive_indices = np.where(fft_freq > 0)
    fft_freq = fft_freq[positive_indices]
    fft_mag = np.abs(fft_signal[positive_indices])

    print("原始频点数:", len(fft_freq))

    # 加速版 1/6 倍频程平滑
    smoothed_spectrum = one_sixth_octave_smoothing_fast(fft_freq, fft_mag)

    print("平滑后频点数:", len(smoothed_spectrum))

    # 防止 log10(0)
    eps = 1e-12

    # 绘图
    plt.figure(figsize=(12, 6))

    # 时间域
    plt.subplot(2, 1, 1)
    plt.plot(t, signal, label='Signal', color='orange')
    plt.xlim(0, duration)
    plt.title('Signal in Time Domain')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper left')
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # 频域
    plt.subplot(2, 1, 2)
    plt.plot(fft_freq, 20 * np.log10(fft_mag + eps), label="original spectrum", alpha=0.5)
    plt.plot(fft_freq, 20 * np.log10(smoothed_spectrum + eps), label="1/6 octave smooth (fast)", color="red")

    plt.xscale("log")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.title('Signal in Frequency Domain')
    plt.xlim(20, 20000)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
