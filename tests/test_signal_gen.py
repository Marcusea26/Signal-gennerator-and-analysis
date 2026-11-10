"""
signal_gen 测试套件（不依赖 pytest，直接用 unittest 运行）。

运行::

    python -m unittest discover -s tests -v
    python run_tests.py
"""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signal_gen.audio_io import quantize_int16, quantize_int24, save_wav, signal_stats
from signal_gen.core import (
    ALL_KINDS,
    DEFAULT_PARAMS,
    FREQ_PARAM_INFO,
    KIND_LABELS,
    KIND_PARAMS,
    MAX_SAMPLES,
    STANDARD_KINDS,
    SignalSpec,
    clamp_spec_to_nyquist,
    generate_signal,
    tukey_window,
)
from signal_gen.standards import (
    EIA426B_CAL_TONE_PEAK_DBFS,
    EIA426B_CREST_DB,
    EIA426B_HP_FC,
    EIA426B_HP_ORDER,
    EIA426B_HP_ROLLOFF_DB_OCT,
    EIA426B_LP_FC,
    EIA426B_LP_ORDER,
    EIA426B_LP_ROLLOFF_DB_OCT,
    EIA426B_SOFT_CLIP_K,
    EIA426B_WEIGHT_KNEE_HZ,
    IEC268_HP_FC,
    IEC268_HP_ORDER,
    IEC268_HP_ROLLOFF_DB_OCT,
    IEC268_LP_FC,
    IEC268_LP_ORDER,
    IEC268_LP_ROLLOFF_DB_OCT,
    IEC268_PROGRAM_CENTERS,
    IEC268_PROGRAM_EXACT_CENTERS,
    IEC268_PROGRAM_LEVELS,
    IEC268_PROGRAM_TABLE,
    IEC268_PROGRAM_TOLERANCES,
    IEC268_PROGRAM_TOTAL_POWER_DB,
    butterworth_magnitude,
    check_iec268_program,
    crest_factor_db,
    eia426b_spectrum,
    iec268_figure2_response,
    iec268_lp_corner,
    iec268_program_density_db,
    iec268_program_psd,
    limit_crest_factor,
    soft_clip,
    soft_clip_to_crest,
    third_octave_band_power,
    third_octave_edges,
    third_octave_exact_center,
)
from signal_gen.spectrum import (
    WINDOWS,
    band_levels_dbfs,
    one_sided_amplitude_spectrum,
    one_sided_power_spectrum,
    one_sided_spectrum_dbfs_power,
    one_sixth_octave_smoothing,
    spectrum_dbfs,
    stft_dbfs,
)


class TestCore(unittest.TestCase):
    def test_time_axis_has_no_endpoint_error(self):
        """回归：multitone 原先缺 endpoint=False，导致频率整体偏移。"""
        spec = SignalSpec(kind="multitone", n_tones=1, f_start=1000.0,
                          f_stop=1000.0, duration=1.0, fs=48000,
                          normalize="none")
        x = generate_signal(spec)
        freqs, _ = one_sided_amplitude_spectrum(x, 48000, window="blackmanharris")
        peak = freqs[np.argmax(np.abs(np.fft.rfft(x)))]
        self.assertAlmostEqual(peak, 1000.0, delta=1.0)

    def test_multitone_peak_frequencies_exact(self):
        """多音的每个分量都应落在标称频率上。"""
        spec = SignalSpec(kind="multitone", n_tones=5, f_start=500.0, f_stop=4000.0,
                          duration=2.0, fs=48000, normalize="none")
        x = generate_signal(spec)
        mag = np.abs(np.fft.rfft(x))
        freqs = np.fft.rfftfreq(x.size, 1 / 48000)
        peaks = freqs[np.argsort(mag)[-5:]]
        expected = np.logspace(np.log10(500.0), np.log10(4000.0), 5)
        for p, e in zip(sorted(peaks), expected):
            self.assertAlmostEqual(p, e, delta=1.0)

    def test_pink_noise_slope_is_minus3db_per_octave(self):
        """粉噪功率谱密度斜率应为 -3.01 dB/octave。

        注意区分两个量：

        * **功率谱密度**（每 Hz 功率）随频率按 -3 dB/oct 下降；
        * **1/6 倍频程频带电平**（频带总功率）则是**平直**的 ——
          粉噪的定义性特征（见
          :meth:`test_pink_noise_octave_bands_are_flat`）。
        """
        spec = SignalSpec(kind="pink", duration=10.0, fs=48000, normalize="none")
        x = generate_signal(spec)
        freqs, power = one_sided_power_spectrum(x, 48000, window="hann")
        psd_db = 10.0 * np.log10(np.maximum(power, 1e-30))
        mask = (freqs >= 100) & (freqs <= 10000)
        slope = np.polyfit(np.log2(freqs[mask]), psd_db[mask], 1)[0]
        self.assertAlmostEqual(slope, -3.01, delta=0.25)

    def test_pink_noise_octave_bands_are_flat(self):
        """粉噪的 1/6 倍频程频带电平必须**平直**（每带功率相同）。"""
        spec = SignalSpec(kind="pink", duration=10.0, fs=48000, normalize="none")
        x = generate_signal(spec)
        freqs, power = one_sided_power_spectrum(x, 48000, window="hann")
        centers, levels = band_levels_dbfs(freqs, power)
        mask = (centers >= 100) & (centers <= 10000)
        slope = np.polyfit(np.log2(centers[mask]), levels[mask], 1)[0]
        self.assertAlmostEqual(slope, 0.0, delta=0.2,
                               msg="粉噪的倍频程曲线没有保持平直")

    def test_colored_noise_beta_mapping(self):
        """beta 与实际谱斜率一致：S ∝ f^-beta（看功率谱密度）。"""
        for kind, beta in (("white", 0.0), ("pink", 1.0), ("red", 2.0)):
            spec = SignalSpec(kind=kind, beta=beta, duration=10.0, fs=48000,
                              normalize="none")
            x = generate_signal(spec)
            freqs, power = one_sided_power_spectrum(x, 48000, window="hann")
            psd_db = 10.0 * np.log10(np.maximum(power, 1e-30))
            mask = (freqs >= 100) & (freqs <= 10000)
            slope = np.polyfit(np.log2(freqs[mask]), psd_db[mask], 1)[0]
            self.assertAlmostEqual(slope, -3.01 * beta, delta=0.3)

    def test_pink_noise_has_no_dc(self):
        spec = SignalSpec(kind="pink", duration=3.0, fs=48000, normalize="none")
        x = generate_signal(spec)
        self.assertLess(abs(float(np.mean(x))), 1e-12)

    def test_pink_noise_does_not_warn_on_divide_by_zero(self):
        """回归：旧版 np.power(0, -0.5) 触发 divide-by-zero 警告。"""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            generate_signal(SignalSpec(kind="pink", duration=1.0, fs=48000))

    def test_same_seed_is_reproducible(self):
        spec = SignalSpec(kind="pink", duration=1.0, seed=42)
        a = generate_signal(spec)
        b = generate_signal(spec)
        np.testing.assert_array_equal(a, b)

    def test_different_seed_differs(self):
        a = generate_signal(SignalSpec(kind="white", duration=1.0, seed=1))
        b = generate_signal(SignalSpec(kind="white", duration=1.0, seed=2))
        self.assertFalse(np.array_equal(a, b))

    def test_stepped_sweep_length(self):
        spec = SignalSpec(kind="stepped", n_steps=10, step_duration=0.5, fs=48000)
        x = generate_signal(spec)
        self.assertEqual(x.size, 10 * int(round(0.5 * 48000)))

    def test_tukey_window_shape(self):
        w = tukey_window(100, alpha=0.5)
        self.assertAlmostEqual(w[0], 0.0, places=6)
        self.assertAlmostEqual(w[-1], 0.0, places=6)
        self.assertAlmostEqual(w[50], 1.0, places=6)

    def test_normalize_rms_target(self):
        spec = SignalSpec(kind="white", normalize="rms", target_dbfs=-18.0, duration=2.0)
        x = generate_signal(spec)
        rms_db = 20 * np.log10(np.sqrt(np.mean(x ** 2)))
        self.assertAlmostEqual(rms_db, -18.0, delta=0.2)


class TestSweepSignals(unittest.TestCase):
    """扫频 / 步进扫频的正确性回归。"""

    FS = 48000

    def _step_purity(self, seg, f_target, fs):
        """台阶内目标频率的纯度 (dB)：目标 bin 能量 / 其余能量。"""
        n = seg.size
        w = np.hanning(n)
        mag = np.abs(np.fft.rfft(seg * w)) ** 2
        k = int(round(f_target * n / fs))
        lo, hi = max(0, k - 3), min(mag.size, k + 4)
        tone = mag[lo:hi].sum()
        other = mag.sum() - tone
        return 10 * np.log10(tone / other) if other > 0 else np.inf

    def _reference_purity(self, f, n, fs):
        """理想正弦在同样度量下的纯度，作为上界参考。"""
        t = np.arange(n) / fs
        return self._step_purity(np.sin(2 * np.pi * f * t), f, fs)

    # ---- 连续扫频 ----
    def test_continuous_sweep_frequency_trajectory_is_exact(self):
        """对数扫频的 log10(f(t)) 必须严格线性（频率轨迹精确）。"""
        spec = SignalSpec(kind="sweep", sweep_method="log", duration=5.0,
                          fs=self.FS, f_start=20.0, f_stop=20000.0, normalize="none")
        x = generate_signal(spec)
        # 用 STFT 轨迹反推频率，再检查 log10(f) 对时间的线性度
        from scipy.signal import spectrogram
        f_spec, t_spec, Sxx = spectrogram(x, fs=self.FS, nperseg=2048,
                                          noverlap=1536, scaling="spectrum",
                                          mode="magnitude")
        track = f_spec[np.argmax(Sxx, axis=0)]
        mid = track[20:-20]
        slope, intercept = np.polyfit(t_spec[20:-20], np.log10(mid), 1)
        resid = np.log10(mid) - (slope * t_spec[20:-20] + intercept)
        self.assertLess(float(np.std(resid)), 0.06,
                        msg="对数扫频的频率轨迹偏离直线，扫频实现有问题")

    def test_continuous_sweep_linear_method_is_not_log(self):
        """线性扫频的 log10(f(t)) 应当明显偏离直线（与对数扫频区分）。"""
        spec = SignalSpec(kind="sweep", sweep_method="linear", duration=5.0,
                          fs=self.FS, f_start=20.0, f_stop=20000.0, normalize="none")
        x = generate_signal(spec)
        from scipy.signal import spectrogram
        f_spec, t_spec, Sxx = spectrogram(x, fs=self.FS, nperseg=2048,
                                          noverlap=1536, scaling="spectrum",
                                          mode="magnitude")
        track = f_spec[np.argmax(Sxx, axis=0)]
        mid = track[20:-20]
        slope, intercept = np.polyfit(t_spec[20:-20], np.log10(mid), 1)
        resid = np.log10(mid) - (slope * t_spec[20:-20] + intercept)
        self.assertGreater(float(np.std(resid)), 0.06)

    # ---- 步进扫频 ----
    def test_stepped_sweep_phase_is_continuous(self):
        """回归：旧版每步重置相位 + Tukey 淡入淡出。

        现在应为「幅度恒定 + 相位连续」，中间台阶必须与解析解逐点一致。
        """
        n_steps, step_dur = 20, 0.4
        fs = self.FS
        spec = SignalSpec(kind="stepped", n_steps=n_steps, step_duration=step_dur,
                          fs=fs, f_start=20.0, f_stop=20000.0,
                          fade_ratio=0.0, normalize="none")
        x = generate_signal(spec)

        n_step = int(round(step_dur * fs))
        freqs = np.logspace(np.log10(20.0), np.log10(20000.0), n_steps)

        # 解析重建：相位跨台阶累加
        expected = np.empty_like(x)
        phase = 0.0
        t = np.arange(n_step) / fs
        for k, f in enumerate(freqs):
            ph = 2 * np.pi * f * t + phase
            expected[k * n_step:(k + 1) * n_step] = np.sin(ph)
            phase = float(ph[-1] + 2 * np.pi * f / fs)

        np.testing.assert_allclose(x, expected, atol=1e-10)

    def test_stepped_sweep_has_no_silent_gaps(self):
        """回归：旧版每步淡入淡出到 0，产生 2.6% 接近静音的样点。"""
        spec = SignalSpec(kind="stepped", n_steps=50, step_duration=0.5,
                          fs=self.FS, f_start=20.0, f_stop=20000.0,
                          fade_ratio=0.0, normalize="none")
        x = generate_signal(spec)
        near_zero = int(np.sum(np.abs(x) < 1e-6))
        ratio = near_zero / x.size
        self.assertLess(ratio, 0.0005,
                        msg=f"接近静音样点占比 {ratio:.4%}，旧版为 2.6%")

    def test_stepped_sweep_boundary_has_no_click(self):
        """台阶边界的跳变不得超过信号内部的正常跳变。"""
        spec = SignalSpec(kind="stepped", n_steps=50, step_duration=0.5,
                          fs=self.FS, f_start=20.0, f_stop=20000.0,
                          fade_ratio=0.0, normalize="none")
        x = generate_signal(spec)
        n_step = int(round(0.5 * self.FS))
        d = np.abs(np.diff(x))
        boundaries = np.array([(k + 1) * n_step - 1 for k in range(49)])
        interior = np.array([d[i] for i in range(1000, x.size - 1000, 997)])
        self.assertLessEqual(float(d[boundaries].max()), float(interior.max()) * 1.05,
                             msg="台阶边界存在跳变（咔哒声）")

    def test_stepped_sweep_low_frequency_step_is_pure(self):
        """回归：旧版 Tukey 让低频台阶严重失真（纯度仅 44 dB）。

        改为幅度恒定 + 相位连续后，每个台阶都应接近理想正弦的纯度。
        """
        n_steps, step_dur = 50, 0.5
        fs = self.FS
        spec = SignalSpec(kind="stepped", n_steps=n_steps, step_duration=step_dur,
                          fs=fs, f_start=20.0, f_stop=20000.0,
                          fade_ratio=0.0, normalize="none")
        x = generate_signal(spec)
        n_step = int(round(step_dur * fs))
        freqs = np.logspace(np.log10(20.0), np.log10(20000.0), n_steps)

        for k in (1, 6, 12, 24, 36, 48):
            seg = x[k * n_step:(k + 1) * n_step]
            purity = self._step_purity(seg, freqs[k], fs)
            ideal = self._reference_purity(freqs[k], n_step, fs)
            self.assertGreaterEqual(
                purity, ideal - 0.5,
                msg=f"台阶 {k} ({freqs[k]:.1f} Hz) 纯度 {purity:.2f} dB "
                    f"低于理想值 {ideal:.2f} dB",
            )

    def test_stepped_sweep_twenty_hz_step_purity_excellent(self):
        """20 Hz 台阶是 10 个整周期，纯度应达到 100 dB 量级。"""
        spec = SignalSpec(kind="stepped", n_steps=50, step_duration=0.5,
                          fs=self.FS, f_start=20.0, f_stop=20000.0,
                          fade_ratio=0.0, normalize="none")
        x = generate_signal(spec)
        n_step = int(round(0.5 * self.FS))
        purity = self._step_purity(x[:n_step], 20.0, self.FS)
        self.assertGreater(purity, 100.0, msg=f"20 Hz 台阶纯度仅 {purity:.1f} dB")

    def test_stepped_sweep_global_fade_only_touches_edges(self):
        """首尾渐变只作用于整段首尾，不应逐台阶产生静音间隙。"""
        spec = SignalSpec(kind="stepped", n_steps=10, step_duration=0.5,
                          fs=self.FS, f_start=20.0, f_stop=20000.0,
                          fade_ratio=0.25, normalize="none")
        x = generate_signal(spec)
        n_step = int(round(0.5 * self.FS))
        # 中间台阶的峰值应保持 1.0（不被淡入淡出削弱）
        middle = x[4 * n_step:6 * n_step]
        self.assertGreater(float(np.max(np.abs(middle))), 0.99)
        # 静音样点比例依然极低
        ratio = float(np.sum(np.abs(x) < 1e-6)) / x.size
        self.assertLess(ratio, 0.0005)


class TestNyquistClamp(unittest.TestCase):
    """频率超出奈奎斯特频率时必须钳制，否则会混叠成别的频率。"""

    def test_clamp_spec_reports_note(self):
        spec = SignalSpec(kind="sweep", fs=8000, f_start=20.0, f_stop=20000.0)
        clamped, notes = clamp_spec_to_nyquist(spec)
        self.assertLessEqual(clamped.f_stop, clamped.f_limit)
        self.assertTrue(any("奈奎斯特" in n for n in notes))

    def test_clamp_spec_swaps_reversed_range(self):
        spec = SignalSpec(kind="sweep", fs=48000, f_start=5000.0, f_stop=1000.0)
        clamped, notes = clamp_spec_to_nyquist(spec)
        self.assertLess(clamped.f_start, clamped.f_stop)
        self.assertTrue(any("交换" in n for n in notes))

    def test_clamp_spec_keeps_valid_range_untouched(self):
        spec = SignalSpec(kind="sweep", fs=48000, f_start=20.0, f_stop=20000.0)
        clamped, notes = clamp_spec_to_nyquist(spec)
        self.assertAlmostEqual(clamped.f_start, 20.0)
        self.assertAlmostEqual(clamped.f_stop, 20000.0)
        self.assertEqual(notes, [])

    def test_sweep_does_not_alias_beyond_nyquist(self):
        """回归：fs=8 kHz 却要 20 kHz，信号会折叠成完全不同的频率。"""
        spec = SignalSpec(kind="sweep", fs=8000, duration=2.0,
                          f_start=20.0, f_stop=20000.0, normalize="none")
        x = generate_signal(spec)
        freqs, db = spectrum_dbfs(x, 8000, window="blackmanharris")
        peak = freqs[int(np.argmax(db))]
        self.assertLessEqual(peak, 0.99 * 8000 / 2 + 1.0,
                             msg=f"频谱峰值落在 {peak:.1f} Hz，发生混叠")

    def test_multitone_does_not_alias_beyond_nyquist(self):
        spec = SignalSpec(kind="multitone", fs=8000, n_tones=10, duration=2.0,
                          f_start=20.0, f_stop=20000.0, normalize="none")
        x = generate_signal(spec)
        freqs, db = spectrum_dbfs(x, 8000, window="blackmanharris")
        self.assertLessEqual(float(freqs[int(np.argmax(db))]), 0.99 * 8000 / 2 + 1.0)

    def test_sine_f1_is_clamped(self):
        spec = SignalSpec(kind="sine", fs=8000, f1=20000.0, duration=1.0,
                          normalize="none")
        clamped, notes = clamp_spec_to_nyquist(spec)
        self.assertLessEqual(clamped.f1, clamped.f_limit)
        x = generate_signal(clamped)
        freqs, db = spectrum_dbfs(x, 8000, window="blackmanharris")
        self.assertLessEqual(float(freqs[int(np.argmax(db))]), 0.99 * 8000 / 2 + 1.0)

    def test_oversized_request_is_rejected(self):
        spec = SignalSpec(kind="stepped", n_steps=2000, step_duration=60.0, fs=48000)
        with self.assertRaises(ValueError):
            generate_signal(spec)

    def test_max_samples_constant_is_sane(self):
        self.assertGreater(MAX_SAMPLES, 1_000_000)


class TestSpectrum(unittest.TestCase):
    def test_full_scale_sine_reads_zero_dbfs(self):
        """归一化后满幅正弦应为 0 dBFS，与记录长度无关。"""
        for dur in (1.0, 2.0, 5.0):
            n = int(48000 * dur)
            t = np.arange(n) / 48000
            x = np.sin(2 * np.pi * 1000 * t)
            freqs, db = spectrum_dbfs(x, 48000, window="blackmanharris")
            peak = db[np.argmax(np.abs(np.fft.rfft(x * np.hanning(n))))]
            self.assertLess(peak, 3.0, msg=f"dur={dur}")
            self.assertGreater(peak, -3.0, msg=f"dur={dur}")

    def test_dbfs_is_length_independent(self):
        """回归：旧版纵轴随 N 漂移 6 dB/oct。"""
        readings = []
        for dur in (1.0, 2.0, 4.0):
            n = int(48000 * dur)
            t = np.arange(n) / 48000
            x = np.sin(2 * np.pi * 1000 * t)
            _, db = spectrum_dbfs(x, 48000, window="rectangular")
            readings.append(db[np.argmax(np.abs(np.fft.rfft(x)))])
        self.assertLess(max(readings) - min(readings), 0.6)

    def test_band_smoothing_integrates_band_power(self):
        """1/6 倍频程返回的必须是**频带总功率**（对功率谱积分）。

        旧版返回"每 bin 平均功率"，读数既不对（满幅正弦差 28 dB）
        也不稳（记录长度减到 1/4，读数就漂 9 dB）。
        """
        rng = np.random.default_rng(0)
        n = 240000
        x = rng.standard_normal(n)
        freqs, amp = one_sided_amplitude_spectrum(x, 48000, window="rectangular")
        power = amp ** 2
        centers, band_power = one_sixth_octave_smoothing(freqs, power)
        manual = []
        for fc in centers:
            lo, hi = fc / 2 ** (1 / 12), fc * 2 ** (1 / 12)
            idx = (freqs >= lo) & (freqs <= hi)
            manual.append(power[idx].sum())
        np.testing.assert_allclose(band_power, manual, rtol=1e-12)

    def test_band_level_of_full_scale_sine_is_zero_dbfs(self):
        """满幅正弦落在 1/6 倍频程频带里 → 该频带读数必须是 0 dBFS。

        这是倍频程电平标度的"锚点"：旧版这里读 -27.7 dBFS。
        """
        n = 240000
        t = np.arange(n) / 48000
        x = np.sin(2 * np.pi * 1000.0 * t)
        for window in ("rectangular", "hann", "blackmanharris", "nuttall"):
            freqs, power = one_sided_power_spectrum(x, 48000, window=window)
            centers, levels = band_levels_dbfs(freqs, power)
            i = int(np.argmin(np.abs(centers - 1000.0)))
            self.assertAlmostEqual(
                levels[i], 0.0, delta=0.25,
                msg=f"{window} 窗下满幅正弦的倍频程读数不是 0 dBFS",
            )

    def test_band_level_is_independent_of_record_length(self):
        """回归：旧版倍频程读数随记录长度漂移（4 倍长度 → 9 dB）。"""
        n = 240000
        t = np.arange(n) / 48000
        x = np.sin(2 * np.pi * 1000.0 * t)
        readings = []
        for div in (1, 2, 4, 8):
            freqs, power = one_sided_power_spectrum(x[: n // div], 48000,
                                                    window="nuttall")
            centers, levels = band_levels_dbfs(freqs, power)
            i = int(np.argmin(np.abs(centers - 1000.0)))
            readings.append(levels[i])
        self.assertLess(max(readings) - min(readings), 0.2,
                        msg=f"倍频程读数随记录长度漂移 {max(readings) - min(readings):.3f} dB")

    def test_white_noise_octave_bands_rise_by_3db_per_octave(self):
        """白噪的每带功率正比于带宽 → 倍频程曲线应 +3 dB/oct 上升。"""
        rng = np.random.default_rng(5)
        n = 240000
        x = rng.standard_normal(n)
        freqs, power = one_sided_power_spectrum(x, 48000, window="nuttall")
        centers, levels = band_levels_dbfs(freqs, power)
        mask = (centers >= 100) & (centers <= 10000)
        slope = np.polyfit(np.log2(centers[mask]), levels[mask], 1)[0]
        self.assertAlmostEqual(slope, 3.01, delta=0.35)

    def test_band_smoothing_no_infinite_loop_with_zero(self):
        """回归：f_min=0 时旧版 while 死循环。"""
        freqs = np.fft.rfftfreq(240000, 1 / 48000)
        power = np.ones_like(freqs)
        centers, _ = one_sixth_octave_smoothing(freqs, power, f_min=20.0, f_max=24000.0)
        self.assertGreater(len(centers), 0)
        self.assertTrue(np.all(np.diff(centers) > 0))

    def test_truncated_edge_bands_excluded(self):
        """首尾被数据边界截断的频带应被剔除。"""
        freqs = np.fft.rfftfreq(240000, 1 / 48000)
        power = np.ones_like(freqs)
        centers, _ = one_sixth_octave_smoothing(freqs, power, f_min=20.0, f_max=24000.0)
        edge_ratio = 2 ** (1 / 12)
        self.assertGreaterEqual(centers[0] / edge_ratio, freqs[0])
        self.assertLessEqual(centers[-1] * edge_ratio, freqs[-1])

    def test_blackman_harris_improves_dynamic_range(self):
        """回归：加窗应显著改善多音信号的本底动态范围。"""
        spec = SignalSpec(kind="multitone", n_tones=200, duration=5.0, fs=48000,
                          normalize="none")
        x = generate_signal(spec)

        # 多音的标称频率，用于定位音调之间的"空隙"
        tones = np.logspace(np.log10(20.0), np.log10(20000.0), 200)

        def gap_floor_db(window):
            freqs, db = spectrum_dbfs(x, 48000, window=window)
            levels = []
            for i in range(len(tones) - 1):
                a, b = tones[i], tones[i + 1]
                # 只取位于 [a, b] 区间内、且远离两端音调的空隙
                lo = a + 0.35 * (b - a)
                hi = b - 0.35 * (b - a)
                idx = (freqs >= lo) & (freqs <= hi)
                if np.any(idx):
                    power = np.mean(10 ** (db[idx] / 10))
                    levels.append(10 * np.log10(power))
            return float(np.mean(levels))

        rect = gap_floor_db("rectangular")
        bh = gap_floor_db("blackmanharris")
        self.assertLess(bh, rect - 40.0)

    def test_combined_fft_matches_separate_calls(self):
        """合并的单次 FFT 必须与分别调用两次的结果完全一致。"""
        x = generate_signal(SignalSpec(kind="pink", duration=2.0, fs=48000))
        f1, db1 = spectrum_dbfs(x, 48000, window="blackmanharris")
        f2, p2 = one_sided_power_spectrum(x, 48000, window="blackmanharris")
        f3, db3, p3 = one_sided_spectrum_dbfs_power(x, 48000, window="blackmanharris")
        np.testing.assert_allclose(f1, f3)
        np.testing.assert_allclose(db1, db3)
        np.testing.assert_allclose(p2, p3)

    def test_band_smoothing_searchsorted_matches_mask(self):
        """searchsorted 分带结果必须与布尔掩码逐位一致。"""
        freqs = np.fft.rfftfreq(240000, 1 / 48000)
        rng = np.random.default_rng(7)
        power = rng.random(freqs.size)
        centers, band_power = one_sixth_octave_smoothing(freqs, power)
        manual = []
        for fc in centers:
            lo, hi = fc / 2 ** (1 / 12), fc * 2 ** (1 / 12)
            idx = (freqs >= lo) & (freqs <= hi)
            manual.append(power[idx].sum())
        np.testing.assert_allclose(band_power, manual, rtol=1e-12)

    def test_band_smoothing_handles_unsorted_frequency_axis(self):
        """频率轴乱序时内部自动排序，结果仍应正确。"""
        freqs = np.fft.rfftfreq(24000, 1 / 48000)
        power = np.ones_like(freqs)
        order = np.random.default_rng(3).permutation(freqs.size)
        c_sorted, p_sorted = one_sixth_octave_smoothing(freqs, power)
        c_shuffled, p_shuffled = one_sixth_octave_smoothing(freqs[order], power[order])
        np.testing.assert_allclose(c_shuffled, c_sorted)
        np.testing.assert_allclose(p_shuffled, p_sorted)

    def test_stft_grid_is_reduced(self):
        """STFT 网格必须被压到"够看"的规模（绘图性能的直接决定因素）。"""
        x = generate_signal(SignalSpec(kind="pink", duration=5.0, fs=48000))
        f_spec, t_spec, db = stft_dbfs(x, 48000)
        self.assertLessEqual(f_spec.size, 320)
        self.assertLessEqual(t_spec.size, 300)
        self.assertEqual(db.shape, (f_spec.size, t_spec.size))

    def test_stft_preserves_sweep_trajectory(self):
        """抽稀后瞬时频率轨迹仍然可还原（log10(f) 对时间线性）。"""
        x = generate_signal(SignalSpec(kind="sweep", sweep_method="log", duration=5.0,
                                       fs=48000, f_start=20.0, f_stop=20000.0,
                                       normalize="none"))
        f_spec, t_spec, db = stft_dbfs(x, 48000)
        track = f_spec[np.argmax(db, axis=0)]
        self.assertLess(float(track[0]), 60.0)
        self.assertGreater(float(track[-1]), 18000.0)
        slope, intercept = np.polyfit(t_spec, np.log10(track), 1)
        resid = np.log10(track) - (slope * t_spec + intercept)
        self.assertLess(float(np.std(resid)), 0.08,
                        msg="抽稀后扫频轨迹失真")


class TestAudioIO(unittest.TestCase):
    def test_no_int16_overflow_wraparound(self):
        """回归：astype(int16) 对超量程会静默回绕成反向爆音。"""
        x = np.array([0.0, 1.5, -1.5, 1.0])
        q = quantize_int16(x, dither=False)
        self.assertEqual(int(q[0]), 0)
        self.assertEqual(int(q[3]), 32767)
        self.assertTrue(np.all(np.abs(q.astype(np.int64)) <= 32767))

    def test_clipping_is_bounded(self):
        x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        q = quantize_int16(x, dither=False)
        self.assertTrue(np.all(np.abs(q.astype(np.int64)) <= 32767))

    def test_dither_decorrelates_quantization_error(self):
        """抖动应让量化误差与信号解相关（相关性显著下降）。"""
        rng = np.random.default_rng(0)
        x = rng.standard_normal(20000) * 0.1
        q_no = quantize_int16(x, dither=False).astype(np.float64) / 32767.0
        q_di = quantize_int16(x, dither=True).astype(np.float64) / 32767.0
        c_no = np.corrcoef(x, x - q_no)[0, 1]
        c_di = np.corrcoef(x, x - q_di)[0, 1]
        self.assertLess(abs(c_di), abs(c_no))

    def test_dither_does_not_push_full_scale_peak_into_wraparound(self):
        """回归：先 clip 后加抖动，会把满幅样点推出量程并回绕成 -32768。

        旧顺序下，20000 样点的满幅正弦里有 2 ~ 4 个正峰被翻成 -1.0；
        只含一个 +1.0 的信号会出现上百个 -1.0。
        """
        for seed in range(5):
            sig = np.sin(np.arange(20000) * 0.01)
            q = quantize_int16(sig, seed=seed, dither=True)
            positives = np.flatnonzero(sig > 0.99)
            self.assertGreater(positives.size, 0)
            self.assertTrue(
                np.all(q[positives] > 0),
                msg=f"seed={seed} 有满幅正峰被量化成负值（回绕）",
            )

    def test_exact_full_scale_sample_maps_to_positive_max(self):
        """精确的 +1.0 必须映射到 +32767，不能溢出成 -32768。"""
        x = np.zeros(64)
        x[32] = 1.0
        for seed in range(8):
            q = quantize_int16(x, seed=seed, dither=True)
            self.assertEqual(int(q[32]), 32767, msg=f"seed={seed} 满幅正峰量化错误")

    def test_int24_full_scale_stays_in_range(self):
        """24 bit 同样不能溢出：+1.0 -> 8388607（不是 8388608）。"""
        x = np.zeros(64)
        x[32] = 1.0
        for seed in range(4):
            q = quantize_int24(x, seed=seed, dither=True)
            self.assertEqual(int(q[32]), 8388607, msg=f"seed={seed} 满幅正峰量化错误")

    def test_sweep_export_has_no_wrapped_peaks(self):
        """回归：导出扫频 WAV 时，满幅正峰不得出现反号。"""
        import tempfile

        spec = SignalSpec(kind="sweep", duration=3.0, fs=48000,
                          f_start=20.0, f_stop=20000.0, sweep_method="log")
        x = generate_signal(spec)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_wav(x, Path(tmp) / "s.wav", fs=48000, bits=16)
            from scipy.io import wavfile

            _fs, data = wavfile.read(path)
        positives = np.flatnonzero(x > 0.99)
        self.assertGreater(positives.size, 0)
        self.assertTrue(np.all(data[positives] > 0),
                        msg="导出文件里出现被回绕的正峰")
        # 量化后中段包络仍应为 1
        mid = np.abs(data) / 32767.0
        self.assertGreater(float(mid[int(mid.size * 0.2):int(mid.size * 0.8)].max()), 0.99)

    def test_save_wav_roundtrip(self):
        import tempfile
        from scipy.io import wavfile

        spec = SignalSpec(kind="sine", f1=1000.0, duration=1.0, fs=48000,
                          normalize="peak")
        x = generate_signal(spec)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_wav(x, Path(tmp) / "t.wav", fs=48000, bits=16)
            fs, data = wavfile.read(path)
            self.assertEqual(fs, 48000)
            self.assertEqual(data.size, x.size)
            self.assertEqual(data.dtype, np.int16)

    def test_save_wav_24bit(self):
        import tempfile
        from scipy.io import wavfile

        x = generate_signal(SignalSpec(kind="sine", duration=0.5))
        with tempfile.TemporaryDirectory() as tmp:
            path = save_wav(x, Path(tmp) / "t24.wav", fs=48000, bits=24)
            fs, data = wavfile.read(path)
            self.assertEqual(fs, 48000)
            self.assertEqual(data.dtype, np.int32)

    def test_stats(self):
        x = np.sin(2 * np.pi * 1000 * np.arange(48000) / 48000)
        st = signal_stats(x)
        self.assertAlmostEqual(st["peak_dbfs"], 0.0, delta=0.1)
        self.assertAlmostEqual(st["crest_db"], 3.01, delta=0.1)


class TestPlayer(unittest.TestCase):
    """播放链路（Windows winsound）。"""

    def test_backend_is_reported(self):
        from signal_gen.player import PLAYBACK_BACKEND, playback_supported

        self.assertIsInstance(PLAYBACK_BACKEND, str)
        self.assertEqual(playback_supported(), PLAYBACK_BACKEND != "不可用")

    def test_empty_signal_is_rejected(self):
        from signal_gen.player import AudioPlayer

        player = AudioPlayer()
        if not player.supported:
            self.skipTest("当前系统不支持内置播放")
        with self.assertRaises(ValueError):
            player.play(np.zeros(0), 48000)

    def test_play_and_stop_roundtrip(self):
        from signal_gen.player import AudioPlayer

        player = AudioPlayer()
        if not player.supported:
            self.skipTest("当前系统不支持内置播放")

        # 极短、极低电平，实际几乎无声，只验证"写 WAV -> 播放 -> 停止"链路
        x = 0.01 * np.sin(2 * np.pi * 1000 * np.arange(480) / 48000)
        try:
            duration = player.play(x, 48000, gain=1.0)
        except RuntimeError as exc:
            self.skipTest(f"本机没有可用的音频输出设备: {exc}")

        self.assertAlmostEqual(duration, 480 / 48000, places=6)
        self.assertTrue(player.playing)
        path = player.current_path
        self.assertIsNotNone(path)
        self.assertTrue(path.exists())

        player.stop()
        self.assertFalse(player.playing)
        self.assertIsNone(player.current_path)
        self.assertFalse(path.exists(), "停止播放后临时文件应被清理")

    def test_playback_stops_automatically_when_finished(self):
        """回归：播完之后必须自动停止，不能等用户手动点停止。"""
        import time

        from signal_gen.player import AudioPlayer

        player = AudioPlayer()
        if not player.supported:
            self.skipTest("当前系统不支持内置播放")

        x = 0.01 * np.sin(2 * np.pi * 1000 * np.arange(480) / 48000)
        try:
            player.play(x, 48000, gain=1.0)
        except RuntimeError as exc:
            self.skipTest(f"本机没有可用的音频输出设备: {exc}")

        self.assertTrue(player.playing)
        path = player.current_path

        deadline = time.monotonic() + 8.0
        while player.playing and time.monotonic() < deadline:
            time.sleep(0.02)

        self.assertFalse(player.playing, "播放结束后没有自动停止")
        self.assertIsNone(player.current_path, "播放结束后没有清理临时文件路径")
        if path is not None:
            self.assertFalse(path.exists(), "播放结束后临时文件应被清理")


class TestStandardSignals(unittest.TestCase):
    """四种标准测试信号（EIA RS-426-B / IEC 60268-1 ×3）。

    测量一律用**不加窗的 DFT 分箱积分**：DFT 是正交分解，某个频带内所有
    bin 的 |X|² 之和就是该频带的能量，不存在窗函数泄漏；而加窗测量会引入
    0.3~0.8 dB 的泄漏伪影（与信号本身无关）。
    """

    FS = 48000
    DURATION = 10.0

    # 1/3 倍频程中心频率（IEC 61260 / ANSI S1.11，1/3 倍频程带宽比 2^(1/6)）
    CENTERS = np.array([
        20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0,
        200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0,
        1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0,
        10000.0, 12500.0, 16000.0, 20000.0,
    ])

    def _band_levels(self, signal, fs=None):
        """返回 ``(中心频率, 1/3 倍频程电平 dB)``，用不加窗的 DFT 分箱积分。"""
        fs = fs or self.FS
        freqs, _db, power = one_sided_spectrum_dbfs_power(
            signal, fs, window="rectangular"
        )
        centers, band_power = third_octave_band_power(freqs, power, self.CENTERS)
        return centers, 10.0 * np.log10(np.maximum(band_power, 1e-30))

    def _psd_levels(self, signal, fs=None):
        """返回 ``(中心频率, 1/3 倍频程平均功率谱密度 dB/Hz)``。

        频带电平（频带总功率）与谱密度（每 Hz 功率）不是一回事：
        粉噪 PSD ∝ f^-1，但 1/3 倍频程的带宽也随频率线性增长，
        因此**频带电平恒定**，而谱密度按 -3 dB/oct 下降。
        """
        fs = fs or self.FS
        freqs, _db, power = one_sided_spectrum_dbfs_power(
            signal, fs, window="rectangular"
        )
        lo, hi = third_octave_edges(self.CENTERS)
        out_c, out_d = [], []
        for c, a, b in zip(self.CENTERS, lo, hi):
            i0 = int(np.searchsorted(freqs, a, side="left"))
            i1 = int(np.searchsorted(freqs, b, side="left"))
            if i1 <= i0:
                continue
            density = float(power[i0:i1].sum()) / (b - a)
            out_c.append(float(c))
            out_d.append(10.0 * np.log10(max(density, 1e-30)))
        return np.asarray(out_c), np.asarray(out_d)

    def _generate(self, kind, **kwargs):
        kwargs.setdefault("fs", self.FS)
        kwargs.setdefault("duration", self.DURATION)
        kwargs.setdefault("seed", 0)
        kwargs.setdefault("normalize", "none")
        spec = SignalSpec(kind=kind, **kwargs)
        return generate_signal(spec)

    # ------------------------------------------------------------------
    # 1/3 倍频程频带边界（ANSI S1.11-2004 / IEC 61260-1）
    # ------------------------------------------------------------------

    def test_band_edges_follow_the_standard(self):
        """**分析用**的标准滤波带必须符合 ANSI S1.11 / IEC 61260 定义。

        标准定义：精确中心 ``1000·10^(n/10)``，边界 = 精确中心 × ``10^(±0.05)``。
        真实分析仪就是这么测的，所以整形与验收都用这一套。

        下面几个数值直接取自 ANSI S1.11-2004 的频带边界表。
        """
        lo, hi = third_octave_edges(IEC268_PROGRAM_CENTERS)

        # 上边界 / 下边界 = 10^0.1（1/3 倍频程）
        np.testing.assert_allclose(hi / lo, 10.0 ** 0.1, rtol=1e-12)

        # 抽几个频带对照 ANSI S1.11-2004 表格
        table = {
            20.0: (17.783, 22.387),
            100.0: (89.127, 112.202),
            1000.0: (891.251, 1122.018),
            20000.0: (17782.794, 22387.214),
        }
        for center, (ref_lo, ref_hi) in table.items():
            idx = IEC268_PROGRAM_CENTERS.index(center)
            self.assertAlmostEqual(lo[idx], ref_lo, delta=ref_lo * 1e-4,
                                   msg=f"{center:g} Hz 频带下边界不符标准")
            self.assertAlmostEqual(hi[idx], ref_hi, delta=ref_hi * 1e-4,
                                   msg=f"{center:g} Hz 频带上边界不符标准")

    def test_standard_edges_tile_exactly(self):
        """**整形与校准用的**标准频带必须严丝合缝。

        标准频带按"精确中心 × 10^(±0.05)"划分，精确中心本身按 10^0.1
        等比排列，因此第 k 频带的上边界恰好等于第 k+1 频带的下边界 ——
        既不重叠，也不留洞。

        早前用的是"名义中心 × 2^(±1/6)"，相邻频带重叠 0.79 %、还会留下
        最大 1.57 % 的空洞（12.5 kHz 与 16 kHz 之间有 223.6 Hz 的死区），
        那不是标准频带，真实分析仪测出来的分带功率与标准表对不上。
        """
        lo, hi = third_octave_edges(IEC268_PROGRAM_CENTERS)
        np.testing.assert_allclose(hi[:-1], lo[1:], rtol=1e-9,
                                   err_msg="标准频带之间出现了缝隙或重叠")

        # 历史近似带（standard=False）确实有空洞也有重叠 —— 留着做对照
        lo_legacy, hi_legacy = third_octave_edges(IEC268_PROGRAM_CENTERS,
                                                  standard=False)
        gap = lo_legacy[1:] - hi_legacy[:-1]
        overlap = hi_legacy[:-1] - lo_legacy[1:]
        self.assertGreater(gap.max(), 100.0,
                           "名义中心 × 2^(±1/6) 应当留有空洞")
        self.assertGreater(overlap.max(), 1.0,
                           "名义中心 × 2^(±1/6) 应当存在重叠")

    def test_standard_bands_leave_no_dead_zones(self):
        """标准频带本身不应在相邻频带之间留洞。"""
        lo, hi = third_octave_edges(IEC268_PROGRAM_CENTERS)
        self.assertLessEqual((lo[1:] - hi[:-1]).max(), 1e-6,
                             "标准频带本身不应留洞")

    def test_exact_centers_match_the_standard_sequence(self):
        """IEC 61260 / ANSI S1.11 的精确中心序列是标准频带的基准。

        精确中心 ``1000·10^(n/10)`` 与标准表上的名义值最多差 1 %，
        频带边界就是按精确中心 × 10^(±0.05) 定义的。
        """
        for nominal, exact in zip(IEC268_PROGRAM_CENTERS,
                                   IEC268_PROGRAM_EXACT_CENTERS):
            self.assertAlmostEqual(exact, third_octave_exact_center(nominal), places=9)
            rel = abs(exact - nominal) / nominal
            self.assertLess(rel, 0.012, f"{nominal:g} Hz 与精确中心相差 {rel:.2%}")

        # 精确中心必须按 10^0.1 等比排列（相邻比恒定）
        ratios = np.diff(np.log10(IEC268_PROGRAM_EXACT_CENTERS))
        np.testing.assert_allclose(ratios, np.log10(10.0 ** 0.1), rtol=1e-9)

    # ------------------------------------------------------------------
    # IEC 60268-1 Clause 7 —— 模拟节目信号（simulated programme signal）
    # ------------------------------------------------------------------

    def _program_check(self, kind="iec268", **kwargs):
        """生成节目信号并按 Table II 判定其符合性。"""
        x = self._generate(kind, **kwargs)
        centers, levels = self._band_levels(x)
        return check_iec268_program(centers, levels)

    def test_iec268_program_matches_table_ii_band_by_band(self):
        """``iec268`` 必须逐带落在 Table II 的容差窗内。

        Table II 是**相对**频谱，所以判据是"存在一个统一电平，能把全部频带
        都放进各自的容差窗"（见 :func:`check_iec268_program`）。

        本实现把每个频带的闭式积分**精确**落在表值上（节点迭代拟合到
        1e-10 dB），有限记录的统计涨落只有百分之几分贝 —— 实测 30 s 记录
        逐带最大偏差 0.009 dB，10 s 记录 0.022 dB。
        """
        check = self._program_check("iec268")
        self.assertTrue(check.passes, f"未通过 Table II：裕量 {check.margin_db:.3f} dB")
        # 裕量几乎等于最窄的那条容差（±0.5 dB）→ 说明频谱本身就落在表值上
        self.assertGreater(check.margin_db, 0.4, f"裕量只有 {check.margin_db:.3f} dB")
        deviation = np.abs(check.deviations_db)
        self.assertLess(deviation.max(), 0.1,
                        f"最大逐带偏差 {deviation.max():.3f} dB")

    def test_iec268_program_relative_levels_match_the_table(self):
        """实测的频带相对电平必须与 Table II 的印值一致（±0.1 dB）。"""
        check = self._program_check("iec268")
        measured = check.band_levels - check.band_levels[0]
        table = check.relative_levels - check.relative_levels[0]
        np.testing.assert_allclose(measured, table, atol=0.1)

    def test_iec268_program_total_power_is_12_5_db_above_one_band(self):
        """Clause 7 的注：整个信号的功率约比一个频带的 0 dB 高 12.5 dB。

        那是 31 个相对电平的功率和：``10·log10(Σ 10^(T_i/10))`` = 12.56 dB。
        """
        self.assertAlmostEqual(IEC268_PROGRAM_TOTAL_POWER_DB, 12.56, delta=0.01)

        x = self._generate("iec268")
        n = x.size
        freqs = np.fft.rfftfreq(n, 1.0 / self.FS)
        power = np.abs(np.fft.rfft(x)) ** 2
        lo, hi = third_octave_edges([250.0])
        i0 = int(np.searchsorted(freqs, lo[0], side="left"))
        i1 = int(np.searchsorted(freqs, hi[0], side="left"))
        band = power[i0:i1].sum()
        self.assertGreater(band, 0.0)
        self.assertAlmostEqual(10 * np.log10(power.sum() / band), 12.56, delta=0.2)

    def test_iec268_program_is_not_amplitude_limited(self):
        """Clause 7 要求 *"without amplitude limiting"* → 默认不限幅。"""
        self.assertEqual(DEFAULT_PARAMS["iec268"]["crest_factor_db"], 0.0)
        x = self._generate("iec268")
        self.assertGreater(crest_factor_db(x), 10.0)

    def test_iec268_program_has_no_energy_outside_the_table_bands(self):
        """Table II 只覆盖 17.78 Hz – 22.39 kHz，之外不应有能量。"""
        x = self._generate("iec268")
        n = x.size
        freqs = np.fft.rfftfreq(n, 1.0 / self.FS)
        power = np.abs(np.fft.rfft(x)) ** 2
        total = power.sum()
        self.assertLess(power[freqs < 17.5].sum() / total, 1e-9, "17.5 Hz 以下不应有能量")
        self.assertLess(power[freqs > 22500.0].sum() / total, 1e-9, "22.5 kHz 以上不应有能量")

    def test_iec268_program_psd_integrates_to_the_table(self):
        """目标 PSD 积分后，每个 1/3 倍频程频带的功率 (dB) 必须等于表值。"""
        f = np.logspace(np.log10(17.0), np.log10(22400.0), 200000)
        psd = iec268_program_psd(f)
        step = np.gradient(f)
        for i, (center, table) in enumerate(
            zip(IEC268_PROGRAM_CENTERS, IEC268_PROGRAM_LEVELS)
        ):
            exact = IEC268_PROGRAM_EXACT_CENTERS[i]
            m = (f >= exact * 10.0 ** -0.05) & (f < exact * 10.0 ** 0.05)
            power = float(np.sum(psd[m] * step[m]))
            self.assertAlmostEqual(10 * np.log10(power), table, delta=0.01,
                                   msg=f"{center:g} Hz 频带积分与表值不符")

    def test_iec268_program_density_is_straight_in_log_frequency(self):
        """密度在 lg f 上是分段直线：节点之间线性，节点取各频带精确中心。"""
        centers = np.asarray(IEC268_PROGRAM_EXACT_CENTERS)
        nodes = iec268_program_density_db(centers)
        # 节点电平与频带电平差一个由斜率决定的平均因子，但节点之间必须严格线性
        mids = 0.5 * (nodes[:-1] + nodes[1:])
        mid_freqs = np.sqrt(centers[:-1] * centers[1:])
        values = iec268_program_density_db(mid_freqs)
        np.testing.assert_allclose(values, mids, atol=1e-9)

    def test_iec268_table_data_shape(self):
        """Table II 的数据本身：31 个频带、容差上下相等、总功率 12.56 dB。"""
        self.assertEqual(len(IEC268_PROGRAM_TABLE), 31)
        self.assertEqual(len(IEC268_PROGRAM_CENTERS), 31)
        self.assertEqual(IEC268_PROGRAM_CENTERS[0], 20.0)
        self.assertEqual(IEC268_PROGRAM_CENTERS[-1], 20000.0)
        for level, tolerance in IEC268_PROGRAM_TABLE.values():
            self.assertGreater(tolerance, 0.0)
        # 名义值升序，且与精确中心一一对应
        ratios = np.diff(np.log10(IEC268_PROGRAM_EXACT_CENTERS))
        np.testing.assert_allclose(ratios, np.log10(10.0 ** 0.1), rtol=1e-9)

    def test_iec268_fig2_response_matches_the_circuit(self):
        """Figure 2 的传递函数：通带损耗约 4 dB，两端滚降。

        元件值来自标准 Figure 2：源阻抗 430 Ω、串联 2.2 µF、节点 A
        （3.3 kΩ‖91 nF）、330 Ω + 2.2 µF、节点 B（3.3 kΩ‖68 nF）、
        0.47 µF、10 kΩ 输出电阻。
        """
        f = np.array([100.0, 1000.0, 10000.0, 20000.0])
        db = 20 * np.log10(np.abs(iec268_figure2_response(f)))
        self.assertAlmostEqual(db[1], -4.2, delta=0.5)   # 通带损耗约 4 dB
        self.assertLess(db[3], db[1] - 15.0)             # 高频滚降（并联电容）
        self.assertLess(db[0], db[1] - 0.5)              # 低频滚降（串联电容）

    def test_iec268_fig2_conforms_to_table_ii(self):
        """Figure 2 实现同样落在 Table II 容差内 —— 但裕量只剩约 0.18 dB。

        这正是"Figure 2 是一种实现、Table II 才是定义"的含义：电路实现
        的最差频带几乎贴着容差限，短记录甚至可能越界。
        """
        check = self._program_check("iec268_fig2")
        self.assertTrue(check.passes, f"未通过 Table II：裕量 {check.margin_db:.3f} dB")
        self.assertLess(check.margin_db, 0.35,
                        f"Figure 2 的裕量只剩 {check.margin_db:.3f} dB")
        self.assertGreater(check.margin_db, 0.0)

    def test_iec268_fig2_is_not_amplitude_limited(self):
        """Figure 2 也是 Clause 7 的实现 → 默认同样不限幅。"""
        self.assertEqual(DEFAULT_PARAMS["iec268_fig2"]["crest_factor_db"], 0.0)
        x = self._generate("iec268_fig2")
        self.assertGreater(crest_factor_db(x), 10.0)

    # ------------------------------------------------------------------
    # IEC 60268-1 附录 C / IEC 60065 —— 带限粉噪
    # ------------------------------------------------------------------

    def test_iec268_pink_is_flat_inside_22_4hz_22_4khz(self):
        """22.4 Hz–22.4 kHz 内应基本平坦。

        注意：这里平坦的是**滤波器**，不是噪声本身。粉噪的频带电平天然
        恒定，但滤波器的过渡带会让紧邻拐点的频带略有下沉：

        * 31.5 Hz 频带在高通拐点（22.7 Hz）的 1.4 倍处 → 约 -1.1 dB；
        * 20 kHz 频带跨在低通拐点（22.4 kHz）旁边 → 约 -1.7 dB。

        这两条都是标准滤波器自己的形状，不是实现误差，所以分别给容差。
        """
        x = self._generate("iec268_pink")
        centers, levels = self._band_levels(x)
        ref = levels[centers == 1000.0][0]

        # 通带主体（50 Hz – 16 kHz）：±0.6 dB
        inside = (centers >= 50.0) & (centers <= 16000.0)
        deviation = np.abs(levels[inside] - ref)
        self.assertLess(
            deviation.max(), 0.6,
            f"带内最大偏差 {deviation.max():.2f} dB",
        )

        # 过渡带（31.5 Hz）：标准滤波器的滚降起点，允许约 1.5 dB
        low = levels[centers == 31.5][0]
        self.assertLess(low - ref, -0.5, "31.5 Hz 频带应已开始滚降")
        self.assertGreater(low - ref, -1.5, "31.5 Hz 频带不应塌得太厉害")

    def test_iec268_pink_highpass_rolls_off_below_22_7hz(self):
        """12 dB/oct 高通（-3 dB @ 22.7 Hz）应把 20 Hz 频带压下去。

        20 Hz 频带（17.78 – 22.39 Hz）几乎整个落在拐点以下，
        实测相对 1 kHz 约 -4.3 dB。
        """
        x = self._generate("iec268_pink")
        centers, levels = self._band_levels(x)
        ref = levels[centers == 1000.0][0]
        low = levels[centers == 20.0][0]
        self.assertLess(low - ref, -3.5, f"20 Hz 相对 1 kHz 只有 {low - ref:.2f} dB")

    def test_iec268_pink_lowpass_rolls_off_above_22_4khz(self):
        """18 dB/oct 低通（-3 dB @ 22.4 kHz）应把 20 kHz 频带适度压下去。

        低通的 -3 dB 点在通带上沿 22.4 kHz，所以 20 kHz 频带
        （17.78 – 22.39 kHz）只有约 -1.7 dB —— 关键是**不能**像早前
        把拐点放在 19.6 kHz 那样把它压掉 8 dB 以上。
        """
        x = self._generate("iec268_pink")
        centers, levels = self._band_levels(x)
        ref = levels[centers == 1000.0][0]
        high = levels[centers == 20000.0][0]
        self.assertLess(high - ref, -0.5, f"20 kHz 相对 1 kHz 只有 {high - ref:.2f} dB")
        self.assertGreater(high - ref, -3.5,
                           f"20 kHz 相对 1 kHz 衰减过大：{high - ref:.2f} dB")

    def test_iec268_pink_is_not_the_programme_signal(self):
        """带限粉噪**不是** Clause 7 的节目信号 —— 它的逐带电平不满足 Table II。

        这条测试守着最容易混淆的地方：Clause 7 的节目信号按 Table II 整形，
        而附录 C 的粉噪只规定滤波器（通带平坦），两者的频谱形状完全不同。
        """
        check = self._program_check("iec268_pink")
        self.assertFalse(check.passes, "带限粉噪不应满足 Table II")

    def test_iec268_pink_crest_factor_is_6db(self):
        """IEC 60268-5 的 clipped pink noise：峰值因子 2:1（6 dB）。"""
        x = self._generate("iec268_pink")
        self.assertAlmostEqual(crest_factor_db(x), 6.0, delta=0.02)

    def test_iec268_filter_slopes_follow_the_standard(self):
        """滤波器的滚降斜率必须是标准规定的 12 / 18 dB/oct。

        早前把"12 dB/oct"当成了"6 阶"，实际做成 36 dB/oct；低通更离谱，
        54 dB/oct。这里直接断言解析式的渐近斜率。

        这两个拐点现在只服务于附录 C 的带限粉噪（``iec268_pink``）——
        Clause 7 的节目信号不走 Butterworth，而是按 Table II 逐带整形。
        """
        self.assertEqual(IEC268_HP_ORDER * 6, IEC268_HP_ROLLOFF_DB_OCT)
        self.assertEqual(IEC268_LP_ORDER * 6, IEC268_LP_ROLLOFF_DB_OCT)
        self.assertEqual(IEC268_HP_ROLLOFF_DB_OCT, 12.0)
        self.assertEqual(IEC268_LP_ROLLOFF_DB_OCT, 18.0)

        # 拐点处 -3 dB
        mag = butterworth_magnitude(np.array([IEC268_HP_FC]), IEC268_HP_FC,
                                    IEC268_HP_ORDER, "high")
        self.assertAlmostEqual(20 * np.log10(mag[0]), -3.01, delta=0.05)

        # 远离拐点一个十倍频程：Butterworth 的渐近段 |H| ≈ (fc/f)^N，
        # 即每个十倍频程衰减 20·N dB（= 斜率 × 10 / log2(10) dB）
        cases = (
            (IEC268_HP_FC / 10.0, IEC268_HP_FC, IEC268_HP_ORDER, "high"),
            (IEC268_LP_FC * 10.0, IEC268_LP_FC, IEC268_LP_ORDER, "low"),
        )
        for f, fc, order, btype in cases:
            mag = butterworth_magnitude(np.array([f]), fc, order, btype)
            self.assertAlmostEqual(20 * np.log10(mag[0]), -20.0 * order, delta=1.0,
                                   msg=f"{btype} @ {fc:g} Hz 滚降斜率不符")

    def test_iec268_lowpass_corner_follows_nyquist(self):
        """采样率过低时低通拐点必须跟着奈奎斯特频率走，不能混叠。

        fs=48 kHz → 拐点 22.4 kHz（标准值），8 kHz 仍在平坦区；
        fs=16 kHz → 拐点被压到 7.92 kHz，8 kHz 频带应明显塌下去。
        """
        self.assertAlmostEqual(iec268_lp_corner(48000), 22400.0, places=6)
        self.assertAlmostEqual(iec268_lp_corner(16000), 7920.0, places=6)

        def _rolloff(fs):
            x = self._generate("iec268_pink", fs=fs)
            self.assertEqual(x.size, int(fs * self.DURATION))
            centers, levels = self._band_levels(x, fs=fs)
            return levels[centers == 8000.0][0] - levels[centers == 6300.0][0]

        self.assertGreater(_rolloff(48000), -1.5, "fs=48 kHz 时 8 kHz 应仍在平坦区")
        self.assertLess(_rolloff(16000), -3.0, "fs=16 kHz 时 8 kHz 频带应塌下去")

    # ------------------------------------------------------------------
    # EIA RS-426-B
    # ------------------------------------------------------------------

    def test_eia426b_is_spectral_weighted_not_band_pass_pink(self):
        """**EIA RS-426-B 不是带通粉噪** —— 这是本次修正的核心。

        标准原文（Keele, AES 2001 转述）：*The specified weighting (as
        measured by a constant percentage bandwidth analyzer), bandlimits
        the noise to 40 Hz and 10 kHz, is flat between 40 Hz and 1 kHz, and
        then rolls off at 3 dB/octave (10 dB/decade) between 1 kHz and
        10 kHz.*

        因此恒定百分比带宽（1/3 倍频程）读数应当：

        * 40 Hz – 1 kHz **平坦**（⇒ 功率谱 ∝ 1/f，粉噪只是这一段的形状）；
        * 1 kHz – 10 kHz 按 **10 dB/decade** 下降（⇒ 功率谱 ∝ 1/f²）。

        如果错做成"粉噪 + 带通"，1 kHz 以上就会保持平坦（斜率 0 dB/dec），
        高频能量比标准多出好几个倍频程的斜率。
        """
        x = self._generate("eia426b")
        centers, levels = self._band_levels(x)
        idx = {c: i for i, c in enumerate(centers)}

        # 40 Hz – 1 kHz：频带电平平坦
        flat = [levels[idx[c]] for c in (50.0, 63.0, 80.0, 100.0, 125.0, 160.0,
                                         200.0, 250.0, 315.0, 400.0, 500.0,
                                         630.0, 800.0, 1000.0)]
        self.assertLess(
            max(flat) - min(flat), 0.6,
            "40 Hz – 1 kHz 的 1/3 倍频程读数应平坦（标准：is flat between "
            "40 Hz and 1 kHz）",
        )

        # 1 kHz – 10 kHz：按 10 dB/decade 滚降
        decades = math.log10(8000.0 / 1250.0)
        slope = (levels[idx[8000.0]] - levels[idx[1250.0]]) / decades
        self.assertAlmostEqual(
            slope, -10.0, delta=1.5,
            msg="1 kHz – 10 kHz 应按 3 dB/octave（10 dB/decade）滚降；"
                "若是带通粉噪则接近 0 dB/dec",
        )

    def test_eia426b_psd_slope_changes_at_1khz(self):
        """功率谱密度在 1 kHz 处由 -3 dB/oct 变成 -6 dB/oct。

        频带电平的斜率 = 谱密度斜率 + 3 dB/oct（1/3 倍频程带宽随频率增长），
        所以"频带电平平坦"对应"谱密度 -3 dB/oct"，"频带电平 -3 dB/oct"
        对应"谱密度 -6 dB/oct"。
        """
        x = self._generate("eia426b")
        pc, psd = self._psd_levels(x)
        idx = {c: i for i, c in enumerate(pc)}

        low = (psd[idx[500.0]] - psd[idx[125.0]]) / math.log2(500.0 / 125.0)
        high = (psd[idx[8000.0]] - psd[idx[2000.0]]) / math.log2(8000.0 / 2000.0)
        self.assertAlmostEqual(low, -3.0, delta=0.3)
        self.assertAlmostEqual(high, -6.0, delta=0.5)

    def test_eia426b_weighting_function(self):
        """加权谱函数本身：1 kHz 以下 ∝ f^-0.5，1 kHz 以上再乘 f^-0.5。"""
        f = np.array([100.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])
        a = eia426b_spectrum(f)
        # 1 kHz 以下：幅度 ∝ f^-0.5 → 每倍频程 -10·log10(2) = -3.01 dB
        self.assertAlmostEqual(
            20.0 * np.log10(a[2] / a[1]), -10.0 * math.log10(2.0), delta=0.01)
        # 1 kHz 以上：幅度 ∝ f^-1 → 每倍频程 -20·log10(2) = -6.02 dB
        self.assertAlmostEqual(
            20.0 * np.log10(a[3] / a[2]), -20.0 * math.log10(2.0), delta=0.01)
        self.assertAlmostEqual(
            20.0 * np.log10(a[4] / a[3]), -20.0 * math.log10(2.0), delta=0.01)
        self.assertAlmostEqual(
            20.0 * np.log10(a[5] / a[4]), -20.0 * math.log10(2.0), delta=0.01)

    def test_eia426b_band_limits_are_40hz_and_10khz(self):
        """限带由 40 Hz 高通（≥6 阶）与 10 kHz 低通（≥4 阶）完成。

        标准原文：*a 40-Hz high-pass filter of at least 6th-order minimum and
        a 10-kHz low-pass filter of at least 4th-order minimum, both
        maximally-flat Butterworth types*。
        """
        self.assertEqual(EIA426B_HP_FC, 40.0)
        self.assertEqual(EIA426B_LP_FC, 10000.0)
        self.assertGreaterEqual(EIA426B_HP_ORDER, 6)
        self.assertGreaterEqual(EIA426B_LP_ORDER, 4)
        self.assertEqual(EIA426B_HP_ROLLOFF_DB_OCT, 6.0 * EIA426B_HP_ORDER)
        self.assertEqual(EIA426B_LP_ROLLOFF_DB_OCT, 6.0 * EIA426B_LP_ORDER)

        x = self._generate("eia426b")
        centers, levels = self._band_levels(x)
        idx = {c: i for i, c in enumerate(centers)}
        ref = levels[idx[1000.0]]

        # 40 Hz 以下的频带应被高通信号压下去
        self.assertLess(levels[idx[25.0]] - ref, -12.0)
        self.assertLess(levels[idx[31.5]] - ref, -6.0)
        # 10 kHz 以上同理
        self.assertLess(levels[idx[12500.0]] - ref, -20.0)
        self.assertLess(levels[idx[16000.0]] - ref, -30.0)

    def test_eia426b_absolute_level_matches_the_standard(self):
        """绝对电平：峰值 −6 dBFS、RMS −12 dBFS。

        标准规定寿命测试噪声的 RMS 比 1 kHz 校准音低 3 dB。校准音是
        峰值 −6 dBFS 的正弦，RMS 为 −9 dBFS；噪声峰值因子 6 dB，因此
        噪声**峰值与校准音相同**（−6 dBFS）、RMS 为 −12 dBFS。
        """
        x = self._generate("eia426b")
        peak_db = 20.0 * math.log10(float(np.max(np.abs(x))))
        rms_db = 20.0 * math.log10(float(np.sqrt(np.mean(np.square(x)))))
        self.assertAlmostEqual(peak_db, EIA426B_CAL_TONE_PEAK_DBFS, delta=0.05)
        self.assertAlmostEqual(
            rms_db, EIA426B_CAL_TONE_PEAK_DBFS - EIA426B_CREST_DB, delta=0.05)
        # 与校准音（正弦，峰值 −6 dBFS）的 RMS 相差 3 dB
        tone_rms_db = EIA426B_CAL_TONE_PEAK_DBFS - 3.0103
        self.assertAlmostEqual(rms_db, tone_rms_db - 3.0, delta=0.05)

    def test_eia426b_uses_soft_clipping(self):
        """EIA RS-426-B 要求 **soft clipped**（Keele Eq. 1，k = 4）。

        软削波与硬削波的可观测差别：

        * 峰值因子同样是 6 dB；
        * 软削波是**圆拐点**，峰值的概率密度连续 → 峭度（kurtosis）
          明显低于硬削波；
        * Keele 实测他生成的 CD 轨道：``V_kurt = -0.931``、峰值因子 6.0 dB。

        这里的实现应当复现出同样的峭度 —— 说明削波函数与 k 取值都对。
        """
        x = self._generate("eia426b")
        self.assertAlmostEqual(crest_factor_db(x), 6.0, delta=0.02)

        m = float(np.mean(x))
        s = float(np.std(x))
        kurt = float(np.mean((x - m) ** 4)) / s ** 4 - 3.0
        # Keele 实测 −0.931；硬削波约 −0.6，软削波约 −0.93
        self.assertAlmostEqual(kurt, -0.931, delta=0.05,
                               msg="峭度与 Keele 实测不符，说明软削波模型不对")

    def test_eia426b_spectrum_matches_keele_figure6(self):
        """与 Keele 论文 Fig. 6（计算所得的 1/3 倍频程频谱）逐带对照。

        Fig. 6 的数值由论文矢量图反解得到（相对电平），Keele 的原始
        噪声是有限长记录 + 模拟滤波器，1/3 倍频程读数带 ±1 dB 的统计
        起伏，因此逐带容差取 2 dB。

        最低的两个频带（20 Hz / 25 Hz）不参与比较：那里 40 Hz 高通的
        过渡带占主导，而 Fig. 6 在该处的曲线与标准限带模型自相矛盾
        （图中 20→25 Hz 只降 5.1 dB，而 6 阶 40 Hz 高通必然给出约 11.4 dB），
        反解值本身不可靠。
        """
        keele = {
            31.56: -22.99, 39.65: -13.95,
            49.83: -10.49, 63.07: -10.65, 79.25: -10.34, 99.58: -10.49,
            126.05: -10.18, 158.39: -10.18, 199.02: -9.94, 250.07: -10.18,
            316.55: -9.86, 397.75: -10.02, 499.78: -10.18, 627.99: -10.57,
            789.09: -10.89, 998.84: -11.36, 1255.06: -11.75, 1577.02: -12.30,
            1981.57: -12.93, 2508.30: -13.64, 3151.75: -14.26, 3989.53: -15.05,
            5012.95: -15.84, 6298.91: -16.78, 7914.74: -18.12, 9945.07: -22.59,
        }
        x = self._generate("eia426b", duration=30.0)
        centers, levels = self._band_levels(x)

        # Keele 的参考电平是任意的，统一以 250 Hz 带对齐
        our_base = levels[list(centers).index(250.0)]
        worst = 0.0
        for c, ref in keele.items():
            # Fig. 6 的横坐标是**精确**中心频率，这里换算回名义中心
            nominal = min(self.CENTERS, key=lambda t: abs(math.log(t / c)))
            got = levels[list(centers).index(nominal)] - our_base
            want = ref - keele[250.07]
            worst = max(worst, abs(got - want))
        self.assertLess(
            worst, 2.0,
            f"与 Keele Fig. 6 的最大逐带偏差 {worst:.2f} dB 超出容差",
        )

    # ------------------------------------------------------------------
    # 峰值因子
    # ------------------------------------------------------------------

    def test_crest_factor_is_exact_for_all_standards(self):
        """标准要求的 6 dB 峰值因子必须精确实现。

        ``iec268`` / ``iec268_fig2`` 的 Clause 7 节目信号默认**不限幅**
        （标准原文 *"without amplitude limiting"*），但显式给出
        ``crest_factor_db=6`` 时必须精确命中 —— 这条路对应 IEC 60268-7:2010
        8.3.2 那种"带附加削波"（*"with additional clipping"*）的用法。
        """
        for kind in ("eia426b", "iec268", "iec268_fig2", "iec268_pink"):
            for seed in (0, 11):
                x = self._generate(kind, seed=seed, crest_factor_db=6.0)
                self.assertAlmostEqual(
                    crest_factor_db(x), 6.0, delta=0.02,
                    msg=f"{kind} seed={seed} 峰值因子未达标",
                )

    def test_crest_factor_zero_means_no_limiting(self):
        """crest_factor_db = 0 表示不限幅。"""
        x = self._generate("iec268", crest_factor_db=0.0)
        self.assertGreater(crest_factor_db(x), 10.0)

    def test_limit_crest_factor_helper(self):
        """限幅工具本身：目标峰值因子必须精确命中。"""
        rng = np.random.default_rng(3)
        noise = rng.standard_normal(200000)
        for target in (3.0, 6.0, 12.0):
            out = limit_crest_factor(noise, target)
            self.assertAlmostEqual(crest_factor_db(out), target, delta=0.01)

    def test_soft_clip_matches_keele_eq1(self):
        """软削波函数必须等于 Keele 论文 Eq.(1)。

        原文分段式::

            y = [ x^k / (x^k + 1) ]^(1/k)      , x ≥ 0
            y = −[ (−x)^k / ((−x)^k + 1) ]^(1/k) , x < 0

        等价于 ``y = sign(x)·|x| / (1 + |x|^k)^(1/k)``。
        """
        for k in (1.0, 2.0, 4.0, 100.0):
            x = np.array([-5.0, -1.0, -0.5, 0.0, 0.5, 1.0, 5.0])
            got = soft_clip(x, k)
            a = np.abs(x)
            want = np.sign(x) * (a ** k / (a ** k + 1.0)) ** (1.0 / k)
            np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-15)

    def test_soft_clip_is_linear_near_zero_and_saturates(self):
        """小信号斜率为 1（线性无失真），大信号渐近饱和于 ±1。"""
        k = 4.0
        # 小信号线性（论文：|x| ≤ 0.1 时斜率为 1）
        x = np.array([0.001, 0.01, 0.05])
        np.testing.assert_allclose(soft_clip(x, k), x, rtol=1e-5, atol=1e-12)
        # 永不越过 ±1，且单调趋于 1
        big = soft_clip(np.array([10.0, 100.0, 1000.0]), k)
        self.assertTrue(np.all(big < 1.0))
        self.assertTrue(np.all(np.diff(big) > 0.0))
        self.assertAlmostEqual(float(big[-1]), 1.0, delta=1e-3)
        # 极大输入也不会溢出
        huge = soft_clip(np.array([1e6, 1e12, 1e30]), k)
        self.assertTrue(np.all(huge <= 1.0 + 1e-12))
        # 奇对称
        np.testing.assert_allclose(soft_clip(-x, k), -soft_clip(x, k),
                                   rtol=1e-12, atol=1e-15)

    def test_soft_clip_to_crest_hits_target(self):
        """二分求输入增益必须把峰值因子压到目标值。"""
        rng = np.random.default_rng(5)
        noise = rng.standard_normal(200000)
        for target in (4.0, 6.0, 8.0, 10.0):
            out = soft_clip_to_crest(noise, target, k=4.0)
            self.assertAlmostEqual(crest_factor_db(out), target, delta=0.02,
                                   msg=f"软削波未把峰值因子压到 {target} dB")

    def test_soft_clip_is_rounder_than_hard_clip(self):
        """软削波的峭度应明显低于硬削波（峰值处概率密度连续）。"""
        rng = np.random.default_rng(7)
        noise = rng.standard_normal(400000)
        hard = limit_crest_factor(noise, 6.0)
        soft = soft_clip_to_crest(noise, 6.0, k=4.0)

        def _kurt(x):
            m = float(np.mean(x))
            s = float(np.std(x))
            return float(np.mean((x - m) ** 4)) / s ** 4 - 3.0

        self.assertLess(_kurt(soft), _kurt(hard) - 0.15)
        # Keele 实测的软削波峭度
        self.assertAlmostEqual(_kurt(soft), -0.931, delta=0.05)

    def test_limit_crest_factor_keeps_signal_when_already_ok(self):
        """本来就满足要求的信号不应被改动。"""
        rng = np.random.default_rng(4)
        noise = rng.standard_normal(10000)
        out = limit_crest_factor(noise, 30.0)
        np.testing.assert_array_equal(out, noise)

    # ------------------------------------------------------------------
    # 通用性
    # ------------------------------------------------------------------

    def test_standard_kinds_are_registered(self):
        for kind in ("eia426b", "iec268", "iec268_fig2", "iec268_pink"):
            self.assertIn(kind, ALL_KINDS)
            self.assertIn(kind, KIND_LABELS)
            self.assertIn(kind, DEFAULT_PARAMS)
            self.assertIn(kind, KIND_PARAMS)
            self.assertIn(kind, STANDARD_KINDS)

    def test_standard_signals_are_deterministic(self):
        """同 seed 必须逐比特复现。"""
        for kind in ("eia426b", "iec268", "iec268_fig2", "iec268_pink"):
            a = self._generate(kind, seed=5)
            b = self._generate(kind, seed=5)
            np.testing.assert_array_equal(a, b)
            c = self._generate(kind, seed=6)
            self.assertFalse(np.array_equal(a, c), f"{kind} 不同 seed 应有差异")

    def test_standard_signals_are_not_silent(self):
        for kind in ("eia426b", "iec268", "iec268_fig2", "iec268_pink"):
            x = self._generate(kind, normalize="rms")
            self.assertGreater(float(np.sqrt(np.mean(x ** 2))), 0.0)
            self.assertTrue(np.all(np.isfinite(x)))

    def test_butterworth_magnitude_at_minus3db(self):
        """Butterworth 解析幅度响应在 -3 dB 点应当准确。

        注意 Butterworth 的 -3 dB 点精确值是 ``1/sqrt(2)`` = **-3.0103 dB**，
        而不是整数 -3.000 dB，所以容差至少要 0.01 dB。
        """
        freqs = np.array([22.7])
        expected = 20.0 * np.log10(1.0 / np.sqrt(2.0))
        self.assertAlmostEqual(expected, -3.0, delta=0.02)
        mag = butterworth_magnitude(freqs, 22.7, order=6, btype="high")
        self.assertAlmostEqual(20.0 * np.log10(mag[0]), expected, delta=1e-9)
        mag = butterworth_magnitude(freqs, 22.7, order=6, btype="low")
        self.assertAlmostEqual(20.0 * np.log10(mag[0]), expected, delta=1e-9)


class TestWindowAndGui(unittest.TestCase):
    """窗函数选择、电平标度与界面结构的回归。"""

    # ------------------------------------------------------------------
    # STFT 电平标度
    # ------------------------------------------------------------------
    def test_stft_level_scale_matches_amplitude_spectrum(self):
        """回归：STFT 满幅正弦曾读 +49.5 dBFS（应为 0 dBFS）。

        STFT 与整体频谱必须用同一套幅度标度，否则两页图的纵轴
        无法互相对照。
        """
        n = 240000
        t = np.arange(n) / 48000
        x = np.sin(2 * np.pi * 1000.0 * t)
        for window in ("hann", "nuttall", "blackmanharris"):
            f_spec, _t_spec, db = stft_dbfs(x, 48000, window=window)
            i = int(np.argmin(np.abs(f_spec - 1000.0)))
            self.assertAlmostEqual(
                float(db[i].max()), 0.0, delta=0.5,
                msg=f"{window} 窗下 STFT 的满幅正弦读数不是 0 dBFS",
            )

    def test_stft_white_noise_floor_is_finite(self):
        """白噪的 STFT 不应出现 NaN / inf。"""
        rng = np.random.default_rng(9)
        x = rng.standard_normal(120000)
        f_spec, t_spec, db = stft_dbfs(x, 48000)
        self.assertTrue(np.all(np.isfinite(db)))
        self.assertEqual(db.shape, (f_spec.size, t_spec.size))

    # ------------------------------------------------------------------
    # 窗函数
    # ------------------------------------------------------------------
    def test_default_window_is_rectangular(self):
        """默认窗应为矩形窗（不加窗，频率分辨率最高）。"""
        from signal_gen.spectrum import DEFAULT_WINDOW

        self.assertEqual(DEFAULT_WINDOW, "rectangular")
        self.assertIn(DEFAULT_WINDOW, WINDOWS)

    def test_default_window_has_narrowest_main_lobe(self):
        """默认窗的 ENBW 必须是最窄的（矩形窗 = 1.00 bin）。"""
        from signal_gen.spectrum import DEFAULT_WINDOW, window_metrics

        default = window_metrics(DEFAULT_WINDOW, fs=48000, n=8192)
        for nm in ("hann", "blackman", "blackmanharris", "nuttall", "flattop"):
            other = window_metrics(nm, fs=48000, n=8192)
            self.assertLess(
                default["enbw_bins"], other["enbw_bins"],
                msg=f"默认窗的 ENBW 应小于 {nm}",
            )

    def test_nuttall_beats_blackman_harris(self):
        """实测：nuttall 的 ENBW 更窄、栅栏损失相当 → 全面优于 BH。"""
        from signal_gen.spectrum import window_metrics

        nuttall = window_metrics("nuttall", fs=48000, n=8192)
        bh = window_metrics("blackmanharris", fs=48000, n=8192)
        self.assertLess(nuttall["enbw_bins"], bh["enbw_bins"])
        self.assertLess(nuttall["scalloping_db"], bh["scalloping_db"] + 0.1)

    def test_flattop_has_negligible_scalloping_loss(self):
        """平顶窗的栅栏损失应接近 0 dB（幅度测量最准）。"""
        from signal_gen.spectrum import window_metrics

        m = window_metrics("flattop", fs=48000, n=8192)
        self.assertGreater(m["scalloping_db"], -0.1)
        self.assertLess(abs(m["scalloping_db"]), 0.1)

    def test_window_metrics_are_sane(self):
        """各窗的 ENBW 必须随窗变宽而单调不减。"""
        from signal_gen.spectrum import window_metrics

        enbw = [window_metrics(nm, fs=48000, n=8192)["enbw_bins"]
                for nm in ("rectangular", "hann", "blackman",
                           "blackmanharris", "nuttall", "flattop")]
        self.assertLess(enbw[0], enbw[1])
        self.assertLess(enbw[1], enbw[2])
        self.assertLess(enbw[-1], 4.0)

    # ------------------------------------------------------------------
    # 信号类型命名
    # ------------------------------------------------------------------
    def test_kind_labels_are_english(self):
        """类型名必须是常用英文命名（不含中文字符）。"""
        for kind in ALL_KINDS:
            label = KIND_LABELS[kind]
            self.assertTrue(label, msg=f"{kind} 缺少英文名")
            self.assertFalse(
                any("\u4e00" <= ch <= "\u9fff" for ch in label),
                msg=f"{kind} 的类型名里还有中文：{label}",
            )

    def test_kind_descriptions_cover_all_kinds(self):
        """每种类型都要有中文说明（CLI 列表与界面提示用）。"""
        from signal_gen.core import KIND_DESCRIPTIONS

        for kind in ALL_KINDS:
            self.assertIn(kind, KIND_DESCRIPTIONS)
            self.assertTrue(KIND_DESCRIPTIONS[kind].strip())

    # ------------------------------------------------------------------
    # 界面结构
    # ------------------------------------------------------------------
    def test_octave_band_has_no_separate_tab(self):
        """1/6 倍频程不再单独占一个标签页（叠在频谱页上）。"""
        from signal_gen.gui import TAB_KEYS

        self.assertEqual(TAB_KEYS, ("time", "spec", "stft"))
        self.assertNotIn("band", TAB_KEYS)

    def test_minmax_envelope_keeps_high_frequency_peaks(self):
        """回归：时域抽稀造成的"时间点缺失"观感。

        15 kHz 正弦在 48 kHz 下每周期只有 3.2 个样点，隔点抽稀会
        产生严重混叠；min/max 包络必须把峰值完整保留下来。
        """
        from signal_gen.gui import minmax_envelope

        n = 48000
        t = np.arange(n) / 48000
        x = np.sin(2 * np.pi * 15000.0 * t)
        tt, lo, hi = minmax_envelope(x, 48000, ncols=1600)
        self.assertEqual(tt.size, 1600)
        self.assertGreater(float(hi.max()), 0.99)
        self.assertLess(float(lo.min()), -0.99)
        # 时间轴必须覆盖整段信号（回归：步长曾写成 ncols/fs，整条轴被拉长）
        self.assertAlmostEqual(float(tt[-1]), 1.0, delta=0.01)
        self.assertAlmostEqual(float(tt[0]), 1.0 / 3200, delta=1e-6)

    def test_minmax_envelope_tracks_amplitude_envelope(self):
        """包络必须忠实还原信号的幅度包络（这是抽稀做不到的）。"""
        from signal_gen.gui import minmax_envelope

        n = 48000
        t = np.arange(n) / 48000
        env = np.linspace(0.1, 1.0, n)
        x = env * np.sin(2 * np.pi * 15000.0 * t)

        tt, _lo, hi = minmax_envelope(x, 48000, ncols=1600)
        idx = np.clip((tt * 48000).astype(np.int64), 0, n - 1)
        self.assertLess(float(np.abs(hi - env[idx]).max()), 0.02)

        # 旧做法（隔点抽稀）会明显偏离真实包络
        step = max(1, n // 1600)
        naive = x[::step]
        naive_env = np.abs(naive)
        self.assertGreater(float(np.abs(naive_env - env[::step]).max()), 0.02)

    def test_minmax_envelope_handles_short_signals(self):
        from signal_gen.gui import minmax_envelope

        x = np.linspace(-1.0, 1.0, 100)
        tt, lo, hi = minmax_envelope(x, 48000, ncols=1600)
        self.assertEqual(tt.size, 100)
        np.testing.assert_allclose(lo, x)
        np.testing.assert_allclose(hi, x)

    def test_minmax_envelope_handles_empty_signal(self):
        from signal_gen.gui import minmax_envelope

        tt, lo, hi = minmax_envelope(np.zeros(0), 48000)
        self.assertEqual(tt.size, 0)
        self.assertEqual(lo.size, 0)
        self.assertEqual(hi.size, 0)


class TestFrequencyRangeParams(unittest.TestCase):
    """起始 / 终止频率：所有信号类型都能设置，含义随类型而变。"""

    FS = 48000

    # 1/3 倍频程的名义中心频率（用于标准信号的频带测量）
    CENTERS = np.array([
        20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0,
        200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0,
        1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0,
        10000.0, 12500.0, 16000.0, 20000.0,
    ])

    # ------------------------------------------------------------------
    # 参数表结构
    # ------------------------------------------------------------------

    def test_every_kind_declares_a_frequency_range(self):
        for kind in ALL_KINDS:
            params = KIND_PARAMS[kind]
            self.assertIn("f_start", params, f"{kind} 没有起始频率控件")
            if kind != "sine":
                self.assertIn("f_stop", params, f"{kind} 没有终止频率控件")

    def test_default_params_cover_every_kind(self):
        for kind in ALL_KINDS:
            self.assertIn(kind, DEFAULT_PARAMS)
            self.assertIn("f_start", DEFAULT_PARAMS[kind])
            if kind != "sine":
                self.assertIn("f_stop", DEFAULT_PARAMS[kind])

    def test_frequency_param_info_covers_every_kind(self):
        """每个类型都要说明这两个参数到底是什么意思（GUI 动态显示）。"""
        for kind in ALL_KINDS:
            self.assertIn(kind, FREQ_PARAM_INFO)
            self.assertIn("f_start", FREQ_PARAM_INFO[kind])
            self.assertIn("f_stop", FREQ_PARAM_INFO[kind])

    def test_library_defaults_match_the_parameter_table(self):
        """库 API 的默认行为必须与 GUI / CLI 用的是同一套默认值。"""
        for kind in ALL_KINDS:
            spec = SignalSpec(kind=kind)
            for key, want in DEFAULT_PARAMS[kind].items():
                if key == "crest_factor_db":
                    continue  # None = 用标准默认值，行为等价
                self.assertEqual(getattr(spec, key), want, f"{kind}.{key}")

    def test_explicit_values_are_never_overwritten(self):
        """显式传进去的值不能被"按类型补默认值"覆盖掉。"""
        spec = SignalSpec(kind="iec268", f_stop=18000.0)
        self.assertEqual(spec.f_stop, 18000.0)
        self.assertEqual(spec.f_start, DEFAULT_PARAMS["iec268"]["f_start"])
        spec = SignalSpec(kind="iec268", f_start=100.0)
        self.assertEqual(spec.f_start, 100.0)
        self.assertEqual(spec.f_stop, DEFAULT_PARAMS["iec268"]["f_stop"])
        self.assertEqual(SignalSpec(kind="red", beta=1.5).beta, 1.5)
        self.assertEqual(SignalSpec(kind="sweep", fade_ratio=0.3).fade_ratio, 0.3)

    # ------------------------------------------------------------------
    # 单频正弦：f_start 就是信号频率
    # ------------------------------------------------------------------

    def test_sine_uses_f_start_as_signal_frequency(self):
        self.assertEqual(SignalSpec(kind="sine").f1, 1000.0)
        self.assertEqual(SignalSpec(kind="sine", f_start=500.0).f1, 500.0)
        self.assertEqual(SignalSpec(kind="sine", f1=500.0).f1, 500.0)
        self.assertEqual(SignalSpec(kind="sine", f_start=20.0).f1, 20.0)

    def test_sine_frequency_actually_controls_the_tone(self):
        x = generate_signal(SignalSpec(kind="sine", f_start=500.0,
                                      duration=1.0, fs=self.FS, normalize="none"))
        freqs, db = spectrum_dbfs(x, self.FS, window="blackmanharris")
        self.assertAlmostEqual(float(freqs[int(np.argmax(db))]), 500.0, delta=2.0)

    def test_sine_has_no_terminal_frequency(self):
        """单频信号只有一个频率，终止频率对它没有意义。"""
        self.assertNotIn("f_stop", KIND_PARAMS["sine"])

    # ------------------------------------------------------------------
    # 宽带噪声的带限
    # ------------------------------------------------------------------

    def test_noise_is_band_limited(self):
        for kind in ("white", "pink", "red", "blue"):
            x = generate_signal(SignalSpec(kind=kind, duration=4.0, fs=self.FS,
                                          normalize="none",
                                          f_start=200.0, f_stop=2000.0))
            freqs, power = one_sided_power_spectrum(x, self.FS, window="nuttall")
            total = power.sum()
            self.assertLess(power[freqs < 190.0].sum() / total, 1e-6,
                            f"{kind} 噪声在带外仍有能量")
            self.assertLess(power[freqs > 2100.0].sum() / total, 1e-6,
                            f"{kind} 噪声在带外仍有能量")

    def test_default_noise_band_is_20hz_to_20khz(self):
        spec = SignalSpec(kind="white")
        self.assertEqual(spec.f_start, 20.0)
        self.assertEqual(spec.f_stop, 20000.0)
        x = generate_signal(spec.with_(duration=4.0, fs=self.FS, normalize="none"))
        freqs, power = one_sided_power_spectrum(x, self.FS, window="nuttall")
        total = power.sum()
        self.assertLess(power[freqs < 19.0].sum() / total, 1e-6)
        self.assertLess(power[freqs > 20600.0].sum() / total, 1e-6)

    def test_band_limit_keeps_the_in_band_slope(self):
        """带限不能破坏 1/f 谱斜率 —— 这正是不用 Butterworth 高通的原因。

        Butterworth 高通在 20 Hz 拐点会让 100 Hz 处低 28 dB，斜率就毁了；
        砖墙式限制（带内完全平坦）才不会。
        """
        for kind, beta in (("pink", 1.0), ("red", 2.0), ("blue", -1.0)):
            x = generate_signal(SignalSpec(kind=kind, beta=beta, duration=10.0,
                                          fs=self.FS, normalize="none",
                                          f_start=20.0, f_stop=20000.0))
            freqs, power = one_sided_power_spectrum(x, self.FS, window="hann")
            psd_db = 10.0 * np.log10(np.maximum(power, 1e-30))
            m = (freqs >= 100) & (freqs <= 10000)
            slope = float(np.polyfit(np.log2(freqs[m]), psd_db[m], 1)[0])
            self.assertAlmostEqual(slope, -3.01 * beta, delta=0.3,
                                   msg=f"{kind} 的谱斜率被带限破坏了")

    def test_band_limit_does_not_tilt_the_band_interior(self):
        """带内不应出现下倾（白噪的倍频程曲线仍应 +3 dB/oct 上升）。"""
        x = generate_signal(SignalSpec(kind="white", duration=10.0, fs=self.FS,
                                      normalize="none",
                                      f_start=200.0, f_stop=2000.0))
        freqs, power = one_sided_power_spectrum(x, self.FS, window="hann")
        centers, levels = band_levels_dbfs(freqs, power)
        m = (centers >= 250) & (centers <= 1600)
        slope = float(np.polyfit(np.log2(centers[m]), levels[m], 1)[0])
        self.assertAlmostEqual(slope, 3.0, delta=0.8,
                               msg="带内倍频程曲线被带限压成了下倾")

    # ------------------------------------------------------------------
    # 标准信号的频带参数化
    # ------------------------------------------------------------------

    def test_iec268_default_band_is_the_standard_one(self):
        """Clause 7 的 Table II 覆盖 20 Hz – 20 kHz 共 31 个 1/3 倍频程频带。

        节目信号的 ``f_start`` / ``f_stop`` 语义是"纳入整形的频带范围"，
        不是滤波器的 -3 dB 拐点 —— 拐点语义属于附录 C 的带限粉噪
        （``iec268_pink``）。
        """
        spec = SignalSpec(kind="iec268")
        self.assertAlmostEqual(spec.f_start, 20.0)
        self.assertAlmostEqual(spec.f_stop, 20000.0)
        spec = SignalSpec(kind="iec268_fig2")
        self.assertAlmostEqual(spec.f_start, 20.0)
        self.assertAlmostEqual(spec.f_stop, 20000.0)

    def test_iec268_pink_default_corners_are_the_standard_ones(self):
        """附录 C / IEC 60065 带限粉噪：-3 dB 拐点 22.7 Hz / 22.4 kHz。"""
        spec = SignalSpec(kind="iec268_pink")
        self.assertAlmostEqual(spec.f_start, IEC268_HP_FC)
        self.assertAlmostEqual(spec.f_stop, IEC268_LP_FC)

    def test_iec268_pink_corners_follow_the_specified_range(self):
        """IEC 60268-1 的 -3 dB 拐点就是 f_start / f_stop。"""
        x = generate_signal(SignalSpec(kind="iec268_pink", duration=10.0, fs=self.FS,
                                      normalize="none",
                                      f_start=100.0, f_stop=5000.0))
        freqs, _db, power = one_sided_spectrum_dbfs_power(
            x, self.FS, window="rectangular")
        _c, band_power = third_octave_band_power(freqs, power, self.CENTERS)
        levels = 10.0 * np.log10(np.maximum(band_power, 1e-30))
        ref = levels[self.CENTERS == 1000.0][0]
        # 高通拐点抬到 100 Hz → 20 Hz 频带被压得很低
        self.assertLess(levels[self.CENTERS == 20.0][0] - ref, -20.0)
        # 低通拐点降到 5 kHz → 10 kHz 频带被压下去
        self.assertLess(levels[self.CENTERS == 10000.0][0] - ref, -15.0)

    def test_iec268_program_band_subset_drops_the_bands_above_f_stop(self):
        """收窄频带范围后，超出范围的频带整体消失（不留半个频带）。"""
        x = generate_signal(SignalSpec(kind="iec268", duration=10.0, fs=self.FS,
                                      normalize="none",
                                      f_start=100.0, f_stop=1000.0))
        freqs = np.fft.rfftfreq(x.size, 1.0 / self.FS)
        power = np.abs(np.fft.rfft(x)) ** 2
        total = power.sum()
        # 100 Hz 频带的下沿是 88.8 Hz（精确中心 × 10^-0.05）
        self.assertLess(power[freqs < 88.0].sum() / total, 1e-9)
        # 1 kHz 频带的上沿是 1122 Hz
        self.assertLess(power[freqs > 1130.0].sum() / total, 1e-9)

    def test_iec268_fig2_band_subset_drops_the_bands_above_f_stop(self):
        """Figure 2 实现同样按"频带范围"整带取舍。"""
        x = generate_signal(SignalSpec(kind="iec268_fig2", duration=10.0, fs=self.FS,
                                      normalize="none",
                                      f_start=100.0, f_stop=1000.0))
        freqs = np.fft.rfftfreq(x.size, 1.0 / self.FS)
        power = np.abs(np.fft.rfft(x)) ** 2
        total = power.sum()
        self.assertLess(power[freqs < 88.0].sum() / total, 1e-9)
        self.assertLess(power[freqs > 1130.0].sum() / total, 1e-9)

    def test_standard_band_deviation_is_reported(self):
        _spec, notes = clamp_spec_to_nyquist(
            SignalSpec(kind="iec268", f_start=100.0))
        self.assertTrue(any("偏离" in n for n in notes))
        _spec, notes = clamp_spec_to_nyquist(
            SignalSpec(kind="iec268_fig2", f_stop=10000.0))
        self.assertTrue(any("偏离" in n for n in notes))
        _spec, notes = clamp_spec_to_nyquist(
            SignalSpec(kind="iec268_pink", f_stop=10000.0))
        self.assertTrue(any("偏离" in n for n in notes))
        # 默认频带不该有提示
        _spec, notes = clamp_spec_to_nyquist(SignalSpec(kind="iec268"))
        self.assertEqual(notes, [])
        # EIA RS-426-B 的频带由标准固定为 40 Hz – 10 kHz，改动即"偏离"
        _spec, notes = clamp_spec_to_nyquist(
            SignalSpec(kind="eia426b", f_start=50.0))
        self.assertTrue(any("偏离" in n for n in notes))
        _spec, notes = clamp_spec_to_nyquist(SignalSpec(kind="eia426b"))
        self.assertEqual(notes, [])

    # ------------------------------------------------------------------
    # GUI
    # ------------------------------------------------------------------

    def test_time_plot_has_no_inset(self):
        """回归：时域图曾带一个"前 20 ms 细节"的放大插图，现已去掉。"""
        import inspect

        from signal_gen.gui import SignalGeneratorApp

        source = inspect.getsource(SignalGeneratorApp._draw_time)
        self.assertNotIn("inset", source)

    def test_gui_frequency_labels_follow_the_signal_kind(self):
        """频率参数的标签与可用性必须随信号类型变化。"""
        try:
            from signal_gen.gui import SignalGeneratorApp

            app = SignalGeneratorApp()
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"无法创建 GUI：{exc}")
        try:
            app.withdraw()
            app._updating = True  # 不触发重算，只看控件状态
            for kind in ALL_KINDS:
                app.kind_var.set(KIND_LABELS[kind])
                app._on_kind_change()
                self.assertEqual(
                    str(app._spec_labels["f_start"].cget("text")),
                    FREQ_PARAM_INFO[kind]["f_start"][0],
                    msg=f"{kind} 的起始频率标签不对",
                )
                self.assertEqual(
                    str(app._spec_labels["f_stop"].cget("text")),
                    FREQ_PARAM_INFO[kind]["f_stop"][0],
                    msg=f"{kind} 的终止频率标签不对",
                )
                expected = "disabled" if kind == "sine" else "normal"
                self.assertEqual(
                    str(app._spec_widgets["f_stop"].cget("state")), expected,
                    msg=f"{kind} 的终止频率控件状态不对",
                )
        finally:
            app.destroy()


if __name__ == "__main__":
    unittest.main(verbosity=2)
