"""
外部音频分析（:mod:`signal_gen.report`）与统一音频读取的测试。

这一组覆盖的是从旧脚本「读取信号文件 → 分析处理 → 画图」并进来的能力：

* :func:`signal_gen.audio_io.load_audio` / :func:`signal_gen.audio_io.audio_info`
  —— 统一读取入口（WAV 零依赖，其它格式可选 soundfile）；
* :func:`signal_gen.report.analyze_file` —— 读文件 → 全部读数 + 1/6 倍频程；
* :func:`signal_gen.report.build_report_figure` / :func:`signal_gen.report.report_file`
  —— 双栏分析报告图；
* CLI 的 ``analyze --plot``。

**口径约定**（与生成信号完全一致）：满幅正弦 = 0 dBFS；
1/6 倍频程按 IEC 61260 用**频带功率积分**，不是旧脚本的算术平均。

运行::

    python -m unittest discover -s tests
    python run_tests.py TestReport
"""

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 无头后端：必须在 pyplot 被导入前设定
import matplotlib  # noqa: E402

matplotlib.use("Agg")

from signal_gen import cli as cli_module  # noqa: E402
from signal_gen.audio_io import (  # noqa: E402
    NON_WAV_SUFFIXES,
    audio_info,
    load_audio,
    save_wav,
    wav_info,
)
from signal_gen.core import SignalSpec, generate_signal  # noqa: E402
from signal_gen.audio_io import signal_stats  # noqa: E402
from signal_gen.report import (  # noqa: E402
    analyze_file,
    build_report_figure,
    format_band_table,
    minmax_envelope,
    report_file,
)

def _write_test_wav(path: Path, signal: np.ndarray, fs: int = 48000,
                    bits: int = 16, dither: bool = False) -> Path:
    """写一个测试用 WAV（关掉抖动，让量化误差可预期）。"""
    save_wav(signal, path, fs=fs, bits=bits, dither=dither)
    return path


class TestLoadAudio(unittest.TestCase):
    """统一读取入口 :func:`load_audio`。"""

    def setUp(self) -> None:
        matplotlib.use("Agg")

    def test_reads_a_wav_file(self):
        x = np.sin(2 * np.pi * 1000 * np.arange(4800) / 48000)
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_test_wav(Path(tmp) / "a.wav", x)
            y, fs = load_audio(path)
            self.assertEqual(fs, 48000)
            self.assertEqual(y.shape, x.shape)
            self.assertLess(float(np.abs(y - x).max()), 2.0 / 32768.0)

    def test_wave_suffix_is_treated_as_wav(self):
        """``.wave`` 与 ``.wav`` 是同一个格式，不能因为后缀不同就去找 soundfile。"""
        x = np.sin(2 * np.pi * 500 * np.arange(480) / 48000)
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_test_wav(Path(tmp) / "a.wave", x)
            y, fs = load_audio(path)
            self.assertEqual(fs, 48000)
            self.assertEqual(y.size, x.size)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_audio(Path(tempfile.gettempdir()) / "绝对不存在.wav")

    def test_non_wav_without_soundfile_gives_a_clear_error(self):
        """环境里没装 soundfile 时，错误信息必须说清怎么装 —— 而不是抛 ImportError。"""
        from signal_gen.audio_io import _soundfile_module

        if _soundfile_module() is not None:
            self.skipTest("本机装了 soundfile，非 WAV 路径已由 soundfile 负责")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.flac"
            path.write_bytes(b"fLaC" + b"\x00" * 64)
            with self.assertRaises(ValueError) as ctx:
                load_audio(path)
            message = str(ctx.exception)
            self.assertIn("soundfile", message)
            self.assertIn("pip install soundfile", message)

    def test_non_wav_suffixes_is_a_single_source(self):
        """``NON_WAV_SUFFIXES`` 只能有一处定义，包顶层与 report 都是同一份。"""
        import signal_gen
        from signal_gen import report as report_module

        self.assertIs(signal_gen.NON_WAV_SUFFIXES, NON_WAV_SUFFIXES)
        self.assertIs(report_module.NON_WAV_SUFFIXES, NON_WAV_SUFFIXES)
        self.assertIn(".flac", NON_WAV_SUFFIXES)
        self.assertNotIn(".wav", NON_WAV_SUFFIXES)


class TestAudioInfo(unittest.TestCase):
    """:func:`audio_info` —— 不解码采样数据就能拿到格式信息。"""

    def test_wav_info_is_consistent_with_wav_info(self):
        x = np.zeros(1000, dtype=np.float64)
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_test_wav(Path(tmp) / "i.wav", x)
            info = audio_info(path)
            reference = wav_info(path)
            for key in ("channels", "sample_rate", "bits", "frames", "duration", "encoding"):
                self.assertEqual(info[key], reference[key], msg=f"{key} 与 wav_info 不一致")
            self.assertEqual(info["format"], "WAV")

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            audio_info(Path(tempfile.gettempdir()) / "绝对不存在.wav")


class TestAnalyzeFile(unittest.TestCase):
    """:func:`analyze_file` —— 读文件 → 读数 + 频谱 + 1/6 倍频程。"""

    def setUp(self) -> None:
        matplotlib.use("Agg")

    def test_returns_the_documented_fields(self):
        x = generate_signal(SignalSpec(kind="pink", duration=1.0, fs=48000))
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_test_wav(Path(tmp) / "a.wav", x)
            result = analyze_file(path)

        for key in ("path", "name", "signal", "fs", "n_samples", "duration",
                    "info", "stats", "freqs", "db", "centers", "bands",
                    "window", "order"):
            self.assertIn(key, result, msg=f"缺少字段 {key!r}")

        self.assertEqual(result["fs"], 48000)
        self.assertEqual(result["n_samples"], 48000)
        self.assertAlmostEqual(result["duration"], 1.0)
        self.assertEqual(result["order"], 6)

    def test_readings_use_the_absolute_dbfs_scale(self):
        """满幅正弦：峰值 0 dBFS、RMS -3.01 dBFS、峰值因子 3.01 dB。

        旧脚本会先把波形归一化到峰值 1.0，**绝对刻度就丢了**；
        现在外部文件与内部生成的信号可以直接对比。
        """
        n = 48000
        x = np.sin(2 * np.pi * 1000 * np.arange(n) / 48000)
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_test_wav(Path(tmp) / "sine.wav", x)
            stats = analyze_file(path)["stats"]

        self.assertAlmostEqual(stats["peak_dbfs"], 0.0, delta=0.05)
        self.assertAlmostEqual(stats["rms_dbfs"], -3.01, delta=0.1)
        self.assertAlmostEqual(stats["crest_db"], 3.01, delta=0.1)

    def test_band_levels_are_already_in_dbfs(self):
        """1/6 倍频程读数必须**已经是 dB 值**（历史上被重复取过 log10，全变成 -200）。"""
        n = 48000
        x = np.sin(2 * np.pi * 1000 * np.arange(n) / 48000)
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_test_wav(Path(tmp) / "sine.wav", x)
            result = analyze_file(path)

        bands = result["bands"]
        self.assertTrue(np.all(bands > -200.0), msg="频带读数看起来被重复取了 log10")

        # 1 kHz 正弦所在的频带应读到约 0 dBFS（功率积分口径）
        centers = result["centers"]
        index = int(np.argmin(np.abs(centers - 1000.0)))
        self.assertAlmostEqual(float(bands[index]), 0.0, delta=0.5)

    def test_matches_the_generated_signal(self):
        """同一个波形：生成路径与分析路径的频带读数必须一致。"""
        spec = SignalSpec(kind="pink", duration=1.0, fs=48000)
        x = generate_signal(spec)
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_test_wav(Path(tmp) / "p.wav", x, bits=24)
            result = analyze_file(path)

        expected = signal_stats(x)
        self.assertAlmostEqual(result["stats"]["rms_dbfs"], expected["rms_dbfs"], delta=0.01)
        self.assertAlmostEqual(result["stats"]["peak_dbfs"], expected["peak_dbfs"], delta=0.01)
        # 粉噪的 1/6 倍频程曲线应当**平直**（这正是粉噪的定义性特征）
        bands = result["bands"]
        self.assertLess(float(bands.max() - bands.min()), 6.0)

    def test_band_range_respects_f_min_and_f_max(self):
        x = generate_signal(SignalSpec(kind="pink", duration=1.0, fs=48000))
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_test_wav(Path(tmp) / "p.wav", x)
            result = analyze_file(path, f_min=100.0, f_max=1000.0)

        centers = result["centers"]
        self.assertTrue(centers.size > 0)
        self.assertGreaterEqual(float(centers.min()), 100.0)
        self.assertLessEqual(float(centers.max()), 1000.0)

    def test_order_controls_the_band_count(self):
        """1/1 倍频程的频带数必须明显少于 1/6 倍频程。"""
        x = generate_signal(SignalSpec(kind="pink", duration=1.0, fs=48000))
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_test_wav(Path(tmp) / "p.wav", x)
            coarse = analyze_file(path, order=1)["centers"]
            fine = analyze_file(path, order=6)["centers"]
        self.assertLess(coarse.size, fine.size)
        self.assertAlmostEqual(coarse.size * 6.0, fine.size, delta=1.0)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            analyze_file(Path(tempfile.gettempdir()) / "绝对不存在.wav")


class TestBandTable(unittest.TestCase):
    """:func:`format_band_table` —— 把频带结果排成对齐的文本表。"""

    def test_empty_input_is_reported_clearly(self):
        text = format_band_table(np.zeros(0), np.zeros(0))
        self.assertIn("没有", text)

    def test_line_count_matches_the_number_of_bands(self):
        centers = np.array([20.0, 25.0, 31.5])
        bands = np.array([-47.4, -50.1, -60.0])
        text = format_band_table(centers, bands)
        lines = [line for line in text.splitlines() if line.strip()]
        self.assertEqual(len(lines), 3)
        self.assertIn("-47.40", text)
        self.assertIn("20.00", text)


class TestReportFigure(unittest.TestCase):
    """双栏分析报告图。"""

    def setUp(self) -> None:
        matplotlib.use("Agg")

    def _sample_result(self):
        x = generate_signal(SignalSpec(kind="sine", duration=0.5, fs=48000))
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_test_wav(Path(tmp) / "r.wav", x)
            return analyze_file(path)

    def test_figure_has_two_axes(self):
        import matplotlib.pyplot as plt

        result = self._sample_result()
        fig = build_report_figure(result)
        try:
            self.assertEqual(len(fig.axes), 2)
        finally:
            plt.close(fig)

    def test_figure_has_a_suptitle_with_the_readings(self):
        import matplotlib.pyplot as plt

        result = self._sample_result()
        fig = build_report_figure(result)
        try:
            self.assertTrue(fig._suptitle is not None)
            text = fig._suptitle.get_text()
            self.assertIn(result["name"], text)
            self.assertIn("48000", text)
        finally:
            plt.close(fig)

    def test_report_file_writes_a_png(self):
        x = generate_signal(SignalSpec(kind="sine", duration=0.5, fs=48000))
        with tempfile.TemporaryDirectory() as tmp:
            wav = _write_test_wav(Path(tmp) / "r.wav", x)
            out = Path(tmp) / "report.png"
            path, result = report_file(wav, out)

            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 1000)
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(result["n_samples"], 24000)

    def test_report_file_creates_missing_parent_directories(self):
        x = generate_signal(SignalSpec(kind="sine", duration=0.2, fs=48000))
        with tempfile.TemporaryDirectory() as tmp:
            wav = _write_test_wav(Path(tmp) / "r.wav", x)
            out = Path(tmp) / "nested" / "deeper" / "report.png"
            path, _result = report_file(wav, out)
            self.assertTrue(path.is_file())


class TestMinMaxEnvelope(unittest.TestCase):
    """min/max 包络：时域绘图不能靠隔点抽稀（高频段必然混叠）。"""

    def test_short_signals_are_returned_unchanged(self):
        x = np.linspace(-1.0, 1.0, 100)
        t, lo, hi = minmax_envelope(x, 48000, 1600)
        self.assertEqual(t.size, 100)
        np.testing.assert_allclose(lo, x)
        np.testing.assert_allclose(hi, x)

    def test_long_signals_are_binned(self):
        x = np.zeros(48000)
        x[0] = 1.0
        x[-1] = -1.0
        t, lo, hi = minmax_envelope(x, 48000, 1600)
        self.assertEqual(t.size, 1600)
        self.assertAlmostEqual(float(hi.max()), 1.0)
        self.assertAlmostEqual(float(lo.min()), -1.0)

    def test_peak_is_preserved(self):
        """抽稀会漏掉峰值；包络必须把峰值留住。"""
        rng = np.random.default_rng(0)
        x = rng.normal(0.0, 0.2, 48000)
        x[12345] = 0.95
        _t, lo, hi = minmax_envelope(x, 48000, 1600)
        self.assertAlmostEqual(float(hi.max()), 0.95)


class TestCliAnalyzePlot(unittest.TestCase):
    """CLI：``analyze --plot`` 导出双栏报告图。"""

    def setUp(self) -> None:
        matplotlib.use("Agg")

    def test_plot_option_writes_a_png(self):
        x = generate_signal(SignalSpec(kind="sine", duration=0.5, fs=48000))
        with tempfile.TemporaryDirectory() as tmp:
            wav = _write_test_wav(Path(tmp) / "c.wav", x)
            out = Path(tmp) / "cli_report.png"
            code = cli_module.main(["analyze", str(wav), "--plot", str(out)])
            self.assertEqual(code, 0)
            self.assertTrue(out.is_file())
            self.assertGreater(out.stat().st_size, 1000)
            self.assertEqual(out.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_output_contains_the_readings_and_the_band_table(self):
        x = generate_signal(SignalSpec(kind="sine", duration=0.5, fs=48000))
        buffer = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            wav = _write_test_wav(Path(tmp) / "c.wav", x)
            with contextlib.redirect_stdout(buffer):
                code = cli_module.main(["analyze", str(wav)])
        self.assertEqual(code, 0)
        text = buffer.getvalue()
        self.assertIn("dBFS", text)
        self.assertIn("倍频程", text)
        self.assertIn("峰值因子", text)

    def test_missing_file_returns_two(self):
        missing = str(Path(tempfile.gettempdir()) / "绝对不存在.wav")
        self.assertEqual(cli_module.main(["analyze", missing]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
