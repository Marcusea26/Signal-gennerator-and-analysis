"""
新增功能的测试：WAV 读取、CLI 子命令。

与 :mod:`tests.test_signal_gen` 分开成独立文件 —— 前者是既有功能的
回归保护，这一组是**新功能**的验收，边界更清晰。

运行::

    python -m unittest discover -s tests
    python run_tests.py TestCli
"""

import importlib
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signal_gen import cli as cli_module
from signal_gen.audio_io import load_wav, save_wav, wav_info
from signal_gen.core import SignalSpec, generate_signal


def _write_wav(path: Path, data: bytes, *, channels: int = 1, fs: int = 48000,
               bits: int = 16, audio_format: int = 1, extra_fmt: bytes = b"") -> Path:
    """手工拼一个 WAV 文件（用于构造标准库 ``wave`` 写不出来的格式）。"""
    block_align = channels * max(1, bits // 8)
    byte_rate = fs * block_align
    fmt_body = struct.pack(
        "<HHIIHH",
        audio_format,
        channels,
        fs,
        byte_rate,
        block_align,
        bits,
    ) + extra_fmt

    chunks = b""
    chunks += b"fmt " + struct.pack("<I", len(fmt_body)) + fmt_body
    chunks += b"data" + struct.pack("<I", len(data)) + data
    if len(data) & 1:
        chunks += b"\x00"  # chunk 按偶数字节对齐

    blob = b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks
    path.write_bytes(blob)
    return path


class TestPackageExports(unittest.TestCase):
    """包级导出面：模块 ``__all__`` 必须都能从包顶层拿到。"""

    #: 参与漂移检查的子模块
    MODULES = ("core", "standards", "spectrum", "audio_io", "report", "fonts", "player")

    def test_every_module_export_is_reachable_from_package(self):
        import signal_gen

        pkg = set(signal_gen.__all__)
        for name in self.MODULES:
            module = importlib.import_module(f"signal_gen.{name}")
            for exported in getattr(module, "__all__", []):
                self.assertIn(
                    exported, pkg,
                    msg=f"signal_gen.{name}.__all__ 里的 {exported!r} 没有从包顶层导出",
                )

    def test_package_all_names_actually_exist(self):
        import signal_gen

        for name in signal_gen.__all__:
            self.assertTrue(hasattr(signal_gen, name), msg=f"{name!r} 导出了但不存在")

    def test_package_all_has_no_duplicates(self):
        import signal_gen

        names = signal_gen.__all__
        self.assertEqual(len(names), len(set(names)), msg="__all__ 里有重复项")


class TestLoadWav(unittest.TestCase):
    """任意格式 WAV 的读取。"""

    def test_pcm16_roundtrip(self):
        x = np.sin(2 * np.pi * 1000 * np.arange(4800) / 48000)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.wav"
            save_wav(x, path, fs=48000, bits=16, dither=False)
            y, fs = load_wav(path)
            self.assertEqual(fs, 48000)
            self.assertEqual(y.shape, x.shape)
            # 16 bit 的量化误差是 1 LSB
            self.assertLess(float(np.abs(y - x).max()), 2.0 / 32768.0)

    def test_pcm24_is_one_lsb_accurate(self):
        """24 bit 的满量程是 ±2**23 —— 除数用错会让电平差 256 倍。"""
        values = np.array([0.0, 0.5, -0.5, 0.25, -0.25])
        data = b"".join(
            int(round(v * 8388608.0)).to_bytes(3, "little", signed=True) for v in values
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_wav(Path(tmp) / "p24.wav", data, bits=24)
            y, fs = load_wav(path)
            self.assertEqual(fs, 48000)
            self.assertEqual(y.size, values.size)
            self.assertLess(float(np.abs(y - values).max()), 1.0 / 8388608.0)

    def test_pcm8_is_unsigned(self):
        """8 bit PCM 是无符号的（中点 128），不是有符号。"""
        data = bytes([128, 255, 0, 192])
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_wav(Path(tmp) / "p8.wav", data, bits=8)
            y, _fs = load_wav(path)
            np.testing.assert_allclose(y, [0.0, 127.0 / 128.0, -1.0, 64.0 / 128.0])

    def test_pcm32(self):
        values = np.array([0.0, 0.5, -0.5])
        data = b"".join(
            int(round(v * 2147483648.0)).to_bytes(4, "little", signed=True) for v in values
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_wav(Path(tmp) / "p32.wav", data, bits=32)
            y, _fs = load_wav(path)
            self.assertLess(float(np.abs(y - values).max()), 1.0 / 2147483648.0)

    def test_ieee_float32(self):
        values = np.array([0.0, 0.5, -0.5, 0.25])
        data = values.astype("<f4").tobytes()
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_wav(Path(tmp) / "f32.wav", data, bits=32, audio_format=3)
            y, _fs = load_wav(path)
            np.testing.assert_allclose(y, values, atol=1e-7)

    def test_ieee_float64(self):
        values = np.array([0.0, 0.5, -0.5, 0.25])
        data = values.astype("<f8").tobytes()
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_wav(Path(tmp) / "f64.wav", data, bits=64, audio_format=3)
            y, _fs = load_wav(path)
            np.testing.assert_allclose(y, values, atol=1e-15)

    def test_stereo_is_downmixed_to_mono(self):
        """多声道下混成单声道（各声道取平均）。"""
        left = np.array([1.0, 0.0, -1.0])
        right = np.array([0.0, 1.0, 0.0])
        interleaved = np.empty(6, dtype=np.float64)
        interleaved[0::2] = left
        interleaved[1::2] = right
        data = interleaved.astype("<f4").tobytes()
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_wav(
                Path(tmp) / "st.wav", data, channels=2, bits=32, audio_format=3
            )
            y, _fs = load_wav(path)
            self.assertEqual(y.ndim, 1)
            self.assertEqual(y.size, 3)
            np.testing.assert_allclose(y, (left + right) / 2.0)

    def test_mono_result_is_one_dimensional(self):
        """单声道不能残留成 (N, 1) 的二维形状。"""
        data = np.array([0.0, 0.5, -0.5]).astype("<f4").tobytes()
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_wav(Path(tmp) / "m.wav", data, bits=32, audio_format=3)
            y, _fs = load_wav(path)
            self.assertEqual(y.ndim, 1)
            self.assertEqual(y.shape, (3,))

    def test_wav_info_reports_format(self):
        data = np.zeros(100, dtype=np.float64).astype("<f4").tobytes()
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_wav(Path(tmp) / "i.wav", data, bits=32, audio_format=3)
            info = wav_info(path)
            self.assertEqual(info["sample_rate"], 48000)
            self.assertEqual(info["channels"], 1)
            self.assertEqual(info["bits"], 32)
            self.assertEqual(info["frames"], 100)
            self.assertAlmostEqual(info["duration"], 100 / 48000.0)
            self.assertIn("浮点", info["encoding"])

    def test_invalid_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.wav"
            path.write_bytes(b"NOTAWAVFILE")
            with self.assertRaises(ValueError):
                load_wav(path)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_wav(Path(tempfile.gettempdir()) / "绝对不存在.wav")


class TestCli(unittest.TestCase):
    """CLI 子命令与参数优先级。"""

    def test_parser_exposes_all_subcommands(self):
        parser = cli_module.build_parser()
        cases = {
            "list": ["list"],
            "generate": ["generate", "pink"],
            "plot": ["plot", "pink"],
            "analyze": ["analyze", "x.wav"],
        }
        for command, argv in cases.items():
            args = parser.parse_args(argv)
            self.assertTrue(hasattr(args, "func"), msg=f"{command} 没有绑定处理函数")

    def test_generate_requires_a_kind(self):
        """``generate`` 不带类型时 argparse 直接报错退出（SystemExit）。"""
        parser = cli_module.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["generate"])

    def test_spec_from_args_uses_type_defaults(self):
        """CLI 的默认参数必须来自 ``DEFAULT_PARAMS``（按信号类型取值）。

        ``iec268`` 现在是 Clause 7 的模拟节目信号：频带范围 = Table II 的
        31 个 1/3 倍频程频带（20 Hz – 20 kHz），峰值因子 0（不做幅度限制）。
        """
        parser = cli_module.build_parser()
        args = parser.parse_args(["generate", "iec268"])
        spec = cli_module._spec_from_args(args)
        self.assertEqual(spec.kind, "iec268")
        self.assertAlmostEqual(spec.f_start, 20.0)
        self.assertAlmostEqual(spec.f_stop, 20000.0)
        self.assertAlmostEqual(spec.crest_factor_db, 0.0)

    def test_spec_from_args_uses_type_defaults_for_pink_noise(self):
        """附录 C 带限粉噪：Butterworth 拐点 22.7 Hz / 22.4 kHz，峰值因子 6 dB。"""
        parser = cli_module.build_parser()
        args = parser.parse_args(["generate", "iec268_pink"])
        spec = cli_module._spec_from_args(args)
        self.assertEqual(spec.kind, "iec268_pink")
        self.assertAlmostEqual(spec.f_start, 22.7)
        self.assertAlmostEqual(spec.f_stop, 22400.0)
        self.assertAlmostEqual(spec.crest_factor_db, 6.0)

    def test_spec_from_args_cli_overrides_defaults(self):
        parser = cli_module.build_parser()
        args = parser.parse_args(["generate", "iec268", "--f-start", "50"])
        spec = cli_module._spec_from_args(args)
        self.assertAlmostEqual(spec.f_start, 50.0)

    def test_sine_frequency_is_consistent(self):
        """--f1 与 --f-start 在单频正弦下是同一个量，必须保持一致。"""
        parser = cli_module.build_parser()
        for argv in (["generate", "sine", "--f1", "500"],
                     ["generate", "sine", "--f-start", "500"]):
            spec = cli_module._spec_from_args(parser.parse_args(argv))
            self.assertAlmostEqual(spec.f1, 500.0)
            self.assertAlmostEqual(spec.f_start, 500.0)

    def test_sine_defaults_to_1000(self):
        parser = cli_module.build_parser()
        spec = cli_module._spec_from_args(parser.parse_args(["generate", "sine"]))
        self.assertAlmostEqual(spec.f1, 1000.0)
        self.assertAlmostEqual(spec.f_start, 1000.0)

    def test_harmonics_option_is_parsed(self):
        parser = cli_module.build_parser()
        args = parser.parse_args(["generate", "sine", "--harmonics", "0.5,0.25"])
        spec = cli_module._spec_from_args(args)
        self.assertEqual(spec.harmonics, [0.5, 0.25])

    def test_harmonics_option_rejects_garbage(self):
        parser = cli_module.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["generate", "sine", "--harmonics", "abc"])

    def test_analyze_reads_a_wav_file(self):
        x = generate_signal(SignalSpec(kind="pink", duration=1.0, fs=48000))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.wav"
            save_wav(x, path, fs=48000, bits=16)
            parser = cli_module.build_parser()
            args = parser.parse_args(["analyze", str(path)])
            self.assertEqual(args.func(args), 0)

    def test_analyze_reports_a_missing_file(self):
        """文件不存在时给出清晰错误并返回 2，而不是抛栈。"""
        missing = str(Path(tempfile.gettempdir()) / "绝对不存在.wav")
        self.assertEqual(cli_module.main(["analyze", missing]), 2)


class TestCliConsoleEncoding(unittest.TestCase):
    """Windows 中文控制台（GBK）下不能因为 ``²`` / ``⚠`` 崩掉。"""

    def test_configure_console_reconfigures_a_gbk_stream(self):
        """GBK 输出流必须被改成 UTF-8 —— 否则 ``1/f²`` 直接抛 UnicodeEncodeError。"""
        import io

        buf = io.BytesIO()
        wrapper = io.TextIOWrapper(buf, encoding="gbk", errors="strict")
        original = sys.stdout
        try:
            sys.stdout = wrapper
            cli_module._configure_console()
            sys.stdout.write("1/f²  ⚠\n")
            sys.stdout.flush()
        finally:
            sys.stdout = original

        # 改写后的编码必须是 UTF-8（而不是原来的 GBK）
        self.assertEqual(wrapper.encoding.lower().replace("-", ""), "utf8")
        text = buf.getvalue().decode("utf-8")
        self.assertIn("1/f²", text)
        self.assertIn("⚠", text)

    def test_list_command_survives_gbk_console(self):
        """完整跑一遍 ``list``：改动前这里会因 ``²`` 崩掉整个命令。"""
        import io

        buf = io.BytesIO()
        wrapper = io.TextIOWrapper(buf, encoding="gbk", errors="strict")
        original = sys.stdout
        try:
            sys.stdout = wrapper
            code = cli_module.main(["list"])
            wrapper.flush()
        finally:
            sys.stdout = original

        self.assertEqual(code, 0)
        text = buf.getvalue().decode("utf-8")
        self.assertIn("1/f²", text)


if __name__ == "__main__":
    unittest.main()
