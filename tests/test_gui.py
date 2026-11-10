"""
GUI 层的自动化测试：谐波输入、规格回填、外部 WAV 分析。

GUI 需要 tkinter 与显示环境，跑不起来时**跳过**而不是让整套测试失败
—— 这样在没有桌面环境的 CI 上也能跑。

运行::

    python -m unittest discover -s tests
    python run_tests.py TestGui
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signal_gen.audio_io import save_wav
from signal_gen.core import KIND_LABELS, SignalSpec, generate_signal
from signal_gen import gui as gui_module
from signal_gen.gui import MODE_FILE, MODE_GENERATE


class _GuiTestCase(unittest.TestCase):
    """提供可用的 :class:`SignalGeneratorApp` 实例（跑不起来就跳过）。"""

    def setUp(self) -> None:
        try:
            from signal_gen.gui import SignalGeneratorApp

            self.app = SignalGeneratorApp()
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"无法创建 GUI：{exc}")
        self.app.withdraw()

    def tearDown(self) -> None:
        try:
            self.app.destroy()
        except Exception:  # noqa: BLE001
            pass


class TestGuiSpecRoundTrip(_GuiTestCase):
    """规格 → 界面 → 再收回来，必须**无损**往返。"""

    def test_spec_roundtrip_through_the_gui_is_lossless(self):
        """把规格灌进界面再收回来，各字段必须完全一致。"""
        app = self.app
        original = SignalSpec(
            kind="sine", fs=44100, duration=3.5, seed=11,
            f_start=1234.0, amplitude=0.4, harmonics=[0.5, 0.125],
            normalize="rms", target_dbfs=-21.0,
        )

        app._updating = True
        app._apply_spec_to_vars(original)
        restored, _notes = app._collect_spec()

        self.assertEqual(restored.kind, original.kind)
        self.assertEqual(restored.fs, original.fs)
        self.assertEqual(restored.duration, original.duration)
        self.assertEqual(restored.seed, original.seed)
        self.assertAlmostEqual(restored.f1, original.f1)
        self.assertAlmostEqual(restored.amplitude, original.amplitude)
        self.assertEqual(restored.harmonics, original.harmonics)
        self.assertEqual(restored.normalize, original.normalize)
        self.assertAlmostEqual(restored.target_dbfs, original.target_dbfs)

    def test_harmonics_are_only_used_by_sine(self):
        """谐波只对单频正弦有效，换成别的类型就不该再带上。"""
        app = self.app
        app._updating = True
        app._apply_spec_to_vars(
            SignalSpec(kind="sine", f_start=1000.0, harmonics=[0.5])
        )
        self.assertEqual(str(app._vars["harmonics"].get()), "0.5")

        app.kind_var.set(KIND_LABELS["pink"])
        app._on_kind_change()
        spec, _notes = app._collect_spec()
        self.assertEqual(spec.harmonics, [])

    def test_invalid_harmonics_text_is_reported(self):
        app = self.app
        app._updating = True
        app.kind_var.set(KIND_LABELS["sine"])
        app._on_kind_change()
        app._vars["harmonics"].set("0.5, abc")
        with self.assertRaises(ValueError):
            app._collect_spec()


class TestGuiAnalyzeExternalWav(_GuiTestCase):
    """「打开 WAV」：外部文件走的是和生成信号完全相同的分析路径。"""

    def test_analyze_payload_matches_the_generated_signal(self):
        """对同一个波形，生成路径与分析路径必须给出相同的频域结果。"""
        from signal_gen.gui import analyze_payload, compute_payload

        spec = SignalSpec(kind="pink", duration=1.0, fs=48000)
        signal = generate_signal(spec)

        generated = compute_payload(spec, "hann", False)
        analyzed = analyze_payload(signal, spec.fs, "hann", False)

        np.testing.assert_allclose(analyzed["db"], generated["db"], atol=1e-9)
        np.testing.assert_allclose(analyzed["bands"], generated["bands"], atol=1e-9)
        self.assertEqual(analyzed["fs"], spec.fs)
        self.assertEqual(analyzed["n_samples"], signal.size)

    def test_file_worker_produces_a_usable_payload(self):
        """后台工作线程读文件 → 分析 → 回填结果。"""
        from signal_gen.gui import analyze_payload

        x = generate_signal(SignalSpec(kind="sine", duration=0.5, fs=48000))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.wav"
            save_wav(x, path, fs=48000, bits=16)

            payload = analyze_payload(x, 48000, "hann", False)
            payload["spec"] = None
            payload["title"] = path.name

            # 回填到界面：绘图与统计都只依赖 payload，不依赖 spec
            self.app._apply_payload(payload)
            self.assertIsNone(self.app._spec)
            self.assertIsNotNone(self.app._signal)
            self.assertIn(path.name, str(self.app.stats_label.cget("text")))

    def test_external_wav_analysis_keeps_the_sample_count(self):
        from signal_gen.gui import analyze_payload

        x = np.zeros(1234, dtype=np.float64)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "y.wav"
            save_wav(x, path, fs=48000, bits=16)

            from signal_gen.audio_io import load_wav

            signal, fs = load_wav(path)
            payload = analyze_payload(signal, fs, "hann", False)
            self.assertEqual(payload["n_samples"], 1234)
            self.assertEqual(payload["fs"], 48000)


class TestGuiExportReport(_GuiTestCase):
    """「导出分析图」按钮：导出双栏报告图（时域 + 频域含 1/6 倍频程）。"""

    def test_export_report_writes_a_png(self):
        """按钮必须真的落盘一张 PNG，而不只是弹出保存对话框。"""
        from signal_gen import gui as gui_module

        x = generate_signal(SignalSpec(kind="sine", duration=0.5, fs=48000))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "gui_report.png"

            payload = gui_module.analyze_payload(x, 48000, "hann", False)
            payload["spec"] = None
            payload["title"] = "x.wav"
            self.app._apply_payload(payload)

            original_ask = gui_module.filedialog.asksaveasfilename
            original_info = gui_module.messagebox.showinfo
            try:
                gui_module.filedialog.asksaveasfilename = lambda *a, **k: str(out)
                gui_module.messagebox.showinfo = lambda *a, **k: None
                self.app._export_report()
            finally:
                gui_module.filedialog.asksaveasfilename = original_ask
                gui_module.messagebox.showinfo = original_info

            self.assertTrue(out.is_file(), msg="没有生成 PNG")
            self.assertGreater(out.stat().st_size, 1000)
            self.assertEqual(out.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_export_report_without_data_warns_instead_of_crashing(self):
        """还没生成信号就点按钮：给提示，不能抛栈。"""
        from signal_gen import gui as gui_module

        original_warn = gui_module.messagebox.showwarning
        try:
            gui_module.messagebox.showwarning = lambda *a, **k: None
            self.app._export_report()
        finally:
            gui_module.messagebox.showwarning = original_warn

    def test_minmax_envelope_is_a_single_source(self):
        """``gui.minmax_envelope`` 与 ``report.minmax_envelope`` 必须是同一个函数。"""
        from signal_gen import gui as gui_module
        from signal_gen import report as report_module

        self.assertIs(gui_module.minmax_envelope, report_module.minmax_envelope)


class TestGuiModes(_GuiTestCase):
    """双工作模式：生成信号 / 分析文件。"""

    def _make_wav(self, tmp: str, name: str = "m.wav") -> Path:
        x = generate_signal(SignalSpec(kind="sine", duration=0.5, fs=48000))
        path = Path(tmp) / name
        save_wav(x, path, fs=48000, bits=16)
        return path

    @staticmethod
    def _packed(widget) -> bool:
        """控件当前是否处于显示状态（未 pack 就不是）。"""
        return bool(widget.winfo_manager())

    def test_generation_mode_shows_the_generation_panel(self):
        app = self.app
        self.assertTrue(self._packed(app._gen_body))
        self.assertFalse(self._packed(app._file_body))
        self.assertEqual(app._mode, MODE_GENERATE)

    def test_switching_to_file_mode_swaps_the_panels(self):
        """切到「分析文件」：生成参数隐藏、文件信息显示。"""
        app = self.app
        app.mode_var.set(MODE_FILE)
        app._on_mode_change()

        self.assertEqual(app._mode, MODE_FILE)
        self.assertFalse(self._packed(app._gen_body), msg="生成参数不该再显示")
        self.assertTrue(self._packed(app._file_body), msg="文件信息该显示出来")

    def test_switching_back_restores_the_generation_panel(self):
        app = self.app
        app.mode_var.set(MODE_FILE)
        app._on_mode_change()
        app.mode_var.set(MODE_GENERATE)
        app._on_mode_change()

        self.assertEqual(app._mode, MODE_GENERATE)
        self.assertTrue(self._packed(app._gen_body))
        self.assertFalse(self._packed(app._file_body))

    def test_regenerate_button_is_disabled_in_file_mode(self):
        """文件模式下没有"生成参数"，「重新生成」不该可用。"""
        app = self.app
        app.mode_var.set(MODE_FILE)
        app._on_mode_change()
        self.assertEqual(str(app.btn_regen.cget("state")), "disabled")

        app.mode_var.set(MODE_GENERATE)
        app._on_mode_change()
        self.assertEqual(str(app.btn_regen.cget("state")), "normal")

    def test_file_mode_never_regenerates_a_signal(self):
        """**核心回归**：文件模式下改分析选项，必须重算文件而不是生成新信号。

        旧行为下这里会走 ``_worker``（生成信号），把打开的文件顶掉。
        """
        app = self.app
        calls = []
        app._file_worker = lambda *a, **k: calls.append("file")
        app._worker = lambda *a, **k: calls.append("gen")

        with tempfile.TemporaryDirectory() as tmp:
            app._file_path = str(self._make_wav(tmp))
            app._mode = MODE_FILE
            app._start_job()

        self.assertEqual(calls, ["file"], msg="文件模式应当重算文件，而不是重新生成信号")

    def test_file_mode_without_a_file_does_not_generate_anything(self):
        """文件模式但没打开文件时，不能凭空生成一个信号顶替。"""
        app = self.app
        calls = []
        app._file_worker = lambda *a, **k: calls.append("file")
        app._worker = lambda *a, **k: calls.append("gen")

        app._mode = MODE_FILE
        app._file_path = None
        app._start_job()

        self.assertEqual(calls, [])

    def test_source_badge_distinguishes_the_two_modes(self):
        app = self.app
        app.mode_var.set(MODE_FILE)
        app._on_mode_change()
        self.assertEqual(app.source_label.cget("text"), "文件")

        app.mode_var.set(MODE_GENERATE)
        app._on_mode_change()
        self.assertEqual(app.source_label.cget("text"), "生成")

    def test_file_info_panel_shows_the_opened_file(self):
        """打开文件后，左栏「文件信息」要显示格式 / 采样率 / 声道 / 时长。"""
        app = self.app
        with tempfile.TemporaryDirectory() as tmp:
            path = self._make_wav(tmp, "面板.wav")
            app._file_path = str(path)
            app._file_info = {
                "format": "WAV", "sample_rate": 48000, "channels": 1,
                "bits": 16, "encoding": "PCM", "duration": 0.5, "frames": 24000,
            }
            app._update_file_panel()

            self.assertIn("面板.wav", app._file_name_label.cget("text"))
            meta = app._file_meta_label.cget("text")
            self.assertIn("48000", meta)
            self.assertIn("16 bit", meta)
            self.assertIn("0.500", meta)

    def test_failed_open_falls_back_to_generation_mode(self):
        """**首次打开**失败要退回生成模式。

        否则徽章写着"文件"、图表却还是旧的生成结果 —— 两边对不上，
        用户会以为自己在看那个（根本没打开成功的）文件。
        """
        app = self.app
        errors = []
        original = gui_module.messagebox.showerror
        try:
            gui_module.messagebox.showerror = lambda *a, **k: errors.append(a)
            app.mode_var.set(MODE_FILE)
            app._on_mode_change()
            app._pending_open = True
            app._job_running = True
            app._queue.put((app._job_id, None, (ValueError("坏文件"), "tb")))
            app._poll_results()
        finally:
            gui_module.messagebox.showerror = original

        self.assertEqual(app._mode, MODE_GENERATE)
        self.assertTrue(self._packed(app._gen_body))
        self.assertFalse(self._packed(app._file_body))
        self.assertEqual(app.source_label.cget("text"), "生成")
        self.assertEqual(str(app.btn_regen.cget("state")), "normal")
        self.assertTrue(errors)


class TestGuiOctaveOrder(_GuiTestCase):
    """倍频程阶数：GUI 与 CLI 的 ``--order`` 同义。"""

    def test_order_is_carried_into_the_payload(self):
        from signal_gen.gui import analyze_payload

        x = generate_signal(SignalSpec(kind="pink", duration=1.0, fs=48000))
        payload = analyze_payload(x, 48000, "hann", False, order=3)
        self.assertEqual(payload["order"], 3)

    def test_order_changes_the_number_of_bands(self):
        from signal_gen.gui import analyze_payload

        x = generate_signal(SignalSpec(kind="pink", duration=1.0, fs=48000))
        coarse = analyze_payload(x, 48000, "hann", False, order=1)["centers"]
        fine = analyze_payload(x, 48000, "hann", False, order=6)["centers"]
        self.assertLess(coarse.size, fine.size)

    def test_default_order_is_one_sixth(self):
        self.assertEqual(self.app._current_order(), 6)

    def test_selected_order_is_read_back(self):
        app = self.app
        app._vars["order"].set("1/3")
        self.assertEqual(app._current_order(), 3)
        app._vars["order"].set("1/12")
        self.assertEqual(app._current_order(), 12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
