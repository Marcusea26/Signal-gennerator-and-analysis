"""
图形界面（tkinter + matplotlib）。

功能：
    * 左侧参数面板随信号类型动态启用/禁用控件，支持鼠标滚轮滚动
    * 右侧三个标签页：时域波形 / 频谱（含 1/6 倍频程曲线）/ 频谱图(STFT)
    * 底部实时统计：峰值 / RMS / 峰值因子 / 削波样点
    * **一键播放**生成的信号（Windows 内置 winsound 异步播放，播完自动停止）
    * 一键导出 WAV（16 或 24 bit，可开关抖动）

性能设计
--------
第三轮针对"界面卡顿、不流畅"做了 6 项优化：

1. **后台线程计算**：信号生成 + 频谱 + STFT 全部在工作线程完成，
   主线程只负责绘图，界面在计算期间依然可以拖动窗口、切换标签页。
2. **单次 FFT**：原始频谱与 1/6 倍频程电平共用同一次 ``rfft``
   （旧版调用两次，等于白白多算一遍）。
3. **searchsorted 分频带**：倍频程频带归属用二分定位，
   取代旧版"每个频带做一次全数组布尔掩码"（62 × 240 000 次比较）。
4. **去抖**：参数变化后 220 ms 内没有新输入才真正重算，
   连续拖动滑块不会触发一串重复计算。
5. **只重绘可见标签页**：旧版每次重绘全部 4 页，现按需重绘，
   切到某页时才补画那一页。
6. **draw_idle + layout engine**：``canvas.draw_idle()`` 合并同一帧内的
   重绘请求；布局交给 figure 的 layout engine，省掉每次 ``tight_layout()``
   的额外文本测量。

频率轴默认显示 **20 Hz – 20 kHz**（不超过当前采样率的奈奎斯特频率）。

启动::

    python -m signal_gen.gui
    python gui.py
"""

from __future__ import annotations

import queue
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from .fonts import configure_chinese_font

configure_chinese_font()

from .audio_io import (
    BITS_CHOICES,
    audio_info,
    load_audio,
    save_wav,
    signal_stats,
)
from .core import (
    ALL_KINDS,
    DEFAULT_PARAMS,
    FREQ_PARAM_INFO,
    KIND_DESCRIPTIONS,
    KIND_LABELS,
    KIND_PARAMS,
    PARAM_SPECS,
    SignalSpec,
    clamp_spec_to_nyquist,
    generate_signal,
)
from .player import PLAYBACK_BACKEND, AudioPlayer
from .report import build_report_figure, minmax_envelope
from .spectrum import (
    DEFAULT_WINDOW,
    WINDOW_LABELS,
    WINDOWS,
    band_levels_dbfs,
    one_sided_spectrum_dbfs_power,
    stft_dbfs,
    window_metrics,
)

__all__ = [
    "SignalGeneratorApp",
    "main",
    "minmax_envelope",
    "TAB_KEYS",
    "MODE_GENERATE",
    "MODE_FILE",
    "OCTAVE_ORDERS",
]

#: 三个标签页的内部标识（1/6 倍频程曲线叠在频谱页上，不再单独占一页）
TAB_KEYS: Tuple[str, ...] = ("time", "spec", "stft")

#: 频率轴显示范围（默认 20 Hz – 20 kHz）
VIEW_F_LO = 20.0
VIEW_F_HI = 20000.0

#: 计算窗函数指标用的参考长度（栅栏损失等指标与长度基本无关）
_WINDOW_METRIC_N = 8192

#: 工作模式 —— **生成信号**（左栏是生成参数）
MODE_GENERATE = "generate"
#: 工作模式 —— **分析文件**（左栏是文件信息）
MODE_FILE = "file"

#: 可选的倍频程阶数（与 CLI 的 ``--order`` 一致）
OCTAVE_ORDERS = (1, 3, 6, 12, 24)

#: 倍频程阶数的中文名
_ORDER_LABELS = {1: "1/1", 3: "1/3", 6: "1/6", 12: "1/12", 24: "1/24"}

#: 倍频程阶数中文名 → 阶数（反查表）
_ORDER_BY_LABEL = {label: order for order, label in _ORDER_LABELS.items()}


# ---------------------------------------------------------------------------
# 后台计算
# ---------------------------------------------------------------------------

def _compute_stft(signal: np.ndarray, fs: int, window: str = "hann"):
    """计算 STFT（dBFS），在工作线程中调用。"""
    return stft_dbfs(signal, fs, window=window)


def analyze_payload(
    signal: np.ndarray,
    fs: int,
    window: str,
    show_stft: bool,
    order: int = 6,
) -> Dict[str, Any]:
    """对**已有的一段波形**做全部频域分析（供工作线程调用）。

    这是 :func:`compute_payload` 的后半段 —— 抽出来是为了让**外部音频文件**
    也能走完全相同的分析路径：同一套窗函数、同一条 1/6 倍频程曲线、
    同一张 STFT 图，分析口径与生成的信号完全一致。

    参数
    ----
    order
        倍频程阶数，6 表示 1/6 倍频程（与 CLI 的 ``--order`` 同义）。
    """
    # 一次 FFT 同时得到 dBFS 幅度谱与功率谱
    freqs, db, power = one_sided_spectrum_dbfs_power(signal, fs, window=window)

    f_max = min(fs / 2.0, 24000.0)
    centers, bands = band_levels_dbfs(
        freqs, power, f_min=VIEW_F_LO, f_max=f_max, order=int(order)
    )

    stft = _compute_stft(signal, fs, window) if show_stft else None

    return {
        "signal": signal,
        "fs": int(fs),
        "n_samples": int(signal.size),
        "freqs": freqs,
        "db": db,
        "centers": centers,
        "bands": bands,
        "stft": stft,
        "stats": signal_stats(signal),
        "order": int(order),
    }


def compute_payload(
    spec: SignalSpec,
    window: str,
    show_stft: bool,
    order: int = 6,
) -> Dict[str, Any]:
    """生成信号并完成全部频域分析（供工作线程调用）。"""
    signal = generate_signal(spec)
    payload = analyze_payload(signal, spec.fs, window, show_stft, order=order)
    payload["spec"] = spec
    payload["title"] = KIND_LABELS[spec.kind]
    return payload


# ---------------------------------------------------------------------------
# 主窗口
# ---------------------------------------------------------------------------

class SignalGeneratorApp(tk.Tk):
    """信号生成器主窗口。"""

    #: 参数变化后的去抖延迟 (ms)
    DEBOUNCE_MS = 180
    #: 后台结果轮询间隔 (ms)
    POLL_MS = 30
    #: 播放状态轮询间隔 (ms)
    PLAY_POLL_MS = 100
    #: 时域图的列数上限（每列画 min/max 包络）
    TIME_COLUMNS = 1600

    def __init__(self) -> None:
        super().__init__()
        self.title("信号生成与分析 · signal_gen")
        self.geometry("1360x840")
        self.minsize(1100, 680)

        self._signal: Optional[np.ndarray] = None
        self._spec: Optional[SignalSpec] = None
        self._notes: List[str] = []
        self._result: Optional[Dict[str, Any]] = None
        self._vars: Dict[str, tk.Variable] = {}
        self._param_widgets: Dict[str, tk.Widget] = {}
        self._updating = False

        # ---- 工作模式（生成信号 / 分析文件）----
        #: 当前工作模式：:data:`MODE_GENERATE` 或 :data:`MODE_FILE`
        self._mode = MODE_GENERATE
        #: 最近一次打开的音频文件路径
        self._file_path: Optional[str] = None
        #: 最近一次**文件**分析结果（切回文件模式时直接复用，不必重读）
        self._file_payload: Optional[Dict[str, Any]] = None
        #: 已打开文件的格式信息
        self._file_info: Optional[Dict[str, Any]] = None
        #: 正在"首次打开"文件 —— 打开失败时要退回生成模式
        self._pending_open = False

        # ---- 后台任务 ----
        self._queue: "queue.Queue[Tuple[int, Optional[Dict[str, Any]], Optional[Tuple]]]" = (
            queue.Queue()
        )
        self._job_id = 0
        self._job_running = False
        #: 已应用的结果计数（每应用一次 +1），便于外部判断"是否有新结果"
        self._result_seq = 0
        self._poll_id: Optional[str] = None
        self._debounce_id: Optional[str] = None

        # ---- 绘图状态 ----
        self._dirty: Set[str] = set(TAB_KEYS)
        #: STFT 色条（常驻复用，避免每次重绘都新建坐标轴）
        self._stft_cbar = None
        #: 左侧滚动容器（供滚轮事件调用）
        self._scroll_canvas: Optional[tk.Canvas] = None

        # ---- 播放 ----
        self.player = AudioPlayer()
        self._play_poll_id: Optional[str] = None

        self._build_style()
        self._build_layout()
        self._update_window_hint()
        self._on_kind_change()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # 样式与布局
    # ------------------------------------------------------------------
    def _build_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TLabel", font=("Microsoft YaHei UI", 9))
        style.configure("TButton", font=("Microsoft YaHei UI", 9))
        style.configure("TCheckbutton", font=("Microsoft YaHei UI", 9))
        style.configure("Header.TLabelframe.Label", font=("Microsoft YaHei UI", 10, "bold"))
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 14, "bold"))
        style.configure("Hint.TLabel", foreground="#666666", font=("Microsoft YaHei UI", 8))
        style.configure("Warn.TLabel", foreground="#c0392b", font=("Microsoft YaHei UI", 8))
        # 来源徽章：区分"生成的信号"与"打开的文件"
        style.configure("Source.TLabel", foreground="#1f6fb2", font=("Microsoft YaHei UI", 9, "bold"))
        # 文件信息里的路径等次要文本
        style.configure("Path.TLabel", foreground="#555555", font=("Microsoft YaHei UI", 8))

    def _build_layout(self) -> None:
        # 顶部标题栏
        header = ttk.Frame(self, padding=(14, 10, 14, 6))
        header.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(header, text="信号生成与分析", style="Title.TLabel").pack(side=tk.LEFT)
        ttk.Label(
            header,
            text="  可复现随机 · dBFS 绝对刻度 · 1/6 倍频程频带功率 · 频率轴 20 Hz–20 kHz",
            style="Hint.TLabel",
        ).pack(side=tk.LEFT)

        main = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))

        # ---- 左侧参数面板 ----
        left = ttk.Frame(main, width=330)
        main.add(left, weight=0)

        # 滚动容器，窗口变小时参数仍可访问
        canvas = tk.Canvas(left, highlightthickness=0, borderwidth=0, width=330)
        scroll = ttk.Scrollbar(left, orient=tk.VERTICAL, command=canvas.yview)
        inner = ttk.Frame(canvas)

        inner.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.bind(
            "<Configure>",
            lambda e: canvas.itemconfigure(win_id, width=e.width),
        )
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._scroll_canvas = canvas

        self._build_param_panel(inner)

        # 滚轮控制左侧面板滚动
        self._bind_panel_wheel(canvas)
        self._bind_panel_wheel(inner)

        # ---- 右侧图表区 ----
        right = ttk.Frame(main)
        main.add(right, weight=1)

        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.fig_time = Figure(figsize=(7, 4), dpi=100)
        self.ax_time = self.fig_time.add_subplot(111)
        self.canvas_time = self._add_tab(self.notebook, "时域波形", self.fig_time)

        self.fig_spec = Figure(figsize=(7, 4), dpi=100)
        self.ax_spec = self.fig_spec.add_subplot(111)
        self.canvas_spec = self._add_tab(self.notebook, "频谱 (dBFS)", self.fig_spec)

        self.fig_stft = Figure(figsize=(7, 4), dpi=100)
        self.ax_stft = self.fig_stft.add_subplot(111)
        self.canvas_stft = self._add_tab(self.notebook, "频谱图 STFT", self.fig_stft)

        #: 标签页 → 图，供「保存当前视图 PNG」定位
        self._tab_figures = {
            "time": self.fig_time,
            "spec": self.fig_spec,
            "stft": self.fig_stft,
        }

        # 只在切换标签页时补画尚未更新的那一页
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        # ---- 底部操作栏 ----
        bottom = ttk.Frame(self, padding=(14, 4, 14, 12))
        bottom.pack(side=tk.BOTTOM, fill=tk.X)

        # 第一行：来源徽章 + 读数 + 状态/告警
        info_row = ttk.Frame(bottom)
        info_row.pack(side=tk.TOP, fill=tk.X, pady=(0, 6))

        self.source_label = ttk.Label(info_row, text="生成", style="Source.TLabel")
        self.source_label.pack(side=tk.LEFT)

        self.stats_label = ttk.Label(info_row, text="—", style="Hint.TLabel")
        self.stats_label.pack(side=tk.LEFT, padx=(8, 0))

        self.warn_label = ttk.Label(info_row, text="", style="Warn.TLabel")
        self.warn_label.pack(side=tk.RIGHT, padx=(10, 0))

        self.state_label = ttk.Label(info_row, text="", style="Hint.TLabel")
        self.state_label.pack(side=tk.RIGHT)

        # 第二行：按钮（按语义分组，用分隔线隔开）
        btns = ttk.Frame(bottom)
        btns.pack(side=tk.TOP, fill=tk.X)

        self.btn_open = ttk.Button(btns, text="打开音频文件…", command=self._open_audio)
        self.btn_open.pack(side=tk.LEFT, padx=(0, 4))

        self.btn_regen = ttk.Button(
            btns, text="重新生成", command=lambda: self._regenerate(immediate=True)
        )
        self.btn_regen.pack(side=tk.LEFT, padx=4)

        ttk.Separator(btns, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        self.btn_export_wav = ttk.Button(btns, text="导出 WAV", command=self._export_wav)
        self.btn_export_wav.pack(side=tk.LEFT, padx=4)

        self.btn_save_png = ttk.Button(
            btns, text="保存当前视图 PNG", command=self._save_figure
        )
        self.btn_save_png.pack(side=tk.LEFT, padx=4)

        self.btn_export_report = ttk.Button(
            btns, text="导出分析图", command=self._export_report
        )
        self.btn_export_report.pack(side=tk.LEFT, padx=4)

        # ---- 播放控件（靠右）----
        play_box = ttk.Frame(btns)
        play_box.pack(side=tk.RIGHT)

        ttk.Label(play_box, text="音量", style="Hint.TLabel").pack(side=tk.LEFT, padx=(0, 3))
        self._vars["play_gain"] = tk.IntVar(value=50)
        ttk.Spinbox(
            play_box, from_=0, to=100, increment=5,
            textvariable=self._vars["play_gain"], width=4,
        ).pack(side=tk.LEFT)

        self.play_button = ttk.Button(play_box, text="▶ 播放", command=self._play)
        self.play_button.pack(side=tk.LEFT, padx=(8, 0))

        self.stop_button = ttk.Button(play_box, text="■ 停止", command=self._stop_play)
        self.stop_button.pack(side=tk.LEFT, padx=(4, 0))

        if not self.player.supported:
            self.play_button.configure(state="disabled")
            self.stop_button.configure(state="disabled")
            self.warn_label.configure(
                text=f"（播放不可用：当前系统无 {PLAYBACK_BACKEND} 后端）"
            )

    def _add_tab(self, notebook: ttk.Notebook, title: str, fig: Figure) -> FigureCanvasTkAgg:
        frame = ttk.Frame(notebook)
        notebook.add(frame, text=title)
        # 布局交给 layout engine，省掉每次绘制时的 tight_layout() 测量
        try:
            fig.set_layout_engine("tight")
        except Exception:  # noqa: BLE001 - 老版本 matplotlib 兜底
            pass
        canvas = FigureCanvasTkAgg(fig, master=frame)
        toolbar = NavigationToolbar2Tk(canvas, frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        return canvas

    # ------------------------------------------------------------------
    # 参数面板
    # ------------------------------------------------------------------
    def _build_param_panel(self, parent: ttk.Frame) -> None:
        """构建左侧面板：**模式切换 + 三个可切换区块**。

        面板按工作模式切换可见性：

        * 「生成信号」→ 显示生成参数（信号类型 / 通用参数 / 类型专属参数）；
        * 「分析文件」→ 显示文件信息（格式 / 采样率 / 声道 / 位深 / 时长）。

        「分析选项」（窗函数 / 倍频程阶数 / STFT / 导出设置）**两种模式都显示** ——
        分析外部文件用的就是同一套窗函数与同一条倍频程曲线。
        """
        # ---- 工作模式切换 ----
        mode_box = ttk.LabelFrame(parent, text="工作模式", padding=10)
        mode_box.pack(fill=tk.X, padx=8, pady=(8, 4))
        mode_box.columnconfigure(1, weight=1)

        self.mode_var = tk.StringVar(value=MODE_GENERATE)
        ttk.Radiobutton(
            mode_box, text="生成信号", value=MODE_GENERATE,
            variable=self.mode_var, command=self._on_mode_change,
        ).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            mode_box, text="分析文件", value=MODE_FILE,
            variable=self.mode_var, command=self._on_mode_change,
        ).grid(row=0, column=1, sticky="w")

        self.mode_hint = ttk.Label(mode_box, text="", style="Hint.TLabel", wraplength=300)
        self.mode_hint.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))

        # ---- 生成参数（仅「生成信号」模式可见）----
        self._gen_body = ttk.Frame(parent)
        self._gen_body.pack(fill=tk.X)
        self._build_gen_panel(self._gen_body)

        # ---- 文件信息（仅「分析文件」模式可见）----
        self._file_body = ttk.Frame(parent)
        self._file_body.pack(fill=tk.X)
        self._build_file_panel(self._file_body)
        self._file_body.pack_forget()

        # ---- 分析选项（两种模式都适用）----
        self._analysis_box = ttk.LabelFrame(parent, text="分析选项", padding=10)
        self._analysis_box.pack(fill=tk.X, padx=8, pady=4)
        self._build_analysis_options(self._analysis_box)

        # ---- 提示文字 ----
        ttk.Label(
            parent,
            text=(
                "提示：修改参数后自动重算（带去抖），也可按 Enter 或点「重新生成」。\n"
                "左侧面板可用鼠标滚轮上下滚动。\n"
                "起始 / 终止频率：所有类型都能设，含义随类型变化（见标签）。\n"
                "渐变比例：连续扫频按总时长计，步进扫频按每步时长计（只作用于首尾）。\n"
                "渐变会让扫频信号的 STFT 频谱图首尾变暗 —— 这是抑制边界咔哒声的\n"
                "代价，把渐变比例调成 0 可以去掉，但首尾会出现不连续。\n"
                "标准信号：EIA RS-426-B / IEC 60268-1 Clause 7 的频谱"
                "按标准整形；峰值因子 0 表示不限幅。\n"
                "IEC 60268-1 Clause 7 的模拟节目信号按 Table II 逐带整形，"
                "标准要求“不做幅度限制”，所以默认峰值因子为 0。\n"
                "电平口径：满幅正弦为 0 dBFS；倍频程曲线是频带总功率"
                "（粉噪下应为平线）。"
            ),
            style="Hint.TLabel",
            wraplength=300,
        ).pack(fill=tk.X, padx=14, pady=(2, 10))

    def _build_gen_panel(self, parent: ttk.Frame) -> None:
        """生成参数：信号类型 + 通用参数 + 类型专属参数。"""
        # 信号类型
        box = ttk.LabelFrame(parent, text="信号类型", padding=10)
        box.pack(fill=tk.X, padx=8, pady=(8, 4))
        box.columnconfigure(0, weight=1)

        # 初始值必须与下拉框的显示格式一致，否则下拉框显示空白
        self.kind_var = tk.StringVar(value=KIND_LABELS["pink"])
        kind_box = ttk.Combobox(
            box,
            textvariable=self.kind_var,
            values=[KIND_LABELS[k] for k in ALL_KINDS],
            state="readonly",
            width=30,
        )
        kind_box.grid(row=0, column=0, sticky="ew")
        kind_box.bind("<<ComboboxSelected>>", lambda e: self._on_kind_change())
        self.kind_box = kind_box

        self.kind_hint = ttk.Label(box, text="", style="Hint.TLabel", wraplength=300)
        self.kind_hint.grid(row=1, column=0, sticky="w", pady=(4, 0))

        # 通用参数
        gen = ttk.LabelFrame(parent, text="通用参数", padding=10)
        gen.pack(fill=tk.X, padx=8, pady=4)
        gen.columnconfigure(1, weight=1)

        row = 0
        # 采样率
        self._vars["fs"] = tk.IntVar(value=48000)
        ttk.Label(gen, text=PARAM_SPECS["fs"]["label"]).grid(
            row=row, column=0, sticky="w", pady=2
        )
        cb = ttk.Combobox(
            gen,
            textvariable=self._vars["fs"],
            values=[str(v) for v in PARAM_SPECS["fs"]["choices"]],
            state="readonly",
            width=12,
        )
        cb.grid(row=row, column=1, sticky="ew", pady=2)
        cb.bind("<<ComboboxSelected>>", lambda e: self._on_fs_change())
        self._param_widgets["fs"] = cb
        row += 1

        # 随机种子
        self._vars["seed"] = tk.IntVar(value=0)
        ttk.Label(gen, text=PARAM_SPECS["seed"]["label"]).grid(
            row=row, column=0, sticky="w", pady=2
        )
        seed_box = ttk.Frame(gen)
        seed_box.grid(row=row, column=1, sticky="ew", pady=2)
        seed_box.columnconfigure(0, weight=1)
        ttk.Spinbox(
            seed_box, from_=0, to=2**31 - 1, textvariable=self._vars["seed"], width=10
        ).grid(row=0, column=0, sticky="ew")
        ttk.Button(seed_box, text="随机", width=5, command=self._random_seed).grid(
            row=0, column=1, padx=(4, 0)
        )
        self._param_widgets["seed"] = seed_box
        row += 1

        # 归一化
        self._vars["normalize"] = tk.StringVar(value="peak")
        ttk.Label(gen, text="归一化").grid(row=row, column=0, sticky="w", pady=2)
        nb = ttk.Combobox(
            gen,
            textvariable=self._vars["normalize"],
            values=["peak", "rms", "none"],
            state="readonly",
            width=12,
        )
        nb.grid(row=row, column=1, sticky="ew", pady=2)
        self._param_widgets["normalize"] = nb
        nb.bind("<<ComboboxSelected>>", lambda e: self._sync_normalize_state())
        row += 1

        self._vars["target_dbfs"] = tk.DoubleVar(value=-3.0)
        ttk.Label(gen, text=PARAM_SPECS["target_dbfs"]["label"]).grid(
            row=row, column=0, sticky="w", pady=2
        )
        dbfs_widget = ttk.Spinbox(
            gen, from_=-60.0, to=0.0, increment=0.5,
            textvariable=self._vars["target_dbfs"], width=12,
        )
        dbfs_widget.grid(row=row, column=1, sticky="ew", pady=2)
        self._param_widgets["target_dbfs"] = dbfs_widget
        row += 1

        # ---- 类型专属参数 ----
        self.specific_frame = ttk.LabelFrame(parent, text="类型专属参数", padding=10)
        self.specific_frame.pack(fill=tk.X, padx=8, pady=4)
        self.specific_frame.columnconfigure(1, weight=1)

        self._spec_widgets: Dict[str, tk.Widget] = {}
        self._spec_labels: Dict[str, tk.Widget] = {}
        spec_row = 0
        for key in ("duration", "f_start", "f_stop", "beta", "n_tones", "n_steps",
                    "step_duration", "fade_ratio", "amplitude",
                    "crest_factor_db"):
            meta = PARAM_SPECS[key]
            self._vars[key] = tk.DoubleVar(value=float(meta["default"]))
            label = ttk.Label(self.specific_frame, text=meta["label"])
            label.grid(row=spec_row, column=0, sticky="w", pady=2)
            self._spec_labels[key] = label
            widget = ttk.Spinbox(
                self.specific_frame,
                from_=meta["min"],
                to=meta["max"],
                increment=meta.get("step", 1),
                textvariable=self._vars[key],
                width=12,
            )
            widget.grid(row=spec_row, column=1, sticky="ew", pady=2)
            widget.bind("<Return>", lambda e: self._regenerate())
            widget.bind("<FocusOut>", lambda e: self._regenerate())
            self._spec_widgets[key] = widget
            spec_row += 1

        # 扫频方式
        self._vars["sweep_method"] = tk.StringVar(value="log")
        ttk.Label(self.specific_frame, text="扫频方式").grid(
            row=spec_row, column=0, sticky="w", pady=2
        )
        cb = ttk.Combobox(
            self.specific_frame,
            textvariable=self._vars["sweep_method"],
            values=["log", "linear"],
            state="readonly",
            width=12,
        )
        cb.grid(row=spec_row, column=1, sticky="ew", pady=2)
        cb.bind("<<ComboboxSelected>>", lambda e: self._regenerate())
        self._spec_widgets["sweep_method"] = cb
        spec_row += 1

        # 谐波幅度（单频正弦专用）：一列文本，不是单个数值
        self._vars["harmonics"] = tk.StringVar(value="")
        ttk.Label(self.specific_frame, text=PARAM_SPECS["harmonics"]["label"]).grid(
            row=spec_row, column=0, sticky="w", pady=2
        )
        harmonics_entry = ttk.Entry(
            self.specific_frame, textvariable=self._vars["harmonics"], width=16
        )
        harmonics_entry.grid(row=spec_row, column=1, sticky="ew", pady=2)
        harmonics_entry.bind("<Return>", lambda e: self._regenerate())
        harmonics_entry.bind("<FocusOut>", lambda e: self._regenerate())
        self._spec_widgets["harmonics"] = harmonics_entry

    def _build_file_panel(self, parent: ttk.Frame) -> None:
        """文件信息：分析模式下显示当前打开的音频文件。"""
        box = ttk.LabelFrame(parent, text="文件信息", padding=10)
        box.pack(fill=tk.X, padx=8, pady=(8, 4))
        box.columnconfigure(0, weight=1)

        self._file_name_label = ttk.Label(box, text="（尚未打开文件）", wraplength=290)
        self._file_name_label.grid(row=0, column=0, sticky="w")

        self._file_path_label = ttk.Label(box, text="", style="Path.TLabel", wraplength=290)
        self._file_path_label.grid(row=1, column=0, sticky="w", pady=(2, 6))

        self._file_meta_label = ttk.Label(box, text="", style="Hint.TLabel", wraplength=290)
        self._file_meta_label.grid(row=2, column=0, sticky="w")

    def _build_analysis_options(self, parent: ttk.Frame) -> None:
        """分析选项：窗函数 / 倍频程阶数 / STFT / 导出设置。

        这一块在**两种工作模式下都显示** —— 分析外部文件用的就是同一套
        窗函数与同一条倍频程曲线，口径必须一致。
        """
        parent.columnconfigure(1, weight=1)

        self._vars["window"] = tk.StringVar(value=DEFAULT_WINDOW)
        ttk.Label(parent, text="窗函数").grid(row=0, column=0, sticky="w", pady=2)
        cb = ttk.Combobox(
            parent,
            textvariable=self._vars["window"],
            values=list(WINDOWS.keys()),
            state="readonly",
            width=16,
        )
        cb.grid(row=0, column=1, sticky="ew", pady=2)
        cb.bind("<<ComboboxSelected>>", lambda e: self._on_window_change())
        self._window_box = cb

        self._window_hint = ttk.Label(parent, text="", style="Hint.TLabel", wraplength=300)
        self._window_hint.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))

        # 倍频程阶数：与 CLI 的 --order 同义，默认 1/6
        self._vars["order"] = tk.StringVar(value=_ORDER_LABELS[6])
        ttk.Label(parent, text="倍频程阶数").grid(row=2, column=0, sticky="w", pady=(6, 2))
        cb = ttk.Combobox(
            parent,
            textvariable=self._vars["order"],
            values=[_ORDER_LABELS[o] for o in OCTAVE_ORDERS],
            state="readonly",
            width=8,
        )
        cb.grid(row=2, column=1, sticky="w", pady=(6, 2))
        cb.bind("<<ComboboxSelected>>", lambda e: self._regenerate())
        self._order_box = cb

        self._order_hint = ttk.Label(
            parent,
            text="1/6 倍频程为默认口径（IEC 61260 / ANSI S1.11）。",
            style="Hint.TLabel",
            wraplength=300,
        )
        self._order_hint.grid(row=3, column=0, columnspan=2, sticky="w", pady=(2, 0))

        self._vars["show_stft"] = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            parent, text="绘制 STFT 频谱图", variable=self._vars["show_stft"],
            command=self._regenerate,
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(6, 0))

        self._vars["dither"] = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            parent, text="导出时启用 TPDF 抖动", variable=self._vars["dither"]
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(4, 0))

        self._vars["bits"] = tk.IntVar(value=16)
        ttk.Label(parent, text="导出位深").grid(row=6, column=0, sticky="w", pady=(6, 2))
        cb = ttk.Combobox(
            parent,
            textvariable=self._vars["bits"],
            values=[str(b) for b in BITS_CHOICES],
            state="readonly",
            width=8,
        )
        cb.grid(row=6, column=1, sticky="w", pady=(6, 2))

    # ------------------------------------------------------------------
    # 规格回填
    # ------------------------------------------------------------------

    def _apply_spec_to_vars(self, spec: SignalSpec) -> None:
        """把 :class:`SignalSpec` 的值回填到界面控件上。

        回填的是**已解析完成**的规格（含类型默认值），所以每个字段都是
        确定的；谐波是列表，界面上以逗号分隔的文本呈现。
        """
        self._updating = True
        try:
            self.kind_var.set(KIND_LABELS[spec.kind])
            self._on_kind_change()

            for key, cast in (
                ("fs", int),
                ("duration", float),
                ("seed", int),
                ("f_start", float),
                ("f_stop", float),
                ("beta", float),
                ("crest_factor_db", float),
                ("n_tones", int),
                ("n_steps", int),
                ("step_duration", float),
                ("fade_ratio", float),
                ("amplitude", float),
                ("target_dbfs", float),
            ):
                value = getattr(spec, key, None)
                if value is None or key not in self._vars:
                    continue
                try:
                    self._vars[key].set(cast(value))
                except (tk.TclError, ValueError):
                    continue

            self._vars["normalize"].set(spec.normalize)
            self._vars["sweep_method"].set(spec.sweep_method)
            self._vars["harmonics"].set(
                ", ".join(f"{h:g}" for h in spec.harmonics)
            )
            self._sync_fs_limits()
            self._sync_normalize_state()
        finally:
            self._updating = False

    def _export_report(self) -> None:
        """导出**双栏分析报告图**（时域 + 频域含 1/6 倍频程）。

        与「保存频谱图 PNG」的区别：那个只存当前频谱页，这张存的是
        完整报告（上半时域、下半频域），可直接贴进测试报告。

        复用界面上已经算好的结果，**不重新分析** —— 所以对长文件也是瞬时的。
        """
        if self._result is None or self._signal is None:
            messagebox.showwarning("无法导出", "请先生成信号或打开一个音频文件")
            return

        title = str(self._result.get("title") or "signal")
        stem = Path(title).stem or "signal"
        path = filedialog.asksaveasfilename(
            title="导出分析图",
            initialfile=f"{stem}_analysis.png",
            defaultextension=".png",
            filetypes=[("PNG 图片", "*.png"), ("所有文件", "*.*")],
        )
        if not path:
            return

        try:
            import matplotlib.pyplot as plt

            payload = self._result
            file_info = payload.get("file_info")
            spec = payload.get("spec")
            if file_info:
                info = {
                    "sample_rate": file_info["sample_rate"],
                    "channels": file_info["channels"],
                    "bits": file_info["bits"],
                    "encoding": file_info["encoding"],
                }
            elif spec is not None:
                info = {
                    "sample_rate": spec.fs,
                    "channels": 1,
                    "bits": int(self._vars["bits"].get()),
                    "encoding": "PCM",
                }
            else:
                info = {
                    "sample_rate": payload["fs"],
                    "channels": 1,
                    "bits": 0,
                    "encoding": "dBFS",
                }

            result = {
                "name": title,
                "signal": payload["signal"],
                "fs": payload["fs"],
                "duration": payload["n_samples"] / float(payload["fs"]),
                "freqs": payload["freqs"],
                "db": payload["db"],
                "centers": payload["centers"],
                "bands": payload["bands"],
                "stats": payload["stats"],
                "order": int(payload.get("order", 6) or 6),
                "info": info,
            }
            fig = build_report_figure(result, title=title)
            try:
                fig.savefig(path, dpi=140)
            finally:
                plt.close(fig)
            messagebox.showinfo("导出完成", f"已保存到：\n{path}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("导出失败", str(exc))

    # ------------------------------------------------------------------
    # 参数联动
    # ------------------------------------------------------------------
    def _random_seed(self) -> None:
        self._vars["seed"].set(int(np.random.default_rng().integers(0, 2**31 - 1)))
        self._regenerate()

    def _sync_normalize_state(self) -> None:
        """目标电平只在 rms 模式下生效。"""
        mode = str(self._vars["normalize"].get())
        state = "normal" if mode == "rms" else "disabled"
        try:
            self._param_widgets["target_dbfs"].configure(state=state)
        except tk.TclError:
            pass
        self._regenerate()

    def _on_fs_change(self) -> None:
        """采样率变化时，终止频率上限要跟着奈奎斯特频率走。"""
        self._sync_fs_limits()
        self._regenerate()

    def _sync_fs_limits(self) -> None:
        """采样率变化时，起始 / 终止频率的上限都要跟着奈奎斯特频率走。"""
        try:
            fs = int(self._vars["fs"].get())
        except (tk.TclError, ValueError):
            return
        limit = 0.99 * fs / 2.0
        for key in ("f_start", "f_stop"):
            widget = self._spec_widgets.get(key)
            if widget is not None:
                try:
                    widget.configure(to=limit)
                except tk.TclError:
                    pass
        try:
            cur = float(self._vars["f_stop"].get())
        except (tk.TclError, ValueError):
            cur = None
        if cur is not None and cur > limit:
            self._vars["f_stop"].set(round(limit, 1))
        try:
            cur = float(self._vars["f_start"].get())
        except (tk.TclError, ValueError):
            cur = None
        if cur is not None and cur > limit:
            self._vars["f_start"].set(round(limit, 1))

    # ------------------------------------------------------------------
    # 状态同步
    # ------------------------------------------------------------------
    def _current_kind(self) -> str:
        """从下拉框的显示文本反查内部类型标识。"""
        text = self.kind_var.get()
        for kind, label in KIND_LABELS.items():
            if label == text:
                return kind
        # 兜底：按选中序号反查
        try:
            idx = self.kind_box.current()
            if 0 <= idx < len(ALL_KINDS):
                return ALL_KINDS[idx]
        except tk.TclError:
            pass
        return "white"

    def _on_kind_change(self) -> None:
        kind = self._current_kind()
        defaults = DEFAULT_PARAMS.get(kind, {})
        for key, value in defaults.items():
            if key in self._vars:
                self._vars[key].set(value)

        relevant = set(KIND_PARAMS.get(kind, []))
        for key, widget in self._spec_widgets.items():
            state = "normal" if key in relevant else "disabled"
            try:
                widget.configure(state=state)
            except tk.TclError:
                pass

        # 起始 / 终止频率的含义随类型而变，标签与说明要跟着换
        self._sync_freq_labels(kind)

        try:
            self.kind_hint.configure(text=KIND_DESCRIPTIONS.get(kind, ""))
        except tk.TclError:
            pass

        # 文件模式下生成参数不适用，切换类型不该把文件分析结果顶掉
        if self._mode == MODE_FILE:
            return

        self._regenerate()

    def _sync_freq_labels(self, kind: str) -> None:
        """按信号类型刷新 ``f_start`` / ``f_stop`` 的标签与说明。

        同一对参数在不同类型下含义完全不同（噪声=带限、多音=音调范围、
        扫频=起止频率、标准信号=频带边界），所以标签必须随类型变化，
        否则用户很容易把"带限下限"当成"扫频起点"来理解。
        """
        info = FREQ_PARAM_INFO.get(kind, {})
        for key in ("f_start", "f_stop"):
            label_text, _note = info.get(key, (PARAM_SPECS[key]["label"], ""))
            widget = self._spec_labels.get(key)
            if widget is not None:
                try:
                    widget.configure(text=label_text)
                except tk.TclError:
                    pass

    def _on_window_change(self) -> None:
        """切换窗函数：刷新指标提示并重算。"""
        self._update_window_hint()
        self._regenerate()

    def _update_window_hint(self) -> None:
        """显示所选窗函数的关键指标（ENBW / 栅栏损失）。"""
        try:
            window = str(self._vars["window"].get())
            fs = int(self._vars["fs"].get())
        except (tk.TclError, ValueError):
            return
        m = window_metrics(window, fs=fs, n=_WINDOW_METRIC_N)
        text = (
            f"ENBW {m['enbw_bins']:.2f} bin · 栅栏损失 {m['scalloping_db']:+.2f} dB\n"
            f"{WINDOW_LABELS.get(window, '')}"
        )
        try:
            self._window_hint.configure(text=text)
        except tk.TclError:
            pass

    # ------------------------------------------------------------------
    # 左侧面板滚轮
    # ------------------------------------------------------------------
    def _bind_panel_wheel(self, widget: tk.Misc) -> None:
        """递归给面板控件绑定滚轮。

        用 ``add="+"`` 保留控件自身的绑定；事件处理函数返回 ``"break"``，
        于是下拉框不会因为滚轮而误改选中项，数值框也不会误改数值 ——
        滚轮在左侧面板里统一用于滚动。
        """
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            try:
                widget.bind(seq, self._on_panel_wheel, add="+")
            except tk.TclError:
                pass
        for child in widget.winfo_children():
            self._bind_panel_wheel(child)

    def _on_panel_wheel(self, event: tk.Event) -> str:
        """滚轮滚动左侧参数面板。"""
        delta = 0
        num = getattr(event, "num", None)
        if num == 4:          # X11 / Linux 向上
            delta = -1
        elif num == 5:        # X11 / Linux 向下
            delta = 1
        else:
            step = int(getattr(event, "delta", 0) or 0)
            if step > 0:
                delta = -1
            elif step < 0:
                delta = 1
        if delta:
            try:
                self._scroll_canvas.yview_scroll(delta, "units")
            except tk.TclError:
                pass
        return "break"

    def _collect_spec(self) -> Tuple[SignalSpec, List[str]]:
        kind = self._current_kind()
        defaults = DEFAULT_PARAMS.get(kind, {})

        def get(key, cast=float):
            if key not in self._vars:
                return defaults.get(key)
            try:
                return cast(self._vars[key].get())
            except (tk.TclError, ValueError):
                return defaults.get(key)

        f_start = float(get("f_start"))
        f_stop = float(get("f_stop"))
        # 单频正弦只有一个频率：界面上用"起始频率"承载它，终止频率不适用
        f1 = f_start if kind == "sine" else 1000.0

        # 谐波：一列逗号分隔的相对幅度（"0.5,0.25"）
        harmonics: List[float] = []
        if kind == "sine":
            raw = str(self._vars["harmonics"].get()).strip()
            for piece in raw.replace("，", ",").split(","):
                piece = piece.strip()
                if not piece:
                    continue
                try:
                    harmonics.append(float(piece))
                except ValueError as exc:
                    raise ValueError(f"谐波幅度里有无法解析的数值: {piece!r}") from exc

        spec = SignalSpec(
            kind=kind,
            fs=int(get("fs", int)),
            duration=float(get("duration")),
            seed=int(get("seed", int)),
            f_start=f_start,
            f_stop=f_stop,
            beta=float(get("beta")),
            crest_factor_db=float(get("crest_factor_db")),
            n_tones=int(get("n_tones", int)),
            n_steps=int(get("n_steps", int)),
            step_duration=float(get("step_duration")),
            fade_ratio=float(get("fade_ratio")),
            f1=f1,
            amplitude=float(get("amplitude")),
            harmonics=harmonics,
            sweep_method=str(self._vars["sweep_method"].get()),
            normalize=str(self._vars["normalize"].get()),
            target_dbfs=float(get("target_dbfs")),
        )
        # 频率超出奈奎斯特会被混叠成别的频率，这里统一钳制
        return clamp_spec_to_nyquist(spec)

    # ------------------------------------------------------------------
    # 工作模式：生成信号 / 分析文件
    # ------------------------------------------------------------------
    def _current_order(self) -> int:
        """当前选择的倍频程阶数（默认 1/6）。"""
        try:
            label = str(self._vars["order"].get())
        except (tk.TclError, KeyError):
            return 6
        return _ORDER_BY_LABEL.get(label, 6)

    def _toggle_panel(self, panel: tk.Widget, visible: bool) -> None:
        """显示 / 隐藏左栏里的一个区块。

        用 ``pack(before=...)`` 而不是 ``pack()`` —— 否则重新显示的区块会被
        追加到**末尾**，跑到「分析选项」后面去，顺序就乱了。
        """
        if visible:
            panel.pack(fill=tk.X, before=self._analysis_box)
        else:
            panel.pack_forget()

    def _on_mode_change(self) -> None:
        """切换工作模式。

        * 切到「生成信号」→ 用当前生成参数重新生成并显示；
        * 切到「分析文件」→ 复用已打开文件的分析结果（不必重读）；
          还没打开过文件时提示用户去打开。
        """
        mode = str(self.mode_var.get())
        if mode not in (MODE_GENERATE, MODE_FILE):
            mode = MODE_GENERATE
        self._mode = mode

        is_file = mode == MODE_FILE
        self._toggle_panel(self._gen_body, not is_file)
        self._toggle_panel(self._file_body, is_file)

        # 「重新生成」只在生成模式下有意义（文件模式下改的是分析选项）
        try:
            self.btn_regen.configure(state="disabled" if is_file else "normal")
        except tk.TclError:
            pass

        try:
            self.source_label.configure(text="文件" if is_file else "生成")
        except tk.TclError:
            pass

        if is_file:
            if self._file_payload is None:
                try:
                    self.mode_hint.configure(
                        text="还没有打开文件 —— 点下方「打开音频文件…」选一个。"
                    )
                except tk.TclError:
                    pass
                self._status("等待打开文件")
                return
            try:
                self.mode_hint.configure(
                    text="分析已打开的文件：窗函数与倍频程阶数对它同样生效。"
                )
            except tk.TclError:
                pass
            # 复用已算好的结果，不重新读文件
            self._apply_payload(self._file_payload)
            return

        try:
            self.mode_hint.configure(
                text="按参数生成可复现信号（相同参数 + 相同种子 → 逐比特相同输出）。"
            )
        except tk.TclError:
            pass
        self._regenerate(immediate=True)

    def _update_file_panel(self) -> None:
        """刷新左栏「文件信息」区块的内容。"""
        info = self._file_info
        path = self._file_path
        if not path:
            try:
                self._file_name_label.configure(text="（尚未打开文件）")
                self._file_path_label.configure(text="")
                self._file_meta_label.configure(text="")
            except tk.TclError:
                pass
            return

        try:
            self._file_name_label.configure(text=Path(path).name)
            self._file_path_label.configure(text=str(path))
        except tk.TclError:
            pass

        if not info:
            try:
                self._file_meta_label.configure(text="（格式信息不可用）")
            except tk.TclError:
                pass
            return

        try:
            self._file_meta_label.configure(
                text=(
                    f"格式 {info.get('format', '?')} · {info.get('sample_rate', 0)} Hz · "
                    f"{info.get('channels', 0)} ch · {info.get('bits', 0)} bit "
                    f"{info.get('encoding', '')}\n"
                    f"时长 {info.get('duration', 0.0):.3f} s · "
                    f"{info.get('frames', 0)} 样本"
                )
            )
        except tk.TclError:
            pass

    # ------------------------------------------------------------------
    # 生成调度（去抖 + 后台线程）
    # ------------------------------------------------------------------
    def _regenerate(self, immediate: bool = False) -> None:
        """请求重算。带去抖，避免连续输入触发一串重复计算。

        生成模式下重算的是**新生成的信号**；文件模式下重算的是
        **已打开文件的分析结果**（改窗函数 / 倍频程阶数 / STFT 时）。
        """
        if self._updating:
            return
        if self._debounce_id is not None:
            try:
                self.after_cancel(self._debounce_id)
            except tk.TclError:
                pass
            self._debounce_id = None

        if immediate:
            self._start_job()
        else:
            self._debounce_id = self.after(self.DEBOUNCE_MS, self._on_debounce_fire)

    def _on_debounce_fire(self) -> None:
        self._debounce_id = None
        self._start_job()

    def _start_job(self) -> None:
        """在主线程收集参数，把计算丢给工作线程。"""
        window = str(self._vars["window"].get())
        show_stft = bool(self._vars["show_stft"].get())
        order = self._current_order()

        self._job_id += 1
        job_id = self._job_id
        self._job_running = True

        # 文件模式：重算的是同一个文件的分析结果
        if self._mode == MODE_FILE:
            if not self._file_path:
                # 没打开文件就不能凭空生成一个信号来顶替 —— 那会让用户
                # 以为自己在看文件的分析结果
                self._job_running = False
                self._status("等待打开文件")
                return
            self._notes = []
            self._status("重新分析中…")
            thread = threading.Thread(
                target=self._file_worker,
                args=(job_id, self._file_path, window, show_stft, order),
                daemon=True,
            )
            thread.start()
            self._ensure_polling()
            return

        # 生成模式：按参数生成新信号
        try:
            spec, notes = self._collect_spec()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("参数错误", str(exc))
            return

        self._notes = notes
        self._status("计算中…")

        thread = threading.Thread(
            target=self._worker,
            args=(job_id, spec, window, show_stft, order),
            daemon=True,
        )
        thread.start()
        self._ensure_polling()

    def _worker(
        self, job_id: int, spec: SignalSpec, window: str, show_stft: bool, order: int = 6
    ) -> None:
        try:
            payload = compute_payload(spec, window, show_stft, order=order)
        except Exception as exc:  # noqa: BLE001
            self._queue.put((job_id, None, (exc, traceback.format_exc())))
        else:
            self._queue.put((job_id, payload, None))

    # ------------------------------------------------------------------
    # 打开外部音频文件
    # ------------------------------------------------------------------

    def _open_audio(self) -> None:
        """打开一个已有的音频文件并分析它。

        分析走的是**和生成信号完全相同**的频域路径（同一套窗函数、
        同一条倍频程曲线、同一张 STFT 图），所以外部文件与内部
        生成的信号可以直接对比。

        WAV 走自研的零依赖 RIFF 解析器；其它格式（FLAC / OGG / AIFF …）
        需要环境里装了 ``soundfile``。
        """
        path = filedialog.askopenfilename(
            title="打开音频文件",
            filetypes=[
                ("WAV 音频", "*.wav *.wave"),
                ("所有音频文件", "*.wav *.wave *.flac *.ogg *.oga *.aiff *.aif "
                                 "*.au *.snd *.caf *.mp3 *.m4a"),
                ("所有文件", "*.*"),
            ],
        )
        if not path:
            return

        window = str(self._vars["window"].get())
        show_stft = bool(self._vars["show_stft"].get())
        order = self._current_order()

        self._file_path = path
        self._pending_open = True
        self._job_id += 1
        job_id = self._job_id
        self._job_running = True
        self._notes = []
        self._status("读取并分析中…")

        # 打开文件即切到「分析文件」模式（左栏换成文件信息）
        self._mode = MODE_FILE
        try:
            self.mode_var.set(MODE_FILE)
            self._toggle_panel(self._gen_body, False)
            self._toggle_panel(self._file_body, True)
            self.source_label.configure(text="文件")
            self.btn_regen.configure(state="disabled")
            self.mode_hint.configure(
                text="分析已打开的文件：窗函数与倍频程阶数对它同样生效。"
            )
        except tk.TclError:
            pass

        thread = threading.Thread(
            target=self._file_worker,
            args=(job_id, path, window, show_stft, order),
            daemon=True,
        )
        thread.start()
        self._ensure_polling()

    def _file_worker(
        self, job_id: int, path: str, window: str, show_stft: bool, order: int = 6
    ) -> None:
        """工作线程：读取音频文件并完成全部频域分析。"""
        try:
            signal, fs = load_audio(path)
            if signal.size == 0:
                raise ValueError("文件里没有采样数据")
            payload = analyze_payload(signal, fs, window, show_stft, order=order)
            # 外部文件没有"生成参数"，spec 留空
            payload["spec"] = None
            payload["title"] = Path(path).name
            # 文件格式信息（导出分析图时显示）
            try:
                payload["file_info"] = audio_info(path)
            except Exception:  # noqa: BLE001 - 取不到格式信息不该影响分析
                payload["file_info"] = None
        except Exception as exc:  # noqa: BLE001
            self._queue.put((job_id, None, (exc, traceback.format_exc())))
        else:
            self._queue.put((job_id, payload, None))

    def _ensure_polling(self) -> None:
        if self._poll_id is None:
            self._poll_id = self.after(self.POLL_MS, self._poll_results)

    def _poll_results(self) -> None:
        """主线程取回后台计算结果并刷新界面。"""
        self._poll_id = None

        payload = None
        error = None
        while True:
            try:
                job_id, item, err = self._queue.get_nowait()
            except queue.Empty:
                break
            if job_id != self._job_id:
                continue  # 过期结果，丢弃
            if err is not None:
                error = err
            else:
                payload = item

        if payload is not None:
            self._job_running = False
            self._pending_open = False
            self._apply_payload(payload)
        elif error is not None:
            self._job_running = False
            exc, tb = error
            self._status("计算失败")
            if self._pending_open:
                # 首次打开就失败了：退回生成模式。
                # 否则徽章写着"文件"、图表却还是旧的生成结果，两边对不上。
                self._pending_open = False
                self._mode = MODE_GENERATE
                try:
                    self.mode_var.set(MODE_GENERATE)
                    self._toggle_panel(self._gen_body, True)
                    self._toggle_panel(self._file_body, False)
                    self.source_label.configure(text="生成")
                    self.btn_regen.configure(state="normal")
                    self.mode_hint.configure(
                        text="按参数生成可复现信号（相同参数 + 相同种子 → 逐比特相同输出）。"
                    )
                except tk.TclError:
                    pass
            messagebox.showerror(
                "分析失败" if self._mode == MODE_FILE else "生成失败", f"{exc}\n\n{tb}"
            )
        elif self._job_running:
            pass

        if self._job_running:
            self._ensure_polling()

    def _apply_payload(self, payload: Dict[str, Any]) -> None:
        self._result = payload
        self._signal = payload["signal"]
        # 外部文件分析时没有"生成参数"，spec 为 None
        self._spec = payload.get("spec")
        self._result_seq += 1
        self._dirty = set(TAB_KEYS)
        self._redraw()
        self._update_stats(payload)

        # 文件分析的结果单独缓存一份：切回「分析文件」模式时直接复用，
        # 不必重新读盘（长文件尤其明显）。
        if self._mode == MODE_FILE:
            self._file_payload = payload
            self._file_info = payload.get("file_info")
            self._update_file_panel()

        self._status("")

    # ------------------------------------------------------------------
    # 绘图
    # ------------------------------------------------------------------
    def _active_tab_key(self) -> str:
        try:
            idx = self.notebook.index(self.notebook.select())
        except tk.TclError:
            idx = 0
        if 0 <= idx < len(TAB_KEYS):
            return TAB_KEYS[idx]
        return TAB_KEYS[0]

    def _on_tab_changed(self, _event=None) -> None:
        """切到某一页时才补画那一页（旧版每次重绘全部 4 页）。"""
        key = self._active_tab_key()
        if key in self._dirty:
            self._redraw(key)

    def _redraw(self, key: Optional[str] = None) -> None:
        if self._result is None:
            return
        key = key or self._active_tab_key()
        if key == "time":
            self._draw_time(self._result)
        elif key == "spec":
            self._draw_spec(self._result)
        elif key == "stft":
            self._draw_stft_tab(self._result)
        self._dirty.discard(key)

    def _view_range(self, fs: int) -> Tuple[float, float]:
        """频率轴显示范围：默认 20 Hz – 20 kHz，且不超过奈奎斯特频率。"""
        nyq = fs / 2.0
        lo = min(VIEW_F_LO, nyq * 0.99)
        hi = min(VIEW_F_HI, nyq * 0.99)
        if hi <= lo:
            hi = lo * 1.0001
        return lo, hi

    def _draw_time(self, payload: Dict[str, Any]) -> None:
        signal = payload["signal"]
        fs = float(payload["fs"])
        ax = self.ax_time
        ax.clear()
        n = signal.size

        # **逐列画 min/max 包络**，不能简单抽稀（详见 minmax_envelope 的说明）
        t, lo, hi = minmax_envelope(signal, fs, self.TIME_COLUMNS)
        if t.size < n:
            ax.fill_between(t, lo, hi, color="#e07b39", lw=0.0, alpha=0.9)
        else:
            ax.plot(t, lo, lw=0.7, color="#e07b39")

        ax.set_title(f"{payload.get('title', '信号')} — 时域波形", fontsize=10)
        ax.set_xlabel("时间 (s)")
        ax.set_ylabel("幅度")
        ax.set_xlim(0, n / fs)
        ax.grid(True, which="both", ls="--", lw=0.4, alpha=0.6)

        self.canvas_time.draw_idle()

    def _draw_spec(self, payload: Dict[str, Any]) -> None:
        freqs = payload["freqs"]
        db = payload["db"]
        centers = payload["centers"]
        bands = payload["bands"]
        window = str(self._vars["window"].get())
        order = int(payload.get("order", 6) or 6)

        ax = self.ax_spec
        ax.clear()
        ax.plot(freqs, db, lw=0.5, alpha=0.55, color="#4a7ec7", label="原始频谱")
        ax.plot(centers, bands, lw=1.6, color="#d62728",
                marker="o", ms=2.5, label=f"{_ORDER_LABELS.get(order, '1/6')} 倍频程（频带功率）")
        ax.set_xscale("log")
        ax.set_ylim(-140, 5)
        lo, hi = self._view_range(payload["fs"])
        ax.set_xlim(lo, hi)
        ax.set_xlabel("频率 (Hz)")
        ax.set_ylabel("电平 (dBFS)")
        ax.set_title(
            f"{payload.get('title', '信号')} — 频谱（窗函数 {window}）", fontsize=10
        )
        ax.grid(True, which="both", ls="--", lw=0.4, alpha=0.6)
        ax.legend(loc="upper right", fontsize=8)
        self.canvas_spec.draw_idle()

    def _draw_stft_tab(self, payload: Dict[str, Any]) -> None:
        ax = self.ax_stft
        ax.clear()
        stft = payload.get("stft")
        if stft is None:
            # 关闭 STFT 时隐藏色条，避免残留
            if self._stft_cbar is not None:
                self._stft_cbar.ax.set_visible(False)
            ax.text(0.5, 0.5, "（已关闭 STFT 绘制）", ha="center", va="center",
                    transform=ax.transAxes, color="#888888")
        else:
            f_spec, t_spec, db = stft
            mesh = ax.pcolormesh(t_spec, f_spec, db, shading="auto", cmap="viridis")
            if self._stft_cbar is None:
                # 色条只创建一次并常驻：每次 fig.colorbar() 都会往 figure 里
                # 再塞一个坐标轴，ax.clear() 并不会把它删掉，反复重绘会让
                # 坐标轴数量线性增长、界面越用越卡。
                self._stft_cbar = self.fig_stft.colorbar(mesh, ax=ax, label="dBFS")
            else:
                self._stft_cbar.update_normal(mesh)
                self._stft_cbar.ax.set_visible(True)
            ax.set_yscale("log")
            lo, hi = self._view_range(payload["fs"])
            ax.set_ylim(lo, hi)
            ax.set_xlabel("时间 (s)")
            ax.set_ylabel("频率 (Hz)")
            ax.set_title(
                f"{payload.get('title', '信号')} — 短时傅里叶变换 (STFT)", fontsize=10
            )
            ax.grid(True, which="both", ls="--", lw=0.3, alpha=0.4)
        self.canvas_stft.draw_idle()

    # ------------------------------------------------------------------
    # 状态栏
    # ------------------------------------------------------------------
    def _status(self, text: str) -> None:
        """显示临时状态（计算中… / 等待打开文件）。

        单独占一个标签，**不覆盖读数** —— 读数是"结果"，状态是"过程"，
        两者混在一行会让用户以为读数被冲掉了。
        """
        try:
            self.state_label.configure(text=text)
        except tk.TclError:
            pass

    def _update_stats(self, payload: Dict[str, Any]) -> None:
        stats = payload["stats"]
        fs = int(payload["fs"])
        n = int(payload["n_samples"])
        warn = "  ⚠ 削波!" if stats["clipped_samples"] else ""
        source = payload.get("title", "")
        prefix = f"{source}  |  " if source else ""

        # 文件模式下额外给出格式信息，让"这组读数属于哪个文件"一目了然
        info = payload.get("file_info")
        meta = ""
        if info:
            meta = (
                f"{info.get('format', '?')} · {info.get('sample_rate', 0)} Hz · "
                f"{info.get('channels', 0)} ch · {info.get('bits', 0)} bit "
                f"{info.get('encoding', '')}  |  "
            )

        try:
            self.source_label.configure(
                text="文件" if self._mode == MODE_FILE else "生成"
            )
        except tk.TclError:
            pass

        self.stats_label.configure(
            text=(
                f"{prefix}{meta}时长 {n / fs:.3f} s · {n} 样本  |  "
                f"峰值 {stats['peak_dbfs']:.2f} dBFS  |  RMS {stats['rms_dbfs']:.2f} dBFS  |  "
                f"峰值因子 {stats['crest_db']:.2f} dB  |  直流 {stats['dc']:+.2e}{warn}"
            )
        )
        if self._notes:
            self.warn_label.configure(text="⚠ " + "；".join(self._notes))
        elif self.player.supported:
            self.warn_label.configure(text="")

    # ------------------------------------------------------------------
    # 播放
    # ------------------------------------------------------------------
    def _play(self) -> None:
        if self._signal is None or self._result is None:
            messagebox.showwarning("无法播放", "请先生成信号或打开一个音频文件")
            return
        if not self.player.supported:
            messagebox.showwarning("无法播放", "当前系统没有内置音频播放后端")
            return
        try:
            gain = max(0.0, min(100.0, int(self._vars["play_gain"].get()))) / 100.0
        except (tk.TclError, ValueError):
            gain = 0.5
        try:
            self.player.play(self._signal, int(self._result["fs"]), gain=gain)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("播放失败", str(exc))
            return
        self.play_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self._ensure_play_polling()

    def _ensure_play_polling(self) -> None:
        """开始轮询播放状态，播完自动恢复按钮。"""
        if self._play_poll_id is None:
            self._play_poll_id = self.after(self.PLAY_POLL_MS, self._poll_play_state)

    def _poll_play_state(self) -> None:
        """播放结束（或被停止）时把界面切回空闲状态。"""
        self._play_poll_id = None
        if self.player.playing:
            self._ensure_play_polling()
            return
        self._finish_play()

    def _finish_play(self) -> None:
        """复位播放控件并清理临时文件。"""
        try:
            self.player.stop()
        except Exception:  # noqa: BLE001
            pass
        self.play_button.configure(state="normal")
        self.stop_button.configure(state="disabled")

    def _stop_play(self) -> None:
        self._cancel_play_polling()
        self._finish_play()

    def _cancel_play_polling(self) -> None:
        if self._play_poll_id is not None:
            try:
                self.after_cancel(self._play_poll_id)
            except tk.TclError:
                pass
            self._play_poll_id = None

    # ------------------------------------------------------------------
    # 导出
    # ------------------------------------------------------------------
    def _export_wav(self) -> None:
        if self._signal is None or self._result is None:
            messagebox.showwarning("无法导出", "请先生成信号或打开一个音频文件")
            return
        spec = self._spec
        kind = spec.kind if spec is not None else "signal"
        initial = f"{kind}.wav"
        path = filedialog.asksaveasfilename(
            title="导出 WAV",
            initialfile=initial,
            defaultextension=".wav",
            filetypes=[("WAV 音频", "*.wav"), ("所有文件", "*.*")],
        )
        if not path:
            return
        try:
            bits = int(self._vars["bits"].get())
            saved = save_wav(
                self._signal,
                path,
                fs=int(self._result["fs"]),
                bits=bits,
                seed=(spec.seed if spec is not None else 0),
                dither=bool(self._vars["dither"].get()),
            )
            messagebox.showinfo("导出完成", f"已保存到：\n{saved}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("导出失败", str(exc))

    def _save_figure(self) -> None:
        """保存**当前标签页**的图为 PNG。

        存"当前视图"而不是固定存频谱页 —— 用户停在 STFT 页时点保存，
        拿到的就该是 STFT 那张图。
        """
        if self._result is None:
            messagebox.showwarning("无法保存", "请先生成信号或打开一个音频文件")
            return
        key = self._active_tab_key()
        fig = self._tab_figures.get(key, self.fig_spec)
        spec = self._spec
        stem = Path(str(self._result.get("title") or (spec.kind if spec else "signal"))).stem
        path = filedialog.asksaveasfilename(
            title="保存当前视图",
            initialfile=f"{stem or 'signal'}_{key}.png",
            defaultextension=".png",
            filetypes=[("PNG 图片", "*.png"), ("所有文件", "*.*")],
        )
        if not path:
            return
        try:
            fig.savefig(path, dpi=140)
            messagebox.showinfo("保存完成", f"已保存到：\n{path}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("保存失败", str(exc))

    # ------------------------------------------------------------------
    def _on_close(self) -> None:
        """关闭窗口：停掉播放、清掉定时器与临时文件。"""
        try:
            self.player.stop()
        except Exception:  # noqa: BLE001
            pass
        self._cancel_play_polling()
        for attr in ("_poll_id", "_debounce_id"):
            ident = getattr(self, attr, None)
            if ident is not None:
                try:
                    self.after_cancel(ident)
                except tk.TclError:
                    pass
                setattr(self, attr, None)
        self.destroy()


def main() -> int:
    app = SignalGeneratorApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
