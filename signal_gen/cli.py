"""
命令行入口。

用法示例::

    python -m signal_gen list
    python -m signal_gen generate pink --duration 5 --seed 42 --out pink.wav
    python -m signal_gen generate multitone --n-tones 200 --normalize rms --target-dbfs -18
    python -m signal_gen generate stepped --n-steps 50 --step-duration 0.5
    python -m signal_gen plot sweep --sweep-method linear --out sweep.png
    python -m signal_gen analyze out/pink.wav
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, List, Optional, Sequence

from . import __version__
from .audio_io import BITS_CHOICES, output_dir, save_wav, signal_stats
from .core import (
    ALL_KINDS,
    KIND_DESCRIPTIONS,
    KIND_LABELS,
    SignalSpec,
    clamp_spec_to_nyquist,
    default_params_for,
    generate_signal,
)
from .report import (
    analyze_file,
    build_report_figure,
    format_band_table,
    minmax_envelope,
)
from .spectrum import (
    DEFAULT_WINDOW,
    WINDOWS,
    band_levels_dbfs,
    one_sided_spectrum_dbfs_power,
)

#: :class:`SignalSpec` 的 dataclass 默认值 —— 命令行参数没给时的最后一级兜底。
_DATACLASS_DEFAULTS = {f.name: f.default for f in fields(SignalSpec)}


def _configure_console() -> None:
    """让控制台能输出中文与特殊符号。

    Windows 中文环境的控制台代码页默认是 **GBK (cp936)**，而项目里的
    ``1/f²``（U+00B2）和 ``⚠``（U+26A0）在 GBK 下**无法编码** ——
    直接 ``print`` 会抛 :class:`UnicodeEncodeError` 让整个命令崩掉
    （``python -m signal_gen list`` 在默认中文控制台上就会挂）。

    这里做两件事：

    1. Windows 10 1803 之后的 conhost 与 Windows Terminal 都支持 UTF-8，
       先把控制台代码页切到 65001（UTF-8），再把 Python 的输出流改成
       UTF-8 —— 这样终端里显示正常，**重定向到文件**时也是 UTF-8 字节，
       与控制台语言无关；
    2. 万一切换失败（比如 conhost 太老），至少给输出流加上
       ``errors="replace"``，最差情况是字符被替换成 ``?``，而不是崩掉。
    """
    switched = False
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            ctypes.windll.kernel32.SetConsoleCP(65001)
            switched = True
        except Exception:  # noqa: BLE001 - 切换失败就退回兜底方案
            switched = False

    for stream in (sys.stdout, sys.stderr):
        try:
            if switched or not stream.isatty():
                # 输出被重定向到文件/管道：统一用 UTF-8 字节
                stream.reconfigure(encoding="utf-8", errors="replace")
            else:
                # 控制台切不动：保留原编码，只是不再因不可编码字符崩溃
                stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


def _parse_harmonics(text: str) -> List[float]:
    """解析 ``--harmonics`` 的逗号分隔列表。

    ``--harmonics 0.5,0.25`` 表示 2 次谐波相对基频的幅度 0.5、
    3 次谐波 0.25（对应 :attr:`SignalSpec.harmonics`）。
    """
    values: List[float] = []
    for piece in str(text).split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            values.append(float(piece))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"谐波列表里有无法解析的数值: {piece!r}"
            ) from exc
    if not values:
        raise argparse.ArgumentTypeError("谐波列表不能为空")
    return values


def _add_common(parser: argparse.ArgumentParser) -> None:
    """添加生成信号时共用的参数。

    **所有选项的默认值都是 ``None``** —— 这样才能区分"用户没写"和"用户写了
    跟默认值一样的值"，从而在装配规格时正确地只覆盖显式给出的参数。
    """
    parser.add_argument("--fs", type=int, default=None,
                        help="采样率 (Hz)，默认 48000")
    parser.add_argument("--duration", type=float, default=None,
                        help="信号时长 (s)，默认 5")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子，默认 0")
    parser.add_argument(
        "--normalize", choices=["peak", "rms", "none"], default=None,
        help="归一化方式：peak 峰值 / rms 均方根 / none 不处理，默认 peak",
    )
    parser.add_argument(
        "--target-dbfs", type=float, default=None,
        help="rms 模式下的目标电平 (dBFS)，默认 -3",
    )
    parser.add_argument("--f-start", type=float, default=None,
                        help="起始频率 (Hz)，含义随类型而变：噪声=带限下限、"
                             "多音=最低音调、扫频=起点、单频=信号频率、"
                             "标准信号=频带下限（默认随类型而定）")
    parser.add_argument("--f-stop", type=float, default=None,
                        help="终止频率 (Hz)，含义随类型而变：噪声=带限上限、"
                             "多音=最高音调、扫频=终点、标准信号=频带上限"
                             "（默认随类型而定；单频正弦不适用）")
    parser.add_argument("--beta", type=float, default=None,
                        help="噪声谱斜率 β（S∝f^-β）：0 白 / 1 粉 / 2 红 / -1 蓝"
                             "（默认随类型而定）")
    parser.add_argument("--crest-factor-db", type=float, default=None,
                        help="峰值因子 (dB)，0 表示不限幅；默认随类型而定"
                             "（EIA RS-426-B / IEC 60268-5 带限粉噪为 6 dB；"
                             "IEC 60268-1 Clause 7 节目信号为 0 —— 标准要求"
                             "“不做幅度限制”）")
    parser.add_argument("--n-tones", type=int, default=None,
                        help="多音数量，默认 200")
    parser.add_argument("--n-steps", type=int, default=None,
                        help="步进扫频台阶数，默认 50")
    parser.add_argument("--step-duration", type=float, default=None,
                        help="步进扫频每步时长 (s)，默认 0.5")
    parser.add_argument("--fade-ratio", type=float, default=None,
                        help="渐变比例（0~1），默认随类型而定"
                             "（连续扫频 0.05，步进扫频 0.25）")
    parser.add_argument("--f1", type=float, default=None,
                        help="单频正弦频率 (Hz)，等价于该类型的 --f-start，默认 1000")
    parser.add_argument("--amplitude", type=float, default=None,
                        help="单频正弦幅度，默认 1.0")
    parser.add_argument("--harmonics", type=_parse_harmonics, default=None,
                        help="谐波相对幅度列表（逗号分隔），如 --harmonics 0.5,0.25 "
                             "表示 2 次谐波 0.5、3 次谐波 0.25（仅单频正弦有效）")
    parser.add_argument("--sweep-method", choices=["log", "linear"], default=None,
                        help="扫频方式：log 等比 / linear 线性，默认 log")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="signal_gen",
        description="信号生成与分析工具（测试信号标定）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="可用类型: " + ", ".join(ALL_KINDS),
    )
    parser.add_argument("--version", action="version", version=f"signal_gen {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="列出所有信号类型")
    p_list.set_defaults(func=_cmd_list)

    p_gen = sub.add_parser("generate", help="生成信号并写入 WAV")
    p_gen.add_argument("kind", choices=ALL_KINDS, help="信号类型")
    p_gen.add_argument("--out", default=None, help="输出文件路径（默认写到 output/ 目录）")
    p_gen.add_argument("--bits", type=int, default=16, choices=BITS_CHOICES,
                       help="量化位深，默认 16")
    p_gen.add_argument("--no-dither", action="store_true", help="关闭 TPDF 抖动")
    _add_common(p_gen)
    p_gen.set_defaults(func=_cmd_generate)

    p_plot = sub.add_parser("plot", help="生成信号并绘制频谱图")
    p_plot.add_argument("kind", choices=ALL_KINDS, help="信号类型")
    p_plot.add_argument("--out", default=None, help="输出 PNG 路径")
    p_plot.add_argument("--window", default=DEFAULT_WINDOW, choices=list(WINDOWS.keys()),
                        help=f"窗函数，默认 {DEFAULT_WINDOW}")
    p_plot.add_argument("--show", action="store_true", help="同时弹出窗口显示")
    _add_common(p_plot)
    p_plot.set_defaults(func=_cmd_plot)

    # ---- analyze：分析任意音频文件 ----
    p_analyze = sub.add_parser("analyze", help="分析一个音频文件（读数 + 倍频程表 + 报告图）")
    p_analyze.add_argument("path", help="要分析的音频文件路径（WAV / FLAC / OGG …）")
    p_analyze.add_argument("--window", default=DEFAULT_WINDOW, choices=list(WINDOWS.keys()),
                           help=f"分析窗函数，默认 {DEFAULT_WINDOW}")
    p_analyze.add_argument("--order", type=int, default=6,
                           help="倍频程阶数：1=整倍频程，3=1/3，6=1/6（默认 6）")
    p_analyze.add_argument("--f-min", type=float, default=20.0,
                           help="分析下限 (Hz)，默认 20")
    p_analyze.add_argument("--f-max", type=float, default=24000.0,
                           help="分析上限 (Hz)，默认 24000")
    p_analyze.add_argument("--plot", default=None, metavar="PNG",
                           help="同时导出双栏分析报告图（时域 + 频域）到指定路径")
    p_analyze.set_defaults(func=_cmd_analyze)

    return parser


def _spec_from_args(args: argparse.Namespace) -> SignalSpec:
    """把命令行参数装配成 :class:`SignalSpec`。

    取值优先级（逐项独立判断）：

    1. **命令行显式写出的值**；
    2. 该信号类型的默认值（:data:`signal_gen.core.DEFAULT_PARAMS`）；
    3. :class:`SignalSpec` 的 dataclass 默认值。
    """
    kind = getattr(args, "kind", None)
    if kind is None:
        raise ValueError("必须指定信号类型")
    type_defaults = default_params_for(kind)

    def resolve(name: str) -> Any:
        """逐项取值：命令行 > 类型默认值 > dataclass 默认值。

        最后两级返回的可能是 ``None``（那些"可按类型解析"的字段），
        交给 :meth:`SignalSpec.__post_init__` 补齐即可。
        """
        cli = getattr(args, name, None)
        if cli is not None:
            return cli
        if name in type_defaults:
            return type_defaults[name]
        return _DATACLASS_DEFAULTS.get(name)

    def num(name: str, caster):
        value = resolve(name)
        return None if value is None else caster(value)

    # ---- 单频正弦：只有一个频率，--f1 与 --f-start 是同一个量的两种写法 ----
    #
    # 两个选项在这里被**合并**成一个频率，避免出现"改了 f1 却 f_start 还是旧值"
    # 的半吊子状态。
    f1_value: Optional[float]
    f_start_value: Optional[float]
    if kind == "sine":
        cli_f1 = getattr(args, "f1", None)
        cli_f_start = getattr(args, "f_start", None)
        if cli_f1 is not None:
            freq = float(cli_f1)
        elif cli_f_start is not None:
            freq = float(cli_f_start)
        else:
            freq = resolve("f1")
            if freq is None:
                freq = resolve("f_start")
        f1_value = float(freq)
        f_start_value = f1_value
    else:
        # f1 只对单频正弦有意义：非正弦类型不带 f1
        f1_value = num("f1", float) if getattr(args, "f1", None) is not None else None
        f_start_value = num("f_start", float)

    # 谐波（相对幅度列表）：只有命令行能设置
    if getattr(args, "harmonics", None) is not None:
        harmonics = [float(v) for v in args.harmonics]
    else:
        harmonics = []

    return SignalSpec(
        kind=kind,
        fs=num("fs", int),
        duration=num("duration", float),
        seed=num("seed", int),
        f_start=f_start_value,
        f_stop=num("f_stop", float),
        beta=num("beta", float),
        crest_factor_db=resolve("crest_factor_db"),
        n_tones=num("n_tones", int),
        n_steps=num("n_steps", int),
        step_duration=num("step_duration", float),
        fade_ratio=num("fade_ratio", float),
        f1=f1_value,
        amplitude=num("amplitude", float),
        harmonics=harmonics,
        sweep_method=resolve("sweep_method"),
        normalize=resolve("normalize"),
        target_dbfs=num("target_dbfs", float),
    )


# ---------------------------------------------------------------------------
# 子命令实现
# ---------------------------------------------------------------------------

def _cmd_list(args: argparse.Namespace) -> int:
    print(f"signal_gen {__version__} —— 可用信号类型：\n")
    for kind in ALL_KINDS:
        print(f"  {kind:12s} {KIND_LABELS[kind]}")
        print(f"  {'':12s}   {KIND_DESCRIPTIONS.get(kind, '')}")
    print(f"\n默认输出目录：{output_dir()}")
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    spec = _spec_from_args(args)
    spec, notes = clamp_spec_to_nyquist(spec)
    signal = generate_signal(spec)
    stats = signal_stats(signal)

    out = args.out
    if out is None:
        out = f"{spec.kind}.wav"
    path = save_wav(
        signal, out, fs=spec.fs, bits=args.bits,
        seed=spec.seed, dither=not args.no_dither,
    )

    print(f"已生成 {KIND_LABELS[spec.kind]}")
    print(f"  文件      : {path}")
    print(f"  采样率    : {spec.fs} Hz")
    print(f"  时长      : {signal.size / spec.fs:.3f} s ({signal.size} 样本)")
    print(f"  峰值      : {stats['peak_dbfs']:.2f} dBFS")
    print(f"  RMS       : {stats['rms_dbfs']:.2f} dBFS")
    print(f"  峰值因子  : {stats['crest_db']:.2f} dB")
    if stats["clipped_samples"]:
        print(f"  ⚠ 削波样点: {stats['clipped_samples']}")
    for note in notes:
        print(f"  ⚠ {note}")
    return 0


def _cmd_plot(args: argparse.Namespace) -> int:
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .fonts import configure_chinese_font
    configure_chinese_font()

    spec = _spec_from_args(args)
    signal = generate_signal(spec)

    # 一次 FFT 同时拿到幅度谱与功率谱
    freqs, db, power = one_sided_spectrum_dbfs_power(
        signal, spec.fs, window=args.window
    )
    f_max = min(spec.fs / 2.0, 24000.0)
    centers, bands = band_levels_dbfs(freqs, power, f_min=20.0, f_max=f_max)

    fig, axes = plt.subplots(3, 1, figsize=(11, 10))

    # 时域用 min/max 包络：直接抽稀在扫频高频段必然混叠（见 report.minmax_envelope）
    t, lo, hi = minmax_envelope(signal, spec.fs, 1600)
    if t.size < signal.size:
        axes[0].fill_between(t, lo, hi, color="orange", lw=0.0, alpha=0.9)
    else:
        axes[0].plot(t, lo, lw=0.7, color="orange")
    axes[0].set_title(f"{KIND_LABELS[spec.kind]} — 时域")
    axes[0].set_xlabel("时间 (s)")
    axes[0].set_ylabel("幅度")
    axes[0].grid(True, which="both", ls="--", lw=0.5)

    axes[1].plot(freqs, db, lw=0.5, alpha=0.6, label=f"原始频谱 ({args.window})")
    axes[1].plot(centers, bands, color="red", lw=1.6, marker="o", ms=2.5,
                 label="1/6 倍频程（频带功率）")
    axes[1].set_xscale("log")
    axes[1].set_xlim(20.0, min(20000.0, spec.fs / 2))
    axes[1].set_ylim(-140, 5)
    axes[1].set_title("频域 (dBFS)")
    axes[1].set_xlabel("频率 (Hz)")
    axes[1].set_ylabel("电平 (dBFS)")
    axes[1].grid(True, which="both", ls="--", lw=0.5)
    axes[1].legend(loc="upper right")

    axes[2].plot(centers, bands, "o-", color="red", lw=1.6, ms=3)
    axes[2].set_xscale("log")
    axes[2].set_xlim(20.0, min(20000.0, spec.fs / 2))
    axes[2].set_title("1/6 倍频程频带功率 (dBFS)")
    axes[2].set_xlabel("中心频率 (Hz)")
    axes[2].set_ylabel("电平 (dBFS)")
    axes[2].grid(True, which="both", ls="--", lw=0.5)

    fig.tight_layout()

    out = args.out
    if out is None:
        out = f"{spec.kind}_spectrum.png"
    path = Path(out)
    if not path.is_absolute():
        path = output_dir() / path
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    print(f"频谱图已保存: {path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    """分析一个音频文件：基本读数 + 倍频程频带功率表（+ 可选报告图）。"""
    path = Path(args.path)
    result = analyze_file(
        path,
        window=args.window,
        order=args.order,
        f_min=args.f_min,
        f_max=args.f_max,
    )

    info = result["info"]
    stats = result["stats"]
    centers = result["centers"]
    bands = result["bands"]

    print(f"分析 {path}")
    print(f"  采样率    : {info['sample_rate']} Hz")
    print(f"  声道      : {info['channels']}"
          + ("（已下混成单声道）" if info["channels"] > 1 else ""))
    print(f"  位深/编码 : {info['bits']} bit {info['encoding']}")
    print(f"  时长      : {info['duration']:.3f} s ({result['n_samples']} 样本)")
    print()
    print("  峰值      : {:+.2f} dBFS".format(stats["peak_dbfs"]))
    print("  RMS       : {:+.2f} dBFS".format(stats["rms_dbfs"]))
    print("  峰值因子  : {:.2f} dB".format(stats["crest_db"]))
    print("  直流偏移  : {:+.3e}".format(stats["dc"]))
    if stats["clipped_samples"]:
        print(f"  ⚠ 超满幅样点: {stats['clipped_samples']}")

    order_name = {1: "1", 2: "1/2", 3: "1/3", 6: "1/6", 12: "1/12",
                  24: "1/24"}.get(args.order, f"1/{args.order}")
    print()
    print(f"{order_name} 倍频程频带功率 (dBFS，窗函数 {args.window})")
    print(format_band_table(centers, bands))

    if args.plot:
        out = Path(args.plot)
        if not out.is_absolute():
            out = output_dir() / out
        out.parent.mkdir(parents=True, exist_ok=True)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = build_report_figure(result)
        try:
            fig.savefig(out, dpi=140)
        finally:
            plt.close(fig)
        print()
        print(f"分析报告图已保存: {out}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    _configure_console()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ValueError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
