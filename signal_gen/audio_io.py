"""
量化与 WAV 读写。

写入要点
--------

1. **抖动先加、clip 后做**。NumPy 2.x 下 ``astype(np.int16)`` 对超量程值会
   *静默回绕*：``1.5 * 32767 -> -16386``，产生反向爆音。旧实现把 clip 放在抖动之前，
   抖动会把满幅样点推出量程，``round(x * 32767)`` 得到 32768 后回绕成 -32768 ——
   满幅正峰直接变成 -1.0。现在顺序改为「加抖动 → clip → 缩放取整」。
2. **TPDF 抖动**。直接截断的量化误差与信号相关（实测 RMS 0.579 LSB），
   加入幅值为 1 LSB 的三角分布抖动后误差与信号解相关，噪声更自然。
3. **输出路径基于 ``__file__``**，不再受当前工作目录影响。
4. 支持 16 bit 与 24 bit PCM，24 bit 为真正的 24 位打包格式。

读取要点
--------

:func:`load_wav` 自己解析 RIFF 容器（而不是用标准库 ``wave``），因为
``wave`` 不认 IEEE 浮点 WAV（format 3）—— 而浮点 WAV 在音频工具里很常见。
支持 PCM 8/16/24/32 bit 与 IEEE 浮点 32/64 bit，多声道会下混成单声道。

:func:`load_audio` 在此之上再放宽一层：**任何**能被 libsndfile 读的格式
（FLAC / OGG / AIFF / MP3 …）都能读进来 —— 前提是环境里装了 ``soundfile``。
没有装也不影响 WAV：WAV 走自研解析器，不依赖任何第三方库。
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

__all__ = [
    "BITS_CHOICES",
    "NON_WAV_SUFFIXES",
    "quantize_int16",
    "quantize_int24",
    "save_wav",
    "load_wav",
    "load_audio",
    "wav_info",
    "audio_info",
    "signal_stats",
    "output_dir",
]

#: 可选位深
BITS_CHOICES = [16, 24]

#: WAV 的 fmt 块里的音频格式标记
_WAVE_FORMAT_PCM = 0x0001
_WAVE_FORMAT_IEEE_FLOAT = 0x0003
_WAVE_FORMAT_EXTENSIBLE = 0xFFFE


def output_dir() -> Path:
    """默认输出目录：包所在的 ``output`` 子目录（不依赖当前工作目录）。"""
    return Path(__file__).resolve().parent.parent / "output"


def _dither(signal: np.ndarray, rng: np.random.Generator, levels: int) -> np.ndarray:
    """加入 TPDF 抖动（峰值为 1 LSB 的三角分布）。

    三角分布抖动会让量化误差与输入信号完全解相关，代价是本底噪声
    理论上升约 3 dB —— 对于测试信号这是可接受的代价。
    """
    tpdf = rng.random(signal.size) - rng.random(signal.size)
    return signal + tpdf / levels


def quantize_int16(
    signal: np.ndarray,
    seed: int = 0,
    dither: bool = True,
) -> np.ndarray:
    """量化为 16 bit PCM 整数。

    **抖动必须先加、clip 必须后做**。旧实现顺序反了：先 ``clip(-1, 1)`` 再加抖动，
    抖动幅度（1 LSB）会把满幅样点推出量程，``round(x * 32767)`` 得到 **32768**，
    ``astype(np.int16)`` 静默回绕成 **-32768** —— 满幅正峰变成 -1.0 的反向爆音。
    实测：20000 样点的满幅正弦里有 2 ~ 4 个正峰被翻转；只含一个 +1.0 的信号
    会出现 113 个 -1.0。

    修正后的顺序是「加抖动 → clip → 缩放取整 → 整数钳制」，
    满幅正峰稳定映射到 +32767，负峰映射到 -32767。
    """
    x = np.asarray(signal, dtype=np.float64)
    if dither:
        x = _dither(x, np.random.default_rng(seed), 32768.0)
    x = np.clip(x, -1.0, 1.0)
    # 乘 32767 而非 32768，保证 +1.0 映射到 32767 而不是溢出
    q = np.round(x * 32767.0)
    return np.clip(q, -32768.0, 32767.0).astype(np.int16)


def quantize_int24(
    signal: np.ndarray,
    seed: int = 0,
    dither: bool = True,
) -> np.ndarray:
    """量化为 24 bit PCM 整数（存放在 int32 容器中）。

    与 :func:`quantize_int16` 同样的顺序问题：抖动在 clip 之后加会把样点推出量程，
    24 bit 的 +8388608 打包进 3 字节后会变成 0（+1.0 变成 +0.0）。
    """
    x = np.asarray(signal, dtype=np.float64)
    if dither:
        x = _dither(x, np.random.default_rng(seed), 8388608.0)
    x = np.clip(x, -1.0, 1.0)
    q = np.round(x * 8388607.0)
    return np.clip(q, -8388608.0, 8388607.0).astype(np.int32)


def _write_wav_pcm(
    path: Path,
    samples: np.ndarray,
    fs: int,
    bits: int,
) -> None:
    """写入 PCM WAV。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(bits // 8)
        wf.setframerate(fs)
        if bits == 16:
            wf.writeframes(samples.astype("<i2").tobytes())
        elif bits == 24:
            # int32 -> 3 字节小端打包
            packed = np.empty(samples.size * 3, dtype=np.uint8)
            view = samples.astype("<i4").view(np.uint8).reshape(-1, 4)
            packed[0::3] = view[:, 0]
            packed[1::3] = view[:, 1]
            packed[2::3] = view[:, 2]
            wf.writeframes(packed.tobytes())
        else:
            raise ValueError(f"不支持的位深: {bits}")


def save_wav(
    signal: np.ndarray,
    path,
    fs: int = 48000,
    bits: int = 16,
    seed: int = 0,
    dither: bool = True,
) -> Path:
    """把浮点信号写入 WAV 文件。

    参数
    ----
    signal
        浮点样本，范围应在 [-1, 1]，超出部分会被 clip。
    path
        输出路径；相对路径按 :func:`output_dir` 解析。
    bits
        16 或 24。
    seed
        抖动随机种子。
    dither
        是否加入 TPDF 抖动。
    """
    path = Path(path)
    if not path.is_absolute():
        path = output_dir() / path

    if bits == 16:
        _write_wav_pcm(path, quantize_int16(signal, seed=seed, dither=dither), fs, 16)
    elif bits == 24:
        _write_wav_pcm(path, quantize_int24(signal, seed=seed, dither=dither), fs, 24)
    else:
        raise ValueError(f"不支持的位深: {bits}（可选 {BITS_CHOICES}）")
    return path


def signal_stats(signal: np.ndarray) -> Dict[str, float]:
    """统计一组信号的关键指标，便于 GUI / CLI 展示。"""
    x = np.asarray(signal, dtype=np.float64)
    if x.size == 0:
        return {}
    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(np.square(x))))
    crest = peak / rms if rms > 0 else 0.0
    dc = float(np.mean(x))
    clipped = int(np.sum(np.abs(x) > 1.0))
    return {
        "peak": peak,
        "peak_dbfs": 20.0 * np.log10(max(peak, 1e-20)),
        "rms": rms,
        "rms_dbfs": 20.0 * np.log10(max(rms, 1e-20)),
        "crest_db": 20.0 * np.log10(max(crest, 1e-20)),
        "dc": dc,
        "clipped_samples": clipped,
    }


# ---------------------------------------------------------------------------
# 读取
# ---------------------------------------------------------------------------

def _decode_pcm(data: bytes, bits: int) -> np.ndarray:
    """把 PCM 字节解码成 [-1, 1] 的 float64。

    注意必须用 ``np.frombuffer`` **重解释**字节（而不是 ``astype``）——
    ``astype`` 是逐元素转换，16 bit 的两个字节会被当成两个独立样本。
    """
    if bits == 8:
        # 8 bit PCM 是**无符号**的（中点 128）
        raw = np.frombuffer(data, dtype=np.uint8)
        return (raw.astype(np.float64) - 128.0) / 128.0
    if bits == 16:
        return np.frombuffer(data, dtype="<i2").astype(np.float64) / 32768.0
    if bits == 24:
        # 24 bit 存成 3 字节小端：先拼成 int32 再做符号扩展
        # 满量程是 ±2**23（不是 2**31 —— 位深决定量程）
        b = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3)
        value = (
            b[:, 0].astype(np.int32)
            | (b[:, 1].astype(np.int32) << 8)
            | (b[:, 2].astype(np.int32) << 16)
        )
        value = np.where(value >= (1 << 23), value - (1 << 24), value)
        return value.astype(np.float64) / 8388608.0
    if bits == 32:
        return np.frombuffer(data, dtype="<i4").astype(np.float64) / 2147483648.0
    raise ValueError(f"不支持的 PCM 位深: {bits} bit")


def _read_riff(path: Path) -> Tuple[Dict[str, Any], bytes]:
    """解析 RIFF / RF64 容器，返回 ``(格式信息, data 块的字节)``。

    自己解析而不用标准库 ``wave``，是因为 ``wave`` 不认 IEEE 浮点 WAV
    （format 3），而浮点 WAV 在音频工具里很常见。
    """
    blob = path.read_bytes()
    if len(blob) < 12:
        raise ValueError(f"文件太小，不是有效的 WAV: {path}")

    riff_id = blob[0:4]
    if riff_id not in (b"RIFF", b"RF64") or blob[8:12] != b"WAVE":
        raise ValueError(f"不是有效的 WAV 文件（缺少 RIFF/WAVE 头）: {path}")

    total = len(blob)
    pos = 12
    fmt: Optional[Dict[str, Any]] = None
    data_bytes: Optional[bytes] = None
    ds64_size: Optional[int] = None

    while pos + 8 <= total:
        chunk_id = blob[pos:pos + 4]
        size = int.from_bytes(blob[pos + 4:pos + 8], "little")
        body_start = pos + 8
        body_end = body_start + size
        if body_end > total:
            # 有些写文件工具把 data 长度写成 0 或 0xFFFFFFFF —— 取剩余全部
            body_end = total
            size = total - body_start

        body = blob[body_start:body_end]
        if chunk_id == b"fmt ":
            fmt = _parse_fmt_chunk(body)
        elif chunk_id == b"ds64":
            # RF64 的真实长度在 ds64 块里
            if len(body) >= 16:
                ds64_size = int.from_bytes(body[8:16], "little")
        elif chunk_id == b"data":
            data_bytes = body

        pos = body_start + size + (size & 1)  # chunk 按偶数字节对齐

    if fmt is None:
        raise ValueError(f"WAV 缺少 fmt 块: {path}")
    if data_bytes is None:
        raise ValueError(f"WAV 缺少 data 块: {path}")
    if ds64_size is not None and len(data_bytes) < ds64_size:
        raise ValueError(f"WAV 的 data 块长度与 ds64 声明不符: {path}")

    return fmt, data_bytes


def _parse_fmt_chunk(body: bytes) -> Dict[str, Any]:
    """解析 ``fmt `` 块。"""
    if len(body) < 16:
        raise ValueError("WAV 的 fmt 块长度不足")

    audio_format = int.from_bytes(body[0:2], "little")
    channels = int.from_bytes(body[2:4], "little")
    sample_rate = int.from_bytes(body[4:8], "little")
    block_align = int.from_bytes(body[12:14], "little")
    bits = int.from_bytes(body[14:16], "little")

    if audio_format == _WAVE_FORMAT_EXTENSIBLE and len(body) >= 40:
        # 可扩展格式：真正的格式藏在 SubFormat GUID 的头两个字节
        audio_format = int.from_bytes(body[24:26], "little")

    if channels < 1:
        raise ValueError(f"WAV 的声道数非法: {channels}")
    if block_align < 1:
        block_align = channels * max(1, bits // 8)

    return {
        "audio_format": audio_format,
        "channels": channels,
        "sample_rate": sample_rate,
        "block_align": block_align,
        "bits": bits,
    }


def wav_info(path) -> Dict[str, Any]:
    """读取 WAV 的基本信息（不解码采样数据）。

    返回字典字段：``path`` / ``channels`` / ``sample_rate`` / ``bits`` /
    ``duration`` / ``frames`` / ``encoding``。
    """
    p = Path(path)
    fmt, data_bytes = _read_riff(p)

    bytes_per_frame = int(fmt["block_align"])
    frames = len(data_bytes) // bytes_per_frame if bytes_per_frame else 0
    sample_rate = int(fmt["sample_rate"])

    if fmt["audio_format"] == _WAVE_FORMAT_IEEE_FLOAT:
        encoding = "IEEE 浮点"
    elif fmt["audio_format"] == _WAVE_FORMAT_PCM:
        encoding = "PCM"
    else:
        encoding = f"未知 (0x{fmt['audio_format']:04X})"

    return {
        "path": p,
        "channels": int(fmt["channels"]),
        "sample_rate": sample_rate,
        "bits": int(fmt["bits"]),
        "frames": frames,
        "duration": frames / sample_rate if sample_rate else 0.0,
        "encoding": encoding,
    }


def load_wav(path) -> Tuple[np.ndarray, int]:
    """读取 WAV 文件，返回 ``(浮点信号, 采样率)``。

    浮点信号的范围是 [-1, 1]（与 :func:`save_wav` 的约定一致），多声道
    文件会**下混成单声道**（各声道取平均）—— 分析用的都是单声道数据。

    支持的格式：

    * PCM 8 / 16 / 24 / 32 bit；
    * IEEE 浮点 32 / 64 bit（很多音频软件导出的格式）；
    * WAVE_FORMAT_EXTENSIBLE（按其 SubFormat 判定）。

    参数
    ----
    path
        WAV 文件路径。

    异常
    ----
    FileNotFoundError
        文件不存在。
    ValueError
        文件不是合法的 WAV，或位深不受支持。
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"WAV 文件不存在: {p}")

    fmt, data_bytes = _read_riff(p)
    channels = int(fmt["channels"])
    bits = int(fmt["bits"])
    block_align = int(fmt["block_align"])
    audio_format = int(fmt["audio_format"])

    # 丢掉不完整的尾帧
    usable = (len(data_bytes) // block_align) * block_align
    data_bytes = data_bytes[:usable]

    if audio_format == _WAVE_FORMAT_IEEE_FLOAT:
        if bits == 32:
            samples = np.frombuffer(data_bytes, dtype="<f4").astype(np.float64)
        elif bits == 64:
            samples = np.frombuffer(data_bytes, dtype="<f8").astype(np.float64)
        else:
            raise ValueError(f"不支持的浮点位深: {bits} bit")
    elif audio_format == _WAVE_FORMAT_PCM:
        samples = _decode_pcm(data_bytes, bits)
    else:
        raise ValueError(
            f"不支持的 WAV 编码 (0x{audio_format:04X})，"
            "仅支持 PCM 与 IEEE 浮点"
        )

    frames = samples.size // channels
    if frames == 0:
        return np.zeros(0, dtype=np.float64), int(fmt["sample_rate"])

    # 多声道下混成单声道（单声道时 mean 只是把 (N,1) 压回 (N,)）
    samples = samples[: frames * channels].reshape(frames, channels).mean(axis=1)

    return np.ascontiguousarray(samples, dtype=np.float64), int(fmt["sample_rate"])


# ---------------------------------------------------------------------------
# 通用音频读取（WAV 之外的格式交给 soundfile）
# ---------------------------------------------------------------------------

#: 常见但**不是** WAV 的音频扩展名 —— 用来说明"装上 soundfile 就能读这些"
NON_WAV_SUFFIXES = (".flac", ".ogg", ".oga", ".aiff", ".aif", ".aifc",
                    ".au", ".snd", ".w64", ".caf", ".mp3", ".m4a")


def _soundfile_module():
    """返回 ``soundfile`` 模块；没装就返回 ``None``。

    不装 soundfile 不影响 WAV —— WAV 走自研解析器，不依赖第三方库。
    """
    try:
        import soundfile  # noqa: PLC0415 - 延迟导入：这是可选依赖
    except Exception:  # noqa: BLE001 - 没装 / 装坏都当作不可用
        return None
    return soundfile


def load_audio(path) -> Tuple[np.ndarray, int]:
    """读取任意音频文件，返回 ``(浮点信号, 采样率)``。

    与 :func:`load_wav` 的关系：

    * **WAV 文件**走自研的 RIFF 解析器 —— 不依赖任何第三方库，
      且支持 IEEE 浮点 WAV（标准库 ``wave`` 不认）；
    * **其它格式**（FLAC / OGG / AIFF / MP3 …）交给 ``soundfile``
      （即 libsndfile），需要环境里额外装它。

    多声道一律**下混成单声道**（各声道取平均），返回值范围 [-1, 1]。

    参数
    ----
    path
        音频文件路径。

    异常
    ----
    FileNotFoundError
        文件不存在。
    ValueError
        文件格式无法识别（非 WAV 且没有可用的 ``soundfile``）。
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"音频文件不存在: {p}")

    suffix = p.suffix.lower()
    if suffix == ".wav" or suffix == ".wave":
        return load_wav(p)

    sf = _soundfile_module()
    if sf is None:
        raise ValueError(
            f"无法读取 {p.name}：这不是 WAV 文件，而读取 {suffix or '该'} 格式"
            "需要 soundfile（pip install soundfile）。"
            "WAV 文件不需要任何额外依赖。"
        )

    data, fs = sf.read(str(p), always_2d=True)
    # (N, C) -> (N,)：各声道取平均
    signal = np.asarray(data, dtype=np.float64).mean(axis=1)
    return np.ascontiguousarray(signal), int(fs)


def audio_info(path) -> Dict[str, Any]:
    """读取音频文件的基本信息（不解码采样数据）。

    WAV 用自研解析器，其它格式交给 ``soundfile``。返回字段与
    :func:`wav_info` 保持一致（``path`` / ``channels`` / ``sample_rate`` /
    ``bits`` / ``duration`` / ``frames`` / ``encoding``），
    额外给出 ``format``（容器格式名）。
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"音频文件不存在: {p}")

    if p.suffix.lower() in (".wav", ".wave"):
        info = dict(wav_info(p))
        info["format"] = "WAV"
        return info

    sf = _soundfile_module()
    if sf is None:
        raise ValueError(
            f"无法读取 {p.name}：这不是 WAV 文件，而读取 {p.suffix or '该'} 格式"
            "需要 soundfile（pip install soundfile）。"
        )

    raw = sf.info(str(p))
    subtype = str(getattr(raw, "subtype", "") or "")
    encoding = subtype or "未知"
    # 浮点类编码明确标出来，避免和 PCM 整数混为一谈
    if "FLOAT" in subtype.upper():
        encoding = f"{subtype}（浮点）"
    return {
        "path": p,
        "format": str(getattr(raw, "format", "") or p.suffix.lstrip(".")),
        "channels": int(raw.channels),
        "sample_rate": int(raw.samplerate),
        "bits": int(getattr(raw, "bits_per_sample", 0) or 0),
        "frames": int(raw.frames),
        "duration": float(raw.frames) / float(raw.samplerate) if raw.samplerate else 0.0,
        "encoding": encoding,
    }
