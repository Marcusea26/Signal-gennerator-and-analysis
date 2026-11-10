"""
信号播放。

只依赖标准库 ``winsound``（Windows 自带），因此不需要 sounddevice /
simpleaudio 等第三方音频库。非 Windows 平台会明确提示"不支持播放"，
而不是静默失败。

实现要点：

1. ``winsound`` 只能播放 WAV 文件，且不支持 24 bit PCM，
   因此播放前统一转成 **16 bit WAV** 写到系统临时目录；
2. 用 ``SND_ASYNC`` 异步播放，播放期间 GUI 不阻塞；
3. 用 ``SND_PURGE`` 停止播放，并清理临时文件；
4. 临时文件在停止/退出时删除，不会堆积；
5. **播放结束自动停止**：用墙钟定时器在播放结束后触发一次清理，
   调用方轮询 :attr:`AudioPlayer.playing` 即可知道播放是否还在进行。

为什么是"墙钟"而不是完成回调
----------------------------
``winsound`` 没有播放完成回调。实测（本机 Windows + waveOut）：

* ``SND_SYNC`` 阻塞播放能精确知道何时结束，但**另一个线程调用
  ``SND_PURGE`` 根本打不断它**（实测 8 秒的播放纹丝不动）——
  这会让"停止"按钮失效；
* ``SND_ASYNC`` 播放可以被 ``SND_PURGE`` 立刻打断，但拿不到完成事件。

所以只能折中：用 ``SND_ASYNC`` 播放（保证"停止"按钮有效），
再用墙钟推算播放结束时刻。实测声卡的固定开销约 **0.25 s**
（3 秒文件实测 3.256 s，5 秒文件实测 5.243 s），
因此结束时刻取 ``时长 + END_MARGIN_S``，既不会误切掉尾音，
也不会让界面多等太久。
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from .audio_io import save_wav

__all__ = ["AudioPlayer", "PLAYBACK_BACKEND", "playback_supported"]

try:  # pragma: no cover - 平台相关
    import winsound
except ImportError:  # pragma: no cover - 非 Windows
    winsound = None  # type: ignore[assignment]


def playback_supported() -> bool:
    """当前系统是否支持内置播放。"""
    return winsound is not None


#: 播放后端名称（用于界面提示）
PLAYBACK_BACKEND = "winsound (Windows)" if playback_supported() else "不可用"


class AudioPlayer:
    """把浮点信号写成临时 WAV 并异步播放。"""

    #: 播放结束后额外等待的秒数 —— 覆盖声卡缓冲排空的开销（实测约 0.25 s）
    END_MARGIN_S = 0.35

    def __init__(self) -> None:
        self._path: Optional[Path] = None
        self._playing = False
        self._timer: Optional[threading.Timer] = None

    # ------------------------------------------------------------------
    @property
    def supported(self) -> bool:
        return playback_supported()

    @property
    def playing(self) -> bool:
        """当前是否正在播放。"""
        return self._playing

    @property
    def current_path(self) -> Optional[Path]:
        return self._path

    # ------------------------------------------------------------------
    def play(self, signal: np.ndarray, fs: int, gain: float = 1.0) -> float:
        """播放信号，返回信号时长 (s)。

        参数
        ----
        signal
            浮点样本，超出 [-1, 1] 的部分会被裁剪。
        fs
            采样率 (Hz)。
        gain
            播放增益（线性），用于避免满幅信号长时间聆听过于刺耳。
        """
        if not self.supported:
            raise RuntimeError(
                "当前系统没有内置音频播放后端（内置播放仅支持 Windows 的 winsound）"
            )

        x = np.asarray(signal, dtype=np.float64)
        if x.size == 0:
            raise ValueError("信号为空，无法播放")
        x = np.clip(x * float(gain), -1.0, 1.0)

        self.stop()

        fd, name = tempfile.mkstemp(prefix="signal_gen_play_", suffix=".wav")
        os.close(fd)
        path = Path(name)
        # winsound 不支持 24 bit PCM，统一用 16 bit；播放不需要抖动
        save_wav(x, path, fs=int(fs), bits=16, dither=False)

        try:
            winsound.PlaySound(  # type: ignore[union-attr]
                str(path), winsound.SND_FILENAME | winsound.SND_ASYNC
            )
        except RuntimeError as exc:
            self._cleanup()
            raise RuntimeError(f"播放失败: {exc}") from exc

        self._path = path
        self._playing = True

        # 播完自动收尾：到点后停止播放并删除临时文件
        duration = x.size / float(fs)
        timer = threading.Timer(duration + self.END_MARGIN_S, self._on_finished)
        timer.daemon = True
        self._timer = timer
        timer.start()
        return duration

    def stop(self) -> None:
        """停止播放并清理临时文件。"""
        self._cancel_timer()
        self._purge()
        self._cleanup()
        # 清理完成后再清标志 —— 外部看到 playing=False 时，
        # 临时文件一定已经处理干净了
        self._playing = False

    # ------------------------------------------------------------------
    def _on_finished(self) -> None:
        """播放自然结束后的收尾（由墙钟定时器触发）。"""
        self._timer = None
        # 正常情况下声音已经放完，这里再 PURGE 一次只是保险
        self._purge()
        self._cleanup()
        self._playing = False

    def _cancel_timer(self) -> None:
        timer = self._timer
        self._timer = None
        if timer is not None:
            timer.cancel()

    def _purge(self) -> None:
        if self.supported:
            try:
                winsound.PlaySound(None, winsound.SND_PURGE)  # type: ignore[union-attr]
            except RuntimeError:
                pass

    def _cleanup(self) -> None:
        path = self._path
        if path is None:
            return
        self._path = None
        try:
            path.unlink(missing_ok=True)
        except OSError:
            # 声卡可能还占着文件句柄；等播放彻底结束后
            # 下一次 stop() / play() 会再清一次
            pass
