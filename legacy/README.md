# legacy —— 归档的早期脚本

这里的脚本是 `signal_gen` 包**成型之前**的独立实现，现已全部下沉到包内，
仅作为历史参考保留，不再维护、不参与测试。

| 文件 | 说明 | 已被取代的入口 |
|---|---|---|
| `chirp.py` | 扫频 / 步进扫频 | `python -m signal_gen generate sweep` / `generate stepped` |
| `multitone.py` | 多音信号 | `python -m signal_gen generate multitone` |
| `pink noise.py` | 粉噪声 | `python -m signal_gen generate pink` |
| `white noise.py` | 白噪声 | `python -m signal_gen generate white` |
| `step sweep signal.py` | 步进扫频 | `python -m signal_gen generate stepped` |
| `read signal.py` | 读外部音频 → 分析 → 双栏报告图 | `python -m signal_gen analyze 文件.wav --plot 报告.png` / `signal_gen.report` |

## 为什么归档

* 这些文件名带空格（`pink noise.py`），无法被 `import`，只能当脚本执行；
* 它们的实现与 `signal_gen/core.py` 完全重复 —— 留在根目录会造成
  "到底哪个是权威实现"的困惑；
* `tests/` 不覆盖它们，任何修改都不会被回归测试守住。

## 还能运行吗

能。每个文件都加了 `sys.path` 引导，直接运行即可：

```bash
python legacy/chirp.py
python legacy/"pink noise.py"
```

它们仍然调用的是**当前的** `signal_gen.core`，所以看到的行为与命令行一致；
但里面 `main()` 用的绘图代码是旧写法（抽稀绘图而非 min/max 包络），
仅供对照。

> **`read signal.py` 是个例外**：它额外依赖 `soundfile`（本机未安装），
> 而且硬编码的输入文件 `E:\Projects\D242\EIA-426B\EIA-426B 40-200Hz CF 6dB.wav`
> 已不存在 —— 直接运行会立刻报错。它的能力已经并入 `signal_gen.report`，
> 请改用 `python -m signal_gen analyze <文件> --plot <报告图>`。
