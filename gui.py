"""图形界面启动脚本。

用法::

    python gui.py
"""

import sys
from pathlib import Path

# 保证直接运行本文件时也能 import 到同级的 signal_gen 包
sys.path.insert(0, str(Path(__file__).resolve().parent))

from signal_gen.gui import main

if __name__ == "__main__":
    sys.exit(main())
