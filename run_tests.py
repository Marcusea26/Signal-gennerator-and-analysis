"""
测试入口（不依赖 pytest，直接用标准库 unittest）。

用法::

    python run_tests.py            # 跑全部用例
    python run_tests.py -v         # 逐条打印用例名
    python run_tests.py TestCore   # 只跑某个测试类

等价于::

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# 让 tests 包能被 import 到（tests/test_signal_gen.py 自己也做了同样的事）
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

TESTS_DIR = ROOT / "tests"


def build_suite(names: list[str] | None = None) -> unittest.TestSuite:
    """构造测试套件。

    ``names`` 为空时加载 tests 目录下的全部用例；否则按名字筛选
    （支持 ``TestCore`` 这样的类名，也支持 ``tests.test_signal_gen.TestCore``）。
    """
    loader = unittest.TestLoader()
    discovered = loader.discover(str(TESTS_DIR), top_level_dir=str(ROOT))
    if not names:
        return discovered

    selected = unittest.TestSuite()
    for test in _flatten(discovered):
        if any(name in test.id() for name in names):
            selected.addTest(test)
    return selected


def _flatten(suite: unittest.TestSuite):
    """展开嵌套的测试套件。"""
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _flatten(item)
        else:
            yield item


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    verbose = "-v" in argv or "--verbose" in argv
    names = [a for a in argv if not a.startswith("-")]

    suite = build_suite(names)
    if suite.countTestCases() == 0:
        print("没有匹配的测试用例", file=sys.stderr)
        return 1

    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
