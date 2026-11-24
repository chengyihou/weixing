import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from helpers import make_model_dir  # noqa: E402


@pytest.fixture(scope='session')
def ts_model_dir(tmp_path_factory):
    """供测试侧各用例共用的最小可用模型文件夹（只读使用）"""
    return make_model_dir(tmp_path_factory.mktemp('model'))


@pytest.fixture(autouse=True)
def stop_after_each():
    """
    每个用例结束后停掉监听，避免相互影响（test_stream 里的 _STATE 是模块级单例）。
    用 sys.modules 判断而不是 import：Task 1 的用例跑的时候 test_stream.py 还不存在。
    """
    yield
    ts = sys.modules.get('test_stream')
    if ts is not None:
        ts._stop_for_tests()
