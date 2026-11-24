"""_wait_until_stable 的检查：写一半的包不读不删，写完才处理。"""

import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import test_stream as ts                    # noqa: E402
from helpers import drop_packet             # noqa: E402


@pytest.fixture
def watch_dir(tmp_path):
    d = tmp_path / 'stream'
    d.mkdir()
    return d


def test_wait_returns_false_for_missing_file(tmp_path):
    assert ts._wait_until_stable(str(tmp_path / 'nope.iq'), checks=1, interval=0.01) is False


def test_wait_returns_false_for_empty_file(tmp_path):
    p = tmp_path / 'empty.iq'
    p.write_bytes(b'')
    assert ts._wait_until_stable(str(p), checks=1, interval=0.01, max_wait=0.3) is False


def test_wait_returns_true_for_stable_file(tmp_path):
    p = tmp_path / 'ok.iq'
    p.write_bytes(b'\x00' * 1024)
    assert ts._wait_until_stable(str(p), checks=2, interval=0.02, max_wait=2.0) is True


def test_wait_keeps_waiting_while_growing(tmp_path):
    """文件持续增长时不应判为稳定"""
    p = tmp_path / 'grow.iq'
    p.write_bytes(b'\x00' * 1024)
    stop = threading.Event()

    def writer():
        while not stop.is_set():
            with open(p, 'ab') as f:
                f.write(b'\x00' * 512)
            time.sleep(0.05)

    t = threading.Thread(target=writer, daemon=True)
    t.start()
    try:
        assert ts._wait_until_stable(str(p), checks=3, interval=0.1, max_wait=0.8) is False
    finally:
        stop.set()
        t.join(timeout=2)


def test_half_written_packet_is_not_processed_nor_deleted(ts_model_dir, watch_dir):
    """包还在增长时不回调、不删文件；写完后处理并删除"""
    got = []
    ts.start_test(ts_model_dir, str(watch_dir), got.append)

    p = watch_dir / 'slow.iq'
    with open(p, 'wb') as f:
        f.write(b'\x00' * (65536 * 2 * 2))       # 先写一部分
        f.flush()
        time.sleep(0.5)
        assert got == [], '文件还在写，不应产生结果'
        assert p.exists(), '文件还在写，不应删除'
        f.write(b'\x00' * (65536 * 2 * 2))       # 再写一部分，触发新的 modified

    # 写完后等处理
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and not got:
        time.sleep(0.1)

    assert got, '写完后应被处理'
    assert not p.exists(), '处理后应删除'
