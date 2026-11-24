import math
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import test_stream as ts                                                    # noqa: E402
from helpers import drop_packet, make_model_dir, wait_until                 # noqa: E402

# 回调应收到两项（link 放最前；score 是居中偏差；测试侧不输出阈值，也不发原始误差）
KEYS = {'link', 'type', 'score', 'score_sigma'}


@pytest.fixture
def watch_dir(tmp_path):
    d = tmp_path / 'stream'
    d.mkdir()
    return d


# 每个用例跑完自动停监听：autouse fixture `stop_after_each` 在 tests/conftest.py 里


def start(model_dir, watch_dir, on_result=None):
    return ts.start_test(model_dir=model_dir, watch_dir=str(watch_dir),
                         on_result=on_result or (lambda r: None))


def snapshot(d):
    """文件夹里每个文件的 (大小, 修改时间)，用来证明没被改动"""
    out = {}
    for root, _, files in os.walk(d):
        for f in files:
            p = os.path.join(root, f)
            st = os.stat(p)
            out[os.path.relpath(p, d)] = (st.st_size, st.st_mtime_ns)
    return out


# ---------- 大小端 ----------

def test_start_test_byte_order_little_endian(ts_model_dir, watch_dir):
    """byte_order='l' 时，小端推流包能被正常读出并处理"""
    received = []
    ts.start_test(model_dir=ts_model_dir, watch_dir=str(watch_dir),
                  on_result=received.append, byte_order='l')
    assert ts.get_config().state.byte_order == 'l'
    time.sleep(0.5)

    p = drop_packet(watch_dir, 'le.iq', 65536 * 7, seed=31, byte_order='l')

    assert wait_until(lambda: received)
    assert not math.isnan(received[0]['score'])
    assert wait_until(lambda: not os.path.exists(p))


def test_start_test_bad_byte_order_does_not_start(ts_model_dir, watch_dir):
    with pytest.raises(ValueError, match='byte_order'):
        ts.start_test(model_dir=ts_model_dir, watch_dir=str(watch_dir),
                      on_result=lambda r: None, byte_order='x')
    assert not ts.is_running()


# ---------- 启动 ----------

def test_start_test_returns_immediately(ts_model_dir, watch_dir):
    t0 = time.time()
    status = start(ts_model_dir, watch_dir)
    assert status == 'running'
    assert time.time() - t0 < 10.0, '目录为空时 start_test 必须立刻返回，不能阻塞'
    assert ts.is_running()


def test_start_test_twice_raises(ts_model_dir, watch_dir):
    """同一个 link（默认）再启动：拒绝，提示用 change_config"""
    start(ts_model_dir, watch_dir)
    with pytest.raises(ValueError, match='change_config'):
        start(ts_model_dir, watch_dir)


def test_start_test_bad_model_dir_does_not_start(tmp_path, watch_dir):
    with pytest.raises(FileNotFoundError):
        start(str(tmp_path / 'nope'), watch_dir)
    assert not ts.is_running()


def test_start_test_missing_checkpoint_does_not_start(tmp_path, watch_dir):
    d = make_model_dir(tmp_path, write_ckpt=False)
    with pytest.raises(FileNotFoundError):
        start(d, watch_dir)
    assert not ts.is_running()


def test_start_test_does_not_need_cal_block_err(tmp_path, watch_dir):
    """测试侧不再读 cal_block_err.npy：没有这个文件也能正常启动"""
    d = make_model_dir(tmp_path, write_cal=False)

    start(d, watch_dir)

    assert ts.is_running()


def test_start_test_bad_watch_dir_and_callback(tmp_path, ts_model_dir, watch_dir):
    with pytest.raises(ValueError, match='watch_dir'):
        ts.start_test(model_dir=ts_model_dir, watch_dir=str(tmp_path / 'nope'),
                      on_result=lambda r: None)
    with pytest.raises(ValueError, match='on_result'):
        ts.start_test(model_dir=ts_model_dir, watch_dir=str(watch_dir),
                      on_result='not callable')
    assert not ts.is_running()


# ---------- 来包即测、测完就删 ----------

def test_packet_is_tested_callback_fires_and_file_deleted(ts_model_dir, watch_dir):
    received = []
    start(ts_model_dir, watch_dir, on_result=received.append)
    time.sleep(0.5)

    p = drop_packet(watch_dir, 'pkg.iq', 65536 * 7, seed=7)

    assert wait_until(lambda: received), '回调没有被触发'
    r = received[0]
    assert set(r) == KEYS, '回调只有 link 和居中偏差分数'
    assert r['link'] == 'default'
    assert r['type'] == 'ae', '回调用 type 区分 ae / ml'
    assert isinstance(r['score'], float) and not math.isnan(r['score'])
    assert r['score'] >= 0, 'score 是偏差的中位数，非负'
    cfg = ts.get_config()
    assert r['score_sigma'] == pytest.approx(r['score'] / cfg.state.sigma, abs=1e-4)
    assert wait_until(lambda: not os.path.exists(p)), '处理完的包必须被删除'


def test_two_packets_in_a_row(ts_model_dir, watch_dir):
    received = []
    start(ts_model_dir, watch_dir, on_result=received.append)
    time.sleep(0.5)

    a = drop_packet(watch_dir, 'a.iq', 65536 * 7, seed=1)
    assert wait_until(lambda: len(received) == 1)
    b = drop_packet(watch_dir, 'b.iq', 65536 * 7, seed=2)
    assert wait_until(lambda: len(received) == 2)

    assert wait_until(lambda: not os.path.exists(a) and not os.path.exists(b))


def test_packet_dropped_before_start_is_processed(ts_model_dir, watch_dir):
    """启动前就躺在监控目录里的包，start_test 之后必须被处理并删除"""
    received = []
    p = drop_packet(watch_dir, 'early.iq', 65536 * 7, seed=21)   # 先放包，后启动

    start(ts_model_dir, watch_dir, on_result=received.append)

    assert wait_until(lambda: received), '启动前已存在的包没有被处理'
    assert set(received[0]) == KEYS
    assert received[0]['score'] is not None and not math.isnan(received[0]['score'])
    assert received[0]['score_sigma'] is not None
    assert wait_until(lambda: not os.path.exists(p)), '启动前的包处理完也必须删除'
    assert len(received) == 1, '同一个包只应回调一次'


def test_bad_packet_gives_none_and_is_deleted(ts_model_dir, watch_dir):
    """不足一块的包：回调里 score 为 None，文件照样删除"""
    received = []
    start(ts_model_dir, watch_dir, on_result=received.append)
    time.sleep(0.5)

    p = drop_packet(watch_dir, 'tiny.iq', 1000)

    assert wait_until(lambda: received)
    assert set(received[0]) == KEYS
    assert received[0]['score'] is None, '坏包回调给 None'
    assert wait_until(lambda: not os.path.exists(p)), '坏包也必须删除'


def test_unexpected_error_gives_none_prints_reason_and_deletes(
        ts_model_dir, watch_dir, monkeypatch, capsys):
    """处理中出意外错误：回调为 None，原因打在屏幕上，监听不死，文件照样删除"""
    received = []

    def boom(cfg, path):
        raise RuntimeError('推理炸了')
    monkeypatch.setattr(ts, 'process_packet', boom)

    start(ts_model_dir, watch_dir, on_result=received.append)
    time.sleep(0.5)

    p = drop_packet(watch_dir, 'x.iq', 65536 * 2)

    assert wait_until(lambda: received)
    assert received[0]['score'] is None
    assert wait_until(lambda: not os.path.exists(p))
    out = capsys.readouterr().out
    assert 'RuntimeError' in out and '推理炸了' in out, '原因应打在屏幕上'
    assert ts.is_running()


def test_handle_packet_skips_vanished_file(ts_model_dir, watch_dir):
    received = []
    start(ts_model_dir, watch_dir, on_result=received.append)

    assert ts.handle_packet(ts.get_config(), str(watch_dir / 'gone.iq')) is None
    assert received == []


def test_callback_exception_does_not_kill_watcher(ts_model_dir, watch_dir):
    calls = []

    def bad_callback(r):
        calls.append(r)
        raise RuntimeError('外部程序的函数炸了')

    start(ts_model_dir, watch_dir, on_result=bad_callback)
    time.sleep(0.5)

    p1 = drop_packet(watch_dir, 'a.iq', 65536 * 7, seed=13)
    assert wait_until(lambda: len(calls) == 1), '第一次回调没触发'
    assert ts.is_running(), '回调抛异常把监听搞死了'
    assert wait_until(lambda: not os.path.exists(p1)), '回调抛异常时，包仍应被删除'

    drop_packet(watch_dir, 'b.iq', 65536 * 7, seed=14)
    assert wait_until(lambda: len(calls) == 2), '回调抛异常后，后续的包没被测'


# ---------- 测试过程不改动模型文件夹 ----------

def test_model_dir_is_never_written(tmp_path, watch_dir):
    d = make_model_dir(tmp_path / 'm')
    before = snapshot(d)
    received = []

    start(d, watch_dir, on_result=received.append)
    time.sleep(0.5)
    drop_packet(watch_dir, 'a.iq', 65536 * 3, seed=1)
    drop_packet(watch_dir, 'tiny.iq', 1000)
    assert wait_until(lambda: len(received) == 2)
    ts.change_config(d)

    assert snapshot(d) == before, '测试过程不应改动模型文件夹'


# ---------- change_config ----------

def test_change_config_swaps_model_as_a_whole(tmp_path, ts_model_dir, watch_dir):
    start(ts_model_dir, watch_dir)
    cfg = ts.get_config()
    old_state = cfg.state
    observer = ts._STREAMS['default']['observer']
    new_dir = make_model_dir(tmp_path / 'other')

    returned = ts.change_config(new_dir)

    assert cfg.state is not old_state
    assert cfg.state.net is not old_state.net
    assert cfg.state.model_dir == new_dir
    assert returned == new_dir
    assert ts.get_config() is cfg, 'change_config 不能重建配置对象'
    assert ts._STREAMS['default']['observer'] is observer and ts.is_running(), '不能动监听'


def test_change_config_keeps_byte_order(tmp_path, ts_model_dir, watch_dir):
    ts.start_test(model_dir=ts_model_dir, watch_dir=str(watch_dir),
                  on_result=lambda r: None, byte_order='l')
    new_dir = make_model_dir(tmp_path / 'other')

    ts.change_config(new_dir)

    assert ts.get_config().state.byte_order == 'l', '换模型不能改大小端'


def test_change_config_bad_model_dir_keeps_old_state(tmp_path, ts_model_dir, watch_dir):
    start(ts_model_dir, watch_dir)
    cfg = ts.get_config()
    old_state = cfg.state
    broken = make_model_dir(tmp_path / 'broken', write_ckpt=False)

    with pytest.raises(FileNotFoundError):
        ts.change_config(broken)
    with pytest.raises(FileNotFoundError):
        ts.change_config(str(tmp_path / 'nope'))

    assert cfg.state is old_state, '换模型失败时原配置不能变'


def test_change_config_without_start_raises(tmp_path):
    d = make_model_dir(tmp_path / 'm')
    with pytest.raises(RuntimeError, match='start_test'):
        ts.change_config(d)


def test_packet_uses_one_consistent_snapshot(tmp_path, ts_model_dir, watch_dir, monkeypatch):
    """处理包的过程中换了模型：这个包仍用开始时的模型，下一个包才用新的"""
    start(ts_model_dir, watch_dir)
    cfg = ts.get_config()
    old_net = cfg.state.net
    other = make_model_dir(tmp_path / 'other')
    p = tmp_path / 'pkg.iq'
    from helpers import write_iq
    write_iq(str(p), 65536 * 2, seed=3)

    real = ts.block_errs
    seen = []

    def sneaky(net, blocks):
        seen.append(net)
        ts.change_config(other)              # 推理进行到一半，模型被换了
        return real(net, blocks)
    monkeypatch.setattr(ts, 'block_errs', sneaky)

    r = ts.process_packet(cfg, str(p))

    assert seen and seen[0] is old_net, '这个包必须用开始时的模型'
    assert 'threshold' not in r, '测试侧不再输出阈值'
    assert cfg.state.net is not old_net, '配置本身已经换成新的'
