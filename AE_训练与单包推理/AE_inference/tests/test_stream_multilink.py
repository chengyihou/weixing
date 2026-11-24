"""多路（link）并行的检查：各路独立监听、独立热更新、互不干扰。"""

import os
import sys
import threading
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import test_stream as ts                                   # noqa: E402
from helpers import drop_packet, make_model_dir, wait_until   # noqa: E402


@pytest.fixture
def dirs(tmp_path):
    """三个互不相同的监控目录"""
    out = {}
    for name in ('a', 'b', 'c'):
        d = tmp_path / name
        d.mkdir()
        out[name] = str(d)
    return out


def start(link, model_dir, watch_dir, sink, q=99):
    return ts.start_test(model_dir, watch_dir, sink.append, link=link)


# ---------- 同时开多路 ----------

def test_three_links_run_at_once(ts_model_dir, dirs):
    ra, rb, rc = [], [], []
    start('A', ts_model_dir, dirs['a'], ra)
    start('B', ts_model_dir, dirs['b'], rb)
    start('C', ts_model_dir, dirs['c'], rc)

    assert ts.running_links() == ['A', 'B', 'C']
    assert ts.is_running('A') and ts.is_running('B') and ts.is_running('C')


def test_each_link_gets_only_its_own_packets_and_link_name(ts_model_dir, dirs):
    ra, rb = [], []
    start('A', ts_model_dir, dirs['a'], ra)
    start('B', ts_model_dir, dirs['b'], rb)
    time.sleep(0.5)

    drop_packet(dirs['a'], 'x.iq', 65536 * 2, seed=1)
    assert wait_until(lambda: ra), 'A 路没收到自己的包'
    time.sleep(1.0)

    assert rb == [], 'B 路不应收到 A 目录里的包'
    assert ra[0]['link'] == 'A'
    assert list(ra[0])[0] == 'link', 'link 要在最前面'

    drop_packet(dirs['b'], 'y.iq', 65536 * 2, seed=2)
    assert wait_until(lambda: rb)
    assert rb[0]['link'] == 'B'


# ---------- 重复拒绝 ----------

def test_duplicate_link_rejected(ts_model_dir, dirs):
    start('A', ts_model_dir, dirs['a'], [])
    with pytest.raises(ValueError, match='link 已在监听'):
        start('A', ts_model_dir, dirs['b'], [])
    assert ts.running_links() == ['A'], '失败的启动不能留下残留'


def test_same_watch_dir_rejected(ts_model_dir, dirs):
    start('A', ts_model_dir, dirs['a'], [])
    with pytest.raises(ValueError, match='抢包'):
        start('B', ts_model_dir, dirs['a'], [])
    assert ts.running_links() == ['A']


def test_same_dir_via_different_spelling_rejected(ts_model_dir, dirs):
    """同一目录换个写法（末尾斜杠）也要识别"""
    start('A', ts_model_dir, dirs['a'], [])
    with pytest.raises(ValueError, match='抢包'):
        start('B', ts_model_dir, dirs['a'] + os.sep, [])


def test_concurrent_start_same_link_only_one_wins(monkeypatch, ts_model_dir, dirs):
    """两路同时启动同一个 link：检查与登记在同一把锁里，只能有一个成功"""
    import watchdog.observers as wo

    real_start = wo.Observer.start

    def slow_start(self):
        time.sleep(0.3)                 # 放大「检查之后、登记之前」的窗口
        return real_start(self)

    monkeypatch.setattr(wo.Observer, 'start', slow_start)

    outcomes = []

    def worker():
        try:
            ts.start_test(ts_model_dir, dirs['a'], lambda r: None, link='A')
            outcomes.append('ok')
        except ValueError as e:
            outcomes.append(str(e))

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert outcomes.count('ok') == 1, f'应当恰好一路成功：{outcomes}'
    assert any('link 已在监听' in o for o in outcomes), outcomes
    assert ts.running_links() == ['A']


@pytest.mark.parametrize('bad', ['', '   ', None, 123])
def test_bad_link_rejected(ts_model_dir, dirs, bad):
    with pytest.raises(ValueError, match='link'):
        start(bad, ts_model_dir, dirs['a'], [])
    assert not ts.is_running()


# ---------- change_config 按 link 定位 ----------

def test_change_config_only_changes_target_link(tmp_path, ts_model_dir, dirs):
    start('A', ts_model_dir, dirs['a'], [])
    start('B', ts_model_dir, dirs['b'], [])
    net_b = ts.get_config('B').state.net
    dir_b = ts.get_config('B').state.model_dir
    other = make_model_dir(tmp_path / 'other')

    returned = ts.change_config(other, link='A')

    assert returned == other
    assert ts.get_config('A').state.model_dir == other
    assert ts.get_config('B').state.model_dir == dir_b, 'B 的模型不能变'
    assert ts.get_config('B').state.net is net_b, 'B 的模型对象不能被换'


def test_change_config_without_link_needs_one_stream_only(tmp_path, ts_model_dir, dirs):
    """多于一路时不指定 link 必须报错（避免误改），只有一路时可省略"""
    a1 = make_model_dir(tmp_path / 'm1')
    a2 = make_model_dir(tmp_path / 'm2')
    start('A', ts_model_dir, dirs['a'], [])
    assert ts.change_config(a1) is not None                   # 单路：可省略

    start('B', ts_model_dir, dirs['b'], [])
    with pytest.raises(ValueError, match='必须指定 link'):
        ts.change_config(a2)
    assert ts.get_config('A').state.model_dir == a1, '报错时不能改动任何一路'


def test_change_config_unknown_link(tmp_path, ts_model_dir, dirs):
    start('A', ts_model_dir, dirs['a'], [])
    with pytest.raises(RuntimeError):
        ts.change_config(make_model_dir(tmp_path / 'm'), link='NOPE')


def test_failed_change_keeps_target_and_others_untouched(tmp_path, ts_model_dir, dirs):
    start('A', ts_model_dir, dirs['a'], [])
    start('B', ts_model_dir, dirs['b'], [])
    old_a = ts.get_config('A').state
    old_b = ts.get_config('B').state

    with pytest.raises(FileNotFoundError):
        ts.change_config(str(tmp_path / 'nope'), link='A')

    assert ts.get_config('A').state is old_a
    assert ts.get_config('B').state is old_b


# ---------- stop ----------

def test_stop_only_stops_that_link(ts_model_dir, dirs):
    ra, rb = [], []
    start('A', ts_model_dir, dirs['a'], ra)
    start('B', ts_model_dir, dirs['b'], rb)

    ts.stop('A')

    assert not ts.is_running('A')
    assert ts.is_running('B')
    assert ts.running_links() == ['B']

    time.sleep(0.5)
    drop_packet(dirs['b'], 'y.iq', 65536 * 2)
    assert wait_until(lambda: rb), 'A 被停后 B 仍应正常处理'


def test_stop_unknown_link(ts_model_dir, dirs):
    start('A', ts_model_dir, dirs['a'], [])
    with pytest.raises(ValueError, match='link 不存在'):
        ts.stop('NOPE')
    assert ts.is_running('A')


def test_stopped_link_can_be_restarted(ts_model_dir, dirs):
    start('A', ts_model_dir, dirs['a'], [])
    ts.stop('A')
    start('A', ts_model_dir, dirs['a'], [])
    assert ts.is_running('A')


# ---------- 一路出错，其他路不受影响 ----------

def test_one_link_failing_to_start_does_not_affect_others(tmp_path, ts_model_dir, dirs):
    start('A', ts_model_dir, dirs['a'], [])
    with pytest.raises(FileNotFoundError):
        start('B', str(tmp_path / 'nope'), dirs['b'], [])

    assert ts.running_links() == ['A']
    assert ts.is_running('A')


def test_callback_exception_in_one_link_does_not_affect_others(ts_model_dir, dirs):
    rb = []

    def bad(_r):
        raise RuntimeError('A 的回调炸了')

    ts.start_test(ts_model_dir, dirs["a"], bad, link="A")
    start('B', ts_model_dir, dirs['b'], rb)
    time.sleep(0.5)

    drop_packet(dirs['a'], 'x.iq', 65536 * 2)
    drop_packet(dirs['b'], 'y.iq', 65536 * 2)

    assert wait_until(lambda: rb), 'A 回调炸了不应影响 B'
    assert ts.is_running('A') and ts.is_running('B')


def test_each_link_has_its_own_lock(ts_model_dir, dirs):
    """每路一把锁：A 的锁被占着时，B 的包仍能处理"""
    ra, rb = [], []
    start('A', ts_model_dir, dirs['a'], ra)
    start('B', ts_model_dir, dirs['b'], rb)
    time.sleep(0.5)

    assert ts._link_lock('A') is not ts._link_lock('B')

    with ts._link_lock('A'):                        # 模拟 A 路正在处理、锁被占着
        drop_packet(dirs['b'], 'y.iq', 65536 * 2)
        assert wait_until(lambda: rb), 'A 的锁被占时 B 仍应能处理（否则全串行了）'
        assert ra == []
