"""start_stream / reload_model 的检查：消息格式、发送失败不影响监听、取消只停自己。"""

import asyncio
import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import test_stream as ts                    # noqa: E402
from helpers import drop_packet, make_model_dir   # noqa: E402


@pytest.fixture
def watch_dir(tmp_path):
    d = tmp_path / 'stream'
    d.mkdir()
    return d


class FakeClient:
    def __init__(self, fail=False):
        self.sent = []
        self.fail = fail

    async def send(self, message):
        if self.fail:
            raise ConnectionError('模拟断线')
        self.sent.append(message)


def _feed(watch_dir, delay=0.5, name='pkg.iq', blocks=3):
    threading.Thread(
        target=lambda: (time.sleep(delay), drop_packet(watch_dir, name, 65536 * blocks)),
        daemon=True,
    ).start()


async def _wait_for(pred, timeout=30.0):
    deadline = time.monotonic() + timeout
    while not pred() and time.monotonic() < deadline:
        await asyncio.sleep(0.05)
    return pred()


async def _cancel(*tasks):
    for t in tasks:
        t.cancel()
    for t in tasks:
        try:
            await t
        except asyncio.CancelledError:
            pass


# ---------- start_stream ----------

def test_sends_inferring_success_message(monkeypatch, ts_model_dir, watch_dir):
    fake = FakeClient()
    monkeypatch.setattr(ts, 'client', fake)
    _feed(watch_dir)

    async def go():
        task = asyncio.create_task(ts.start_stream('ae_1', ts_model_dir, str(watch_dir)))
        await _wait_for(lambda: fake.sent)
        await _cancel(task)

    asyncio.run(go())

    assert fake.sent, '结果应由 start_stream 自己发出'
    m = fake.sent[0]
    assert m['type'] == 'inferring-success'
    assert set(m) == {'type', 'data'}
    assert list(m['data'])[0] == 'link', 'link 必须在 data 内部最前面'
    assert m['data']['link'] == 'ae_1'
    assert set(m['data']) == {'link', 'type', 'score', 'score_sigma'}, \
        '测试侧发 link、type、居中偏差和 σ 倍数，不含阈值，也不发原始误差'


def test_send_failure_does_not_stop_listener(monkeypatch, ts_model_dir, watch_dir):
    monkeypatch.setattr(ts, 'client', FakeClient(fail=True))
    _feed(watch_dir, name='a.iq')

    async def go():
        task = asyncio.create_task(ts.start_stream('ae_1', ts_model_dir, str(watch_dir)))
        await asyncio.sleep(3.0)
        assert ts.is_running('ae_1'), '发送失败不应终止监听'
        await _cancel(task)

    asyncio.run(go())


def test_cancel_stops_only_that_link(monkeypatch, ts_model_dir, tmp_path):
    monkeypatch.setattr(ts, 'client', FakeClient())
    da, db = tmp_path / 'a', tmp_path / 'b'
    da.mkdir()
    db.mkdir()

    async def go():
        ta = asyncio.create_task(ts.start_stream('A', ts_model_dir, str(da)))
        tb = asyncio.create_task(ts.start_stream('B', ts_model_dir, str(db)))
        await _wait_for(lambda: ts.is_running('A') and ts.is_running('B'))
        await _cancel(ta)
        await asyncio.sleep(0.2)
        assert not ts.is_running('A'), '取消 A 后 A 应已停止'
        assert ts.is_running('B'), '取消 A 不应影响 B'
        await _cancel(tb)

    asyncio.run(go())
    assert not ts.is_running()


def test_bad_link_raises(monkeypatch, ts_model_dir, watch_dir):
    monkeypatch.setattr(ts, 'client', FakeClient())
    with pytest.raises(ValueError, match='link'):
        asyncio.run(ts.start_stream('', ts_model_dir, str(watch_dir)))
    assert not ts.is_running()


def test_cancel_during_startup_leaves_no_link_running(monkeypatch, ts_model_dir, watch_dir):
    """启动阶段（线程还在 start_test 里）就取消：必须等线程跑完再清理，不能留下没人管的 link"""
    monkeypatch.setattr(ts, 'client', FakeClient())
    real_start = ts.start_test
    entered = threading.Event()

    def slow_start(*args, **kwargs):
        entered.set()
        time.sleep(1.5)                     # 模拟补扫积压的包
        return real_start(*args, **kwargs)

    monkeypatch.setattr(ts, 'start_test', slow_start)

    async def go():
        task = asyncio.create_task(ts.start_stream('ae_1', ts_model_dir, str(watch_dir)))
        await asyncio.to_thread(entered.wait)
        assert not ts.is_running('ae_1'), '这一步线程还没启动完监听'
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(go())

    assert not ts.is_running('ae_1'), '取消后不能留下监听器'
    assert ts.running_links() == []
    # 同一个 link 能重新启动，说明没有残留占用（start_test 不经过 client，这里直接调）
    ts.start_test(ts_model_dir, str(watch_dir), lambda r: None, link='ae_1')
    assert ts.is_running('ae_1')


def test_start_stream_twice_does_not_stop_the_first(monkeypatch, ts_model_dir, watch_dir):
    """重复启动同一个 link：第二次抛 ValueError，第一次的监听必须还活着"""
    monkeypatch.setattr(ts, 'client', FakeClient())

    async def go():
        first = asyncio.create_task(ts.start_stream('ae_1', ts_model_dir, str(watch_dir)))
        await _wait_for(lambda: ts.is_running('ae_1'))
        assert ts.is_running('ae_1')

        with pytest.raises(ValueError, match='link 已在监听'):
            await asyncio.wait_for(
                ts.start_stream('ae_1', ts_model_dir, str(watch_dir)), timeout=10)

        assert ts.is_running('ae_1'), '第二次启动失败不能把第一次的监听停掉'
        await _cancel(first)

    asyncio.run(go())


def test_missing_model_raises(monkeypatch, tmp_path, watch_dir):
    monkeypatch.setattr(ts, 'client', FakeClient())
    with pytest.raises(FileNotFoundError):
        asyncio.run(ts.start_stream('ae_1', str(tmp_path / 'nope'), str(watch_dir)))


# ---------- reload_model ----------

def test_reload_model_sends_change_success_with_link(monkeypatch, tmp_path, ts_model_dir, watch_dir):
    fake = FakeClient()
    monkeypatch.setattr(ts, 'client', fake)
    other = make_model_dir(tmp_path / 'other', cal_values=[0.5] * 20)

    async def go():
        task = asyncio.create_task(ts.start_stream('ae_1', ts_model_dir, str(watch_dir)))
        await _wait_for(lambda: ts.is_running('ae_1'))
        new_dir = await ts.reload_model('ae_1', other)
        await _cancel(task)
        return new_dir

    new_dir = asyncio.run(go())

    assert new_dir == other
    assert {'type': 'change-success', 'link': 'ae_1'} in fake.sent


def test_reload_model_failure_sends_nothing(monkeypatch, ts_model_dir, tmp_path, watch_dir):
    fake = FakeClient()
    monkeypatch.setattr(ts, 'client', fake)

    async def go():
        task = asyncio.create_task(ts.start_stream('ae_1', ts_model_dir, str(watch_dir)))
        await _wait_for(lambda: ts.is_running('ae_1'))
        with pytest.raises(FileNotFoundError):
            await ts.reload_model('ae_1', str(tmp_path / 'nope'))
        await _cancel(task)

    asyncio.run(go())
    assert all(m.get('type') != 'change-success' for m in fake.sent), '失败时不能发成功消息'
