import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import test_stream as ts                  # noqa: E402
from helpers import make_model_dir        # noqa: E402


# ---------- load_bundle：只读模型权重，不碰校准误差 ----------

def test_load_bundle_ok(tmp_path):
    d = make_model_dir(tmp_path)

    b = ts.load_bundle(d)

    assert isinstance(b.net, torch.nn.Module)
    assert not b.net.training                         # 已切到 eval
    assert b.model_dir == d
    assert b.byte_order == 'b'


def test_load_bundle_has_no_threshold_fields(tmp_path):
    """测试侧不持有阈值：Bundle 里不应再有 quantile / threshold / cal_block_err"""
    b = ts.load_bundle(make_model_dir(tmp_path))

    assert not hasattr(b, 'threshold')
    assert not hasattr(b, 'quantile')
    assert not hasattr(b, 'cal_block_err')


def test_load_bundle_ok_without_cal_block_err(tmp_path):
    """没有 cal_block_err.npy 也能加载（测试侧不再读它）"""
    d = make_model_dir(tmp_path, write_cal=False)

    assert ts.load_bundle(d).model_dir == d


def test_load_bundle_missing_checkpoint(tmp_path):
    d = make_model_dir(tmp_path, write_ckpt=False)
    with pytest.raises(FileNotFoundError, match='last.pth'):
        ts.load_bundle(d)


def test_load_bundle_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        ts.load_bundle(str(tmp_path / 'nope'))


# ---------- load_config：读输入 + 回调 ----------

def test_load_config_ok(tmp_path, ts_model_dir):
    watch = tmp_path / 'stream'
    watch.mkdir()
    cb = lambda r: None            # noqa: E731

    cfg = ts.load_config(ts_model_dir, str(watch), cb)

    assert isinstance(cfg.state, ts.Bundle)
    assert cfg.state.byte_order == 'b'
    assert cfg.watch_dir == str(watch)
    assert cfg.on_result is cb
    assert cfg.link == 'default'


def test_load_config_link(tmp_path, ts_model_dir):
    watch = tmp_path / 'stream'
    watch.mkdir()

    cfg = ts.load_config(ts_model_dir, str(watch), lambda r: None, link='ae_1')

    assert cfg.link == 'ae_1'


def test_load_config_bad_watch_dir(tmp_path, ts_model_dir):
    with pytest.raises(ValueError, match='watch_dir'):
        ts.load_config(ts_model_dir, str(tmp_path / 'nope'), lambda r: None)


def test_load_config_bad_callback(tmp_path, ts_model_dir):
    with pytest.raises(ValueError, match='on_result'):
        ts.load_config(ts_model_dir, str(tmp_path), 'not callable')


def test_load_config_missing_checkpoint(tmp_path):
    d = make_model_dir(tmp_path / 'm', write_ckpt=False)
    with pytest.raises(FileNotFoundError):
        ts.load_config(d, str(tmp_path), lambda r: None)
