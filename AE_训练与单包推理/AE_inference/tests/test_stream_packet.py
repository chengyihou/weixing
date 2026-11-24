import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import test_stream as ts                                    # noqa: E402
from helpers import write_iq                                # noqa: E402


@pytest.fixture
def cfg(ts_model_dir):
    return ts.Config(state=ts.load_bundle(ts_model_dir), watch_dir='.', on_result=lambda r: None)


def test_process_packet_seven_blocks_drops_tail(cfg, tmp_path):
    p = tmp_path / 'pkg.iq'
    write_iq(str(p), 65536 * 7 + 999, seed=3)         # 7 块 + 残点

    r = ts.process_packet(cfg, str(p))

    assert r['blocks'] == 7
    assert r['block_err'].shape == (7,)
    assert r['median_block_err'] == pytest.approx(float(np.median(r['block_err'])))
    center = cfg.state.center
    assert r['score'] == pytest.approx(float(np.median(np.abs(r['block_err'] - center))))
    assert r['packet'] == 'pkg.iq'
    assert 'reason' not in r
    assert os.path.exists(p), 'process_packet 不负责删文件'


def test_even_block_count_uses_mean_of_middle_two(cfg, tmp_path):
    p = tmp_path / 'pkg8.iq'
    write_iq(str(p), 65536 * 8, seed=4)

    r = ts.process_packet(cfg, str(p))

    s = np.sort(r['block_err'])
    assert r['blocks'] == 8
    assert r['median_block_err'] == pytest.approx(float((s[3] + s[4]) / 2))
    dev = np.sort(np.abs(r['block_err'] - cfg.state.center))
    assert r['score'] == pytest.approx(float((dev[3] + dev[4]) / 2))


def test_block_err_uses_61_overlapping_windows(cfg, tmp_path):
    """块误差必须与训练/校准同口径：每块 61 个窗口（4096 / 步长 1024），窗口先除以自身 RMS。用最朴素的逐窗口写法做参照"""
    p = tmp_path / 'pkg.iq'
    write_iq(str(p), 65536 * 2, seed=9)
    blocks = ts.to_blocks(ts.load_iq(str(p)))
    assert blocks.shape == (2, 65536)
    expect = []
    with torch.no_grad():
        for b in blocks:
            errs = []
            for k in range(61):
                seg_iq = b[k * 1024:k * 1024 + 4096]
                seg = np.stack([seg_iq.real, seg_iq.imag], axis=0).astype(np.float32)  # (2, 4096)
                seg = (seg / max(np.sqrt((seg ** 2).mean()), 1e-8)).astype(np.float32)
                x = torch.from_numpy(seg)[None].to(ts.DEVICE)
                errs.append(float((cfg.state.net(x) - x).pow(2).mean()))
            expect.append(np.mean(errs))

    r = ts.process_packet(cfg, str(p))

    assert np.allclose(r['block_err'], expect, rtol=1e-4)


def test_load_iq_little_endian_same_signal(cfg, tmp_path):
    """byte_order='l' 读小端包，与同样内容的大端包得到同一结果"""
    b = tmp_path / 'big.iq'
    l = tmp_path / 'lit.iq'
    write_iq(str(b), 65536 * 2, seed=8, byte_order='b')
    write_iq(str(l), 65536 * 2, seed=8, byte_order='l')

    cfg_l = ts.Config(state=ts.replace(cfg.state, byte_order='l'), watch_dir='.',
                      on_result=lambda r: None)
    r_b = ts.process_packet(cfg, str(b))
    r_l = ts.process_packet(cfg_l, str(l))

    assert r_l['blocks'] == r_b['blocks'] == 2
    assert r_l['block_err'] == pytest.approx(r_b['block_err'])


def test_load_iq_bad_byte_order(cfg, tmp_path):
    p = tmp_path / 'pkg.iq'
    write_iq(str(p), 65536 * 2, seed=9)

    with pytest.raises(ValueError, match='byte_order'):
        ts.load_iq(str(p), byte_order='x')


def test_process_packet_too_short(cfg, tmp_path):
    p = tmp_path / 'tiny.iq'
    write_iq(str(p), 1000)

    r = ts.process_packet(cfg, str(p))

    assert r['blocks'] == 0
    assert math.isnan(r['median_block_err'])
    assert math.isnan(r['score'])
    assert '不足一个块' in r['reason']


def test_process_packet_empty_file(cfg, tmp_path):
    p = tmp_path / 'empty.iq'
    p.write_bytes(b'')

    r = ts.process_packet(cfg, str(p))

    assert math.isnan(r['median_block_err'])
    assert math.isnan(r['score'])
    assert '空文件' in r['reason']


def test_process_packet_odd_int16_count(cfg, tmp_path):
    p = tmp_path / 'odd.iq'
    np.array([1, 2, 3], dtype='>i2').tofile(str(p))      # 3 个 int16，无法组成 I/Q 对

    r = ts.process_packet(cfg, str(p))

    assert math.isnan(r['median_block_err'])
    assert math.isnan(r['score'])
    assert '奇数' in r['reason']


def test_format_result_normal_packet(cfg, tmp_path):
    p = tmp_path / 'pkg.iq'
    write_iq(str(p), 65536 * 7, seed=11)

    line = ts.format_result(ts.process_packet(cfg, str(p)))

    assert 'pkg.iq' in line
    assert 'median_block_err=' in line
    assert 'score=' in line


def test_format_result_bad_packet_shows_reason(cfg, tmp_path):
    p = tmp_path / 'tiny.iq'
    write_iq(str(p), 1000)

    line = ts.format_result(ts.process_packet(cfg, str(p)))

    assert 'tiny.iq' in line
    assert '不足一个块' in line
