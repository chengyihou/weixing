import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Conv1DAE          # noqa: E402


def write_iq(path, n_samples, seed=0, scale=1000, byte_order='b'):
    """写一个 .iq：int16，I/Q 交错（byte_order 'b' 大端 / 'l' 小端）"""
    dtype = '>i2' if byte_order == 'b' else '<i2'
    rng = np.random.default_rng(seed)
    data = rng.integers(-scale, scale, size=n_samples * 2, dtype=np.int16)
    data.astype(dtype).tofile(path)


def drop_packet(watch_dir, name, n_samples, seed=0, byte_order='b'):
    """
    模拟"整包放入"：先在监听目录旁边的暂存目录写好，再整体移进监听目录，
    这样 watchdog 看到文件时它已经是完整的。返回最终路径。
    """
    staging = os.path.join(os.path.dirname(str(watch_dir)), '_staging')
    os.makedirs(staging, exist_ok=True)
    tmp = os.path.join(staging, name)
    write_iq(tmp, n_samples, seed=seed, byte_order=byte_order)
    final = os.path.join(str(watch_dir), name)
    os.replace(tmp, final)
    return final


def wait_until(cond, timeout=60.0, interval=0.2):
    """轮询到 cond() 为真或超时，返回最后一次 cond() 的布尔值"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(interval)
    return bool(cond())


def make_model_dir(root, cal_values=None, write_ckpt=True, write_cal=True,
                   write_info=True, center=None):
    """
    造一个最小可用的模型文件夹（随机初始化的模型 + 合成的校准块误差），
    内容与 train() 最终留下的一致：checkpoints/last.pth、cal_block_err.npy、model_info.json。
    center 不给时取 cal_values 的中位数；write_info=False 用来模拟没有 model_info.json 的旧模型。
    """
    d = os.path.join(str(root), '003_20261014_153012')
    os.makedirs(d, exist_ok=True)
    if write_ckpt:
        torch.manual_seed(0)
        os.makedirs(os.path.join(d, 'checkpoints'), exist_ok=True)
        torch.save(Conv1DAE(seq_len=4096, latent_dim=8).state_dict(),
                   os.path.join(d, 'checkpoints', 'last.pth'))
    if cal_values is None and (write_cal or write_info):
        cal_values = np.random.default_rng(0).gamma(2.0, 0.001, size=200)
    if write_cal:
        np.save(os.path.join(d, 'cal_block_err.npy'), np.asarray(cal_values, dtype=np.float64))
    if write_info:
        c = float(np.median(cal_values)) if center is None else float(center)
        s = float(1.4826 * np.median(np.abs(np.asarray(cal_values, dtype=float) - c)))
        with open(os.path.join(d, 'model_info.json'), 'w', encoding='utf-8') as f:
            json.dump({'cal_block_center': c, 'cal_block_sigma': s}, f)
    return d
