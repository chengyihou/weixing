import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# ============================================================
# dataset.py —— 块数据集（窗口在块内实时切，逐窗口 RMS 归一化取自 AE/ae_1008_best.py）
#
# 暴露：
#   window_rms_norm(x)
#       - x: (2, seq_len)，除以整个窗口 I/Q 共同的 RMS，下限 1e-8
#   WindowDataset(blocks, label, seq_len=4096, stride=1024, kept=61)
#       - blocks: (n_blocks, block_samples, 2) 原始块（未归一化）
#       - 窗口编号按块顺序排列：第 b 块的第 k 个窗口 = 样本 b*kept + k，
#         采样点 [k*stride, k*stride+seq_len)
#       - __getitem__ 返回 (tensor(2, seq_len), label_long)
#       - 不 shuffle 时误差数组可直接 reshape(-1, kept) 得到 (块, 窗口)
#   IQData(data_dir, batch_size, num_workers, use_gpu, seq_len, seed)
#       - 读 data_dir 下 train/cal 两个 *_blocks.npy、meta.json 里列出的异常 *_blocks.npy
#       - train_loader: shuffle + drop_last；其余不 shuffle
#       - loaders() 返回 (train_loader, cal_loader, anomaly_loaders)
#         anomaly_loaders: {异常名: loader}
#
# 数据契约：
#   - label: 0=normal, 1=anomaly（异常文件的所有块）
#   - 训练只用基准训练块；cal（校准块，仅正常）只用来定阈值；
#     异常文件只在测试阶段使用
#   - 归一化只依赖当前窗口自己，不依赖同一段里的其他数据
# ============================================================


def window_rms_norm(x):
    # x: (2, seq_len) → 除以整个窗口的 RMS（先转 float64 再除，与 AE 的 WindowRMSNorm 一致）
    x = x.astype(np.float64)
    rms = np.sqrt((x ** 2).mean())
    return (x / max(rms, 1e-8)).astype(np.float32)



class WindowDataset(Dataset):
    def __init__(self, blocks, label, seq_len=4096, stride=1024, kept=61):
        self.blocks = blocks                                # (n_blocks, block_samples, 2)
        self.label = int(label)
        self.seq_len = seq_len
        self.stride = stride
        self.kept = kept
        if len(blocks) > 0 and (kept - 1) * stride + seq_len > blocks.shape[1]:
            raise ValueError('块长度不足以放下 kept 个窗口')

    def __len__(self):
        return len(self.blocks) * self.kept

    def __getitem__(self, index):
        b, k = divmod(index, self.kept)
        start = k * self.stride
        seg = self.blocks[b, start:start + self.seq_len]    # (seq_len, 2)
        x = window_rms_norm(seg.T)                          # (2, seq_len)

        return torch.from_numpy(x), self.label



class IQData(object):
    def __init__(self, data_dir, batch_size=64, num_workers=0, use_gpu=False,
                 seq_len=4096, seed=0):

        # 路径数据
        data_dir = Path(data_dir)
        with open(data_dir / 'meta.json', 'r', encoding='utf-8') as f:
            meta = json.load(f)
        if meta['seq_len'] != seq_len:
            raise ValueError(f"--seq-len={seq_len} 与预处理时的 seq_len={meta['seq_len']} 不一致")
        stride, kept = meta['stride'], meta['kept']
        self.kept = kept
        self.anomaly_names = list(meta['anomalies'])

        def make(name, label):
            blocks = np.load(data_dir / f'{name}_blocks.npy')
            return WindowDataset(blocks, label, seq_len=seq_len, stride=stride, kept=kept)

        self.train_set = make('train', 0)
        self.cal_set = make('cal', 0)
        self.anomaly_sets = {name: make(name, 1) for name in self.anomaly_names}

        # 异常文件的不重叠版：窗口步长 = seq_len，每块 block_windows 个窗口，
        # 由相邻 group 个原块首尾拼成（块在预处理时就是连续切出来的，拼接即连续信号）
        self.kept_no = meta['block_windows']
        block_samples = meta['block_samples']
        group = self.kept_no * seq_len // block_samples
        if group * block_samples != self.kept_no * seq_len:
            raise ValueError('block_windows * seq_len 不是 block_samples 的整数倍')

        def make_no(name):
            blocks = np.load(data_dir / f'{name}_blocks.npy')
            n = len(blocks) // group * group                # 尾部不足 group 个块的丢掉
            blocks = blocks[:n].reshape(n // group, group * block_samples, 2)
            return WindowDataset(blocks, 1, seq_len=seq_len, stride=seq_len, kept=self.kept_no)

        self.anomaly_sets_no = {name: make_no(name) for name in self.anomaly_names}

        # 关键参数设置
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_gpu = use_gpu
        pin_memory = bool(use_gpu)

        def eval_loader(dataset):
            return DataLoader(
                dataset, batch_size=batch_size, shuffle=False,
                pin_memory=pin_memory, num_workers=num_workers,
            )

        self.train_loader = DataLoader(
            self.train_set, batch_size=batch_size, shuffle=True,
            drop_last=True, pin_memory=pin_memory, num_workers=num_workers,
        )
        self.cal_loader = eval_loader(self.cal_set)
        self.anomaly_loaders = {name: eval_loader(ds) for name, ds in self.anomaly_sets.items()}
        self.anomaly_loaders_no = {name: eval_loader(ds) for name, ds in self.anomaly_sets_no.items()}

        print(
            f"Train: {len(self.train_set)}  "
            f"Cal: {len(self.cal_set)}  "
            + '  '.join(f'{n}: {len(ds)}' for n, ds in self.anomaly_sets.items())
            + '  (windows)'
        )

    def loaders(self):
        return self.train_loader, self.cal_loader, self.anomaly_loaders
