import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path

# ============================================================
# dataset.py —— IQ 窗数据集 + 三拆分容器
#
# 暴露：
#   IQDataset(data, targets, seq_len=4096)
#       - data: (N, 8192) numpy，前 4096 实部、后 4096 虚部
#       - __getitem__ 返回 (tensor(2, seq_len), label_long)
#   IQData(data_dir, batch_size, num_workers, use_gpu, val_ratio, seq_len, seed)
#       - 读 data_dir 下 train_data.npy / train_label.npy / test_data.npy / test_label.npy
#       - train_data 按 val_ratio 顺序切 train_set / val_set（前 80%/后 20%，全 normal，label=0）
#       - test_set: 整个 test_data（normal + anomaly，用 test_label）
#       - loaders() 返回 (train_loader, val_loader, test_loader)
#         train: shuffle + drop_last；val/test: 不 shuffle
#
#
# 数据契约：
#   - .npy 原始形状 (N, 8192)，前一半实部后一半虚部
#   - label: 0=normal, 1=anomaly
#   - 不做归一化（与 HRRP_OSR 一致）
#   - val 来源 = train_data 的 20%（held-out normal，真正 unseen）
#   - test_normal 仍来自 test_data.npy（真正 unseen）
#
# 切分策略：按前 80%/后 20% 顺序切分 train/val（不随机）。
# ============================================================


class IQDataset(Dataset):
    def __init__(self, data, targets, seq_len=4096):
        self.seq_len = seq_len                              # 获取单个样本的长度
        tensor = torch.from_numpy(data).float()             # (N, 8192)
        real = tensor[:, :seq_len]                          # (N, 4096)
        imag = tensor[:, seq_len:2*seq_len]                 # (N, 4096)
        self.data = torch.stack([real, imag], dim=1)        # (N, 2, 4096)
        self.targets = torch.from_numpy(np.asarray(targets)).long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]



class IQData(object):
    def __init__(self, data_dir, batch_size=64, num_workers=0, use_gpu=False,
                 val_ratio=0.2, seq_len=4096, seed=0):
        
        # 路径数据
        data_dir = Path(data_dir)

        # train_x = np.load(data_dir / 'train_data.npy')
        # train_y = np.load(data_dir / 'train_label.npy')
        # test_x = np.load(data_dir / 'test_data.npy')
        # test_y = np.load(data_dir / 'test_label.npy')

        # # 数据集的划分
        # full_train = IQDataset(train_x, train_y, seq_len=seq_len)
        # n = len(full_train)
        # n_val = int(round(n * val_ratio))
        # n_train = n - n_val
        # train_idx = list(range(n_train))        # 前 80% normal 用于训练
        # val_idx = list(range(n_train, n))       # 后 20% normal 用于验证
        # self.train_set = Subset(full_train, train_idx)
        # self.val_set = Subset(full_train, val_idx)
        # self.test_set = IQDataset(test_x, test_y, seq_len=seq_len)

        train_x = np.load(data_dir / "train_data.npy")
        train_y = np.load(data_dir / "train_label.npy")
        val_x = np.load(data_dir / "val_data.npy")
        val_y = np.load(data_dir / "val_label.npy")
        test_x = np.load(data_dir / "test_data.npy")
        test_y = np.load(data_dir / "test_label.npy")

        self.train_set = IQDataset(train_x, train_y, seq_len=seq_len)
        self.val_set = IQDataset(val_x, val_y, seq_len=seq_len)
        self.test_set = IQDataset(test_x, test_y, seq_len=seq_len)

        # 关键参数设置
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_gpu = use_gpu
        pin_memory = bool(use_gpu)
        
        self.train_loader = DataLoader(
            self.train_set, batch_size=batch_size, shuffle=True,
            drop_last=True, pin_memory=pin_memory, num_workers=num_workers,
        )
        self.val_loader = DataLoader(
            self.val_set, batch_size=batch_size, shuffle=False,
            pin_memory=pin_memory, num_workers=num_workers,
        )
        self.test_loader = DataLoader(
            self.test_set, batch_size=batch_size, shuffle=False,
            pin_memory=pin_memory, num_workers=num_workers,
        )
        
        print(
            f"Train: {len(self.train_set)}  "
            f"Val: {len(self.val_set)}  "
            f"Test: {len(self.test_set)}"
        )

    def loaders(self):
        return self.train_loader, self.val_loader, self.test_loader