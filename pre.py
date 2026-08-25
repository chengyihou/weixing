import argparse
import os
from pathlib import Path
import numpy as np


DEFAULT_KEEP_SIGNAL_IDS = [
    3, 4, 8, 10, 11,
    25, 26, 37, 39,
    44, 47, 49, 51, 52,
    65, 66, 70, 71, 72,
]

DEFAULT_ROOT = (
    "/home/ubuntu/[260603]\u897f\u5b89\u91c7\u96c6\u6570\u636e/"
    "UESTC_train_data/train_data"
)
DEFAULT_SAVE_ROOT = (
    "/home/ubuntu/[260603]\u897f\u5b89\u91c7\u96c6\u6570\u636e/"
    "UESTC_train_data/dr_data"
)


def parse_filename(filename):
    """
    Parse one .dat filename.

    Example:
        20250624_103015_18_0.5_2_1_1_train.dat

    Returns:
        dict with tx, sat, signal_id, or None if the file should be skipped.
    """

    name = Path(filename).name
    if name.endswith(".dat"):
        name = name[:-4]
        # name.endswith(".dat")：检查字符串是否以 .dat 结尾。
        # name[:-4]：保留从开头到倒数第 4 个字符之前的内容，即删除最后 4 个字符。即去掉 .dat 后缀。
    parts = name.split("_")            # 拆分
    if len(parts) != 8:                # 确保文件名包括8部分 # 文件名按 _ 拆分后正好有 8 段。
        return None
    signal_id = int(parts[2])          # 信号编号
    uplink = int(parts[4])             # 上下行
    if uplink != 2:
        return None
    tx = int(parts[5])
    return {
        "tx": tx,
        "signal_id": signal_id,
    }


def load_dat(path):
    """
    Extract and concatenate IQ.

    Input:
        One .dat file.

    Returns:
        (N, 2), odd column is I and even column is Q.
    """
    raw = np.fromfile(path, dtype=np.float64)
    if len(raw) % 2 != 0:
        raise ValueError(f"{path} IQ data length is not even.")

    i_data = raw[0::2]
    q_data = raw[1::2]

    iq = np.stack((i_data, q_data), axis=1) # 第一列是 I，第二列是 Q，每一列是一个样本点的 I 或 Q 值。最终返回的 iq 数组形状为 (N, 2)，其中 N 是样本点的数量。

    return iq


def power_normalize(iq):  # 整段(信号) 的 2 范数归一化
    """
    Normalize power so mean(abs(x) ** 2) == 1.

    Input:
        (N, 2) IQ data.

    Returns:
        Real and imaginary parts after normalization, shaped as (N, 2).
    """
    x = iq[:, 0] + 1j * iq[:, 1]
    power = np.mean(np.abs(x) ** 2)
    if power < 1e-12:
        return iq
    x = x / np.sqrt(power)                  # 代码执行的是整段 .dat 数据的平均功率归一化：
    iq = np.stack((x.real, x.imag), axis=1)

    return iq.astype(np.float64)


def split_segment(iq, segment_length=1280): # 拆分1280
    """
    Split IQ by segment_length.
    Input:
        (N, 2) IQ.

    Output:
        (num, segment_length, 2) IQ.
    """
    num = len(iq) // segment_length
    iq = iq[:num * segment_length]
    iq = iq.reshape(num, segment_length, 2)
    return iq


def save_dataset(save_dir, data, label, sat):  # ？？？？？ 问题很大
    """
    Save dataset:
        save_dir/
            data.npy
            label.npy
            sat.npy
    """
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, "data.npy"), data)
    np.save(os.path.join(save_dir, "label.npy"), label)
    np.save(os.path.join(save_dir, "sat.npy"), sat)

    print("=" * 60)
    print(save_dir)
    print("data :", data.shape)
    print("label:", label.shape)
    print("sat  :", sat.shape)
    print("=" * 60)


def _list_dat_files(root): # 递归查找 root 目录及其所有子目录中的 .dat 文件，并返回排序后的完整路径列表。
    dat_files = [] # 完整的文件路径名
    # os.walk(root) 会递归遍历目录，每次返回三个内容：
    # 当前目录路径, 当前目录下的子目录列表, 当前目录下的文件列表
    # 
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(".dat"):
                dat_files.append(os.path.join(dirpath, filename))
    return sorted(dat_files)


def _build_split(files, segment_length):
    data_list = []
    label_list = []

    for path, meta in files:
        iq = load_dat(path)
        iq = power_normalize(iq)
        data = split_segment(iq, segment_length=segment_length)

        if len(data) == 0:
            continue

        data_list.append(data)
        label_list.append(
            np.full((len(data),), meta["signal_id"], dtype=np.int64)
        )
    if not data_list:
        return (
            np.empty((0, segment_length, 2), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
        )
    return (
        np.concatenate(data_list, axis=0),
        np.concatenate(label_list, axis=0),
    )


def build_dataset(               # 主要代码
    root,
    save_root,
    train_ratio=0.6,
    seed=42,
    keep_signal_ids=None,
    segment_length=1280,
):
    """
    Traverse .dat files, parse filenames, keep downlink signals, split by file,
    load IQ, normalize power, segment data, and save Train/Val/Test.
    """
    # 不是“parse_filename 选择后缀”，而是：
    # _list_dat_files：选 .dat
    # parse_filename：按文件名字段筛 uplink / tx / signal_id / sat
    # keep_signal_ids：再筛指定编号

    keep_signal_ids = (
        set(keep_signal_ids) if keep_signal_ids is not None else None    # 把最开始的信号编号列表转换为集合，以便在后续处理中进行快速查找。
    )
    files = []
    for path in _list_dat_files(root): # 这里是提取所有的.dat 文件路径，并对每个文件路径进行了sort。
        meta = parse_filename(path)  # 这里实际上是按照要求提取指定的文件
        if meta is None:
            continue
        if keep_signal_ids is not None and meta["signal_id"] not in keep_signal_ids:
            continue
        files.append((path, meta))

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(files))
    files = [files[i] for i in order]

    n_total = len(files)
    n_train = int(n_total * train_ratio)
    n_val = (n_total - n_train) // 2

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    print("files:", n_total)
    print("train files:", len(train_files))
    print("val files:", len(val_files))
    print("test files:", len(test_files))

    train_data, train_label, train_sat = _build_split(
        train_files,
        segment_length=segment_length,
    )
    val_data, val_label, val_sat = _build_split(
        val_files,
        segment_length=segment_length,
    )
    test_data, test_label, test_sat = _build_split(
        test_files,
        segment_length=segment_length,
    )

    save_dataset(
        os.path.join(save_root, "train_data"),
        train_data,
        train_label,
        train_sat,
    )
    save_dataset(
        os.path.join(save_root, "val_data"),
        val_data,
        val_label,
        val_sat,
    )
    save_dataset(
        os.path.join(save_root, "test_data"),
        test_data,
        test_label,
        test_sat,
    )


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--save-root", default=DEFAULT_SAVE_ROOT)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--segment-length", type=int, default=1280)
    parser.add_argument(
        "--keep-signal-ids",
        type=int,
        nargs="*",
        default=DEFAULT_KEEP_SIGNAL_IDS,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_dataset(
        root=args.root,
        save_root=args.save_root,
        train_ratio=args.train_ratio,
        seed=args.seed,
        keep_signal_ids=args.keep_signal_ids,
        segment_length=args.segment_length,
    )
