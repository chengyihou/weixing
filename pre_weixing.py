import os
import numpy as np

# 遍历所有.dat文件解析文件名，得到 tx、上下行、signal_id。
# 读文件拆分 IQ 组成 (N,2)，整文件功率归一化，按 4096 切片后生成 Train/Val/Test。

"""
遍历 root 下的文件
↓
解析文件名，得到 signal_id / uplink / tx
↓
按 (signal_id, uplink) 组合筛选训练文件和测试文件
↓
训练文件的切片前 80% → 训练，后 20% → 验证
测试文件的全部切片 → 测试
↓
读取每个 .dat
↓
拆分 I/Q → 整文件功率归一化 → 4096 点切片
↓
训练/验证 label=0；测试按 tx 是否在训练中出现赋 label=0/1
↓
保存 train_data / val_data / test_data

save_root/
├── train_data/
│   ├── data.npy
│   └── label.npy
├── val_data/
│   ├── data.npy
│   └── label.npy
└── test_data/
    ├── data.npy
    └── label.npy

"""

def parse_filename(filename):
    """
    解析文件名

    输入：
        20250624_103015_18_0.5_2_1_1_train.dat
        日期-时间-信号序号-带宽-上下行信号-站1234-采集设备20个一组，前20个一个转发器，后20个一个转发器，40个点换站

    返回：
        tx：发射机编号
        uplink：上下行编号
        signal_id：信号序号
    """
    # 按 "_" 分割文件名
    parts = filename.split("_")
    # 确保读取的文件包括8部分
    if len(parts) != 8:
        return None
    signal_id = int(parts[2])     # 第3部分：信号序号
    uplink = int(parts[4])        # 第5部分：上下行（上行1，下行2）
    tx = int(parts[5])            # 第6部分：tx标签
    return {
        "tx": tx,
        "uplink": uplink,
        "signal_id": signal_id,
    }


def load_dat(path):
    """
    提取并拼接IQ
    输入：读取一个.dat文件
    返回：(N,2)奇数列为I，偶数列为Q
    """
    # 读取所有float64数据
    raw = np.fromfile(path, dtype=np.float64)
    if len(raw) % 2 != 0:
        raise ValueError(f"{path} IQ数据长度不是偶数！ ")
    # 奇数位置(I)，从下标0开始，每隔1个取一个元素
    I = raw[0::2]
    # 偶数位置(Q)，从下标1开始，每隔1个取一个元素
    Q = raw[1::2]
    # 拼成(N,2)
    iq = np.stack((I, Q), axis=1)
    return iq


def power_normalize(iq):
    """
    功率归一化 mean(|x|²)=1
    输入：拼接成(N,2)的data
    返回：转复数，归一化后重新拆为IQ(N,2)
    """
    # 组成复数
    x = iq[:, 0] + 1j * iq[:, 1]
    # 平均功率
    power = np.mean(np.abs(x) ** 2)
    # 防止除0
    if power < 1e-12:
        return iq
    # 归一化
    x = x / np.sqrt(power)
    # 再拆回IQ
    iq = np.stack((x.real, x.imag), axis=1)
    return iq.astype(np.float64)


def split_segment(iq, segment_length=4096):
    """
    按4096为长度切分
    输入：(N,2)IQ
    输出：(num,4096,2)IQ
    """
    # 能切多少段
    num = len(iq) // segment_length
    # 去掉剩余不足一个完整片段的数据 ＜（＾－＾）＞  √
    iq = iq[:num * segment_length]
    # reshape
    iq = iq.reshape(num, segment_length, 2)
    return iq


def save_dataset(save_dir, data, label):
    """
    保存数据集

    save_dir/
        data.npy
        label.npy
    """

    # 如果目录不存在则创建
    os.makedirs(save_dir, exist_ok=True)
    data = np.concatenate((data[:, :, 0], data[:, :, 1]), axis=1)
    np.save(os.path.join(save_dir, "data.npy"), data)
    np.save(os.path.join(save_dir, "label.npy"), label)

    print("=" * 60)
    print(save_dir)
    print("data :", data.shape)
    print("label:", label.shape)
    print("=" * 60)


def build_dataset(
        root,
        save_root,
        train_selectors,
        test_selectors,
        train_ratio=0.8,
        segment_length=4096,
):
    """
    root:
        原始dat文件目录
    save_root:
        保存目录
    最终生成
    save_root/
        train_data/
        val_data/
        test_data/
    """
    train_selectors = set(train_selectors)
    test_selectors = set(test_selectors)


    # 训练集和验证集
    train_files = []
    test_files = []
    for file in os.listdir(root):
        if not file.endswith(".dat"):
            continue
        info = parse_filename(file)
        if info is None:
            continue
        selector = (info["signal_id"], info["uplink"])
        if selector in train_selectors:
            train_files.append((file, info))  # 路径和信息
        elif selector in test_selectors:
            test_files.append((file, info))   # 路径和信息

    def empty_data():
        return np.empty((0, segment_length, 2), dtype=np.float32)

    def combine(parts):
        if not parts:
            return empty_data()
        return np.concatenate(parts, axis=0).astype(np.float32, copy=False)

    train_parts = []
    val_parts = []
    for file, _ in train_files:
        path = os.path.join(root, file)                               # 第一步，读取文件
        iq = power_normalize(load_dat(path))                          # 第二步，功率归一化
        segments = split_segment(iq, segment_length=segment_length)   # 第三步，按4096切片


        cut = int(len(segments) * train_ratio)                        # 第四步，按比例切分训练集和验证集
        train_parts.append(segments[:cut])                            # 第五步，训练集和验证集
        val_parts.append(segments[cut:])

        cut = int(len(segments) * train_ratio)
        idx = np.random.default_rng(0).permutation(len(segments))
        train_parts.append(segments[idx[:cut]])
        val_parts.append(segments[idx[cut:]])

    train_data = combine(train_parts)
    val_data = combine(val_parts)
    train_label = np.zeros(len(train_data), dtype=np.uint8)           # 符合训练集和验证集的label都是0
    val_label = np.zeros(len(val_data), dtype=np.uint8)


    # 测试集
    training_tx_set = {info["tx"] for _, info in train_files}         # 训练集的tx集合，用于测试集的label判断
    test_parts = []
    test_label_parts = []
    for file, info in test_files:
        path = os.path.join(root, file)
        iq = power_normalize(load_dat(path))
        segments = split_segment(iq, segment_length=segment_length)
        test_parts.append(segments)
        label = 0 if info["tx"] in training_tx_set else 1
        test_label_parts.append(np.full(len(segments), label, dtype=np.uint8))

    test_data = combine(test_parts)
    if test_label_parts:
        test_label = np.concatenate(test_label_parts, axis=0)
    else:
        test_label = np.empty((0,), dtype=np.uint8)

    print("Train files:", len(train_files), "segments:", len(train_data))
    print("Val segments:", len(val_data))
    print("Test files:", len(test_files), "segments:", len(test_data))

    # 保存
    save_dataset(
        os.path.join(save_root, "train_data"),
        train_data,
        train_label,
    )

    save_dataset(
        os.path.join(save_root, "val_data"),
        val_data,
        val_label,
    )

    save_dataset(
        os.path.join(save_root, "test_data"),
        test_data,
        test_label,
    )


if __name__ == "__main__":
    # 按 (signal_id, uplink) 手动指定训练文件和测试文件。
    train_selectors = [
        (1, 1),
        (2, 1),
        (3, 0),
        (4, 1),
        (5, 0),
        (6, 1),
        (7, 0),
        (8, 1),
        (9, 0),
        (10, 1),

    ]

    test_selectors = [
        (1, 0),
        (2, 1),
        (3, 0),
        (4, 1),
        (5, 0),
        (6, 1),
        (7, 0),
        (8, 1),
        (9, 0),
        (10, 1),

    ]

    build_dataset(
        root="/home/ubuntu/[260603]西安采集数据/UESTC_train_data/train_data",
        save_root="/home/ubuntu/[260603]西安采集数据/UESTC_train_data/dr_data",
        train_selectors=train_selectors,
        test_selectors=test_selectors,
    )
