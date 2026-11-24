import os
import json
import numpy as np

# ============================================================
# pre_blocks.py —— 预处理：.iq → 块（内容取自 AE/ae_1008_best.py）
#
# 读 root 下所有 .iq（按字节序参数决定大端 / 小端，I/Q 交错，/32768）；坏文件跳过。
#
# 窗口 / 块（AE 的做法 + 排除块末窗口防泄漏）：
#   - 窗口长 SEQ_LEN=4096，步长 STRIDE=1024，相邻窗口重叠 75%
#   - AE 的一个块 = 连续 64 个窗口；窗口 i 与 j 只要 |i-j| < SEQ_LEN/STRIDE=4 就共享采样点，
#     所以相邻两块在交界处最多有 3 个窗口与邻块共享采样点
#   - 这里把每块最后 3 个窗口排除，块内只保留 KEPT=61 个窗口
#   - 61 个窗口恰好覆盖 60*1024+4096 = 65536 个采样点，块与块首尾相接、互不重叠
#   → 因此一个块 = 原始信号中连续的 65536 个采样点，边界是 65536 的整数倍，
#     防泄漏由结构保证；窗口不预先展开，由 dataset.py 在块内按 STRIDE 实时切
#
# 流程：
#   读取 root 下的所有 .iq → 逐个切块 → 合并 → (N,2) → 每 65536 点一块，尾部不足一块的丢掉
#   基准块按 AE 的方式随机划分（seed）：64% 训练 / 16% 校准 / 其余测试
#   异常 .iq 不划分，所有块都进测试
#   归一化不在这里做（逐窗口 RMS 在 dataset.py 里做）
#
# save_root/
# ├── train_blocks.npy   (n_train, 65536, 2)  float32
# ├── cal_blocks.npy     (n_cal,   65536, 2)
# ├── <异常名>_blocks.npy (n,      65536, 2)   每个异常文件一个
# └── meta.json          窗口/块参数、划分的块编号、异常文件名
# ============================================================

SEQ_LEN = 4096
STRIDE = 1024
BLOCK_WINDOWS = 64                                  # AE 的块大小（含被排除的窗口）
EXCLUDE = SEQ_LEN // STRIDE - 1                     # 块末排除窗口数 = 3
KEPT = BLOCK_WINDOWS - EXCLUDE                      # 块内保留窗口数 = 61
BLOCK_SAMPLES = (KEPT - 1) * STRIDE + SEQ_LEN       # 一个块覆盖的采样点数 = 65536
assert BLOCK_SAMPLES == BLOCK_WINDOWS * STRIDE      # 下一块正好从这里开始，保证块间不重叠


def load_iq(path, byte_order="b"):
    """
    读取 .iq：int16（byte_order 'b'=大端 >i2，'l'=小端 <i2），I/Q 交错（I0,Q0,I1,Q1,...），
    除以 32768 缩放到 [-1,1)。与 AE/iq2csv.py 的读法一致。
    返回：(N,2) float32；无法组成 I/Q 对时抛 ValueError（build_dataset 会跳过这类文件）
    """
    if byte_order == "b":
        dtype = ">i2"
    elif byte_order == "l":
        dtype = "<i2"
    else:
        raise ValueError(f"byte_order 只能是 'l' 或 'b'，收到 {byte_order!r}")

    raw = np.fromfile(path, dtype=dtype)
    if raw.size == 0:
        raise ValueError(f"{path} 为空文件！")
    if raw.size % 2 != 0:
        raise ValueError(f"{path} int16 数量为奇数，无法组成 I/Q 对！")
    return raw.reshape(-1, 2).astype(np.float32) / 32768.0


def to_blocks(iq):
    """
    (N,2) → (n_blocks, 65536, 2)，从头开始每 65536 点一块，尾部不足一块的丢掉
    """
    n = len(iq) // BLOCK_SAMPLES
    return iq[:n * BLOCK_SAMPLES].reshape(n, BLOCK_SAMPLES, 2)


def split_blocks(n_blocks, seed=42, train_ratio=0.75):
    """
    块编号随机打乱后切成 训练 / 校准：前 int(n*train_ratio) 块训练，其余全部校准。
    不再产出测试块（训练样本的测试集已按用户要求取消）。
    返回两个块编号数组
    """
    order = np.random.default_rng(seed).permutation(n_blocks)
    n_train = int(n_blocks * train_ratio)
    train_ids = order[:n_train]
    cal_ids = order[n_train:]
    if len(train_ids) == 0 or len(cal_ids) == 0:
        raise ValueError(
            f"基准只有 {n_blocks} 个块，划分后 训练/校准 = "
            f"{len(train_ids)}/{len(cal_ids)}，存在空集合；数据太短。"
            f"（每块 {BLOCK_SAMPLES} 个采样点，q99 要稳建议校准块数 >= 几十）"
        )
    return train_ids, cal_ids


def save_blocks(save_root, name, blocks):
    os.makedirs(save_root, exist_ok=True)
    np.save(os.path.join(save_root, f"{name}_blocks.npy"), np.ascontiguousarray(blocks, dtype=np.float32))
    print(f"{name}_blocks: {blocks.shape}")


def load_blocks_from_dir(root, byte_order="b"):
    """
    扫 root 下（不递归）**所有** .iq 文件，按文件名排序，逐个切块（尾部不足一块的丢掉）后按顺序合并。

    空文件 / int16 个数为奇数 / 不足一块的文件**跳过并打印警告**，不报错；
    一个块都切不出来时才抛 ValueError。
    返回：(n_blocks, 65536, 2) float32 数组。
    """
    paths = sorted(p for p in os.listdir(root) if p.lower().endswith(".iq"))
    if not paths:
        raise ValueError(f"{root} 里没有任何 .iq 文件")

    parts, skipped = [], 0
    for name in paths:
        try:
            iq = load_iq(os.path.join(root, name), byte_order=byte_order)
        except ValueError as e:
            print(f"[warn] 跳过 {name}：{e}")
            skipped += 1
            continue
        blocks = to_blocks(iq)
        if len(blocks) == 0:
            print(f"[warn] 跳过 {name}：不足一个块（{BLOCK_SAMPLES} 个采样点）")
            skipped += 1
            continue
        parts.append(blocks)

    if not parts:
        raise ValueError(
            f"{root} 里 {len(paths)} 个 .iq 文件全部无效（跳过 {skipped} 个），一个块都切不出来"
        )

    print(f"{root}: {len(paths)} 个 .iq 文件，用上 {len(parts)} 个，跳过 {skipped} 个")
    return np.concatenate(parts)


def build_dataset(
        root,
        save_root,
        byte_order="b",
        seed=42,
):
    """
    root:
        放有若干 .iq 文件的目录；每个文件各自切块后按文件名排序合并，单个大文件时结果与只读它一样
    save_root:
        保存目录
    """
    normal_blocks = load_blocks_from_dir(root, byte_order=byte_order)
    train_ids, cal_ids = split_blocks(len(normal_blocks), seed=seed)

    save_blocks(save_root, "train", normal_blocks[train_ids])
    save_blocks(save_root, "cal", normal_blocks[cal_ids])

    meta = {
        "seq_len": SEQ_LEN,
        "stride": STRIDE,
        "block_windows": BLOCK_WINDOWS,
        "exclude": EXCLUDE,
        "kept": KEPT,
        "block_samples": BLOCK_SAMPLES,
        "seed": seed,
        "byte_order": byte_order,
        "anomalies": [],
        "n_normal_blocks": int(len(normal_blocks)),
        "train_block_ids": train_ids.tolist(),     # 合并后块序列中的块编号，第 b 块 = 采样点 [b*65536,(b+1)*65536)
        "cal_block_ids": cal_ids.tolist(),
    }
    with open(os.path.join(save_root, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print(f"基准块 {len(normal_blocks)}：训练 {len(train_ids)} / 校准 {len(cal_ids)}")
    print(f"每块保留 {KEPT} 个窗口 → 训练窗口 {len(train_ids) * KEPT}，校准窗口 {len(cal_ids) * KEPT}")
    print("=" * 60)


if __name__ == "__main__":
    build_dataset(
        root="/data/hdd1/project/anomaly_detection/dataset",
        save_root="/data/hdd1/project/anomaly_detection/dataset/ae_blocks",
    )
