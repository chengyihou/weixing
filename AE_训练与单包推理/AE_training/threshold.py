import numpy as np
import torch

# ============================================================
# threshold.py —— 重构误差计算 + 块误差 + 块阈值（取自 AE/ae_1008_best.py）
#
# 暴露：
#   compute_errors(net, loader, device)
#       - 跑一遍 loader（不 shuffle），对每个窗口算重构 MSE
#       - 返回 np.ndarray, shape=(N,)，顺序与 loader 一致
#   block_errors(errors, kept)
#       - 窗口误差 (N,) → reshape(-1, kept) 后每行取平均 → 块误差 (N/kept,)
#       - N 必须是 kept 的整数倍（dataset.py 的窗口就是按块排列的）
#   select_block_threshold(cal_block_err, quantile)
#       - 校准块的块误差取 q 分位数（默认 0.99）作为块阈值（单侧上界，保留备用）
#   select_center_threshold(cal_block_err, quantile)
#       - 返回 (中心, 偏差阈值)：中心 = 中位数，阈值 = |误差-中心| 的 q 分位数
#       - 判定 |block_err - 中心| > 阈值 → 异常，一个阈值同时管偏高和偏低
#
# 阈值只来自校准块（模型没见过的正常数据），不来自训练块：
#   训练误差天然偏低，拿它定阈值会贴死正常分布下界。
# ============================================================


def compute_errors(net, loader, device=torch.device('cpu')):

    net.eval()
    errors = []

    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            recon = net(data)
            err = (recon - data).pow(2).mean(dim=(1, 2))
            errors.append(err.cpu().numpy())

    if not errors:
        return np.array([], dtype=np.float64)

    return np.concatenate(errors, axis=0).astype(np.float64)



def block_errors(errors, kept):

    errors = np.asarray(errors, dtype=np.float64)
    if len(errors) % kept != 0:
        raise ValueError(f'窗口误差数 {len(errors)} 不是 kept={kept} 的整数倍')

    return errors.reshape(-1, kept).mean(axis=1)



def select_block_threshold(cal_block_err, quantile=0.99):

    if len(cal_block_err) == 0:
        raise ValueError('empty cal block errors; cannot select threshold')

    return float(np.quantile(cal_block_err, quantile))


def select_center_threshold(cal_block_err, quantile=0.99):
    """
    居中偏差口径：(中心, 偏差阈值, 稳健 σ)。

      中心 = 校准块误差的中位数（与 quantile 无关，同一个模型固定）
      阈值 = |校准块误差 - 中心| 的 q 分位数（原始单位）
      σ    = 1.4826 × median(|校准块误差 - 中心|)，MAD 估计

    判定：median(|block_err - 中心|) > 阈值 → 异常（偏高、偏低都算）。
    一个阈值同时管两侧；不需要额外余量（位置误差由中位数吸收，宽度由绝对偏差吸收）。
    σ 只供展示层把分数换算成"几个 σ"，不参与判定。
    """
    cal = np.asarray(cal_block_err, dtype=np.float64)
    if len(cal) == 0:
        raise ValueError('empty cal block errors; cannot select center threshold')

    center = float(np.median(cal))
    dev = np.abs(cal - center)
    thr = float(np.quantile(dev, quantile))
    sigma = float(1.4826 * np.median(dev))
    return center, thr, sigma
