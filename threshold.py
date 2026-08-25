import numpy as np
import torch

# ============================================================
# threshold.py —— 重构误差计算 + 阈值选择
#
# 两个函数，职责严格分层：
#   1. compute_errors(net, loader, device)
#      - 跑一遍 loader，对每个样本算重构 MSE（标量）
#      - 模型相关：要 forward net
#      - 返回 np.ndarray, shape=(N,)，N = loader 样本数
#
#   2. select_threshold(errors, method, percentile)
#      - 从误差分布选一个标量阈值
#      - 纯统计函数，不碰模型、无副作用
#      - 三种 method：
#          percentile     —— 经验分位数（默认 95，约 5% 假阳）
#          mean_plus_std  —— mean+1σ，假设近似正态，对尾部不敏感
#          max            —— 最保守，零假阳，对单点极端值敏感
#
# 调用约定（main.py 里）：
#   train_err = compute_errors(net, train_loader, device)   # baseline
#   val_err   = compute_errors(net, val_loader,   device)   # unseen normal
#   thr       = select_threshold(val_err, ...)              # 阈值只来自 val ，应该没错
#
# 关键设计：阈值只来自 val（unseen normal），不来自 train。
#   train 误差天然偏低，拿它定阈值会贴死正常分布下界，全打成异常。
#   train_err 仅用于：和 val 比对、画直方图、喂给 eval 算泛化标志。
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



def select_threshold(errors, method='percentile', percentile=95.0):

    if len(errors) == 0:
        raise ValueError('empty errors array; cannot select threshold')

    if method == 'percentile':
        return float(np.percentile(errors, percentile))
    if method == 'mean_plus_std':
        return float(errors.mean() + errors.std())
    if method == 'max':
        return float(errors.max())

    raise ValueError(f'unknown threshold method: {method}')
