import torch
import torch.nn as nn

# ============================================================
# loss.py —— 重构损失
#
# 也就是三种损失
# build_loss 选取那个损失
#
# 暴露：
#   ReconLoss            - MSE（默认）
#   L1ReconLoss          - L1
#   ComplexAwareReconLoss- MSE + α·幅度差，对复数信号的幅度敏感
#   build_loss(name)     - 工厂，按 --loss 配置项构造
#
# 调用约定（train.py 里）：
#   criterion = build_loss(options['loss'])
#   loss = criterion(recon, data)
#
# reduction='mean' 时返回标量（训练用）；'none' 时返回逐样本张量
# （阈值计算/评估可能用到）。当前 train 都用 'mean'。
# ============================================================


class ReconLoss(nn.Module):                # MSE 重构损失
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, recon, x):           # x 是原始数据， recon 是重构图像
        return nn.functional.mse_loss(recon, x, reduction=self.reduction)



class L1ReconLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, recon, x):           # x 是原始数据， recon 是重构图像
        return nn.functional.l1_loss(recon, x, reduction=self.reduction)



class ComplexAwareReconLoss(nn.Module):    # MSE + α·幅度差，对复数信号的幅度敏感
    def __init__(self, reduction='mean', alpha=0.5):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, recon, x):
        diff = recon - x
        mse = (diff ** 2).mean(dim=(1, 2)) if self.reduction == 'none' else (diff ** 2).mean() # MSE
        mag_diff = (recon.norm(dim=1) - x.norm(dim=1)).abs().mean(dim=-1) # 幅度感知项：专门计算重建信号与原始信号在幅度（模长）上的差异，并取绝对值。
        if self.reduction == 'none':
            mag = mag_diff

            return mse + self.alpha * mag

        return mse + self.alpha * mag_diff






def build_loss(name='mse', **kwargs):       # 不同的调用
    name = name.lower()                     # 取小写
    if name == 'mse':                       # MSE 重构损失  
        return ReconLoss(**kwargs)
    if name == 'l1':                        # L1 重构损失
        return L1ReconLoss(**kwargs)
    if name in ('complex', 'complex_aware'): # 复数感知重构损失 ?
        return ComplexAwareReconLoss(**kwargs)

    raise ValueError(f'unknown loss: {name}')
