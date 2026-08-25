import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

# ============================================================
# eval.py —— 测试评估
#
# 暴露：
#   collect_errors_and_labels(net, loader, device)
#       - 跑一遍 loader，返回 (errors, labels) 两个 (N,) numpy 数组
#   evaluate(net, test_loader, threshold, train_err, val_err, device)
#       - 在 test 集（unseen normal + anomaly）上评估
#       - 返回 dict：AUROC / F1 / 混淆矩阵 / normal/anomaly 误差均值 /
#         unseen_high_flag（val 误差均值是否 > 2× train，泛化失败哨兵）
#
# 评估约定：
#   - 异常为 positive（label=1），正常为 negative（label=0）
#   - AUROC = roc_auc_score(labels, errors)，用误差本身作分数（越大越像异常）
#   - F1 @ 选定阈值（threshold 来自 val，由 threshold.py 算）
#
# unseen_high_flag：
#   若 val（unseen normal）误差均值远高于 train，说明 AE 没泛化到 unseen
#   normal，阈值不可信，方案不成立——得回看模型，不是评估能救的。
# ============================================================


def collect_errors_and_labels(net, loader, device=torch.device('cpu')):

    net.eval()
    all_err, all_lbl = [], []

    with torch.no_grad():
        for data, label in loader:
            data = data.to(device)  # gpu
            recon = net(data)
            err = (recon - data).pow(2).mean(dim=(1, 2)).cpu().numpy()   # MSE
            all_err.append(err)            # 测试样本的重构损失列表
            all_lbl.append(label.numpy())  # 测试样本的标签列表
    if not all_err:
        return np.array([]), np.array([])

    return np.concatenate(all_err), np.concatenate(all_lbl)



def evaluate(
    net, test_loader, threshold,
    train_err=None, val_err=None,
    device=torch.device('cpu'),
):  # 导入网络和配置

    errors, labels = collect_errors_and_labels(net, test_loader, device=device) 
    if len(errors) == 0:
        return {
            'test_err': errors,
            'test_labels': labels,
            'Accuracy': float('nan'),
            'correct_count': 0,
            'total_count': 0,
            'AUROC': float('nan'),
            'F1': float('nan'),
        }

    preds = (errors > threshold).astype(np.int32) # 
    labels_bin = labels.astype(np.int32) # 
    # labels_bin  # 真实标签：0=真正常, 1=真异常
    # preds       # 预测标签：0=判正常, 1=判异常

    correct_count = int(np.sum(preds == labels_bin))
    total_count = len(labels_bin)
    accuracy = float(correct_count / total_count)


    if len(np.unique(labels_bin)) == 2:
        auroc = float(roc_auc_score(labels_bin, errors))
    else:
        auroc = float('nan')
    if len(np.unique(labels_bin)) == 2:
        f1 = float(f1_score(labels_bin, preds, zero_division=0))
    else:
        f1 = float('nan')
    if len(np.unique(labels_bin)) == 2:
        cm = confusion_matrix(labels_bin, preds, labels=[0, 1])
    else:
        cm = None



    results = {                         # 字典保存
        'test_err': errors,
        'test_labels': labels,
        'threshold': float(threshold),
        'Accuracy': accuracy,
        'correct_count': correct_count,
        'total_count': total_count,
        'AUROC': auroc,
        'F1': f1,
        'confusion_matrix': cm,
    }

    if val_err is not None and len(val_err) > 0:
        val_mean = float(np.mean(val_err))
        train_mean = float(np.mean(train_err)) if train_err is not None and len(train_err) > 0 else None
        unseen_high_flag = bool(train_mean is not None and val_mean > 2.0 * train_mean)
        results['val_error_mean'] = val_mean
        results['train_error_mean'] = train_mean
        results['unseen_high_flag'] = unseen_high_flag # 如果 flag 亮了，前面 AUROC / F1 再好看都不能信——因为阈值本身不可信。
        # val/train 误差比值
    normal_err = errors[labels_bin == 0]    # 挑出真正常样本的误差
    anomaly_err = errors[labels_bin == 1]   # 挑出真异常样本的误差
    results['normal_error_mean'] = float(np.mean(normal_err)) if len(normal_err) > 0 else None      # ？
    results['anomaly_error_mean'] = float(np.mean(anomaly_err)) if len(anomaly_err) > 0 else None

    return results



    # 双重的阈值判断方法
    # - train_err：在 train 集（模型见过的那批样本）上算的误差
    # - val_err：在 val 集（从 train 切出来的 20%，模型没见过，但也是正常样本）上算的误差

    # 4. 比值 2 是什么
    # unseen_high_flag = val_mean > 2.0 * train_mean
    # 就是一个经验阈值：
    # val_mean / train_mean > 2  →  flag 亮，方案不成立
    # 意思是：如果 val 集的平均误差是 train 集的 2 倍以上，认定模型在死记硬背（学法 B），不是真泛化。
    # 为什么是 2 倍不是 1.5 或 3 倍？经验值，不是定理。正常情况下 val 误差应该比 train
    # 高一点点（因为没见过），但不会高太多。高 2 倍已经是明显的"记忆而非学习"信号。
