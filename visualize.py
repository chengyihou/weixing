import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# visualize.py —— 评估可视化
#
# 暴露：
#   plot_error_histogram(train_err, val_err, test_err, test_labels, thr, save_path)
#       - 4 组误差分布叠在一张图：train / val / test_normal / test_anomaly
#       - 竖虚线标 threshold
#       - 期望：train/val/test_normal 重叠，test_anomaly 右移
#   plot_latent_2d(z, labels, save_path)
#       - latent 用 t-SNE 降 2D（若 latent_dim != 2），按 label 染色
#   plot_roc(errors, labels, save_path)
#       - 用误差本身作分数画 ROC（异常为 positive）
#
# matplotlib 用 Agg 后端（无窗口，纯存图）。
# 所有函数自己 makedirs，不依赖调用方。
# ============================================================


def plot_error_histogram(
    train_err, val_err, test_err, test_labels, thr,
    save_path='error_hist.png',
):
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, max(np.percentile(train_err, 99) * 2, thr * 1.5), 60)

    ax.hist(
        train_err, bins=bins, alpha=0.5,
        label='train (normal)', color='steelblue', density=True,
    )
    if val_err is not None and len(val_err) > 0:
        ax.hist(
            val_err, bins=bins, alpha=0.5,
            label='val (unseen normal)', color='orange', density=True,
        )
    if test_err is not None and len(test_err) > 0:
        test_labels = np.asarray(test_labels)
        normal = test_err[test_labels == 0]
        anomaly = test_err[test_labels == 1]
        if len(normal) > 0:
            ax.hist(normal, bins=bins, alpha=0.5, label='test normal', color='green', density=True)
        if len(anomaly) > 0:
            ax.hist(anomaly, bins=bins, alpha=0.5, label='test anomaly', color='red', density=True)
    ax.axvline(thr, color='black', linestyle='--', label=f'thr={thr:.4f}')
    ax.set_xlabel('Reconstruction error (MSE)')
    ax.set_ylabel('Density')
    ax.set_title('Error distribution')
    ax.legend()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)



def plot_latent_2d(z, labels, save_path='latent_2d.png'):
    z = np.asarray(z)
    labels = np.asarray(labels)
    if z.shape[1] != 2:
        try:
            from sklearn.manifold import TSNE
            z2 = TSNE(n_components=2, init='pca', learning_rate='auto').fit_transform(z)
        except Exception:
            return
    else:
        z2 = z
    fig, ax = plt.subplots(figsize=(6, 6))
    for cls, color, name in [(0, 'steelblue', 'normal'), (1, 'red', 'anomaly')]:
        mask = labels == cls
        if mask.sum() > 0:
            ax.scatter(z2[mask, 0], z2[mask, 1], s=5, alpha=0.5, c=color, label=name)
    ax.set_title('Latent space (2D)')
    ax.legend()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)



def plot_roc(errors, labels, save_path='roc.png'):
    from sklearn.metrics import roc_curve
    labels = np.asarray(labels)
    errors = np.asarray(errors)
    if len(np.unique(labels)) != 2:
        return
    fpr, tpr, _ = roc_curve(labels, errors)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2)
    ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title('ROC')
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
