import os
import sys
import argparse
import time
import json
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import set_seed, Logger, mkdir_if_missing, save_networks, load_networks
from dataset import IQData
from models import AE
from loss import build_loss
from train import train_epoch, validate_epoch
from threshold import compute_errors, select_threshold
from eval import evaluate

# ============================================================
# main.py —— 主入口
#
# 编排全流程（训练 / 测试两个对等分支）：
#   公共：argparse → device → Logger → Seed → Dataset → Model
#   训练模式（不传 --eval）：
#     1. 构造 criterion / optimizer / scheduler
#     2. for epoch: train_epoch → validate_epoch → save 'best'/'last'
#     3. 结束后 load 'best'（若有）
#     4. compute_errors(train/val) → select_threshold(val_err) → thr
#     5. 存 best.pth / last.pth / threshold.json / train_err.npy / val_err.npy
#     不跑 test。
#   测试模式（传 --eval）：
#     1. load 'best'（必须存在）
#     2. 读 threshold.json + train_err.npy + val_err.npy
#     3. evaluate(test_loader, thr) → AUROC / F1 / 混淆矩阵 / unseen_high_flag
#     4. 存 test_err.npy / test_labels.npy
#
# 关键决策：
#   - 阈值只来自 val（unseen normal），不来自 train
#   - best 模型按 val_loss 选，不是按 test 指标选（避免 test 泄漏）
#   - 阈值持久化为 threshold.json，测试模式直接读不再重算
#   - --eval 开关对齐参考项目 OSR.py 的分层形式
# ============================================================


parser = argparse.ArgumentParser('AE anomaly detection on IQ windows')

# data                   # 关键改目录
parser.add_argument(
    "--data-dir",
    type=str,
    default=os.path.dirname(os.path.abspath(__file__)),
    help="数据目录",
)

parser.add_argument('--seq-len', type=int, default=4096) # IQ 窗长度
parser.add_argument('--batch-size', type=int, default=32) # 批大小
parser.add_argument('--num-workers', type=int, default=0) # DataLoader 进程数（0=主进程内加载）
parser.add_argument('--val-ratio', type=float, default=0.2) # val 占 train 比例
# model
parser.add_argument('--model', type=str, default='AE') # 模型名
parser.add_argument('--latent-dim', type=int, default=128) # 潜空间维度
# training




parser.add_argument('--lr', type=float, default=1e-3) # 学习率




parser.add_argument('--weight-decay', type=float, default=0.0) # 权重衰减
parser.add_argument('--max-epoch', type=int, default=50) # 训练轮数
parser.add_argument(
    '--scheduler', type=str, default='step',
    choices=['cosine', 'step', 'none'],
) # 学习率调度
parser.add_argument('--stepsize', type=int, default=20) # StepLR 步长 # 每20步调整学习率 ？
parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'l1', 'complex']) # 重构损失
parser.add_argument('--seed', type=int, default=0) # 随机种子
parser.add_argument('--gpu', type=int, default=0) # GPU 索引
parser.add_argument('--use-cpu', action='store_true', default=False) # 强制 CPU
parser.add_argument('--print-freq', type=int, default=10) # 打印间隔
# eval
parser.add_argument(
    '--threshold-method', type=str, default='percentile',
    choices=['percentile', 'mean_plus_std', 'max'],
) # 阈值方法
parser.add_argument('--threshold-percentile', type=float, default=85) # 分位数，用于提取阈值
parser.add_argument('--eval-freq', type=int, default=1) # 评估间隔
parser.add_argument('--eval', action='store_true', default=True) # 测试模式开关：传则只 eval，不传则只 train
# paths
parser.add_argument('--out-dir', type=str, default='./output/run1') # 输出目录
parser.add_argument('--resume', type=str, default=None) # 恢复检查点

"""
output/run1/
├── checkpoints/
│   ├── best.pth          # 验证损失最低的模型参数
│   └── last.pth          # 最后一轮模型参数
├── log.txt               # 训练日志
├── threshold.json        # 阈值及相关统计
├── train_err.npy         # 最佳模型在训练集上的逐样本重构误差
├── val_err.npy           # 最佳模型在验证集上的逐样本重构误差
├── log_eval.txt          # 测试日志、AUROC、F1、混淆矩阵等
├── test_err.npy          # 测试集逐样本重构误差
└── test_labels.npy       # 测试集真实标签
"""

def build_model(options):
    return AE(in_channels=2, latent_dim=options['latent_dim'])



def get_device(options):
    # 默认走 cuda。--use-cpu 强制 CPU；无可用 cuda 时自动回退 CPU。
    if options['use_cpu']:
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device('cuda', options['gpu'])

    print('WARNING: CUDA not available, falling back to CPU.') # cuda 不可用，回退 CPU

    return torch.device('cpu')



def main_worker(options):

    # Device
    device = get_device(options)
    options['device'] = device
    options['use_gpu'] = device.type == 'cuda'

    # Logger
    out_dir = options['out_dir']
    mkdir_if_missing(out_dir)
    log_name = 'log_eval.txt' if options['eval'] else 'log.txt' # 测试模式另起日志，避免覆盖训练 log
    logger = Logger(fpath=os.path.join(out_dir, log_name))
    logger.write(f'Options: {options}\n')
    logger.write(f'Device: {device}\n')
    logger.flush()

    # Seed
    set_seed(options['seed'])

    # Dataset
    data = IQData(
        data_dir=options['data_dir'],
        batch_size=options['batch_size'],
        num_workers=options['num_workers'],
        use_gpu=options['use_gpu'],
        val_ratio=options['val_ratio'],
        seq_len=options['seq_len'],
        seed=options['seed'],
    )
    train_loader, val_loader, test_loader = data.loaders()  # ？
    logger.write(
        f'train windows: {len(data.train_set)} | '
        f'val windows: {len(data.val_set)} | '
        f'test windows: {len(data.test_set)}\n'
    )
    logger.flush()
    # Model
    net = build_model(options).to(device)  # 只是把模型结构定死


    """
    # 测试模式：load 模型 + 读阈值/误差缓存，只跑 test
    """
    if options['eval']:
        # ===== 测试模式：load 模型 + 读阈值/误差缓存，只跑 test =====
        # Load checkpoint & threshold
        ckpt_path = os.path.join(out_dir, 'checkpoints', 'best.pth')
        thr_path = os.path.join(out_dir, 'threshold.json')
        train_err_path = os.path.join(out_dir, 'train_err.npy')
        val_err_path = os.path.join(out_dir, 'val_err.npy')

        if not os.path.exists(ckpt_path):                          # 没跑训练就测试 → 直接报错
            logger.write(f'ERROR: checkpoint not found: {ckpt_path}\n')
            raise FileNotFoundError(f'checkpoint not found: {ckpt_path}')
        if not os.path.exists(thr_path):                           # 阈值没存 → 报错
            logger.write(f'ERROR: threshold.json not found: {thr_path}\n')
            raise FileNotFoundError(f'threshold.json not found: {thr_path}')
        if not (os.path.exists(train_err_path) and os.path.exists(val_err_path)): # 误差缓存缺失 → 报错
            logger.write(f'ERROR: train_err/val_err npy not found in {out_dir}\n')
            raise FileNotFoundError(f'train_err/val_err npy not found in {out_dir}')

        net = load_networks(net, out_dir, name='best', map_location=str(device))
        with open(thr_path, 'r') as f:                             # 阈值 + 统计从训练阶段继承
            thr_info = json.load(f)
        thr = float(thr_info['threshold'])
        train_err = np.load(train_err_path)
        val_err = np.load(val_err_path)
        # 说明 完整的离线评估实验
        logger.write(
            f'loaded best.pth | threshold={thr:.6f} '
            f'(method={thr_info.get("method")}, p={thr_info.get("percentile")}) | '
            f'train_mean={thr_info.get("train_error_mean")} '
            f'val_mean={thr_info.get("val_error_mean")}\n'
        )
        logger.flush()


        # Evaluate
        results = {'train_err': train_err, 'val_err': val_err, 'threshold': thr}
        if test_loader is not None:
            test_results = evaluate(
                net, test_loader, thr,
                train_err=train_err, val_err=val_err, device=device,
            )
            accuracy = test_results['Accuracy']
            correct_count = test_results['correct_count']
            total_count = test_results['total_count']
            normal_mean = test_results.get('normal_error_mean')
            anomaly_mean = test_results.get('anomaly_error_mean')
            normal_mean_text = f'{normal_mean:.6f}' if normal_mean is not None else 'N/A'
            anomaly_mean_text = f'{anomaly_mean:.6f}' if anomaly_mean is not None else 'N/A'
            logger.write(
                'Test metrics:\n'
                f'  Accuracy : {accuracy:.6f} '
                f'({correct_count}/{total_count}, {accuracy * 100:.2f}%)\n'
                f'  AUROC    : {test_results.get("AUROC"):.6f}\n'
                f'  F1       : {test_results.get("F1"):.6f}\n'
                '\nReconstruction error:\n'
                f'  Normal mean  : {normal_mean_text}\n'
                f'  Anomaly mean : {anomaly_mean_text}\n'
                '\nGeneralization:\n'
                f'  unseen_high_flag: {test_results.get("unseen_high_flag")}\n'
            )
            if test_results.get('confusion_matrix') is not None:
                logger.write(
                    '\nConfusion matrix:\n'
                    '  rows=true [normal, anomaly]\n'
                    '  cols=pred [normal, anomaly]\n'
                    f'{test_results["confusion_matrix"]}\n'
                )
            results.update(test_results)

        # Save
        if 'test_err' in results:
            np.save(os.path.join(out_dir, 'test_err.npy'), results['test_err'])
            np.save(os.path.join(out_dir, 'test_labels.npy'), results['test_labels'])
        logger.write('Done.\n')
        logger.close()
        return



    # ===== 训练模式：训练循环 → 选阈值 → 存模型/阈值/误差缓存 =====
    # Loss / Optimizer / Scheduler
    """
        # 训练模式：训练循环 → 选阈值 → 存模型/阈值/误差缓存
        # 流程：
        # 1. 构造 criterion / optimizer / scheduler
        # 2. for epoch: train_epoch → validate_epoch → save 'best'/'last'
        # 3. 结束后 load 'best'（若有）
        # 4. compute_errors(train/val) → select_threshold(val_err) → thr
        # 5. 存 best.pth / last.pth / threshold.json / train
    """
    criterion = build_loss(options['loss'])                           # 损失函数
    optimizer = torch.optim.Adam(                                     # 优化器
        net.parameters(),
        lr=options['lr'],
        weight_decay=options['weight_decay'],
    )
    scheduler = None                               
    if options['scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=options['max_epoch'],
        )
    elif options['scheduler'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=options['stepsize'], gamma=0.5,
        )
    best_val = float('inf')

    # Train loop
    for epoch in range(1, options['max_epoch'] + 1):
        t0 = time.time() # 训练计时开始
        tr_loss = train_epoch(net, train_loader, optimizer, criterion,
                              epoch=epoch, use_gpu=options['use_gpu'],
                              print_freq=options['print_freq'], device=device)
        val_loss = validate_epoch(
            net, val_loader, criterion,
            use_gpu=options['use_gpu'], device=device,
        )
        if scheduler is not None:
            scheduler.step()
        logger.write(                                    # 统计训练过程中随 epoch 变化的损失情况
            f'Epoch {epoch}/{options["max_epoch"]} | '
            f'train_loss={tr_loss:.6f} val_loss={val_loss:.6f} '
            f'time={time.time()-t0:.1f}s\n'
        )
        logger.flush() # 写入损失和训练时间到日志文件
        if val_loss < best_val:
            best_val = val_loss
            save_networks(net, out_dir, name='best')     # 保存验证集上验证的损失最小的模型
        save_networks(net, out_dir, name='last')         # 保存验证集的最后一个模型
    if os.path.exists(os.path.join(out_dir, 'checkpoints', 'best.pth')):
        net = load_networks(net, out_dir, name='best', map_location=str(device))

    # Threshold
    train_err = compute_errors(net, train_loader, device=device)
    val_err = compute_errors(net, val_loader, device=device)
    thr = select_threshold(
        val_err, method=options['threshold_method'], percentile=options['threshold_percentile'],
    )
    logger.write(
        f'threshold ({options["threshold_method"]}, '
        f'p={options["threshold_percentile"]}): {thr:.6f}\n'
    )
    logger.write(
        f'train_err mean={np.mean(train_err):.6f} | '
        f'val_err mean={np.mean(val_err):.6f}\n'
    )
    logger.flush() # 从内存写入硬盘

    # Save threshold & errors
    thr_info = {                      # thr 单独为一个词典写入json
        'threshold': float(thr),
        'method': options['threshold_method'],
        'percentile': options['threshold_percentile'],
        'train_error_mean': float(np.mean(train_err)),
        'val_error_mean': float(np.mean(val_err)),
    }
    with open(os.path.join(out_dir, 'threshold.json'), 'w') as f:
        json.dump(thr_info, f, indent=2)
    np.save(os.path.join(out_dir, 'train_err.npy'), train_err)
    np.save(os.path.join(out_dir, 'val_err.npy'), val_err)
    logger.write('Done. (train mode — 模型/阈值/误差已存盘，未跑 test)\n')
    logger.close()





if __name__ == '__main__':
    args = parser.parse_args()
    options = vars(args)
    main_worker(options)
