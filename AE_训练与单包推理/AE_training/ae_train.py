import os
import sys
import argparse
import time
import json
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import set_seed, Logger, mkdir_if_missing, save_networks
from dataset import IQData
from models import Conv1DAE
from loss import build_loss
from train import train_epoch
from threshold import compute_errors, block_errors, select_center_threshold
import shutil
from pre_blocks import build_dataset

# ============================================================
# ae_train.py —— 训练入口（只训练；测试由单独的测试项目 test_stream.py 完成）
#
# 两种用法：
#   1) 命令行：python ae_train.py --data-dir <预处理好的块目录> --out-dir <输出目录>
#   2) 被外部程序调用：from ae_train import train; train(data_root, model_id)
#      （见文件末尾的 train()：它自己做预处理，再调用 main_worker 训练，并整理产物）
#
# main_worker 的流程：
#   argparse → device → Logger → Seed → Dataset → Model
#   1. 构造 criterion / optimizer / scheduler
#   2. for epoch: train_epoch → save 'last'（只用基准训练块训练，取最后一轮）
#   3. compute_errors(cal) → 窗口误差按块平均 → 块误差
#   4. 校准块的块误差取 --block-quantile 分位数（默认 0.99）→ 块阈值
#   5. 存 last.pth / threshold.json / cal_block_err.npy
#
# 关键决策：
#   - 数据划分与块的构造见 pre_blocks.py：窗口 4096、步长 1024、块内保留 61 个窗口、块间不共享采样点
#   - 验证集 = 校准块（只含正常数据），没有选 epoch 用的指标，取最后一轮
#   - 阈值只来自校准块的块误差，不来自训练块
# ============================================================

parser = argparse.ArgumentParser('AE anomaly detection on IQ blocks')

# data                 
parser.add_argument(
    "--data-dir",
    type=str,
    default="/data/hdd1/project/anomaly_detection/dataset/ae_blocks",
    help="数据目录（pre_blocks.py 的 save_root）",
)

parser.add_argument('--seq-len', type=int, default=4096) # IQ 窗长度（须与预处理一致）
parser.add_argument('--batch-size', type=int, default=64) # 批大小
parser.add_argument('--num-workers', type=int, default=0) # DataLoader 进程数（0=主进程内加载）
# model
parser.add_argument('--model', type=str, default='AE') # 模型名
parser.add_argument('--latent-dim', type=int, default=12) # 瓶颈通道数（瓶颈为 latent_dim × seq_len/32）
# training
parser.add_argument('--lr', type=float, default=1e-3) # 学习率
parser.add_argument('--weight-decay', type=float, default=1e-5) # 权重衰减
parser.add_argument('--max-epoch', type=int, default=400) # 训练轮数
parser.add_argument(
    '--scheduler', type=str, default='none',
    choices=['cosine', 'step', 'none'],
) # 学习率调度（AE 不用调度）
parser.add_argument('--stepsize', type=int, default=20) # StepLR 步长
parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'l1', 'complex']) # 重构损失
parser.add_argument('--seed', type=int, default=42) # 随机种子
parser.add_argument('--gpu', type=int, default=0) # GPU 索引
parser.add_argument('--use-cpu', action='store_true', default=False) # 强制 CPU
parser.add_argument('--print-freq', type=int, default=100) # 打印间隔
# threshold
parser.add_argument('--block-quantile', type=float, default=0.99) # 块阈值：校准块的块误差分位数
parser.add_argument('--threshold-margin', type=float, default=0.0) # 阈值余量（σ 倍数）：先加在分位数阈值上，默认 0 = 不启用
parser.add_argument('--threshold-scale', type=float, default=2.0) # 阈值倍数：最终偏差阈值 = (分位数阈值 + 余量×σ) × 倍数
# paths
parser.add_argument('--out-dir', type=str, default='./output/run1') # 输出目录
parser.add_argument('--resume', type=str, default=None) # 恢复检查点

"""
命令行用法下，--out-dir（默认 ./output/run1/）里会有：
├── checkpoints/
│   └── last.pth          # 最后一轮模型参数
├── log.txt               # 训练日志
├── threshold.json        # 块阈值及相关统计
└── cal_block_err.npy     # 校准块的块误差
（train() 的产物目录另见 train() 的说明）
"""

def build_model(options):
    return Conv1DAE(seq_len=options['seq_len'], latent_dim=options['latent_dim'])



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
    logger = Logger(fpath=os.path.join(out_dir, 'log.txt'))
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
        seq_len=options['seq_len'],
        seed=options['seed'],
    )
    train_loader, cal_loader, _ = data.loaders()
    kept = data.kept
    logger.write(
        f'train windows: {len(data.train_set)} | '
        f'cal windows: {len(data.cal_set)} ({len(data.cal_set) // kept} blocks) | '
        + ' | '.join(f'{n} windows: {len(ds)} ({len(ds) // kept} blocks)' for n, ds in data.anomaly_sets.items())
        + f' | kept windows per block: {kept}\n'
    )
    logger.flush()
    # Model
    net = build_model(options).to(device)  # 只是把模型结构定死


    # ===== 训练循环 → 选阈值 → 存模型/阈值/误差缓存 =====
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

    # Train loop
    for epoch in range(1, options['max_epoch'] + 1):
        t0 = time.time() # 训练计时开始
        tr_loss = train_epoch(net, train_loader, optimizer, criterion,
                              epoch=epoch, use_gpu=options['use_gpu'],
                              print_freq=options['print_freq'], device=device)
        if scheduler is not None:
            scheduler.step()
        logger.write(                                    # 统计训练过程中随 epoch 变化的损失
            f'Epoch {epoch}/{options["max_epoch"]} | '
            f'train_loss={tr_loss:.6f} time={time.time()-t0:.1f}s\n'
        )
        logger.flush() # 写入损失和训练时间到日志文件
        save_networks(net, out_dir, name='last')         # 保存最后一轮模型

    # Threshold：居中偏差口径 —— 中心 = 校准块中位数，阈值 = |误差-中心| 的 q 分位数
    cal_err = compute_errors(net, cal_loader, device=device)
    cal_block_err = block_errors(cal_err, kept)
    center, thr, sigma = select_center_threshold(cal_block_err, options['block_quantile'])
    margin = _check_margin(options.get('threshold_margin', 0.0))
    scale = _check_scale(options.get('threshold_scale', 1.0))
    thr = (thr + margin * sigma) * scale             # σ 倍数余量先加，倍数最后乘：thr_sigma 变成「(原值+余量)×倍数」
    logger.write(
        f'center threshold (q{options["block_quantile"]} of |err-center|, '
        f'margin {margin}sigma, scale {scale} over '
        f'{len(cal_block_err)} cal blocks): center={center:.6f} thr={thr:.6f} '
        f'sigma={sigma:.6f} (thr={thr / sigma:.2f}sigma) | '
        f'cal block mean={np.mean(cal_block_err):.6f} max={np.max(cal_block_err):.6f}\n'
    )
    logger.flush() # 从内存写入硬盘

    # Save threshold & errors
    thr_info = {                      # thr 单独为一个词典写入json
        'threshold': float(thr),
        'center': float(center),
        'sigma': float(sigma),
        'quantile': options['block_quantile'],
        'threshold_margin': float(margin),
        'threshold_scale': float(scale),
        'kept': kept,
        'cal_blocks': int(len(cal_block_err)),
        'cal_block_mean': float(np.mean(cal_block_err)),
        'last_train_loss': float(tr_loss),
    }
    with open(os.path.join(out_dir, 'threshold.json'), 'w') as f:
        json.dump(thr_info, f, indent=2)
    np.save(os.path.join(out_dir, 'cal_block_err.npy'), cal_block_err)
    logger.write('Done.\n')
    logger.close()





#MODELS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')   # 模型文件夹的父目录
MODELS_ROOT = "/home/ubuntu/文档/models/ae"

def default_options():
    """命令行参数的全部默认值，供 train() 构造 options"""
    return vars(parser.parse_args([]))


def write_model_info(out_dir, options, meta, created_at, thr_info):
    """把模型编号 / 数据来源 / 时间戳 / 超参 / 量化点与块阈值写进 out_dir/model_info.json"""
    info = {
        'model_id': str(options['model_id']),
        'data_root': options.get('data_root'),
        'created_at': created_at,
        'seq_len': options['seq_len'],
        'stride': meta['stride'],
        'kept': meta['kept'],
        'block_samples': meta['block_samples'],
        'n_train_blocks': len(meta['train_block_ids']),
        'n_cal_blocks': len(meta['cal_block_ids']),
        'max_epoch': options['max_epoch'],
        'seed': options['seed'],
        'latent_dim': options['latent_dim'],
        'block_quantile': thr_info['quantile'],          # 百分数，如 99（与 train() 的 quantile 同单位）
        'cal_block_center': thr_info['center'],          # 居中偏差的中心：校准块误差中位数
        'cal_block_sigma': thr_info['sigma'],            # 稳健 σ（1.4826×MAD），供展示层换算成「几个 σ」
        'cal_block_threshold': thr_info['threshold'],    # 偏离中心的阈值（原始单位，已含余量）
        'threshold_margin': thr_info['threshold_margin'],  # 阈值余量（σ 倍数），compute_threshold 据此重算
        'threshold_scale': thr_info['threshold_scale'],    # 阈值倍数，compute_threshold 据此重算
    }
    with open(os.path.join(out_dir, 'model_info.json'), 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    return info


def normalize_quantile(value):
    """百分数 0~100（如 99）→ (百分数 float, 0~1 小数)。非数字或越界抛 ValueError。"""
    try:
        pct = float(str(value).strip().rstrip('%'))
    except (TypeError, ValueError):
        raise ValueError(f'百分位数不合法：{value!r}')
    if not (0 < pct < 100):
        raise ValueError(f'百分位数必须在 0 到 100 之间（不含）：{value!r}')
    return pct, pct / 100.0


def _check_margin(value):
    """阈值余量（σ 倍数）：必须是 >= 0 的有限数；None 当 0.0，非法值抛 ValueError。"""
    margin = 0.0 if value is None else float(value)
    if not np.isfinite(margin) or margin < 0:
        raise ValueError(f'阈值余量必须是 >= 0 的有限数（σ 倍数）：{value!r}')
    return margin


def _read_margin(model_dir):
    """读 model_info.json 的 threshold_margin；没有这个键（旧模型）→ 0.0。文件读不出也按 0 处理。"""
    info_path = os.path.join(str(model_dir), 'model_info.json')
    if not os.path.isfile(info_path):
        return 0.0
    try:
        with open(info_path, 'r', encoding='utf-8') as f:
            value = json.load(f).get('threshold_margin')
    except (OSError, ValueError):
        return 0.0
    return _check_margin(value)


def _check_scale(value):
    """阈值倍数：必须是 > 0 的有限数；None 当 1.0，非法值抛 ValueError。"""
    scale = 1.0 if value is None else float(value)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f'阈值倍数必须是 > 0 的有限数：{value!r}')
    return scale


def _read_scale(model_dir):
    """读 model_info.json 的 threshold_scale；没有这个键（旧模型）→ 1.0。文件读不出也按 1.0 处理。"""
    info_path = os.path.join(str(model_dir), 'model_info.json')
    if not os.path.isfile(info_path):
        return 1.0
    try:
        with open(info_path, 'r', encoding='utf-8') as f:
            value = json.load(f).get('threshold_scale')
    except (OSError, ValueError):
        return 1.0
    return _check_scale(value)


def _read_center(model_dir, cal_block_err):
    """
    取居中偏差的中心：优先读 model_info.json 的 cal_block_center（旧模型没有这个键），
    缺了就退回「校准块误差的中位数」现算 —— 两者算法相同，结果一致。
    """
    info_path = os.path.join(str(model_dir), 'model_info.json')
    if os.path.isfile(info_path):
        try:
            with open(info_path, 'r', encoding='utf-8') as f:
                center = json.load(f).get('cal_block_center')
            if center is not None:
                return float(center)
        except (OSError, ValueError):
            pass
    return float(np.median(cal_block_err))


def _read_sigma(model_dir, cal_block_err, center):
    """
    取稳健 σ：优先读 model_info.json 的 cal_block_sigma（旧模型没这个键），
    缺了就用 1.4826×MAD 现算 —— 与训练时同一公式，结果一致。
    """
    info_path = os.path.join(str(model_dir), 'model_info.json')
    if os.path.isfile(info_path):
        try:
            with open(info_path, 'r', encoding='utf-8') as f:
                sigma = json.load(f).get('cal_block_sigma')
            if sigma is not None:
                return float(sigma)
        except (OSError, ValueError):
            pass
    return float(1.4826 * np.median(np.abs(cal_block_err - center)))


def compute_threshold(quantile, model_dir, model_id):
    """
    按指定百分位数从已训练模型重算**居中偏差阈值**：只读 cal_block_err.npy，不训练、不写、不改任何文件。

    居中偏差口径下，中心（校准块误差中位数）与百分位数无关，同一个模型固定；
    只有偏差阈值随 quantile 变。调用方用 |块误差 - 中心| > 阈值 判定。
    阈值 = (|校准块误差 - 中心| 的 q 分位数 + 模型记录的余量×σ) × 模型记录的倍数
    （旧模型没有余量/倍数键，按 0 和 1.0 处理）。

    quantile  : 阈值百分位数，**百分数 0~100**（99 = 99%）
    model_dir : 模型文件夹地址
    model_id  : 模型编号，原样返回

    返回 (偏差阈值, 模型编号, σ 倍数)。阈值是原始单位；σ 倍数是 threshold/sigma，
    只供展示层画图用（score_sigma 与它比较），判定用原始单位即可。
    模型文件夹或 cal_block_err.npy 不存在抛 FileNotFoundError。
    中心优先取 model_info.json 里的 cal_block_center，旧模型没有该键时用中位数现算；
    σ 优先取 cal_block_sigma，缺了用 1.4826×MAD 现算；σ=0 时倍数给 inf。
    """
    pct, q = normalize_quantile(quantile)
    cal_path = os.path.join(str(model_dir), 'cal_block_err.npy')
    if not os.path.isfile(cal_path):
        raise FileNotFoundError(f'校准块误差缺失：{cal_path}')
    cal_block_err = np.load(cal_path)
    if len(cal_block_err) == 0:
        raise ValueError(f'校准块误差为空：{cal_path}')
    center = _read_center(model_dir, cal_block_err)
    thr = float(np.quantile(np.abs(cal_block_err - center), q))
    sigma = _read_sigma(model_dir, cal_block_err, center)
    thr = (thr + _read_margin(model_dir) * sigma) * _read_scale(model_dir)   # 与训练同一口径：先加余量，再乘倍数
    thr_sigma = round(thr / sigma,4) if sigma > 0 else float('inf')
    return thr, model_id, thr_sigma


def train(model_id, data_root, byte_order='b', quantile=99):
    """
    对外训练接口：外部程序直接调这个函数，阻塞到训练完成。

    model_id   : 模型编号（字符串），写进文件夹名和 model_info.json
    data_root  : 基准文件夹，扫其中所有 .iq（单个大文件或多个小包都可以）
    byte_order : 'b' 大端（默认）/ 'l' 小端；只决定怎么把字节读成 int16
    quantile   : 阈值百分位数，**百分数 0~100**（99 = 99%，默认 99）

    超参数全部取命令行的默认值（400 轮、lr 1e-3、seed 42、latent_dim 8 等），外部不可改；
    想改超参请改 ae_train.py 里 parser 的默认值。
    产物放在 <MODELS_ROOT>/<model_id>_<时间戳>/（MODELS_ROOT 是 ae_train.py 所在目录下的 models/）。
    训练在临时目录 <MODELS_ROOT>/.<model_id>_<时间戳>.tmp/ 里进行，全部成功后才改名为正式目录。
    训练成功后删掉 blocks/、threshold.json、log.txt，只留：
    checkpoints/last.pth、cal_block_err.npy、model_info.json。训练中途失败或被中断则删除临时目录，不留残留。
    返回: {'model_id', 'model_dir', 'center', 'sigma', 'threshold'}。
          model_dir 是绝对路径（文件夹名可从其最后一级目录取）；
          center 是校准块误差的中位数，threshold 是 |校准块误差 - center| 按 quantile 算出的偏差阈值，
          sigma 是 1.4826×MAD（稳健标准差，只供展示层把分数换算成「几个 σ」）。
          判定：median(|块误差 - center|) > threshold → 异常（偏高、偏低都算）。
    同模型编号、同一秒重复调用：后到的抛 FileExistsError，不写任何文件，先到的不受影响。
    """
    if byte_order not in ('l', 'b'):
        raise ValueError(f"byte_order 只能是 'l' 或 'b'，收到 {byte_order!r}")
    pct, q = normalize_quantile(quantile)

    options = default_options()
    options['block_quantile'] = q                      # main_worker 里的分位数是 0~1 小数
    _check_margin(options.get('threshold_margin', 0.0))   # 早失败：建任何目录之前就校验余量
    _check_scale(options.get('threshold_scale', 1.0))     # 同上，校验倍数
    created_at = time.strftime('%Y%m%d_%H%M%S')
    final_dir = os.path.join(MODELS_ROOT, f'{model_id}_{created_at}')       # 正式目录：只在全部成功后才出现
    out_dir = os.path.join(MODELS_ROOT, f'.{model_id}_{created_at}.tmp')    # 训练期间的临时目录（同一文件系统，rename 才是原子的）
    if os.path.exists(final_dir):
        raise FileExistsError(f'模型目录已存在：{final_dir}')              # 建任何东西之前就拒绝，不进清理分支
    os.makedirs(out_dir)       # 不加 exist_ok：临时目录已存在（同编号同一秒另一次训练在跑）就抛 FileExistsError，且不进下面的清理
    try:
        data_dir = os.path.join(out_dir, 'blocks')
        options.update({'out_dir': out_dir, 'data_dir': data_dir})

        build_dataset(root=str(data_root), save_root=data_dir, byte_order=byte_order)
        main_worker(options)

        with open(os.path.join(data_dir, 'meta.json'), 'r', encoding='utf-8') as f:
            meta = json.load(f)
        with open(os.path.join(out_dir, 'threshold.json'), 'r', encoding='utf-8') as f:
            thr_info = json.load(f)
        thr_info['quantile'] = pct                          # model_info.json 里记百分数，与输入同单位
        write_model_info(out_dir, {**options, 'model_id': str(model_id), 'data_root': str(data_root)},
                         meta, created_at, thr_info)

        shutil.rmtree(data_dir)                                   # 训练成功后清理测试用不到的东西
        for name in ('threshold.json', 'log.txt'):
            os.remove(os.path.join(out_dir, name))
        os.rename(out_dir, final_dir)                             # 最后一步：临时目录变正式目录
    except BaseException:                                         # 含 Ctrl+C；只删本次创建的临时目录
        shutil.rmtree(out_dir, ignore_errors=True)
        raise

    sigma = float(thr_info['sigma'])
    return {
        'model_id': str(model_id),
        'model_dir': final_dir,
        'center': float(thr_info['center']),
        'sigma': sigma,
        'threshold': float(thr_info['threshold']),
        'thr_sigma': round(float(thr_info['threshold']) / sigma,4) if sigma > 0 else float('inf'),
    }


if __name__ == '__main__':
    args = parser.parse_args()
    options = vars(args)
    main_worker(options)
