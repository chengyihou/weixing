'''
main_offline.py —— 离线对比脚本：基准文件夹训练，若干异常 .iq 检测（本地验证脚本，非项目一部分）

只调用两边项目现有的函数，不修改它们任何一行：
  训练：AE_training/ae_train.py      train(...) / compute_threshold(...)
  推理：AE_inference/test_stream.py  load_bundle(...) / process_packet(...)

判定口径（居中偏差，与项目一致）：
  center / threshold 由 train() 返回；score = median(|每个块的块误差 - center|)
  score > threshold -> 异常（偏高、偏低都算）

流程：
  1. BASELINE 是一个文件夹（也可以是单个 .iq 或路径列表），扫其中所有 .iq，按文件名排序
  2. 每个基准文件各自取末尾 HOLDOUT_RATIO 的块留出，其余写入临时训练目录
  3. 所有文件的训练段一起交给 train()；所有留出段合起来当“没见过的正常数据”
  4. 留出段 + 每个异常文件算块误差，与 threshold 比较
  5. 结果保存到 offline_results/<时间戳>_<model_id>/ ：report.json、report.txt、block_err.npz
  6. 删掉临时目录（模型文件夹保留）

改参数只需改下面「可改的常量」区。
'''

import json
import os
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, 'AE_inference'), os.path.join(_HERE, 'AE_training')):
    if _p not in sys.path:
        sys.path.insert(0, _p)          # 后插入的排在前面：AE_training 优先

# 两个项目里唯一重名的模块是 models.py，按约定两份必须一致，取哪份都一样
import test_stream                      # noqa: E402  AE_inference
from ae_train import train, compute_threshold   # noqa: E402  AE_training

BLOCK_SAMPLES = test_stream.BLOCK_SAMPLES     # 65536，与两边一致

# train() 内部还要对收到的数据再切训练/校准，块数太少切不动
MIN_TRAIN_BLOCKS = 5


# ==================== 可改的常量 ====================
BASELINE = r'E:\\path\\to\\jizhun_folder'      # 基准文件夹：里面只放基准 .iq（也可写单个文件或路径列表）

ANOMALIES = [                                  # 异常 .iq，手动写地址
    r'E:\\path\\to\\yichangA.iq',
    r'E:\\path\\to\\yichangB.iq',
    r'E:\\path\\to\\yichangC.iq',
    r'E:\\path\\to\\yichangD.iq',
]

MODEL_ID = 'offline_001'        # 模型编号，训练产物文件夹名的一部分
BYTE_ORDER = 'b'                # 'b' 大端（默认）/ 'l' 小端；所有基准和异常文件共用
QUANTILE = 99                   # 阈值百分位数（0–100）：99 = 99% 的正常块偏差小于它
HOLDOUT_RATIO = 0.2             # 每个基准文件各自取末尾这么多比例的块当“没见过的正常数据”
REPEAT = []                     # 可选：换个百分位数重算阈值做对比，如 [95, 90]
RESULT_ROOT = os.path.join(_HERE, 'offline_results')   # 每次运行的结果存到这下面
# ===================================================


# ==================== 读文件 / 切块 ====================

def _list_baseline(baseline):
    '''基准 -> 文件列表：文件夹（扫 .iq，按文件名排序）/ 单个文件 / 路径列表'''
    if isinstance(baseline, (list, tuple)):
        files = [str(p) for p in baseline]
    elif os.path.isdir(baseline):
        files = [os.path.join(baseline, n) for n in sorted(os.listdir(baseline))
                 if n.lower().endswith('.iq')]
    elif os.path.isfile(baseline):
        files = [str(baseline)]
    else:
        raise FileNotFoundError(f'基准不存在：{baseline}')
    if not files:
        raise ValueError(f'基准里没有任何 .iq 文件：{baseline}')
    return files


def _read_raw(path, byte_order):
    '''读 .iq 的原始 int16。空文件或奇数个直接报错。'''
    dtype = '>i2' if byte_order == 'b' else '<i2'
    raw = np.fromfile(path, dtype=dtype)
    if raw.size == 0:
        raise ValueError(f'文件为空：{path}')
    if raw.size % 2:
        raise ValueError(f'int16 个数为奇数，无法组成 I/Q 对：{path}（{raw.size} 个）')
    return raw


def split_baseline(raw, holdout_ratio):
    '''
    单个基准文件按块边界切成 (训练段, 留出段)：末尾 holdout_ratio 的块留出。
    两段保持原字节序，可直接写回 .iq。总块数和最小块数的检查在 run() 里按全部文件合计做。

    返回 (训练段, 留出段, 丢弃的 I/Q 点数, 总块数, 训练块数, 留出块数)。
    '''
    n_samples = raw.size // 2
    n_blocks = n_samples // BLOCK_SAMPLES
    if n_blocks == 0:
        raise ValueError(f'不足一个块（需 {BLOCK_SAMPLES} 个 I/Q 点，实有 {n_samples}）')

    dropped = n_samples - n_blocks * BLOCK_SAMPLES
    n_hold = int(round(n_blocks * holdout_ratio))
    n_train = n_blocks - n_hold

    cut = n_train * BLOCK_SAMPLES * 2           # 单位是 int16 个数
    end = n_blocks * BLOCK_SAMPLES * 2
    return raw[:cut], raw[cut:end], dropped, n_blocks, n_train, n_hold


def _block_err_of(cfg, path, label):
    '''一个 .iq 的块误差数组；坏文件打警告并返回 None（不中断整个脚本）。'''
    r = test_stream.process_packet(cfg, path)
    if 'reason' in r:
        print(f'[warn] {label}: {r["reason"]}')
        return None
    return r['block_err']


def _stats(block_err, center, threshold):
    '''
    与推流侧同一口径：score = median(|块误差 - center|)，score > threshold 即该文件异常。
    另外给出块级的偏高/偏低计数，只作方向诊断，不参与判定。
    '''
    dev = np.abs(block_err - center)
    score = float(np.median(dev))                          # 包/文件级分数，与 test_stream 一致
    above = int(np.sum(block_err - center > threshold))    # 偏高块数
    below = int(np.sum(center - block_err > threshold))    # 偏低块数
    return {
        'blocks': int(len(block_err)),
        'score': score,
        'flag': int(score > threshold),                    # 文件级判定：1=异常
        'above': above,
        'below': below,
        'ratio': (above + below) / len(block_err),         # 块级异常比例（诊断用）
        'score_p95': float(np.quantile(dev, 0.95)),
        'err_mean': float(np.mean(block_err)),
        'err_median': float(np.median(block_err)),
        'err_min': float(np.min(block_err)),
        'err_p95': float(np.quantile(block_err, 0.95)),
    }


def _sg(x, sg):
    '''换算成几个 σ；σ=0 时给 inf，不抛零除'''
    return x / sg if sg else float('inf')


# ==================== 结果保存 ====================

def _jsonable(o):
    '''numpy 类型转原生类型，inf/nan 转 None，保证能 json.dump'''
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def _save_json(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(_jsonable(obj), f, indent=2, ensure_ascii=False)


# ==================== 主流程 ====================

def run(baseline=BASELINE, anomalies=ANOMALIES, model_id=MODEL_ID,
        byte_order=BYTE_ORDER, quantile=QUANTILE, holdout_ratio=HOLDOUT_RATIO,
        repeat=REPEAT):
    '''
    基准训练 -> 留出段 + 异常文件检测 -> 打印并保存 -> 返回汇总字典。

    baseline      : 基准文件夹（扫其中所有 .iq）/ 单个 .iq / 路径列表
    anomalies     : 异常 .iq 路径列表
    model_id      : 模型编号（写进模型文件夹名）
    byte_order    : 'b' 大端 / 'l' 小端，所有文件共用
    quantile      : 阈值百分位数，0–100（决定 threshold）
    holdout_ratio : 每个基准文件各自取末尾多少比例的块留出
    repeat        : 可选，额外用这些百分位数重算阈值并对比

    某个文件坏掉时对应项缺失，其余照常。
    '''
    if byte_order not in ('b', 'l'):
        raise ValueError(f'byte_order 只能是 l 或 b，收到 {byte_order!r}')
    files = _list_baseline(baseline)
    anomalies = list(anomalies)
    if not anomalies:
        raise ValueError('至少要给一个异常文件')
    for p in anomalies:
        if not os.path.isfile(p):
            print(f'[warn] 异常文件不存在，会按坏文件处理：{p}')

    stamp = time.strftime('%Y%m%d_%H%M%S')
    errs = {}
    hold = None
    # 训练段和留出段必须放**不同目录**：train() 会把 data_root 下所有 .iq 都当训练数据
    tmp_dir = tempfile.mkdtemp(prefix='ae_offline_')
    try:
        train_in = os.path.join(tmp_dir, 'train')
        hold_dir = os.path.join(tmp_dir, 'holdout')
        os.makedirs(train_in)
        os.makedirs(hold_dir)

        print(f'基准 : {baseline}  （{len(files)} 个文件）')
        base_files, hold_files = [], []
        for i, p in enumerate(files):
            name = os.path.basename(p)
            try:
                raw = _read_raw(p, byte_order)
                tr, ho, dropped, nb, nt, nh = split_baseline(raw, holdout_ratio)
            except (ValueError, OSError) as e:
                print(f'[warn] 跳过基准文件 {name}: {e}')
                continue
            tag = f'{i:03d}_{name}'                    # 序号放前面，train() 按文件名排序，顺序固定
            if not tag.lower().endswith('.iq'):
                tag += '.iq'
            if tr.size:
                tr.tofile(os.path.join(train_in, tag))
            if ho.size:
                hp = os.path.join(hold_dir, tag)
                ho.tofile(hp)
                hold_files.append((name, hp))
            base_files.append({'file': p, 'blocks': nb, 'train': nt,
                               'holdout': nh, 'dropped': dropped})

        n_train = sum(f['train'] for f in base_files)
        n_hold = sum(f['holdout'] for f in base_files)
        print(f'  可用基准文件 {len(base_files)} 个 | 训练共 {n_train} 块 | 留出共 {n_hold} 块')
        if n_hold < 1:
            raise ValueError(f'留出段共 {n_hold} 块；请增大 HOLDOUT_RATIO 或补充基准数据')
        if n_train < MIN_TRAIN_BLOCKS:
            raise ValueError(f'训练段共 {n_train} 块，至少要 {MIN_TRAIN_BLOCKS} 块'
                             f'（train() 内部还要切成训练/校准）')

        print(f'\n开始训练（model_id={model_id!r}，{n_train} 块）...')
        info = train(model_id, train_in, byte_order=byte_order, quantile=quantile)
        model_dir = info['model_dir']
        center, sigma = float(info['center']), float(info['sigma'])
        threshold = float(info['threshold'])
        thr_sigma = float(info['thr_sigma'])       # 直接用 train() 给的，不自己再除
        print(f'训练完成 : {model_dir}')
        print(f'中心     : {center:.6f}  (校准块误差中位数)')
        print(f'σ        : {sigma:.6f}  (1.4826×MAD)')
        print(f'阈值     : {threshold:.6f}  (= {thr_sigma:.2f}σ，|误差-中心| 的 q{quantile})')

        # 训练一结束就先落盘一份，后面评估出错也不会白跑
        out_dir = os.path.join(RESULT_ROOT, f'{stamp}_{model_id}')
        os.makedirs(out_dir, exist_ok=True)
        report = {'status': 'trained', 'time': stamp, 'out_dir': out_dir,
                  'baseline_path': str(baseline), 'baseline_files': base_files,
                  'anomaly_paths': anomalies, 'byte_order': byte_order,
                  'quantile': quantile, 'holdout_ratio': holdout_ratio,
                  'n_train': n_train, 'n_hold': n_hold,
                  'model_id': model_id, 'model_dir': model_dir,
                  'center': center, 'sigma': sigma, 'threshold': threshold,
                  'thr_sigma': thr_sigma, 'baseline': None, 'anomalies': {},
                  'alternatives': []}
        _save_json(os.path.join(out_dir, 'report.json'), report)

        # 与推流侧同一套：load_bundle 读模型，process_packet 读包算块误差
        cfg = SimpleNamespace(
            state=test_stream.load_bundle(model_dir, byte_order=byte_order))
        print(f'推流侧 center : {cfg.state.center}')

        hold_list = []
        for name, hp in hold_files:
            e = _block_err_of(cfg, hp, f'基准留出段 {name}')
            if e is not None:
                hold_list.append(e)
        hold = np.concatenate(hold_list) if hold_list else None
        if hold is not None:
            report['baseline'] = _stats(hold, center, threshold)

        for i, p in enumerate(anomalies):
            name = os.path.basename(p)
            if name in errs:                          # 不同目录下同名文件，加序号区分
                name = f'{i}_{name}'
            err = _block_err_of(cfg, p, name)
            if err is not None:
                errs[name] = err                      # 块误差只算一次，REPEAT 复用
                report['anomalies'][name] = _stats(err, center, threshold)

        # 可选：换百分位数重算阈值，看检出率怎么变
        for q in repeat:
            alt_thr, _, alt_thr_sigma = compute_threshold(q, model_dir, info['model_id'])
            alt = {'quantile': float(q), 'threshold': float(alt_thr),
                   'thr_sigma': float(alt_thr_sigma), 'baseline': None,
                   'anomalies': {}}
            if hold is not None:
                alt['baseline'] = _stats(hold, center, alt_thr)
            for name, err in errs.items():
                alt['anomalies'][name] = _stats(err, center, alt_thr)
            report['alternatives'].append(alt)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # ---- 打印并保存 ----
    report['status'] = 'done'
    arrays, index = {}, {}
    if hold is not None:
        arrays['holdout'] = hold
        index['holdout'] = '基准留出段（所有基准文件的留出段按顺序拼接）'
    for i, (name, err) in enumerate(errs.items()):
        arrays[f'anomaly_{i}'] = err
        index[f'anomaly_{i}'] = name
    report['block_err_index'] = index

    lines = _report_lines(report)
    print('\n' + '\n'.join(lines))
    with open(os.path.join(out_dir, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    _save_json(os.path.join(out_dir, 'report.json'), report)
    if arrays:
        np.savez(os.path.join(out_dir, 'block_err.npz'), **arrays)
    print(f'\n结果已保存 : {out_dir}')
    return report


def _stat_lines(s, sg, pad):
    n = s['blocks']
    return [
        pad + '块数 %d | score=%.6f (%.2fσ) -> %s' % (
            n, s['score'], _sg(s['score'], sg), '异常' if s['flag'] else '正常'),
        pad + '块级诊断：偏高 %d (%.1f%%)，偏低 %d (%.1f%%)' % (
            s['above'], s['above'] / n * 100, s['below'], s['below'] / n * 100),
        pad + '块误差 均值=%.6f 中位数=%.6f 最小=%.6f p95=%.6f' % (
            s['err_mean'], s['err_median'], s['err_min'], s['err_p95']),
    ]


def _report_lines(report):
    c, thr, sg = report['center'], report['threshold'], report['sigma']
    L = ['=' * 78, '模型 %s' % report['model_dir'],
         '判定：score = median(|块误差 - %.6f|)，score > %.6f (%.2fσ) -> 异常' % (
             c, thr, _sg(thr, sg)),
         '      （与推流侧同一口径；块级的偏高/偏低只作方向诊断，不参与判定）', '']

    bf = report['baseline_files']
    L.append('[基准文件] 共 %d 个，训练 %d 块，留出 %d 块（每个文件各取末尾 %.0f%% 留出）' % (
        len(bf), report['n_train'], report['n_hold'], report['holdout_ratio'] * 100))
    for f in bf[:20]:
        L.append('  %s: 总块 %d | 训练 %d | 留出 %d' % (
            os.path.basename(f['file']), f['blocks'], f['train'], f['holdout']))
    if len(bf) > 20:
        L.append('  ... 其余 %d 个文件见 report.json' % (len(bf) - 20))

    L.append('')
    L.append('[基准留出段] 模型没见过、也未参与定阈值')
    b = report['baseline']
    if b is None:
        L.append('  该段无法读取，跳过')
    else:
        L += _stat_lines(b, sg, '  ')

    L.append('')
    L.append('[异常文件]')
    if not report['anomalies']:
        L.append('  没有可用的异常文件')
    for name, s in report['anomalies'].items():
        L.append('  ' + name)
        L += _stat_lines(s, sg, '    ')

    if report['alternatives']:
        L.append('')
        L.append('[换百分位数对比] 中心与 σ 不变，只有阈值变')
        for alt in report['alternatives']:
            balt = alt['baseline']
            base_txt = ('基准 score=%.6f (%s)' % (
                balt['score'], '异常' if balt['flag'] else '正常')) if balt else '基准 -'
            L.append('  q%g: 阈值 %.6f (%.2fσ) | %s' % (
                alt['quantile'], alt['threshold'], alt['thr_sigma'], base_txt))
            for name, s in alt['anomalies'].items():
                L.append('    %s: score=%.6f (%s)' % (
                    name, s['score'], '异常' if s['flag'] else '正常'))
    L.append('=' * 78)
    return L


def main():
    run()


if __name__ == '__main__':
    main()
