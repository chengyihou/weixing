"""
test_stream.py —— 单包实时推理入口（watchdog 监听 → 读包 → 测包 → 回调 → 删包）

对外接口:
  start_test(model_dir, watch_dir, on_result) —— 同步：启动监听，结果走回调
  await start_stream(link, model_dir, watch_dir) —— 异步：启动监听，结果由本模块直接发送
  change_config(model_dir, link=None)         —— 运行中换模型（不重启监听）
  await reload_model(link, model_dir)         —— 异步换模型，成功后发送 change-success
  is_running() / get_config()                          —— 查询当前状态

推理流程:
  1. 每来一个 .iq 文件，读进来、按训练口径切块(65536 点一块,尾部丢弃)
  2. 每块算 61 个重叠窗口(4096 点、步长 1024)的重构误差，块误差 = 61 窗口均值
  3. 包级判定：每块到中心的距离取中位数，通过 on_result({'link': <名>, 'score': <数>}) 送出
  4. 删包，继续等下一个

约束:
  - 启动时读一次模型权重，之后测包只用内存里的模型；不读校准误差、不计算阈值。
  - change_config 只改内存里的模型，不碰 watchdog 监听。
  - 测试过程不往模型文件夹写任何东西。
  - 测完的包删掉，监听目录不留残留。
"""
import asyncio
import json
import math
import os
import re
import threading
import time
from dataclasses import dataclass, replace
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from models import Conv1DAE

# 甲方的通信库（只存在于部署机器上）；本机没有时置 None，测试照常跑，发送时抛明确错误
try:
    from websocket_client import client
except ImportError:
    client = None

# ---------- 常量：与训练项目一致 ----------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64                   # 与训练默认批大小一致
SEQ_LEN = 4096                    # 训练用的窗口长度
STRIDE = 1024                     # 训练用的步长
KEPT = 61                         # 每块的窗口数 (floor((65536 - 4096) / 1024) + 1)
BLOCK_SAMPLES = 65536             # 包内划块：65536 点一块
LATENT_DIM = 8                    # 训练默认潜变量维度
DEFAULT_LINK = 'default'          # 不传 link 时使用的默认流名（单路场景与旧接口完全兼容）


# ========== 读取部分 ==========

@dataclass(frozen=True)
class Bundle:
    """
    不可变：模型 + 大小端 + 居中偏差的中心。
    center 是模型属性（校准块误差的中位数），不是业务阈值；真正的阈值由调用方保存。
    """
    model_dir: str
    net: nn.Module
    byte_order: str = 'b'    # 推流包的大小端：'b' 大端（默认）/ 'l' 小端
    center: Optional[float] = None   # 居中偏差的中心；模型没训练出 center 时为 None
    sigma: Optional[float] = None    # 稳健 σ（1.4826×MAD），只用于把 score 换算成「几个 σ」展示


class Config:
    """运行配置：当前 Bundle + 监听目录 + 回调 + 所属的 link"""
    def __init__(self, state, watch_dir=None, on_result=None, link=DEFAULT_LINK):
        self.state = state
        self.watch_dir = watch_dir
        self.on_result = on_result
        self.link = link


def load_iq(path, byte_order='b'):
    """读 .iq：int16（byte_order 'b'=大端，'l'=小端）I/Q 交错 → 复数 float32"""
    if byte_order == 'b':
        dtype = '>i2'
    elif byte_order == 'l':
        dtype = '<i2'
    else:
        raise ValueError(f"byte_order 只能是 'l' 或 'b'，收到 {byte_order!r}")
    raw = np.fromfile(path, dtype=dtype)
    if len(raw) % 2 != 0:
        raise ValueError(f'int16 个数为奇数，无法组成 I/Q 对：{len(raw)}')
    i, q = raw[::2].astype(np.float32), raw[1::2].astype(np.float32)
    return i + 1j * q


def to_blocks(iq):
    """切块：65536 点一块，尾部不足一块丢弃。返回 (n_blocks, BLOCK_SAMPLES) 的复数数组"""
    n = len(iq) // BLOCK_SAMPLES
    return iq[:n * BLOCK_SAMPLES].reshape(n, BLOCK_SAMPLES)


def _read_center(model_dir):
    """从 model_info.json 读居中偏差的中心和稳健 σ；缺哪个就给 None"""
    info_path = os.path.join(str(model_dir), 'model_info.json')
    if not os.path.isfile(info_path):
        return None, None
    try:
        with open(info_path, 'r', encoding='utf-8') as f:
            info = json.load(f)
    except (OSError, ValueError):
        return None, None
    center = info.get('cal_block_center')
    sigma = info.get('cal_block_sigma')
    return (float(center) if center is not None else None,
            float(sigma) if sigma is not None else None)


def load_bundle(model_dir, byte_order='b'):
    """
    读模型权重 + 居中偏差的中心。缺文件抛 FileNotFoundError，模型切到 eval() 状态。
    测试侧不读 cal_block_err.npy、不计算阈值；只读 model_info.json 里的 center（模型属性）。
    """
    ckpt_path = os.path.join(str(model_dir), 'checkpoints', 'last.pth')
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f'模型权重缺失：{ckpt_path}')
    if not os.path.isdir(str(model_dir)):
        raise FileNotFoundError(f'模型文件夹不存在：{model_dir}')

    net = Conv1DAE(seq_len=SEQ_LEN, latent_dim=LATENT_DIM).to(DEVICE)
    net.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    net.eval()

    center, sigma = _read_center(model_dir)
    return Bundle(model_dir=str(model_dir), net=net, byte_order=byte_order,
                  center=center, sigma=sigma)


def load_config(model_dir, watch_dir, on_result, byte_order='b', link=DEFAULT_LINK):
    """对外读取接口：只读模型权重；byte_order 是推流包的大小端"""
    if not isinstance(link, str) or not link.strip():
        raise ValueError(f'link 必须是非空字符串：{link!r}')
    if byte_order not in ('l', 'b'):
        raise ValueError(f"byte_order 只能是 'l' 或 'b'，收到 {byte_order!r}")
    if not os.path.isdir(str(watch_dir)):
        raise ValueError(f'watch_dir 必须是已存在的目录：{watch_dir}')
    if not callable(on_result):
        raise ValueError('on_result 必须是可调用对象')
    bundle = load_bundle(model_dir, byte_order=byte_order)
    return Config(state=bundle, watch_dir=str(watch_dir), on_result=on_result, link=link)

# ========== 测一个包(内部) ==========

def _skipped(path, reason):
    """坏包的结果"""
    return {
        'packet': os.path.basename(path),
        'blocks': 0,
        'block_err': np.array([]),
        'median_block_err': float('nan'),
        'score': float('nan'),
        'score_sigma': float('nan'),
        'reason': reason,
    }


def block_errs(net, blocks):
    """
    61 窗重叠口径：每块 61 个窗口（4096 / 步长 1024），窗口先除以自身 RMS，
    块误差 = 61 个窗口的平均 MSE。返回 (n_blocks,) 数组。

    窗口不再逐个送模型：一次把 61 个窗口攒成批（BATCH_SIZE 分批），减少 GPU 上的
    逐次同步开销。BatchNorm 在 eval() 下逐样本独立，合批与逐窗结果一致（差 ~1e-7）。
    """
    n = len(blocks)
    errs = np.zeros(n, dtype=np.float32)
    # 61 个窗口的起始下标一次性算好：(KEPT, SEQ_LEN)
    idx = np.arange(KEPT)[:, None] * STRIDE + np.arange(SEQ_LEN)[None, :]
    with torch.no_grad():
        for i, b in enumerate(blocks):
            # (KEPT, 2, SEQ_LEN)：第 0 通道实部、第 1 通道虚部
            seg = np.stack([b.real[idx], b.imag[idx]], axis=1).astype(np.float32)
            rms = np.maximum(np.sqrt((seg ** 2).mean(axis=(1, 2), keepdims=True)), 1e-8)
            x = torch.from_numpy(seg / rms).to(DEVICE)
            parts = []
            for j in range(0, len(x), BATCH_SIZE):
                xb = x[j:j + BATCH_SIZE]
                parts.append((net(xb) - xb).pow(2).mean(dim=(1, 2)))
            errs[i] = float(torch.cat(parts).mean())
    return errs


def process_packet(cfg, path):
    """
    测一个包：读 → 切块 → 算块误差 → 取中位数。
    不删文件（删文件在 handle_packet）。
    """
    b = cfg.state                    # 快照：本包全程只用这一套（模型 / 大小端 / center）
    try:
        iq = load_iq(path, byte_order=b.byte_order)
    except Exception as e:
        return _skipped(path, f'读取失败: {e}')

    if len(iq) == 0:
        return _skipped(path, '空文件')

    blocks = to_blocks(iq)
    if len(blocks) == 0:
        return _skipped(path, '不足一个块')

    block_err = block_errs(b.net, blocks)
    median_block_err = float(np.median(block_err))

    # score：每个块先算到中心的距离，再取中位数（比「中位数的偏差」稳，块间一高一低不会抵消）
    # score_sigma 也在这里用同一个 b.sigma 算：不能留到 handle_packet 里读 cfg.state，
    # 否则 change_config 恰好插在中间时，score 用旧 center、sigma 却用新模型，两者对不上。
    if b.center is None:
        score = float('nan')         # 旧模型没有 center，算不出偏差
        score_sigma = float('nan')
    else:
        score = float(np.median(np.abs(block_err - b.center)))
        score_sigma = score / b.sigma if b.sigma else float('nan')

    return {
        'packet': os.path.basename(path),
        'blocks': len(blocks),
        'block_err': block_err,
        'median_block_err': median_block_err,
        'score': score,
        'score_sigma': score_sigma,
    }


def format_result(r):
    """屏幕上的一行文本（两个数都打：median_block_err 看水平，score 看偏离）"""
    if 'reason' in r:
        return f"[SKIP] {r['packet']}: {r['reason']}"
    return (f"[OK] {r['packet']}: blocks={r['blocks']} "
            f"median_block_err={r['median_block_err']:.6f} score={r['score']:.6f}")


def handle_packet(cfg, path):
    """
    处理一个包：测 → 屏幕打一行 → 回调 → 删文件。
    回调收三项：{'link', 'score', 'score_sigma'}。
      score       —— median(|块误差_i - center|)，原始单位；调用方拿它和 train() 返回的 threshold 比较
      score_sigma —— score / sigma，即「偏离几个 σ」，只供展示层画图，判定以 score 为准
    原始的 median_block_err 只打在屏幕上（排查用），不发给调用方。
    坏包 / 意外错误 / 模型没有 center 时 score 为 None（score_sigma 同理）。
    阈值不在这里——训练时算好并由 train() 返回。
    文件夹里最多一个包，处理完一律删掉等下一个：正常包、坏包、处理中出意外，都删。
    文件已经不在了（被别的程序移走等）→ 直接跳过，返回 None。
    回调、删除出错都只打印警告，不能让 watchdog 线程死掉。
    """
    if not os.path.exists(path):
        return None

    try:
        r = process_packet(cfg, path)
    except Exception as e:
        r = _skipped(path, f'{type(e).__name__}: {e}')

    print(format_result(r))

    try:
        score = r['score']
        score_sigma = r['score_sigma']
        ok = not math.isnan(score)
        if not ok and 'reason' not in r:
            print('[warn] 模型里没有 center，score 无法计算；请用新版 train() 重新训练')
        cfg.on_result({'link': cfg.link, 
                       'type':"ae",                                   # link 放最前面
                       'score': float(score) if ok else None,               # 坏包给 None，避免 JSON 里的裸 NaN
                       'score_sigma': round(float(score_sigma),4)
                       if (ok and not math.isnan(score_sigma)) else None})
    except Exception as e:
        print(f'[warn] on_result 抛异常，已忽略: {type(e).__name__}: {e}')

    try:
        os.remove(path)
    except OSError as e:
        print(f'[warn] 删除失败 {path}: {e}')
    return r


def _wait_until_stable(path, checks=4, interval=0.3, max_wait=30.0):
    """
    等文件写完：大小和 mtime 连续 checks 次都不变才算稳定。
    返回 True=可以读；False=超时或文件已消失（调用方不测、不删）。
    """
    deadline = time.monotonic() + max_wait
    prev, stable = None, 0
    while time.monotonic() < deadline:
        try:
            st = os.stat(path)
        except OSError:
            return False                      # 文件已不在
        cur = (st.st_size, st.st_mtime_ns)
        if cur[0] > 0 and cur == prev:
            stable += 1
            if stable >= checks:
                return True
        else:
            stable = 0
        prev = cur
        time.sleep(interval)
    return False


def _dispatch(cfg, path):
    """watchdog 事件 / 补扫的入口：只认 .iq；先等写入稳定；本路串行处理；异常都吞掉"""
    if not path.lower().endswith('.iq'):
        return
    if not _wait_until_stable(path):          # 锁外等待，不阻塞其他包
        print(f'[warn] {path} 尚未写入完成或已消失，跳过（文件保留）')
        return
    try:
        with _link_lock(cfg.link):             # 每路一把锁：一路处理慢不拖累其他路
            handle_packet(cfg, path)
    except Exception as e:
        print(f'[warn] 处理 {path} 时出错，已忽略: {type(e).__name__}: {e}')


# ========== 改配置 ==========

def change_config(model_dir, link=None):
    """
    就地换指定那一路（link）的模型，监听不中断，其他路不受影响。
    只读一次新的模型权重，装载成功后才一次性替换；失败则原配置原封不动。

    link：要改哪一路。只有一路在跑时可省略；多于一路时必须指定，否则抛 ValueError（避免误改）。
    大小端不在这里改，要换格式请重新 start_test。
    阈值也不在这里改：测试侧不持有阈值，阈值由训练时算出并通过 train() 返回给调用方。
    正在处理的包按旧配置测完，下一个包起用新模型。
    """
    cfg = get_config(link)
    if cfg is None:
        raise RuntimeError('尚未启动监听，无法改配置；请先调用 start_test(...)')

    old = cfg.state
    new = load_bundle(model_dir, byte_order=old.byte_order)

    cfg.state = new
    print(f'[config:{cfg.link}] 模型 {new.model_dir}')
    return new.model_dir


# ========== 监听 ==========

def start_watchdog(cfg, real_dir=None):
    """
    开这一路（cfg.link）的监听：watchdog 的 Observer 在后台线程里盯着 cfg.watch_dir，
    来一个 .iq 包就测一个、删一个。
    重复检查（同一 link / 同一目录）和登记在 `_STARTING` 里串行，两路同时启动时不会都通过检查。
    登记完状态后，把目录里**已有**的 .iq 包补扫一遍（启动前就在目录里的包，watchdog 不会通知）；
    补扫在本函数内同步执行，目录里有积压的包时，测完才返回。
    """
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    class Handler(FileSystemEventHandler):
        def on_created(self, event):
            if not event.is_directory:
                _dispatch(cfg, event.src_path)

        def on_moved(self, event):
            if not event.is_directory:
                _dispatch(cfg, event.dest_path)

    real_dir = real_dir or os.path.realpath(cfg.watch_dir)
    observer = Observer()
    observer.schedule(Handler(), cfg.watch_dir, recursive=False)

    # _STARTING 串行化「检查 → 启动 → 登记」，两路同时启动只有一个能过；
    # 不占用 _STATE_LOCK，observer.start() 期间事件派发不会卡住
    with _STARTING:
        with _STATE_LOCK:
            if cfg.link in _STREAMS:
                raise ValueError(f'link 已在监听：{cfg.link!r}（换模型请用 change_config，重启请先 stop）')
            for other, entry in _STREAMS.items():
                if os.path.realpath(entry['cfg'].watch_dir) == real_dir:
                    raise ValueError(f'watch_dir 已被 link {other!r} 监听：{cfg.watch_dir}（同一目录会抢包）')
        observer.start()              # 只起线程，毫秒级
        with _STATE_LOCK:
            _STREAMS[cfg.link] = {'cfg': cfg, 'observer': observer, 'lock': threading.Lock()}

    # 补扫：监听和状态都已就绪，所以补扫期间新到的包也不会漏；
    # 两条路径都经 _dispatch，由本路的锁串行，同一个包只会被处理一次。
    for name in sorted(os.listdir(cfg.watch_dir)):
        _dispatch(cfg, os.path.join(cfg.watch_dir, name))


# ========== 状态：按 link 分桶，每路 {cfg, observer, lock} ==========

_STREAMS = {}
_STATE_LOCK = threading.Lock()          # 只保护 _STREAMS 这个字典本身
_STARTING = threading.Lock()            # 串行化启动：检查 link/目录 与登记 之间不放开


def _link_lock(link):
    """取这一路的处理锁；这一路已被停掉时给一把临时锁，让在途的包能收尾"""
    with _STATE_LOCK:
        entry = _STREAMS.get(link)
        return entry['lock'] if entry else threading.Lock()


def _resolve(link):
    """link 为 None 时：只有一路则取它，没有返回 None，多于一路抛 ValueError。"""
    if link is not None:
        return link
    if not _STREAMS:
        return None
    if len(_STREAMS) == 1:
        return next(iter(_STREAMS))
    raise ValueError(f'有 {len(_STREAMS)} 路在监听，必须指定 link：{sorted(_STREAMS)}')


def is_running(link=None):
    """link 为 None：是否有任意一路在跑；指定 link：那一路是否在跑"""
    with _STATE_LOCK:
        return bool(_STREAMS) if link is None else link in _STREAMS


def running_links():
    """当前所有在跑的 link"""
    with _STATE_LOCK:
        return sorted(_STREAMS)


def get_config(link=None):
    """指定 link 的配置；未启动返回 None。link 省略时：一路则取它，多于一路抛 ValueError"""
    with _STATE_LOCK:
        key = _resolve(link)
        entry = _STREAMS.get(key) if key is not None else None
        return entry['cfg'] if entry else None


def stop(link):
    """只停指定的那一路，其他路继续。该 link 不存在抛 ValueError。"""
    with _STATE_LOCK:
        entry = _STREAMS.pop(link, None)
    if entry is None:
        raise ValueError(f'link 不存在：{link!r}')
    entry['observer'].stop()
    entry['observer'].join(timeout=5)


def start_test(model_dir, watch_dir, on_result, byte_order='b', link=DEFAULT_LINK):
    """
    对外入口：读取（load_config）→ 监控（start_watchdog）→ 返回 'running'。

    model_dir  : 所选择模型的文件夹（train() 产出的那个）；测试过程只读它，不往里写
    watch_dir  : 推流包所在文件夹
    on_result  : 每个包测完调一次，参数是 {'link', 'score'}；
                 接口不做正常/异常判定，阈值由训练时算出并返回给调用方
    byte_order : 推流包的大小端，'b' 大端（默认）/ 'l' 小端
    link       : 这一路的唯一名字（非空字符串）。同一进程可同时开多路，各自独立监听、独立热更新

    返回 'running'。真正的结果通过 on_result 送出。
    模型文件夹缺文件、百分位数非法、目录不存在 → 直接抛异常，不启动监听。
    同一个 link 已在监听、或另一路已在监听同一个 watch_dir → ValueError（后者否则会抢包）。
    启动后会把 watch_dir 里已有的 .iq 包补测一遍；补扫在本函数内同步执行，目录里有积压的包时，测完才返回。
    """
    cfg = load_config(model_dir, watch_dir, on_result, byte_order=byte_order, link=link)
    real_dir = os.path.realpath(cfg.watch_dir)     # 重复检查在 start_watchdog 的锁内做

    s = cfg.state

    start_watchdog(cfg, real_dir)          # 重复启动在这里抛 ValueError，下面的启动信息不会误打

    print(
        f'[test:{link}] 模型文件夹 : {s.model_dir}\n'
        f'[test:{link}] 监控文件夹 : {cfg.watch_dir}\n'
        f'[test:{link}] 块口径     : {KEPT} 窗 / 步长 {STRIDE}（与训练一致）\n'
        f'[test:{link}] 设备       : {DEVICE}\n'
        f'[test:{link}] 已开始监听\n'
    )
    return 'running'


async def _send(message):
    """发一条消息；没装通信库或发送失败都只打警告，不影响监听。返回是否发出。"""
    if client is None:
        print(f'[warn] 未安装 websocket_client，消息未发送：{message}')
        return False
    try:
        await client.send(message)
        return True
    except Exception as e:
        print(f'[warn] 发送失败，已忽略: {type(e).__name__}: {e}')
        return False


async def start_stream(link, model_dir, watch_dir, byte_order='b'):
    """
    异步入口：启动这一路监听，每测完一个包**由本函数自己发出去**（调用方不用管发送）。

        await test_stream.start_stream('ae_003', model_dir, watch_dir)

    link       : 这一路的唯一名字；同一进程可并发开多路（各自 create_task / gather）
    model_dir  : 所选择模型的文件夹（train() 产出的那个）；测试过程只读它
    watch_dir  : 推流包所在文件夹
    byte_order : 推流包的大小端，'b' 大端（默认）/ 'l' 小端

    每个包发送：
        {"type": "inferring-success", "data": {"link", "score", "score_sigma"}}
    坏包三者中除 link 外都是 None（原始的 median_block_err 只打在屏幕上，不发送）。发送失败只打警告，继续处理下一个包（包已删除，不重试）。
    启动阶段的异常（模型缺失、目录不存在、link 或目录重复）直接抛出。
    本函数不会自行结束；调用方用取消（asyncio 任务取消）或进程退出来停止——取消只停这一路。
    """
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue()

    def on_result(r):
        # 在监听线程里被调用：投递回事件循环
        loop.call_soon_threadsafe(queue.put_nowait, r)

    # start_test 会同步补扫目录里已有的包，放到线程里避免卡住事件循环。
    # shield：取消只打断这边的等待，不打断线程；启动阶段被取消时，finally 里等线程跑完再清理，
    # 否则线程可能在协程这边退出之后才把监听器登记起来，留下没人管的 link（线程本身取消不掉）。
    fut = asyncio.ensure_future(
        asyncio.to_thread(start_test, model_dir, watch_dir, on_result, byte_order, link))

    try:
        await asyncio.shield(fut)
        while True:
            result = await queue.get()
            await _send({'type': 'inferring-success', 'data': result})
    finally:
        if not fut.done():
            await asyncio.wait({fut})             # 等启动线程跑完（可能正在补扫积压的包）
        # 只清理**本次调用自己启动**的那一路：启动失败（link 重复、模型缺失）时
        # fut 带异常，此时同名的那一路属于别的调用，绝不能停
        if not fut.cancelled() and fut.exception() is None and is_running(link):
            await asyncio.to_thread(stop, link)


async def reload_model(link, model_dir):
    """
    异步换指定那一路（link）的模型，成功后发送 {"type": "change-success", "link": link}。

        await test_stream.reload_model('ae_003', '新的模型目录')

    等价于在工作线程里调 change_config(model_dir, link)，不阻塞事件循环；
    失败（link 不存在、模型缺失）直接抛出，**不发送**成功消息，原配置不变。
    返回新的模型目录。
    """
    new_dir = await asyncio.to_thread(change_config, model_dir, link)
    await _send({'type': 'change-success', "data": {'link': link}})
    return new_dir


def _stop_for_tests():
    """停掉所有路、清空状态。仅供测试使用，不对外暴露。"""
    with _STATE_LOCK:
        entries = list(_STREAMS.values())
        _STREAMS.clear()
    for entry in entries:
        entry['observer'].stop()
        entry['observer'].join(timeout=5)
