import os
import sys
import errno
import os.path as osp
import random
import numpy as np
import torch

# ============================================================
# utils.py —— 通用工具层
#
# 暴露：
#   mkdir_if_missing(dir)              - 递归建目录，已存在不报错
#   AverageMeter                       - 滑动累加器（val/avg/sum/count）
#   Logger(fpath)                      - 同时写 stdout 和文件的日志器
#                                      - 它就是一个双通道管子：你写进去的每句话，同时出现在屏幕上 + 落到文件里。
#                                      -  
#   set_seed(seed)                     - random / numpy / torch 全设
#   save_networks(net, dir, name)      - 存到 dir/checkpoints/{name}.pth
#   load_networks(net, dir, name)      - 从 dir/checkpoints/{name}.pth 读回
#
# 被几乎所有其它模块 import。最先实现、依赖最少。
# ============================================================


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise



class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        
        if self.count > 0: # 多了一层防卫
            self.avg = self.sum / self.count



class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w', encoding='utf-8')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        if self.console is not None:
            try:
                self.console.flush()
            except Exception:
                pass
        if self.file is not None:
            self.file.close()



def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def save_networks(networks, result_dir, name=''):
    mkdir_if_missing(osp.join(result_dir, 'checkpoints'))
    weights = networks.state_dict()
    filename = '{}/checkpoints/{}.pth'.format(result_dir, name)
    torch.save(weights, filename)



def load_networks(networks, result_dir, name='', map_location='cpu'):
    filename = '{}/checkpoints/{}.pth'.format(result_dir, name)
    state = torch.load(filename, map_location=map_location, weights_only=True)
    networks.load_state_dict(state)

    return networks
