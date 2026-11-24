import torch
from utils import AverageMeter

# ============================================================
# train.py —— 单 epoch 训练 / 验证循环
#
# 暴露：
#   train_epoch(net, loader, optimizer, criterion, epoch, ...)
#       - 跑一遍 train_loader，每个 batch 做 forward→loss→backward→step
#       - 返回该 epoch 的平均 loss（losses.avg）
#   validate_epoch(net, loader, criterion, ...)
#       - 跑一遍 val_loader，只 forward 不反向
#       - 返回平均 loss（当前训练流程没有调用 validate_epoch：没有选 epoch 用的指标，取最后一轮）
#
# 调用约定（ae_train.py 里）：
#   for epoch in range(1, max_epoch+1):
#       tr_loss = train_epoch(net, train_loader, optimizer, criterion, ...)
#       save_networks(net, out_dir, 'last')
#
# 设计：epoch 循环留在 main_worker，本文件只做单步。便于单独测训练步。
# ============================================================



def train_epoch(
    net, loader, optimizer, criterion,
    epoch=None, use_gpu=False, print_freq=10,
    device=torch.device('cpu'),
):

    net.train()
    losses = AverageMeter()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    for batch_idx, (data, _) in enumerate(loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon = net(data)
        loss = criterion(recon, data)
        loss.backward()
        optimizer.step()

        # 打印训练损失
        losses.update(loss.item(), data.size(0))
        if (batch_idx + 1) % print_freq == 0:
            print("Epoch {} | Batch {}/{}\t train_loss_val:{:.6f} train_loss_avg:{:.6f}".format(
                epoch, batch_idx + 1, len(loader), losses.val, losses.avg))

    return losses.avg



def validate_epoch(net, loader, criterion, use_gpu=False, device=torch.device('cpu')):
    # 验证集的数据
    net.eval()
    losses = AverageMeter()

    with torch.no_grad():
        for data, _ in loader:         # 没有标签的数据
            data = data.to(device)
            recon = net(data)
            loss = criterion(recon, data)
            losses.update(loss.item(), data.size(0))

    return losses.avg
