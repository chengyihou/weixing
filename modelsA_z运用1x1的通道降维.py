import torch
import torch.nn as nn

# ============================================================
# models.py —— AE 自编码器（1D CNN，全卷积张量瓶颈）
#
# 暴露：
#   Encoder(in_channels=2, latent_dim=64)
#       - 4 层 [Conv1d(k=9, s=2, p=4)→BN→LeakyReLU]，下采样 16 倍
#       - 末层 1×1 Conv 降通道到 latent_dim，保留时序 L=256
#       - 输出 z: (B, latent_dim, 256) 特征图（非向量）
#   Decoder(latent_dim=64, out_channels=2)
#       - 首层 1×1 Conv 升维回 128 通道
#       - 4 层 [ConvTranspose1d(k=9, s=2, p=4, op=1)→BN→LeakyReLU]，上采样 16 倍
#       - 末层 Conv1d(k=9, p=4) 输出重构
#   AE(in_channels=2, latent_dim=64)
#       - forward(x) → recon
#       - encode(x) → z    (B, latent_dim, 256)
#
# 形状约定：
#   输入 x: (B, 2, 4096)，2 通道 = [实部, 虚部]
#   latent z: (B, latent_dim, 256)  特征图，保留时序位置
#   重构 recon: (B, 2, 4096)
#
# 关键设计：
#   - 全卷积瓶颈（1×1 Conv），不 flatten，不接大 fc
#   - encoder/decoder 镜像对称（通道 32/64/128/128 对应 128/128/64/32）
#   - 下采样用 stride=2 conv（学参数），不用 MaxPool
#   - 上采样用 ConvTranspose(stride=2)，与 encoder 对称
#   - decoder 末层纯 Conv1d 不接 BN/ReLU（标准 AE 做法）
# ============================================================



class Encoder(nn.Module):
    def __init__(self, in_channels=2, latent_dim=64):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, 32,  kernel_size=9, stride=2, padding=4, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)        # (B,32,2048)

        self.conv2 = nn.Conv1d(32, 64,  kernel_size=9, stride=2, padding=4, bias=False)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)        # (B,64,1024)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=9, stride=2, padding=4, bias=False)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)        # (B,128,512)

        self.conv4 = nn.Conv1d(128, 128, kernel_size=9, stride=2, padding=4, bias=False) # 没有加通道的降维
        self.bn4 = nn.BatchNorm1d(128)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)        # (B,128,256)

        self.latent_conv = nn.Conv1d(128, latent_dim, kernel_size=1, bias=False)  # 1×1 压缩通道，保留时序 L=256

    def forward(self, x):
        h = self.relu1(self.bn1(self.conv1(x)))
        h = self.relu2(self.bn2(self.conv2(h)))
        h = self.relu3(self.bn3(self.conv3(h)))
        h = self.relu4(self.bn4(self.conv4(h)))
        z = self.latent_conv(h)                             # (B, latent_dim, 256) 特征图

        return z



class Decoder(nn.Module):
    def __init__(self, latent_dim=64, out_channels=2):
        super().__init__()

        self.latent_deconv = nn.Conv1d(latent_dim, 128, kernel_size=1, bias=False)  # 1×1 升维回 128 通道

        self.deconv1 = nn.ConvTranspose1d(128, 128, kernel_size=9, stride=2, padding=4, output_padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)        # (B,128,512)

        self.deconv2 = nn.ConvTranspose1d(128, 64,  kernel_size=9, stride=2, padding=4, output_padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)        # (B,64,1024)

        self.deconv3 = nn.ConvTranspose1d(64,  32,  kernel_size=9, stride=2, padding=4, output_padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(32)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)        # (B,32,2048)

        self.deconv4 = nn.ConvTranspose1d(32,  32,  kernel_size=9, stride=2, padding=4, output_padding=1, bias=False)
        self.bn4 = nn.BatchNorm1d(32)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)        # (B,32,4096)

        self.out_conv = nn.Conv1d(32, out_channels, kernel_size=9, padding=4)  # (B,2,4096)

    def forward(self, z):
        h = self.latent_deconv(z)                           # (B,128,256)
        h = self.relu1(self.bn1(self.deconv1(h)))           # (B,128,512)
        h = self.relu2(self.bn2(self.deconv2(h)))           # (B,64,1024)
        h = self.relu3(self.bn3(self.deconv3(h)))           # (B,32,2048)
        h = self.relu4(self.bn4(self.deconv4(h)))           # (B,32,4096)
        out = self.out_conv(h)                              # (B,2,4096)

        return out



class AE(nn.Module):
    def __init__(self, in_channels=2, latent_dim=64):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(latent_dim, in_channels)

    def forward(self, x):     # 两步过程
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x):
        return self.encoder(x)
