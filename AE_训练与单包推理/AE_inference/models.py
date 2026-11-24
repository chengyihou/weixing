import torch.nn as nn

# ============================================================
# models.py —— AE 自编码器（1D 全卷积，网络结构取自 AE/ae_1008_best.py）
#
# 暴露：
#   ConvBlock(cin, cout, kernel)
#       - Conv1d(stride=2) → BN → LeakyReLU(0.1)，长度减半
#   DeconvBlock(cin, cout, kernel)
#       - ConvTranspose1d(stride=2, output_padding=1) → BN → LeakyReLU(0.1)，长度翻倍
#   Conv1DAE(seq_len=4096, base_channels=32, latent_dim=8)
#       - encoder: 5 层 ConvBlock，核 7/5/3/3/3，通道 2→c→2c→4c→4c→latent_dim，下采样 32 倍
#       - decoder: 4 层 DeconvBlock + 末层 ConvTranspose1d（无 BN/激活）
#       - forward(x) → recon
#
# 形状约定：
#   输入 x: (B, 2, 4096)，2 通道 = [实部, 虚部]
#   瓶颈 z: (B, latent_dim, seq_len/32)，默认 (B, 8, 128)，共 1024 个值
#   重构 recon: (B, 2, 4096)
#
# 注意：
#   - latent_dim 是瓶颈的"通道数"，不是展平后的向量维度
#   - seq_len 必须能被 32 整除
#   - 没有 encode()，也没有全连接层
# ============================================================



class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, kernel: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, kernel, stride=2, padding=kernel // 2, bias=False),
            nn.BatchNorm1d(cout),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.net(x)



class DeconvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, kernel: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose1d(cin, cout, kernel, stride=2, padding=kernel // 2,
                               output_padding=1, bias=False),
            nn.BatchNorm1d(cout),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.net(x)



class Conv1DAE(nn.Module):
    def __init__(self, seq_len=4096, base_channels=32, latent_dim=8):
        super().__init__()
        if seq_len % 32:
            raise ValueError("seq_len must be divisible by 32")
        c = base_channels
        self.encoder = nn.Sequential(
            ConvBlock(2, c, 7), ConvBlock(c, 2*c, 5), ConvBlock(2*c, 4*c, 3),
            ConvBlock(4*c, 4*c, 3), ConvBlock(4*c, latent_dim, 3),
        )
        self.decoder = nn.Sequential(
            DeconvBlock(latent_dim, 4*c, 3), DeconvBlock(4*c, 4*c, 3),
            DeconvBlock(4*c, 2*c, 3), DeconvBlock(2*c, c, 5),
            nn.ConvTranspose1d(c, 2, 7, stride=2, padding=3, output_padding=1),
        )
        self.seq_len = seq_len

    def forward(self, x):
        return self.decoder(self.encoder(x))
