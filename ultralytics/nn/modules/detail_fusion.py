import torch
import torch.nn as nn
import torch.nn.functional as F

class DetailInject(nn.Module):
    """
    V4: localization-friendly fusion
    feat' = feat + alpha * F([feat, gate*detail])
    - gate: from feat -> 1ch spatial mask (objectness-like)
    - F: 1x1 residual fusion (zero-init) so it starts as identity
    - alpha: bounded by tanh to protect AP75
    """
    def __init__(self, c1, init_scale=0.0, s_max=0.06):
        super().__init__()
        self.s_max = float(s_max)
        self.alpha = nn.Parameter(torch.tensor(float(init_scale)))

        # spatial gate (very light)
        self.g_dw = nn.Conv2d(c1, c1, 3, 1, 1, groups=c1, bias=False)
        self.g_pw = nn.Conv2d(c1, 1, 1, 1, 0, bias=True)

        # residual fusion (1x1): [feat, gated_detail] -> delta
        self.fuse = nn.Conv2d(2 * c1, c1, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(c1)

        # 初始化：让模块起步≈不改变特征（保护训练稳定 & AP75）
        nn.init.zeros_(self.g_dw.weight)
        nn.init.zeros_(self.g_pw.weight)
        nn.init.zeros_(self.g_pw.bias)
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.bn.weight)
        nn.init.zeros_(self.bn.bias)

    def forward(self, x):
        feat, detail = x
        gate = torch.sigmoid(self.g_pw(self.g_dw(feat)))          # (B,1,H,W)
        d = detail * gate
        delta = self.bn(self.fuse(torch.cat([feat, d], dim=1)))   # (B,C,H,W)

        a = self.s_max * torch.tanh(self.alpha)                   # bounded
        return feat + a * delta


class DetailExtract(nn.Module):
    """
    V4: denoise-friendly detail
    detail = gain * refine(x - blur(x))
    """
    def __init__(self, c1):
        super().__init__()
        self.blur = nn.AvgPool2d(3, 1, 1)
        self.dw = nn.Conv2d(c1, c1, 3, 1, 1, groups=c1, bias=False)
        self.pw = nn.Conv2d(c1, c1, 1, 1, 0, bias=False)
        self.gain = nn.Parameter(torch.tensor(0.0))

        nn.init.zeros_(self.dw.weight)
        nn.init.zeros_(self.pw.weight)

    def forward(self, x):
        hf = x - self.blur(x)
        d = self.pw(self.dw(hf))
        g = torch.tanh(self.gain)
        return g * d
