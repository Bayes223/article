import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalStdPool(nn.Module):
    def forward(self, x):
        return x.flatten(2).std(dim=2)  # (B,C)


class UCTToken(nn.Module):
    """
    UCTToken (名字不变) - 稳健多尺度条件token
    - 支持单输入 feature 或多输入 [P3,P4,P5]
    - 仅使用 mean(+std) 的稳健统计（避免你之前 token 噪声过大）
    Args (YAML): [d, hidden, use_std, fuse_attn]
    """
    def __init__(self, c1, d=128, hidden=256, use_std=True, fuse_attn=True):
        super().__init__()
        self.use_std = use_std
        self.fuse_attn = fuse_attn

        # c1 can be int or list[int]
        if isinstance(c1, (list, tuple)):
            self.c_list = list(c1)
        else:
            self.c_list = [int(c1)]

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gsp = GlobalStdPool()

        self.norm = nn.ModuleList()
        self.mlp = nn.ModuleList()
        for ci in self.c_list:
            in_dim = ci * (1 + int(use_std))
            self.norm.append(nn.LayerNorm(in_dim))
            self.mlp.append(nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.SiLU(inplace=True),
                nn.Linear(hidden, d),
            ))

        # 多尺度融合权重（可选）
        if fuse_attn and len(self.c_list) > 1:
            self.fuse = nn.Sequential(
                nn.Linear(d * len(self.c_list), d),
                nn.SiLU(inplace=True),
                nn.Linear(d, len(self.c_list))
            )
        else:
            self.fuse = None

    def _one(self, x, i):
        b, c, _, _ = x.shape
        mu = self.gap(x).view(b, c)
        feats = [mu]
        if self.use_std:
            feats.append(self.gsp(x))
        v = torch.cat(feats, dim=1)
        v = self.norm[i](v)
        return self.mlp[i](v)  # (B,d)

    def forward(self, x):
        xs = list(x) if isinstance(x, (list, tuple)) else [x]
        assert len(xs) == len(self.c_list), f"UCTToken expects {len(self.c_list)} inputs, got {len(xs)}"

        toks = [self._one(xs[i], i) for i in range(len(xs))]
        if len(toks) == 1:
            return toks[0]

        if self.fuse is None:
            return sum(toks) / len(toks)

        cat = torch.cat(toks, dim=1)             # (B, d*K)
        w = torch.softmax(self.fuse(cat), 1)     # (B, K)
        tok = torch.stack(toks, dim=1)           # (B, K, d)
        return (tok * w.unsqueeze(-1)).sum(1)    # (B, d)


class UCTFiLM(nn.Module):
    """
    UCTFiLM (名字不变) - 稳健条件调制（弱启动 + 有界小扰动 + 空间门控）
    关键：避免你现在出现的 mAP75 大掉、n/s/m 变差的问题。

    y = x + alpha * mask * (x * gamma + beta(optional))

    - alpha: 用 sigmoid(logit)*alpha_max 控制“调制强度”，默认初始非常小(≈0)，训练稳定但可学。
    - gamma/beta: tanh 限幅，避免过强扰动导致定位发散。
    - mask: 由特征自身预测的空间门控（按group输出），默认bias为负 -> 初始 mask 很小，不会一开始乱调制。

    Args (兼容你之前的 YAML 写法):
      [groups, d, k_expert(忽略), clamp, alpha_init, use_spatial, alpha_max, use_beta]

    注意：
    - k_expert 参数为了兼容旧 YAML，直接忽略即可。
    - alpha_init 解释为 “初始alpha比例” in [0,1]，旧yaml填0.0刚好等价“初始不调制”。
    """
    def __init__(self,
                 c1,
                 groups=16,
                 d=128,
                 k_expert=0,        # 兼容旧yaml，忽略
                 clamp=0.08,
                 alpha_init=0.0,    # 初始alpha比例(0~1)
                 use_spatial=True,
                 alpha_max=0.12,
                 use_beta=False):
        super().__init__()
        assert c1 % groups == 0, f"channels {c1} must be divisible by groups {groups}"
        self.c1 = c1
        self.groups = groups
        self.clamp = float(clamp)
        self.alpha_max = float(alpha_max)
        self.use_spatial = bool(use_spatial)
        self.use_beta = bool(use_beta)

        out_dim = 2 * groups if self.use_beta else groups
        self.proj = nn.Linear(d, out_dim)
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.proj.bias)

        # alpha = sigmoid(logit)*alpha_max
        # alpha_init 是比例(0~1)，0表示几乎不调制（稳定），>0 可更快起效
        alpha_init = float(alpha_init)
        if alpha_init <= 0.0:
            logit = -10.0  # sigmoid≈0
        elif alpha_init >= 1.0:
            logit = 10.0
        else:
            logit = math.log(alpha_init / (1.0 - alpha_init))
        self.alpha_logit = nn.Parameter(torch.tensor(logit, dtype=torch.float32))

        if self.use_spatial:
            # mask: (B,g,H,W)，bias 负值 => 初始 mask 小（稳定）
            self.sgate = nn.Conv2d(c1, groups, 1, 1, 0, bias=True)
            nn.init.zeros_(self.sgate.weight)
            nn.init.constant_(self.sgate.bias, -2.0)  # sigmoid≈0.12
        else:
            self.sgate = None

    def forward(self, x):
        feat, token = x if isinstance(x, (list, tuple)) else (x, None)
        if token is None:
            return feat

        b, c, h, w = feat.shape
        g = self.groups

        gb = self.proj(token)  # (B, g) or (B, 2g)
        if self.use_beta:
            gamma, beta = gb[:, :g], gb[:, g:]
            beta = torch.tanh(beta).view(b, g, 1, 1) * self.clamp
        else:
            gamma = gb
            beta = None

        gamma = torch.tanh(gamma).view(b, g, 1, 1) * self.clamp

        alpha = torch.sigmoid(self.alpha_logit) * self.alpha_max  # scalar

        if self.sgate is None:
            mask = 1.0
        else:
            mask = torch.sigmoid(self.sgate(feat)).view(b, g, 1, h, w)  # (B,g,1,H,W)

        feat_g = feat.view(b, g, c // g, h, w)
        mod = feat_g * gamma.unsqueeze(2)
        if beta is not None:
            mod = mod + beta.unsqueeze(2)

        y = feat_g + alpha * mask * mod
        return y.view_as(feat)
