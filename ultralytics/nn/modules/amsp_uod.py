
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class BrokenBlock(nn.Module):
    """ 比起Tensorflow2.x 没想到还是Torch更反人类 """
    def __init__(self,  dim_len, dim=1, dim_shape=4, group=1, *args, **kwargs):
        super(BrokenBlock, self).__init__(*args, **kwargs)
        self.dim = dim
        self.group = group
        self.dim_len = dim_len
        self.view = list(np.ones(dim_shape, int))

    def getShuffle(self, shape):
        # 生成随机索引
        self.view[self.dim] = shape[self.dim]
        perm = torch.randperm(self.dim_len//self.group).unsqueeze(1)
        indices = torch.cat([perm * self.group + _ for _ in range(self.group)], dim=1)
        indices = indices.view(self.view).expand(shape)
        return indices

    def forward(self, x):
        if self.training:
            return torch.gather(x, self.dim, self.getShuffle(x.shape).to(x.device))
        else:
            return x
        
        
class SC(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, length=4, *args, **kwargs):
        super(SC, self).__init__(*args, **kwargs)
        self.length = length
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = nn.Conv2d(c_//length, c_//length, 5, 1, 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c_)
        self.act = nn.SiLU()
        self.brk = BrokenBlock(c_, group=c_//length)

    def forward(self, x):
        x = self.cv1(x)
        if self.training:
            y = self.brk(x)
        else:
            y = x
        y = y.chunk(self.length, 1)
        out = [self.cv2(_) for _ in y]
        out = self.act(self.bn(torch.cat(out, 1)))
        return torch.cat([x, out], 1)
    
class PW(nn.Module):
    def __init__(self, in_channels, out_channels, ratio = 32):
        super(PW, self).__init__()
        self.csp = Conv(in_channels, out_channels)
        self.pool_h, self.pool_w = PoolBlock((1, None)), PoolBlock((None, 1))
        c = max(8, in_channels//ratio)
        self.cv1 = nn.Conv2d(in_channels=out_channels, out_channels=c, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(c)
        self.act1 = nn.Hardswish()

        self.cv2 = nn.Conv2d(in_channels=c, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.cv3 = nn.Conv2d(in_channels=c, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.act2 = nn.Sigmoid()

        self.brk = BrokenBlock(out_channels, group=2)

    def forward(self, x):
        x = self.csp(x)
        b,c,h,w = x.shape
        x_h, x_w = self.pool_h(x), self.pool_w(x).permute(0, 1, 3, 2)

        x_tmp = torch.cat([x_h, x_w], dim=-1)
        x_bk = self.brk(x_tmp)
        x_bk = self.act1(self.bn(self.cv1(x_bk)))

        out_h, out_w = torch.split(x_bk, [h,w], dim=-1)
        out_h = self.cv2(out_h)
        out_w = self.cv3(out_w)
        out = out_w * out_h.permute(0, 1, 3, 2)

        return x * self.act2(out)

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    
class PoolBlock(nn.Module):
    def __init__(self, size):
        super(PoolBlock, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d(size)
        self.max = nn.AdaptiveMaxPool2d(size)

    def forward(self, x):
        avg = self.avg(x)
        max = self.max(x)
        return avg + max


class RepBlock(nn.Module):
    def __init__(self, c, n, shortcut, g, e):
        super(RepBlock, self).__init__()
        self.m0 = Bottleneck(c, c, shortcut, g, e=1.0)
        self.m = nn.ModuleList(Bottleneck(2 * c, c, shortcut, g, e=1.0) for _ in range(n-1))

    def forward(self, x):
        y = []
        a = x
        b = self.m0(a)
        y.append(b)
        for m in self.m:
            c = torch.cat([a, b], 1)
            a = b
            b = m(c)
            y.append(b)

        return y
    
class PWConv(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(PWConv, self).__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((4 + 3) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = RepBlock(self.c, 3, shortcut, g, e=1.0)
        self.pw = PW(2 * self.c, 2 * self.c)


    def forward(self, x):
        t = self.cv1(x)
        y = list(t.chunk(2, 1))
        y.extend(self.m(y[-1]))
        y.append(self.pw(t))
        return self.cv2(torch.cat(y, 1))
    
    def forward_split(self, x):
        t = self.cv1(x)
        y = list(t.split((self.c, self.c), 1))
        y.extend(self.m(y[-1]))
        y.append(self.pw(t))
        return self.cv2(torch.cat(y, 1))