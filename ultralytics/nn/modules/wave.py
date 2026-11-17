import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import pywt
import numpy as np
# 你项目里如果能直接用 ultralytics.nn.modules.Conv，建议用它统一风格
try:
    from ultralytics.nn.modules import Conv
except Exception:
    Conv = None

class DWTFunction_2D(Function):
    @staticmethod
    def forward(ctx, input, matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1):
        # ✅ 关键：矩阵和输入对齐 dtype/device（AMP 下 input 可能是 half）
        dtype = input.dtype
        device = input.device
        matrix_Low_0  = matrix_Low_0.to(device=device, dtype=dtype)
        matrix_Low_1  = matrix_Low_1.to(device=device, dtype=dtype)
        matrix_High_0 = matrix_High_0.to(device=device, dtype=dtype)
        matrix_High_1 = matrix_High_1.to(device=device, dtype=dtype)

        # 原逻辑不变
        L = torch.matmul(matrix_Low_0, input)
        H = torch.matmul(matrix_High_0, input)
        LL = torch.matmul(L, matrix_Low_1)
        LH = torch.matmul(L, matrix_High_1)
        HL = torch.matmul(H, matrix_Low_1)
        HH = torch.matmul(H, matrix_High_1)

        # 如果你 backward 里也用到了矩阵，建议 ctx.save_for_backward 保存“对齐后的矩阵”
        ctx.save_for_backward(matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1)
        return LL, LH, HL, HH
    @staticmethod
    def backward(ctx, grad_LL, grad_LH, grad_HL, grad_HH):
        matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1 = ctx.saved_variables
        # pdb.set_trace()
        mL1t = matrix_Low_1.t().to(dtype=grad_LL.dtype, device=grad_LL.device)
        mH1t = matrix_High_1.t().to(dtype=grad_LL.dtype, device=grad_LL.device)
        
        grad_L = torch.matmul(grad_LL, mL1t) + torch.matmul(grad_LH, mH1t)
        # grad_L = torch.add(torch.matmul(grad_LL, matrix_Low_1.t().to(torch.float16)), torch.matmul(grad_LH, matrix_High_1.t().to(torch.float16)))
        grad_H = torch.matmul(grad_HL, mL1t) + torch.matmul(grad_HH, mH1t)
        # grad_H = torch.add(torch.matmul(grad_HL, matrix_Low_1.t().to(torch.float16)), torch.matmul(grad_HH, matrix_High_1.t().to(torch.float16)))
        grad_input = torch.add(torch.matmul(matrix_Low_0.t().to(torch.float16), grad_L.to(torch.float16)), torch.matmul(matrix_High_0.t().to(torch.float16), grad_H.to(torch.float16)))
        return grad_input, None, None, None, None
    
class ChannelAttention(nn.Module):
    def __init__(self, c, ratio=4):
        super().__init__()
        hidden = max(c // ratio, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c, hidden, 1, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, c, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        a = self.fc2(self.act(self.fc1(self.avg_pool(x))))
        w = self.sigmoid(a)
        return x * w



class SpatialAttention(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        assert k in (3, 7)
        p = 3 if k == 7 else 1
        self.conv = nn.Conv2d(2, 1, k, padding=p, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        w = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * w


class WF(nn.Module):
    """Weighted Fusion: 与你原版 WF 的思想一致：学习两个分支权重再做一次轻量卷积后处理。"""
    def __init__(self, c, eps=1e-8):
        super().__init__()
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.eps = eps
        if Conv is not None:
            self.post = Conv(c, c, k=3, s=1)
        else:
            self.post = nn.Sequential(
                nn.Conv2d(c, c, 3, 1, 1, bias=False),
                nn.BatchNorm2d(c),
                nn.SiLU(inplace=True),
            )

    def forward(self, x1, x2):
        w = F.relu(self.w)
        w = w / (w.sum() + self.eps)
        y = w[0] * x1 + w[1] * x2
        return self.post(y)


class HaarDWT2D(nn.Module):
    """
    用固定 Haar 小波核做 2D DWT：
    输入: (B,C,2H,2W) -> 输出 4个子带 (B,C,H,W): LL,LH,HL,HH
    实现：group conv，避免你原版每次构造矩阵的巨大开销。
    """
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

        # Haar 1D: low=[1/sqrt2, 1/sqrt2], high=[-1/sqrt2, 1/sqrt2]
        s = 1.0 / math.sqrt(2.0)
        low = torch.tensor([s, s], dtype=torch.float32)
        high = torch.tensor([-s, s], dtype=torch.float32)

        # 2D 外积得到 4 个 2x2 核
        ll = torch.ger(low, low)    # (2,2)
        lh = torch.ger(low, high)
        hl = torch.ger(high, low)
        hh = torch.ger(high, high)

        # 组织成 (4*C, 1, 2, 2) 的 depthwise weight
        base = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)  # (4,1,2,2)
        weight = base.repeat(channels, 1, 1, 1)                   # (4*C,1,2,2)

        # register_buffer：不参与训练、跟随 device
        self.register_buffer("weight", weight, persistent=False)

    def forward(self, x):
        # x: (B,C,2H,2W) -> y: (B,4C,H,W)
        y = F.conv2d(x, self.weight, stride=2, padding=0, groups=self.channels)
        ll, lh, hl, hh = y.chunk(4, dim=1)
        return ll, lh, hl, hh


class DWT_2D(nn.Module):
    """
    input: the 2D data to be decomposed -- (N, C, H, W)
    output -- lfc: (N, C, H/2, W/2)
              hfc_lh: (N, C, H/2, W/2)
              hfc_hl: (N, C, H/2, W/2)
              hfc_hh: (N, C, H/2, W/2)
    """
    def __init__(self, wavename):
        """
        2D discrete wavelet transform (DWT) for 2D image decomposition
        :param wavename: pywt.wavelist(); in the paper, 'chx.y' denotes 'biorx.y'.
        """
        super(DWT_2D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        """
        生成变换矩阵
        generating the matrices: \mathcal{L}, \mathcal{H}
        :return: self.matrix_low = \mathcal{L}, self.matrix_high = \mathcal{H}
        """
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros( ( L,      L1 + self.band_length - 2 ) )
        matrix_g = np.zeros( ( L1 - L, L1 + self.band_length - 2 ) )
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(self.input_height / 2)),0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(self.input_width / 2)),0:(self.input_width + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:,(self.band_length_half-1):end]
        matrix_h_1 = matrix_h_1[:,(self.band_length_half-1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_g_0 = matrix_g_0[:,(self.band_length_half-1):end]
        matrix_g_1 = matrix_g_1[:,(self.band_length_half-1):end]
        matrix_g_1 = np.transpose(matrix_g_1)

        if torch.cuda.is_available():
            self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
            self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
        else:
            self.matrix_low_0 = torch.Tensor(matrix_h_0)
            self.matrix_low_1 = torch.Tensor(matrix_h_1)
            self.matrix_high_0 = torch.Tensor(matrix_g_0)
            self.matrix_high_1 = torch.Tensor(matrix_g_1)

    def forward(self, input):
        """
        input_lfc = \mathcal{L} * input * \mathcal{L}^T
        input_hfc_lh = \mathcal{H} * input * \mathcal{L}^T
        input_hfc_hl = \mathcal{L} * input * \mathcal{H}^T
        input_hfc_hh = \mathcal{H} * input * \mathcal{H}^T
        :param input: the 2D data to be decomposed
        :return: the low-frequency and high-frequency components of the input 2D data
        """
        assert len(input.size()) == 4
        self.input_height = input.size()[-2]
        self.input_width = input.size()[-1]
        self.get_matrix()
        return DWTFunction_2D.apply(input, self.matrix_low_0, self.matrix_low_1, self.matrix_high_0, self.matrix_high_1)

class Downsamplewave(nn.Module):
    def __init__(self, wavename = 'haar'):
        super(Downsamplewave, self).__init__()
        # self.dwt = DWT_2D_tiny(wavename = wavename) #return LL
        self.dwt = DWT_2D(wavename = wavename)   #return LL,LH,HL,HH

    def forward(self, input):
        # LL = self.dwt(input)
        # return LL
        LL,LH,HL,HH = self.dwt(input)
        return LL,LH+HL+HH
    
class CSS(nn.Module):
    """
    YOLO12 版 CS：保留你原版逻辑（小波 LL/高频 + CA/SA + WF + residual），
    但小波分解改为固定 Haar 卷积核（无 mmcv/mmseg/pywt 依赖）。
    """
    def __init__(self, in_channels: int, ca_ratio: int = 4, sa_kernel: int = 3):
        super().__init__()
        # self.up = nn.Upsample(scale_factor=2, mode="nearest")
        # self.dwt = HaarDWT2D(in_channels)
        
        self.wave = nn.Sequential(*[nn.Upsample(scale_factor=2),Downsamplewave(wavename='haar')])

        self.ca = ChannelAttention(in_channels, ratio=ca_ratio)
        self.sa = SpatialAttention(k=sa_kernel)
        self.wf = WF(in_channels)

    def forward(self, x):
        # 对齐你原版：ll, hs = self.wave(x) 其中 hs=LH+HL+HH
        # x_up = self.up(x)
        # ll, lh, hl, hh = self.dwt(x_up)
        ll,hs = self.wave(x)
        # hs = lh + hl + hh

        ll = self.ca(ll) + x
        hs = self.sa(hs) + x
        y = self.wf(hs, ll)
        return y + x

class CS(nn.Module):
    def __init__(self, in_channels, ca_ratio=4, sa_kernel=3):
        super().__init__()
        self.wave = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # ✅ 替换 nearest
            Downsamplewave(wavename='haar')
        )
        self.ca = ChannelAttention(in_channels, ratio=ca_ratio)
        self.sa = SpatialAttention(k=sa_kernel)
        self.wf = WF(in_channels)

        self.alpha = nn.Parameter(torch.zeros(1))  # ✅ 初始 0，先别破坏定位

    def forward(self, x):
        ll, hs = self.wave(x)

        ll = self.ca(ll)
        hs = self.sa(hs)
        y  = self.wf(hs, ll)

        # ✅ 只做“可控残差修正”
        return x + self.alpha * (y - x)

import torch
import torch.nn as nn

# 依赖：你工程里已有
# from .conv import Conv                      # 你的 YOLO Conv
# from .wave import Downsamplewave            # 你的 haar 小波
# from .attention import ChannelAttention, SpatialAttention
# from .wf import WF
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
    
class ConvM(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt


# class CSConv(nn.Module):
#     """
#     频域保持下采样 (FPD-Stem) - 消融实验版本
#     支持模式 (mode):
#         - 'irreversible': 不可逆的传统步长卷积下采样 (Baseline)
#         - 'haar': Haar 小波基 (滤波器长度 2)
#         - 'db2': Daubechies 2 小波基 (滤波器长度 4)
#         - 'db4': Daubechies 4 小波基 (滤波器长度 8)
#     """
#     def __init__(self, c1, c2, k=3, s=2, act=True, mode='irreversible'):
#         super().__init__()
#         self.mode = mode.lower()
        
#         if self.mode == 'irreversible':
#             # ---------------------------------------------------------
#             # 消融实验 1: 不可逆下采样 (证明 Stride=2 导致高频信息丢失)
#             # ---------------------------------------------------------
#             self.stem = ConvM(c1, c2, k=k, s=2, p=k//2, act=act)
        
#         elif self.mode in ['haar', 'db2', 'db4']:
#             # ---------------------------------------------------------
#             # 消融实验 2: 小波基选择 (Haar vs db2 vs db4)
#             # ---------------------------------------------------------
#             # 获取 1D 小波分解低通(lo)和高通(hi)滤波器
#             wavelet = pywt.Wavelet(self.mode)
#             dec_lo = torch.tensor(wavelet.dec_lo, dtype=torch.float32)
#             dec_hi = torch.tensor(wavelet.dec_hi, dtype=torch.float32)
            
#             # 张量外积生成 2D 滤波器 (LL, LH, HL, HH)
#             ll = torch.outer(dec_lo, dec_lo)
#             lh = torch.outer(dec_hi, dec_lo)
#             hl = torch.outer(dec_lo, dec_hi)
#             hh = torch.outer(dec_hi, dec_hi)
            
#             # 堆叠成卷积核形状: [4, 1, K, K]
#             filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
            
#             # 为了使用 depthwise convolution (groups=C)，复制滤波器 C 次
#             # 最终权重形状: [4*c1, 1, K, K]
#             self.register_buffer('wavelet_weights', filters.repeat(c1, 1, 1, 1))
            
#             # 滤波器长度计算 padding，确保下采样后尺寸严格为 H/2, W/2
#             self.filter_len = wavelet.dec_len
#             self.pad = self.filter_len // 2 - 1
            
#             # 融合层：接收 4*c1 个通道，输出 c2 个通道 (注意这里 stride=1)
#             self.fusion_conv = ConvM(c1 * 4, c2, k=3, s=1, p=1, act=act)
            
#         else:
#             raise ValueError(f"Unsupported mode: {self.mode}")

#     def forward(self, x):
#         if self.mode == 'irreversible':
#             return self.stem(x)
            
#         # 强制使用 float32 进行小波卷积，防止权重在 FP16 下精度溢出
#         with torch.cuda.amp.autocast(enabled=False):
#             x_f = x.float()
            
#             # 同样强制将权重转为 float32，防止 YOLO 在 val 阶段调用 model.half() 导致权重变半精度
#             weight_f = self.wavelet_weights.float()
            
#             B, C, H, W = x_f.shape
            
#             # 动态 Padding 处理
#             pad_h = H % 2
#             pad_w = W % 2
#             x_pad = F.pad(x_f, (self.pad, self.pad + pad_w, self.pad, self.pad + pad_h), mode='reflect')
            
#             # 使用 float32 的特征和 float32 的权重进行卷积
#             out_dwt = F.conv2d(x_pad, weight_f, stride=2, groups=C)
            
#             # 恢复到与输入 x 相同的原始数据类型 (支持训练 FP16 和验证 FP16)
#             out_dwt = out_dwt.to(x.dtype)

#         # [B, 4*C, H/2, W/2] -> [B, c2, H/2, W/2]
#         return self.fusion_conv(out_dwt)

class CSConv(nn.Module):
    """
    Haar-Downsampling Stem (HDS-Stem)
    
    原理:
    利用 Haar 小波变换 (DWT) 实现无损下采样，替代传统的 Conv(stride=2)。
    input(3, H, W) -> DWT -> (12, H/2, W/2) -> Conv3x3 -> (c2, H/2, W/2)
    
    优势:
    1. 信息零损失 (Zero Information Loss): 保留了所有频段的物理信息。
    2. 极其稳健: 网络自动学习如何利用高频信息，避免人工设计的注意力引入偏差。
    """
    def __init__(self, c1, c2, k=3, s=2, act=True):
        super().__init__()
        # Haar DWT 会把通道数变成原来的 4 倍 (3 -> 12)
        in_channels = c1 * 4
        
        # 使用 3x3 卷积进行特征融合与提取
        # 注意：这里 stride=1，因为 DWT 已经把尺寸缩小了一半
        self.fusion_conv = ConvM(in_channels, c2, k=3, p=1,s=1, act=act)

    def forward(self, x):
        # x: [B, 3, H, W]
        
        # 1. Haar 小波变换 (无参数，极快)
        # 强制使用 float32 防止精度溢出
        with torch.cuda.amp.autocast(enabled=False):
            x_f = x.float()
            
            # 兼容性处理: 如果长宽是奇数，进行 Padding
            B, C, H, W = x_f.shape
            pad_h = H % 2
            pad_w = W % 2
            if pad_h or pad_w:
                x_f = F.pad(x_f, (0, pad_w, 0, pad_h), mode='reflect')
            
            # 切片提取 4 个分量
            x00 = x_f[:, :, 0::2, 0::2] # LL
            x01 = x_f[:, :, 0::2, 1::2] # LH
            x10 = x_f[:, :, 1::2, 0::2] # HL
            x11 = x_f[:, :, 1::2, 1::2] # HH
            
            # DWT 公式
            ll = x00 + x01 + x10 + x11
            lh = x00 - x01 + x10 - x11
            hl = x00 + x01 - x10 - x11
            hh = x00 - x01 - x10 + x11
            
            # 拼接: [B, 12, H/2, W/2]
            out = torch.cat([ll, lh, hl, hh], dim=1)
            
            # 转回半精度 (如果在训练中使用混合精度)
            if x.dtype == torch.float16:
                out = out.half()

        # 2. 卷积融合
        # import pdb
        # pdb.set_trace()
        return self.fusion_conv(out)

class StemS2D(nn.Module):
    """space-to-depth stem：把 2x2 邻域直接搬进通道（x00,x01,x10,x11），不做任何滤波。

    与 Haar 版本共享同一个 3x3 融合卷积（4*c1 -> c2），因此参数量、计算量完全相同，
    唯一差别是"通道里装的是原始像素"还是"装的是频域组合"。
    """

    def __init__(self, c1, c2, k=3, s=2, act=True):
        super().__init__()
        self.fusion_conv = ConvM(c1 * 4, c2, k=3, p=1, s=1, act=act)

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            x_f = x.float()
            B, C, H, W = x_f.shape
            pad_h, pad_w = H % 2, W % 2
            if pad_h or pad_w:
                x_f = F.pad(x_f, (0, pad_w, 0, pad_h), mode='reflect')
            x00 = x_f[:, :, 0::2, 0::2]
            x01 = x_f[:, :, 0::2, 1::2]
            x10 = x_f[:, :, 1::2, 0::2]
            x11 = x_f[:, :, 1::2, 1::2]
            out = torch.cat([x00, x01, x10, x11], dim=1)
            if x.dtype == torch.float16:
                out = out.half()
        return self.fusion_conv(out)


class StemLL(nn.Module):
    """Haar 仅保留 LL 子带，LH/HL/HH 置零，再走同一个 3x3 融合卷积。

    通道数与 Haar 版本一致（4*c1），因此参数/计算量完全相同：
    用来隔离"高频信息本身"与"4 通道投影容量"两个因素。
    """

    def __init__(self, c1, c2, k=3, s=2, act=True):
        super().__init__()
        self.fusion_conv = ConvM(c1 * 4, c2, k=3, p=1, s=1, act=act)

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            x_f = x.float()
            B, C, H, W = x_f.shape
            pad_h, pad_w = H % 2, W % 2
            if pad_h or pad_w:
                x_f = F.pad(x_f, (0, pad_w, 0, pad_h), mode='reflect')
            x00 = x_f[:, :, 0::2, 0::2]
            x01 = x_f[:, :, 0::2, 1::2]
            x10 = x_f[:, :, 1::2, 0::2]
            x11 = x_f[:, :, 1::2, 1::2]
            ll = x00 + x01 + x10 + x11
            z = torch.zeros_like(ll)
            out = torch.cat([ll, z, z, z], dim=1)
            if x.dtype == torch.float16:
                out = out.half()
        return self.fusion_conv(out)
