# ultralytics/nn/modules/waveformer_yolo.py
import math
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F

# 你 YOLO 工程里一般都有 Conv（ultralytics.nn.modules.conv import Conv）
from ultralytics.nn.modules.conv import Conv


# -------------------------
# Small utilities
# -------------------------
class LayerNorm2d(nn.LayerNorm):
    """LN over channel dim for NCHW."""
    def forward(self, x: torch.Tensor):
        # x: (B,C,H,W) -> (B,H,W,C)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class DropPath(nn.Module):
    """Minimal DropPath to avoid timm dependency."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * random


class Mlp(nn.Module):
    """YOLO-friendly MLP (1x1 conv) for NCHW."""
    def __init__(self, dim, mlp_ratio=2.0, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Conv2d(dim, hidden, 1, 1, 0)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden, dim, 1, 1, 0)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# -------------------------
# Core operator: Wave2D (patched for AMP safety)
# -------------------------
class Wave2D(nn.Module):
    """
    This is adapted from your WaveFormer_dct.py Wave2D,
    but patched to be AMP-safe inside YOLO training:
    - Force internal computations in FP32 (autocast disabled) to avoid dtype mismatch
    - Cache cos/decay maps as FP32 and reuse
    """
    def __init__(self, infer_mode=False, res=14, dim=96, hidden_dim=96, **kwargs):
        super().__init__()
        self.res = res
        # IMPORTANT: WaveFormer's original uses groups=hidden_dim, so dim==hidden_dim is safest.
        assert dim == hidden_dim, "Wave2D in YOLO建议 dim==hidden_dim (depthwise)"
        self.dwconv = nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.infer_mode = infer_mode

        self.to_k = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.GELU(),
        )
        self.c = nn.Parameter(torch.ones(1) * 1.0)
        self.alpha = nn.Parameter(torch.ones(1) * 0.1)
        
        # self.register_buffer("cosH_k_base", cosH_k.float(), persistent=False)


    @staticmethod
    def get_cos_map(N: int, device, dtype=torch.float32):
        weight_x = (torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(1, -1) + 0.5) / N
        weight_n = torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(-1, 1)
        weight = torch.cos(weight_n * weight_x * torch.pi) * math.sqrt(2 / N)
        weight[0, :] = weight[0, :] / math.sqrt(2)
        return weight  # (N, N)

    @staticmethod
    def get_decay_map(resolution, device, dtype=torch.float32):
        resh, resw = resolution
        weight_n = torch.linspace(0, torch.pi, resh + 1, device=device, dtype=dtype)[:resh].view(-1, 1)
        weight_m = torch.linspace(0, torch.pi, resw + 1, device=device, dtype=dtype)[:resw].view(1, -1)
        weight = torch.pow(weight_n, 2) + torch.pow(weight_m, 2)
        weight = torch.exp(-weight)
        return weight  # (H, W)

    def forward(self, x: torch.Tensor, freq_embed=None):
        """
        x: (B,C,H,W)
        freq_embed: (B,H,W,C) or None
        """
        x_dtype = x.dtype
        device_type = "cuda" if x.is_cuda else "cpu"
        amp_ctx = torch.amp.autocast(device_type=device_type, enabled=False) if x.is_cuda else contextlib.nullcontext()

        with amp_ctx:
            x = x.float()
            B, C, H, W = x.shape

            # feature pre-proj
            dtype = self.dwconv.weight.dtype
            if x.dtype != dtype:
                x = x.to(dtype)
            if freq_embed is not None and freq_embed.dtype != dtype:
                freq_embed = freq_embed.to(dtype)
            x = self.dwconv(x)
            x = self.linear(x.permute(0, 2, 3, 1).contiguous())  # (B,H,W,2C)
            x, z = x.chunk(chunks=2, dim=-1)  # both (B,H,W,C)

            # cache maps (FP32)
            key = (H, W, x.device)
            cached = getattr(self, "_cache", None)
            if cached is None:
                cached = {}
                setattr(self, "_cache", cached)

            if key not in cached:
                cosH = self.get_cos_map(H, device=x.device, dtype=torch.float32).detach()
                cosW = self.get_cos_map(W, device=x.device, dtype=torch.float32).detach()
                decay = self.get_decay_map((H, W), device=x.device, dtype=torch.float32).detach()
                cached[key] = (cosH, cosW, decay)
            else:
                cosH, cosW, decay = cached[key]

            # DCT-like projections via conv1d
            x_perm = x.permute(0, 3, 2, 1).contiguous()     # (B,C,W,H)
            x_flat = x_perm.view(-1, 1, H)                  # (B*C*W,1,H)
            
            # cosH_k = cosH.view(H, 1, H)  # (H,1,H)
            # cosW_k = cosW.view(W, 1, W)  # (W,1,W)
            cosH_k = cosH.view(H, 1, H).contiguous()
            cosH_k = cosH_k.to(device=x_flat.device, dtype=x_flat.dtype)
            cosW_k = cosW.view(W, 1, W).contiguous()
            cosW_k = cosW_k.to(device=x_flat.device, dtype=x_flat.dtype) 

            # x_u0

            x_u0 = F.conv1d(x_flat, cosH_k).squeeze(-1)      # (B*C*W,H)
            x_u0 = x_u0.view(B, C, W, H).permute(0, 3, 2, 1).contiguous()  # (B,H,W,C)

            x_perm = x_u0.permute(0, 3, 1, 2).contiguous()   # (B,C,H,W)
            x_flat = x_perm.view(-1, 1, W)                   # (B*C*H,1,W)
            x_u0 = F.conv1d(x_flat, cosW_k).squeeze(-1)       # (B*C*H,W)
            x_u0 = x_u0.view(B, C, H, W).permute(0, 2, 3, 1).contiguous()  # (B,H,W,C)

            # x_v0 (same as x_u0 in your file; keep for consistency)
            x_perm = x.permute(0, 3, 2, 1).contiguous()
            x_flat = x_perm.view(-1, 1, H)
            x_v0 = F.conv1d(x_flat, cosH_k).squeeze(-1)
            x_v0 = x_v0.view(B, C, W, H).permute(0, 3, 2, 1).contiguous()

            x_perm = x_v0.permute(0, 3, 1, 2).contiguous()
            x_flat = x_perm.view(-1, 1, W)
            x_v0 = F.conv1d(x_flat, cosW_k).squeeze(-1)
            x_v0 = x_v0.view(B, C, H, W).permute(0, 2, 3, 1).contiguous()

            # freq embedding
            # if freq_embed is None:
            #     freq_embed = torch.zeros(B, H, W, self.hidden_dim, device=x.device, dtype=torch.float32)
            # else:
            #     freq_embed = freq_embed.float()

            # t = self.to_k(freq_embed)  # (B,H,W,C)
            # freq embedding (确保 dtype 与 to_k 权重一致)
            w_dtype = next(self.to_k.parameters()).dtype  # 通常是 fp16（AMP 下）
            w_device = next(self.to_k.parameters()).device
            
            if freq_embed is None:
                freq_embed = torch.zeros(B, H, W, self.hidden_dim, device=w_device, dtype=w_dtype)
            else:
                freq_embed = freq_embed.to(device=w_device, dtype=w_dtype)
            
            t = self.to_k(freq_embed)  # ✅ 不再 dtype mismatch

            c_t = self.c * t
            cos_term = torch.cos(c_t)
            eps = 1e-8
            sin_term = torch.sin(c_t) / (self.c + eps)

            wave_term = cos_term * x_u0
            velocity_term = sin_term * (x_v0 + (self.alpha / 2) * x_u0)
            final_term = wave_term + velocity_term

            # IDCT-like projections (use transposed kernels)
            # cosH_t = cosH.t().contiguous().view(H, 1, H)
            # cosW_t = cosW.t().contiguous().view(W, 1, W)
            cosH_t = cosH.t().contiguous().view(H, 1, H)
            cosH_t = cosH_t.to(device=x_flat.device, dtype=x_flat.dtype)  # ✅ 建议用 x_h 的 dtype/device
            cosW_t = cosW.t().contiguous().view(W, 1, W)
            cosW_t = cosW_t.to(device=x_flat.device, dtype=x_flat.dtype)  # ✅ 注意是 cosW_t，不是 cosW


            x_w = final_term.permute(0, 1, 3, 2).contiguous().view(B * H * C, 1, W)
            x_w = F.conv1d(x_w, cosW_t).squeeze(-1)  # (B*H*C,W)
            x_w = x_w.view(B, H, C, W).permute(0, 1, 3, 2).contiguous()  # (B,H,W,C)

            x_h = x_w.permute(0, 2, 3, 1).contiguous().view(B * W * C, 1, H)
            x_h = F.conv1d(x_h, cosH_t).squeeze(-1)                           # (B*W*C, H)

            x_final = x_h.view(B, W, C, H).permute(0, 2, 3, 1).contiguous() 

            # output gating (match your file's style)
            x_ln = self.out_norm(x_final.permute(0, 2, 3, 1).contiguous())
            gate = F.silu(z)
            x_gated = x_ln * gate
            x_out = self.out_linear(x_gated).permute(0, 3, 1, 2).contiguous()  # (B,C,H,W)

        return x_out.to(dtype=x_dtype)


# -------------------------
# YOLO module: WaveA2C2f (no WaveBlock)
# -------------------------
class WaveUnit(nn.Module):
    """Wave2D + MLP, residual style."""
    def __init__(self, dim, mlp_ratio=2.0, drop_path=0.0):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.op = Wave2D(res=14, dim=dim, hidden_dim=dim, infer_mode=False)
        self.dp = DropPath(drop_path)
        self.norm2 = LayerNorm2d(dim)
        self.mlp = Mlp(dim, mlp_ratio=mlp_ratio)

    def forward(self, x, freq_embed):
        x = x + self.dp(self.op(self.norm1(x), freq_embed))
        x = x + self.dp(self.mlp(self.norm2(x)))
        return x


class WaveA2C2f(nn.Module):
    """
    Drop-in replacement for A2C2f in YOLOv12:
    - keep YAML arg compatibility: [c2, a2, area]  (a2/area仅占位，不强依赖)
    - internal: cv1 -> n * WaveUnit -> cv2
    - freq_embed: learnable base grid (r0 x r0) interpolated to current (H,W)
    """
    def __init__(
        self,
        c1,
        c2,
        n=1,
        a2=True,          # placeholder for YAML compatibility
        area=1,           # placeholder for YAML compatibility
        residual=False,   # placeholder
        mlp_ratio=2.0,
        e=0.5,
        g=1,
        shortcut=True,
        freq_base=20,     # base resolution for learnable freq grid (P5@640≈20)
        drop_path=0.0
    ):
        super().__init__()
        c_ = int(c2 * e)

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1, 1)

        # learnable freq grid: (1, C, r0, r0) -> interpolate to (B,H,W,C)
        self.freq = nn.Parameter(torch.zeros(1, c_, freq_base, freq_base))
        nn.init.trunc_normal_(self.freq, std=0.02)

        self.m = nn.ModuleList([WaveUnit(c_, mlp_ratio=mlp_ratio, drop_path=drop_path) for _ in range(n)])

    def _freq_embed(self, x):
        # x: (B,C,H,W)
        B, C, H, W = x.shape
        f = F.interpolate(self.freq, size=(H, W), mode="bilinear", align_corners=False)  # (1,C,H,W)
        f = f.permute(0, 2, 3, 1).contiguous()  # (1,H,W,C)
        f = f.expand(B, H, W, C).to(dtype=x.dtype, device=x.device).contiguous()
        return f

    def forward(self, x):
        y = [self.cv1(x)]
        freq = self._freq_embed(y[0])
        for block in self.m:
            y.append(block(y[-1], freq))
        return self.cv2(torch.cat(y, 1))
