import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    # YOLO 标准卷积模块，用于替代直接的 nn.Conv2d 以保持格式统一
    default_act = nn.SiLU()
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k//2 if p is None else p, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DHTAstatic_hypergraph(nn.Module):
    """
    Dynamic Hypergraph Token Attention (DHTA) with ablation switches

    Supported ablation modes:
    - "full": dynamic association + aggregation + broadcast   (完整 DHTA)
    - "static_hypergraph": fixed grouping, no dynamic incidence generation
    - "dynamic_association_only": only dynamic assignment, no edge update / no broadcast loop
    - "dynamic_agg": dynamic association + aggregation (+ edge update), no broadcast
    """

    def __init__(self, dim, num_edges=32, act=True, ablation_mode="dynamic_agg"):
        super().__init__()
        self.num_edges = num_edges
        self.ablation_mode = ablation_mode

        assert ablation_mode in [
            "full",
            "static_hypergraph",
            "dynamic_association_only",
            "dynamic_agg",
        ], f"Unsupported ablation_mode: {ablation_mode}"

        # local query branch
        self.q = nn.Conv2d(dim, dim, 1, bias=False)
        self.v = nn.Conv2d(dim, dim, 1, bias=False)

        # dynamic incidence generator
        self.k = nn.Conv2d(dim, num_edges, 1, bias=False)

        # hyperedge update
        self.edge_update = nn.Sequential(
            nn.Conv1d(dim, dim, 1, bias=False),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, 1, bias=False)
        )

        # final projection
        self.proj = Conv(dim, dim, k=1, act=act)

    def _static_mats(self, B, N, device, dtype):
        """固定分组的关联/广播矩阵与输入无关，构造一次后按 (N, device, dtype) 缓存。

        incidence: (1, E, N) 行归一化，组内节点权重 1/count，行和为 1
        broadcast: (1, E, N) 0/1 掩码，每个节点只属于一个超边，列和为 1
        """
        key = (B, N, str(device), dtype)
        cache = self.__dict__.setdefault("_static_cache", {})
        if key not in cache:
            E = self.num_edges
            mask = torch.zeros(B, E, N, device=device, dtype=dtype)
            gid = torch.arange(N, device=device) * E // N
            gid = torch.clamp(gid, max=E - 1)
            mask.scatter_(1, gid.view(1, 1, N).expand(B, 1, N), 1.0)
            count = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
            cache[key] = (mask / count, mask)
        return cache[key]

    def _build_static_incidence(self, B, N, device, dtype):
        return self._static_mats(B, N, device, dtype)[0]

    def _build_static_broadcast(self, B, N, device, dtype):
        return self._static_mats(B, N, device, dtype)[1]

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DHTAstatic_hypergraph(nn.Module):
    """
    Dynamic Hypergraph Token Attention (DHTA) with ablation switches

    Supported ablation modes:
    - "full": dynamic association + aggregation + broadcast   (完整 DHTA)
    - "static_hypergraph": fixed grouping, no dynamic incidence generation
    - "dynamic_association_only": only dynamic assignment, no edge update / no broadcast loop
    - "dynamic_agg": dynamic association + aggregation (+ edge update), no broadcast
    """

    def __init__(self, dim, num_edges=32, act=True, ablation_mode="dynamic_agg"):
        super().__init__()
        self.num_edges = num_edges
        self.ablation_mode = ablation_mode

        assert ablation_mode in [
            "full",
            "static_hypergraph",
            "dynamic_association_only",
            "dynamic_agg",
        ], f"Unsupported ablation_mode: {ablation_mode}"

        # local query branch
        self.q = nn.Conv2d(dim, dim, 1, bias=False)
        self.v = nn.Conv2d(dim, dim, 1, bias=False)

        # dynamic incidence generator
        self.k = nn.Conv2d(dim, num_edges, 1, bias=False)

        # hyperedge update
        self.edge_update = nn.Sequential(
            nn.Conv1d(dim, dim, 1, bias=False),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, 1, bias=False)
        )

        # final projection
        self.proj = Conv(dim, dim, k=1, act=act)

    def _build_static_incidence(self, B, N, device, dtype):
        """
        构建固定分组的 static hypergraph incidence matrix
        shape: (B, E, N)

        简单做法：
        按节点索引平均分成 E 组，每个节点只属于一个固定超边。
        """
        E = self.num_edges
        incidence = torch.zeros(B, E, N, device=device, dtype=dtype)

        # 每个节点按 index 均匀映射到某个超边
        group_ids = torch.arange(N, device=device) * E // N
        group_ids = torch.clamp(group_ids, max=E - 1)  # [N]

        mask = torch.zeros(B, E, N, device=device, dtype=dtype)
        mask.scatter_(1, group_ids.view(1, 1, N).expand(B, 1, N), 1.0)
        count = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        incidence = mask / count

        return incidence

    def _build_static_broadcast(self, B, N, device, dtype):
        """
        静态支路的 hyperedge -> node 广播矩阵。

        原实现直接复用行归一化的 incidence（每个节点被额外缩放 1/count），
        与动态支路 softmax(dim=E)（每个节点对超边的权重求和为 1）口径不一致，
        导致静态行的输出尺度小一个量级、比较不公平。这里改为列归一化：
        每个节点只属于一个固定超边，故其自身超边权重为 1，列和为 1。
        """
        E = self.num_edges
        group_ids = torch.arange(N, device=device) * E // N
        group_ids = torch.clamp(group_ids, max=E - 1)
        mat = torch.zeros(B, E, N, device=device, dtype=dtype)
        mat.scatter_(1, group_ids.view(1, 1, N).expand(B, 1, N), 1.0)
        return mat

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        E = self.num_edges

        q = self.q(x)  # (B, C, H, W)
        v = self.v(x).contiguous().reshape(B, C, N)  # (B, C, N)

        # --------------------------------------------------
        # 1) incidence matrix generation
        # --------------------------------------------------
        if self.ablation_mode == "static_hypergraph":
            incidence_matrix_N = self._build_static_incidence(
                B=B, N=N, device=x.device, dtype=v.dtype
            )  # (B, E, N)
            k = None
        else:
            k = self.k(x).contiguous().reshape(B, E, N)  # (B, E, N)
            incidence_matrix_N = F.softmax(k, dim=-1)    # node -> hyperedge assignment

        # --------------------------------------------------
        # 2) dynamic association only
        #    用动态分配直接生成一个粗粒度上下文，不做 edge_update / broadcast
        # --------------------------------------------------
        if self.ablation_mode == "dynamic_association_only":
            # 节点 -> 超边
            hyperedges = torch.bmm(incidence_matrix_N, v.transpose(1, 2).contiguous())  # (B, E, C)

            # 直接将超边特征平均回写到所有节点（不使用对称广播，不做 edge_update）
            hyper_mean = hyperedges.mean(dim=1, keepdim=True)           # (B, 1, C)
            out = hyper_mean.transpose(1, 2).contiguous().expand(B, C, N)
            out = out.contiguous().reshape(B, C, H, W)

            out = out + q
            return (x + self.proj(out)).contiguous()

        # --------------------------------------------------
        # 3) node -> hyperedge aggregation
        # --------------------------------------------------
        hyperedges = torch.bmm(
            incidence_matrix_N,
            v.transpose(1, 2).contiguous()
        )  # (B, E, C)

        # --------------------------------------------------
        # 4) hyperedge update
        # --------------------------------------------------
        hyperedges = hyperedges.transpose(1, 2).contiguous()  # (B, C, E)
        hyperedges = self.edge_update(hyperedges)             # (B, C, E)

        # --------------------------------------------------
        # 5) dynamic association + aggregation only
        #    不广播回节点，只把聚合后的全局信息平均展开
        # --------------------------------------------------
        if self.ablation_mode == "dynamic_agg":
            hyper_mean = hyperedges.mean(dim=-1, keepdim=True)         # (B, C, 1)
            out = hyper_mean.expand(B, C, N).contiguous().reshape(B, C, H, W)

            out = out + q
            return (x + self.proj(out)).contiguous()

        # --------------------------------------------------
        # 6) full DHTA: hyperedge -> node broadcast
        # --------------------------------------------------
        if self.ablation_mode == "static_hypergraph":
            # static hypergraph 使用列归一化的固定广播矩阵（与动态支路口径一致）
            incidence_matrix_E = self._build_static_broadcast(
                B=B, N=N, device=x.device, dtype=v.dtype
            )
        else:
            incidence_matrix_E = F.softmax(k, dim=1)  # (B, E, N)

        out = torch.bmm(
            incidence_matrix_E.transpose(1, 2).contiguous(),
            hyperedges.transpose(1, 2).contiguous()
        )  # (B, N, C)

        out = out.transpose(1, 2).contiguous().reshape(B, C, H, W)
        out = out + q

        return (x + self.proj(out)).contiguous()

class DHTA(nn.Module):
    """
    Dynamic Hypergraph Token Attention (DHTA)
    
    原理：
    利用动态超图机制，实现空间像素的全局动态聚类。
    具有 O(N) 的线性计算复杂度，完美适配高分辨率水下图像的实时检测。
    """
    def __init__(self, dim, num_edges=16, act=True):
        """
        :param dim: 输入特征图的通道数 (Channel)
        :param num_edges: 超边的数量 (Hyperedges, 即聚类中心的数量)。通常设为 16, 32 或 64。
        """
        super().__init__()
        self.num_edges = num_edges
        
        # 1. 节点特征映射
        # q: 用于后续与超边特征融合
        # v: 用于提供像素的实际特征内容
        self.q = nn.Conv2d(dim, dim, 1, bias=False)
        self.v = nn.Conv2d(dim, dim, 1, bias=False)
        
        # 2. 动态关联矩阵生成器 (Incidence Matrix Generator)
        # 将维度映射到超边数量，生成像素与超边的关联度
        self.k = nn.Conv2d(dim, num_edges, 1, bias=False) 
        
        # 3. 超边特征更新网络 (Hyperedge Update)
        # 对聚类后的全局特征进行非线性处理
        self.edge_update = nn.Sequential(
            nn.Conv1d(dim, dim, 1, bias=False),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, 1, bias=False)
        )
        
        # 4. 最终输出映射
        self.proj = Conv(dim, dim, k=1, act=act)
    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W  # 节点总数 (N)
        
        # --- [节点特征准备] ---
        q = self.q(x)  # (B, C, H, W)
        
        # 强制连续后再 reshape
        v = self.v(x).contiguous().reshape(B, C, N)  
        k = self.k(x).contiguous().reshape(B, self.num_edges, N)
        
        # --- [Step 1: 动态生成关联矩阵] ---
        incidence_matrix_N = F.softmax(k, dim=-1) 
        
        # --- [Step 2: 节点 -> 超边] ---
        hyperedges = torch.bmm(incidence_matrix_N, v.transpose(1, 2).contiguous()) 
        
        # --- [Step 3: 超边内部信息交互] ---
        hyperedges = hyperedges.transpose(1, 2).contiguous()  # (B, C, E)
        hyperedges = self.edge_update(hyperedges)             # (B, C, E)
        
        # --- [Step 4: 超边 -> 节点] ---
        incidence_matrix_E = F.softmax(k, dim=1)  # (B, E, N)
        
        out = torch.bmm(incidence_matrix_E.transpose(1, 2).contiguous(), 
                        hyperedges.transpose(1, 2).contiguous())
        
        # 【核心修复】：先 transpose，再强制 contiguous，最后 reshape
        out = out.transpose(1, 2).contiguous().reshape(B, C, H, W)
        
        out = out + q
        
        # 保证最终输出给 YOLO 下一层时，内存绝对连续
        return (x + self.proj(out)).contiguous()

import torch
import torch.nn as nn
import torch.nn.functional as F


class DHTAsss(nn.Module):
    """
    Dynamic Hypergraph Token Attention (DHTA) with assignment ablations

    assignment_mode:
        - "soft"      : soft assignment (original DHTA)
        - "hard"      : hard assignment / top-1
        - "topk"      : top-k assignment
        - "uniform"   : uniform assignment
        - "no_learned": no learned association (fixed / input-independent association)
    """

    def __init__(self, dim, num_edges=32, act=True, assignment_mode="topk", topk=4):
        super().__init__()
        self.num_edges = num_edges
        self.assignment_mode = assignment_mode
        self.topk = topk

        assert assignment_mode in ["soft", "hard", "topk", "uniform", "no_learned"], \
            f"Unsupported assignment_mode: {assignment_mode}"

        # 1. 节点特征映射
        self.q = nn.Conv2d(dim, dim, 1, bias=False)
        self.v = nn.Conv2d(dim, dim, 1, bias=False)

        # 2. 动态关联矩阵生成器
        self.k = nn.Conv2d(dim, num_edges, 1, bias=False)

        # 3. 超边特征更新
        self.edge_update = nn.Sequential(
            nn.Conv1d(dim, dim, 1, bias=False),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, 1, bias=False)
        )

        # 4. 输出映射
        self.proj = Conv(dim, dim, k=1, act=act)

    def _uniform_assignment(self, B, E, N, device, dtype):
        # 每个节点对每条超边赋予相同权重
        mat = torch.full((B, E, N), 1.0 / N, device=device, dtype=dtype)
        return mat

    def _fixed_assignment(self, B, E, N, device, dtype):
        """
        no learned association:
        不使用 self.k(x)，而是基于节点索引构造一个固定的输入无关分配。
        这里采用均匀分组方式：按节点序号划分到不同超边。
        """
        mat = torch.zeros(B, E, N, device=device, dtype=dtype)
        group_ids = torch.arange(N, device=device) * E // N
        group_ids = torch.clamp(group_ids, max=E - 1)

        for e in range(E):
            mask = (group_ids == e).to(dtype)
            count = mask.sum().clamp(min=1.0)
            mat[:, e, :] = mask / count
        return mat

    def _build_incidence_N(self, k):
        """
        Build node->hyperedge incidence matrix H^(N), shape: (B, E, N)
        Normalized over node dimension N
        """
        B, E, N = k.shape
        device, dtype = k.device, k.dtype

        if self.assignment_mode == "soft":
            # 原始 DHTA：soft assignment
            return F.softmax(k, dim=-1)

        elif self.assignment_mode == "hard":
            # top-1 hard assignment
            idx = torch.argmax(k, dim=1)  # (B, N)
            mat = torch.zeros(B, E, N, device=device, dtype=dtype)
            mat.scatter_(1, idx.unsqueeze(1), 1.0)

            # 按超边内节点数归一化，避免不同超边尺度不一致
            denom = mat.sum(dim=-1, keepdim=True).clamp(min=1.0)
            mat = mat / denom
            return mat

        elif self.assignment_mode == "topk":
            # top-k assignment
            k_val, k_idx = torch.topk(k, k=min(self.topk, E), dim=1)  # (B, K, N)
            mat = torch.zeros(B, E, N, device=device, dtype=dtype)
            mat.scatter_(1, k_idx, 1.0)

            # 对每个超边沿节点维归一化
            denom = mat.sum(dim=-1, keepdim=True).clamp(min=1.0)
            mat = mat / denom
            return mat

        elif self.assignment_mode == "uniform":
            return self._uniform_assignment(B, E, N, device, dtype)

        elif self.assignment_mode == "no_learned":
            return self._fixed_assignment(B, E, N, device, dtype)

        else:
            raise ValueError(f"Unknown assignment_mode: {self.assignment_mode}")

    def _build_incidence_E(self, k, incidence_N):
        """
        Build hyperedge->node broadcast matrix H^(E), shape: (B, E, N)
        Normalized over hyperedge dimension E
        """
        B, E, N = incidence_N.shape
        device, dtype = incidence_N.device, incidence_N.dtype

        if self.assignment_mode == "soft":
            return F.softmax(k, dim=1)

        elif self.assignment_mode == "hard":
            idx = torch.argmax(k, dim=1)  # (B, N)
            mat = torch.zeros(B, E, N, device=device, dtype=dtype)
            mat.scatter_(1, idx.unsqueeze(1), 1.0)
            return mat

        elif self.assignment_mode == "topk":
            _, k_idx = torch.topk(k, k=min(self.topk, E), dim=1)
            mat = torch.zeros(B, E, N, device=device, dtype=dtype)
            mat.scatter_(1, k_idx, 1.0)

            # 对每个节点沿超边维归一化
            denom = mat.sum(dim=1, keepdim=True).clamp(min=1.0)
            mat = mat / denom
            return mat

        elif self.assignment_mode == "uniform":
            # 对每个节点，所有超边平均广播
            return torch.full((B, E, N), 1.0 / E, device=device, dtype=dtype)

        elif self.assignment_mode == "no_learned":
            # 固定关联下，广播也使用输入无关的固定矩阵
            mat = self._fixed_assignment(B, E, N, device, dtype)

            # 转成按超边维归一的广播矩阵
            denom = mat.sum(dim=1, keepdim=True).clamp(min=1.0)
            mat = mat / denom
            return mat

        else:
            raise ValueError(f"Unknown assignment_mode: {self.assignment_mode}")

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        q = self.q(x)                                  # (B, C, H, W)
        v = self.v(x).contiguous().reshape(B, C, N)    # (B, C, N)

        if self.assignment_mode in ["soft", "hard", "topk"]:
            k = self.k(x).contiguous().reshape(B, self.num_edges, N)  # (B, E, N)
        else:
            k = None

        # Step 1: incidence generation
        if k is None:
            dummy = torch.zeros(B, self.num_edges, N, device=x.device, dtype=v.dtype)
            incidence_matrix_N = self._build_incidence_N(dummy)
            incidence_matrix_E = self._build_incidence_E(dummy, incidence_matrix_N)
        else:
            incidence_matrix_N = self._build_incidence_N(k)
            incidence_matrix_E = self._build_incidence_E(k, incidence_matrix_N)

        # Step 2: node -> hyperedge
        hyperedges = torch.bmm(
            incidence_matrix_N,
            v.transpose(1, 2).contiguous()
        )  # (B, E, C)

        # Step 3: hyperedge update
        hyperedges = hyperedges.transpose(1, 2).contiguous()  # (B, C, E)
        hyperedges = self.edge_update(hyperedges)             # (B, C, E)

        # Step 4: hyperedge -> node
        out = torch.bmm(
            incidence_matrix_E.transpose(1, 2).contiguous(),
            hyperedges.transpose(1, 2).contiguous()
        )  # (B, N, C)

        out = out.transpose(1, 2).contiguous().reshape(B, C, H, W)
        out = out + q

        return (x + self.proj(out)).contiguous()