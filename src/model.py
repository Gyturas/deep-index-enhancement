"""深度截面因子模型：L0 / L1A / L1B / L2 四级，用 ``level`` 开关控制。

    L0   纯时序编码，个股之间不交换信息
    L1A  + 末步截面 attention          O(N^2)
    L1B  + 逐时间步截面 attention      O(N^2 L)，MASTER 式
    L2   在 L1 之上 + 市场状态门控

设计上和 Oxford 那版（v1）最大的两处不同：

1. **属性 embedding 取代个股 embedding**。v1 照搬 DeePM，给每个资产一个可学习的
   ticker embedding——期货成立（螺纹永远是螺纹），个股不成立：成分股半年调整、
   上市退市、主业变更；1000 个 embedding 纯粹用来背身份，是过拟合装置；
   而且新进池的股票没有训练过的向量，冷启动无解。
   这里改成**属性相加**：
       e_i = E_ind2[申万二级] + E_ind1[申万一级]
           + E_size[市值档] + E_liq[流动性档] + E_age[年限档]
   参数量从 O(股票数) 降到 O(档数)，天然泛化到新成分股。

2. **输出是打分不是仓位**，末端只做截面标准化，不过 tanh。
   tanh 会压缩尾部，而排序任务恰恰关心尾部。
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ====================================================================== 基础块
class SharedDropout(nn.Module):
    """时间维共享掩码的 dropout（变分 dropout，Gal & Ghahramani 2016）。

    v1 实测的教训：标准 dropout 在每个时间步独立采样掩码，会让相邻两日的输出
    出现纯噪音跳变——日均换手从 0.92 涨到 11.01、成本涨 29 倍。
    v2 训练时虽然不直接算成本，但同样的抖动会污染时序表示。
    掩码沿倒数第二维（时间维）广播，同一序列内所有时间步共享。
    """

    def __init__(self, p: float):
        super().__init__()
        self.p = float(p)

    def forward(self, x):
        if not self.training or self.p <= 0:
            return x
        shape = list(x.shape)
        shape[-2] = 1
        keep = x.new_empty(shape).bernoulli_(1 - self.p) / (1 - self.p)
        return x * keep


class ResSwiGLU(nn.Module):
    """Post-Norm + SwiGLU + 残差。全网复用的标准非线性块。"""

    def __init__(self, d: int, mult: int = 2, dropout: float = 0.3, seq_dim: bool = True):
        super().__init__()
        h = d * mult
        self.w1, self.v, self.w2 = nn.Linear(d, h), nn.Linear(d, h), nn.Linear(h, d)
        self.drop = SharedDropout(dropout) if seq_dim else nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d)

    def forward(self, x):
        return self.ln(x + self.drop(self.w2(self.w1(x) * F.silu(self.v(x)))))


def sinusoidal_pe(L: int, d: int, device) -> torch.Tensor:
    pos = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(0, d, 2, device=device, dtype=torch.float32)
    ang = pos / torch.pow(10000.0, i / d)
    pe = torch.zeros(L, d, device=device)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(ang), torch.cos(ang)
    return pe


# ====================================================================== 属性身份
class AttributeContext(nn.Module):
    """s_i = Linear(e_i)，e_i 由五个属性 embedding **相加**而成。

    相加而不是拼接：拼接会让维度随属性数线性增长，而这些属性本就该被理解为
    同一个"身份空间"里的几个正交方向。相加也让新增属性不用改下游维度。

    未知取值统一映到 index 0（各 embedding 表的第 0 行），因此建表时要 +1 预留。
    """

    def __init__(self, n_ind1: int, n_ind2: int, d: int, n_bucket: int = 12):
        super().__init__()
        self.e_ind1 = nn.Embedding(n_ind1 + 1, d, padding_idx=0)
        self.e_ind2 = nn.Embedding(n_ind2 + 1, d, padding_idx=0)
        self.e_size = nn.Embedding(n_bucket + 1, d, padding_idx=0)
        self.e_liq = nn.Embedding(n_bucket + 1, d, padding_idx=0)
        self.e_age = nn.Embedding(n_bucket + 1, d, padding_idx=0)
        self.proj = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d))

    def forward(self, attr: dict) -> torch.Tensor:
        e = (self.e_ind1(attr["ind1"]) + self.e_ind2(attr["ind2"])
             + self.e_size(attr["size"]) + self.e_liq(attr["liq"])
             + self.e_age(attr["age"]))
        return self.proj(e)                               # (B, N, d)


# ====================================================================== 时序
class TemporalEncoder(nn.Module):
    """Linear 投影 + 位置编码 -> FiLM(属性调制) -> LSTM -> 因果 MHA -> adapter。

    LSTM 与因果注意力分工不同：LSTM 是非线性低通滤波，压高频噪音、处理路径依赖；
    注意力绕过 LSTM 的记忆衰减，直接看回看窗口内的任意时点。
    权重在全部股票间共享（channel-independent），这既是正则化，
    也让样本量变成 "股票数 x 天数" 而非 "天数"。
    """

    def __init__(self, n_feat: int, d: int, n_head: int = 4, dropout: float = 0.3):
        super().__init__()
        self.d = d
        self.inp = nn.Linear(n_feat, d)
        self.film = nn.Linear(d, 2 * d)                   # 由属性向量生成 gamma / beta
        self.lstm = nn.LSTM(d, d, batch_first=True)
        self.init = nn.Linear(d, 2 * d)                   # LSTM 初始态也由属性给
        self.ad1 = ResSwiGLU(d, dropout=dropout)
        self.mha = nn.MultiheadAttention(d, n_head, dropout=0.0, batch_first=True)
        self.ad2 = ResSwiGLU(d, dropout=dropout)

    def forward(self, x, s):
        # x: (B,N,L,F)   s: (B,N,d)
        B, N, L, _ = x.shape
        h = self.inp(x) + sinusoidal_pe(L, self.d, x.device)
        g, b = self.film(s).chunk(2, dim=-1)
        h = g.unsqueeze(2) * h + b.unsqueeze(2)           # FiLM

        h = h.reshape(B * N, L, self.d)
        h0, c0 = torch.tanh(self.init(s)).reshape(B * N, 1, 2 * self.d).chunk(2, -1)
        out, _ = self.lstm(h, (h0.transpose(0, 1).contiguous(),
                               c0.transpose(0, 1).contiguous()))
        z = self.ad1(out)
        causal = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), 1)
        a, _ = self.mha(z, z, z, attn_mask=causal, need_weights=False)
        return self.ad2(a).reshape(B, N, L, self.d)


# ====================================================================== 截面
class CrossSectionAttn(nn.Module):
    """截面自注意力 + ReZero 门。

    ReZero 的 alpha 初始化为 0，所以这层一开始是恒等映射，优化器只在确有增益时
    才引入截面交互。训练完看 alpha 有没有离开 0，本身就是一个诊断量。

    不建预定义行业图：MASTER 表里用预定义图的 GAT 在 CSI800 上 RankIC 0.042、
    RankICIR 0.35 是全表最差之一。行业信息走属性 embedding 做条件，不做消息传递。
    """

    def __init__(self, d: int, n_head: int = 4, dropout: float = 0.3):
        super().__init__()
        self.mha = nn.MultiheadAttention(d, n_head, dropout=0.0, batch_first=True)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.ln = nn.LayerNorm(d)
        self.ad = ResSwiGLU(d, dropout=dropout, seq_dim=False)

    def forward(self, H, mask):
        """H: (M, N, d)   mask: (M, N) True=有效"""
        kpm = ~mask
        kpm = kpm & ~kpm.all(-1, keepdim=True)            # 防整行全屏蔽导致 NaN
        a, _ = self.mha(H, H, H, key_padding_mask=kpm, need_weights=False)
        return self.ad(self.ln(H + self.alpha * a))


class MarketGate(nn.Module):
    """市场状态门控（MASTER 式）：按当前市况动态放大/抑制每一维特征。

    alpha = F * softmax(W m + b) / beta

    乘以 F 是关键——让权重和为 F 而不是 1，这样和"每维都是 1"的均匀分布相比，
    alpha_f > 1 是放大、< 1 是抑制。直接用 softmax 会把所有特征一起缩小。
    温度 beta 控制分布尖锐程度（MASTER 在 CSI300 用 5、CSI800 用 2）。

    这和 FiLM 是**正交**的两个条件：FiLM 问"哪类股票适合什么因子"，
    门控问"什么市况下哪类因子有效"。
    """

    def __init__(self, n_mkt: int, n_feat: int, beta: float = 5.0):
        super().__init__()
        self.fc = nn.Linear(n_mkt, n_feat)
        self.beta, self.n_feat = beta, n_feat

    def forward(self, x, m):
        # x: (B,N,L,F)   m: (B,F_mkt)
        a = self.n_feat * torch.softmax(self.fc(m) / self.beta, dim=-1)
        return x * a[:, None, None, :]


# ====================================================================== 整体
class DeepFactor(nn.Module):
    def __init__(self, n_feat: int, n_ind1: int, n_ind2: int, *, level: str = "L0",
                 d: int = 64, n_head: int = 4, dropout: float = 0.3,
                 n_bucket: int = 12, n_mkt: int = 66, gate_beta: float = 5.0):
        super().__init__()
        assert level in ("L0", "L1A", "L1B", "L2"), level
        self.level = level
        self.gate = MarketGate(n_mkt, n_feat, gate_beta) if level == "L2" else None
        self.ctx = AttributeContext(n_ind1, n_ind2, d, n_bucket)
        self.backbone = TemporalEncoder(n_feat, d, n_head, dropout)
        need_xsec = level in ("L1A", "L1B", "L2")
        self.xsec = CrossSectionAttn(d, n_head, dropout) if need_xsec else None
        # L2 默认沿用 L1A 的截面位置；要用逐时间步版本传 level="L1B" 后再开门控
        self.xsec_per_step = (level == "L1B")
        self.head = nn.Linear(d, 1)

    # ------------------------------------------------------------------
    def forward(self, batch: dict) -> torch.Tensor:
        x = batch["x"]                                     # (B,N,L,F)
        mask = batch["mask"]                               # (B,N) bool
        if self.gate is not None:
            x = self.gate(x, batch["market"])

        s = self.ctx(batch["attr"])                        # (B,N,d)
        H = self.backbone(x, s)                            # (B,N,L,d)
        B, N, L, d = H.shape

        if self.xsec is not None and self.xsec_per_step:
            Ht = H.permute(0, 2, 1, 3).reshape(B * L, N, d)
            mt = mask.unsqueeze(1).expand(B, L, N).reshape(B * L, N)
            Ht = self.xsec(Ht, mt)
            H = Ht.reshape(B, L, N, d).permute(0, 2, 1, 3)

        Z = H[:, :, -1, :]                                 # 取末步 (B,N,d)

        if self.xsec is not None and not self.xsec_per_step:
            Z = self.xsec(Z, mask)

        score = self.head(Z).squeeze(-1)                   # (B,N)
        return xsec_standardize(score, mask)

    @property
    def rezero_alpha(self) -> float:
        """截面层的 ReZero 门。训练后若仍贴近 0，说明模型判定这层没用。"""
        return float(self.xsec.alpha.detach()) if self.xsec is not None else float("nan")


def xsec_standardize(score: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """日内截面 z-score，只在有效股票上统计。无效位置置 0（损失里会被掩掉）。"""
    m = mask.float()
    n = m.sum(-1, keepdim=True).clamp(min=2.0)
    mu = (score * m).sum(-1, keepdim=True) / n
    var = (((score - mu) * m) ** 2).sum(-1, keepdim=True) / n
    return ((score - mu) / (var.sqrt() + 1e-6) * m)
