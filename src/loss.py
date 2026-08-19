"""损失：日频截面 IC + pairwise 排序 hinge。

为什么不用 MSE 当主损失
-----------------------
Chen-Pelger-Zhu (Management Science) 的对照：同样的深度网络、同样的数据，
目标函数从"MSE 预测收益"换成有经济含义的目标，样本外年化夏普 1.5 -> 2.6。
他们的原话是 off-the-shelf 的预测方法可能连线性的无套利模型都打不过。

对指增来说，预测的实际用法是**排序选股**，所以损失应该对齐排序而不是对齐数值：

* IC 损失量纲无关，不会被少数极端收益主导；
* 排序 hinge 直接惩罚"把该排前面的排到后面"（Feng et al., TOIS 2019）。

所有统计都在 mask 内做——mask 已经排除了当日不可建仓的样本
（停牌、一字板、收盘涨跌停、新股、非成分股）。
"""
from __future__ import annotations
import torch

EPS = 1e-8


def cross_sectional_ic(pred: torch.Tensor, label: torch.Tensor,
                       mask: torch.Tensor) -> torch.Tensor:
    """逐日截面 Pearson 相关。pred/label/mask: (B, N) -> (B,)

    有效股票数不足 2 的日子返回 0（不贡献梯度，也不算进均值的分子）。
    """
    m = mask.float()
    n = m.sum(-1, keepdim=True)
    ok = (n.squeeze(-1) >= 2)

    n = n.clamp(min=2.0)
    pc = (pred - (pred * m).sum(-1, keepdim=True) / n) * m
    lc = (label - (label * m).sum(-1, keepdim=True) / n) * m
    num = (pc * lc).sum(-1)
    den = pc.pow(2).sum(-1).sqrt() * lc.pow(2).sum(-1).sqrt()
    return torch.where(ok, num / (den + EPS), torch.zeros_like(num))


def rank_hinge(pred: torch.Tensor, label: torch.Tensor, mask: torch.Tensor,
               pairs_per_stock: int = 4, generator=None) -> torch.Tensor:
    """pairwise 排序 hinge，随机抽样股票对。

        L = mean over (i,j) of  max(0, -(p_i - p_j)(y_i - y_j))

    全配对是 O(N^2) ~ 1e6/天，抽 4N 对即可——梯度方向不变，开销降两个量级。
    """
    B, N = pred.shape
    K = max(pairs_per_stock * N, 1)
    dev = pred.device
    i = torch.randint(0, N, (B, K), device=dev, generator=generator)
    j = torch.randint(0, N, (B, K), device=dev, generator=generator)

    valid = mask.gather(1, i) & mask.gather(1, j) & (i != j)
    dp = pred.gather(1, i) - pred.gather(1, j)
    dy = label.gather(1, i) - label.gather(1, j)
    h = torch.relu(-dp * dy) * valid.float()
    return h.sum() / valid.float().sum().clamp(min=1.0)


def objective(pred, label, mask, lam_rank: float = 0.1, pairs_per_stock: int = 4,
              kind: str = "ic"):
    """返回 (loss, 诊断字典)。

    kind="ic"  主损失是截面 IC（默认）
    kind="mse" 纯 MSE，用来验证 Chen-Pelger-Zhu 的论断——他们发现同样的网络
               换成 MSE 预测收益，样本外夏普从 2.6 掉到 1.5。这一档是对照，
               不是候选方案。
    """
    if kind == "mse":
        m = mask.float()
        se = ((pred - label) ** 2 * m).sum() / m.sum().clamp(min=1.0)
        ic = cross_sectional_ic(pred, label, mask)
        n_day = (mask.sum(-1) >= 2).float().sum().clamp(min=1.0)
        return se, {"ic": (ic.sum() / n_day).detach(),
                    "rank_hinge": torch.zeros((), device=pred.device),
                    "ic_std": ic.detach().std(), "n_day": int(n_day)}

    ic = cross_sectional_ic(pred, label, mask)
    n_day = (mask.sum(-1) >= 2).float().sum().clamp(min=1.0)
    ic_mean = ic.sum() / n_day

    loss = -ic_mean
    rk = torch.zeros((), device=pred.device)
    if lam_rank > 0:
        rk = rank_hinge(pred, label, mask, pairs_per_stock)
        loss = loss + lam_rank * rk

    return loss, {"ic": ic_mean.detach(), "rank_hinge": rk.detach(),
                  "ic_std": ic.detach().std(), "n_day": int(n_day)}


# ---------------------------------------------------------------- 评估指标
@torch.no_grad()
def ic_series(pred, label, mask) -> torch.Tensor:
    """逐日 IC 序列，用于算 ICIR。"""
    return cross_sectional_ic(pred, label, mask)


@torch.no_grad()
def rank_ic_series(pred, label, mask) -> torch.Tensor:
    """逐日 RankIC（Spearman）：先在 mask 内做截面 rank，再算 Pearson。"""
    out = []
    for b in range(pred.shape[0]):
        m = mask[b]
        if m.sum() < 2:
            out.append(torch.zeros((), device=pred.device))
            continue
        p = _masked_rank(pred[b], m)
        l = _masked_rank(label[b], m)
        out.append(cross_sectional_ic(p[None], l[None], m[None])[0])
    return torch.stack(out)


def _masked_rank(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """只在 m 为真的位置上排名，映到 [-1, 1]；其余置 0。"""
    v = x[m]
    order = v.argsort()
    r = torch.empty_like(order, dtype=torch.float32)
    r[order] = torch.arange(len(v), device=x.device, dtype=torch.float32)
    r = 2.0 * (r / max(len(v) - 1, 1)) - 1.0
    out = torch.zeros_like(x)
    out[m] = r
    return out


@torch.no_grad()
def summarize(pred, label, mask) -> dict:
    """一次算齐 IC / RankIC / ICIR / RankICIR。"""
    ic = ic_series(pred, label, mask)
    ric = rank_ic_series(pred, label, mask)
    keep = (mask.sum(-1) >= 2)
    ic, ric = ic[keep], ric[keep]
    if ic.numel() == 0:
        return {k: float("nan") for k in ("ic", "rank_ic", "icir", "rank_icir", "n_day")}
    return {
        "ic": float(ic.mean()), "rank_ic": float(ric.mean()),
        "icir": float(ic.mean() / (ic.std() + EPS)),
        "rank_icir": float(ric.mean() / (ric.std() + EPS)),
        "n_day": int(ic.numel()),
    }


# ================================================================ 信号平滑与换手
@torch.no_grad()
def smooth_signal(pred: torch.Tensor, mask: torch.Tensor, lam: float) -> torch.Tensor:
    """日频部分调仓：p_t = lam * p_{t-1} + (1-lam) * p_new_t。

    展开后 k 天前信号的权重是 (1-lam)*lam^k，**平均信息年龄 = lam/(1-lam)**。
    所以 h 日预测期对应 lam ≈ h/(h+1)：h=5 -> lam≈0.833，实用上取 0.8（年龄 4 天）。

    两个作用：
    1. 消除调仓日择时运气——日频部分调仓等价于同时持有 h 个错开一天的组合
       （Jegadeesh-Titman 的 overlapping portfolios）；
    2. EWMA 是低通滤波，压掉预测里的高频噪音。

    pred/mask: (B, N)，B 必须按时间**升序**。不可交易的位置不参与更新，沿用上期值。
    """
    out = torch.zeros_like(pred)
    prev = torch.zeros_like(pred[0])
    for t in range(pred.shape[0]):
        m = mask[t].float()
        cur = lam * prev + (1 - lam) * pred[t]
        # 当日不可交易的股票保持上期信号（本来也调不动）
        cur = torch.where(mask[t], cur, prev)
        out[t] = cur * m
        prev = cur
    return out


@torch.no_grad()
def turnover(pred: torch.Tensor, mask: torch.Tensor) -> float:
    """信号层面的日均绝对变动，作为换手的代理。"""
    d = (pred[1:] - pred[:-1]).abs() * (mask[1:] & mask[:-1]).float()
    n = (mask[1:] & mask[:-1]).float().sum().clamp(min=1.0)
    return float(d.sum() / n)


def newey_west_se(x: torch.Tensor, lag: int) -> float:
    """Newey-West 标准误。

    h 日标签在相邻日之间有 h-1 天重叠，IC 序列强自相关，
    直接用 std/sqrt(T) 会**低估**标准误、让 ICIR 与 t 值偏乐观。
    lag 取 h-1。
    """
    v = (x - x.mean()).double()
    T = v.numel()
    if T < 3:
        return float("nan")
    s = float((v * v).mean())
    for k in range(1, min(lag, T - 1) + 1):
        w = 1.0 - k / (lag + 1)
        s += 2.0 * w * float((v[k:] * v[:-k]).mean())
    s = max(s, 1e-18)
    return float((s / T) ** 0.5)


@torch.no_grad()
def summarize_full(pred, label, mask, h: int = 5,
                   lams=(0.0, 0.5, 0.8, 0.9)) -> dict:
    """完整评估：原始 IC/RankIC + NW 调整的显著性 + 各 lam 下的换手匹配 IC。"""
    base = summarize(pred, label, mask)
    ic = ic_series(pred, label, mask)
    keep = (mask.sum(-1) >= 2)
    icv = ic[keep]
    se = newey_west_se(icv, max(h - 1, 1))
    base["icir_nw"] = float(icv.mean() / se) / (len(icv) ** 0.5) if se == se else float("nan")
    base["t_ic_nw"] = float(icv.mean() / se) if se == se else float("nan")
    base["turnover_raw"] = turnover(pred, mask)

    base["lam_sweep"] = {}
    for lam in lams:
        p = smooth_signal(pred, mask, lam) if lam > 0 else pred
        s = summarize(p, label, mask)
        base["lam_sweep"][f"lam{lam}"] = {
            "rank_ic": s["rank_ic"], "ic": s["ic"],
            "turnover": turnover(p, mask),
            "mean_info_age": lam / (1 - lam) if lam < 1 else float("inf"),
        }
    return base
