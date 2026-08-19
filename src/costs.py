"""A 股交易成本模型。

论文的成本是"tick size × 流动性系数"的期货口径，A 股要重写。四项：

    c = 印花税(卖出单边) + 佣金 + 半价差 + 冲击成本(平方根律)

**印花税单边这一非对称性必须保留**——它让"卖出"比"买入"贵，
会实实在在改变最优换手方向；对称化会让模型学到错的换手结构。

冲击项用平方根律 [Almgren et al. 2005]：
    impact = kappa * sqrt( 成交额 / ADV )
ADV 用 20 日均成交额（面板 B_turnover20 的量纲是换手率，这里直接从 amount 算）。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------- 费率
STAMP_SELL = 5e-4          # 印花税，卖出单边 0.05%（2023-08-28 起由 0.1% 减半）
STAMP_SELL_PRE = 1e-3      # 2023-08-28 之前 0.1%
STAMP_CUT_DATE = "2023-08-28"
COMMISSION = 2.5e-4        # 佣金，双边（含规费；机构实际更低，作为保守值）
MIN_HALF_SPREAD = 5e-4     # 半价差下限：A 股最小变动 0.01 元，低价股相对价差更大
KAPPA = 0.1                # 冲击系数，平方根律前置常数


def half_spread(close: pd.DataFrame) -> pd.DataFrame:
    """半价差 ≈ 0.5 × tick / price，并施加下限。

    注意：close 是**后复权**价，不能直接用来推 tick 相对大小。
    这里用后复权价的**截面相对水平**做近似不成立，因此改用一个保守常数下限，
    真实价差需要 Level-1 快照数据才能标定——见计划书"已知缺口"。
    """
    return pd.DataFrame(MIN_HALF_SPREAD, index=close.index, columns=close.columns,
                        dtype="float32")


def stamp_rate(idx: pd.DatetimeIndex) -> pd.Series:
    """逐日印花税率（卖出单边）。"""
    r = pd.Series(STAMP_SELL_PRE, index=idx, dtype="float32")
    r[idx >= pd.Timestamp(STAMP_CUT_DATE)] = STAMP_SELL
    return r


def build_cost_panels(close: pd.DataFrame, amount: pd.DataFrame,
                      adv_win: int = 20) -> dict[str, pd.DataFrame]:
    """返回训练所需的成本相关面板。

    - ``linear``: 与方向无关的比例成本（佣金 + 半价差）
    - ``stamp``:  卖出方向额外成本（印花税），逐日广播
    - ``adv``:    20 日平均成交额（元），用于冲击项
    """
    adv = amount.rolling(adv_win, min_periods=5).mean().astype("float32")
    linear = (COMMISSION + half_spread(close)).astype("float32")
    stamp = pd.DataFrame(np.repeat(stamp_rate(close.index).to_numpy()[:, None],
                                   close.shape[1], axis=1),
                         index=close.index, columns=close.columns, dtype="float32")
    return {"linear": linear, "stamp": stamp, "adv": adv}


def turnover_cost(w: torch.Tensor, w_prev: torch.Tensor, linear: torch.Tensor,
                  stamp: torch.Tensor, adv: torch.Tensor, gmv: float,
                  kappa: float = KAPPA) -> torch.Tensor:
    """逐格换手成本（比例，作用在 |Δw| 上）。

    参数
    ----
    w, w_prev : (..., N) 目标与上期名义权重
    linear, stamp, adv : (..., N) 成本面板
    gmv : 组合总名义规模（元），用于把权重变动折成成交额

    返回
    ----
    (..., N) 的成本贡献，已乘上 |Δw|。
    """
    dw = w - w_prev
    absdw = dw.abs()
    sells = torch.clamp(-dw, min=0.0)                      # 卖出部分才交印花税

    # 冲击项必须作用在**真实成交金额**上。w 是波动率放大后的名义敞口
    # (w = p / sigma，sigma≈0.025 时量级约 40)，直接 absdw * gmv 会得到
    # 几十亿的假成交额，冲击率触顶后成本恒为天量，净收益变成一条确定的大负数——
    # 这正是首版冒烟测试里初始夏普 -43.67 的来源。
    # 正确做法：先把 w 归一化成组合权重占比，折成金额，再算冲击**率**。
    gross_exp = w.abs().sum(-1, keepdim=True).clamp(min=1e-6)
    notional = (absdw / gross_exp) * gmv
    # sqrt 在 0 处导数无穷，masked / 未换手的格子恰好为 0，会直接产出 NaN 梯度。
    # eps 必须加在**根号内**。
    impact = kappa * torch.sqrt(notional / (adv + 1.0) + 1e-12)
    impact = torch.clamp(impact, max=0.02)                 # 冲击率封顶 2%

    return absdw * (linear + impact) + sells * stamp
