"""标签：未来 h 日**残差收益**。

为什么要残差化
--------------
原始收益里绝大部分是市场 beta 和风格暴露。做指增是跟指数比的，beta 是免费的，不是 alpha。
直接拿原始收益当标签，模型的容量会被浪费在学"大盘涨了""小盘占优"上。
剥掉之后模型只需要去找剩下的东西。

做法：**逐日截面正交化**
------------------------
对每个交易日 t，在当日截面上做加权最小二乘

    r_{i, t->t+h} = b0 + b_size * lnmv_{i,t} + sum_k b_k * IND_{i,k,t} + eps_{i,t}

取 eps 作为标签。等权市场收益被截距吸收，所以不需要单独放市场项。

关于前视
--------
这里用**当日**截面系数，不构成前视：我们不是在预测，是在定义标签，
y 和 x 同期，做的是截面正交化。模型训练时只看到残差标签，
推理时根本不需要标签。

但要注意两点：
1. 若改成时序回归估 beta（比如滚动 252 日估个股对市场的 beta），
   那个 beta **必须**来自 t 之前的窗口，两种做法不能混。
2. 残差 IC 高不等于组合收益高——组合最终赚的是**原始**收益。
   所以评估必须同时报因子层面（IC）和组合层面（超额收益）两套指标。

不可交易样本
------------
若 t 日收盘无法建仓，该样本标签直接作废（不进损失）。依据是 A 股上
"涨跌停价污染"的实证：模型会去学预测涨停，IC 好看但那部分收益买不进。
这里用 tradable_strict（比 tradable 多排除收盘涨跌停）。
"""
from __future__ import annotations
import sys, json, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config as C
import splits as S

LAB = C.ROOT / "data" / "labels"
LAB.mkdir(parents=True, exist_ok=True)
PQ = dict(compression="zstd", compression_level=3)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def forward_return(close: pd.DataFrame, h: int) -> pd.DataFrame:
    """未来 h 个交易日的收益。

    用 ffill 后的价格取 t+h，避免 t+h 恰好停牌时整段样本被丢掉；
    但起点 t 必须是真实交易日（由 mask 保证）。
    """
    px = close.ffill()
    fwd = (px.shift(-h) / px - 1.0)
    return fwd.where(close.notna()).astype("float32")


def residualize_daily(fwd: np.ndarray, lnmv: np.ndarray, ind: np.ndarray,
                      valid: np.ndarray, n_ind: int) -> np.ndarray:
    """逐日截面正交化。返回与 fwd 同形状的残差，无效格子为 NaN。

    自变量：[常数, lnmv(标准化), 行业哑变量]
    行业哑变量用一级（38 类），二级 179 类在单日 ~1000 只上会过参数化。
    """
    T, N = fwd.shape
    out = np.full((T, N), np.nan, dtype="float32")
    for t in range(T):
        m = valid[t]
        if m.sum() < 100:                       # 截面太小，整天作废
            continue
        y = fwd[t, m].astype("float64")
        z = lnmv[t, m].astype("float64")
        z = (z - z.mean()) / (z.std() + 1e-9)   # 标准化，避免量纲主导
        g = ind[t, m]
        # 设计矩阵：截距 + 市值 + 行业 one-hot（丢掉一个避免共线）
        used = np.unique(g[g >= 0])
        if len(used) < 2:
            continue
        D = np.zeros((m.sum(), len(used) - 1), dtype="float64")
        for k, gid in enumerate(used[1:]):
            D[:, k] = (g == gid)
        X = np.column_stack([np.ones(len(y)), z, D])
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        out[t, m] = (y - X @ beta).astype("float32")
    return out


def to_rank(x: pd.DataFrame) -> pd.DataFrame:
    """日内截面 rank 映到 [-1, 1]。对极端值稳健，且和 RankIC 的口径一致。"""
    r = x.rank(axis=1, pct=True)
    return (2.0 * (r - 0.5)).astype("float32")


def to_zscore(x: pd.DataFrame) -> pd.DataFrame:
    mu = x.mean(axis=1)
    sd = x.std(axis=1)
    return x.sub(mu, axis=0).div(sd + 1e-9, axis=0).clip(-5, 5).astype("float32")


def main():
    t0 = time.time()
    log("载入面板 …")
    close = pd.read_parquet(C.OUT / "A_close.parquet")
    lnmv = pd.read_parquet(C.OUT / "B_lnmv.parquet")
    ind1 = pd.read_parquet(C.OUT / "B_ind1.parquet")
    member = pd.read_parquet(C.OUT / "B_member.parquet")
    strict = pd.read_parquet(C.OUT / "A_tradable_strict.parquet")

    idx, cols = close.index, close.columns
    n_ind = int(ind1.to_numpy().max()) + 1

    # 建仓日必须可交易，且是当日成分股，且市值/行业已知
    base_ok = (strict.to_numpy() & member.to_numpy()
               & np.isfinite(lnmv.to_numpy()) & (ind1.to_numpy() >= 0))
    # 只在"当日有成分股"的日子上统计——2003~2014 中证1000 尚未发布，
    # 把那些日子算进分母会让日均只数腰斩，看着像 bug 其实不是。
    live = member.to_numpy().sum(1) > 0
    log(f"可建仓格子 {int(base_ok.sum()):,} | 有成分股的 {int(live.sum())} 天里，"
        f"日均 {base_ok[live].sum(1).mean():.0f} / {member.to_numpy()[live].sum(1).mean():.0f} 只 "
        f"({base_ok[live].sum(1).mean()/member.to_numpy()[live].sum(1).mean():.1%})")

    stats = {}
    for h in C.LABEL_HORIZONS:
        log(f"--- h = {h} ---")
        fwd = forward_return(close, h)
        # 未来 h 日内必须至少有一次真实行情，否则收益不可实现
        future_ok = close.notna().rolling(h, min_periods=1).sum().shift(-h) > 0
        ok = base_ok & np.isfinite(fwd.to_numpy()) & future_ok.fillna(False).to_numpy()

        resid = residualize_daily(fwd.to_numpy("float32"), lnmv.to_numpy("float32"),
                                  ind1.to_numpy("int16"), ok, n_ind)
        R = pd.DataFrame(resid, index=idx, columns=cols)
        lab = to_rank(R) if C.LABEL_TRANSFORM == "rank" else to_zscore(R)
        lab = lab.where(np.isfinite(resid))
        mask = pd.DataFrame(ok & np.isfinite(resid), index=idx, columns=cols)

        R.to_parquet(LAB / f"resid_h{h}.parquet", **PQ)
        lab.to_parquet(LAB / f"label_h{h}.parquet", **PQ)
        mask.to_parquet(LAB / f"mask_h{h}.parquet", **PQ)

        v = resid[np.isfinite(resid)]
        raw = fwd.to_numpy()[ok]
        stats[f"h{h}"] = {
            "n_valid": int(mask.to_numpy().sum()),
            "mean_n_per_day": float(mask.to_numpy()[live].sum(1).mean()),
            "resid_std": float(v.std()), "raw_std": float(np.nanstd(raw)),
            "var_explained": float(1 - v.var() / np.nanvar(raw)),
        }
        log(f"  有效 {stats[f'h{h}']['n_valid']:,} 格，成分日日均 "
            f"{stats[f'h{h}']['mean_n_per_day']:.0f} 只 | 原始 std {np.nanstd(raw):.4f} "
            f"-> 残差 std {v.std():.4f}（剥掉 {stats[f'h{h}']['var_explained']:.1%} 的方差）")

    (C.META / "labels_stats.json").write_text(
        json.dumps({"resid_mode": C.RESID_MODE, "transform": C.LABEL_TRANSFORM,
                    "horizons": list(C.LABEL_HORIZONS), "stats": stats},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"完成，用时 {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
