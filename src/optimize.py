"""指数增强：打分 -> alpha -> 带约束优化 -> 回测。

信号层（Grinold-Kahn 标准做法）
--------------------------------
    z = Phi^-1( (rank(s) - 0.5) / N )        正态分位数变换
    alpha = IC_hat * sigma_spec * z          转成收益量纲

* 内层不做 winsor——排名对单调变换不变，先 winsor 再排名等于没做。
  N≈900 时 Phi^-1 天然落在 ±3.1 以内，外层 clip 只是保险。
* `IC_hat` 用**滚动 252 日**的 RankIC 均值，且往下收缩（乘 0.6）。
  宁可低估：低估只让优化器保守，高估会让它过度集中。
* `sigma_spec` 取风险模型的特异波动——alpha 是"预期残差收益"，
  量纲必须和残差对齐，不能用总波动。

优化
----
    max  alpha'w - lambda/2 * w' Sigma w
    s.t. sum(w) = 1,  0 <= w_i <= cap
         |行业暴露 - 基准| <= 1%
         |Size 暴露 - 基准| <= 0.1 (标准化单位)
         sum|w - w_prev| <= 换手上限
"""
from __future__ import annotations
import sys, json, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent))
import config as C
import risk as RK

PF = C.ROOT / "runs" / "portfolio"
RD = C.ROOT / "data" / "risk"
TD = 252
COST_BP = 15.0


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def ann(x):
    x = np.asarray(x, float)
    return float((1 + x).prod() ** (TD / len(x)) - 1) if len(x) else np.nan


def sharpe(x):
    x = np.asarray(x, float)
    return float(x.mean() / (x.std(ddof=1) + 1e-12) * np.sqrt(TD))


def nw_t(x, lag=4):
    v = np.asarray(x, float) - np.mean(x)
    T = len(v)
    s = (v * v).mean()
    for k in range(1, min(lag, T - 1) + 1):
        s += 2 * (1 - k / (lag + 1)) * (v[k:] * v[:-k]).mean()
    return float(np.mean(x) / np.sqrt(max(s, 1e-18) / T))


def mdd(x):
    nav = (1 + np.asarray(x, float)).cumprod()
    return float((nav / np.maximum.accumulate(nav) - 1).min())


# ------------------------------------------------------------------ 信号层
def to_normal_score(s: np.ndarray, m: np.ndarray) -> np.ndarray:
    """截面 rank -> 正态分位数。无效位置返回 0。"""
    out = np.zeros_like(s, dtype="float64")
    v = s[m]
    n = len(v)
    if n < 10:
        return out
    r = np.empty(n)
    r[np.argsort(v)] = np.arange(n)
    out[m] = np.clip(norm.ppf((r + 0.5) / n), -3.0, 3.0)
    return out


def daily_rank_ic(scores, masks, labels) -> np.ndarray:
    """逐日 RankIC 序列，一次算完备用。"""
    out = np.full(len(scores), np.nan)
    for i in range(len(scores)):
        m = masks[i]
        if m.sum() < 50:
            continue
        a, b = scores[i][m], labels[i][m]
        if not np.isfinite(b).any():
            continue
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 50:
            continue
        a, b = a[ok], b[ok]
        ra = np.empty(len(a)); ra[np.argsort(a)] = np.arange(len(a))
        rb = np.empty(len(b)); rb[np.argsort(b)] = np.arange(len(b))
        out[i] = np.corrcoef(ra, rb)[0, 1]
    return out


def rolling_ic(ic_series, i, h, win=252, shrink=0.6, prior=0.03, min_obs=60):
    """截至第 i 天可用的滚动 RankIC 均值，往下收缩。

    **前视陷阱**：第 s 天的 RankIC 要拿 score[s] 和 label[s] 比，
    而 label[s] 是 s 往后 h 日的收益——要到 s+h 才知道。
    所以站在第 i 天，最新一个**已知**的 IC 观测是第 i-h 天，窗口必须截到那里。
    少截这 h 天，等于把未来 h 日的收益偷偷用进了 alpha 的量纲里。

    观测不足时退回保守先验，且先验也过同一个收缩系数——宁可低估：
    低估只让优化器保守，高估会让它过度集中。
    """
    hi = max(i - h, 0)
    lo = max(hi - win, 0)
    v = ic_series[lo:hi]
    v = v[np.isfinite(v)]
    if len(v) < min_obs:
        return prior * shrink
    return float(v.mean()) * shrink


# ------------------------------------------------------------------ 优化
def solve(alpha, Sigma_fn, w_prev, w_bench, ind, size, dev_cap,
          lam=10.0, ind_tol=0.01, size_tol=0.10, kappa=0.0015, te_cap=None):
    """带约束二次优化。

    **换手做成惩罚项而不是硬约束。**换手代表成本，本来就属于目标函数；
    做成硬约束会和个股上限/行业中性冲突——上期权重按今日可交易集重新归一后
    常常已经越界，而换手预算不够把它拉回来，于是整个问题 infeasible。
    实测硬约束版本 486 天里 346 天无解。

    改成惩罚后 w = w_bench 永远可行（等权满足所有约束），求解器不会再无解。
    """
    import cvxpy as cp
    n = len(alpha)
    w = cp.Variable(n, nonneg=True)
    Xf, F, dspec = Sigma_fn
    risk = cp.quad_form(Xf.T @ (w - w_bench), cp.psd_wrap(F)) \
        + cp.sum(cp.multiply(dspec, cp.square(w - w_bench)))
    # 个股约束改成**相对基准的偏离上限**（行业标准口径）。
    # 原来的「3 倍等权」是均匀的 0.34%，对市值加权基准没有意义——
    # 指数权重从 0.007% 到 0.49% 差 70 倍。
    cons = [cp.sum(w) == 1, w <= w_bench + dev_cap]
    if te_cap is not None:
        cons += [risk <= (te_cap ** 2) / 252.0]      # 年化跟踪误差上限
    # 行业中性
    for g in np.unique(ind):
        sel = (ind == g)
        cons += [cp.abs(cp.sum(w[sel]) - w_bench[sel].sum()) <= ind_tol]
    # 市值中性
    cons += [cp.abs(size @ (w - w_bench)) <= size_tol]
    # 换手：惩罚项，系数就是单位换手的真实成本
    prob = cp.Problem(cp.Maximize(alpha @ w - lam / 2 * risk
                                  - kappa * cp.norm1(w - w_prev)), cons)
    why = ""
    for slv in (cp.CLARABEL, cp.OSQP):
        try:
            prob.solve(solver=slv, verbose=False)
            if w.value is not None and np.isfinite(w.value).all():
                x = np.maximum(w.value, 0)
                if x.sum() > 1e-9:
                    return x / x.sum(), True, prob.status
            why = prob.status or "no-value"
        except Exception as e:
            why = f"{type(e).__name__}:{str(e)[:40]}"
    k = max(int(round(n * 0.30)), 1)
    top = np.argpartition(-alpha, k - 1)[:k]
    x = np.zeros(n); x[top] = 1.0 / k
    return x, False, why


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--ind-tol", type=float, default=0.01)
    ap.add_argument("--size-tol", type=float, default=0.10)
    ap.add_argument("--kappa", type=float, default=0.0015,
                    help="换手惩罚系数，等于单位换手的真实成本（15bp）")
    ap.add_argument("--h", type=int, default=5, help="标签预测期，用于 alpha 量纲折算")
    ap.add_argument("--dev-cap", type=float, default=0.01,
                    help="个股相对基准的权重偏离上限（绝对，默认 1 个百分点）")
    ap.add_argument("--te-cap", type=float, default=None,
                    help="年化跟踪误差上限，如 0.05")
    ap.add_argument("--folds", default="0,1,2")
    ap.add_argument("--final", action="store_true",
                    help="在锁定模拟盘上回测（读 pf_final_final_pred.npz）")
    a = ap.parse_args()

    idx = pd.read_parquet(C.OUT / "A_close.parquet").index
    lnmv = pd.read_parquet(C.OUT / "B_lnmv.parquet").to_numpy("float32")
    ind1 = pd.read_parquet(C.OUT / "B_ind1.parquet").to_numpy("int16")
    fr = np.load(RD / "factor_returns.npz")
    fret, n_ind = fr["fret"], int(fr["n_ind"])
    spec = np.load(RD / "specific.npz")["spec"]
    beta = np.load(RD / "beta.npy")
    IWT = pd.read_parquet(C.OUT / "B_weight.parquet").to_numpy("float32")
    LAB = pd.read_parquet(C.ROOT / "data" / "labels" / f"label_h{5}.parquet").to_numpy("float32")

    all_ex, all_bm, all_dt, all_to = [], [], [], []
    tags = ["final"] if a.final else [f"L0_fold{x}" for x in a.folds.split(",")]
    for fk in tags:
        z = np.load(PF / f"pf_{fk}_{'final' if a.final else ''}_pred.npz"
                    if a.final else PF / f"pf_{fk}_pred.npz")
        score, mask, fwd = np.nan_to_num(z["score"], nan=-9e9), z["mask"], z["fwd1"]
        dates = pd.to_datetime([d for d in z["dates"]])
        pos = idx.get_indexer(dates)
        # 预测 npz 里没存 label，直接从标签面板按 (日期, 股票) 取
        lab_te = LAB[np.ix_(pos, np.arange(score.shape[1]))]
        ic_ser = daily_rank_ic(score, mask, lab_te)
        log(f"fold{fk} 逐日 RankIC: 均值 {np.nanmean(ic_ser):+.4f} "
            f"有效 {int(np.isfinite(ic_ser).sum())} 天")

        # w_prev 存在**全域**下标上。每日 cols 不同（成分调整、停牌进出），
        # 直接按位置沿用上一天的解，等于把 A 股票的权重安到 B 股票头上——
        # 换手约束和成本都会算在一个无意义的向量上，组合被锚死在垃圾解附近。
        w_prev_full = np.zeros(score.shape[1])
        ex, bm_r, n_ok, n_fb, fails, tos, ics = [], [], 0, 0, {}, [], []
        for i, t in enumerate(pos):
            m = mask[i] & np.isfinite(beta[t]) & np.isfinite(lnmv[t]) & (ind1[t] >= 0)
            k = int(m.sum())
            if k < 100:
                continue
            cols = np.where(m)[0]
            cap = np.exp(lnmv[t][cols].astype("float64"))
            sz = np.nan_to_num(RK.standardize(lnmv[t][cols], cap))
            bt = np.nan_to_num(RK.standardize(beta[t][cols], cap))
            g = ind1[t][cols]

            # 基准 = **中证1000 真实指数权重**（自由流通市值加权），不是等权。
            # 等权 2019-2024 比市值加权高 2.86%/年，那是小市值风格价差不是 alpha；
            # 且 standardize() 是「市值加权均值为 0」的 Barra 口径，
            # 用市值加权当锚，基准的 Size/Beta 暴露天然为 0，中性约束才自洽。
            wraw = np.nan_to_num(IWT[t][cols])
            if wraw.sum() <= 0:
                continue
            wb = wraw / wraw.sum()
            # 信号
            zs = to_normal_score(score[i], m)[cols]
            ic = rolling_ic(ic_ser, i, a.h)
            dv = RK.specific_var(spec, t, cols,
                                 size_bucket=np.clip((sz * 2 + 5).astype(int), 0, 9))
            # **量纲必须对齐**：IC 是对 h 日残差收益估的，特异方差 dv 是日频的。
            # 所以 h 日的特异波动是 sqrt(dv * h)，再除以 h 折回日频预期收益。
            alpha = ic * np.sqrt(dv * a.h) * zs / a.h

            F = RK.factor_cov(fret, t)
            if F is None:
                continue
            D = np.zeros((k, n_ind)); D[np.arange(k), g] = 1.0
            Xf = np.column_stack([np.ones(k), D, sz, bt])
            wp = w_prev_full[cols]
            wp = wp / wp.sum() if wp.sum() > 1e-9 else wb.copy()
            w, ok, why = solve(alpha, (Xf, F, dv), wp, wb, g, sz, a.dev_cap,
                               a.lam, a.ind_tol, a.size_tol, a.kappa, a.te_cap)
            n_ok += ok; n_fb += (not ok)
            if not ok:
                fails[why] = fails.get(why, 0) + 1

            r = fwd[i][cols]
            to = np.abs(w - wp).sum()
            tos.append(to); ics.append(ic)
            ex.append(float(w @ r) - to * COST_BP * 1e-4 - float(wb @ r))
            bm_r.append(float(wb @ r))
            # 权重随价格漂移——隔一天的真实起点是漂移后的权重，不是上期目标权重。
            # 不做这步会低估换手。
            drift = w * (1.0 + np.nan_to_num(r))
            drift = drift / drift.sum() if drift.sum() > 0 else w
            w_prev_full = np.zeros(score.shape[1])
            w_prev_full[cols] = drift

        ex, bm_r = np.array(ex), np.array(bm_r)
        all_ex.append(ex); all_bm.append(bm_r)
        all_dt.append(np.array([str(d.date()) for d in dates[:len(ex)]]))
        all_to.append(np.array(tos))
        log(f"{fk}  {len(ex)} 天 | 求解成功 {n_ok} 回退 {n_fb} | "
            f"超额年化 {ann(ex):+.2%}  IR {sharpe(ex):+.2f}  NW-t {nw_t(ex):+.2f}  "
            f"超额回撤 {mdd(ex):+.1%}")
        log(f"          日均单边换手 {np.mean(tos)/2:.2%}  年化双边 {np.mean(tos)*TD:.1f}x  "
            f"| 滚动IC 用到 [{np.min(ics):.4f}, {np.max(ics):.4f}] 均值 {np.mean(ics):.4f}")
        if fails:
            log(f"   失败原因: {dict(sorted(fails.items(), key=lambda kv:-kv[1])[:4])}")

    EX = np.concatenate(all_ex); BM = np.concatenate(all_bm)
    np.savez_compressed(PF / ("ie_series_final.npz" if a.final else "ie_series.npz"), ex=EX, bench=BM,
                        dates=np.concatenate(all_dt), turnover=np.concatenate(all_to),
                        seam=np.cumsum([len(x) for x in all_ex])[:-1])
    print("=" * 78)
    log(f"三折拼接 {len(EX)} 天")
    log(f"  基准 年化 {ann(BM):+.2%}  夏普 {sharpe(BM):+.2f}  回撤 {mdd(BM):+.1%}")
    log(f"  组合 年化 {ann(EX + BM):+.2%}  夏普 {sharpe(EX + BM):+.2f}  回撤 {mdd(EX + BM):+.1%}")
    log(f"  超额 年化 {ann(EX):+.2%}  IR {sharpe(EX):+.2f}  NW-t {nw_t(EX):+.2f}  "
        f"超额回撤 {mdd(EX):+.1%}")


if __name__ == "__main__":
    main()
