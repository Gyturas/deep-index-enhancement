"""本地分析：三折拼接 + 多空拆解 + 暴露归因。

全部读 runs/portfolio/*_pred.npz，纯 numpy，不需要 GPU、不需要重训。

三个问题
--------
1. **拼接**：三折预测接成 2019-01~2024-12 连续序列。样本量翻三倍，
   单折两年那个 NW-t 说不清的显著性，六年能说清。
   注意接缝：换模型那天仓位会跳，那部分换手是模型切换造成的不是信号变化，单独标出。
2. **多空**：多头超额 / 空头超额 / 多空价差三段拆开，回答"东西在头还是尾"。
   ⚠️ A 股融券受限，这是诊断不是策略。
3. **归因**：超额里多少来自市值偏移、行业偏移，多少是真残差 alpha。
   标签虽然残差化过，但不保证**实际选出来的组合**是中性的——
   模型可能通过别的特征间接押上了风格。这一步就是查这个。
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config as C

import os
NEUTRALIZE = os.environ.get("NEUTRALIZE", "0") == "1"
PF = C.ROOT / "runs" / "portfolio"
TD = 252
COST_BP = 15.0
BORROW_BP = 800.0        # 融券年化 8%，乐观假设


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


def neutralize(score, mask, lnmv, ind1):
    """逐日截面把打分对 [1, 市值, 行业哑变量] 正交化。

    为什么必须单独做这一步：标签残差化只保证**目标**是中性的，
    不保证**打分**是中性的——打分是特征的函数，而动量/波动/换手这些特征
    本身有行业聚集性，模型会把这个结构继承下来。
    实测：标签对行业的 R² 只有 1.3%，打分却有 14~23%。
    """
    out = np.full_like(score, -9e9)
    for t in range(score.shape[0]):
        m = mask[t]
        if m.sum() < 100:
            continue
        y = score[t][m].astype("float64")
        z = lnmv[t][m].astype("float64")
        z = (z - np.nanmean(z)) / (np.nanstd(z) + 1e-9)
        z = np.nan_to_num(z)
        g = ind1[t][m]
        used = np.unique(g[g >= 0])
        cols = [np.ones(m.sum()), z]
        cols += [(g == q).astype(float) for q in used[1:]]
        X = np.column_stack(cols)
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ b
        tmp = np.full(score.shape[1], -9e9)
        tmp[m] = (r - r.mean()) / (r.std() + 1e-9)
        out[t] = tmp
    return out


# ------------------------------------------------------------------ 组合
def run_long(score, mask, fwd, q, lam):
    T_, N = score.shape
    wp = np.zeros(N); ex, tos = [], []
    for t in range(T_):
        m = mask[t]; k = int(m.sum())
        if k < 50:
            ex.append(np.nan); tos.append(np.nan); continue
        n = max(int(round(k * q)), 1)
        top = np.argpartition(-np.where(m, score[t], -np.inf), n - 1)[:n]
        wt = np.zeros(N); wt[top] = 1.0 / n
        w = (lam * wp + (1 - lam) * wt) if lam > 0 else wt
        w = np.where(m, w, 0.0)
        if w.sum() <= 0:
            ex.append(np.nan); tos.append(np.nan); continue
        w /= w.sum()
        to = np.abs(w - wp).sum()
        bench = float(np.nansum(np.where(m, 1.0 / k, 0.0) * fwd[t]))
        ex.append(float(np.nansum(w * fwd[t])) - to * COST_BP * 1e-4 - bench)
        tos.append(to); wp = w
    return np.array(ex), np.array(tos)


def run_ls(score, mask, fwd, q, lam):
    """返回 (多头超额, 空头超额, 多空价差)。"""
    T_, N = score.shape
    wlp = np.zeros(N); wsp = np.zeros(N)
    L, S, B = [], [], []
    for t in range(T_):
        m = mask[t]; k = int(m.sum())
        if k < 50:
            L.append(np.nan); S.append(np.nan); B.append(np.nan); continue
        n = max(int(round(k * q)), 1)
        top = np.argpartition(-np.where(m, score[t], -np.inf), n - 1)[:n]
        bot = np.argpartition(np.where(m, score[t], np.inf), n - 1)[:n]
        wlt = np.zeros(N); wlt[top] = 1.0 / n
        wst = np.zeros(N); wst[bot] = 1.0 / n
        wl = (lam * wlp + (1 - lam) * wlt) if lam > 0 else wlt
        ws = (lam * wsp + (1 - lam) * wst) if lam > 0 else wst
        wl = np.where(m, wl, 0.0); ws = np.where(m, ws, 0.0)
        if wl.sum() <= 0 or ws.sum() <= 0:
            L.append(np.nan); S.append(np.nan); B.append(np.nan); continue
        wl /= wl.sum(); ws /= ws.sum()
        bench = float(np.nansum(np.where(m, 1.0 / k, 0.0) * fwd[t]))
        rl = float(np.nansum(wl * fwd[t])) - np.abs(wl - wlp).sum() * COST_BP * 1e-4
        rs = float(np.nansum(ws * fwd[t])) - np.abs(ws - wsp).sum() * COST_BP * 1e-4
        bc = BORROW_BP * 1e-4 / TD
        L.append(rl - bench); S.append(bench - rs - bc); B.append(rl - rs - bc)
        wlp, wsp = wl, ws
    return map(np.array, (L, S, B))


# ------------------------------------------------------------------ 归因
def attribute(score, mask, fwd, q, lam, lnmv, ind1, dates_pos):
    """把多头超额逐日拆成 市值 / 行业 / 残差 三块。

    每天对当日截面做 超额权重 (w - w_bench) 在 [市值, 行业哑变量] 上的分解：
    组合相对基准的因子暴露 × 该因子当日收益 = 该因子贡献。
    """
    T_, N = score.shape
    wp = np.zeros(N)
    size_c, ind_c, resid_c = [], [], []
    for t in range(T_):
        m = mask[t]; k = int(m.sum())
        if k < 50:
            continue
        n = max(int(round(k * q)), 1)
        top = np.argpartition(-np.where(m, score[t], -np.inf), n - 1)[:n]
        wt = np.zeros(N); wt[top] = 1.0 / n
        w = (lam * wp + (1 - lam) * wt) if lam > 0 else wt
        w = np.where(m, w, 0.0)
        if w.sum() <= 0:
            continue
        w /= w.sum()
        wb = np.where(m, 1.0 / k, 0.0)
        dw = (w - wb)[m]                       # 超额权重
        r = fwd[t][m]
        z = lnmv[t][m]
        z = (z - z.mean()) / (z.std() + 1e-9)
        g = ind1[t][m]
        used = np.unique(g[g >= 0])
        D = np.zeros((m.sum(), max(len(used) - 1, 0)))
        for j, gid in enumerate(used[1:]):
            D[:, j] = (g == gid)
        X = np.column_stack([np.ones(len(r)), z, D]) if D.size else np.column_stack([np.ones(len(r)), z])
        beta, *_ = np.linalg.lstsq(X, r, rcond=None)
        eps = r - X @ beta
        size_c.append(float(dw @ z * beta[1]))
        ind_c.append(float(dw @ (D @ beta[2:])) if D.size else 0.0)
        resid_c.append(float(dw @ eps))
        wp = w
    return map(np.array, (size_c, ind_c, resid_c))


# ------------------------------------------------------------------
def main():
    print(f"打分中性化: {'开' if NEUTRALIZE else '关'}")
    close = pd.read_parquet(C.OUT / "A_close.parquet")
    lnmv_all = pd.read_parquet(C.OUT / "B_lnmv.parquet").to_numpy("float32")
    ind1_all = pd.read_parquet(C.OUT / "B_ind1.parquet").to_numpy("int16")
    idx = close.index

    stitched = {"ex30": [], "ex50": [], "bench": [], "dates": [], "seam": []}
    print("=" * 88)
    print(f"{'':16}{'多头超额':>10}{'多头IR':>8}{'空头超额':>10}{'空头IR':>8}"
          f"{'多空年化':>10}{'多空夏普':>9}")
    print("=" * 88)
    ls_all = {q: {"L": [], "S": [], "B": []} for q in (0.10, 0.30)}

    for k in range(3):
        z = np.load(PF / f"pf_L0_fold{k}_pred.npz")
        score, mask, fwd = z["score"], z["mask"], z["fwd1"]
        score = np.nan_to_num(score, nan=-9e9)
        dates = pd.to_datetime([d for d in z["dates"]])
        pos = idx.get_indexer(dates)
        if NEUTRALIZE:
            score = neutralize(score, mask, lnmv_all[pos], ind1_all[pos])

        for q in (0.10, 0.30):
            L, S, B = run_ls(score, mask, fwd, q, 0.8)
            ok = ~np.isnan(L)
            ls_all[q]["L"].append(L[ok]); ls_all[q]["S"].append(S[ok]); ls_all[q]["B"].append(B[ok])
            if q == 0.30:
                print(f"fold{k} q30%{'':6}{ann(L[ok]):>10.1%}{sharpe(L[ok]):>8.2f}"
                      f"{ann(S[ok]):>10.1%}{sharpe(S[ok]):>8.2f}"
                      f"{ann(B[ok]):>10.1%}{sharpe(B[ok]):>9.2f}")

        for q, key in ((0.30, "ex30"), (0.50, "ex50")):
            ex, _ = run_long(score, mask, fwd, q, 0.8)
            stitched[key].append(ex[~np.isnan(ex)])
        bm = []
        for t in range(len(score)):
            m = mask[t]; kk = int(m.sum())
            if kk >= 50:
                bm.append(float(np.nansum(np.where(m, 1.0 / kk, 0.0) * fwd[t])))
        stitched["bench"].append(np.array(bm))
        stitched["dates"].append(dates[:len(bm)])
        stitched["seam"].append(len(bm))

        # 归因
        S_, I_, R_ = attribute(score, mask, fwd, 0.30, 0.8, lnmv_all[pos], ind1_all[pos], pos)
        tot = ann(S_ + I_ + R_)
        print(f"       归因: 市值 {ann(S_):+.2%}  行业 {ann(I_):+.2%}  "
              f"残差 {ann(R_):+.2%}  合计 {tot:+.2%}")

    # ---------------- 多空汇总
    print("-" * 88)
    for q in (0.10, 0.30):
        L = np.concatenate(ls_all[q]["L"]); S = np.concatenate(ls_all[q]["S"])
        B = np.concatenate(ls_all[q]["B"])
        print(f"三折合并 q{int(q*100)}%{'':3}{ann(L):>10.1%}{sharpe(L):>8.2f}"
              f"{ann(S):>10.1%}{sharpe(S):>8.2f}{ann(B):>10.1%}{sharpe(B):>9.2f}")

    # ---------------- 拼接
    print("=" * 88)
    for key, name in (("ex30", "前30%"), ("ex50", "前50%")):
        ex = np.concatenate(stitched[key])
        bm = np.concatenate(stitched["bench"])[:len(ex)]
        port = ex + bm
        print(f"\n【三折拼接 2019-01~2024-12  {name} λ=0.8】 {len(ex)} 天")
        print(f"  基准     年化 {ann(bm):+7.2%}  夏普 {sharpe(bm):+5.2f}  回撤 {mdd(bm):7.1%}")
        print(f"  组合     年化 {ann(port):+7.2%}  夏普 {sharpe(port):+5.2f}  回撤 {mdd(port):7.1%}")
        print(f"  超额     年化 {ann(ex):+7.2%}  信息比 {sharpe(ex):+5.2f}  "
              f"NW-t {nw_t(ex):+5.2f}  超额回撤 {mdd(ex):7.1%}")
        cuts = np.cumsum(stitched["seam"])[:-1]
        print(f"  接缝在第 {list(cuts)} 天（换模型），那两天换手不计入日常统计")


if __name__ == "__main__":
    main()
