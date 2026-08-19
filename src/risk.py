"""窄版 Barra 式风险模型：Country + Industry(申万一级) + Size + Beta。

    Sigma = X F X' + Delta

* `X` 暴露矩阵 (N x K)，K = 1 + 38 + 2
* `F` 因子协方差 (K x K)，EWMA + Newey-West
* `Delta` 特异方差对角阵，EWMA + 向规模分组均值贝叶斯收缩

照做的 Barra 约定
-----------------
1. 描述子缩尾 ±3 倍标准差；标准化为**市值加权均值 0、等权标准差 1**
   （不是普通 z-score——这是 Barra 的约定，保证市值加权组合的因子暴露为 0）
2. 因子收益用**市值加权 WLS**，权重 sqrt(市值)
3. **行业因子约束为市值加权和为零**——不加这条，行业之和与 Country 完全共线，回归不可解
4. Beta 用 252 日窗口、63 日半衰的 EWMA 回归（不是普通 63 日 OLS）

有意简化掉的
------------
Eigenfactor 风险调整、Volatility Regime 调整、Bias test 校准。
这三个是 Barra 用来修正协方差系统性偏差的，实现复杂、对我们这个规模边际价值存疑。
若实测发现「预测跟踪误差」明显低于「实际跟踪误差」，再回头补 Eigenfactor。
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config as C

OUT = C.ROOT / "data" / "risk"
OUT.mkdir(parents=True, exist_ok=True)

BETA_WIN, BETA_HL = 252, 63       # Beta 回归窗口与半衰期
F_HL_VAR, F_HL_COR = 90, 480      # 因子协方差的方差/相关半衰期（Barra 惯例）
D_HL = 90                         # 特异方差半衰期
NW_LAG = 5


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def ewma_weights(n: int, halflife: float) -> np.ndarray:
    """最近的权重最大，和为 1。"""
    lam = 0.5 ** (1.0 / halflife)
    w = lam ** np.arange(n - 1, -1, -1)
    return w / w.sum()


# ------------------------------------------------------------------ 暴露
def standardize(x: np.ndarray, cap: np.ndarray) -> np.ndarray:
    """Barra 口径：市值加权均值为 0，等权标准差为 1，缩尾 ±3。

    x 与 cap 必须**已经是同一子集**（长度相同），不要一个传全量一个传子集。
    """
    out = np.full(len(x), np.nan, dtype="float64")
    v = np.isfinite(x)
    if v.sum() < 20:
        return out
    xv = x[v].astype("float64")
    w = cap[v] / cap[v].sum()
    mu = float(w @ xv)                       # 市值加权均值
    sd = float(np.std(xv - mu)) + 1e-12      # 等权标准差
    z = (xv - mu) / sd
    z = np.clip(z, -3.0, 3.0)
    # 缩尾后再对齐一次，保证市值加权均值仍为 0
    z = z - float(w @ z)
    out[v] = z
    return out


def rolling_beta(ret: np.ndarray, mkt: np.ndarray) -> np.ndarray:
    """252 日窗口、63 日半衰的 EWMA 对市场回归斜率。"""
    T, N = ret.shape
    w = ewma_weights(BETA_WIN, BETA_HL)
    out = np.full((T, N), np.nan, dtype="float32")
    r = np.nan_to_num(ret)
    ok = np.isfinite(ret)
    for t in range(BETA_WIN, T):
        sl = slice(t - BETA_WIN, t)
        m = mkt[sl]
        cnt = ok[sl].sum(0)
        wm = float(w @ m)
        dm = m - wm
        var = float(w @ (dm * dm)) + 1e-12
        cov = (w[:, None] * dm[:, None] * (r[sl] - (w @ r[sl])[None, :])).sum(0)
        b = cov / var
        out[t] = np.where(cnt >= BETA_WIN // 2, b, np.nan)
    return out


# ------------------------------------------------------------------ 主流程
def build(start: str = "2014-06-01"):
    t0 = time.time()
    log("载入面板 …")
    close = pd.read_parquet(C.OUT / "A_close.parquet")
    ret1 = pd.read_parquet(C.OUT / "A_ret1.parquet")
    lnmv = pd.read_parquet(C.OUT / "B_lnmv.parquet")
    ind1 = pd.read_parquet(C.OUT / "B_ind1.parquet")
    tradable = pd.read_parquet(C.OUT / "A_tradable.parquet")
    member = pd.read_parquet(C.OUT / "B_member.parquet")

    idx, codes = close.index, close.columns
    R = np.clip(np.nan_to_num(ret1.to_numpy("float32"), nan=0.0), -0.25, 0.25)
    LM = lnmv.to_numpy("float32")
    IND = ind1.to_numpy("int16")
    OKT = (tradable.to_numpy() & member.to_numpy() & np.isfinite(LM) & (IND >= 0))
    CAP = np.exp(np.nan_to_num(LM, nan=0.0)).astype("float64")

    # 市场收益：市值加权（Barra 的 Country 因子口径）
    mkt = np.zeros(len(idx))
    for t in range(len(idx)):
        m = OKT[t]
        if m.sum() >= 50:
            w = CAP[t][m] / CAP[t][m].sum()
            mkt[t] = float(w @ R[t][m])

    log("估计 Beta（252 日窗口 / 63 日半衰）…")
    BETA = rolling_beta(np.where(OKT, R, np.nan), mkt)

    t_start = idx.get_indexer([pd.Timestamp(start)])[0]
    t_start = max(t_start, BETA_WIN)
    n_ind = int(IND.max()) + 1
    K = 1 + n_ind + 2                             # Country + 行业 + Size + Beta
    log(f"因子数 K={K}（Country 1 + 行业 {n_ind} + Size 1 + Beta 1）")

    fret = np.full((len(idx), K), np.nan)          # 因子收益
    spec = np.full(R.shape, np.nan, dtype="float32")   # 特异收益
    expo = {}                                      # 逐日暴露，供组合优化用

    for t in range(t_start, len(idx) - 1):
        m = OKT[t] & np.isfinite(BETA[t])
        if m.sum() < 100:
            continue
        cap = CAP[t][m]
        sz = standardize(LM[t][m], cap)
        bt = standardize(BETA[t][m], cap)
        sz = np.nan_to_num(sz); bt = np.nan_to_num(bt)
        g = IND[t][m]

        n = m.sum()
        D = np.zeros((n, n_ind))
        D[np.arange(n), g] = 1.0
        used = D.sum(0) > 0
        X = np.column_stack([np.ones(n), D[:, used], sz, bt])
        Ku = X.shape[1]

        # 行业市值加权和为零的约束：把最后一个用到的行业替换掉
        # 采用等价做法——对行业列做加权中心化，避免与 Country 共线
        wcap = cap / cap.sum()
        Xi = X[:, 1:1 + used.sum()]
        X[:, 1:1 + used.sum()] = Xi - (wcap @ Xi)[None, :]

        y = R[t + 1][m].astype("float64")          # 次日收益
        sw = np.sqrt(cap); sw = sw / sw.sum() * n  # WLS 权重
        Xw = X * sw[:, None]; yw = y * sw
        f, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        e = y - X @ f

        row = np.full(K, np.nan)
        row[0] = f[0]
        row[1:1 + n_ind][used] = f[1:1 + used.sum()]
        row[-2], row[-1] = f[-2], f[-1]
        fret[t] = row
        spec[t + 1][m] = e.astype("float32")
        expo[t] = {"cols": np.where(m)[0], "size": sz.astype("float32"),
                   "beta": bt.astype("float32"), "ind": g.astype("int16")}

    log(f"因子收益估计完成，有效日 {int(np.isfinite(fret[:, 0]).sum())}")

    np.savez_compressed(OUT / "factor_returns.npz", fret=fret.astype("float32"),
                        dates=np.array([str(d.date()) for d in idx]), n_ind=n_ind)
    np.savez_compressed(OUT / "specific.npz", spec=spec)
    np.save(OUT / "beta.npy", BETA)
    with open(OUT / "meta.json", "w") as fh:
        json.dump({"K": K, "n_ind": n_ind, "beta_win": BETA_WIN, "beta_hl": BETA_HL,
                   "f_hl_var": F_HL_VAR, "f_hl_cor": F_HL_COR, "d_hl": D_HL,
                   "nw_lag": NW_LAG, "start": start,
                   "note": "简化版：无 eigenfactor / vol-regime / bias 调整"},
                  fh, ensure_ascii=False, indent=2)
    log(f"完成，用时 {(time.time()-t0)/60:.1f} min  ->  data/risk/")
    return fret, spec, expo


# ------------------------------------------------------------------ 协方差
def factor_cov(fret: np.ndarray, t: int, min_obs: int = 252) -> np.ndarray:
    """t 日可用的因子协方差：EWMA + Newey-West。

    方差与相关用不同半衰期是 Barra 的做法——方差变得快、相关变得慢。
    """
    hist = fret[:t]
    hist = hist[np.isfinite(hist[:, 0])]
    if len(hist) < min_obs:
        return None
    H = np.nan_to_num(hist)
    n = len(H)
    wv = ewma_weights(n, F_HL_VAR)
    wc = ewma_weights(n, F_HL_COR)
    mu_v = wv @ H
    mu_c = wc @ H
    var = wv @ ((H - mu_v) ** 2)
    Dv = np.diag(np.sqrt(np.maximum(var, 1e-16)))
    Z = (H - mu_c) / (np.sqrt(np.maximum(wc @ ((H - mu_c) ** 2), 1e-16))[None, :])
    Cw = (Z * wc[:, None]).T @ Z
    # Newey-West 修相关的自相关
    for k in range(1, NW_LAG + 1):
        w_ = 1 - k / (NW_LAG + 1)
        A = (Z[k:] * wc[k:, None]).T @ Z[:-k]
        Cw += w_ * (A + A.T)
    d = np.sqrt(np.maximum(np.diag(Cw), 1e-16))
    Cw = Cw / np.outer(d, d)
    np.fill_diagonal(Cw, 1.0)
    F = Dv @ Cw @ Dv
    # 保证半正定
    ev, V = np.linalg.eigh((F + F.T) / 2)
    return V @ np.diag(np.maximum(ev, 1e-14)) @ V.T


def specific_var(spec: np.ndarray, t: int, cols: np.ndarray,
                 size_bucket: np.ndarray = None, min_obs: int = 60) -> np.ndarray:
    """特异方差：EWMA + 向规模分组均值贝叶斯收缩。

    单只股票的残差方差估计噪音很大，Barra 的做法是往同规模组的均值收缩，
    观测越少收缩越重。
    """
    lo = max(t - 504, 0)
    H = spec[lo:t, cols]
    n = len(H)
    w = ewma_weights(n, D_HL)
    ok = np.isfinite(H)
    Hf = np.nan_to_num(H)
    cnt = ok.sum(0)
    v = (w[:, None] * Hf ** 2).sum(0) / np.maximum((w[:, None] * ok).sum(0), 1e-12)
    v = np.where(cnt >= min_obs, v, np.nan)
    grand = np.nanmedian(v) if np.isfinite(v).any() else 1e-4
    if size_bucket is not None:
        for b in np.unique(size_bucket):
            sel = size_bucket == b
            gm = np.nanmedian(v[sel]) if np.isfinite(v[sel]).any() else grand
            k = np.clip(cnt[sel] / (cnt[sel] + 120.0), 0.0, 1.0)   # 观测少 -> 收缩重
            v[sel] = k * np.nan_to_num(v[sel], nan=gm) + (1 - k) * gm
    else:
        v = np.where(np.isfinite(v), v, grand)
    return np.maximum(v, 1e-8)


if __name__ == "__main__":
    build()
