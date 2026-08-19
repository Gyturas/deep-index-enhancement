"""按论文 §3.3 构造 DeePM 输入特征，逐个落盘（避免一次性占满内存）。

与论文的两处**有意偏离**，均在计划书 §特征 里记录：

1. MACD 的归一化分母。论文 Eq.(15) 写作除以 σ̂（前文定义为"日收益率"的 EWMA 波动），
   但分子 EWM_S(P)-EWM_L(P) 是**价格量纲**，除以收益率量纲不自洽。这里按该指标的
   原始定义 [Baz et al. 2015; Lim et al. 2019; Wood et al. 2023] 除以**价格的 63 日
   滚动标准差**，再按论文要求用自身 252 日滚动标准差二次归一化。

2. 前收盘取 close.ffill().shift(1)。停牌期间价格面板为 NaN，直接 shift 会把停牌前
   的收盘错位；ffill 后再 shift 与交易所"前收盘"口径一致。

3. **波动率分母的地板与去 ffill**。这是实测踩到的坑，必须处理：
   停牌期用 ffill 补出的常数价格段会让 63 日价格标准差塌到 **恰好 0**
   （实测 702,357 个格子；改用带 NaN 的原始 close 后降到 168,190），
   除以 `0 + 1e-8` 使 MACD 爆到 $10^6\sim10^7$。同理 $\hat\sigma$ 有 129,472
   个格子 $< 0.002$，把多周期收益炸到几十倍量级。
   处理：(a) 波动率分母一律在**真实交易日**上估计，不用 ffill 序列；
   (b) 低于地板的格子**置 NaN 而非产出巨值**——语义是"该日波动率估计不可用"，
   下游掩码可识别；(c) ffill 只保留在分子取历史价的 shift 上。

裁剪（论文 §3.3 第 4 点）：逐特征做 252 日滚动 median / MAD，
裁到 [m ± 5 × 1.4826 × MAD]。全部为滚动窗口，无未来信息。
"""
import sys, gc, json, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config as C

PQ = dict(compression="zstd", compression_level=3)
FEAT = C.ROOT / "data" / "features"
FEAT.mkdir(parents=True, exist_ok=True)
EPS = 1e-8

# 波动率分母的地板。低于地板即判定"估计不可用"，该格子置 NaN。
SIGMA_FLOOR = 0.002        # 日收益率波动下限（0.2%/日，约 3% 年化；A 股实际不会更低）
PXSTD_FLOOR_REL = 1e-3     # 价格标准差下限 = 1e-3 × 当前价格
MACDSTD_FLOOR = 1e-6       # MACD 二次归一化分母下限


def floored(x: pd.DataFrame, floor) -> pd.DataFrame:
    """低于地板的格子置 NaN，并返回被丢弃的比例。"""
    ok = x >= (floor if not isinstance(floor, pd.DataFrame) else floor)
    return x.where(ok)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def robust_clip(x: pd.DataFrame) -> pd.DataFrame:
    """252 日滚动 median ± 5×1.4826×MAD 裁剪（论文 §3.3.4）。"""
    med = x.rolling(C.CLIP_WIN, min_periods=60).median()
    mad = (x - med).abs().rolling(C.CLIP_WIN, min_periods=60).median()
    band = C.CLIP_K * C.MAD_SCALE * mad
    return x.clip(lower=med - band, upper=med + band).astype("float32")


def emit(x: pd.DataFrame, name: str, stats: dict):
    x = robust_clip(x)
    fp = FEAT / f"{name}.parquet"
    x.to_parquet(fp, **PQ)
    v = x.to_numpy()
    fin = np.isfinite(v)
    stats[name] = {"nan_pct": float(1 - fin.mean()),
                   "mean": float(np.nanmean(np.where(fin, v, np.nan))),
                   "std": float(np.nanstd(np.where(fin, v, np.nan))),
                   "p01": float(np.nanpercentile(v[fin], 1)) if fin.any() else None,
                   "p99": float(np.nanpercentile(v[fin], 99)) if fin.any() else None,
                   "mb": round(fp.stat().st_size / 1e6, 1)}
    log(f"  {name:16} {x.shape} nan={stats[name]['nan_pct']:.1%} "
        f"std={stats[name]['std']:.3f} {stats[name]['mb']}MB")
    del x, v
    gc.collect()


def main():
    t0 = time.time()
    A = C.OUT
    log("载入基础面板 …")
    close = pd.read_parquet(A / "A_close.parquet")
    sigma = pd.read_parquet(A / "A_sigma63.parquet")
    log(f"  close {close.shape}")

    stats = {}
    px = close.ffill()                      # 只用于分子取 h 日前的历史价

    # 波动率分母：一律在真实交易日上估计（不用 ffill 序列），并施加地板
    sig = floored(sigma, SIGMA_FLOOR)
    log(f"  σ̂ 低于地板 {SIGMA_FLOOR} 被判不可用: "
        f"{(sigma.notna() & sig.isna()).to_numpy().sum():,} 格")

    # ---- 1. 波动归一化多周期收益 (Eq.14)
    log("特征 1/3：波动归一化收益")
    for h in C.RET_HORIZONS:
        r = (px / px.shift(h) - 1.0) / (sig * np.sqrt(h))
        r = r.where(close.notna())          # 只在真实有行情的格子上保留
        emit(r, f"ret_{h}d", stats)
        del r

    # ---- 2. MACD 多尺度 (Eq.15，分母见模块说明)
    log("特征 2/3：MACD 趋势滤波")
    px_std63 = floored(close.rolling(63, min_periods=20).std(),
                       PXSTD_FLOOR_REL * px)
    log(f"  价格 std63 低于地板被判不可用: "
        f"{(close.rolling(63, min_periods=20).std().notna() & px_std63.isna()).to_numpy().sum():,} 格")
    for s, l in C.MACD_PAIRS:
        m = (px.ewm(span=s, min_periods=s).mean()
             - px.ewm(span=l, min_periods=l).mean()) / px_std63
        m = m / floored(m.rolling(C.MACD_RENORM_WIN, min_periods=60).std(), MACDSTD_FLOOR)
        emit(m.where(close.notna()), f"macd_{s}_{l}", stats)
        del m
    del px_std63
    gc.collect()

    # ---- 3. 对数价格 Z-score
    log("特征 3/3：对数价格 Z-score")
    lp = np.log(px.where(px > 0))
    for w in C.ZSCORE_WINS:
        z = (lp - lp.rolling(w, min_periods=max(10, w // 4)).mean()) / \
            (lp.rolling(w, min_periods=max(10, w // 4)).std() + EPS)
        emit(z.where(close.notna()), f"zscore_{w}d", stats)
        del z

    (C.META / "features_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"完成 {len(stats)} 个特征，用时 {(time.time()-t0)/60:.1f} min")

    # 论文 §3.3.5 的两个互斥子集（不要同时喂进 V-VSN）
    subsets = {
        "raw_momentum": [f"ret_{h}d" for h in C.RET_HORIZONS] +
                        [f"zscore_{w}d" for w in C.ZSCORE_WINS],
        "signal_based": ["ret_1d"] + [f"macd_{s}_{l}" for s, l in C.MACD_PAIRS] +
                        [f"zscore_{w}d" for w in C.ZSCORE_WINS],
    }
    (C.META / "feature_subsets.json").write_text(
        json.dumps(subsets, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"特征子集: raw_momentum({len(subsets['raw_momentum'])}) / "
        f"signal_based({len(subsets['signal_based'])})")


if __name__ == "__main__":
    main()
