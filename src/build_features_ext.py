"""扩充特征集：在原有 9 个纯价格特征之外，补量、换手、波动、日内隔夜、流动性、极值。

为什么必须补
------------
原有 9 维全部由收盘价导出（多周期收益 / MACD / Z-score），彼此高度相关。
那是从 DeePM 继承的限制——原文**故意**只用收盘价，为的是隔离结构先验的贡献，
是实验设计不是实盘配置。

但做截面选股，这个维度差得太远：MASTER 用 Alpha158，HIST 用 Alpha360，
A 股那篇用 213 个因子。拿 9 维价格特征去对标它们的基线数字不公平，
L0 的判负标准会失效——测出来不行，分不清是架构不行还是特征不够。

三条必须遵守的实现纪律（都是 v1 踩出来的）
-----------------------------------------
1. **波动率类分母不在 ffill 序列上估计**。停牌期 ffill 出的常数段会让滚动标准差
   塌到恰好 0，除法直接爆到 1e6 量级。而且 MAD 稳健裁剪救不了——窗口内多数是
   巨值时，滚动 median 本身就是巨值。分母一律用带 NaN 的原始序列算，并加地板。
2. **全部滚动窗口，无未来信息**。写完由 verify.py 的截断重算测试把关。
3. **统一做 252 日 MAD 稳健裁剪**，和原有 9 维口径一致。
"""
from __future__ import annotations
import sys, gc, json, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config as C

FEAT = C.ROOT / "data" / "features"
PQ = dict(compression="zstd", compression_level=3)
EPS = 1e-8

STD_FLOOR_REL = 1e-4      # 滚动标准差的相对地板（相对该序列自身的均值水平）
RET_STD_FLOOR = 2e-3      # 收益率波动地板，同 build_features.py


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def robust_clip(x: pd.DataFrame) -> pd.DataFrame:
    """252 日滚动 median ± 5×1.4826×MAD，和原有特征同口径。"""
    med = x.rolling(C.CLIP_WIN, min_periods=60).median()
    mad = (x - med).abs().rolling(C.CLIP_WIN, min_periods=60).median()
    band = C.CLIP_K * C.MAD_SCALE * mad
    return x.clip(lower=med - band, upper=med + band).astype("float32")


def zs(x: pd.DataFrame, win: int, floor_rel: float = STD_FLOOR_REL) -> pd.DataFrame:
    """时序 z-score。分母加**相对地板**——绝对地板对不同量纲的序列没意义。"""
    mu = x.rolling(win, min_periods=max(5, win // 3)).mean()
    sd = x.rolling(win, min_periods=max(5, win // 3)).std()
    lo = floor_rel * mu.abs().rolling(C.CLIP_WIN, min_periods=60).median()
    sd = sd.where(sd >= lo)                        # 低于地板 -> NaN，不产出巨值
    return (x - mu) / sd


def emit(x: pd.DataFrame, name: str, valid: pd.DataFrame, stats: dict):
    x = robust_clip(x.where(valid))
    fp = FEAT / f"{name}.parquet"
    x.to_parquet(fp, **PQ)
    v = x.to_numpy()
    fin = np.isfinite(v)
    stats[name] = {"nan_pct": round(float(1 - fin.mean()), 4),
                   "std": round(float(np.nanstd(v[fin])), 4) if fin.any() else None,
                   "mb": round(fp.stat().st_size / 1e6, 1)}
    log(f"  {name:16} nan={stats[name]['nan_pct']:.1%} std={stats[name]['std']} "
        f"{stats[name]['mb']}MB")
    del x, v
    gc.collect()


def main():
    t0 = time.time()
    log("载入面板 …")
    A = C.OUT
    close = pd.read_parquet(A / "A_close.parquet")
    openp = pd.read_parquet(A / "A_open.parquet")
    high = pd.read_parquet(A / "A_high.parquet")
    low = pd.read_parquet(A / "A_low.parquet")
    vol = pd.read_parquet(A / "A_vol.parquet")
    amt = pd.read_parquet(A / "A_amount.parquet")
    turn = pd.read_parquet(A / "A_turn.parquet")
    ret1 = pd.read_parquet(A / "A_ret1.parquet")

    valid = close.notna()                       # 只在真实交易日产出特征
    px = close.ffill()                          # 仅用于取历史价，不用于估波动
    stats = {}
    log(f"面板 {close.shape}，有效格子 {int(valid.to_numpy().sum()):,}")

    # ---------------------------------------------------------- 1 短期反转
    log("1/7 短期反转")
    sig = pd.read_parquet(A / "A_sigma63.parquet")
    sig = sig.where(sig >= RET_STD_FLOOR)
    for h in (5, 10):
        emit((px / px.shift(h) - 1.0) / (sig * np.sqrt(h)), f"ret_{h}d", valid, stats)

    # ---------------------------------------------------------- 2 量能
    log("2/7 量能")
    for w in (5, 21, 63):
        emit(zs(vol, w), f"vol_z_{w}", valid, stats)
    emit(zs(amt, 21), "amt_z_21", valid, stats)
    v5 = vol.rolling(5, min_periods=2).mean()
    v21 = vol.rolling(21, min_periods=7).mean()
    emit(v5 / (v21 + EPS) - 1.0, "vol_ratio_5_21", valid, stats)
    del v5, v21; gc.collect()

    # ---------------------------------------------------------- 3 换手
    log("3/7 换手")
    for w in (21, 63):
        emit(zs(turn, w), f"turn_z_{w}", valid, stats)
    emit(np.log1p(turn.rolling(21, min_periods=7).mean()), "turn_mean_21", valid, stats)

    # ---------------------------------------------------------- 4 波动
    log("4/7 波动")
    for w in (21, 63):
        emit(np.log(ret1.rolling(w, min_periods=w // 3).std() + RET_STD_FLOOR),
             f"rv_{w}", valid, stats)
    rv21 = ret1.rolling(21, min_periods=7).std()
    emit(rv21.rolling(63, min_periods=21).std() / (rv21.rolling(63, min_periods=21).mean() + EPS),
         "vov_63", valid, stats)
    del rv21; gc.collect()
    emit(((high - low) / (close + EPS)).rolling(21, min_periods=7).mean(),
         "hl_range_21", valid, stats)

    # ---------------------------------------------------------- 5 日内 / 隔夜
    # A 股这两段含义不同：隔夜跳空反映信息与外盘，日内反映本地交易行为。
    log("5/7 日内 / 隔夜拆分")
    prev = close.ffill().shift(1)
    on = (openp / prev - 1.0)                    # 隔夜
    co = (close / openp - 1.0)                   # 日内
    for nm, s in (("on_ret_21", on), ("co_ret_21", co)):
        emit(s.rolling(21, min_periods=7).mean() / (sig + EPS), nm, valid, stats)
    del prev, on, co; gc.collect()

    # ---------------------------------------------------------- 6 量价关系 / 流动性
    log("6/7 量价关系与流动性")
    dv = np.log(vol + 1.0).diff()
    emit(ret1.rolling(21, min_periods=10).corr(dv), "vp_corr_21", valid, stats)
    del dv; gc.collect()
    illiq = (ret1.abs() / (amt + 1.0)) * 1e9     # Amihud，放大到可读量级
    emit(np.log1p(illiq.rolling(21, min_periods=7).mean()), "illiq_21", valid, stats)
    del illiq; gc.collect()

    # ---------------------------------------------------------- 7 极值与形状
    # MAX 因子（彩票效应）在 A 股显著；偏度捕捉收益分布的不对称
    log("7/7 极值与分布形状")
    emit(ret1.rolling(21, min_periods=7).max() / (sig + EPS), "max_ret_21", valid, stats)
    emit(ret1.rolling(63, min_periods=21).skew(), "skew_63", valid, stats)

    # 对等权市场的 beta（滚动 63 日）
    mkt = ret1.mean(axis=1)
    cov = ret1.rolling(63, min_periods=21).cov(mkt)
    var = mkt.rolling(63, min_periods=21).var()
    emit(cov.div(var + EPS, axis=0), "beta_63", valid, stats)
    del cov, var; gc.collect()

    # ---------------------------------------------------------- 登记特征集
    new = list(stats.keys())
    sets = {
        "price9": C.FEATURE_SETS["price9"],
        "ext_only": new,
        "full": C.FEATURE_SETS["price9"] + new,
    }
    (C.META / "feature_sets.json").write_text(
        json.dumps(sets, ensure_ascii=False, indent=2), encoding="utf-8")
    old = json.loads((C.META / "features_stats.json").read_text())
    old.update(stats)
    (C.META / "features_stats.json").write_text(
        json.dumps(old, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"新增 {len(new)} 个特征，full 集共 {len(sets['full'])} 维，"
        f"用时 {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
