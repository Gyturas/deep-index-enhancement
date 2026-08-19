"""对齐结果的验收测试。

第 1 项是硬性前提：**重建的面板必须能逐格复现源面板的原值**，
否则后续一切结论作废。第 3 项检查滚动窗口有没有引入未来信息。
"""
import sys, json, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config as C

RESULT = {}
FAIL = []


def check(name, ok, detail=""):
    RESULT[name] = {"pass": bool(ok), "detail": detail}
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
    if not ok:
        FAIL.append(name)


# ---------------------------------------------------------------- 1 逐格复现
def t1_replay():
    for src, dst in [("close", "A_close"), ("open", "A_open"), ("high", "A_high"),
                     ("low", "A_low"), ("vol", "A_vol"), ("amount", "A_amount"),
                     ("turn", "A_turn")]:
        a = pd.read_parquet(C.SRC_PRICE / f"{src}.parquet")
        b = pd.read_parquet(C.OUT / f"{dst}.parquet")
        common_d, common_c = b.index, b.columns
        x = a.reindex(index=common_d, columns=common_c).to_numpy("float64")
        y = b.to_numpy("float64")
        both_nan = np.isnan(x) & np.isnan(y)
        eq = both_nan | (np.abs(x - y) <= 1e-6 * np.maximum(1.0, np.abs(x)))
        bad = int((~eq).sum())
        check(f"1.复现-{src}", bad == 0, f"不一致格子 {bad:,} / {x.size:,}")
        del a, b, x, y


# ---------------------------------------------------------------- 2 代码映射
def t2_codes():
    samples = ["600004.SH", "000001.SZ", "300750.SZ", "688981.SH", "920819.BJ"]
    ok = all(C.to_wind(C.to_ts(s)) == s for s in samples)
    check("2.代码往返映射", ok, f"{[C.to_ts(s) for s in samples]}")

    m = pd.read_parquet(C.OUT / "B_member.parquet")
    old = pd.read_pickle(C.SRC_PANEL / "mask.pkl")
    mapped = {C.to_wind(c) for c in old.columns}
    miss = mapped - set(m.columns)
    check("2.旧mask代码可映射", len(miss) == 0, f"未落入新列集的 {len(miss)} 个")


# ---------------------------------------------------------------- 3 无未来信息
def t3_no_lookahead(cut="2022-06-30"):
    """用截断到 cut 的历史重算特征，与全历史版本在 cut 之前逐格比较。"""
    close = pd.read_parquet(C.OUT / "A_close.parquet")
    sig = pd.read_parquet(C.OUT / "A_sigma63.parquet")
    cols = list(close.columns[:400])                    # 抽样 400 只以控制耗时
    full, trunc = close[cols], close.loc[:cut, cols]

    def feat(px, sg):
        p = px.ffill()
        r = (p / p.shift(63) - 1.0) / (sg * np.sqrt(63) + 1e-8)
        r = r.where(px.notna())
        med = r.rolling(C.CLIP_WIN, min_periods=60).median()
        mad = (r - med).abs().rolling(C.CLIP_WIN, min_periods=60).median()
        b = C.CLIP_K * C.MAD_SCALE * mad
        return r.clip(med - b, med + b)

    a = feat(full, sig[cols]).loc[:cut]
    b = feat(trunc, sig.loc[:cut, cols])
    x, y = a.to_numpy("float64"), b.to_numpy("float64")
    both_nan = np.isnan(x) & np.isnan(y)
    eq = both_nan | (np.abs(x - y) <= 1e-8 + 1e-6 * np.abs(x))
    bad = int((~eq).sum())
    check("3.特征无未来信息", bad == 0,
          f"截断@{cut} 重算，不一致 {bad:,} / {x.size:,}")


# ---------------------------------------------------------------- 4 掩码逻辑
def t4_masks():
    has = pd.read_parquet(C.OUT / "A_close.parquet").notna()
    tr = pd.read_parquet(C.OUT / "A_tradable.parquet")
    trs = pd.read_parquet(C.OUT / "A_tradable_strict.parquet")
    ow = pd.read_parquet(C.OUT / "A_oneword.parquet")
    mem = pd.read_parquet(C.OUT / "B_member.parquet")

    check("4.tradable ⊆ 有行情", bool((tr & ~has).to_numpy().sum() == 0),
          f"越界 {int((tr & ~has).to_numpy().sum())}")
    check("4.strict ⊆ tradable", bool((trs & ~tr).to_numpy().sum() == 0),
          f"越界 {int((trs & ~tr).to_numpy().sum())}")
    check("4.一字板已排除", bool((tr & ow).to_numpy().sum() == 0),
          f"残留 {int((tr & ow).to_numpy().sum())}")

    n = mem.sum(1)
    n = n[n > 0]
    check("4.成分股只数≈1000", bool(950 <= n.mean() <= 1005),
          f"日均 {n.mean():.1f}，区间 [{n.min()}, {n.max()}]，有效日 {len(n)}")

    cov = (mem & tr).sum(1) / mem.sum(1).replace(0, np.nan)
    cov = cov.dropna()
    check("4.成分股可交易覆盖率", bool(cov.mean() > 0.90),
          f"日均 {cov.mean():.2%}，最低 {cov.min():.2%} @ {cov.idxmin().date()}")


# ---------------------------------------------------------------- 5 日历
def t5_calendar():
    src = pd.read_parquet(C.SRC_PRICE / "close.parquet", columns=["000001.SZ"]).index
    out = pd.read_parquet(C.OUT / "A_close.parquet").index
    check("5.日历为源日历子集", bool(out.difference(src).empty),
          f"越界日 {len(out.difference(src))}")
    check("5.日历单调无重复", bool(out.is_monotonic_increasing and out.is_unique),
          f"{out[0].date()} ~ {out[-1].date()}, n={len(out)}")


# ---------------------------------------------------------------- 6 特征分布
def t6_features():
    d = C.ROOT / "data" / "features"
    if not d.exists() or not list(d.glob("*.parquet")):
        check("6.特征已生成", False, "features/ 为空")
        return
    bad = []
    for f in sorted(d.glob("*.parquet")):
        v = pd.read_parquet(f).to_numpy()
        fin = np.isfinite(v)
        if not fin.any():
            bad.append(f"{f.stem}:全空")
            continue
        s = float(np.nanstd(v[fin]))
        mx = float(np.nanmax(np.abs(v[fin])))
        if not (0.05 < s < 20) or mx > 500:
            bad.append(f"{f.stem}:std={s:.2f},max={mx:.1f}")
    check("6.特征分布合理", len(bad) == 0, "; ".join(bad) if bad else
          f"{len(list(d.glob('*.parquet')))} 个特征通过")


if __name__ == "__main__":
    t0 = time.time()
    for fn in (t1_replay, t2_codes, t3_no_lookahead, t4_masks, t5_calendar, t6_features):
        try:
            fn()
        except Exception as e:
            check(fn.__name__, False, f"{type(e).__name__}: {str(e)[:120]}")
    (C.META / "verify_report.json").write_text(
        json.dumps(RESULT, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{'='*60}\n{len(RESULT)-len(FAIL)}/{len(RESULT)} 通过，用时 {time.time()-t0:.0f}s")
    if FAIL:
        print("未通过：" + ", ".join(FAIL))
        sys.exit(1)
