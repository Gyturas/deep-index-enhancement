"""把分散在两个旧项目里的面板对齐成一套统一的 parquet。

要解决的三个已知不一致
----------------------
1. 代码制式：价格面板是 Wind 码 ('000001.SZ')，截面面板是天软码 ('SH600004')。
   两者直接 join 会得到全 NaN 而**不报错**——本脚本统一转成 Wind 码。
2. 日期端点：价格面板到 2026-07-24，截面面板到 2026-07-21，且 mask.pkl 末日全 False。
   统一以价格面板的交易日历为主轴，截面面板 reindex 后按原值对齐（不 ffill 到未来）。
3. 列集合：close 5869 列、vol/turn 6228 列，取交集 5862 列。

输出（data/aligned/，全部 parquet + zstd）
------------------------------------------
Tier A  全A，2003-01-02 ~ 末日：open/high/low/close/vol/amount/turn/ret1/sigma63
        + tradable / tradable_strict / oneword / limit_up / limit_down / age
Tier B  中证1000 池：member/weight/ind1/ind2/lnmv/illiq20/turnover20
"""
import sys, json, time, warnings, hashlib
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config as C

PQ = dict(compression="zstd", compression_level=3)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save(df: pd.DataFrame, name: str):
    fp = C.OUT / f"{name}.parquet"
    df.to_parquet(fp, **PQ)
    log(f"  -> {name}.parquet  {df.shape}  {fp.stat().st_size/1e6:.0f}MB")


def fingerprint(df: pd.DataFrame) -> str:
    """对数值面板取稳定指纹，用于事后复现校验。"""
    a = np.nan_to_num(df.to_numpy(dtype="float64"), nan=-9.87654321e30)
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


# ------------------------------------------------------------------ 载入
def load_prices():
    log("载入价格面板 …")
    px = {}
    for src, dst in [("open", "open"), ("high", "high"), ("low", "low"),
                     ("close", "close"), ("vol", "vol"), ("amount", "amount"),
                     ("turn", "turn")]:
        px[dst] = pd.read_parquet(C.SRC_PRICE / f"{src}.parquet")
        log(f"  {src}: {px[dst].shape}")
    return px


def main():
    t0 = time.time()
    px = load_prices()

    # --- 主轴：价格面板的交易日历 ∩ 有成交量的区间；列取 close ∩ vol
    codes = sorted(set(px["close"].columns) & set(px["vol"].columns))
    idx = px["close"].index
    idx = idx[(idx >= C.DATE_START_A)]
    log(f"主轴: {len(idx)} 个交易日 ({idx[0].date()}~{idx[-1].date()}), {len(codes)} 只代码")

    for k in px:
        px[k] = px[k].reindex(index=idx, columns=codes).astype("float32")

    # --- 基础派生量
    log("计算 ret1 / sigma63 …")
    close = px["close"]
    prev = close.ffill().shift(1)                 # 前收盘（跨停牌用最近一次收盘，与交易所口径一致）
    ret1 = (close / prev - 1.0).astype("float32")
    sigma63 = ret1.ewm(span=C.EWMA_SPAN, min_periods=20).std().astype("float32")

    # --- 可交易性
    log("构造可交易掩码 …")
    has_px = close.notna()
    vol = px["vol"]
    traded = has_px & vol.notna() & (vol > 0)     # 有价且有成交
    halted = has_px & (vol.fillna(0) <= 0)        # 有价但零成交 = 停牌/无量

    # 一字板：最高=最低且有成交。不依赖 ST 与板块规则，是"确定买不进/卖不出"的情形。
    oneword = traded & (px["high"] == px["low"])

    # 上市天数（按已出现的有效交易日累计）
    age = has_px.cumsum().where(has_px).astype("float32")

    # --- 涨跌停（需要板块 + ST）
    log("判定涨跌停 …")
    st = load_st_panel(idx, codes)
    board = pd.Series({c: C.board_of(c) for c in codes})
    w = build_limit_width(idx, codes, board, st)
    pct = ret1
    limit_up = traded & (pct >= (w - C.LIMIT_TOL)) & (close >= px["high"] - 1e-9)
    limit_dn = traded & (pct <= -(w - C.LIMIT_TOL)) & (close <= px["low"] + 1e-9)

    new_ok = age >= C.NEW_LISTING_EXCL_DAYS
    tradable = (traded & ~oneword & new_ok).fillna(False)
    tradable_strict = (tradable & ~limit_up & ~limit_dn).fillna(False)

    log(f"  停牌格子 {int(halted.sum().sum()):,} | 一字板 {int(oneword.sum().sum()):,} "
        f"| 涨停 {int(limit_up.sum().sum()):,} | 跌停 {int(limit_dn.sum().sum()):,}")
    log(f"  tradable 覆盖 {tradable.sum().sum()/has_px.sum().sum():.2%} of 有价格子")

    # --- 落盘 Tier A
    log("落盘 Tier A …")
    for k, v in px.items():
        save(v, f"A_{k}")
    save(ret1, "A_ret1"); save(sigma63, "A_sigma63"); save(age, "A_age")
    for nm, v in [("tradable", tradable), ("tradable_strict", tradable_strict),
                  ("oneword", oneword), ("limit_up", limit_up),
                  ("limit_down", limit_dn), ("halted", halted), ("is_st", st)]:
        save(v.astype("bool"), f"A_{nm}")

    # --- Tier B：成分、权重、行业、市值、流动性
    log("构造 Tier B …")
    wt = pd.read_parquet(C.SRC_PRICE / "中证1000权重.parquet")
    wt.columns = [C.to_wind(c) for c in wt.columns]
    wt = wt.reindex(index=idx, columns=codes).astype("float32")
    member = wt.notna()          # 权重面板非空即当日在册成分股
    log(f"  成分股日均只数 {member.sum(1).replace(0, np.nan).mean():.1f}")
    save(wt, "B_weight"); save(member.astype("bool"), "B_member")

    for src_name, dst in [("lnmv", "B_lnmv")]:
        p = pd.read_pickle(C.SRC_PANEL / f"{src_name}.pkl")
        p.columns = [C.to_wind(c) for c in p.columns]
        save(p.reindex(index=idx, columns=codes).astype("float32"), dst)

    aux = pd.read_pickle(C.SRC_PANEL / "aux.pkl")
    for k in ("turnover20", "illiq20", "BP"):
        if k in aux:
            p = aux[k].copy()
            p.columns = [C.to_wind(c) for c in p.columns]
            save(p.reindex(index=idx, columns=codes).astype("float32"), f"B_{k}")

    # 行业：优先用 cjpy 抓到的申万二级；回落到旧面板的申万一级
    ind1, ind2 = load_industry(idx, codes)
    if ind1 is not None:
        save(ind1, "B_ind1"); save(ind2, "B_ind2")

    # --- 交叉校验：新 member 与旧 mask.pkl 的一致率
    cross_check(member, tradable, idx, codes)

    # --- 元数据
    meta = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_dates": len(idx), "n_codes": len(codes),
        "date_first": str(idx[0].date()), "date_last": str(idx[-1].date()),
        "fingerprints": {"close": fingerprint(close), "ret1": fingerprint(ret1),
                         "sigma63": fingerprint(sigma63), "weight": fingerprint(wt)},
        "counts": {"halted": int(halted.sum().sum()), "oneword": int(oneword.sum().sum()),
                   "limit_up": int(limit_up.sum().sum()), "limit_down": int(limit_dn.sum().sum()),
                   "tradable": int(tradable.sum().sum())},
    }
    (C.META / "panels_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"完成，用时 {(time.time()-t0)/60:.1f} min")


# ------------------------------------------------------------------ 辅助
def _meta_frames():
    files = sorted((C.CACHE / "meta_raw").glob("*.parquet"))
    if not files:
        return None
    return files


def load_st_panel(idx, codes) -> pd.DataFrame:
    """把月末采样的 ST 状态展开到日频（在采样日之间按前值保持）。"""
    files = _meta_frames()
    if not files:
        log("  [警告] 无 meta_raw 缓存，ST 全部按 False 处理（涨跌停判定会漏掉 ST 的 5% 档）")
        return pd.DataFrame(False, index=idx, columns=codes)
    rows = {}
    for f in files:
        d = pd.read_parquet(f, columns=["代码", "是否ST"])
        rows[pd.Timestamp(f.stem)] = d.set_index("代码")["是否ST"]
    st = pd.DataFrame(rows).T.sort_index()
    st = st.reindex(columns=codes)
    st = st.reindex(index=idx.union(st.index)).ffill().reindex(idx)
    return (st.fillna(0) > 0)


def build_limit_width(idx, codes, board, st) -> pd.DataFrame:
    """逐 (日, 股) 的涨跌停幅度面板。"""
    w = pd.DataFrame(0.10, index=idx, columns=codes, dtype="float32")
    b = board.reindex(codes)
    for name, mask in [("BSE", b == "BSE"), ("STAR", b == "STAR"),
                       ("GEM", b == "GEM"), ("MAIN", b == "MAIN")]:
        cols = b.index[mask]
        if not len(cols):
            continue
        if name == "BSE":
            w.loc[:, cols] = 0.30
        elif name == "STAR":
            w.loc[:, cols] = 0.20
        elif name == "GEM":
            w.loc[:, cols] = 0.10
            w.loc[w.index >= C.GEM_WIDEN_DATE, cols] = 0.20
        else:
            w.loc[:, cols] = 0.10
    # ST：主板/改革前创业板降到 5%；科创、北交所、改革后创业板不变
    narrow = st & (w <= 0.10 + 1e-9)
    w = w.where(~narrow, 0.05)
    return w.astype("float32")


def load_industry(idx, codes):
    """行业面板以 int16 码存储 + 一份 json 映射。

    直接存字符串会让 5900x5862 的面板膨胀到数 GB（Python object），
    且 ffill 极慢。这里先在稀疏的采样网格（295 x 5862）上 factorize，
    再对整数面板做 reindex/ffill，内存与耗时都降一个量级。
    下游做 embedding 查表本来也需要整数索引。
    """
    files = _meta_frames()
    if not files:
        log("  [警告] 无 meta_raw 缓存，跳过行业面板")
        return None, None
    out, maps = [], {}
    for lvl, col in [(1, "申万一级行业代码"), (2, "申万二级行业代码")]:
        rows = {}
        for f in files:
            d = pd.read_parquet(f, columns=["代码", col]).set_index("代码")
            rows[pd.Timestamp(f.stem)] = d[col]
        samp = pd.DataFrame(rows).T.sort_index().reindex(columns=codes)   # 采样网格，很小
        cats = pd.Index(sorted(pd.unique(samp.to_numpy().ravel())))
        cats = cats[cats.notna() & (cats != "")]
        lut = {c: i for i, c in enumerate(cats)}
        arr = samp.apply(lambda s: s.map(lut)).astype("float32")          # 未知 -> NaN
        p = (arr.reindex(index=idx.union(arr.index)).ffill().reindex(idx)
             .fillna(-1).astype("int16"))
        out.append(p)
        maps[f"ind{lvl}"] = {int(v): str(k) for k, v in lut.items()}
        log(f"  申万{'一' if lvl == 1 else '二'}级: {len(cats)} 类, "
            f"未覆盖格子 {(p < 0).to_numpy().mean():.1%}")
    (C.META / "industry_map.json").write_text(
        json.dumps(maps, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def cross_check(member, tradable, idx, codes):
    """与旧 mask.pkl 对表。

    结论（已核实）：旧 mask.pkl 不是纯粹的成分股名单，而是
    **成分股 ∩ 可交易**——新有旧无的格子里 72% 是停牌日，
    单日最大差异出现在 2015-07-07（千股停牌）。
    本项目把"身份"(B_member) 与"当日可交易"(A_tradable) 拆开，
    因此这里同时报告两种口径的一致率：拆开后 member & tradable 应显著更贴近旧掩码。
    """
    fp = C.SRC_PANEL / "mask.pkl"
    if not fp.exists():
        return
    old = pd.read_pickle(fp)
    old.columns = [C.to_wind(c) for c in old.columns]
    common_d = idx.intersection(old.index)
    common_c = [c for c in codes if c in old.columns]
    b = old.loc[common_d, common_c].fillna(False)
    good = b.sum(1) > 0            # 旧 mask 末日整行 False，是已知残缺日
    b = b.loc[good]

    out = {"overlap_days": int(len(b)), "mean_n_old": float(b.sum(1).mean()),
           "dropped_allfalse_days": int((~good).sum())}
    for tag, a in [("member_only", member.loc[b.index, common_c]),
                   ("member_and_tradable", (member & tradable).loc[b.index, common_c])]:
        agree = float((a == b).to_numpy().mean())
        out[tag] = {"cell_agreement": agree, "mean_n": float(a.sum(1).mean())}
        log(f"  [校验] {tag:20} 一致率 {agree:.4%}  日均只数 {a.sum(1).mean():.1f} "
            f"(旧 {b.sum(1).mean():.1f})")
    (C.META / "member_crosscheck.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
