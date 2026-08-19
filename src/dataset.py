"""面板 -> 张量。v2 按**天**批，不再按窗口批。

和 v1 的关键区别
----------------
v1 的一个样本是 (84 日窗口 x 全体股票)，因为损失是组合夏普，需要一整段连续收益。
v2 的损失是逐日截面 IC，所以一个样本就是**一天**：取该日的全部可交易股票，
每只回看 L=20 天的特征。batch 就是若干天。

这让张量小了一个量级：1000 只 x 20 天 x 9 维 = 18 万个数/天，
32 天一批也才 23MB，比 v1 轻得多。

属性分档
--------
市值/流动性/年限做**逐日截面分位**分档（默认 10 档），而不是全样本分档——
市值的绝对水平一直在漂移，截面分位才是稳定的。0 号档留给"未知"。
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import config as C
import splits as S

FEAT_DIR = C.ROOT / "data" / "features"
LAB_DIR = C.ROOT / "data" / "labels"


def _bucket(df: pd.DataFrame, n: int) -> np.ndarray:
    """逐日截面分位分档 -> 1..n，NaN -> 0。"""
    r = df.rank(axis=1, pct=True).to_numpy("float32")
    b = np.ceil(r * n)
    b = np.where(np.isfinite(b), b, 0.0)
    return np.clip(b, 0, n).astype("int16")


class Panels:
    """一次性载入全部面板，按天切片。内存换速度。"""

    def __init__(self, feature_set: str = None, h: int = None,
                 lookback: int = None, device: str = "cpu",
                 xsec_norm: bool = True):
        self.device = device
        self.h = h or C.LABEL_H_MAIN
        self.L = lookback or C.LOOKBACK
        fs = feature_set or C.FEATURE_SET_DEFAULT
        names = C.FEATURE_SETS.get(fs)
        if names is None:                                   # 扩展特征集写在 json 里
            names = json.loads((C.META / "feature_sets.json").read_text())[fs]
        self.feat_names = names

        close = pd.read_parquet(C.OUT / "A_close.parquet")
        self.index, self.codes = close.index, close.columns
        self.N_all = len(self.codes)

        # 截面任务：默认把每个特征做**逐日截面 rank** 映到 [-1,1]。
        # 时序标准化（原特征已做）解决的是"这只股票现在算不算高"，
        # 截面标准化解决的是"今天这只股票在全池里算不算高"——
        # 后者才是选股任务真正要的信息，也让不同量纲的特征直接可比。
        # **预分配后逐个填充**，不用 np.stack。
        # 30 个特征时 stack 的峰值是 8GB（30 个分量各一份 + 结果一份），
        # 而预分配只要 4GB。pod 上只有 31GB RAM，这个差别是安全边际。
        T, Nn, Fn = len(self.index), len(self.codes), len(names)
        cache = FEAT_DIR.parent / f"cache_{fs}_{'xs' if xsec_norm else 'raw'}.npy"
        if cache.exists():
            X = np.load(cache, mmap_mode=None)
            self._from_cache = True
        else:
            X = np.empty((T, Nn, Fn), dtype="float32")
            for k, n in enumerate(names):
                df = pd.read_parquet(FEAT_DIR / f"{n}.parquet")
                if xsec_norm:
                    df = 2.0 * (df.rank(axis=1, pct=True) - 0.5)
                X[:, :, k] = df.to_numpy("float32")
                del df
            self._from_cache = False
            if C.CACHE_FEATURES:
                np.save(cache, X)                            # 省掉每次启动的截面 rank
        self.feat_valid = np.isfinite(X).all(-1)
        self.X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        self.F = len(names)
        self.xsec_norm = xsec_norm

        self.y = pd.read_parquet(LAB_DIR / f"label_h{self.h}.parquet").to_numpy("float32")
        self.y = np.nan_to_num(self.y, nan=0.0)
        self.m = pd.read_parquet(LAB_DIR / f"mask_h{self.h}.parquet").to_numpy()

        # --- 属性：行业码直接用（已是 int16，-1 表未知 -> +1 后 0 表未知）
        ind1 = pd.read_parquet(C.OUT / "B_ind1.parquet").to_numpy("int16")
        ind2 = pd.read_parquet(C.OUT / "B_ind2.parquet").to_numpy("int16")
        self.attr = {
            "ind1": np.clip(ind1 + 1, 0, None).astype("int16"),
            "ind2": np.clip(ind2 + 1, 0, None).astype("int16"),
            "size": _bucket(pd.read_parquet(C.OUT / "B_lnmv.parquet"), C.N_BUCKET),
            "liq":  _bucket(pd.read_parquet(C.OUT / "B_illiq20.parquet"), C.N_BUCKET),
            "age":  _bucket(pd.read_parquet(C.OUT / "A_age.parquet"), C.N_BUCKET),
        }
        self.n_ind1 = int(self.attr["ind1"].max())
        self.n_ind2 = int(self.attr["ind2"].max())

        # --- 市场状态（L2 用；没有就置零，模型仍能跑）
        mkt_fp = C.ROOT / "data" / "market" / "market_state.parquet"
        if mkt_fp.exists():
            mk = pd.read_parquet(mkt_fp).reindex(self.index).ffill()
            self.market = np.nan_to_num(mk.to_numpy("float32"), nan=0.0)
        else:
            self.market = np.zeros((len(self.index), 66), dtype="float32")
        self.n_mkt = self.market.shape[1]

    # ------------------------------------------------------------------
    def valid_days(self, dates: pd.DatetimeIndex, purpose: str = "train",
                   min_stocks: int = 100) -> np.ndarray:
        """返回可用作样本的行下标：有足够回看历史、且当日有效股票够多。"""
        if purpose != "paper":
            S.assert_no_lockbox(dates, f"valid_days({purpose})")
        pos = self.index.get_indexer(dates)
        pos = pos[pos >= self.L - 1]
        return pos[self.m[pos].sum(1) >= min_stocks]

    def cols_for(self, rows: np.ndarray) -> np.ndarray:
        """这批天里出现过的有效股票并集。"""
        return np.where(self.m[rows].any(0))[0]

    def batch(self, rows: np.ndarray, cols: np.ndarray = None) -> dict:
        """组装一批天。rows 是行下标，cols 不传则自动取并集。"""
        rows = np.asarray(rows)
        if cols is None:
            cols = self.cols_for(rows)
        B, L, Nc = len(rows), self.L, len(cols)

        # 回看窗口：(B, L) 的行下标
        win = rows[:, None] - np.arange(L - 1, -1, -1)[None, :]
        xi = np.ix_(win.ravel(), cols)
        x = self.X[xi].reshape(B, L, Nc, self.F).transpose(0, 2, 1, 3)   # (B,N,L,F)
        feat_ok = self.feat_valid[xi].reshape(B, L, Nc).all(1)           # 窗口内全有效

        ri = np.ix_(rows, cols)
        t = lambda a, dt: torch.as_tensor(a, dtype=dt, device=self.device)
        return {
            "x":      t(np.ascontiguousarray(x), torch.float32),
            "y":      t(self.y[ri], torch.float32),
            "mask":   t(self.m[ri] & feat_ok, torch.bool),
            "market": t(self.market[rows], torch.float32),
            "attr":   {k: t(v[ri].astype("int64"), torch.long)
                       for k, v in self.attr.items()},
            "rows":   rows, "cols": cols,
        }

    def iter_batches(self, rows: np.ndarray, batch_days: int, shuffle=False,
                     rng: np.random.Generator = None):
        r = rows.copy()
        if shuffle:
            (rng or np.random.default_rng(0)).shuffle(r)
        for a in range(0, len(r), batch_days):
            sel = np.sort(r[a:a + batch_days])
            yield self.batch(sel)
