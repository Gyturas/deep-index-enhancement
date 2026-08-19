"""抓取 ST 状态与申万行业分类（cjpy / 天软），按月末采样，可断点续跑。

设计要点
--------
* **可恢复**：每个采样日单独落一个 parquet，重跑时跳过已存在的文件。
  （沿用既有教训：cjpy 长任务必须能复用缓存，不能一次跑到底。）
* **月末采样**：ST 与行业均为事件驱动的慢变量，日频抓取无收益。
  代价是 ST 转换日期被解析到"月"粒度——这一点在计划书 §数据 里显式记录。
  真正"完全不可交易"的一字板由价格规则识别，不依赖 ST。
* **失败不中断**：单日失败记入日志，继续下一日；末尾汇总缺失清单。
"""
import sys, time, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import SRC_PRICE, CACHE, LOGS, META, META_FACTORS, to_ts, to_wind

import cjpy

CH = CACHE / "meta_raw"
CH.mkdir(parents=True, exist_ok=True)


def sample_dates() -> list:
    """交易日历上的每月最后一个交易日 + 全样本最后一日。"""
    idx = pd.read_parquet(SRC_PRICE / "close.parquet", columns=["000001.SZ"]).index
    ends = pd.Series(idx, index=idx).groupby(idx.to_period("M")).max().tolist()
    if idx[-1] not in ends:
        ends.append(idx[-1])
    return sorted(set(ends))


def universe() -> list:
    cols = pd.read_parquet(SRC_PRICE / "close.parquet").columns
    return sorted(cols)


def main(batch: int = 1500):
    dates = sample_dates()
    codes_wind = universe()
    codes_ts = [to_ts(c) for c in codes_wind]
    print(f"采样日 {len(dates)} 个 ({dates[0].date()} ~ {dates[-1].date()})，"
          f"universe {len(codes_ts)} 只，分 {-(-len(codes_ts)//batch)} 批/日", flush=True)

    log = open(LOGS / "fetch_meta.log", "a")
    done = skipped = failed = 0
    t00 = time.time()

    for i, d in enumerate(dates):
        ds = d.strftime("%Y%m%d")
        fp = CH / f"{ds}.parquet"
        if fp.exists():
            skipped += 1
            continue
        t0 = time.time()
        parts, err = [], None
        for k in range(0, len(codes_ts), batch):
            chunk = codes_ts[k:k + batch]
            for attempt in range(3):
                try:
                    r = cjpy.get_factor_data(code=chunk, date=[ds], factors=META_FACTORS)
                    if r is not None and len(r):
                        parts.append(r)
                    break
                except Exception as e:
                    err = f"{type(e).__name__}: {str(e)[:120]}"
                    time.sleep(1.5 * (attempt + 1))
            else:
                print(f"  [{ds}] chunk{k} 三次失败 {err}", file=log, flush=True)
        if not parts:
            failed += 1
            print(f"[{ds}] 无数据 {err}", file=log, flush=True)
            continue
        df = pd.concat(parts, ignore_index=True)
        df["代码"] = df["代码"].map(to_wind)
        df.to_parquet(fp, index=False)
        done += 1
        if done % 10 == 0 or i == len(dates) - 1:
            el = time.time() - t00
            print(f"[{i+1}/{len(dates)}] {ds} n={len(df)} "
                  f"{time.time()-t0:.1f}s | 已抓{done} 跳过{skipped} 失败{failed} "
                  f"| 累计{el/60:.1f}min", flush=True)

    print(f"完成：新抓 {done}，跳过 {skipped}，失败 {failed}", flush=True)
    (META / "fetch_meta_status.json").write_text(json.dumps(
        {"n_dates": len(dates), "fetched": done, "skipped": skipped, "failed": failed,
         "first": str(dates[0].date()), "last": str(dates[-1].date())},
        ensure_ascii=False, indent=2), encoding="utf-8")
    log.close()


if __name__ == "__main__":
    main()
