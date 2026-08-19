"""分位多头组合回测：把因子打分转成实际收益。

为什么必须单独做这件事
----------------------
IC 高不等于赚钱。标签是**残差**收益（剥掉了市场、市值、行业），
但组合最终赚的是**原始**收益。两者之间隔着：

* 剥掉的那部分暴露会不会反过来拖累（比如选出来的票系统性偏小市值）；
* 换手成本；
* 分位截断（只取前 10% 意味着只用了排序的一小段信息）。

所以这里一律用**原始次日收益**结算，只有 IC 那一栏用残差标签。

基准
----
中证1000 成分股**等权**，每日再平衡，同一套可交易掩码。
这是指增的正确基准——不是指数本身（市值加权），因为我们的组合是等权的，
拿等权比等权才公平；市值加权指数另列一行参考。
"""
from __future__ import annotations
import sys, json, argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import config as C
import splits as S
from dataset import Panels
from model import DeepFactor
from loss import smooth_signal
import train as T

OUT = C.ROOT / "runs" / "portfolio"
OUT.mkdir(parents=True, exist_ok=True)
TD = 252


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ------------------------------------------------------------------ 指标
def perf(r: np.ndarray, bench: np.ndarray = None) -> dict:
    r = np.asarray(r, dtype="float64")
    n = len(r)
    if n < 20:
        return {}
    ann = float((1 + r).prod() ** (TD / n) - 1)
    vol = float(r.std(ddof=1) * np.sqrt(TD))
    sr = float(r.mean() / (r.std(ddof=1) + 1e-12) * np.sqrt(TD))
    nav = (1 + r).cumprod()
    mdd = float((nav / np.maximum.accumulate(nav) - 1).min())
    out = {"年化": ann, "波动": vol, "夏普": sr, "最大回撤": mdd,
           "卡玛": ann / abs(mdd) if mdd < 0 else np.nan, "天数": n}
    if bench is not None:
        e = r - np.asarray(bench, dtype="float64")
        out["超额年化"] = float((1 + r).prod() ** (TD / n) - (1 + bench).prod() ** (TD / n))
        out["跟踪误差"] = float(e.std(ddof=1) * np.sqrt(TD))
        out["信息比"] = float(e.mean() / (e.std(ddof=1) + 1e-12) * np.sqrt(TD))
        # Newey-West 修正的超额 t（h 日标签重叠 -> 序列自相关）
        from loss import newey_west_se
        se = newey_west_se(torch.tensor(e), 4)
        out["超额t_NW"] = float(e.mean() / se) if se == se else np.nan
    return out


def backtest(score: np.ndarray, mask: np.ndarray, fwd1: np.ndarray,
             q: float, lam: float, cost_bp: float) -> tuple:
    """分位多头。score/mask/fwd1: (T, N)。返回 (净收益序列, 毛收益, 日均换手)。

    q=0.1 表示每日取打分前 10%，等权持有到次日。
    lam 是日频部分调仓：w_t = lam*w_{t-1} + (1-lam)*w_target。
    """
    T_, N = score.shape
    w_prev = np.zeros(N)
    net, gross, turns = [], [], []
    for t in range(T_):
        m = mask[t]
        k = int(m.sum())
        if k < 50:
            continue
        s = np.where(m, score[t], -np.inf)
        n_pick = max(int(round(k * q)), 1)
        idx = np.argpartition(-s, n_pick - 1)[:n_pick]
        w_tgt = np.zeros(N)
        w_tgt[idx] = 1.0 / n_pick

        w = lam * w_prev + (1 - lam) * w_tgt if lam > 0 else w_tgt
        w = np.where(m, w, 0.0)
        ssum = w.sum()
        if ssum <= 0:
            continue
        w = w / ssum                                  # 满仓归一

        to = np.abs(w - w_prev).sum()
        g = float(np.nansum(w * fwd1[t]))
        gross.append(g)
        net.append(g - to * cost_bp * 1e-4)
        turns.append(to)
        w_prev = w
    return np.array(net), np.array(gross), float(np.mean(turns)) if turns else 0.0


def backtest_ls(score, mask, fwd1, q, lam, cost_bp, borrow_bp_annual=800.0):
    """多空拆解：多头前 q%、空头后 q%，都等权。

    拆成三段回答"空头有没有东西"：
        多头超额 = 前q%收益 − 等权全池
        空头超额 = 等权全池 − 后q%收益      （后q%跑输得越多，空头信号越强）
        多空价差 = 前q% − 后q% = 多头超额 + 空头超额

    ⚠️ A 股这个多空是**不可执行**的诊断，不是策略：
    中证1000 里绝大多数不在融券标的名单，即便在，券源也极少。
    borrow_bp_annual 默认 800bp（8%/年）已经是乐观假设。
    这里算它只为回答"信号是不是对称的"。
    """
    T_, N = score.shape
    wl_prev = np.zeros(N); ws_prev = np.zeros(N)
    L, S, B = [], [], []
    for t in range(T_):
        m = mask[t]
        k = int(m.sum())
        if k < 50:
            continue
        s_ = np.where(m, score[t], -np.inf)
        n = max(int(round(k * q)), 1)
        top = np.argpartition(-s_, n - 1)[:n]
        s2 = np.where(m, score[t], np.inf)
        bot = np.argpartition(s2, n - 1)[:n]

        wl_t = np.zeros(N); wl_t[top] = 1.0 / n
        ws_t = np.zeros(N); ws_t[bot] = 1.0 / n
        wl = (lam * wl_prev + (1 - lam) * wl_t) if lam > 0 else wl_t
        ws = (lam * ws_prev + (1 - lam) * ws_t) if lam > 0 else ws_t
        wl = np.where(m, wl, 0.0); ws = np.where(m, ws, 0.0)
        if wl.sum() <= 0 or ws.sum() <= 0:
            continue
        wl /= wl.sum(); ws /= ws.sum()

        bench = float(np.nansum(np.where(m, 1.0 / k, 0.0) * fwd1[t]))
        rl = float(np.nansum(wl * fwd1[t])) - np.abs(wl - wl_prev).sum() * cost_bp * 1e-4
        rs = float(np.nansum(ws * fwd1[t])) - np.abs(ws - ws_prev).sum() * cost_bp * 1e-4
        L.append(rl - bench)                      # 多头超额
        S.append(bench - rs - borrow_bp_annual * 1e-4 / TD)   # 空头超额，扣融券费
        B.append(rl - rs - borrow_bp_annual * 1e-4 / TD)      # 多空价差
        wl_prev, ws_prev = wl, ws
    return np.array(L), np.array(S), np.array(B)


def bench_equal(mask: np.ndarray, fwd1: np.ndarray, cost_bp: float) -> np.ndarray:
    """等权全池，每日再平衡。基准也要扣换手成本（成分调整与停牌复牌带来的）。"""
    T_, N = mask.shape
    w_prev = np.zeros(N)
    out = []
    for t in range(T_):
        m = mask[t]
        k = int(m.sum())
        if k < 50:
            continue
        w = np.where(m, 1.0 / k, 0.0)
        to = np.abs(w - w_prev).sum()
        out.append(float(np.nansum(w * fwd1[t])) - to * cost_bp * 1e-4)
        w_prev = w
    return np.array(out)


def _run_and_report(score, M, R, a, res, f):
    """回测 + 打印。训练路径与缓存路径共用。"""
    bm = bench_equal(M, R, a.cost_bp)
    row = {"等权全池(基准)": perf(bm)}
    for q in [float(x) for x in a.quantiles.split(",")]:
        for lam in [float(x) for x in a.lams.split(",")]:
            net, gross, to = backtest(score, M, R, q, lam, a.cost_bp)
            if len(net) < 20:
                continue
            key = f"前{int(q*100)}%_lam{lam}"
            row[key] = perf(net, bm[:len(net)])
            row[key]["日均换手"] = to
            row[key]["毛年化"] = float((1 + gross).prod() ** (TD / len(gross)) - 1)
    # --- 多空拆解（诊断，非可执行策略）
    ls = {}
    for q in (0.10, 0.20, 0.30):
        Lx, Sx, Bx = backtest_ls(score, M, R, q, 0.8, a.cost_bp)
        if len(Lx) < 20:
            continue
        ann = lambda x: float((1 + x).prod() ** (TD / len(x)) - 1)
        sr = lambda x: float(x.mean() / (x.std(ddof=1) + 1e-12) * np.sqrt(TD))
        ls[f"q{int(q*100)}"] = {"多头超额年化": ann(Lx), "多头IR": sr(Lx),
                                "空头超额年化": ann(Sx), "空头IR": sr(Sx),
                                "多空年化": ann(Bx), "多空夏普": sr(Bx)}
    res["folds"].setdefault("_ls", {})[f] = ls
    log(f"  {'多空拆解':<14}{'多头超额':>9}{'多头IR':>8}{'空头超额':>9}{'空头IR':>8}"
        f"{'多空年化':>9}{'多空夏普':>9}")
    for k, v in ls.items():
        log(f"  {k:<16}{v['多头超额年化']:>9.1%}{v['多头IR']:>8.2f}"
            f"{v['空头超额年化']:>9.1%}{v['空头IR']:>8.2f}"
            f"{v['多空年化']:>9.1%}{v['多空夏普']:>9.2f}")

    res["folds"][f] = row
    b = row["等权全池(基准)"]
    log(f"  {'组合':<18}{'年化':>8}{'夏普':>7}{'超额年化':>10}{'信息比':>7}"
        f"{'NW-t':>7}{'换手':>7}{'回撤':>8}")
    log(f"  {'等权全池(基准)':<16}{b['年化']:>8.1%}{b['夏普']:>7.2f}"
        f"{'':>10}{'':>7}{'':>7}{'':>7}{b['最大回撤']:>8.1%}")
    for k, v in row.items():
        if k.startswith("前"):
            log(f"  {k:<18}{v['年化']:>8.1%}{v['夏普']:>7.2f}{v['超额年化']:>10.1%}"
                f"{v['信息比']:>7.2f}{v['超额t_NW']:>7.2f}{v['日均换手']:>7.1%}"
                f"{v['最大回撤']:>8.1%}")


# ------------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="L0")
    ap.add_argument("--h", type=int, default=5)
    ap.add_argument("--feature-set", default="full")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--batch-days", type=int, default=16)
    ap.add_argument("--cost-bp", type=float, default=15.0,
                    help="单边成本 bp：印花税5(卖出)+佣金2.5+半价差5+冲击，双边合计约15")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="pf_L0")
    ap.add_argument("--from-cache", action="store_true",
                    help="跳过训练，直接读已存的预测。本地改分位/成本时用这个")
    ap.add_argument("--quantiles", default="0.05,0.10,0.20,0.30,0.50")
    ap.add_argument("--lams", default="0.0,0.8")
    ap.add_argument("--final", action="store_true",
                    help="训练 final 模型并在**锁定模拟盘**上推理。不可逆，只能跑一次。")
    a = ap.parse_args()

    cfg = T.default_cfg()
    cfg.update(level=a.level, h=a.h, feature_set=a.feature_set, seeds=a.seeds,
               iters=a.iters, batch_days=a.batch_days, device=a.device)

    panels = Panels(cfg["feature_set"], cfg["h"], cfg["lookback"], cfg["device"])
    close = pd.read_parquet(C.OUT / "A_close.parquet")
    # **原始**次日收益——组合赚的是这个，不是残差
    fwd1_all = (close.ffill().shift(-1) / close.ffill() - 1.0).to_numpy("float32")
    fwd1_all = np.nan_to_num(np.clip(fwd1_all, -0.25, 0.25), nan=0.0)

    res = {"config": {k: v for k, v in cfg.items()}, "cost_bp": a.cost_bp, "folds": {}}
    if a.final:
        fin = S.final_fit(panels.index)
        fin["fold"] = "final"
        folds, purpose = [fin], "paper"
        log("=" * 70)
        log("【锁定模拟盘】训练 %s~%s  验证 %s~%s  ->  推理 %s~%s" % (
            fin["train"][0].date(), fin["train"][-1].date(),
            fin["val"][0].date(), fin["val"][-1].date(),
            fin["paper"][0].date(), fin["paper"][-1].date()))
        log("这段数据此前从未参与训练/验证/超参/选种子。跑完即失去干净性。")
        log("=" * 70)
    else:
        folds, purpose = S.walk_forward(panels.index), "test"

    for fold in folds:
        f = f"fold{fold['fold']}" if fold["fold"] != "final" else "final"
        te = panels.valid_days(fold[purpose if purpose in fold else "test"], purpose)
        log(f"=== {f} [{fold['label']}] 测试日 {len(te)} ===")

        # --- 训练集成并拿测试段打分（或从缓存读）
        cache = OUT / f"{a.tag}_{f}_pred.npz"
        if a.from_cache and cache.exists():
            z = np.load(cache)
            score, M, R = z["score"], z["mask"], z["fwd1"]
            log(f"  从缓存读取预测 {cache.name}  {score.shape}")
            _run_and_report(score, M, R, a, res, f)
            continue
        acc, cols_ref = None, None
        for sd in range(cfg["seeds"]):
            model, info = T.fit_one(panels, fold, cfg, sd, f"{a.tag}-{f}")
            model.eval()
            with torch.no_grad():
                ps = []
                for b in panels.iter_batches(te, cfg["batch_days"]):
                    p = np.full((b["y"].shape[0], panels.N_all), np.nan, dtype="float32")
                    p[:, b["cols"]] = model(b).cpu().numpy()
                    ps.append(p)
                P = np.concatenate(ps)
            acc = P if acc is None else acc + P
            if sd % 4 == 3:
                log(f"  seed {sd+1}/{cfg['seeds']} val_ema {info['best_val_ema']:+.4f}")
        score = acc / cfg["seeds"]

        # 对齐掩码与收益到同样的 (天, 股票) 网格
        M = np.zeros_like(score, dtype=bool)
        R = np.zeros_like(score)
        for i, t in enumerate(te):
            M[i] = panels.m[t]
            R[i] = fwd1_all[t]
        score = np.nan_to_num(score, nan=-9e9)

        # **预测落盘**：只有 6MB，存下来之后组合构建就能脱离 GPU，
        # 在本地反复改分位、成本、约束都不用重训。
        np.savez_compressed(cache, score=score.astype("float32"), mask=M,
                            fwd1=R.astype("float32"),
                            dates=np.array([str(panels.index[t].date()) for t in te]),
                            codes=np.array(list(panels.codes)))
        log(f"  预测已存 {cache.name} ({cache.stat().st_size/1e6:.1f}MB)")
        _run_and_report(score, M, R, a, res, f)
        continue



    (OUT / f"{a.tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2,
                                                  default=float), encoding="utf-8")
    log(f"结果写入 runs/portfolio/{a.tag}.json")


if __name__ == "__main__":
    main()
