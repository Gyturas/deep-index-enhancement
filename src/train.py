"""训练驱动：walk-forward x 多种子，验证指标是 RankIC。

和 v1 的几处修正
----------------
* 验证指标从组合夏普换成 **RankIC**。IC 的样本量是"天数 x 股票数"，
  同样长度的验证段，估计精度高一个量级——v1 用 63 天的组合夏普选种子，
  四个种子的差距全在一个标准误以内，等于抛硬币。
* 种子**全部平均，不做选择**。验证段短的时候，选择带来的噪音大于收益。
* 每若干步打印训练/验证指标，`python3 -u` 不缓冲，长任务能看到进度。
"""
from __future__ import annotations
import sys, json, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import config as C
import splits as S
from dataset import Panels
from model import DeepFactor
from loss import objective, summarize, summarize_full

RUNS = C.ROOT / "runs"
RUNS.mkdir(exist_ok=True)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def check_machine(cfg):
    """启动自检。上一轮在本机白跑了 10 小时——CPU 时间只有 144 分钟，
    其余全被别的进程（微信小程序占 2 核、Claude 客户端占 1 核）抢走。
    这里先看一眼实际可用资源，不合适就直接警告，别等跑完才发现。"""
    import os, shutil
    n = os.cpu_count() or 1
    try:
        load1 = os.getloadavg()[0]
    except OSError:
        load1 = float("nan")
    free_gb = shutil.disk_usage(str(C.ROOT)).free / 2**30
    log(f"机器自检: {n} 核 | 1 分钟负载 {load1:.1f} | 磁盘剩余 {free_gb:.0f}GB "
        f"| device={cfg['device']}")
    if cfg["device"] == "cpu" and load1 > n * 0.5:
        log(f"  [警告] 负载已占用约 {load1:.1f}/{n} 核，训练会被严重拖慢。"
            f"建议关掉占 CPU 的应用，或改用 --device cuda")


def default_cfg():
    return dict(
        level="L0", feature_set=C.FEATURE_SET_DEFAULT, h=C.LABEL_H_MAIN,
        lookback=C.LOOKBACK, d=64, n_head=4, dropout=0.3, gate_beta=5.0,
        lr=1e-4, weight_decay=1e-4, grad_clip=1.0,
        lam_rank=0.1, pairs_per_stock=4, loss_kind="ic",
        batch_days=32, iters=300, eval_every=10, patience=8, ema=0.45,
        seeds=12, device="cpu",
    )


# ------------------------------------------------------------------ 评估
@torch.no_grad()
def evaluate(model, panels, rows, cfg):
    model.eval()
    P, Y, M = [], [], []
    for b in panels.iter_batches(rows, cfg["batch_days"]):
        P.append(model(b).cpu()); Y.append(b["y"].cpu()); M.append(b["mask"].cpu())
    model.train()
    # 各批股票列不同，按最大列数右侧补零对齐后拼接
    w = max(p.shape[1] for p in P)
    pad = lambda t, v: torch.nn.functional.pad(t, (0, w - t.shape[1]), value=v)
    return summarize(torch.cat([pad(p, 0.) for p in P]),
                     torch.cat([pad(y, 0.) for y in Y]),
                     torch.cat([pad(m, False) for m in M]))


# ------------------------------------------------------------------ 单种子
def fit_one(panels, fold, cfg, seed: int, tag: str):
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    tr = panels.valid_days(fold["train"], "train")
    va = panels.valid_days(fold["val"], "val")

    model = DeepFactor(panels.F, panels.n_ind1, panels.n_ind2, level=cfg["level"],
                       d=cfg["d"], n_head=cfg["n_head"], dropout=cfg["dropout"],
                       n_bucket=C.N_BUCKET + 1, n_mkt=panels.n_mkt,
                       gate_beta=cfg["gate_beta"]).to(cfg["device"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])

    best, ema, bad, best_state, hist = -9e9, None, 0, None, []
    t0 = time.time()
    it = 0
    while it < cfg["iters"]:
        for b in panels.iter_batches(tr, cfg["batch_days"], shuffle=True, rng=rng):
            it += 1
            loss, diag = objective(model(b), b["y"], b["mask"],
                                   cfg["lam_rank"], cfg["pairs_per_stock"],
                                   cfg["loss_kind"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()

            if it % cfg["eval_every"] == 0:
                ev = evaluate(model, panels, va, cfg)
                ema = ev["rank_ic"] if ema is None else \
                    cfg["ema"] * ev["rank_ic"] + (1 - cfg["ema"]) * ema
                hist.append({"it": it, "train_ic": float(diag["ic"]),
                             "val_rank_ic": ev["rank_ic"], "val_ema": ema,
                             "alpha": model.rezero_alpha})
                log(f"    it{it:4d} train_IC {float(diag['ic']):+.4f} "
                    f"val_RankIC {ev['rank_ic']:+.4f} ema {ema:+.4f} "
                    f"alpha {model.rezero_alpha:+.3f} ({(time.time()-t0)/60:.1f}min)")
                if ema > best + 1e-4:
                    best, bad = ema, 0
                    best_state = {k: v.detach().clone()
                                  for k, v in model.state_dict().items()}
                else:
                    bad += 1
            if it >= cfg["iters"] or bad >= cfg["patience"]:
                break
        if bad >= cfg["patience"]:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"seed": seed, "best_val_ema": best, "iters": it,
                   "minutes": (time.time() - t0) / 60,
                   "rezero_alpha": model.rezero_alpha, "hist": hist}


# ------------------------------------------------------------------ 一折
def run_fold(panels, fold, cfg, tag: str, keep_hist: bool = False):
    te = panels.valid_days(fold["test"], "test")
    preds, infos = [], []
    for s in range(cfg["seeds"]):
        model, info = fit_one(panels, fold, cfg, s, tag)
        infos.append(info)
        log(f"  {tag} seed{s:02d} val_ema {info['best_val_ema']:+.4f} "
            f"iters {info['iters']} alpha {info['rezero_alpha']:+.3f} "
            f"{info['minutes']:.1f}min")
        with torch.no_grad():
            model.eval()
            preds.append([(model(b).cpu(), b["y"].cpu(), b["mask"].cpu())
                          for b in panels.iter_batches(te, cfg["batch_days"])])

    # 预测落盘：6MB 而已，存了之后组合构建就能脱离 GPU 在本地反复迭代。
    # （上一轮 L0 没存，导致回测得重训一遍——这是当时的疏漏。）
    K = len(preds)
    # 种子集成：对**打分**求平均（不做选择，验证段短时选择的噪音大于收益）
    P = [sum(preds[k][i][0] for k in range(K)) / K for i in range(len(preds[0]))]
    Y = [preds[0][i][1] for i in range(len(preds[0]))]
    M = [preds[0][i][2] for i in range(len(preds[0]))]
    w = max(p.shape[1] for p in P)
    pad = lambda t, v: torch.nn.functional.pad(t, (0, w - t.shape[1]), value=v)
    # 测试段用完整口径：NW 调整的显著性 + lambda 换手匹配扫描
    ens = summarize_full(torch.cat([pad(p, 0.) for p in P]),
                         torch.cat([pad(y, 0.) for y in Y]),
                         torch.cat([pad(m, False) for m in M]), h=cfg["h"])
    single = summarize(torch.cat([pad(preds[0][i][0], 0.) for i in range(len(P))]),
                       torch.cat([pad(y, 0.) for y in Y]),
                       torch.cat([pad(m, False) for m in M]))
    pdir = RUNS / "preds"; pdir.mkdir(exist_ok=True)
    np.savez_compressed(pdir / f"{tag}.npz",
                        score=torch.cat([pad(p, 0.) for p in P]).numpy().astype("float32"),
                        label=torch.cat([pad(y, 0.) for y in Y]).numpy().astype("float32"),
                        mask=torch.cat([pad(m, False) for m in M]).numpy())

    return {"ensemble": ens, "single_seed0": single,
            "seeds": [i if keep_hist else {k: v for k, v in i.items() if k != "hist"}
                      for i in infos],
            "mean_alpha": float(np.mean([i["rezero_alpha"] for i in infos]))}


# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="L0", choices=list(C.LEVELS))
    ap.add_argument("--h", type=int, default=None)
    ap.add_argument("--feature-set", default=None)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--batch-days", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--loss", default="ic", choices=("ic", "mse"))
    ap.add_argument("--lookback", type=int, default=None)
    ap.add_argument("--fold", type=int, default=None,
                    help="只跑某一折（探路用）；不传则跑全部三折")
    ap.add_argument("--keep-hist", action="store_true",
                    help="保存逐次评估的学习曲线（探路必开）")
    a = ap.parse_args()

    cfg = default_cfg()
    cfg["level"], cfg["device"], cfg["loss_kind"] = a.level, a.device, a.loss
    if a.lookback:
        cfg["lookback"] = a.lookback
    for k in ("h", "seeds", "iters", "batch_days"):
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)
    if a.feature_set:
        cfg["feature_set"] = a.feature_set
    tag = a.tag or f"{cfg['level']}_h{cfg['h']}_{cfg['feature_set']}_{cfg['loss_kind']}"

    check_machine(cfg)
    panels = Panels(cfg["feature_set"], cfg["h"], cfg["lookback"], cfg["device"])
    log(f"{tag} | 特征 {panels.F} 维 | 行业 一级{panels.n_ind1}/二级{panels.n_ind2} "
        f"| 回看 {panels.L} | 市场状态 {panels.n_mkt} 维")
    log(S.summary(panels.index))

    out = {"config": {k: v for k, v in cfg.items()}, "folds": {}}
    folds = S.walk_forward(panels.index)
    if a.fold is not None:
        folds = [x for x in folds if x["fold"] == a.fold]
    for fold in folds:
        f = f"fold{fold['fold']}"
        log(f"=== {f} [{fold['label']}] 训练日 {len(panels.valid_days(fold['train'],'train'))} "
            f"验证日 {len(panels.valid_days(fold['val'],'val'))} "
            f"测试日 {len(panels.valid_days(fold['test'],'test'))} ===")
        r = run_fold(panels, fold, cfg, f"{tag}-{f}", keep_hist=a.keep_hist)
        out["folds"][f] = r
        e = r["ensemble"]
        log(f"  >>> {f} 集成 IC {e['ic']:+.4f}  RankIC {e['rank_ic']:+.4f}  "
            f"ICIR {e['icir']:+.2f}  RankICIR {e['rank_icir']:+.2f}  "
            f"NW-t {e['t_ic_nw']:+.2f}  (单种子 RankIC {r['single_seed0']['rank_ic']:+.4f})")
        sw = " ".join(f"lam{k[3:]}:{v['rank_ic']:+.4f}/换手{v['turnover']:.3f}"
                      for k, v in e["lam_sweep"].items())
        log(f"      换手匹配 {sw}")

    ric = [out["folds"][f]["ensemble"]["rank_ic"] for f in out["folds"]]
    out["summary"] = {"rank_ic_median": float(np.median(ric)),
                      "rank_ic_all": ric,
                      "sign_consistent": bool(all(x > 0 for x in ric))}
    (RUNS / f"{tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    log(f"三折 RankIC {['%+.4f' % x for x in ric]} | 中位 {np.median(ric):+.4f} "
        f"| 符号一致 {out['summary']['sign_consistent']}")


if __name__ == "__main__":
    main()
