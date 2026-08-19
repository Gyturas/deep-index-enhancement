"""三折拼接回测曲线。

三块分开画、**不用双轴**——净值和超额量纲差一个量级，叠在同一张图上
要么超额被压成一条直线，要么净值被拉爆。分面是唯一诚实的画法。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import matplotlib.dates as mdates

sys.path.insert(0, str(Path(__file__).parent))
import config as C

# 参考调色板已验证的固定顺序槽位，不自创颜色
S1, S2 = "#2a78d6", "#eb6834"          # categorical slot 1 / 2
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8880"
GRID, SURF = "#e1e0d9", "#fcfcfb"

for f in ("PingFang HK", "Heiti TC", "Arial Unicode MS", "Songti SC"):
    if any(x.name == f for x in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
plt.rcParams.update({"axes.unicode_minus": False, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF})


def style(ax):
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)


def main():
    P = C.ROOT / "runs" / "portfolio"
    z = np.load(P / "ie_series.npz")
    zf = np.load(P / "ie_series_final.npz")
    # 开发期(三折) 与 锁定模拟盘 首尾相接，模拟盘那段单独着色
    ex = np.concatenate([z["ex"], zf["ex"]])
    bm = np.concatenate([z["bench"], zf["bench"]])
    dt = pd.to_datetime([str(d) for d in np.concatenate([z["dates"], zf["dates"]])])
    seam = np.append(z["seam"], len(z["ex"]))
    n_dev = len(z["ex"])                       # 模拟盘起点
    port = ex + bm
    nav_p, nav_b = (1 + port).cumprod(), (1 + bm).cumprod()
    nav_e = (1 + ex).cumprod()
    dd_e = nav_e / np.maximum.accumulate(nav_e) - 1

    fig, axes = plt.subplots(3, 1, figsize=(11, 10.5), sharex=True,
                             gridspec_kw={"height_ratios": [2.4, 1.6, 1.0], "hspace": 0.16})

    # ---- 1 净值
    ax = axes[0]
    # 模拟盘区间灰底 —— 这段从未参与训练/验证/选参，是唯一干净的样本外
    for ax_ in (axes[0], axes[1], axes[2]):
        ax_.axvspan(dt[n_dev], dt[-1], color=INK3, alpha=0.13, lw=0, zorder=0)
    ax.plot(dt, nav_b, color=S2, lw=2, label="基准（中证1000 指数，市值加权）", zorder=3)
    ax.plot(dt, nav_p, color=S1, lw=2, label="指增组合", zorder=4)
    ax.set_yscale("log")
    ax.set_yticks([0.6, 0.8, 1.0, 1.5, 2.0, 2.5])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylabel("净值（对数轴）", color=INK2, fontsize=10)
    ax.set_title("中证1000 指增：开发期三折 + 锁定模拟盘（灰底）",
                 color=INK, fontsize=13, fontweight="bold", loc="left", pad=14)
    ax.legend(frameon=False, fontsize=10, labelcolor=INK2, loc="upper left")
    # 末端直接标注，identity 不只靠颜色
    for v, c, t in ((nav_p[-1], S1, "组合"), (nav_b[-1], S2, "基准")):
        ax.annotate(f"{t} {v:.2f}", (dt[-1], v), xytext=(6, 0),
                    textcoords="offset points", color=c, fontsize=9,
                    va="center", fontweight="bold")
    style(ax)

    # ---- 2 累计超额（单序列，标题即标识，不需要图例）
    ax = axes[1]
    ax.plot(dt, nav_e, color=S1, lw=2, zorder=3)
    ax.axhline(1.0, color=INK3, lw=1, ls="--", zorder=2)
    ax.set_ylabel("累计超额净值", color=INK2, fontsize=10)
    d_ex, f_ex = ex[:n_dev], ex[n_dev:]
    A = lambda x: (1 + x).prod() ** (252 / len(x)) - 1
    SR = lambda x: x.mean() / x.std(ddof=1) * np.sqrt(252)
    ax.set_title(f"累计超额   开发期 {A(d_ex):+.2%}/IR {SR(d_ex):.2f}   |   "
                 f"模拟盘 {A(f_ex):+.2%}/IR {SR(f_ex):.2f}",
                 color=INK, fontsize=11, loc="left", pad=10)
    ax.annotate(f"{nav_e[-1]:.2f}", (dt[-1], nav_e[-1]), xytext=(6, 0),
                textcoords="offset points", color=S1, fontsize=9,
                va="center", fontweight="bold")
    style(ax)

    # ---- 3 超额回撤
    ax = axes[2]
    ax.fill_between(dt, dd_e * 100, 0, color=INK3, alpha=0.35, lw=0, zorder=3)
    ax.plot(dt, dd_e * 100, color=INK2, lw=1.2, zorder=4)
    ax.set_ylabel("超额回撤 %", color=INK2, fontsize=10)
    ax.set_title(f"超额回撤  最深 {dd_e.min():.1%}", color=INK, fontsize=11,
                 loc="left", pad=10)
    style(ax)

    # ---- 折边界
    for i, s in enumerate(seam):
        for ax_ in axes:
            ax_.axvline(dt[s], color=INK3, lw=1, ls=":", zorder=1)
        lbl = "锁定模拟盘起点" if i == len(seam) - 1 else f"换模型 → fold{i+1}"
        axes[0].annotate(lbl, (dt[s], axes[0].get_ylim()[1]), xytext=(4, -12),
                         textcoords="offset points", color=INK3, fontsize=8,
                         va="top", fontweight="bold" if i == len(seam) - 1 else "normal")
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.text(0.008, 0.005,
             "基准=中证1000 指数(自由流通市值加权) · 扣 15bp 双边 · 行业中性±1% · 市值中性 · "
             "个股偏离≤1pp · 换手惩罚 κ=15bp · 灰底段从未参与训练/验证/选参",
             color=INK3, fontsize=8)
    out = C.ROOT / "docs" / "backtest_ie_final.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"已保存 {out}")
    print(f"组合末值 {nav_p[-1]:.3f}  基准末值 {nav_b[-1]:.3f}  超额末值 {nav_e[-1]:.3f}")


if __name__ == "__main__":
    main()
