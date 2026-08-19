"""时间切分：开发期 / 锁定模拟盘。

铁律
----
**2025-01-01 起的所有数据是锁定的模拟盘（lockbox）**，不得用于：
训练、验证、超参搜索、种子排序、早停、特征标准化统计量的估计。
任何函数只要拿到 ``purpose != "paper"``，返回的区间一定不越过 LOCKBOX_START。

这条不是风格偏好。样本外结论的可信度完全取决于这段数据从未被看过；
一旦被用于任何形式的选择，它就退化成又一个验证集。
"""
from __future__ import annotations
import pandas as pd

# ---------------------------------------------------------------- 边界
LOCKBOX_START = "2025-01-01"     # 含此日起为模拟盘，开发期一律不可见
DEV_END = "2024-12-31"           # 开发期最后一日

BACKBONE_START = "2003-01-02"    # 全A 时序主干预训练起点
XSEC_START = "2014-10-31"        # 截面/图可用起点（中证1000 权重面板首日）

VAL_FRAC = 0.10                  # 验证集占训练段末尾的比例（按时间切，不随机）

# walk-forward：每折的测试区间（均在开发期内）
WF_TEST_BLOCKS = [
    ("2019-01-01", "2020-12-31"),
    ("2021-01-01", "2022-12-31"),
    ("2023-01-01", "2024-12-31"),
]


class LockboxViolation(RuntimeError):
    pass


def assert_no_lockbox(idx: pd.DatetimeIndex, what: str = ""):
    """任何进入训练/验证的日期索引都要过这道闸。"""
    bad = idx[idx >= pd.Timestamp(LOCKBOX_START)]
    if len(bad):
        raise LockboxViolation(
            f"{what}: {len(bad)} 个日期落入锁定模拟盘区间 "
            f"({bad[0].date()} ~ {bad[-1].date()})，起点 {LOCKBOX_START}")


def dev_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """把任意日期索引裁到开发期。"""
    return idx[idx <= pd.Timestamp(DEV_END)]


def paper_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """模拟盘区间。只有 purpose='paper' 的推理路径可以拿到。"""
    return idx[idx >= pd.Timestamp(LOCKBOX_START)]


def walk_forward(idx: pd.DatetimeIndex, start: str = XSEC_START) -> list[dict]:
    """生成 walk-forward 折。训练集扩张，验证集取训练段末尾 VAL_FRAC（按时间）。"""
    idx = dev_index(idx)
    folds = []
    for k, (t0, t1) in enumerate(WF_TEST_BLOCKS):
        tr_all = idx[(idx >= pd.Timestamp(start)) & (idx < pd.Timestamp(t0))]
        if len(tr_all) < 500:
            continue
        cut = int(len(tr_all) * (1 - VAL_FRAC))
        tr, va = tr_all[:cut], tr_all[cut:]
        te = idx[(idx >= pd.Timestamp(t0)) & (idx <= pd.Timestamp(t1))]
        for nm, s in [("train", tr), ("val", va), ("test", te)]:
            assert_no_lockbox(s, f"fold{k}-{nm}")
        folds.append({"fold": k, "train": tr, "val": va, "test": te,
                      "label": f"{t0[:4]}-{t1[:4]}"})
    return folds


def final_fit(idx: pd.DatetimeIndex, start: str = XSEC_START) -> dict:
    """最终模型：用全部开发期数据拟合，推理区间为锁定模拟盘。

    注意 ``paper`` 只用于​推理与事后评估，绝不回流到训练或选种子。
    """
    dev = dev_index(idx)
    tr_all = dev[dev >= pd.Timestamp(start)]
    cut = int(len(tr_all) * (1 - VAL_FRAC))
    tr, va = tr_all[:cut], tr_all[cut:]
    assert_no_lockbox(tr, "final-train")
    assert_no_lockbox(va, "final-val")
    return {"train": tr, "val": va, "paper": paper_index(idx), "label": "final"}


def summary(idx: pd.DatetimeIndex) -> str:
    out = ["时间切分", "=" * 64]
    for f in walk_forward(idx):
        out.append(f"  fold{f['fold']} [{f['label']}] "
                   f"train {f['train'][0].date()}~{f['train'][-1].date()} ({len(f['train'])}d) | "
                   f"val {len(f['val'])}d | test {f['test'][0].date()}~{f['test'][-1].date()} "
                   f"({len(f['test'])}d)")
    fin = final_fit(idx)
    out.append(f"  final    train {fin['train'][0].date()}~{fin['train'][-1].date()} "
               f"({len(fin['train'])}d) | val {len(fin['val'])}d")
    p = fin["paper"]
    out.append(f"  锁定模拟盘 {p[0].date()}~{p[-1].date()} ({len(p)}d) —— 训练/验证/选种子全程不可见")
    return "\n".join(out)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    import config as C
    print(summary(pd.read_parquet(C.OUT / "A_close.parquet").index))
