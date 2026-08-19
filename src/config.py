"""DeePM-中证1000 项目配置：路径、universe、涨跌停板规则。"""
from pathlib import Path
import re

# ---------------------------------------------------------------- 路径
HOME = Path.home()
DESK = HOME / "Desktop"
# ROOT 从本文件位置推导，不硬编码——同一份代码要在本机和 pod 上都能跑。
ROOT = Path(__file__).resolve().parent.parent

SRC_PRICE = DESK / "中证1000个股择时" / "中证1000数据" / "float32"   # 后复权全A面板 (Wind码)
SRC_PANEL = DESK / "中证1000截面" / "results" / "panels"            # 截面辅助面板 (天软码)

OUT = ROOT / "data" / "aligned"
META = ROOT / "data" / "meta"
CACHE = ROOT / "data" / "cache"
LOGS = ROOT / "logs"
for _p in (OUT, META, CACHE, LOGS):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 样本期
# 价格面板从 2002 起（共享时序主干在全A上训练）；成分/行业等截面信息从中证1000发布日起。
# 价格面板自 2002-01-04 起，但 vol/turn 自 2003-01-02 才有；成交量是判定停牌/一字板的
# 必要条件，故 Tier A 起点取二者交集的首日。
DATE_START_A = "2003-01-02"      # Tier A: 全A
DATE_START_B = "2014-10-31"      # Tier B: 中证1000 成分（权重面板首日）

# 涨跌停判定容差：涨停价按未复权价四舍五入到 0.01 元，换算成复权收益率后
# 与名义幅度有微小偏离；低价股偏离更大。0.005 是覆盖多数情形的保守带宽。
LIMIT_TOL = 0.005
NEW_LISTING_EXCL_DAYS = 60       # 上市后剔除的交易日数（论文 burn-in=21 在个股上不足）

# ---------------------------------------------------------------- 代码规范化
_TS_RE = re.compile(r"^(SH|SZ|BJ)(\d{6})$")


def to_wind(code: str) -> str:
    """天软码 'SH600004' -> Wind码 '600004.SH'；已是 Wind 码则原样返回。"""
    m = _TS_RE.match(str(code))
    return f"{m.group(2)}.{m.group(1)}" if m else str(code)


def to_ts(code: str) -> str:
    """Wind码 '600004.SH' -> 天软码 'SH600004'（cjpy 用）。"""
    s = str(code)
    return "".join(s.split(".")[::-1]) if "." in s else s


# ---------------------------------------------------------------- 板块与涨跌停
# 交易所对涨跌停的规定按板块 + 是否ST + 生效日期分档。
GEM_WIDEN_DATE = "2020-08-24"    # 创业板注册制改革，涨跌幅 10% -> 20%
BSE_START_DATE = "2021-11-15"    # 北交所开市，涨跌幅 30%


def board_of(code: str) -> str:
    """按 Wind 码判定板块。"""
    num, _, suf = str(code).partition(".")
    if suf == "BJ":
        return "BSE"
    if num.startswith(("688", "689")):
        return "STAR"          # 科创板
    if num.startswith(("300", "301")):
        return "GEM"           # 创业板
    if num.startswith(("600", "601", "603", "605", "000", "001", "002", "003")):
        return "MAIN"          # 主板（含原中小板）
    return "OTHER"


def limit_width(board: str, is_st: bool, date) -> float:
    """返回该 (板块, ST状态, 日期) 下的涨跌停幅度（小数）。"""
    d = str(date)[:10]
    if board == "BSE":
        return 0.30
    if board == "STAR":
        return 0.20
    if board == "GEM":
        if d >= GEM_WIDEN_DATE:
            return 0.20
        return 0.05 if is_st else 0.10
    if board == "MAIN":
        return 0.05 if is_st else 0.10
    return 0.10


# ---------------------------------------------------------------- 特征参数 (论文 §3.3)
EWMA_SPAN = 63                    # 事前波动率 σ̂ 的 EWMA 跨度
RET_HORIZONS = (1, 21, 63, 252)   # 波动归一化收益的窗口
MACD_PAIRS = ((8, 24), (16, 48), (32, 96))
MACD_RENORM_WIN = 252             # MACD 自身 252 日滚动标准差再归一化
ZSCORE_WINS = (21, 252)           # 对数价格 Z-score 窗口
CLIP_WIN = 252                    # 稳健裁剪的滚动窗口
CLIP_K = 5.0                      # 裁剪带宽倍数
MAD_SCALE = 1.4826                # MAD -> σ 的一致性系数

# ---------------------------------------------------------------- 元数据抓取
META_FETCH_FREQ = "W-FRI"         # ST / 行业的抓取频率（周频，随后按交易日 ffill）
META_FACTORS = ["是否ST", "申万一级行业名称", "申万一级行业代码",
                "申万二级行业名称", "申万二级行业代码"]


# ====================================================================== v2
# 以下为 v2（深度截面因子）新增。v1 的端到端仓位参数已随 legacy 代码一并移除。

# ---------------------------------------------------------------- 标签
LABEL_HORIZONS = (1, 5, 10)       # 未来 h 日残差收益
LABEL_H_MAIN = 5                  # 主选（假设：周频调仓。若生产线是日频/月频需改）
RESID_MODE = "size_ind"           # 残差化程度: "mkt" | "size_ind" | "full_factor"
LABEL_TRANSFORM = "rank"          # "rank"（截面 rank 映到 [-1,1]）| "zscore"

# ---------------------------------------------------------------- 序列
LOOKBACK = 20                     # 回看窗口（v1 是 84；MASTER 用 8）
SEQ_BURN_IN = 5                   # 前几步只用于预热循环状态

# ---------------------------------------------------------------- 属性分档
N_BUCKET = 10                     # 市值/流动性/年限的截面分位档数
ATTR_PANELS = ("B_lnmv", "B_illiq20", "A_age")

# ---------------------------------------------------------------- 特征集
FEATURE_SETS = {
    # v1 继承的纯价格特征
    "price9": ["ret_1d", "ret_21d", "ret_63d", "ret_252d",
               "macd_8_24", "macd_16_48", "macd_32_96",
               "zscore_21d", "zscore_252d"],
    # v2 扩充：加量、换手、波动、日内/隔夜拆分、流动性
    "ext": None,                  # 由 build_features_ext.py 写入 meta/feature_sets.json
}
FEATURE_SET_DEFAULT = "price9"

# 把截面标准化后的特征张量缓存成 .npy。
# 30 特征时每次启动要做 30 次逐日截面 rank，本机实测 216 秒；
# 在按小时计费的 pod 上，多跑几组实验这个开销很可观。代价是磁盘约 4GB。
CACHE_FEATURES = True

# ---------------------------------------------------------------- 模型
LEVELS = ("L0", "L1A", "L1B", "L2")
# L0  纯时序，个股之间不交换信息
# L1A 末步截面 attention        O(N^2)
# L1B 逐时间步截面 attention    O(N^2 L)，MASTER 式
# L2  L1 + 市场状态门控
