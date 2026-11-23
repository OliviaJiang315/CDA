"""第 9 章 用户标签体系与用户画像专题

教材本章没有附带数据文件，因此本脚本用课程中真实出现过的两份数据完成迁移：
- 消费流水（RFM 模型的原始数据）：构建统计标签、规则标签、RFM 模型标签，输出用户标签宽表；
- 银行客户数据（关系模型原始数据）：补充性别、年龄、账户与贷款行为，形成属性画像。

输出两个层面的「画像」：
1. 行为画像：客群规模、消费贡献、R/F/M 特征、品类偏好；
2. 属性画像：性别、年龄、账户频率、贷款金额分层。
并给出标签质量检查（唯一性、覆盖率、取值分布），保证标签可落地。
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
OUT = BASE / "图表输出"
OUT.mkdir(exist_ok=True)
CH1 = BASE.parent / "第1章 数据分析思维"
CH2 = BASE.parent / "第2章 数据分析方法"
RFM_FILE = CH2 / "1、分析的基础范式" / "data-有答案" / "1-2RFM模型-参考.xlsx"
REL_DIR = CH1 / "关系模型演示-不建议用poweBI操作关系型数据模型"

LIGHT_BLUE = "#A8CCE8"
LIGHT_YELLOW = "#FFE699"
DEEP_BLUE = "#4F81BD"
MID_BLUE = "#7FA8CF"
GREY = "#A6A6A6"
TEXT = "#404040"
GRID = "#E3E3E3"
RED = "#C0504D"


def set_style():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.edgecolor"] = "#BFBFBF"


def style_axes(ax, title=None, xlabel=None, ylabel=None, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#BFBFBF")
    ax.tick_params(colors=TEXT, labelsize=10, length=0)
    ax.set_axisbelow(True)
    if grid_axis == "y":
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    elif grid_axis == "x":
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    if title:
        ax.set_title(title, loc="left", fontsize=13.5, fontweight="bold", color=TEXT, pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT, fontsize=10)


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[图表] {path.name}")


def rounded_box(ax, x, y, w, h, text, fc, fs=9.5):
    box = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                         boxstyle="round,pad=0.006,rounding_size=0.03",
                         facecolor=fc, edgecolor=fc, linewidth=1.2, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=TEXT,
            zorder=4, linespacing=1.55)


def read_csv_smart(path, **kwargs):
    for enc in ("utf-8-sig", "gbk", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法识别文件编码：{path}")


def clean_columns(df):
    df = df.copy()
    df.columns = [str(c).replace("\u3000", " ").strip() for c in df.columns]
    return df


# ============================================================================
# 一、标签体系：从原始字段到可运营标签
# ============================================================================
def build_user_tags():
    print("\n" + "=" * 78)
    print("一、用户标签体系：原始标签 -> 统计标签 -> 规则标签 -> 模型标签")
    print("=" * 78)
    raw = clean_columns(pd.read_excel(RFM_FILE, sheet_name="原始数据"))
    raw = raw.loc[:, ~raw.columns.str.startswith("Unnamed")]
    raw["销售日期"] = pd.to_datetime(raw["销售日期"])
    raw["会员创建日期"] = pd.to_datetime(raw["会员创建日期"])
    raw["销售金额"] = pd.to_numeric(raw["销售金额"], errors="coerce")
    raw = raw.dropna(subset=["用户编号", "销售日期", "销售金额"])
    ref = pd.Timestamp("2019-08-03")

    tags = raw.groupby("用户编号").agg(
        最近消费日期=("销售日期", "max"),
        首次消费日期=("销售日期", "min"),
        消费频次=("流水号", "nunique"),
        消费金额=("销售金额", "sum"),
        会员创建日期=("会员创建日期", "min"),
        品类数=("一级品类", "nunique"),
    ).reset_index()
    tags["R_近度"] = (ref - tags["最近消费日期"]).dt.days
    tags["F_频次"] = tags["消费频次"]
    tags["M_金额"] = tags["消费金额"]
    tags["客单价"] = tags["M_金额"] / tags["F_频次"]
    tags["会员天数"] = (ref - tags["会员创建日期"]).dt.days
    tags["月均消费"] = tags["M_金额"] / (tags["会员天数"].clip(lower=1) / 30)

    cat_amt = raw.groupby(["用户编号", "一级品类"])["销售金额"].sum()
    top_cat = cat_amt.groupby("用户编号").idxmax().map(lambda x: x[1])
    tags["偏好品类"] = tags["用户编号"].map(top_cat)

    tags["价值等级"] = pd.cut(tags["M_金额"],
                          bins=[-1, tags["M_金额"].quantile(0.5), tags["M_金额"].quantile(0.9), np.inf],
                          labels=["低价值", "中价值", "高价值"])
    tags["活跃等级"] = pd.cut(tags["R_近度"], bins=[-1, 30, 90, 180, np.inf],
                          labels=["高活跃", "活跃", "沉默", "沉睡"])

    def lifecycle(row):
        if row["R_近度"] > 365:
            return "流失客户"
        if row["R_近度"] > 180:
            return "流失预警"
        if row["会员天数"] <= 180:
            return "新客"
        if row["F_频次"] >= tags["F_频次"].median():
            return "忠实客户"
        return "成长客户"

    tags["生命周期"] = tags.apply(lifecycle, axis=1)

    tags["R程度"] = np.where(tags["R_近度"] <= tags["R_近度"].mean(), "高", "低")
    tags["F程度"] = np.where(tags["F_频次"] >= tags["F_频次"].mean(), "高", "低")
    tags["M程度"] = np.where(tags["M_金额"] >= tags["M_金额"].mean(), "高", "低")
    label_map = {
        ("高", "高", "高"): "重要价值用户", ("高", "低", "高"): "重要发展用户",
        ("低", "高", "高"): "重要保持用户", ("低", "低", "高"): "重要挽留用户",
        ("高", "高", "低"): "一般价值用户", ("高", "低", "低"): "一般发展用户",
        ("低", "高", "低"): "一般保持用户", ("低", "低", "低"): "一般挽留用户",
    }
    tags["RFM客群"] = [label_map[(r, f, m)] for r, f, m in
                     zip(tags["R程度"], tags["F程度"], tags["M程度"])]

    print(f"  原始流水 {len(raw):,} 行，生成 {len(tags):,} 位用户的标签宽表")
    print(f"  标签字段：{len(tags.columns)} 个（原始 1 + 统计 9 + 规则 3 + 模型 1 + 辅助维度）")
    print("\n  标签取值分布：")
    for col in ("价值等级", "活跃等级", "生命周期", "RFM客群"):
        vc = tags[col].value_counts()
        print(f"  [{col}] " + "；".join(f"{k}: {v:,}" for k, v in vc.items()))

    print("\n  标签质量检查：")
    print(f"  - 主键唯一性：用户编号重复 {int(tags['用户编号'].duplicated().sum())} 条")
    for col in ("偏好品类", "价值等级", "活跃等级", "生命周期", "RFM客群"):
        cover = tags[col].notna().mean()
        print(f"  - {col} 覆盖率 {cover:.1%}，取值数 {tags[col].nunique()}")
    print(f"  - 标签宽表已输出：{OUT.parent / '用户标签宽表.csv'}")

    out_cols = ["用户编号", "R_近度", "F_频次", "M_金额", "客单价", "会员天数", "月均消费",
                "品类数", "偏好品类", "价值等级", "活跃等级", "生命周期", "RFM客群"]
    tags[out_cols].to_csv(BASE / "用户标签宽表.csv", index=False, encoding="utf-8-sig")
    return raw, tags


def draw_tag_system():
    fig, ax = plt.subplots(figsize=(14.2, 7.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    layers = [
        ("原始标签", ["用户编号", "销售日期", "销售金额", "一级品类"], LIGHT_YELLOW),
        ("统计标签", ["R 近度", "F 频次", "M 金额", "客单价 / 会员天数"], LIGHT_BLUE),
        ("规则标签", ["高 / 中 / 低价值", "高活跃 / 沉睡", "新客 / 忠实 / 流失"], MID_BLUE),
        ("模型标签", ["RFM 八类客群", "偏好品类", "生命周期"], "#DCE9F6"),
    ]
    xs = [0.13, 0.38, 0.63, 0.88]
    for (name, items, color), x in zip(layers, xs):
        ax.text(x, 0.86, name, ha="center", fontsize=12, fontweight="bold", color=TEXT)
        for i, item in enumerate(items):
            rounded_box(ax, x, 0.70 - i * 0.16, 0.20, 0.115, item, color, fs=9)
        if x < 0.85:
            for yy in (0.70, 0.54, 0.38):
                ax.annotate("", xy=(x + 0.115, yy), xytext=(x + 0.10, yy),
                            arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.2))

    ax.text(0.5, 0.08, "标签不是越多越好：每个标签都要有明确的业务动作（触达、分层、推荐、风控）",
            ha="center", fontsize=10, color=GREY)
    ax.set_title("用户标签体系：从字段到可运营的人群", loc="left", fontsize=15,
                 fontweight="bold", color=TEXT, pad=16)
    save(fig, "01_用户标签体系.png")


# ============================================================================
# 二、行为画像
# ============================================================================
def behavior_profile(raw, tags):
    print("\n" + "=" * 78)
    print("二、行为画像：客群规模、消费贡献、品类偏好")
    print("=" * 78)
    seg = tags.groupby("RFM客群").agg(
        人数=("用户编号", "count"),
        消费金额=("M_金额", "sum"),
        平均R=("R_近度", "mean"),
        平均F=("F_频次", "mean"),
        平均M=("M_金额", "mean"),
    ).sort_values("消费金额", ascending=False)
    seg["人数占比"] = seg["人数"] / seg["人数"].sum()
    seg["金额占比"] = seg["消费金额"] / seg["消费金额"].sum()
    print(seg.assign(消费金额=seg["消费金额"].map(lambda v: f"{v:,.0f}"),
                     平均R=seg["平均R"].map(lambda v: f"{v:.0f} 天"),
                     平均F=seg["平均F"].map(lambda v: f"{v:.1f} 次"),
                     平均M=seg["平均M"].map(lambda v: f"{v:,.0f} 元"),
                     人数占比=seg["人数占比"].map(lambda v: f"{v:.1%}"),
                     金额占比=seg["金额占比"].map(lambda v: f"{v:.1%}")).to_string())

    top_cats = raw.groupby("一级品类")["销售金额"].sum().sort_values(ascending=False).head(8).index
    pref = (raw[raw["一级品类"].isin(top_cats)]
            .groupby(["一级品类", "用户编号"])["销售金额"].sum()
            .reset_index()
            .merge(tags[["用户编号", "RFM客群"]], on="用户编号", how="left")
            .groupby(["RFM客群", "一级品类"])["销售金额"].sum()
            .unstack(fill_value=0))
    pref = pref.div(pref.sum(axis=1), axis=0).reindex(seg.index)[top_cats]

    fig, axes = plt.subplots(2, 2, figsize=(15.2, 9.8))
    ax = axes[0, 0]
    order = seg.index[::-1]
    bars = ax.barh(order, seg.loc[order, "人数"], height=0.6,
                   color=[LIGHT_BLUE if i % 2 == 0 else LIGHT_YELLOW for i in range(len(order))],
                   edgecolor="white")
    for b, v in zip(bars, seg.loc[order, "人数"]):
        ax.text(b.get_width() + seg["人数"].max() * 0.015, b.get_y() + b.get_height() / 2,
                f"{v:,}（{v / seg['人数'].sum():.0%}）", va="center", fontsize=8.5, color=TEXT)
    style_axes(ax, "客群规模", "用户数", None, grid_axis="x")
    ax.yaxis.grid(False)
    ax.set_xlim(0, seg["人数"].max() * 1.32)

    ax = axes[0, 1]
    bars = ax.barh(order, seg.loc[order, "消费金额"] / 10000, height=0.6,
                   color=[LIGHT_YELLOW if i % 2 == 0 else LIGHT_BLUE for i in range(len(order))],
                   edgecolor="white")
    for b, v in zip(bars, seg.loc[order, "消费金额"] / 10000):
        ax.text(b.get_width() + seg["消费金额"].max() / 10000 * 0.015,
                b.get_y() + b.get_height() / 2, f"{v:,.0f}", va="center", fontsize=8.5, color=TEXT)
    style_axes(ax, "客群消费贡献（万元）", "消费金额（万元）", None, grid_axis="x")
    ax.yaxis.grid(False)
    ax.set_xlim(0, seg["消费金额"].max() / 10000 * 1.28)

    ax = axes[1, 0]
    important = {"重要价值用户", "重要保持用户", "重要挽留用户", "重要发展用户"}
    sample = tags.sample(min(3000, len(tags)), random_state=1)
    sample = sample.assign(客群分组=np.where(sample["RFM客群"].isin(important),
                                            "重要客群", "一般客群"))
    for name, color in (("重要客群", MID_BLUE), ("一般客群", "#D9D9D9")):
        part = sample[sample["客群分组"] == name]
        ax.scatter(part["R_近度"], part["F_频次"], s=11, alpha=0.55,
                   color=color, label=name)
    ax.set_xlim(0, tags["R_近度"].quantile(0.98))
    ax.set_ylim(0, tags["F_频次"].quantile(0.98))
    style_axes(ax, "R-F 分布：近度越短、频次越高越优质", "R（距今天数）", "F（消费频次）")
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper right", markerscale=2.2)

    ax = axes[1, 1]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "pref", ["#FFF8E1", LIGHT_YELLOW, "#F3E2A0", LIGHT_BLUE, MID_BLUE, "#3F6F9E"])
    im = ax.imshow(pref.values, cmap=cmap, aspect="auto", vmin=0, vmax=np.nanmax(pref.values))
    ax.set_xticks(range(len(pref.columns)))
    ax.set_xticklabels(pref.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pref.index)))
    ax.set_yticklabels(pref.index, fontsize=8.5)
    for i in range(pref.shape[0]):
        for j in range(pref.shape[1]):
            v = pref.values[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7.5,
                    color="white" if v > 0.25 else TEXT)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("客群 × 品类偏好（金额占比）", loc="left", fontsize=13.5,
                 fontweight="bold", color=TEXT, pad=12)

    fig.suptitle("行为画像：把用户按价值分层，再描述每一层买什么", fontsize=15,
                 fontweight="bold", color=TEXT, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "02_行为画像_客群与品类偏好.png")
    return seg


# ============================================================================
# 三、属性画像（银行客户数据）
# ============================================================================
def attribute_profile():
    print("\n" + "=" * 78)
    print("三、属性画像：银行客户的性别、年龄、账户与贷款行为")
    print("=" * 78)
    clients = read_csv_smart(REL_DIR / "clients.csv")
    disp = read_csv_smart(REL_DIR / "disp.csv")
    accounts = read_csv_smart(REL_DIR / "accounts.csv")
    loans = read_csv_smart(REL_DIR / "loans.csv")
    card = read_csv_smart(REL_DIR / "card.csv")

    clients["出生日期"] = pd.to_datetime(clients["birth_date"], errors="coerce")
    clients["性别"] = clients["sex"].map({"女": "女", "男": "男"}).fillna(clients["sex"])
    ref = pd.Timestamp("1998-12-31")
    clients["年龄"] = ((ref - clients["出生日期"]).dt.days / 365.25).round(0)

    client_loan = (clients.merge(disp, on="client_id", how="left")
                   .merge(loans, on="account_id", how="left")
                   .groupby("client_id")
                   .agg(性别=("性别", "first"), 年龄=("年龄", "first"),
                        贷款笔数=("loan_id", "count"), 贷款金额=("amount", "sum"))
                   .reset_index())
    card_cnt = card.merge(disp, on="disp_id", how="left").groupby("client_id")["card_id"].nunique()
    client_loan["持卡数"] = client_loan["client_id"].map(card_cnt).fillna(0)
    client_loan["贷款分层"] = pd.cut(client_loan["贷款金额"],
                                 bins=[-1, 0, 50000, 200000, 500000, np.inf],
                                 labels=["无贷款", "5万以下", "5-20万", "20-50万", "50万以上"])
    print(f"  客户 {len(client_loan):,} 人，有贷款客户 {(client_loan['贷款笔数'] > 0).sum():,} 人，"
          f"贷款总额 {client_loan['贷款金额'].sum():,.0f}")
    print("\n  性别分布：")
    print(client_loan["性别"].value_counts().to_string())
    print("\n  年龄分布：")
    print(client_loan["年龄"].describe().round(1).to_string())
    print("\n  贷款金额分层：")
    print(client_loan["贷款分层"].value_counts().to_string())
    print("  标签使用说明：属性标签用于圈人，行为标签用于判断价值与时机，两者组合才能形成可执行的人群包。")

    fig, axes = plt.subplots(2, 2, figsize=(15.0, 9.4))
    ax = axes[0, 0]
    gender = client_loan["性别"].value_counts()
    ax.pie(gender.values, labels=gender.index, colors=[LIGHT_BLUE, LIGHT_YELLOW],
           autopct=lambda p: f"{p:.1f}%", startangle=90, counterclock=False,
           wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
           textprops=dict(color=TEXT, fontsize=10))
    ax.text(0, 0, "性别", ha="center", va="center", fontsize=11, color=TEXT)
    ax.set_title("客户性别", loc="left", fontsize=13.5, fontweight="bold", color=TEXT, pad=12)

    ax = axes[0, 1]
    age_bins = [0, 30, 40, 50, 60, 70, 120]
    age_labels = ["30岁以下", "30-39岁", "40-49岁", "50-59岁", "60-69岁", "70岁以上"]
    age = pd.cut(client_loan["年龄"], bins=age_bins, labels=age_labels).value_counts().reindex(age_labels)
    bars = ax.bar(age.index, age.values, width=0.56, color=LIGHT_YELLOW, edgecolor=LIGHT_YELLOW)
    for b, v in zip(bars, age.values):
        ax.text(b.get_x() + b.get_width() / 2, v, int(v), ha="center", va="bottom",
                fontsize=8.5, color=TEXT)
    style_axes(ax, "客户年龄分布", None, "客户数")
    ax.tick_params(axis="x", rotation=20)
    ax.set_ylim(0, age.max() * 1.18)

    ax = axes[1, 0]
    freq = accounts["frequency"].value_counts().head(6)
    bars = ax.bar(freq.index, freq.values, width=0.56, color=LIGHT_BLUE, edgecolor=LIGHT_BLUE)
    for b, v in zip(bars, freq.values):
        ax.text(b.get_x() + b.get_width() / 2, v, int(v), ha="center", va="bottom",
                fontsize=8.5, color=TEXT)
    style_axes(ax, "账户对账单频率（行为属性）", None, "账户数")
    ax.set_ylim(0, freq.max() * 1.18)

    ax = axes[1, 1]
    loan_band = client_loan["贷款分层"].value_counts().reindex(
        ["无贷款", "5万以下", "5-20万", "20-50万", "50万以上"]).fillna(0)
    bars = ax.bar(loan_band.index, loan_band.values, width=0.56,
                  color=["#D9D9D9", LIGHT_BLUE, MID_BLUE, LIGHT_YELLOW, "#F5C46B"],
                  edgecolor="white")
    for b, v in zip(bars, loan_band.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{int(v):,}", ha="center", va="bottom",
                fontsize=8.5, color=TEXT)
    style_axes(ax, "贷款金额分层（价值属性）", None, "客户数")
    ax.tick_params(axis="x", rotation=20)
    ax.set_ylim(0, loan_band.max() * 1.18)

    fig.suptitle("属性画像：人口属性 + 行为属性 + 价值属性", fontsize=15,
                 fontweight="bold", color=TEXT, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "03_属性画像_银行客户.png")


def main():
    set_style()
    raw, tags = build_user_tags()
    draw_tag_system()
    behavior_profile(raw, tags)
    attribute_profile()
    print("\n第 9 章标签与画像完成，图表输出目录：", OUT)


if __name__ == "__main__":
    main()