"""第 7 章 指标体系构建方法：把 Excel 指标模板变成可计算的指标体系

教材给出「指标体系设计模板」：
- Step1 从业务场景出发，明确目标、策略、指标拆分、口径、分析维度；
- Step2 用「指标名称 / 计算规则 / 单位 / 口径 / 事件 / 维度 / 指标类型 / 关联指标」定义每个 KPI。

本脚本完成三件事：
1. 解析模板，构建「目标 -> 一级 -> 二级 -> 派生」指标树；
2. 用事件明细实现指标计算引擎，验证分子分母与口径是否自洽；
3. 输出指标树、指标漏斗与指标字典，展示「指标不是名字，而是一套可复算的口径」。
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
TEMPLATE = BASE / "1、指标体系设计模板.xlsx"

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


def rounded_box(ax, x, y, w, h, title, body="", fc=LIGHT_BLUE, fs=9.5, title_fs=10):
    box = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                         boxstyle="round,pad=0.008,rounding_size=0.035",
                         facecolor=fc, edgecolor=fc, linewidth=1.3, zorder=3)
    ax.add_patch(box)
    ax.text(x, y + h * 0.22, title, ha="center", va="center", fontsize=title_fs,
            fontweight="bold", color=TEXT, zorder=4)
    if body:
        ax.text(x, y - h * 0.2, body, ha="center", va="center", fontsize=fs,
                color=TEXT, zorder=4, linespacing=1.5)


def load_template():
    step1 = pd.read_excel(TEMPLATE, sheet_name="Step1")
    step2 = pd.read_excel(TEMPLATE, sheet_name="Step2")
    step2 = step2.loc[:, :"备注"]
    step2.columns = ["层级", "指标名称", "计算规则", "单位", "口径", "事件",
                     "分析维度", "指标类型", "关联指标", "备注"]
    step2 = step2.dropna(subset=["指标名称"]).reset_index(drop=True)
    step2["层级"] = step2["层级"].ffill()
    return step1, step2


def build_metric_tree(step2):
    print("\n" + "=" * 78)
    print("一、指标体系设计模板解析")
    print("=" * 78)
    print(f"  指标字典 {len(step2)} 条，层级：{' / '.join(step2['层级'].unique())}")

    for kind, part in step2.groupby("层级", sort=False):
        names = "、".join(part["指标名称"].astype(str).tolist())
        print(f"  [{kind}] {names}")

    print("\n  指标定义明细（关键 5 字段）：")
    print(step2[["层级", "指标名称", "计算规则", "单位", "指标类型"]].to_string(index=False))

    duplicates = step2["指标名称"][step2["指标名称"].duplicated()].tolist()
    missing_rule = step2.loc[step2["计算规则"].isna(), "指标名称"].tolist()
    missing_caliber = step2.loc[step2["口径"].isna(), "指标名称"].tolist()
    print("\n  指标质量检查：")
    print(f"  - 重复指标名：{duplicates or '无'}")
    print(f"  - 缺少计算规则：{missing_rule or '无'}")
    print(f"  - 缺少业务口径：{missing_caliber or '无'}")
    print(f"  - 派生指标（需要上游基础指标）：{int((step2['指标类型'] == '派生指标').sum())} 个")
    return step2


def build_demo_events(n_users=5000, seed=42):
    rng = np.random.default_rng(seed)
    channels = ["渠道大类A", "渠道大类B", "渠道大类C"]
    channel = rng.choice(channels, size=n_users, p=[0.45, 0.35, 0.20])
    register_day = rng.integers(0, 30, size=n_users)
    apply_flag = rng.random(n_users) < 0.24
    apply_day = np.minimum(register_day + rng.integers(0, 31, size=n_users), 30)
    audit_flag = apply_flag & (rng.random(n_users) < 0.84)
    pass_flag = audit_flag & (rng.random(n_users) < 0.69)

    rows = []
    for uid, ch, day in zip(range(1, n_users + 1), channel, register_day):
        rows.append((uid, "注册点击完成", pd.Timestamp("2025-06-01") + pd.Timedelta(days=int(day)), ch))
    for uid in np.where(apply_flag)[0] + 1:
        rows.append((uid, "申请贷款", pd.Timestamp("2025-06-01") + pd.Timedelta(days=int(apply_day[uid - 1])), channel[uid - 1]))
    for uid in np.where(audit_flag)[0] + 1:
        rows.append((uid, "审核完成", pd.Timestamp("2025-06-01") + pd.Timedelta(days=int(min(apply_day[uid - 1] + 1, 30))), channel[uid - 1]))
    for uid in np.where(pass_flag)[0] + 1:
        rows.append((uid, "审批通过", pd.Timestamp("2025-06-01") + pd.Timedelta(days=int(min(apply_day[uid - 1] + 2, 30))), channel[uid - 1]))
    events = pd.DataFrame(rows, columns=["用户ID", "事件", "事件时间", "渠道大类"])
    return events


def metric_engine(events):
    print("\n" + "=" * 78)
    print("二、指标计算引擎：同一套事件明细，算出四个指标")
    print("=" * 78)
    reg = events.loc[events["事件"] == "注册点击完成", "用户ID"].nunique()
    apply_users = events.loc[events["事件"] == "申请贷款", "用户ID"].nunique()
    audit_users = events.loc[events["事件"] == "审核完成", "用户ID"].nunique()
    pass_users = events.loc[events["事件"] == "审批通过", "用户ID"].nunique()

    reg_time = (events[events["事件"] == "注册点击完成"]
                .set_index("用户ID")["事件时间"])
    apply_time = (events[events["事件"] == "申请贷款"]
                  .groupby("用户ID")["事件时间"].min())
    within_30 = ((apply_time - reg_time.reindex(apply_time.index)).dt.days <= 30).sum()

    metrics = pd.DataFrame([
        {"指标": "注册用户数", "口径": "注册点击完成的去重用户数", "值": reg, "单位": "人"},
        {"指标": "进件用户数", "口径": "提交贷款申请的去重用户数", "值": apply_users, "单位": "人"},
        {"指标": "审核完成用户数", "口径": "审核节点完成的去重用户数", "值": audit_users, "单位": "人"},
        {"指标": "审批通过用户数", "口径": "审批结果为通过的去重用户数", "值": pass_users, "单位": "人"},
        {"指标": "30天进件转化率", "口径": "注册后30天内申请贷款用户数 / 注册用户数",
         "值": within_30 / reg, "单位": "%"},
        {"指标": "审核通过率", "口径": "审批通过用户数 / 进件用户数",
         "值": pass_users / apply_users, "单位": "%"},
    ])
    metrics["显示值"] = np.where(metrics["单位"] == "%", metrics["值"].map(lambda v: f"{v:.1%}"),
                            metrics["值"].map(lambda v: f"{v:,.0f}"))
    print(metrics[["指标", "显示值", "口径"]].to_string(index=False))

    print("\n  口径自洽性校验：")
    print(f"  - 进件用户数 <= 注册用户数：{apply_users <= reg}（{apply_users:,} <= {reg:,}）")
    print(f"  - 审核完成 <= 进件：{audit_users <= apply_users}")
    print(f"  - 审批通过 <= 审核完成：{pass_users <= audit_users}")
    print(f"  - 30 天内进件人数 {within_30:,} <= 进件总数 {apply_users:,}：{within_30 <= apply_users}")

    print("\n  分渠道指标：")
    by_channel = events.groupby("渠道大类").agg(
        注册用户数=("用户ID", lambda s: s.nunique()))
    ch_apply = (events[events["事件"] == "申请贷款"]
                .groupby("渠道大类")["用户ID"].nunique())
    by_channel["进件用户数"] = ch_apply.reindex(by_channel.index).fillna(0).astype(int)
    by_channel["进件转化率"] = by_channel["进件用户数"] / by_channel["注册用户数"]
    print(by_channel.assign(进件转化率=by_channel["进件转化率"].map(lambda v: f"{v:.1%}")).to_string())
    return metrics, by_channel


def draw_metric_tree():
    fig, ax = plt.subplots(figsize=(13.4, 8.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    rounded_box(ax, 0.5, 0.91, 0.34, 0.11, "核心目标",
                "提升贷款业务有效转化", LIGHT_YELLOW, title_fs=11)
    level1 = [("注册用户数", 0.19), ("进件用户数", 0.50), ("审批通过用户数", 0.81)]
    for name, x in level1:
        rounded_box(ax, x, 0.70, 0.23, 0.10, name, "基础指标", LIGHT_BLUE)
        ax.plot([0.5, x], [0.855, 0.75], color=GREY, linewidth=1.4, zorder=1)

    level2 = [("30天进件转化率", 0.34), ("审核通过率", 0.66)]
    for name, x in level2:
        rounded_box(ax, x, 0.47, 0.26, 0.10, name, "派生指标 = 分子 / 分母", LIGHT_YELLOW)
    ax.plot([0.19, 0.34], [0.65, 0.52], color=GREY, linewidth=1.4, zorder=1)
    ax.plot([0.50, 0.34], [0.65, 0.52], color=GREY, linewidth=1.4, zorder=1)
    ax.plot([0.50, 0.66], [0.65, 0.52], color=GREY, linewidth=1.4, zorder=1)
    ax.plot([0.81, 0.66], [0.65, 0.52], color=GREY, linewidth=1.4, zorder=1)

    level3 = [("渠道转化贡献", 0.15), ("审核拒绝原因分布", 0.39),
              ("注册用户数", 0.63), ("进件用户数", 0.87)]
    for name, x in level3:
        rounded_box(ax, x, 0.23, 0.2, 0.095, name, "下钻维度", "#DCE9F6", fs=9)
        ax.plot([0.34, x], [0.42, 0.28], color=GREY, linewidth=1.2, zorder=1, alpha=0.8)
    ax.plot([0.66, 0.63], [0.42, 0.28], color=GREY, linewidth=1.2, zorder=1, alpha=0.8)
    ax.plot([0.66, 0.87], [0.42, 0.28], color=GREY, linewidth=1.2, zorder=1, alpha=0.8)

    ax.text(0.5, 0.06, "指标树 = 目标 -> 基础指标 -> 派生指标 -> 可下钻维度",
            ha="center", fontsize=10, color=GREY)
    ax.set_title("指标体系结构：每一个数字都能沿树回溯到事件与口径", loc="left",
                 fontsize=15, fontweight="bold", color=TEXT, pad=16)
    save(fig, "01_指标体系树.png")


def metric_charts(metrics, by_channel):
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.8),
                             gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    steps = metrics.iloc[:4]
    colors = [LIGHT_BLUE, MID_BLUE, LIGHT_YELLOW, "#F5C46B"]
    y = np.arange(len(steps))[::-1]
    for (i, row), yi in zip(steps.iterrows(), y):
        width = row["值"] / steps["值"].max()
        box = FancyBboxPatch((0.5 - width / 2, yi + 0.18), width, 0.6,
                             boxstyle="round,pad=0.003,rounding_size=0.05",
                             facecolor=colors[i], edgecolor="white", linewidth=1.2)
        ax.add_patch(box)
        ax.text(0.5, yi + 0.48, f"{row['指标']}  {int(row['值']):,} 人",
                ha="center", va="center", fontsize=10, color=TEXT)
        if i > 0:
            rate = row["值"] / steps["值"].iloc[i - 1]
            ax.text(0.5, yi + 0.85, f"环节转化 {rate:.1%}", ha="center", fontsize=8.5, color=RED)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, len(steps))
    ax.axis("off")
    ax.set_title("指标漏斗：注册 -> 进件 -> 审核 -> 通过", loc="left", fontsize=13.5,
                 fontweight="bold", color=TEXT, pad=14)

    ax = axes[1]
    bars = ax.bar(by_channel.index, by_channel["进件转化率"], width=0.55,
                  color=[LIGHT_BLUE, LIGHT_YELLOW, MID_BLUE], edgecolor="white")
    for b, v in zip(bars, by_channel["进件转化率"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.1%}", ha="center",
                va="bottom", fontsize=9.5, color=TEXT)
    style_axes(ax, "分渠道进件转化率", None, "进件转化率")
    ax.yaxis.set_major_formatter(lambda v, p: f"{v:.0%}")
    ax.set_ylim(0, by_channel["进件转化率"].max() * 1.2)
    fig.suptitle("指标不是孤立的数字：上游决定下游，分子分母必须同口径", fontsize=15,
                 fontweight="bold", color=TEXT, y=1.0)
    save(fig, "02_指标漏斗与渠道转化.png")


def main():
    set_style()
    step1, step2 = load_template()
    build_metric_tree(step2)
    events = build_demo_events()
    metrics, by_channel = metric_engine(events)
    draw_metric_tree()
    metric_charts(metrics, by_channel)
    print("\n第 7 章指标体系构建完成，图表输出目录：", OUT)


if __name__ == "__main__":
    main()