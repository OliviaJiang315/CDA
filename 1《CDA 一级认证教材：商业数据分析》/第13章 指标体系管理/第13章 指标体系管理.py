"""第 13 章 指标体系管理：数据标准与指标标准的落地校验

教材提供两份 Excel 模板：
- 基础数据标准编制模板：22 个字段，覆盖识别、分类、管理、业务、技术五类信息；
- 指标数据标准编制模板：23 个字段，覆盖指标业务含义、业务口径、技术口径、维度、状态等。

本脚本把「标准管理」变成可执行的程序：
1. 读取标准模板，按编号规则、必填项、业务定义、技术属性逐项校验；
2. 解析指标技术口径，生成可执行的 SQL 模板（参数化统计日期）；
3. 输出标准分布热力图（部门 × 标准大类）与指标标准生命周期状态机。
"""

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
OUT = BASE / "图表输出"
OUT.mkdir(exist_ok=True)
BASIC_FILE = BASE / "1、基础数据标准编制模板.xlsx"
IND_FILE = BASE / "2、指标数据标准编制模板.xlsx"

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


def style_axes(ax, title=None, xlabel=None, ylabel=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#BFBFBF")
    ax.tick_params(colors=TEXT, labelsize=10, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
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


def clean_columns(df):
    df = df.copy()
    df.columns = [str(c).replace("\u3000", " ").strip() for c in df.columns]
    return df


def filled(value):
    return value is not None and not pd.isna(value) and str(value).strip() not in ("", "/", "nan")


def rounded_box(ax, x, y, w, h, text, fc, fs=9.5):
    box = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                         boxstyle="round,pad=0.006,rounding_size=0.028",
                         facecolor=fc, edgecolor=fc, linewidth=1.2, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=TEXT,
            zorder=4, linespacing=1.6)


# ============================================================================
# 一、读取标准模板
# ============================================================================
def load_standards():
    print("\n" + "=" * 78)
    print("一、读取基础数据标准与指标数据标准模板")
    print("=" * 78)
    basic = clean_columns(pd.read_excel(BASIC_FILE, sheet_name="客户主题为例", header=1))
    basic = basic.dropna(how="all").dropna(subset=["标准编号"]).reset_index(drop=True)
    indicators = clean_columns(pd.read_excel(IND_FILE, sheet_name="标准定义", header=1))
    indicators = indicators.dropna(how="all").dropna(subset=["编号"]).reset_index(drop=True)
    dims = clean_columns(pd.read_excel(IND_FILE, sheet_name="维度定义", header=0))

    print(f"  基础数据标准 {len(basic)} 条，字段 {len(basic.columns)} 个")
    print(f"  指标数据标准 {len(indicators)} 条，字段 {len(indicators.columns)} 个")
    print(f"  维度定义 {len(dims)} 条")
    print("\n  基础标准关键字段：")
    print(basic[["标准编号", "标准中文名称", "标准大类", "标准小类", "归口管理部门",
                 "标准类型", "长度", "度量单位"]].to_string(index=False))
    print("\n  指标标准关键字段：")
    print(indicators[["编号", "中文名称", "指标业务口径", "指标类型", "数据类型",
                      "关联标准", "指标状态"]].to_string(index=False))
    return basic, indicators, dims


# ============================================================================
# 二、标准校验
# ============================================================================
def validate_standards(basic, indicators):
    print("\n" + "=" * 78)
    print("二、标准校验：编号、必填、定义、技术属性、生命周期")
    print("=" * 78)
    results = []

    for _, row in basic.iterrows():
        code = str(row["标准编号"]).strip()
        checks = [
            ("编号规则", r"^[A-Z]{2}\d{6}$", code, "两位主题码 + 6 位流水，如 CM000001"),
            ("必填-中文名称", None, row.get("标准中文名称"),
             "标准中文名称不能为空"),
            ("必填-归口部门", None, row.get("归口管理部门"),
             "必须有最终业务解释权的部门"),
            ("必填-管理负责人", None, row.get("管理负责人"),
             "标准负责人不能为空"),
            ("必填-确认情况", None, row.get("确认情况"),
             "需经归口管理部门确认"),
            ("业务定义", None, row.get("标准业务定义"),
             "业务定义需说明概念与边界"),
            ("长度合法", r"^\d+$", row.get("长度"),
             "长度必须为正整数"),
        ]
        for rule, pattern, value, desc in checks:
            if rule == "长度合法":
                try:
                    ok = float(value) > 0 and float(value).is_integer()
                except (TypeError, ValueError):
                    ok = False
            elif pattern:
                ok = bool(re.match(pattern, str(value).strip())) if filled(value) else False
            else:
                ok = filled(value)
            results.append({
                "标准编号": code,
                "标准名称": str(row.get("标准中文名称", "")).strip(),
                "检查项": rule,
                "实测值": str(value)[:28] if filled(value) else "（空）",
                "结果": "通过" if ok else "不通过",
                "规则说明": desc,
            })

    valid_types = {"代码型", "数值型", "文本型", "日期型", "时间型", "标志型",
                   "编码型", "金额型", "比率型", "比例型"}
    valid_status = {"草稿", "评审中", "已发布", "有效", "失效"}

    for _, row in indicators.iterrows():
        code = str(row["编号"]).strip()
        checks = [
            ("编号规则", r"^[A-Z]{2}-[GP]-\d{3}$", code, "主题码-根/派生-3 位流水"),
            ("必填-中文名称", None, row.get("中文名称"), "指标名称不能为空"),
            ("必填-业务口径", None, row.get("指标业务口径"), "必须说明分子与分母"),
            ("必填-技术口径", None, row.get("技术口径"), "必须有可执行的取数逻辑"),
            ("必填-主责部门", None, row.get("主责部门"), "指标必须有责任部门"),
            ("指标类型", valid_types | {"根指标", "派生指标"}, row.get("指标类型"),
             "根指标 / 派生指标"),
            ("数据类型", {"INT", "FLOAT", "STRING", "DATE", "DATETIME"},
             row.get("数据类型"), "技术数据类型必须在字典内"),
            ("状态合法", valid_status, row.get("指标状态"), "状态必须走生命周期"),
            ("关联标准", None, row.get("关联标准"), "派生指标必须挂接基础标准"),
        ]
        for rule, allowed, value, desc in checks:
            if isinstance(allowed, set):
                ok = filled(value) and str(value).strip() in allowed
            elif allowed is None:
                ok = filled(value)
            else:
                ok = bool(re.match(allowed, str(value).strip())) if filled(value) else False
            if rule == "关联标准" and str(row.get("指标类型")) != "派生指标":
                ok = True
            results.append({
                "标准编号": code,
                "标准名称": str(row.get("中文名称", "")).strip(),
                "检查项": rule,
                "实测值": str(value)[:28] if filled(value) else "（空）",
                "结果": "通过" if ok else "不通过",
                "规则说明": desc,
            })

    result = pd.DataFrame(results)
    passed = int((result["结果"] == "通过").sum())
    total = len(result)
    print(f"  共校验 {total} 项，通过 {passed} 项，通过率 {passed / total:.1%}")
    print("\n  未通过项：")
    failed = result[result["结果"] == "不通过"]
    if len(failed):
        print(failed.to_string(index=False))
    else:
        print("  无")
    result.to_csv(BASE / "标准校验结果.csv", index=False, encoding="utf-8-sig")

    print("\n  指标口径与 SQL 模板：")
    for _, row in indicators.iterrows():
        print(f"  [{row['编号']}] {row['中文名称']}")
        print(f"      业务口径：{row['指标业务口径']}")
        sql = str(row["技术口径"]).replace("&I_XXX", ":stat_date").replace("&I_TRADEDATE", ":stat_date")
        sql = re.sub(r"\s+", " ", sql).strip()
        print(f"      技术口径（参数化后）：{sql[:150]}{'...' if len(sql) > 150 else ''}")

    for _, row in basic.iterrows():
        print(f"\n  基础标准示例：{row['标准编号']} {row['标准中文名称']}，"
              f"归口 {row['归口管理部门']}，类型 {row['标准类型']}，长度 {row['长度']}")
    return result


def validation_chart(result):
    summary = result.groupby(["标准名称", "结果"]).size().unstack(fill_value=0)
    for col in ("通过", "不通过"):
        if col not in summary:
            summary[col] = 0

    fig, axes = plt.subplots(1, 2, figsize=(15.6, 6.8),
                             gridspec_kw={"width_ratios": [1, 1.35]})
    ax = axes[0]
    x = np.arange(len(summary))
    ax.bar(x - 0.17, summary["通过"], width=0.34, color=LIGHT_BLUE,
           edgecolor=LIGHT_BLUE, label="通过")
    ax.bar(x + 0.17, summary["不通过"], width=0.34, color=RED,
           edgecolor=RED, label="不通过")
    for xi, row in enumerate(summary.itertuples()):
        ax.text(xi - 0.17, row.通过 + 0.08, int(row.通过), ha="center", fontsize=8.5, color=TEXT)
        ax.text(xi + 0.17, row.不通过 + 0.08, int(row.不通过), ha="center", fontsize=8.5, color=TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels(summary.index, rotation=12)
    style_axes(ax, "每条标准的校验结果", None, "检查项数")
    ax.legend(frameon=False, loc="upper right")
    ax.set_ylim(0, max(summary.max()) * 1.25)

    ax = axes[1]
    ax.axis("off")
    failed = result[result["结果"] == "不通过"][["标准编号", "检查项", "规则说明"]]
    if len(failed) == 0:
        ax.text(0.5, 0.5, "全部校验通过", ha="center", va="center", fontsize=16, color=DEEP_BLUE)
    else:
        tbl = ax.table(cellText=failed.values, colLabels=["标准编号", "检查项", "规则说明"],
                       loc="center", cellLoc="left", colLoc="center",
                       colWidths=[0.22, 0.2, 0.58])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.2)
        tbl.scale(1, 2.0)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor("#E6E6E6")
            if r == 0:
                cell.set_facecolor(LIGHT_YELLOW)
                cell.set_text_props(color=TEXT, fontweight="bold")
            elif r % 2 == 0:
                cell.set_facecolor("#FBFCFE")
    ax.set_title("未通过项与整改说明", loc="left", fontsize=13.5, fontweight="bold",
                 color=TEXT, pad=12)
    fig.suptitle("标准校验：编号能对上、定义说得清、责任落得下，才算“标准”", fontsize=15,
                 fontweight="bold", color=TEXT, y=1.0)
    save(fig, "01_标准校验结果.png")


# ============================================================================
# 三、标准分布
# ============================================================================
def distribution_heatmap():
    print("\n" + "=" * 78)
    print("三、标准分布：部门 × 标准大类（教材交叉表）")
    print("=" * 78)
    raw = pd.read_excel(BASIC_FILE, sheet_name="标准分布统计", header=None)
    header = raw.iloc[3].tolist()
    data = raw.iloc[4:].copy()
    data = data.iloc[:, :len(header)]
    data.columns = [str(h).strip() for h in header]
    data = data.dropna(subset=["Row Labels"])
    data = data[data["Row Labels"].astype(str) != "Grand Total"]
    data = data.set_index("Row Labels")
    data = data.drop(columns=["Grand Total"], errors="ignore")
    data = data.apply(pd.to_numeric, errors="coerce").fillna(0)

    print("  标准大类 × 部门（部分）：")
    print(data.iloc[:6, :6].astype(int).to_string())
    print(f"  标准总数 {int(data.to_numpy().sum()):,}，"
          f"覆盖 {len(data)} 个标准大类、{len(data.columns)} 个部门")
    by_dept = data.sum().sort_values(ascending=False)
    print(f"  标准最多的部门：{by_dept.index[0]}（{int(by_dept.iloc[0])} 条）；"
          f"最少的部门：{by_dept.index[-1]}（{int(by_dept.iloc[-1])} 条）")

    fig, ax = plt.subplots(figsize=(14.6, 7.2))
    cmap = LinearSegmentedColormap.from_list(
        "std", ["#FFF8E1", LIGHT_YELLOW, "#F3E2A0", LIGHT_BLUE, MID_BLUE, "#3F6F9E"])
    im = ax.imshow(data.values, cmap=cmap, aspect="auto", vmin=0, vmax=data.to_numpy().max())
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=38, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index, fontsize=9)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data.values[i, j]
            if v:
                ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=7.5,
                        color="white" if v > data.to_numpy().max() * 0.55 else TEXT)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("数据标准分布：部门 × 标准大类", loc="left", fontsize=14.5,
                 fontweight="bold", color=TEXT, pad=14)
    cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, length=0)
    save(fig, "02_标准分布热力图.png")
    return data


# ============================================================================
# 四、生命周期与口径落地
# ============================================================================
def lifecycle_and_sql():
    ind = clean_columns(pd.read_excel(IND_FILE, sheet_name="标准定义", header=1))
    row = ind.iloc[0]
    sql = re.sub(r"\s+", " ", str(row["技术口径"]).replace("&I_XXX", ":stat_date")).strip()

    fig, axes = plt.subplots(1, 2, figsize=(15.6, 7.0),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    states = [
        ("草稿", "业务提出\n口径初步成形", LIGHT_GREY), 
        ("评审中", "业务 + 技术\n双评审", LIGHT_YELLOW),
        ("已发布", "进入指标字典\n对外可引用", LIGHT_BLUE),
        ("有效", "线上运行\n纳入监控", MID_BLUE),
        ("失效", "停止使用\n保留历史", "#D9D9D9"),
    ]
    xs = np.linspace(0.10, 0.90, 5)
    for (name, body, color), x in zip(states, xs):
        rounded_box(ax, x, 0.56, 0.16, 0.22, f"{name}\n{body}", color, fs=9)
        if x < 0.85:
            ax.annotate("", xy=(x + 0.093, 0.56), xytext=(x + 0.073, 0.56),
                        arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.5))
    ax.annotate("", xy=(0.10, 0.31), xytext=(0.90, 0.31),
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.4,
                                connectionstyle="arc3,rad=0.35"))
    ax.text(0.5, 0.19, "失效后可重新起草新版本，但旧版本必须留痕", ha="center",
            fontsize=9, color=GREY)
    ax.set_title("指标标准生命周期", loc="left", fontsize=13.5, fontweight="bold",
                 color=TEXT, pad=12)

    ax = axes[1]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    rounded_box(ax, 0.5, 0.83, 0.86, 0.18,
                f"业务定义\n{str(row['指标业务含义'])[:42]}", LIGHT_YELLOW, fs=8.5)
    rounded_box(ax, 0.5, 0.58, 0.86, 0.18,
                f"业务口径\n{str(row['指标业务口径'])[:42]}", LIGHT_BLUE, fs=8.5)
    rounded_box(ax, 0.5, 0.31, 0.86, 0.22,
                "技术口径（SQL 模板）\n" + sql[:110] + ("..." if len(sql) > 110 else ""),
                "#EEF4FA", fs=8)
    ax.annotate("", xy=(0.5, 0.47), xytext=(0.5, 0.495),
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.5))
    ax.annotate("", xy=(0.5, 0.21), xytext=(0.5, 0.44),
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.5))
    ax.set_title("指标口径的三层落地", loc="left", fontsize=13.5, fontweight="bold",
                 color=TEXT, pad=12)
    fig.suptitle("指标管理的终点不是文档，而是「能被系统执行的 SQL」", fontsize=15,
                 fontweight="bold", color=TEXT, y=1.0)
    save(fig, "03_生命周期与口径落地.png")
    print("\n  示例指标：", row["编号"], row["中文名称"])
    print("  业务含义：", row["指标业务含义"])
    print("  业务口径：", row["指标业务口径"])
    print("  技术口径：", sql[:200])
    print("  管理要点：口径变更必须走版本与生效日期，历史报表要用历史版本复算。")


LIGHT_GREY = "#D9D9D9"


def main():
    set_style()
    basic, indicators, dims = load_standards()
    result = validate_standards(basic, indicators)
    validation_chart(result)
    distribution_heatmap()
    lifecycle_and_sql()
    print("\n第 13 章指标管理完成，图表输出目录：", OUT)


if __name__ == "__main__":
    main()