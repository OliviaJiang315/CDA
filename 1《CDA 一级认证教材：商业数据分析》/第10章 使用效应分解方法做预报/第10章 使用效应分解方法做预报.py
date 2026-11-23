"""第 10 章 使用效应分解方法做预报

教材用 Power BI / WPS AI 做时间序列预测，核心是「效应分解」：
观测值 = 趋势效应 × 季节效应 × 随机效应（乘法），或 趋势 + 季节 + 随机（加法）。

本脚本用 Python 复现两个案例：
1. AirPassengers 月度航空客流：乘法分解、对数变换、Holt-Winters 预测与误差评估；
2. 信用卡消费额（含节日 / 去除节日）：量化节日效应、星期效应，并用「趋势 + 星期」外推。
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
OUT = BASE / "图表输出"
OUT.mkdir(exist_ok=True)
AIR_CSV = BASE / "AirPassengers.csv"
AIR_XLSX = BASE / "AirPassengers.xlsx"
CARD_HOLIDAY = BASE / "信用卡消费额_含节日.xlsx"
CARD_ADJ = BASE / "信用卡消费额_去除节日.xlsx"

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


def load_air():
    df = pd.read_csv(AIR_CSV)
    df["DATE"] = pd.to_datetime(df["DATE"], format="%m/%d/%Y")
    s = df.set_index("DATE")["AIR"].astype(float).asfreq("MS")
    return s


# ============================================================================
# 一、AirPassengers：乘法分解
# ============================================================================
def air_decomposition():
    print("\n" + "=" * 78)
    print("一、AirPassengers 月度客流：效应分解")
    print("=" * 78)
    s = load_air()
    print(f"  序列长度 {len(s)} 个月（{s.index.min():%Y-%m} ~ {s.index.max():%Y-%m}），"
          f"均值 {s.mean():.1f}，最小值 {s.min():.0f}，最大值 {s.max():.0f}")

    decomp = seasonal_decompose(s, model="multiplicative", period=12)
    trend = decomp.trend.dropna()
    seasonal = decomp.seasonal.dropna()
    resid = decomp.resid.dropna()
    print(f"  乘法模型：AIR = 趋势 × 季节 × 随机")
    print(f"  趋势从 {trend.iloc[0]:.1f} 升至 {trend.iloc[-1]:.1f}，"
          f"累计增长 {trend.iloc[-1] / trend.iloc[0] - 1:+.1%}")
    print(f"  季节指数范围 {seasonal.min():.3f} ~ {seasonal.max():.3f}，"
          f"峰月 {seasonal.idxmax():%m月}，谷月 {seasonal.idxmin():%m月}")
    print(f"  随机项标准差 {resid.std():.4f}")

    var_share = pd.Series({
        "趋势效应": np.var(trend),
        "季节效应": np.var(seasonal.reindex(s.index).ffill().bfill()),
        "随机效应": np.var(resid),
    })
    var_share = var_share / var_share.sum()
    print("\n  各效应方差贡献：")
    print(var_share.map(lambda v: f"{v:.1%}").to_string())

    log_s = np.log(s)
    decomp_log = seasonal_decompose(log_s, model="additive", period=12)
    print(f"\n  对数变换后加法模型：ln(AIR) 残差标准差 {decomp_log.resid.dropna().std():.4f}"
          f"（原序列乘法残差 {resid.std():.4f}）")

    fig, axes = plt.subplots(4, 1, figsize=(13.6, 11.2),
                             gridspec_kw={"height_ratios": [1.5, 1, 0.8, 0.7], "hspace": 0.35})
    ax = axes[0]
    ax.plot(s.index, s.values, color=DEEP_BLUE, linewidth=2.0)
    ax.fill_between(s.index, s.values, color=LIGHT_BLUE, alpha=0.25)
    style_axes(ax, "原始序列（月度航空客流）", None, "旅客数")

    ax = axes[1]
    ax.plot(trend.index, trend.values, color=DEEP_BLUE, linewidth=2.4)
    style_axes(ax, "趋势效应（居中移动平均）", None, "趋势值")

    ax = axes[2]
    season_pattern = seasonal.groupby(seasonal.index.month).mean()
    month_labels = [f"{m} 月" for m in season_pattern.index][::-1]
    bars = ax.barh(month_labels, season_pattern.values[::-1], height=0.62,
                   color=LIGHT_YELLOW, edgecolor=LIGHT_YELLOW)
    for b, v in zip(bars, season_pattern.values[::-1]):
        ax.text(v + season_pattern.max() * 0.012, b.get_y() + b.get_height() / 2,
                f"{v:.2f}", va="center", fontsize=8.5, color=TEXT)
    style_axes(ax, "季节效应（各月平均指数）", "季节指数", None)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_xlim(0, season_pattern.max() * 1.16)

    ax = axes[3]
    ax.plot(resid.index, resid.values, color=GREY, linewidth=1.6)
    ax.axhline(0, color=RED, linewidth=1.0, linestyle="--")
    style_axes(ax, "随机效应（残差）", None, "残差")
    ax.set_xlim(resid.index.min(), resid.index.max())
    ax.set_xlabel("月份", color=TEXT, fontsize=10)
    fig.suptitle("效应分解：把一条曲线拆成趋势、季节、随机三股力量", fontsize=16,
                 fontweight="bold", color=TEXT, y=0.995)
    save(fig, "01_AirPassengers_效应分解.png")

    fig, ax = plt.subplots(figsize=(12.4, 6.4))
    bars = ax.bar(var_share.index, var_share.values, width=0.5,
                  color=[LIGHT_BLUE, LIGHT_YELLOW, GREY], edgecolor="white")
    for b, v in zip(bars, var_share.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1%}", ha="center", va="bottom",
                fontsize=11, color=TEXT)
    style_axes(ax, "各效应对波动的方差贡献", None, "方差占比")
    ax.yaxis.set_major_formatter(lambda v, p: f"{v:.0%}")
    ax.set_ylim(0, var_share.max() * 1.2)
    save(fig, "02_效应方差贡献.png")
    return s


def air_forecast(s):
    print("\n" + "=" * 78)
    print("二、Holt-Winters：用趋势 + 季节做 12 个月预测")
    print("=" * 78)
    train, test = s.iloc[:-12], s.iloc[-12:]
    model = ExponentialSmoothing(
        train, trend="add", seasonal="mul", seasonal_periods=12,
        initialization_method="estimated",
    ).fit()
    forecast = model.forecast(12)
    mape = float(np.mean(np.abs((test - forecast) / test)))
    rmse = float(np.sqrt(np.mean((test - forecast) ** 2)))
    print(f"  训练集 {len(train)} 个月，测试集 {len(test)} 个月")
    print(f"  模型参数：趋势={model.model.trend}，季节={model.model.seasonal}，周期=12")
    print(f"  留出法评估：MAPE={mape:.2%}，RMSE={rmse:.1f}")
    print("\n  预测明细（最后 6 个月）：")
    cmp = pd.DataFrame({"实际": test, "预测": forecast.round(1)})
    cmp["误差率"] = (cmp["预测"] / cmp["实际"] - 1)
    print(cmp.tail(6).assign(误差率=cmp["误差率"].map(lambda v: f"{v:+.1%}")).to_string())

    full = ExponentialSmoothing(
        s, trend="add", seasonal="mul", seasonal_periods=12,
        initialization_method="estimated",
    ).fit()
    future = full.forecast(12)
    resid_std = float(np.std(full.resid, ddof=1))
    upper = future + 1.96 * resid_std
    lower = future - 1.96 * resid_std

    fig, ax = plt.subplots(figsize=(13.6, 7.2))
    ax.plot(s.index, s.values, color=DEEP_BLUE, linewidth=2.0, label="历史实际")
    ax.plot(forecast.index, forecast.values, color=LIGHT_YELLOW, linewidth=2.6,
            marker="o", markersize=4, label="留出期预测")
    ax.plot(future.index, future.values, color=RED, linewidth=2.4, linestyle="--",
            label="未来 12 个月预测")
    ax.fill_between(future.index, lower, upper, color=LIGHT_YELLOW, alpha=0.35,
                    label="95% 近似区间")
    style_axes(ax, "AirPassengers：趋势 + 乘法季节的预测", "月份", "旅客数")
    ax.legend(frameon=False, loc="upper left", ncol=2)
    ax.text(0.02, 0.72, f"留出期 MAPE = {mape:.2%}\nRMSE = {rmse:.0f}",
            transform=ax.transAxes, fontsize=10, color=TEXT,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#F2F7FC", edgecolor=LIGHT_BLUE))
    save(fig, "03_AirPassengers_预测.png")
    return forecast, future


# ============================================================================
# 三、信用卡消费额：节日效应与星期效应
# ============================================================================
def card_effect():
    print("\n" + "=" * 78)
    print("三、信用卡消费额：节日效应分解与 7 天预测")
    print("=" * 78)
    holiday = pd.read_excel(CARD_HOLIDAY, sheet_name="pos")
    adjusted = pd.read_excel(CARD_ADJ, sheet_name="Sheet1")
    holiday.columns = ["date", "amount"]
    adjusted.columns = ["date", "amount"]
    holiday["date"] = pd.to_datetime(holiday["date"])
    adjusted["date"] = pd.to_datetime(adjusted["date"])
    df = holiday.merge(adjusted, on="date", suffixes=("_含节日", "_去节日"))
    df["含节日_万元"] = df["amount_含节日"] / 10000
    df["节日效应"] = df["含节日_万元"] / df["amount_去节日"] - 1
    df["weekday"] = df["date"].dt.dayofweek
    df["是否周末"] = df["weekday"] >= 5

    festival = df[df["节日效应"].abs() > 0.03].sort_values("节日效应", ascending=False)
    print(f"  观测区间 {df['date'].min():%Y-%m-%d} ~ {df['date'].max():%Y-%m-%d}，共 {len(df)} 天")
    print(f"  含节日日均 {df['含节日_万元'].mean():.1f} 万元，"
          f"去除节日日均 {df['amount_去节日'].mean():.1f} 万元")
    print("\n  实际值相对「去除节日基准」偏离超过 3% 的日期：")
    print(festival[["date", "含节日_万元", "amount_去节日", "节日效应"]]
          .assign(含节日_万元=lambda d: d["含节日_万元"].round(1),
                  amount_去节日=lambda d: d["amount_去节日"].round(1),
                  节日效应=lambda d: d["节日效应"].map(lambda v: f"{v:+.1%}")).to_string(index=False))

    y = df["amount_去节日"].to_numpy()
    t = np.arange(len(y))
    coef = np.polyfit(t, y, 1)
    trend = np.polyval(coef, t)
    detrended = y - trend
    weekday_effect = pd.Series(detrended).groupby(df["weekday"].values).mean()
    weekday_effect.index = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    print("\n  去节日序列的星期效应（去趋势后的平均偏离，万元）：")
    print(weekday_effect.round(2).to_string())
    print(f"  周末较工作日平均效应：{weekday_effect.iloc[5:].mean() - weekday_effect.iloc[:5].mean():+.2f} 万元")

    future_dates = pd.date_range(df["date"].max() + pd.Timedelta(days=1), periods=7, freq="D")
    future_t = np.arange(len(y), len(y) + 7)
    future_trend = np.polyval(coef, future_t)
    future = pd.Series(
        future_trend + weekday_effect.reindex(
            [["周一", "周二", "周三", "周四", "周五", "周六", "周日"][d.weekday()] for d in future_dates]).to_numpy(),
        index=future_dates,
    )
    print("\n  未来 7 天预测（趋势 + 星期效应）：")
    print(future.round(1).to_string())

    fig, axes = plt.subplots(2, 2, figsize=(15.2, 9.4))
    ax = axes[0, 0]
    ax.plot(df["date"], df["含节日_万元"], color=GREY, linewidth=1.8, label="含节日")
    ax.plot(df["date"], df["amount_去节日"], color=DEEP_BLUE, linewidth=2.4, label="去除节日")
    for _, row in festival.iterrows():
        ax.scatter(row["date"], row["含节日_万元"], s=46, color=RED, zorder=4)
    style_axes(ax, "含节日实际值 vs 去除节日基准", None, "消费额（万元）")
    ax.legend(frameon=False, loc="upper left")
    ax.tick_params(axis="x", rotation=35)

    ax = axes[0, 1]
    colors = [RED if v > 0 else MID_BLUE for v in df["节日效应"].values]
    ax.bar(df["date"], df["节日效应"].values, width=0.72, color=colors, edgecolor=colors)
    ax.axhline(0, color=GREY, linewidth=1.0)
    style_axes(ax, "逐日偏离基准（实际 / 基准 - 1）", None, "偏离度")
    ax.yaxis.set_major_formatter(lambda v, p: f"{v:.0%}")
    ax.tick_params(axis="x", rotation=35)

    ax = axes[1, 0]
    bars = ax.bar(weekday_effect.index, weekday_effect.values, width=0.56,
                  color=[LIGHT_YELLOW if i >= 5 else LIGHT_BLUE for i in range(7)],
                  edgecolor="white")
    for b, v in zip(bars, weekday_effect.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:+.1f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8.5, color=TEXT)
    ax.axhline(0, color=GREY, linewidth=1.0)
    style_axes(ax, "星期效应：周末 vs 工作日", None, "偏离（万元）")

    ax = axes[1, 1]
    ax.plot(df["date"], df["amount_去节日"], color=DEEP_BLUE, linewidth=2.2, label="历史")
    ax.plot(df["date"], trend, color=GREY, linewidth=1.4, linestyle="--", label="线性趋势")
    ax.plot(future.index, future.values, color=RED, linewidth=2.4, marker="o",
            markersize=5, label="未来 7 天预测")
    style_axes(ax, "去节日序列：线性趋势 + 星期效应外推", None, "消费额（万元）")
    ax.legend(frameon=False, loc="upper left")
    ax.tick_params(axis="x", rotation=35)

    fig.suptitle("效应分解的两条落地路径：解释历史波动，外推短期走势", fontsize=15,
                 fontweight="bold", color=TEXT, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "04_信用卡消费额_节日效应与预测.png")
    return df, future


def main():
    set_style()
    s = air_decomposition()
    air_forecast(s)
    card_effect()
    print("\n第 10 章效应分解与预测完成，图表输出目录：", OUT)


if __name__ == "__main__":
    main()