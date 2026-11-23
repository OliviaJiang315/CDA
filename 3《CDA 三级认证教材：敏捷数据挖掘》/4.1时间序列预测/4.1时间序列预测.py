# -*- coding: utf-8 -*-
"""
4.1 时间序列预测

项目背景
--------
教材用 4 个案例覆盖时间序列预测的常见方法：

    4.1.1  航空公司乘客数预测：趋势分解法 / Prophet（教材标注“不建议学习”）
    4.1.2  拖拉机销量预测：SARIMA（季节性差分自回归移动平均）
    4.1.3  谷歌股价预测：LSTM 循环神经网络
    ARIMA  模拟序列：AR / MA / ARMA / ARIMA 的原理与定阶

数据文件
--------
    air.csv                  月度航空乘客数（DATE, AIR）
    tractor_sales.csv        月度拖拉机销量（Year_Month, Number_of_Sold）
    google_stock_prics.csv   谷歌股票日线（Date, High, Low, Open, Close, Volume, Adj Close）
    ts_simu200.csv           200 期模拟时间序列（AR1_a, MA1_a, ARMA_11_b, ARIMA_110）

依赖
----
    必装：numpy pandas matplotlib statsmodels
    选装：prophet（或旧版 fbprophet）、tensorflow
    未安装选装包时，对应案例会打印提示并跳过，不影响其他案例。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

# 图表统一输出到章节目录下的 figures/，便于离线查看与归档
FIG_PREFIX = '4.1时间序列预测'
FIG_DIR = BASE_DIR / "figures"
_fig_counter = [0]


def save_show(*_args, **_kwargs):
    """保存当前图形到 figures/ 目录后再显示。"""
    FIG_DIR.mkdir(exist_ok=True)
    _fig_counter[0] += 1
    fig_path = FIG_DIR / ("%s_%02d.png" % (FIG_PREFIX, _fig_counter[0]))
    try:
        plt.tight_layout()
    except Exception:
        pass
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close("all")
    return fig_path

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 4.1.1 趋势分解 / Prophet：航空乘客数预测
# ============================================================
def run_prophet_demo():
    try:
        try:
            from prophet import Prophet
            from prophet.plot import add_changepoints_to_plot
        except ImportError:
            from fbprophet import Prophet
            from fbprophet.plot import add_changepoints_to_plot
    except ImportError:
        print("未安装 prophet/fbprophet，跳过 4.1.1（可 pip install prophet）。")
        return

    import warnings

    warnings.filterwarnings("ignore")

    df = pd.read_csv(BASE_DIR / "air.csv")
    df["DATE"] = pd.to_datetime(df["DATE"])
    print(df.head())
    print(df.dtypes)

    df["DATE"] = pd.DatetimeIndex(df["DATE"])
    print(df.dtypes)

    # Prophet 要求列名必须为 ds（时间）与 y（目标）
    df = df.rename(columns={"DATE": "ds", "AIR": "y"})
    print(df.head(2))

    ax = df.set_index("ds").plot(figsize=(8, 5))
    ax.set_title("航空公司月度乘客数", fontsize=15)
    ax.set_xlabel("日期", fontsize=15)
    ax.set_ylabel("乘客数", fontsize=15)
    save_show()

    # 加法模型，95% 置信区间
    my_model = Prophet(
        growth="linear",
        yearly_seasonality="auto",
        weekly_seasonality=False,
        seasonality_mode="additive",
        daily_seasonality=False,
        holidays=None,
        interval_width=0.95,
    )
    my_model.fit(df)

    future_dates = my_model.make_future_dataframe(periods=36, freq="MS")
    forecast = my_model.predict(future_dates)
    print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(5))

    fig, ax = plt.subplots(figsize=(8, 5))
    my_model.plot(forecast, ax=ax, uncertainty=True)
    ax.set_title("航空乘客数预测（加法模型）", fontsize=15)
    ax.set_xlabel("日期", fontsize=15)
    ax.set_ylabel("乘客数", fontsize=15)
    save_show()

    # 带节假日效应的乘法模型
    my_model = Prophet(
        growth="linear",
        seasonality_mode="multiplicative",
        holidays=None,
        interval_width=0.95,
    )
    my_model.add_country_holidays("China")
    my_model.fit(df)

    future_dates = my_model.make_future_dataframe(periods=36, freq="MS")
    forecast = my_model.predict(future_dates)

    fig, ax = plt.subplots(figsize=(8, 5))
    my_model.plot(forecast, uncertainty=True, ax=ax)
    ax.set_title("航空乘客数预测（乘法模型 + 节假日效应）", fontsize=15)
    ax.set_xlabel("日期", fontsize=15)
    ax.set_ylabel("乘客数", fontsize=15)
    add_changepoints_to_plot(ax, my_model, forecast)
    save_show()

    comp_fig = my_model.plot_components(forecast, figsize=(7, 5))
    for ax_, title_ in zip(comp_fig.axes, ["趋势项", "年度季节性", "节假日效应"]):
        ax_.set_title(title_)
    comp_fig.suptitle("Prophet 预测分量分解", fontsize=14)
    save_show()

    print(forecast.tail(5))


# ============================================================
# 4.1.2 SARIMA：拖拉机销量预测
# ============================================================
def run_sarima_demo():
    import itertools
    import warnings

    import statsmodels.api as sm

    warnings.filterwarnings("ignore")

    sales_data = pd.read_csv(BASE_DIR / "tractor_sales.csv")
    print(sales_data.head(5))

    dates = pd.to_datetime(sales_data["Year_Month"], format="%Y-%b")
    sales_data.set_index(dates, inplace=True)
    sales_data.rename(columns={"Number_of_Sold": "Tractor-Sales"}, inplace=True)
    sales_ts = sales_data["Tractor-Sales"]

    plt.figure(figsize=(8, 5))
    plt.plot(sales_ts)
    plt.title("拖拉机月度销量", fontsize=12)
    plt.xlabel("年份", fontsize=12)
    plt.ylabel("拖拉机销量", fontsize=12)
    save_show()

    # 乘法季节分解
    decomposition = sm.tsa.seasonal_decompose(sales_ts, model="multiplicative")
    fig = decomposition.plot()
    fig.set_figwidth(8)
    fig.set_figheight(5)
    for ax_, title_ in zip(
        fig.axes, ["实际值", "趋势项", "季节项", "残差项"]
    ):
        ax_.set_title(title_)
    fig.suptitle("拖拉机销量季节分解", fontsize=14)
    save_show()

    # 取对数平稳化方差，再做一阶差分平稳化均值
    sales_ts_log = np.log(sales_ts)
    sales_ts_log.dropna(inplace=True)
    sales_ts_log_diff = sales_ts_log.diff(periods=1)
    sales_ts_log_diff.dropna(inplace=True)

    plt.figure(figsize=(10, 3))
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    sm.graphics.tsa.plot_acf(sales_ts_log_diff, lags=30, ax=ax1)
    ax1.set_title("自相关图（ACF）")
    ax1.set_xlabel("滞后阶数")
    ax1.set_ylabel("自相关系数")
    sm.graphics.tsa.plot_pacf(sales_ts_log_diff, lags=30, ax=ax2)
    ax2.set_title("偏自相关图（PACF）")
    ax2.set_xlabel("滞后阶数")
    ax2.set_ylabel("偏自相关系数")
    plt.suptitle("对数差分序列的自相关结构", fontsize=13)
    save_show()

    # 网格搜索最小 AIC 的 SARIMA(p,d,q)(P,D,Q,s)
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

    best_aic = np.inf
    best_pdq = None
    best_seasonal_pdq = None
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                temp_model = sm.tsa.statespace.SARIMAX(
                    endog=sales_ts_log,
                    order=param,
                    seasonal_order=param_seasonal,
                    enforce_stationarity=True,
                    enforce_invertibility=True,
                )
                results = temp_model.fit()
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_pdq = param
                    best_seasonal_pdq = param_seasonal
            except Exception:
                continue

    print("Best: SARIMAX{}x{}".format(best_pdq, best_seasonal_pdq))
    print("AIC: {}".format(best_aic))

    best_model = sm.tsa.statespace.SARIMAX(
        sales_ts_log,
        order=best_pdq,
        seasonal_order=best_seasonal_pdq,
        enforce_stationarity=True,
        enforce_invertibility=True,
    )
    best_results = best_model.fit()
    print(best_results.summary().tables[0])
    print(best_results.summary().tables[1])

    diag_fig = best_results.plot_diagnostics(lags=30, figsize=(8, 8))
    for ax_, title_ in zip(
        diag_fig.axes, ["标准化残差", "残差直方图", "正态 Q-Q 图", "残差相关图"]
    ):
        ax_.set_title(title_)
    diag_fig.suptitle("SARIMA 模型诊断", fontsize=14)
    save_show()

    n_steps = 36
    forecast = best_results.get_forecast(steps=n_steps, alpha=0.05)
    pred_pr_95 = forecast.predicted_mean
    pred_ci_95 = forecast.conf_int()
    idx = pd.date_range(sales_ts.index[-1], periods=n_steps, freq="MS")

    fc_95 = pd.DataFrame(
        data=np.column_stack([np.exp(pred_pr_95), np.exp(pred_ci_95)]),
        index=idx,
        columns=["forecast", "lower_ci_95", "upper_ci_95"],
    )
    print(fc_95.head(5))

    ax = sales_ts.plot(label="实际值", figsize=(8, 5))
    fc_95["forecast"].plot(ax=ax, linestyle="--", label="预测值", alpha=0.7)
    ax.fill_between(
        fc_95.index,
        fc_95["lower_ci_95"],
        fc_95["upper_ci_95"],
        color="k",
        alpha=0.25,
    )
    ax.set_xlabel("年份", fontsize=15)
    ax.set_ylabel("拖拉机销量", fontsize=15)
    ax.set_title("拖拉机销量预测（95% 置信区间）", fontsize=15)
    plt.legend(loc="best", fontsize=15)
    save_show()


# ============================================================
# 4.1.3 LSTM：谷歌股价预测
# ============================================================
def process_ts(ts, timesteps=4):
    """
    把二维序列转换为 LSTM 的监督学习样本。
    ts         : numpy.array，形状 [n_samples, features]
    timesteps  : 用前多少步预测下一步
    返回 x      : [n_samples, timesteps, features]
         y      : [n_samples, features]
    """
    feature_dims = ts.shape[1]
    x, y = [], []
    for i in range(timesteps, len(ts)):
        x.append(ts[i - timesteps:i, ])
        y.append(ts[i:i + 1, ])

    x, y = np.vstack(x), np.vstack(y)
    x = x.reshape((-1, timesteps, feature_dims))
    y = y.reshape((-1, feature_dims))
    return x, y


def run_lstm_demo():
    try:
        import tensorflow as tf
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.models import Sequential
    except ImportError:
        print("未安装 tensorflow，跳过 4.1.3 LSTM（可 pip install tensorflow）。")
        return

    from sklearn.metrics import mean_absolute_error
    from sklearn.preprocessing import MinMaxScaler

    googl = pd.read_csv(BASE_DIR / "google_stock_prics.csv")
    print(googl.head())

    data_length = len(googl)
    train_size = 0.8
    cutoff = round(data_length * train_size)

    # 选择 Open、Close 两列
    train = googl.iloc[:cutoff, 3:5]
    test = googl.iloc[cutoff:, 3:5]
    print("Train shape:", train.shape, "\nTest shape:", test.shape)

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)

    timesteps = 60
    train_data, train_target = process_ts(train_scaled, timesteps=timesteps)
    test_data, test_target = process_ts(test_scaled, timesteps=timesteps)
    print(train_data[:2])
    print("-" * 20)
    print(train_target[:2])

    feature_dims = train.shape[1]

    regressor = Sequential()
    regressor.add(
        LSTM(units=32, return_sequences=True, input_shape=(timesteps, feature_dims))
    )
    regressor.add(LSTM(units=32, return_sequences=False))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units=feature_dims, activation=None))
    regressor.summary()

    regressor.compile(optimizer="adam", loss="mean_squared_error")
    history = regressor.fit(
        train_data,
        train_target,
        epochs=100,
        batch_size=16,
        validation_data=(test_data, test_target),
        shuffle=False,
    )

    fig = plt.figure(figsize=[10, 5])
    ax = plt.subplot()
    ax.plot(history.history["loss"], "b-", label="训练集")
    ax.plot(history.history["val_loss"], "r--", label="验证集")
    plt.title("损失随训练轮次的变化", fontsize=15)
    plt.ylabel("损失", fontsize=15)
    plt.xlabel("训练轮次", fontsize=15)
    plt.legend(fontsize=15)
    save_show()

    test_predict = regressor.predict(test_data)
    test_predict = scaler.inverse_transform(test_predict)
    test_mae = mean_absolute_error(test_predict[:, 0], test.iloc[timesteps:, -1])
    dummy_mae = mean_absolute_error(
        test.iloc[timesteps:-1, -1], test.iloc[timesteps + 1:, -1]
    )
    print("Mean absolute error : %.2f" % test_mae)
    print("Dummy MSE: %.2f" % dummy_mae)

    plt.figure(figsize=(10, 5))
    plt.plot(test.values[timesteps:, 1], "b-", label="实际股价")
    plt.plot(test_predict[:, 0], "r--", label="预测股价")
    plt.title("谷歌股价预测", fontsize=15)
    plt.xlabel("时间", fontsize=15)
    plt.ylabel("谷歌股价", fontsize=15)
    plt.legend(fontsize=15)
    save_show()


# ============================================================
# 4.1.4 ARIMA 原理与定阶（模拟序列）
# ============================================================
def run_arima_demo():
    import itertools
    import warnings

    import statsmodels.api as sm
    from scipy import stats
    from statsmodels.graphics.api import qqplot
    from statsmodels.tsa.stattools import adfuller

    warnings.filterwarnings("ignore")

    def plot_acf_pacf(series, lags=20, name=""):
        """把 ACF 与 PACF 画在同一张图上，并输出中文标题。"""
        fig_, axes_ = plt.subplots(1, 2, figsize=(10, 3))
        sm.graphics.tsa.plot_acf(series, lags=lags, ax=axes_[0])
        axes_[0].set_title(name + "自相关图（ACF）")
        axes_[0].set_xlabel("滞后阶数")
        axes_[0].set_ylabel("自相关系数")
        sm.graphics.tsa.plot_pacf(series, lags=lags, ax=axes_[1])
        axes_[1].set_title(name + "偏自相关图（PACF）")
        axes_[1].set_xlabel("滞后阶数")
        axes_[1].set_ylabel("偏自相关系数")
        save_show()

    ts_simu200 = pd.read_csv(BASE_DIR / "ts_simu200.csv", index_col="t")
    print(ts_simu200.head())

    dates = pd.date_range(start="2017/01/01", periods=200)
    ts_simu200.set_index(dates, inplace=True)

    # ---------- AR(1) ----------
    dta = ts_simu200["AR1_a"]
    ax_ar = dta.plot(figsize=(8, 5))
    ax_ar.set_title("AR(1) 模拟序列")
    ax_ar.set_ylabel("数值")
    save_show()

    plot_acf_pacf(dta, name="AR(1) 原序列：")

    ar10 = sm.tsa.ARIMA(dta, order=(1, 0, 0)).fit()

    resid = ar10.resid
    plot_acf_pacf(resid, name="AR(1) 残差：")

    print(sm.stats.durbin_watson(ar10.resid.values))
    print(stats.normaltest(resid))

    fig, ax = plt.subplots(figsize=(5, 5))
    qqplot(resid, line="q", fit=True, ax=ax)
    ax.set_title("残差正态 Q-Q 图")
    ax.set_xlabel("理论分位数")
    ax.set_ylabel("样本分位数")
    save_show()

    forcast = ar10.forecast(steps=5)
    print(pd.DataFrame({"forcast": forcast}))

    # ---------- MA(1) ----------
    dta = ts_simu200["MA1_a"]
    plot_acf_pacf(dta, name="MA(1) 原序列：")

    ma01 = sm.tsa.ARIMA(dta, order=(0, 0, 1)).fit()
    resid = ma01.resid
    plot_acf_pacf(resid, name="MA(1) 残差：")

    print(sm.stats.durbin_watson(ma01.resid))
    print(stats.normaltest(resid).pvalue)

    print(
        ma01.predict(
            start=pd.to_datetime("2017-07-20"),
            end=pd.to_datetime("2017-07-25"),
            dynamic=False,
        )
    )

    # ---------- ARMA(1,1) ----------
    dta = ts_simu200["ARMA_11_b"]
    plot_acf_pacf(dta, name="ARMA(1,1) 原序列：")

    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    best_aic = np.inf
    best_pdq = None
    for param in pdq:
        try:
            results = sm.tsa.ARIMA(dta, order=param).fit()
            if results.aic < best_aic:
                best_aic = results.aic
                best_pdq = param
        except Exception:
            continue

    print("Best ARIMA{} model - AIC:{}".format(best_pdq, best_aic))

    arma11 = sm.tsa.ARIMA(dta, order=(1, 0, 1)).fit()
    resid = arma11.resid
    plot_acf_pacf(resid, name="ARMA(1,1) 残差：")

    print(sm.stats.durbin_watson(arma11.resid))
    print(stats.normaltest(resid).pvalue)
    print(arma11.predict(start=200, end=205, dynamic=False))

    # ---------- ARIMA(1,1,0)：先平稳性检验，再差分 ----------
    dta = ts_simu200["ARIMA_110"]
    result = adfuller(dta)
    print("ADF Statistic: %f" % result[0])
    print("p-value: %f" % result[1])

    plot_acf_pacf(dta, name="ARIMA(1,1,0) 原序列：")

    ax_raw = dta.plot()
    ax_raw.set_title("ARIMA(1,1,0) 原序列")
    ax_raw.set_ylabel("数值")
    save_show()

    diff1 = dta.diff(1)
    ax_diff = diff1.plot(figsize=(8, 5))
    ax_diff.set_title("一阶差分后的序列")
    ax_diff.set_ylabel("差分值")
    save_show()
    print("p-value: %f" % adfuller(diff1[1:])[1])

    best_aic = np.inf
    best_pdq = None
    for param in pdq:
        try:
            results = sm.tsa.ARIMA(dta, order=param).fit()
            if results.aic < best_aic:
                best_aic = results.aic
                best_pdq = param
        except Exception:
            continue

    print("Best ARIMA{} model - AIC:{}".format(best_pdq, best_aic))
    arima110 = sm.tsa.ARIMA(dta, order=(1, 1, 0)).fit()
    print(arima110.summary())


if __name__ == "__main__":
    run_prophet_demo()
    run_sarima_demo()
    run_lstm_demo()
    run_arima_demo()
