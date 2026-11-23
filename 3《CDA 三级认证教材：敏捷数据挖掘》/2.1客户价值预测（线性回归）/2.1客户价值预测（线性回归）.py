# -*- coding: utf-8 -*-
"""
2.1 客户价值预测（线性回归）

项目背景
--------
银行希望根据客户的人口统计特征、收入水平与消费能力，预测客户的信用卡平均月支出
（avg_exp，客户价值），从而支撑差异化营销与额度管理。

本脚本覆盖教材中的完整建模流程：
    1. 数据读取与探索性分析（散点图、相关系数）
    2. 一元与多元线性回归
    3. 向前逐步回归（基于 AIC）
    4. 残差诊断、对数变换与异常值识别
    5. 多重共线性（VIF）诊断与变量重构
    6. 类别变量、交互项建模
    7. 样本外预测与结果落盘
    8. 正则化回归：岭回归、Lasso、ElasticNet 与正则路径

数据文件
--------
    creditcard_exp.csv                    训练样本（含 avg_exp）
    creditcard_exp_out_of_sample.csv      样本外客户（无 avg_exp，用于预测）

依赖
----
    numpy pandas matplotlib statsmodels scikit-learn
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import OLSInfluence

BASE_DIR = Path(__file__).resolve().parent

# 图表统一输出到章节目录下的 figures/，便于离线查看与归档
FIG_PREFIX = '2.1客户价值预测'
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

pd.set_option("display.max_columns", 8)

# 中文字体设置，避免图中中文显示为方框
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 一、数据读取与准备
# ============================================================
raw_data = pd.read_csv(BASE_DIR / "creditcard_exp.csv", skipinitialspace=True)
out_of_sample_data = pd.read_csv(
    BASE_DIR / "creditcard_exp_out_of_sample.csv", skipinitialspace=True
)

print("训练数据维度:", raw_data.shape)
print(raw_data.head())

train = (
    raw_data[raw_data["avg_exp"].notnull()]
    .copy()
    .iloc[:, 2:]
    .drop("age2", axis=1)
)

oos_data = out_of_sample_data.copy().drop("age2", axis=1)

print(train.describe(include="all"))


# ============================================================
# 二、探索性分析：客户价值与收入的关系
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))
train.plot("Income", "avg_exp", kind="scatter", ax=ax)
ax.set_ylabel("平均支出", fontsize=15)
ax.set_xlabel("收入", fontsize=15)
ax.set_title("客户收入与平均支出的关系", fontsize=15)
save_show()

print(train[["Income", "avg_exp", "Age", "dist_home_val"]].corr(method="pearson"))


# ============================================================
# 三、一元线性回归：avg_exp ~ Income
# ============================================================
simple_linear_model = ols(formula="avg_exp ~ Income", data=train).fit()
print("一元回归系数:\n", simple_linear_model.params)

summary = simple_linear_model.summary()
for table in summary.tables:
    print(table)

# 训练集拟合值与残差
train_prediction = simple_linear_model.predict(train)
train_resid = simple_linear_model.resid
print(
    pd.DataFrame(
        [train_prediction, train_resid], index=["Pred_avg_exp", "resid"]
    ).T.head()
)

# 样本外预测
print("样本外前 5 个预测值:\n", simple_linear_model.predict(oos_data)[:5])


# ============================================================
# 四、多元线性回归
# ============================================================
multiple_linear_model = ols(
    "avg_exp ~ Income + dist_home_val + dist_avg_income", data=train
).fit()
print(multiple_linear_model.summary())
print("多元回归系数:\n", multiple_linear_model.params)


# ============================================================
# 五、向前逐步回归（AIC 准则）
# ============================================================
def forward_select(data, response):
    """逐步向模型中添加变量，直到 AIC 不再下降为止。"""
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = float("inf"), float("inf")
    while remaining:
        aic_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {}".format(response, " + ".join(selected + [candidate]))
            aic = ols(formula=formula, data=data).fit().aic
            aic_with_candidates.append((aic, candidate))
        aic_with_candidates.sort(reverse=True)
        best_new_score, best_candidate = aic_with_candidates.pop()
        if current_score > best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
            print("aic is {}, continuing!".format(current_score))
        else:
            print("forward selection over!")
            break

    formula = "{} ~ {}".format(response, " + ".join(selected))
    print("final formula is {}".format(formula))
    return ols(formula=formula, data=data).fit()


data_for_select = train[
    ["avg_exp", "Income", "Age", "dist_home_val", "dist_avg_income"]
]
forward_select_model = forward_select(data=data_for_select, response="avg_exp")
print("逐步回归 R 方:", forward_select_model.rsquared)


# ============================================================
# 六、残差诊断工具
# ============================================================
def plot_resid(resid, x, y_hat, **kw):
    """绘制“残差-解释变量”和“残差-拟合值”两张散点图。"""
    plt.figure(figsize=(8, 8))

    ax1 = plt.subplot(211)
    ax1.set_ylabel("残差", fontsize=15)
    ax1.set_xlabel("解释变量", fontsize=15)
    ax1.scatter(x, resid, **kw)
    ax1.set_title("残差与解释变量散点图", fontsize=15)

    ax2 = plt.subplot(212)
    ax2.set_ylabel("残差", fontsize=15)
    ax2.set_xlabel("拟合值", fontsize=15)
    ax2.scatter(y_hat, resid, **kw)
    ax2.set_title("残差与拟合值散点图", fontsize=15)

    plt.tight_layout()
    save_show()


# ============================================================
# 七、对数变换：比较三种函数形式
# ============================================================
train["Income_ln"] = np.log(train["Income"])
oos_data["Income_ln"] = np.log(oos_data["Income"])

linear_model1 = ols(formula="avg_exp ~ Income", data=train).fit()
linear_model2 = ols(formula="avg_exp_ln ~ Income", data=train).fit()
linear_model3 = ols(formula="avg_exp_ln ~ Income_ln", data=train).fit()

plot_resid(linear_model1.resid, train["Income"], linear_model1.predict(train))
plot_resid(linear_model2.resid, train["Income"], linear_model2.predict(train))
plot_resid(linear_model3.resid, train["Income_ln"], linear_model3.predict(train))

fig, ax = plt.subplots(figsize=(6, 6))
sm.qqplot(linear_model3.resid, fit=True, line="45", ax=ax, figure=fig)
ax.set_xlabel("理论分位数", fontsize=15)
ax.set_ylabel("样本分位数", fontsize=15)
ax.set_title("残差正态 Q-Q 图", fontsize=15)
save_show()

import pprint

pprint.pprint(
    {
        "exp ~ Income": linear_model1.rsquared,
        "ln(exp) ~ Income": linear_model2.rsquared,
        "ln(exp) ~ ln(Income)": linear_model3.rsquared,
    }
)


# ============================================================
# 八、异常值识别与处理
# ============================================================
train["resid"] = linear_model3.resid
student_resid = (train["resid"] - train["resid"].mean()) / train["resid"].std()

print("学生化残差绝对值大于 2 的样本:\n", train[np.abs(student_resid) > 2])
print(OLSInfluence(linear_model3).summary_frame().head())

train = train[np.abs(student_resid) <= 2].copy()
linear_model4 = ols("avg_exp_ln ~ Income_ln", train).fit()
plot_resid(linear_model4.resid, train["Income_ln"], linear_model4.predict(train))
print("剔除异常值后的 R 方:", linear_model4.rsquared)


# ============================================================
# 九、多重共线性诊断（VIF）
# ============================================================
def vif(df, y_col):
    """以某个解释变量为因变量、其余变量为自变量回归，计算 VIF。"""
    x_cols = list(df.columns)
    x_cols.remove(y_col)
    formula = y_col + "~" + "+".join(x_cols)
    rsquared = ols(formula, df).fit().rsquared
    return 1.0 / (1.0 - rsquared)


train["dist_home_val_ln"] = np.log(train["dist_home_val"])
train["dist_avg_income_ln"] = np.log(train["dist_avg_income"])
oos_data["dist_home_val_ln"] = np.log(oos_data["dist_home_val"])
oos_data["dist_avg_income_ln"] = np.log(oos_data["dist_avg_income"])

linear_model5 = ols(
    "avg_exp_ln ~ Age + Income_ln + dist_home_val_ln + dist_avg_income_ln", train
).fit()
print(linear_model5.summary())

exog = train[["Age", "Income_ln", "dist_home_val_ln", "dist_avg_income_ln"]]
for y_col in exog.columns:
    print('VIF of "%s" is %.2f' % (y_col, vif(df=exog, y_col=y_col)))

# Income_ln 与 dist_avg_income_ln 高度共线，构造收入占比变量替代 Income_ln
train["high_avg_ratio"] = train["high_avg"] / train["dist_avg_income"]
oos_data["high_avg_ratio"] = oos_data["high_avg"] / oos_data["dist_avg_income"]

exog = train[
    ["Age", "high_avg_ratio", "dist_home_val_ln", "dist_avg_income_ln"]
]
for y_col in exog.columns:
    print('VIF of "%s" is %.2f' % (y_col, vif(df=exog, y_col=y_col)))

var_select_data = train[
    ["avg_exp_ln", "Age", "high_avg_ratio", "dist_home_val_ln", "dist_avg_income_ln"]
]
linear_model6 = forward_select(data=var_select_data, response="avg_exp_ln")
print("r_squared is ", linear_model6.rsquared)


# ============================================================
# 十、类别变量与交互项
# ============================================================
formula = """
avg_exp_ln ~ dist_avg_income_ln + dist_home_val_ln +
C(gender) + C(Ownrent) + C(Selfempl) + C(edu_class)
"""
linear_model7 = ols(formula, train).fit()
print(linear_model7.summary())

formula = """
avg_exp_ln ~ dist_avg_income_ln + dist_home_val_ln +
C(Selfempl) + C(gender) * C(edu_class)
"""
linear_model8 = ols(formula, train).fit()
print(linear_model8.summary())


# ============================================================
# 十一、样本外预测与结果保存
# ============================================================
output_data_dir = BASE_DIR / "data_processed"
output_data_dir.mkdir(exist_ok=True)
output_data_path = output_data_dir / "creditcard_exp_linear_regression.csv"

# 训练集中 edu_class 取值为 1/2/3，而样本外数据中只有 0/1，将 0 归并为 1
oos_data["edu_class"].replace({0: 1}, inplace=True)
oos_data["Pred_avg_exp"] = np.exp(linear_model8.predict(oos_data))
oos_data.to_csv(output_data_path, index=False)
print("样本外预测结果已保存至:", output_data_path)


# ============================================================
# 十二、岭回归（L2 正则）
# ============================================================
continuous_xcols = ["Income", "dist_home_val", "dist_avg_income"]
raw = train[continuous_xcols].copy()
train_scale = (raw - raw.mean()) / raw.std()

oos_raw = oos_data[continuous_xcols].copy()
oos_scale = (oos_raw - raw.mean()) / raw.std()

train_scale["avg_exp"] = train["avg_exp"]

linear_model_ridge = ols(
    "avg_exp ~ Income + dist_home_val + dist_avg_income", data=train_scale
).fit_regularized(alpha=1, L1_wt=0)
print(linear_model_ridge.predict(oos_scale))


# ============================================================
# 十三、scikit-learn 正则化回归
# ============================================================
from sklearn.linear_model import ElasticNetCV, Lasso, LassoCV, Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

continuous_xcols = ["Age", "Income", "dist_home_val", "dist_avg_income"]

scaler = StandardScaler()
scaler.fit(train[continuous_xcols])
X = scaler.transform(train[continuous_xcols])
oos_scale = scaler.transform(oos_data[continuous_xcols])
y = train["avg_exp_ln"]

alphas = np.logspace(-2, 3, 100, base=10)

rcv = RidgeCV(alphas=alphas, cv=3)
rcv.fit(X, y)
rcv_predict = rcv.predict(oos_scale)
print("oos prediction:", np.exp(rcv_predict[:5]))
print("coef of linear model:", rcv.coef_)
print("The best alpha is {}".format(rcv.alpha_))
print("The r-square is {}".format(rcv.score(X, y)))

# 正则系数路径
ridge = Ridge()
coefs = []
for alpha in alphas:
    ridge.set_params(alpha=alpha)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)

ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale("log")
plt.xlabel("正则化参数")
plt.ylabel("回归系数")
plt.title("岭回归系数随正则化参数的变化")
plt.axis("tight")
save_show()

ridge.set_params(alpha=10)
ridge.fit(X, y)
print("Coefs of the model: ", ridge.coef_)
print("The r-square of the model:", ridge.score(X, y))
print("Prediction of the model: ", np.exp(ridge.predict(oos_scale)[:5]))


# ============================================================
# 十四、Lasso 回归（L1 正则）
# ============================================================
lmr1 = ols(
    "avg_exp ~ Age + Income + dist_home_val + dist_avg_income", data=train
).fit_regularized(alpha=1, L1_wt=1)

lasso_alphas = np.logspace(-3, 0, 100, base=10)
lcv = LassoCV(alphas=lasso_alphas, cv=10)
lcv.fit(X, y)
print("The best alpha is {}".format(lcv.alpha_))
print("The r-square is {}".format(lcv.score(X, y)))

lasso = Lasso()
lasso_coefs = []
for alpha in lasso_alphas:
    lasso.set_params(alpha=alpha)
    lasso.fit(X, y)
    lasso_coefs.append(lasso.coef_)

ax = plt.gca()
ax.plot(lasso_alphas, lasso_coefs)
ax.set_xscale("log")
plt.xlabel("正则化参数")
plt.ylabel("回归系数")
plt.title("Lasso 回归系数随正则化参数的变化")
plt.axis("tight")
save_show()

lasso.set_params(alpha=0.02)
lasso.fit(X, y)
print("Coefs of the model: ", lasso.coef_)
print("The best r-square of the model:", lasso.score(X, y))
print("Prediction of the model: ", np.exp(lasso.predict(oos_scale)[:5]))


# ============================================================
# 十五、ElasticNet 回归（L1 + L2）
# ============================================================
l1_ratio = [0.01, 0.1, 0.2, 0.4, 0.8, 0.95, 0.99, 1]

encv = ElasticNetCV(l1_ratio=l1_ratio, n_alphas=100, n_jobs=-1, cv=4)
encv.fit(X, y)

print("The best l1_ratio is {}".format(encv.l1_ratio_))
print("The best alpha is {}".format(encv.alpha_))
print("The best r-square of the model is:", encv.score(X, y))
print("The prediction of the ElasticNet is:", np.exp(encv.predict(oos_scale)[:5]))
