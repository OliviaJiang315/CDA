# -*- coding: utf-8 -*-
"""
6 特征工程

项目背景
--------
特征工程直接决定模型上限。教材以电信客户账单数据为例，覆盖：

    1. 数据清洗：离群值截断、缺失值处理
    2. 数值变换：log 变换、标准化、归一化、范数归一化
    3. 离散化：等宽分箱、等频分箱、人工分箱
    4. 类别编码：独热编码
    5. 特征构造：单位价格、TAC 编码、在网时长
    6. 降维：PCA、LDA
    7. 特征选择：方差过滤、卡方、ANOVA、F 检验、RFE、树模型重要性

数据文件
--------
    telecom_bill.csv
        ID, churn, join_time, vip, product, expenditure, call, traffic, state, IMEI

依赖
----
    numpy pandas matplotlib scikit-learn
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

# 图表统一输出到章节目录下的 figures/，便于离线查看与归档
FIG_PREFIX = '6 特征工程'
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
pd.set_option("display.max_columns", 10)

# 中文字体设置，避免图中中文显示为方框
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 一、读取数据与简单清洗
# ============================================================
raw = pd.read_csv(BASE_DIR / "telecom_bill.csv")
print(raw.head())

# traffic >= 5000 的样本明显异常，剔除
print(raw[raw.traffic >= 5000].head())
data = raw[raw.traffic < 5000].copy()

statesCount = data["ID"].groupby(data.state).count()
print(statesCount.head(10))

plt.figure(figsize=(6, 6))
plt.pie(
    statesCount.values,
    labels=["状态 %g" % s for s in statesCount.index],
    colors=("b", "g", "r", "c", "m", "y", "tan", "w"),
    autopct="%1.1f%%",
)
plt.title("客户所属状态占比")
plt.axis("equal")
save_show()

# 状态 2、4 视为缺失
new_state = data["state"].replace([2, 4], np.nan)
print(new_state.head())


# ============================================================
# 二、离群值处理：3σ 截断
# ============================================================
plt.boxplot(data.expenditure)
plt.title("消费支出箱线图（原始）")
plt.ylabel("消费支出")
save_show()

tmp = data.expenditure.copy()
mu = data.expenditure.mean()
sigma = data.expenditure.std()
lb = mu - 3 * sigma
hb = mu + 3 * sigma
tmp[tmp < lb] = lb
tmp[tmp > hb] = hb

plt.boxplot(tmp)
plt.title("消费支出箱线图（3σ 截断后）")
plt.ylabel("消费支出")
save_show()

data.expenditure.hist(figsize=(8, 4), bins=40)
plt.title("消费支出分布")
plt.xlabel("消费支出")
plt.ylabel("频数")
save_show()


# ============================================================
# 三、数值变换：log1p
# ============================================================
data["log_expd"] = np.log1p(tmp)
data.log_expd.hist(figsize=(8, 4), bins=20)
plt.title("对数消费支出分布")
plt.xlabel("对数（消费支出 + 1）")
plt.ylabel("频数")
save_show()

from sklearn.preprocessing import FunctionTransformer

functionTF = FunctionTransformer(np.log1p)
log_data = functionTF.fit_transform(data[["call", "traffic"]])
data["log_call"] = log_data.iloc[:, 0]
data["log_traffic"] = log_data.iloc[:, 1]
print(data.head())


# ============================================================
# 四、缺失值处理
# ============================================================
print(data.count())
print(data.isnull().sum())
print(data.dropna(thresh=1, subset=["state", "IMEI"]).count())

# 注意：全部填 0 只适用于该演示数据，一般业务中不可这样处理
print(data.fillna(value=0, inplace=False).head())

top_state = data["state"].mode()[0]
data.fillna({"IMEI": 0, "state": top_state}, inplace=True)
print(data.count())

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
imputer.fit_transform(raw[["state", "IMEI"]])

# 二值化：是否登记 IMEI
from sklearn.preprocessing import Binarizer

binarizer = Binarizer(threshold=0, copy=True)
data["has_IMEI"] = binarizer.fit_transform(data[["IMEI", ]])
print(data.head())

print(
    pd.cut(
        data["IMEI"], bins=[-float("inf"), 0, float("inf")], labels=False
    ).head()
)


# ============================================================
# 五、在网时长与离散化
# ============================================================
dura_col = datetime(2015, 1, 1) - pd.to_datetime(data["join_time"])
data["duration"] = dura_col.map(lambda x: x.days / 30.0)

equal_width = pd.cut(data["duration"], 5)
print(data["churn"].groupby(equal_width).agg(["count", "mean", "sum"]))

sample_labels, retbins = pd.cut(
    data["duration"], 5, labels=range(5), retbins=True
)
print(sample_labels[:5])
print(retbins)

equal_depth = pd.qcut(data["duration"], 14)
discretization_dura = data["churn"].groupby(equal_depth).agg(
    ["count", "mean", "sum"]
)
print(discretization_dura.head())

discretization_dura["mean"].plot(kind="bar")
plt.title("等频分箱下各区间流失率")
plt.xlabel("在网时长区间（月）")
plt.ylabel("流失率")
save_show()

bins = [-np.inf, 5, 12, 36, np.inf]
artificial_width = pd.cut(data["duration"], bins)
data["churn"].groupby(artificial_width).agg(["mean"]).plot(kind="bar")
plt.title("人工分箱下各区间流失率")
plt.xlabel("在网时长区间（月）")
plt.ylabel("流失率")
save_show()


# ============================================================
# 六、类别编码
# ============================================================
dummies = pd.get_dummies(data.state, prefix="s")
print(data.join(dummies).head())

from sklearn.preprocessing import OneHotEncoder

print(OneHotEncoder().fit_transform(data[["state", ]]).toarray())


# ============================================================
# 七、连续变量标准化 / 归一化
# ============================================================
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler

consumptions = data.loc[:, "log_expd":"log_traffic"]
std_scaler = StandardScaler()
std_data = std_scaler.fit_transform(consumptions)

std_df = pd.DataFrame(
    std_data,
    columns=["s_expd", "s_call", "s_traffic"],
    index=data.index,
)
data1 = data.join(std_df)
print(data1.head())

range_scaler = MinMaxScaler()
print(range_scaler.fit_transform(consumptions)[:5])

print(Normalizer().fit_transform(data1.iloc[:, -3:]))


# ============================================================
# 八、特征构造
# ============================================================
data1["unit_price"] = data1["expenditure"] / (data1["traffic"] + 1.0)
data1["TAC"] = data1["IMEI"].map(lambda x: str(x)[:6])
print(data1.head())


# ============================================================
# 九、降维：PCA 与 LDA
# ============================================================
from sklearn.decomposition import PCA

pca = PCA(n_components=2, whiten=True)
pca.fit_transform(data1[["s_expd", "s_call", "s_traffic"]])
print("每个主成分解释的方差比例:", pca.explained_variance_ratio_)
print("每个主成分解释的方差:", pca.explained_variance_)
print("主成分方向:", pca.components_)

print(
    pca.transform(
        pd.DataFrame(
            data=[[-1, 0, 1], [1, 2, 3]],
            columns=["s_expd", "s_call", "s_traffic"],
        )
    )
)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()
lda.fit_transform(data1[["s_expd", "s_call", "s_traffic"]], data1.churn)
print("LDA 投影方向:", lda.coef_)

print(
    lda.transform(
        pd.DataFrame(
            [[-1, 0, 1], [1, 2, 3]],
            columns=["s_expd", "s_call", "s_traffic"],
        )
    )
)

prediction = lda.predict(data1[["s_expd", "s_call", "s_traffic"]])
result = prediction == data1.churn
print("LDA 预测准确率:", result.mean())


# ============================================================
# 十、特征选择
# ============================================================
from sklearn.feature_selection import VarianceThreshold

var_th = VarianceThreshold(threshold=0.09)
var_th.fit_transform(data1[["vip", "has_IMEI"]])
print("各变量方差:", var_th.variances_)

x_categorical = data1[["vip", "state", "has_IMEI"]]
x_continuous = data1[["s_expd", "s_call", "s_traffic", "duration"]]
y_categorical = data1["churn"]
y_continuous = data1["s_expd"]

# 离散变量 vs 离散目标：卡方检验
from sklearn.feature_selection import SelectKBest, chi2

print(chi2(x_categorical, y_categorical))

selector = SelectKBest(chi2, k=2).fit(x_categorical, y_categorical)
selector.transform(x_categorical)
print("卡方保留掩码:", selector.get_support())

# 连续变量 vs 离散目标：ANOVA F 检验
from sklearn.feature_selection import f_classif

print(f_classif(x_continuous, y_categorical))

f_classif_selector = SelectKBest(f_classif, k=2)
print(f_classif_selector.fit_transform(x_continuous, y_categorical))
print("ANOVA 保留掩码:", f_classif_selector.get_support())

# 连续变量 vs 连续目标：F 回归
from sklearn.feature_selection import f_regression

print(f_regression(x_continuous, y_continuous))

f_regression_selector = SelectKBest(f_regression, k=2)
print(f_regression_selector.fit_transform(x_continuous, y_continuous))
print("F 回归保留掩码:", f_regression_selector.get_support())

# 递归特征消除
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

rfe = RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=2)
rfe.fit_transform(x_continuous, y_categorical)
print("RFE 保留掩码:", rfe.get_support())

# 树模型特征重要性
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(random_state=42)
rfc.fit(x_continuous, y_categorical)
print(list(zip(x_continuous.columns, rfc.feature_importances_)))

# 基于模型的特征选择
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel

gbdt_selector = SelectFromModel(
    GradientBoostingClassifier(random_state=42), threshold="mean"
)
gbdt_selector.fit_transform(x_continuous, y_categorical)
print("GBDT 保留掩码:", gbdt_selector.get_support())
