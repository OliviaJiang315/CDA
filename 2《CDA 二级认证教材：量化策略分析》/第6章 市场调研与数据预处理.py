# -*- coding: utf-8 -*-
"""
第6章 市场调研与数据预处理

对应教材：
    第6章-市场调研与数据预处理/1数据采集与处理 1 数据采集方法/抽样.ipynb
    第6章-市场调研与数据预处理/1数据采集与处理 1 数据采集方法/6_1sampling.py
    第6章-市场调研与数据预处理/2 数据预处理基础/数据预处理.ipynb
    第6章-市场调研与数据预处理/2 数据预处理基础/woe.py

内容：
1. 市场调研中的三种抽样方法：简单随机抽样、分层抽样、系统抽样；
2. 数据预处理：脏数据识别、缺失值处理、分布检查；
3. WoE（Weight of Evidence）分箱工具：连续变量/离散变量分箱、IV 计算、
   因子变换、分箱合并、决策树寻优。

依赖：pandas、numpy、scikit-learn、matplotlib。
"""

import os
import math
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHAPTER_DIR = os.path.join(BASE_DIR, "第6章-市场调研与数据预处理")
SAMPLING_DIR = os.path.join(CHAPTER_DIR, "1数据采集与处理 1 数据采集方法")
PREPROCESS_DIR = os.path.join(CHAPTER_DIR, "2 数据预处理基础")
FIG_DIR = os.path.join(BASE_DIR, "图表输出", "第6章 市场调研与数据预处理")
os.makedirs(FIG_DIR, exist_ok=True)

_fig_no = [0]
_orig_show = plt.show


def _save_figure():
    """保存当前图形到本章图表目录。"""
    fig = plt.gcf()
    name = ""
    if fig._suptitle is not None:
        name = fig._suptitle.get_text()
    if not name:
        for ax in fig.axes:
            if ax.get_title():
                name = ax.get_title()
                break
    _fig_no[0] += 1
    safe = "".join(c for c in name if c not in '\\/:*?"<>|').strip() or "图"
    path = os.path.join(FIG_DIR, "%02d_%s.png" % (_fig_no[0], safe[:40]))
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    print("已保存图表：", path)


def _show_and_save(*args, **kwargs):
    _save_figure()
    _orig_show(*args, **kwargs)
    plt.close("all")


plt.show = _show_and_save

warnings.filterwarnings("ignore")


def title(text):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


# ======================================================================
# 第一部分  数据采集：抽样
# ======================================================================
title("6.1 抽样：简单随机、分层、系统抽样")


def get_sample(df, sampling="simple_random", k=1, stratified_col=None):
    """对输入的数据框进行抽样。

    参数：
        df：pandas.DataFrame，总体数据；
        sampling：抽样方法，可选 "simple_random"（简单随机）、
                  "stratified"（分层）、"systematic"（系统）；
        k：抽样个数或抽样比例。0 < k < 1 表示比例；k >= 1 表示个数，
           分层抽样时表示每层的样本量；
        stratified_col：分层所依据的列名列表，仅分层抽样生效。

    返回：
        抽样结果 DataFrame。
    """
    len_df = len(df)
    if k <= 0:
        raise AssertionError("k不能为负数")

    if k >= 1:
        assert isinstance(k, int), "选择抽样个数时, k必须为正整数"
        sample_by_n = True
        if sampling == "stratified":
            n_strata = df.groupby(by=stratified_col)[stratified_col[0]].count().count()
            alln = k * n_strata
            if alln >= len_df:
                raise AssertionError("请确认k乘以层数不能超过总样本量")
    else:
        sample_by_n = False
        if sampling in ("simple_random", "systematic"):
            k = math.ceil(len_df * k)

    if sampling == "simple_random":
        print("使用简单随机抽样")
        idx = random.sample(range(len_df), k)
        return df.iloc[idx, :].copy()

    elif sampling == "systematic":
        print("使用系统抽样")
        step = len_df // k + 1
        start = 0
        idx = range(len_df)[start::step]
        return df.iloc[idx, :].copy()

    elif sampling == "stratified":
        assert stratified_col is not None, "请传入包含需要分层的列名的列表"
        assert all(np.isin(stratified_col, df.columns)), "请检查输入的列名"

        grouped = df.groupby(by=stratified_col)[stratified_col[0]].count()
        if sample_by_n:
            group_k = grouped.map(lambda x: k)
        else:
            group_k = grouped.map(lambda x: math.ceil(x * k))

        pieces = []
        for group_idx in group_k.index:
            df1 = df
            if len(stratified_col) == 1:
                df1 = df1[df1[stratified_col[0]] == group_idx]
            else:
                for i in range(len(group_idx)):
                    df1 = df1[df1[stratified_col[i]] == group_idx[i]]
            idx = random.sample(range(len(df1)), int(group_k[group_idx]))
            pieces.append(df1.iloc[idx, :].copy())
        return pd.concat(pieces, ignore_index=False)

    else:
        raise AssertionError("sampling is illegal")


# 客户数据包含地区（Dist）等信息，用于演示三种抽样
clients = pd.read_csv(os.path.join(SAMPLING_DIR, "clients.csv"), encoding="gbk")
print("总体样本量：", len(clients))
print(clients.columns.tolist())

# 1. 简单随机抽样
srn = get_sample(clients, sampling="simple_random", k=22)   # 按数量
print("简单随机抽样（22 人）：", len(srn))
srp = get_sample(clients, sampling="simple_random", k=0.1)  # 按比例
print("简单随机抽样（10%）：", len(srp))

# 2. 分层抽样：按地区 Dist 分层
strn = get_sample(clients, sampling="stratified", k=2, stratified_col=["Dist"])
print("分层抽样（每层 2 人）：", len(strn))
strp = get_sample(clients, sampling="stratified", k=0.1, stratified_col=["Dist"])
print("分层抽样（每层 10%）：", len(strp))

# 3. 系统抽样
sysn = get_sample(clients, sampling="systematic", k=4)
print("系统抽样（4 人）：", len(sysn))
sysp = get_sample(clients, sampling="systematic", k=0.1)
print("系统抽样（10%）：", len(sysp))


# ======================================================================
# 第二部分  数据预处理基础
# ======================================================================
title("6.2 数据预处理：脏数据识别与缺失值")

camp = pd.read_csv(os.path.join(PREPROCESS_DIR, "teleco_camp_orig.csv"))
print("数据形状：", camp.shape)
print(camp.head())

# 查看收入分布，识别 0 值等异常集中点
plt.figure(figsize=(7, 4))
plt.hist(camp["AvgIncome"], bins=20)
plt.title("AvgIncome 原始分布")
plt.xlabel("AvgIncome")
plt.ylabel("频数")
plt.show()

print("AvgIncome 描述统计：\n", camp["AvgIncome"].describe(include="all"))
print("AvgHomeValue 描述统计：\n", camp["AvgHomeValue"].describe(include="all"))

# 收入中的 0 值应视为缺失值
camp["AvgIncome"] = camp["AvgIncome"].replace({0: np.nan})

# 存在缺失值后，画直方图需显式指定值域
plt.figure(figsize=(7, 4))
plt.hist(
    camp["AvgIncome"],
    bins=20,
    range=(camp["AvgIncome"].min(), camp["AvgIncome"].max()),
)
plt.title("AvgIncome 清洗后分布（0 值已转为缺失）")
plt.xlabel("AvgIncome")
plt.ylabel("频数")
plt.show()

print("清洗后 AvgIncome 缺失率：%.2f%%" % (camp["AvgIncome"].isnull().mean() * 100))
print("清洗后 AvgIncome 描述统计：\n", camp["AvgIncome"].describe(include="all"))


# ======================================================================
# 第三部分  WoE 分箱工具
# ======================================================================
title("6.3 WoE 分箱与 IV 计算")


class DiscreteVarOverFlowError(Exception):
    """离散变量取值过多时抛出的异常。"""


class WoE:
    """连续变量与离散变量的 WoE（Weight of Evidence）分箱工具。

    属性：
        bins：每个分箱的统计量及 WoE；
        iv：该变量的信息值（Information Value）；
        df：带分箱标签和 WoE 的原始样本。
    """

    def __init__(
        self,
        qnt_num=16,
        min_block_size=16,
        spec_values=None,
        v_type="c",
        bins=None,
        t_type="b",
    ):
        """
        参数：
            qnt_num：连续变量初始分箱数上限；
            min_block_size：每个分箱的最小样本量；
            spec_values：特殊值列表或字典 {标签: 取值}；
            v_type：'c' 连续变量，'d' 离散变量；
            bins：预设的连续变量分箱边界；
            t_type：'b' 二分类目标，'c' 连续目标。
        """
        self.__qnt_num = qnt_num
        self._predefined_bins = None if bins is None else np.array(bins)
        self.type = v_type
        self._min_block_size = min_block_size
        self._gb_ratio = None
        self.bins = None
        self.df = None
        self.qnt_num = None
        self.t_type = t_type

        if isinstance(spec_values, dict):
            self.spec_values = {}
            for k, v in spec_values.items():
                if str(v).startswith("d_"):
                    self.spec_values[k] = v
                else:
                    self.spec_values[k] = "d_" + str(v)
        elif spec_values is None:
            self.spec_values = {}
        else:
            self.spec_values = {i: "d_" + str(i) for i in spec_values}

    def fit(self, x, y):
        """拟合 WoE 变换。"""
        if not isinstance(x, pd.Series) or not isinstance(y, pd.Series):
            raise TypeError("pandas.Series type expected")
        if x.size != y.size:
            raise Exception("X size don't match Y size")

        t_bad = np.sum(y)
        if t_bad == 0 or t_bad == y.size:
            raise ValueError("There should be BAD and GOOD observations in the sample")
        if np.max(y) > 1 or np.min(y) < 0:
            raise ValueError("Y range should be between 0 and 1")

        # 离散变量把所有取值作为特殊值
        if self.type == "d":
            sp_values = {i: "d_" + str(i) for i in x.unique()}
            if len(sp_values) > 100:
                raise DiscreteVarOverFlowError(
                    "Discrete variable with too many unique values (more than 100)"
                )
            if self.spec_values:
                sp_values.update(self.spec_values)
            self.spec_values = sp_values

        df = pd.DataFrame({"X": x, "Y": y, "order": np.arange(x.size)})
        df_sp_values, df_cont = self._split_sample(df)
        df_cont, c_bins = self._cont_labels(df_cont)
        df_sp_values, d_bins = self._disc_labels(df_sp_values)

        self.df = pd.concat([df_sp_values, df_cont])
        self.bins = pd.concat([d_bins, c_bins], ignore_index=True)

        self._calc_stat()

        # 排序后便于 transform 时切分
        try:
            self.bins = self.bins.sort_values("bins").reset_index(drop=True)
        except TypeError:
            # 离散变量与连续变量混合时 bins 可能是混合类型，跳过排序
            pass
        self.df = self.df.sort_values("order")
        self.df = self.df.set_index(x.index)
        return self

    def fit_transform(self, x, y):
        """拟合并返回 WoE 变换后的列。"""
        self.fit(x, y)
        return self.df["woe"]

    def _split_sample(self, df):
        if self.type == "d":
            return df, None
        sp_values_flag = df["X"].isin(self.spec_values.keys()).values | df["X"].isnull().values
        df_sp_values = df[sp_values_flag].copy()
        df_cont = df[~sp_values_flag].copy()
        return df_sp_values, df_cont

    def _disc_labels(self, df):
        if df is None:
            return None, pd.DataFrame({"bins": [], "labels": []})
        df["labels"] = df["X"].apply(
            lambda x: self.spec_values[x] if x in self.spec_values.keys() else "d_" + str(x)
        )
        d_bins = pd.DataFrame({"bins": df["X"].unique()})
        d_bins["labels"] = d_bins["bins"].apply(
            lambda x: self.spec_values[x] if x in self.spec_values.keys() else "d_" + str(x)
        )
        # 多个特殊值可能映射到同一标签（例如合并 0、1 两个箱），按标签去重
        d_bins = d_bins.drop_duplicates(subset=["labels"]).reset_index(drop=True)
        return df, d_bins

    def _cont_labels(self, df):
        if df is None or df.empty:
            return df, pd.DataFrame({"bins": [], "labels": []})

        self.qnt_num = (
            int(np.minimum(df["X"].unique().size / self._min_block_size, self.__qnt_num)) + 1
        )
        bins = None
        cuts = None

        if self._predefined_bins is None:
            try:
                cuts, bins = pd.qcut(df["X"], self.qnt_num, retbins=True, labels=False)
            except ValueError as ex:
                if str(ex.args[0]).startswith("Bin edges must be unique"):
                    ex.args = (
                        "Please reduce number of bins or encode frequent items as special values",
                    ) + ex.args
                    raise
            bins = np.append((-float("inf"),), bins[1:-1])
        else:
            bins = self._predefined_bins
            if bins[0] != float("-Inf"):
                bins = np.append((-float("inf"),), bins)
            cuts = pd.cut(
                df["X"],
                bins=np.append(bins, (float("inf"),)),
                labels=np.arange(len(bins)).astype(str),
            )

        df = df.copy()
        df["labels"] = cuts.astype(str)
        c_bins = pd.DataFrame({"bins": bins, "labels": np.arange(len(bins)).astype(str)})
        return df, c_bins

    def _calc_stat(self):
        stat = self.df.groupby("labels")["Y"].agg(
            mean="mean",
            bad=lambda s: np.count_nonzero(s),
            obs="size",
        )
        if self.t_type != "b":
            stat["bad"] = stat["mean"] * stat["obs"]
        stat["good"] = stat["obs"] - stat["bad"]

        t_good = np.maximum(stat["good"].sum(), 0.5)
        t_bad = np.maximum(stat["bad"].sum(), 0.5)
        stat["woe"] = stat.apply(self._bucket_woe, axis=1) + np.log(t_good / t_bad)

        iv_stat = (stat["bad"] / t_bad - stat["good"] / t_good) * stat["woe"]
        self.iv = iv_stat.sum()

        self.bins = pd.merge(stat, self.bins, left_index=True, right_on=["labels"])
        label_woe = self.bins[["woe", "labels"]].drop_duplicates()
        self.df = pd.merge(self.df, label_woe, on="labels")

    def transform(self, x):
        """按已拟合的分箱规则对新数据进行 WoE 变换。"""
        if not isinstance(x, pd.Series):
            raise TypeError("pandas.Series type expected")
        if self.bins is None:
            raise Exception("Fit the model first, please")

        df = pd.DataFrame({"X": x, "order": np.arange(x.size)})
        df_sp_values, df_cont = self._split_sample(df)

        def get_sp_label(x_):
            if x_ in self.spec_values.keys():
                return self.spec_values[x_]
            str_x = "d_" + str(x_)
            if str_x in list(self.bins["labels"]):
                return str_x
            raise ValueError("Value " + str_x + " does not exist in the training set")

        df_sp_values["labels"] = df_sp_values["X"].apply(get_sp_label)

        c_bins = self.bins[self.bins["labels"].apply(lambda z: not str(z).startswith("d_"))]
        if self.type != "d":
            cuts = pd.cut(
                df_cont["X"],
                bins=np.append(c_bins["bins"].values, (float("inf"),)),
                labels=c_bins["labels"].values,
            )
            df_cont["labels"] = cuts.astype(str)

        df = pd.concat([df_sp_values, df_cont])
        df = pd.merge(df, self.bins[["woe", "labels"]], on="labels")
        df = df.sort_values("order")
        return df.set_index(x.index)

    def merge(self, label1, label2=None):
        """合并分箱。

        连续变量只需给 label1（与下一个箱合并）；
        离散变量需要给 label1、label2。
        """
        spec_values = self.spec_values.copy()
        c_bins = self.bins[self.bins["labels"].apply(lambda x: not str(x).startswith("d_"))].copy()

        if label2 is None and not str(label1).startswith("d_"):
            c_bins = c_bins[c_bins["labels"] != label1]
        else:
            if not (str(label1).startswith("d_") and str(label2).startswith("d_")):
                raise Exception("Labels should be discrete simultaneously")
            bin1 = self.bins[self.bins["labels"] == label1]["bins"].iloc[0]
            bin2 = self.bins[self.bins["labels"] == label2]["bins"].iloc[0]
            spec_values[bin1] = label1 + "_" + label2
            spec_values[bin2] = label1 + "_" + label2

        new_woe = WoE(
            self.__qnt_num,
            self._min_block_size,
            spec_values,
            self.type,
            c_bins["bins"],
            self.t_type,
        )
        return new_woe.fit(self.df["X"], self.df["Y"])

    def plot(self, figsize=(8, 5)):
        """绘制每个分箱的样本量与 WoE。"""
        index = np.arange(self.bins.shape[0])
        bar_width = 0.8
        woe_fig = plt.figure(figsize=figsize)
        plt.title("分箱样本量与 WoE")
        ax = woe_fig.add_subplot(111)
        ax.set_ylabel("样本量")
        plt.xticks(index + bar_width / 2, self.bins["labels"].astype(str), rotation=45)
        plt.bar(index, self.bins["obs"], bar_width, color="b", label="样本量")

        ax2 = ax.twinx()
        ax2.set_ylabel("WoE 值")
        ax2.plot(
            index + bar_width / 2,
            self.bins["woe"],
            "bo-",
            linewidth=2.0,
            color="r",
            label="WoE",
        )
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(handles1 + handles2, labels1 + labels2)
        woe_fig.autofmt_xdate()
        return woe_fig

    def optimize(self, criterion=None, fix_depth=None, max_depth=None, cv=3):
        """用决策树搜索连续变量的最优分箱边界。"""
        from sklearn import tree
        from sklearn.model_selection import cross_val_score

        tree_type = tree.DecisionTreeClassifier if self.t_type == "b" else tree.DecisionTreeRegressor
        m_depth = int(np.log2(self.__qnt_num)) + 1 if max_depth is None else max_depth

        cont = self.df["labels"].apply(lambda z: not str(z).startswith("d_"))
        x_train = np.array(self.df[cont]["X"]).reshape(-1, 1)
        y_train = np.array(self.df[cont]["Y"])

        if fix_depth is None:
            cv_scores = []
            for i in range(1, m_depth):
                d_tree = (
                    tree_type(max_depth=i)
                    if criterion is None
                    else tree_type(criterion=criterion, max_depth=i)
                )
                scores = cross_val_score(d_tree, x_train, y_train, cv=cv)
                cv_scores.append(scores.mean())
            best = int(np.argmax(cv_scores)) + 1
        else:
            best = fix_depth

        final_tree = tree_type(max_depth=best)
        final_tree.fit(x_train, y_train)
        opt_bins = final_tree.tree_.threshold[final_tree.tree_.threshold > 0]
        opt_bins = np.sort(opt_bins)

        new_woe = WoE(
            self.__qnt_num,
            self._min_block_size,
            self.spec_values,
            self.type,
            opt_bins,
            self.t_type,
        )
        return new_woe.fit(self.df["X"], self.df["Y"])

    @staticmethod
    def _bucket_woe(x):
        t_bad = x["bad"]
        t_good = x["good"]
        t_bad = 0.5 if t_bad == 0 else t_bad
        t_good = 0.5 if t_good == 0 else t_good
        return np.log(t_bad / t_good)


# WoE 使用示例：构造带特殊值（0、1、缺失）的连续变量
if __name__ == "__main__":
    title("6.3.1 WoE 工具演示")

    t_type_ = "c"   # 'b' 二分类目标，'c' 连续目标
    N = 300

    x1 = np.random.rand(N)
    x2 = np.random.rand(N)
    if t_type_ == "b":
        y_ = np.where(np.random.rand(N) + x1 + x2 > 2, 1, 0)
    else:
        y_ = np.random.rand(N) + x1 + x2
        y_ = (y_ - np.min(y_)) / (np.max(y_) - np.min(y_)) / 2

    # 插入特殊值与缺失值
    x1[0:20] = float("nan")
    x1[30:50] = 0.0
    x1[60:80] = 1.0
    x2[0:20] = float("nan")

    woe = WoE(7, 30, spec_values={0: "0", 1: "1"}, v_type="c", t_type=t_type_)
    woe.fit(pd.Series(x1), pd.Series(y_))
    print("分箱统计与 IV：")
    print(woe.bins)
    print("IV =", woe.iv)

    # 用 x1 的规则变换 x2
    transformed = woe.transform(pd.Series(x2))
    print("变换结果前 5 行：\n", transformed.head())

    # 决策树寻优
    woe2 = woe.optimize(max_depth=5)
    print("寻优后 IV =", woe2.iv)

    # 合并离散箱与连续箱
    woe3 = woe.merge("d_0", "d_1")
    woe4 = woe3.merge("2")
    print("合并后分箱：\n", woe4.bins[["labels", "obs", "woe"]])

    # 绘制 WoE 图
    fig = woe.plot()
    plt.show()
    fig = woe2.plot()
    plt.show()

    print("\n第6章全部演示执行完毕。")