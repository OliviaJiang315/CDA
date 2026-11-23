# -*- coding: utf-8 -*-
"""
2.4 交叉销售（关联规则与序列模式）

项目背景
--------
银行希望从客户的跨产品购买记录中发现“买了 A 就倾向买 B”的规律，用于产品推荐与
交叉销售。教材包含两条技术路线：

    2.4.1 关联规则（Apriori）：把每个客户拥有的产品看作一个购物篮，挖掘
          {产品组合} -> {产品} 的支持度、置信度、提升度。
    2.4.2 序列模式（AprioriAll / PrefixSpan）：考虑产品购买的先后顺序，
          挖掘“先买 A、后买 B”的序列模式。

本脚本同时保留了教材的三种实现：
    1. 自写 apriori / aprioriAll / prefixSpan（原始教材 myscript 中的实现）
    2. Orange3 的 FPGrowth 实现（选装）
    3. mlxtend 现成 Apriori 包的实现（选装）

数据文件
--------
    bank_prod_order.csv    字段：OrderNumber, CSR_ID, PROD（GBK 编码）

依赖
----
    必装：numpy pandas matplotlib
    选装：mlxtend、orange3（orangecontrib）
"""

import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

# 图表统一输出到章节目录下的 figures/，便于离线查看与归档
FIG_PREFIX = '2.4交叉销售'
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
DATA_PATH = BASE_DIR / "bank_prod_order.csv"

# 中文字体设置，避免图中中文显示为方框
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 一、自写 Apriori 算法（教材 myscript/apriori.py）
# ============================================================
def createC1(dataSet):
    """把所有购物篮打平，得到 1 项候选集。"""
    C1 = set(itertools.chain(*dataSet))
    return [frozenset([i]) for i in C1]


def scanD(dataSet, Ck, min_support):
    """扫描数据集，统计候选项集支持度并过滤。"""
    support = {}
    for basket in dataSet:
        for candidate in Ck:
            if candidate.issubset(basket):
                support[candidate] = support.get(candidate, 0) + 1

    n = len(dataSet)
    return {k: v / n for k, v in support.items() if v / n >= min_support}


def aprioriGen(Lk):
    """由 k 项频繁集两两合并生成 k+1 项候选集。"""
    lenLk = len(Lk)
    k = len(Lk[0])
    if lenLk > 1 and k > 0:
        return set(
            [
                Lk[i].union(Lk[j])
                for i in range(lenLk - 1)
                for j in range(i + 1, lenLk)
                if len(Lk[i] | Lk[j]) == k + 1
            ]
        )


def apriori(dataSet, min_support=0.5):
    """返回所有满足最小支持度的频繁项集字典 {frozenset: support}。"""
    C1 = createC1(dataSet)
    L1 = scanD(dataSet, C1, min_support)
    L = [L1]
    k = 2
    while len(L[k - 2]) > 1:
        Ck = aprioriGen(list(L[k - 2].keys()))
        Lk = scanD(dataSet, Ck, min_support)
        if len(Lk) > 0:
            L.append(Lk)
            k += 1
        else:
            break

    d = {}
    for Lk in L:
        d.update(Lk)
    return d


def rulesGen(iterable):
    """生成频繁项集的全部非空真子集，组成候选规则左右项。"""
    subSet = []
    for i in range(1, len(iterable)):
        subSet.extend(itertools.combinations(iterable, i))
    return [
        (frozenset(lhs), frozenset(iterable.difference(lhs))) for lhs in subSet
    ]


def arules(dataSet, min_support=0.5):
    """返回 pandas.DataFrame，字段：LHS、RHS、support、confidence、lift。"""
    L = apriori(dataSet, min_support)

    rules = []
    for Lk in L.keys():
        if len(Lk) > 1:
            rules.extend(rulesGen(Lk))

    scl = []
    for rule in rules:
        lhs, rhs = rule
        support = L[lhs | rhs]
        confidence = support / L[lhs]
        lift = confidence / L[rhs]
        scl.append(
            {
                "LHS": lhs,
                "RHS": rhs,
                "support": support,
                "confidence": confidence,
                "lift": lift,
            }
        )
    return pd.DataFrame(scl)


# ============================================================
# 二、2.4.1 关联规则：银行产品交叉销售
# ============================================================
data = pd.read_csv(DATA_PATH, encoding="gbk")
print(data.head())

baskets = data["PROD"].groupby(data["CSR_ID"]).apply(list)
print(baskets.head())

rules = arules(baskets, min_support=0.02)
print(rules.head())

conf = rules[(rules.confidence > 0.1) & (rules.lift > 1)]
print(conf.sort_values(by="support", ascending=False).head())

ax = plt.subplot()
conf.plot.scatter(
    x="lift", y="support", c="confidence", cmap="binary", figsize=(6, 4), ax=ax
)
ax.set_xlabel("提升度")
ax.set_ylabel("支持度")
if ax.collections and ax.collections[0].colorbar is not None:
    ax.collections[0].colorbar.set_label("置信度")
ax.set_title("关联规则的支持度-提升度分布")
save_show()


# ============================================================
# 三、2.4.1 关联规则：Orange3 FPGrowth 实现（选装）
# ============================================================
mapping = data["PROD"].drop_duplicates().to_dict()
inverse_mapping = {v: k for k, v in mapping.items()}

data_encoded = data.copy()
data_encoded["PROD"] = data_encoded["PROD"].replace(inverse_mapping)
baskets_ids = data_encoded["PROD"].groupby(data_encoded["CSR_ID"]).apply(list)
print(baskets_ids.head())

try:
    from orangecontrib.associate.fpgrowth import (
        association_rules,
        frequent_itemsets,
        rules_stats,
    )

    n_examples = len(data_encoded)
    itemsets = dict(frequent_itemsets(baskets_ids.tolist(), 0.02))
    orange_rules = association_rules(itemsets, min_confidence=0.1)
    stats = rules_stats(orange_rules, itemsets, n_examples)

    result = pd.DataFrame(
        stats,
        columns=[
            "atecedent",
            "consequent",
            "support",
            "confidence",
            "coverage",
            "strength",
            "lift",
            "leverage",
        ],
    )
    pid_to_product = lambda fsets: tuple(mapping[fset] for fset in fsets)
    result["atecedent"] = result["atecedent"].map(pid_to_product)
    result["consequent"] = result["consequent"].map(pid_to_product)
    print(result.head())
except ImportError:
    print("未安装 orange3，跳过 FPGrowth 示例（可 pip install orange3）。")


# ============================================================
# 四、自写 AprioriAll 算法（教材 myscript/aprioriAll.py）
# ============================================================
def createLs1(dataSet, min_support):
    """把序列数据打平为一维项集，用 Apriori 得到 1-频繁序列。"""
    n = len(dataSet)
    flattenSet = list(itertools.chain(*dataSet))
    flatten_n = len(flattenSet)

    min_support_new = min_support * n / flatten_n
    litemsets = apriori(flattenSet, min_support=min_support_new)

    mapping_ = {v: k for k, v in enumerate(litemsets)}
    supportLs1 = {
        (mapping_[k],): v * flatten_n / n for k, v in litemsets.items()
    }
    return mapping_, supportLs1


def seqMapping(seq, mapping_):
    newSeq = []
    for iSet in seq:
        newSet = [v for k, v in mapping_.items() if k <= set(iSet)]
        if newSet != []:
            newSeq.append(newSet)
    return newSeq


def transform(dataSet, mapping_):
    transformDS = []
    for seq in dataSet:
        newSeq = seqMapping(seq, mapping_)
        if newSeq != []:
            transformDS.append(newSeq)
    return transformDS


def seqGen(seqA, seqB):
    newA, newB = seqA.copy(), seqB.copy()
    if seqA[:-1] == seqB[:-1]:
        newA.append(seqB[-1])
        newB.append(seqA[-1])
        return [newA, newB]


def CsGen(Ls):
    Cs = []
    for seqA, seqB in itertools.combinations(Ls, 2):
        newSeqs = seqGen(seqA, seqB)
        if newSeqs is not None:
            Cs.extend(newSeqs)
    return [seq for seq in Cs if seq[1:] in Ls]


def isSubSeq(seq, cusSeq):
    nSeq, nCusSeq = len(seq), len(cusSeq)
    if nSeq > nCusSeq:
        return False
    if nSeq == 1:
        return any([seq[0] in i for i in cusSeq])
    if nSeq > 1:
        head = [seq[0] in i for i in cusSeq]
        if any(head):
            split = head.index(True)
            return isSubSeq(seq[1:], cusSeq[split + 1 :])
        return False


def calcSupport(transformDS, Cs, min_support):
    supportLsk = {}
    n = len(transformDS)
    if len(Cs) >= 1:
        for seq in Cs:
            support = (
                sum([isSubSeq(seq, cusSeq) for cusSeq in transformDS]) / n
            )
            if support >= min_support:
                supportLsk.update({tuple(seq): support})
    return [list(k) for k in supportLsk], supportLsk


def isSubSeq2(seq, cusSeq):
    nSeq, nCusSeq = len(seq), len(cusSeq)
    if nSeq > nCusSeq:
        return False
    if nSeq == 1:
        return any([seq[0].issubset(i) for i in cusSeq])
    if nSeq > 1:
        head = [seq[0].issubset(i) for i in cusSeq]
        if any(head):
            split = head.index(True)
            return isSubSeq2(seq[1:], cusSeq[split:])
        return False


def notProperSubSeq(seq, cusSeq):
    if seq == cusSeq:
        return True
    return not isSubSeq2(seq, cusSeq)


def maxLs(Ls, supportLs):
    LsCopy = Ls.copy()
    lenL, lenC = len(Ls), len(LsCopy)

    while lenC > 1 and lenL > 1:
        if LsCopy[lenC - 1] in Ls:
            mask = [notProperSubSeq(seq, LsCopy[lenC - 1]) for seq in Ls]
            Ls = list(itertools.compress(Ls, mask))
            lenL = len(Ls)
        lenC -= 1
    supportLs = {tuple(seq): supportLs[tuple(seq)] for seq in Ls}
    return Ls, supportLs


def aprioriAll(dataSet, min_support=0.25):
    mapping_, supportLs1 = createLs1(dataSet, min_support)

    Ls1 = [list(k) for k in supportLs1]
    transformDS = transform(dataSet, mapping_)

    LsList = [Ls1]
    supportLs = supportLs1.copy()
    k = 1
    while k >= 1 and len(LsList[-1]) > 1:
        Csk = CsGen(LsList[-1])
        Lsk, supportLsk = calcSupport(transformDS, Csk, min_support)
        if len(Lsk) > 0:
            LsList.append(Lsk)
            supportLs.update(supportLsk)
            k += 1
        else:
            break
    Ls = list(itertools.chain(*LsList))

    tr_mapping = {v: k for k, v in mapping_.items()}
    Ls = [[tr_mapping[k] for k in seq] for seq in Ls]
    supportLs = {
        tuple([tr_mapping[i] for i in k]): v for k, v in supportLs.items()
    }

    Ls, supportLs = maxLs(Ls, supportLs)
    return pd.DataFrame(list(supportLs.items()), columns=["sequence", "support"])


# ============================================================
# 五、2.4.2 序列模式：AprioriAll
# ============================================================
seq_data = pd.read_csv(DATA_PATH, encoding="gbk")
baskets_seq = seq_data.groupby(["OrderNumber", "CSR_ID"])["PROD"].apply(list)
print(baskets_seq.head())

baskets_seq = baskets_seq.groupby(level=1).apply(list)
print(baskets_seq.head())

patterns = aprioriAll(baskets_seq, min_support=0.1)
print(patterns.head())


# ============================================================
# 六、PrefixSpan 序列模式（教材 myscript/prefixspan.py）
# ============================================================
PLACE_HOLDER = "_"


class SquencePattern:
    def __init__(self, squence, support):
        self.squence = []
        for s in squence:
            self.squence.append(list(s))
        self.support = support

    def append(self, p):
        if p.squence[0][0] == PLACE_HOLDER:
            first_e = p.squence[0]
            first_e.remove(PLACE_HOLDER)
            self.squence[-1].extend(first_e)
            self.squence.extend(p.squence[1:])
        else:
            self.squence.extend(p.squence)
        self.support = min(self.support, p.support)


def prefixSpan(pattern, S, threshold):
    patterns_ = []
    f_list = frequent_items(S, pattern, threshold)
    for i in f_list:
        p = SquencePattern(pattern.squence, pattern.support)
        p.append(i)
        patterns_.append(p)
        p_S = build_projected_database(S, p)
        p_patterns = prefixSpan(p, p_S, threshold)
        patterns_.extend(p_patterns)
    return patterns_


def frequent_items(S, pattern, threshold):
    items = {}
    _items = {}
    f_list = []
    if S is None or len(S) == 0:
        return []

    last_e = pattern.squence[-1] if len(pattern.squence) != 0 else []
    for s in S:
        is_prefix = True
        for item in last_e:
            if item not in s[0]:
                is_prefix = False
                break
        if is_prefix and len(last_e) > 0:
            index = s[0].index(last_e[-1])
            if index < len(s[0]) - 1:
                for item in s[0][index + 1 :]:
                    if item in _items:
                        _items[item] += 1
                    else:
                        _items[item] = 1
        if PLACE_HOLDER in s[0]:
            for item in s[0][1:]:
                if item in _items:
                    _items[item] += 1
                else:
                    _items[item] = 1
            s = s[1:]
        counted = []
        for element in s:
            for item in element:
                if item not in counted:
                    counted.append(item)
                    if item in items:
                        items[item] += 1
                    else:
                        items[item] = 1
    f_list.extend(
        [
            SquencePattern([[PLACE_HOLDER, k]], v)
            for k, v in _items.items()
            if v >= threshold
        ]
    )
    f_list.extend(
        [SquencePattern([[k]], v) for k, v in items.items() if v >= threshold]
    )
    return sorted(f_list, key=lambda p: p.support)


def build_projected_database(S, pattern):
    p_S = []
    last_e = pattern.squence[-1]
    last_item = last_e[-1]
    for s in S:
        p_s = []
        for element in s:
            is_prefix = False
            if PLACE_HOLDER in element:
                if last_item in element and len(pattern.squence[-1]) > 1:
                    is_prefix = True
            else:
                is_prefix = True
                for item in last_e:
                    if item not in element:
                        is_prefix = False
                        break
            if is_prefix:
                e_index = s.index(element)
                i_index = element.index(last_item)
                if i_index == len(element) - 1:
                    p_s = s[e_index + 1 :]
                else:
                    p_s = s[e_index:]
                    e = element[i_index:]
                    e[0] = PLACE_HOLDER
                    p_s[0] = e
                break
        if len(p_s) != 0:
            p_S.append(p_s)
    return p_S


patterns_sp = prefixSpan(SquencePattern([], 1e4), baskets_seq, threshold=100)
for pattern in patterns_sp:
    print(pattern.squence, "-> support:", pattern.support)


# ============================================================
# 七、mlxtend 现成 Apriori 包（选装）
# ============================================================
try:
    from mlxtend.frequent_patterns import apriori as ml_apriori
    from mlxtend.frequent_patterns import association_rules as ml_association_rules
    from mlxtend.preprocessing import TransactionEncoder

    ml_data = pd.read_csv(DATA_PATH, encoding="gbk")
    grouped = ml_data.groupby("CSR_ID")["PROD"].apply(list)

    te = TransactionEncoder()
    te_ary = te.fit(grouped).transform(grouped)
    ml_df = pd.DataFrame(te_ary, columns=te.columns_)

    def encode_units(x):
        return 1 if x is True else 0

    ml_df = ml_df.applymap(encode_units)

    frequent_itemsets = ml_apriori(ml_df, min_support=0.01, use_colnames=True)
    print(frequent_itemsets)

    ml_rules = ml_association_rules(
        frequent_itemsets, metric="lift", min_threshold=1
    )
    print(ml_rules)

    ml_conf = ml_rules[(ml_rules.confidence > 0.1) & (ml_rules.lift > 1)]
    print(ml_conf.sort_values(by="support", ascending=False).head())

    ax = plt.subplot()
    ml_conf.plot.scatter(
        x="lift", y="support", c="confidence", cmap="binary",
        figsize=(6, 4), ax=ax,
    )
    ax.set_xlabel("提升度")
    ax.set_ylabel("支持度")
    if ax.collections and ax.collections[0].colorbar is not None:
        ax.collections[0].colorbar.set_label("置信度")
    ax.set_title("关联规则的支持度-提升度分布")
    save_show()

    # 课后答案版：查看与指定产品相关的规则
    ml_rules["antecedents"] = ml_rules["antecedents"].astype(str)
    ml_rules["consequents"] = ml_rules["consequents"].astype(str)

    caiwutong_rules = ml_rules[
        ml_rules["antecedents"].str.contains("财务通")
        | ml_rules["consequents"].str.contains("财务通")
    ]
    print("与'财务通'相关的规则:")
    print(caiwutong_rules)

    haoxiangdai_rules = ml_rules[
        ml_rules["consequents"].str.contains("好享贷")
    ]
    print("与'好享贷'相关的规则:")
    print(haoxiangdai_rules)
except ImportError:
    print("未安装 mlxtend，跳过现成 Apriori 包示例（可 pip install mlxtend）。")
