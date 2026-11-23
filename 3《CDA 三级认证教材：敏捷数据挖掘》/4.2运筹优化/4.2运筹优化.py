# -*- coding: utf-8 -*-
"""
4.2 运筹优化

项目背景
--------
教材用 CPLEX 求解三类经典运筹优化问题，它们对应营销资源分配、广告投放等业务决策：

    4.2.1 线性规划  productivity.lp
          最大化 50*x1 + 100*x2
          s.t. x1 + x2 <= 300
               2*x1 + x2 <= 400
               x2 <= 250, x1,x2 >= 0

    4.2.2 整数规划  campaign.lp
          9 个候选项目、3 个渠道，每个项目最多投一个渠道，每个渠道最多投 3 个项目，
          最大化总收益。0-1 整数规划。

    4.2.3 二次规划  qp1.lp
          最小化 [2a^2 + b^2 + a*b]/2 + a + b
          s.t. a + b = 1, a,b >= 0

实现方式
--------
优先使用 CPLEX（与教材一致）；未安装 CPLEX 时自动用 scipy 求解相同的数学模型，
保证在任何环境都能跑通并得到最优解。scipy 版本要求 >= 1.9（提供 milp）。
"""

from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# 4.2.1 线性规划
# ============================================================
def solve_linear():
    print("=" * 60)
    print("4.2.1 线性规划 productivity.lp")
    try:
        import cplex

        op = cplex.Cplex()
        op.read(str(BASE_DIR / "productivity.lp"))
        op.solve()
        print("Max objective value is %s" % op.solution.get_objective_value())
        print("Best Productivity is %s" % op.solution.get_values())
        return op.solution.get_objective_value(), op.solution.get_values()
    except ImportError:
        print("未安装 CPLEX，使用 scipy.optimize.linprog 求解等价模型。")
        from scipy.optimize import linprog

        # 最大化 -> 最小化负目标
        c = [-50.0, -100.0]
        A_ub = [[1.0, 1.0], [2.0, 1.0], [0.0, 1.0]]
        b_ub = [300.0, 400.0, 250.0]
        bounds = [(0.0, None), (0.0, None)]
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        print("status:", res.message)
        print("Max objective value is %s" % (-res.fun))
        print("Best Productivity is %s" % res.x.tolist())
        return -res.fun, res.x.tolist()


# ============================================================
# 4.2.2 整数规划
# ============================================================
PROFIT = np.array(
    [
        [100, 120, 90],
        [50, 70, 75],
        [60, 75, 65],
        [55, 80, 75],
        [75, 60, 50],
        [75, 65, 60],
        [80, 70, 75],
        [65, 60, 60],
        [80, 110, 75],
    ],
    dtype=float,
)


def solve_integer():
    print("=" * 60)
    print("4.2.2 整数规划 campaign.lp")
    try:
        import cplex

        op = cplex.Cplex()
        op.read(str(BASE_DIR / "campaign.lp"))
        op.solve()
        print("Max objective value is %s" % op.solution.get_objective_value())
        print("Best allocation :")
        print(np.array(op.solution.get_values()).reshape([3, -1]).T)
        return op.solution.get_objective_value()
    except ImportError:
        print("未安装 CPLEX，使用穷举法求解 3^9 个 0-1 分配方案（等价最优）。")
        n_items, n_channels = PROFIT.shape
        best_profit = -np.inf
        best_assign = None
        for code in range(n_channels ** n_items):
            assign = []
            value = code
            for _ in range(n_items):
                assign.append(value % n_channels)
                value //= n_channels
            assign = np.array(assign)
            counts = np.bincount(assign, minlength=n_channels)
            if np.any(counts > 3):
                continue
            profit = PROFIT[np.arange(n_items), assign].sum()
            if profit > best_profit:
                best_profit = profit
                best_assign = assign
        print("Max objective value is %s" % best_profit)
        print("每个项目选择的渠道编号（0/1/2）:", best_assign.tolist())
        return best_profit


# ============================================================
# 4.2.3 二次规划
# ============================================================
def solve_quadratic():
    print("=" * 60)
    print("4.2.3 二次规划 qp1.lp")
    try:
        import cplex

        op = cplex.Cplex()
        op.read(str(BASE_DIR / "qp1.lp"))
        op.solve()
        print("objective value:", op.solution.get_objective_value())
        print("values:", op.solution.get_values())
        return op.solution.get_objective_value()
    except ImportError:
        print("未安装 CPLEX，使用 scipy.optimize.minimize(SLSQP) 求解。")
        from scipy.optimize import minimize

        def objective(x):
            a, b = x
            return 0.5 * (2 * a ** 2 + b ** 2 + a * b) + a + b

        cons = ({"type": "eq", "fun": lambda x: x[0] + x[1] - 1.0},)
        res = minimize(
            objective,
            x0=np.array([0.5, 0.5]),
            method="SLSQP",
            bounds=[(0.0, None), (0.0, None)],
            constraints=cons,
        )
        print("success:", res.success)
        print("objective value:", res.fun)
        print("values:", res.x.tolist())
        return res.fun


if __name__ == "__main__":
    solve_linear()
    solve_integer()
    solve_quadratic()