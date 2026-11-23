# -*- coding: utf-8 -*-
"""
4.3 流程挖掘（pm4py）

项目背景
--------
流程挖掘从信息系统产生的“事件日志”中还原真实业务流程，典型任务包括：
    1. 读取 IEEE XES 标准日志
    2. 查看日志 / 轨迹 / 事件的结构
    3. 导出 XES
    4. 发现直接跟随图（DFG）：频次视图与性能视图

数据文件
--------
    running-example.xes     教材示例日志
    exportedLog.xes         导出结果

依赖
----
    pm4py（pip install pm4py）
    未安装 pm4py 时脚本会打印安装提示并跳过。
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def run_process_mining():
    try:
        from pm4py.objects.log.importer.xes import importer as xes_importer
    except ImportError:
        print("未安装 pm4py，跳过 4.3（可 pip install pm4py）。")
        return

    # 1. 读取 XES 日志
    log = xes_importer.apply(str(BASE_DIR / "running-example.xes"))
    print(log)

    print("The first trace in the log: \n", log[0])
    print("-" * 80)
    print("The first event of the first trace: \n", log[0][0])

    print(log[0:2:2][:3])

    print(type(log))
    print(type(log[0]))
    print(type(log[0][0]))

    print("Length of the log is %d" % len(log))
    print("Length of the first trace is %d" % len(log[0]))
    print("Length of the first event is %d" % len(log[0][0]))

    # 2. 导出 XES
    from pm4py.objects.log.exporter.xes import exporter as xes_exporter

    xes_exporter.apply(log, str(BASE_DIR / "exportedLog.xes"))

    # 3. 直接跟随图：频次视图
    from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
    from pm4py.visualization.dfg import visualizer as dfg_visualization

    fig_dir = BASE_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)

    dfg = dfg_discovery.apply(log)
    gviz = dfg_visualization.apply(
        dfg, log=log, variant=dfg_visualization.Variants.FREQUENCY
    )
    try:
        dfg_visualization.save(gviz, str(fig_dir / "4.3流程挖掘_频次DFG.png"))
    except Exception as exc:
        print("频次 DFG 图保存失败：", exc)

    # 4. 直接跟随图：性能（时间）视图
    dfg = dfg_discovery.apply(log, variant=dfg_discovery.Variants.PERFORMANCE)
    gviz = dfg_visualization.apply(
        dfg, log=log, variant=dfg_visualization.Variants.PERFORMANCE
    )
    try:
        dfg_visualization.save(gviz, str(fig_dir / "4.3流程挖掘_性能DFG.png"))
    except Exception as exc:
        print("性能 DFG 图保存失败：", exc)


if __name__ == "__main__":
    run_process_mining()