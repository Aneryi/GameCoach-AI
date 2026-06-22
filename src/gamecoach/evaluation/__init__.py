"""评估模块 — Agent 执行质量指标计算与 LangSmith 上报。

evaluate_run() 从 GameCoachState 提取关键指标，
计算三维评分（execution / data_quality / output_quality），
并通过 LangSmith SDK 上报自定义 feedback。
"""

from gamecoach.evaluation.metrics import evaluate_run

__all__ = ["evaluate_run"]
