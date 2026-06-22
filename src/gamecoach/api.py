"""
GameCoach AI FastAPI Web 服务。

提供 REST API 接口，支持通过 HTTP 请求调用教练服务。
启动方式：uvicorn gamecoach.api:app --host 127.0.0.1 --port 8000
Swagger UI：http://127.0.0.1:8000/docs

API 端点：
- GET  /       — 健康检查
- GET  /demo   — 获取预设演示场景列表
- POST /coach  — 核心教练接口（接收用户问题，返回完整建议）
"""

from __future__ import annotations

import logging
from typing import Optional

# FastAPI: Python 异步 Web 框架，自动生成 OpenAPI/Swagger 文档
from fastapi import FastAPI

# BaseModel: Pydantic 数据模型，用于请求/响应的序列化和校验
# Field: 字段描述，会显示在 Swagger UI 中
from pydantic import BaseModel, Field

from gamecoach.graph.workflow import graph

logger = logging.getLogger(__name__)

# FastAPI 应用实例
app = FastAPI(
    title="GameCoach AI",
    description="基于 LangGraph 的游戏成长教练 Agent API",
    version="0.2.0",
)


# ═══════════════════════════════════════════════════════════════
# 请求/响应 Pydantic 模型
# ═══════════════════════════════════════════════════════════════

class CoachRequest(BaseModel):
    """
    教练请求体。

    player_id 有默认值 "player_001"，Swagger UI 中可以直接测试而无需填写。
    """
    user_message: str = Field(
        description="玩家自然语言输入",
        examples=["我最近胜率很低，怎么上分？"],
    )
    player_id: str = Field(default="player_001", description="玩家 ID")


class MetricsSummary(BaseModel):
    """评估指标摘要（嵌套在 CoachResponse 中返回）。"""
    planner_task_count: int = 0
    routing_path: str = ""
    degraded_node_count: int = 0
    rag_hit_count: int = 0
    response_length: int = 0
    has_match_analysis: bool = False
    has_character_recommendations: bool = False
    has_build_recommendations: bool = False
    has_training_plan: bool = False


class CoachResponse(BaseModel):
    """
    教练响应体。

    包含完整的执行结果：意图分类、任务拆解、路由路径、
    最终回复、评估指标、错误和降级信息。
    """
    status: str = Field(description="ok / error")
    intent: Optional[str] = Field(default=None, description="识别到的用户意图")
    planned_tasks: list[dict] = Field(default_factory=list, description="任务拆解列表")
    routing_path: str = Field(default="", description="节点执行路径")
    final_response: str = Field(default="", description="最终教练建议")
    metrics: MetricsSummary = Field(default_factory=MetricsSummary)
    errors: list[str] = Field(default_factory=list)
    degraded_nodes: list[str] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
# API 路由
# ═══════════════════════════════════════════════════════════════

@app.get("/")
def root():
    """健康检查端点。"""
    return {"service": "GameCoach AI", "version": "0.2.0", "status": "healthy"}


@app.post("/coach", response_model=CoachResponse)
def coach(request: CoachRequest):
    """
    核心教练接口。

    接收玩家自然语言问题，运行完整的 LangGraph 多 Agent 工作流，
    返回包含诊断、数据依据、角色推荐、出装方案和训练计划的完整建议。

    示例请求：
        POST /coach
        {"user_message": "我最近胜率很低，怎么上分？", "player_id": "player_001"}
    """
    graph_input = {
        "user_message": request.user_message,
        "player_id": request.player_id,
    }
    try:
        result = graph.invoke(graph_input)
    except Exception:
        logger.exception("Graph 执行失败")
        return CoachResponse(status="error", errors=["系统执行异常，请稍后重试。"])

    m = result.get("metrics", {})
    return CoachResponse(
        status="ok",
        intent=result.get("intent"),
        planned_tasks=result.get("planned_tasks", []),
        routing_path=m.get("routing_path", ""),
        final_response=result.get("final_response", ""),
        metrics=MetricsSummary(
            planner_task_count=m.get("planner_task_count", 0),
            routing_path=m.get("routing_path", ""),
            degraded_node_count=m.get("degraded_node_count", 0),
            rag_hit_count=m.get("rag_hit_count", 0),
            response_length=m.get("response_length", 0),
            has_match_analysis=m.get("has_match_analysis", False),
            has_character_recommendations=m.get("has_character_recommendations", False),
            has_build_recommendations=m.get("has_build_recommendations", False),
            has_training_plan=m.get("has_training_plan", False),
        ),
        errors=result.get("errors", []),
        degraded_nodes=result.get("degraded_nodes", []),
    )


@app.get("/demo")
def demo_scenarios():
    """返回预设演示场景列表。"""
    return {
        "scenarios": [
            {"id": 0, "name": "上分求助", "message": "我最近胜率很低，怎么上分？"},
            {"id": 1, "name": "角色推荐", "message": "当前版本有什么强势角色？"},
            {"id": 2, "name": "出装建议", "message": "Alpha 出什么装备比较好？"},
            {"id": 3, "name": "训练计划", "message": "帮我制定一个 7 天上分训练计划。"},
        ]
    }


# uvicorn 启动入口：python -m gamecoach.api 或 uvicorn gamecoach.api:app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
