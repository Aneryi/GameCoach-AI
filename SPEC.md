# GameCoach AI: 基于 LangGraph 的游戏成长教练 Agent

## 1. 项目概述

GameCoach AI 是一个面向游戏玩家的智能成长教练 Agent。系统通过 LangGraph 构建多 Agent 工作流，结合 Planning、Tool Calling、RAG、Memory 与 LangSmith 监控能力，帮助玩家分析近期战绩、定位能力短板、推荐英雄与出装，并生成可执行的阶段性训练计划。

该项目目标是对标游戏公司中与 AI Agent、游戏数据分析、玩家成长系统、智能推荐、内容检索增强生成相关的岗位要求，突出工程化 Agent 编排能力与可评估的业务闭环。

## 2. 项目目标

### 2.1 用户目标

用户可以通过自然语言提出游戏成长相关问题，例如：

```text
我最近胜率很低，怎么上分？
我适合练什么英雄？
帮我分析最近 20 场的问题。
我玩射手老是团战暴毙，怎么改？
当前版本有哪些强势英雄适合我？
给我制定一个 7 天上分训练计划。
```

系统需要完成：

- 分析玩家近期战绩表现。
- 识别玩家英雄池、打法偏好与能力短板。
- 推荐适合当前版本与玩家风格的英雄。
- 推荐装备、铭文、符文、天赋或构筑方案。
- 检索并总结攻略、版本指南、职业选手打法。
- 生成可执行的短期成长计划。
- 记录玩家长期偏好、目标和问题。
- 评估建议效果并持续优化。

### 2.2 技术目标

- 使用 Python 3.11 开发。
- 使用 LangGraph 实现多 Agent 状态机编排。
- 使用 Tool Calling 接入战绩、英雄数据库、版本信息和攻略检索工具。
- 使用 RAG 处理攻略、版本指南、职业打法等非结构化知识。
- 使用 Player Memory 存储玩家画像与成长历史。
- 使用 LangSmith 追踪 Agent 执行链路、工具调用与效果指标。
- 提供清晰的模块边界，方便扩展到不同游戏。

## 3. 典型使用流程

### 3.1 用户输入

```text
我最近胜率很低，怎么上分？
```

### 3.2 Planner Agent 拆解

Planner Agent 将用户问题拆成多个子任务：

```json
[
  {
    "task": "match_analysis",
    "description": "分析玩家最近 20 场战绩，找出胜率、KDA、死亡次数、参团率等问题"
  },
  {
    "task": "hero_pool_analysis",
    "description": "分析玩家常用英雄、胜率较高英雄和低效英雄"
  },
  {
    "task": "version_recommendation",
    "description": "结合当前版本强势英雄，推荐适合玩家上分的英雄"
  },
  {
    "task": "training_plan",
    "description": "根据短板生成 7 天成长计划"
  }
]
```

### 3.3 Task Router 分发

Task Router 根据任务类型调用不同 Agent：

- `MatchAnalysis Agent`: 分析战绩数据。
- `Strategy Agent`: 生成打法与训练建议。
- `Build Agent`: 推荐装备、铭文、天赋或构筑方案。
- `RAG Agent`: 检索攻略、版本指南和职业选手内容。
- `Memory Agent`: 读取并更新玩家长期画像。

### 3.4 Response Agent 汇总

最终输出结构化建议：

```text
你的主要问题不是英雄选择，而是中后期团战死亡率过高。

最近 20 场：
- 胜率：40%
- 平均 KDA：2.1
- 平均死亡：6.8
- 参团率：48%

建议：
1. 暂时主练后羿和狄仁杰，降低操作复杂度。
2. 每局 10 分钟前优先保证发育，不主动接无视野团。
3. 团战站位保持在辅助或前排后方 600-800 距离。
4. 未来 7 天重点训练补刀、视野判断和团战进场时机。
```

## 4. 核心功能范围

### 4.1 MVP 功能

MVP 阶段优先完成以下能力：

- 单轮对话式问题理解。
- Planner Agent 任务拆解。
- LangGraph 多节点工作流。
- Mock 战绩查询 Tool。
- Mock 英雄数据库 Tool。
- Mock 版本信息 Tool。
- 本地攻略文档 RAG。
- 玩家 Memory JSON 存取。
- 成长计划生成。
- LangSmith Trace 接入。
- 基础 Agent 评估脚本。

### 4.2 进阶功能

- 多轮对话上下文理解。
- 支持多个游戏配置，例如 MOBA、FPS、卡牌游戏。
- 真实 API 接入游戏战绩平台。
- 个性化英雄推荐排序模型。
- 基于历史执行结果的建议效果评估。
- 自动复盘报告生成。
- Web UI 或 Streamlit Demo。
- 用户任务打卡与训练计划进度追踪。

## 5. 系统架构

### 5.1 总体架构

```text
User
  |
  v
Input Normalizer
  |
  v
Planner Agent
  |
  v
Task Router
  |
  +--> MatchAnalysis Agent
  +--> Strategy Agent
  +--> Build Agent
  +--> RAG Agent
  +--> Memory Agent
  |
  v
Response Agent
  |
  v
Evaluation Logger
  |
  v
LangSmith / Metrics Store
```

### 5.2 LangGraph 节点设计

| 节点 | 职责 | 输入 | 输出 |
| --- | --- | --- | --- |
| `input_normalizer` | 标准化用户输入、识别游戏类型、抽取玩家 ID | 原始用户消息 | 规范化请求 |
| `planner` | 拆解用户意图，生成任务列表 | 规范化请求、Memory | 子任务计划 |
| `router` | 根据任务类型选择 Agent | 子任务计划 | 路由结果 |
| `match_analysis_agent` | 调用战绩工具并分析表现 | 玩家 ID、分析范围 | 战绩分析结果 |
| `strategy_agent` | 生成打法、训练、上分策略 | 战绩分析、Memory、版本信息 | 策略建议 |
| `build_agent` | 推荐英雄构筑、装备和技能加点 | 英雄、版本、玩家风格 | 构筑推荐 |
| `rag_agent` | 检索攻略、版本指南、职业选手打法 | 查询关键词 | 引用资料与摘要 |
| `memory_agent` | 读取、写入玩家偏好和长期目标 | 用户输入、分析结果 | 更新后的玩家画像 |
| `response_agent` | 汇总多 Agent 输出，生成最终答案 | 所有中间结果 | 面向用户的结构化回复 |
| `evaluation_logger` | 记录任务执行与评估数据 | Trace、工具结果、最终回答 | 指标日志 |

### 5.3 LangGraph 状态定义

建议定义统一状态对象 `GameCoachState`：

```python
from typing import Any, Literal, TypedDict


class PlayerMemory(TypedDict, total=False):
    player_id: str
    favorite_heroes: list[str]
    main_roles: list[str]
    weaknesses: list[str]
    goals: list[str]
    preferred_playstyle: str
    rank: str
    updated_at: str


class PlannedTask(TypedDict):
    task_id: str
    task_type: Literal[
        "match_analysis",
        "hero_recommendation",
        "build_recommendation",
        "strategy_generation",
        "rag_lookup",
        "memory_update",
        "training_plan",
    ]
    description: str
    priority: int
    required_tools: list[str]


class GameCoachState(TypedDict, total=False):
    user_message: str
    player_id: str
    game: str
    intent: str
    memory: PlayerMemory
    planned_tasks: list[PlannedTask]
    match_data: dict[str, Any]
    match_analysis: dict[str, Any]
    hero_recommendations: list[dict[str, Any]]
    build_recommendations: list[dict[str, Any]]
    rag_context: list[dict[str, Any]]
    strategy: dict[str, Any]
    training_plan: dict[str, Any]
    final_response: str
    errors: list[str]
    metrics: dict[str, Any]
```

## 6. Agent 详细设计

### 6.1 Planner Agent

职责：

- 理解用户自然语言意图。
- 判断用户是否需要战绩分析、英雄推荐、出装推荐、攻略检索或成长计划。
- 生成结构化任务列表。
- 为每个任务标注优先级和依赖工具。

输入示例：

```text
我最近胜率很低，怎么上分？
```

输出示例：

```json
{
  "intent": "improve_rank",
  "planned_tasks": [
    {
      "task_id": "t1",
      "task_type": "match_analysis",
      "description": "分析最近 20 场胜率、KDA、死亡、参团等数据",
      "priority": 1,
      "required_tools": ["match_history_tool"]
    },
    {
      "task_id": "t2",
      "task_type": "hero_recommendation",
      "description": "根据英雄池和当前版本推荐上分英雄",
      "priority": 2,
      "required_tools": ["hero_database_tool", "patch_meta_tool"]
    },
    {
      "task_id": "t3",
      "task_type": "training_plan",
      "description": "生成 7 天成长计划",
      "priority": 3,
      "required_tools": []
    }
  ]
}
```

### 6.2 MatchAnalysis Agent

职责：

- 调用战绩查询 Tool。
- 计算最近 N 场胜率、KDA、平均击杀、平均死亡、平均助攻。
- 分析英雄维度表现。
- 识别常见问题，例如死亡过多、输出低、参团低、经济低、视野差。

核心分析指标：

- `win_rate`: 最近 N 场胜率。
- `avg_kda`: 平均 KDA。
- `avg_kills`: 平均击杀。
- `avg_deaths`: 平均死亡。
- `avg_assists`: 平均助攻。
- `hero_win_rates`: 各英雄胜率。
- `role_performance`: 各位置表现。
- `early_game_score`: 前期表现评分。
- `teamfight_score`: 团战表现评分。
- `objective_score`: 资源控制评分。

输出示例：

```json
{
  "summary": "最近 20 场胜率偏低，主要问题是死亡次数过高和参团率不足。",
  "metrics": {
    "matches": 20,
    "win_rate": 0.4,
    "avg_kda": 2.1,
    "avg_deaths": 6.8,
    "teamfight_participation": 0.48
  },
  "weaknesses": ["团战站位靠前", "中期无视野带线", "逆风局死亡过多"],
  "strengths": ["对线补刀稳定", "使用射手英雄熟练度较高"]
}
```

### 6.3 Strategy Agent

职责：

- 根据 MatchAnalysis、Memory 和 RAG 资料生成策略建议。
- 将抽象问题转化为可执行动作。
- 针对不同阶段给出建议：对线期、中期转线、团战、逆风局、顺风局。

输出内容：

- 核心问题诊断。
- 优先级排序。
- 对局阶段建议。
- 训练动作。
- 避免事项。

### 6.4 Build Agent

职责：

- 根据英雄、版本和玩家习惯推荐装备。
- 支持顺风、均势、逆风三类构筑。
- 给出选择原因，不只输出装备列表。

输出示例：

```json
{
  "hero": "狄仁杰",
  "scenario": "均势局",
  "items": ["急速战靴", "末世", "无尽战刃", "破晓", "逐日之弓", "魔女斗篷"],
  "rationale": "该构筑兼顾持续输出和容错，适合团战容易被切入的玩家。",
  "alternatives": [
    {
      "condition": "敌方法刺爆发高",
      "replace": "逐日之弓",
      "with": "不祥征兆"
    }
  ]
}
```

### 6.5 RAG Agent

职责：

- 从攻略库、版本指南、职业选手复盘、英雄教学中检索相关内容。
- 对检索结果进行摘要和引用。
- 将非结构化攻略转化为行动建议。

知识库来源：

- 官方版本公告。
- 英雄改动说明。
- 高分段玩家攻略。
- 职业比赛复盘。
- 装备与符文解析。

RAG 检索流程：

```text
用户问题 / Planner 子任务
  -> Query Rewrite
  -> Embedding Search
  -> Rerank
  -> Context Compression
  -> Answer Grounding
```

### 6.6 Memory Agent

职责：

- 读取玩家画像。
- 从对话和分析结果中提取长期偏好。
- 更新玩家目标、英雄池和能力短板。
- 为 Planner 与 Strategy Agent 提供个性化上下文。

Memory Schema：

```json
{
  "player_id": "player_001",
  "favorite_heroes": ["后羿", "狄仁杰"],
  "main_roles": ["射手"],
  "weaknesses": ["团战意识差", "中期容易掉点"],
  "goals": ["上钻石"],
  "preferred_playstyle": "稳健发育型",
  "rank": "铂金 I",
  "training_history": [
    {
      "date": "2026-06-02",
      "task": "团战站位训练",
      "completed": false
    }
  ],
  "updated_at": "2026-06-02T20:00:00+08:00"
}
```

### 6.7 Response Agent

职责：

- 汇总各 Agent 输出。
- 控制最终回答风格，避免信息堆砌。
- 输出玩家能直接执行的建议。
- 在必要时提示数据不足或建议补充玩家 ID、段位、主玩位置。

推荐输出结构：

```text
结论：
你的主要上分阻碍是中期死亡过多，而不是英雄池太浅。

数据依据：
- 最近 20 场胜率：40%
- 平均 KDA：2.1
- 平均死亡：6.8

优先改进：
1. 10 分钟后减少无视野单带。
2. 团战等待敌方刺客露头后再输出。
3. 暂时减少高风险英雄，主练 2 个稳定上分英雄。

推荐英雄：
- 狄仁杰：容错高，适合稳定发育。
- 后羿：团战输出强，但需要注意站位。

7 天计划：
Day 1: 补刀与经济训练。
Day 2: 小地图观察训练。
Day 3: 团战站位复盘。
Day 4: 逆风局少死训练。
Day 5: 英雄池固定训练。
Day 6: 版本强势英雄适应。
Day 7: 综合复盘。
```

## 7. Tool Calling 设计

### 7.1 战绩查询 Tool

工具名：`match_history_tool`

职责：

- 查询玩家最近 N 场对局。
- 返回胜负、英雄、位置、KDA、经济、伤害、参团率等数据。

输入：

```json
{
  "player_id": "player_001",
  "limit": 20,
  "game": "moba"
}
```

输出：

```json
{
  "matches": [
    {
      "match_id": "m_001",
      "result": "loss",
      "hero": "后羿",
      "role": "射手",
      "kills": 4,
      "deaths": 8,
      "assists": 6,
      "gold": 9800,
      "damage": 62000,
      "teamfight_participation": 0.46,
      "duration_minutes": 18
    }
  ]
}
```

### 7.2 英雄数据库 Tool

工具名：`hero_database_tool`

职责：

- 查询英雄基础属性。
- 查询英雄定位、难度、克制关系。
- 查询适合搭配和被克制英雄。

输入：

```json
{
  "hero": "狄仁杰",
  "game": "moba"
}
```

输出：

```json
{
  "hero": "狄仁杰",
  "role": "射手",
  "difficulty": "medium",
  "strengths": ["持续输出", "解控", "对线稳定"],
  "weaknesses": ["缺少位移", "怕强突进"],
  "counters": ["兰陵王", "镜"],
  "countered_by": ["孙尚香", "公孙离"]
}
```

### 7.3 版本信息 Tool

工具名：`patch_meta_tool`

职责：

- 返回当前版本强势英雄。
- 返回英雄增强、削弱和装备改动。
- 为英雄推荐和出装推荐提供版本依据。

输入：

```json
{
  "game": "moba",
  "patch": "latest",
  "role": "射手"
}
```

输出：

```json
{
  "patch": "2026.06",
  "strong_heroes": ["狄仁杰", "孙尚香", "戈娅"],
  "nerfed_heroes": ["后羿"],
  "buffed_items": ["逐日之弓"],
  "meta_summary": "当前版本射手更重视自保和中期参团能力。"
}
```

### 7.4 攻略 RAG Tool

工具名：`guide_rag_tool`

职责：

- 根据用户问题检索攻略知识库。
- 返回相关片段、来源和摘要。

输入：

```json
{
  "query": "射手团战站位技巧",
  "top_k": 5
}
```

输出：

```json
{
  "documents": [
    {
      "source": "marksman_teamfight_guide.md",
      "title": "射手团战站位指南",
      "score": 0.87,
      "snippet": "射手应保持在前排后方，等待敌方关键控制和突进技能交出后再进场输出。"
    }
  ]
}
```

## 8. 数据与目录规划

建议项目结构：

```text
gamecoach-ai/
  README.md
  SPEC.md
  pyproject.toml
  .env.example
  src/
    gamecoach/
      __init__.py
      main.py
      graph/
        state.py
        workflow.py
        nodes.py
        router.py
      agents/
        planner.py
        match_analysis.py
        strategy.py
        build.py
        rag.py
        memory.py
        response.py
      tools/
        match_history.py
        hero_database.py
        patch_meta.py
        guide_rag.py
      memory/
        store.py
        schemas.py
      rag/
        loader.py
        index.py
        retriever.py
      evaluation/
        metrics.py
        evaluator.py
      config/
        settings.py
  data/
    mock/
      matches.json
      heroes.json
      patch_meta.json
    guides/
      marksman_teamfight_guide.md
      patch_guide.md
    memory/
      player_001.json
  tests/
    test_planner.py
    test_match_analysis.py
    test_memory.py
    test_workflow.py
```

## 9. 成长计划生成规格

### 9.1 输入

- 玩家目标：例如上钻石、提高胜率、练会打野。
- 当前弱点：例如团战意识差、补刀差、英雄池浅。
- 玩家可投入时间：例如每天 30 分钟。
- 当前版本推荐方向。

### 9.2 输出字段

```json
{
  "duration_days": 7,
  "goal": "提升射手位胜率",
  "daily_tasks": [
    {
      "day": 1,
      "theme": "补刀与经济",
      "tasks": ["训练营补刀 10 分钟", "排位中 10 分钟经济不低于队伍第二"],
      "success_criteria": ["10 分钟经济 >= 4500", "死亡 <= 2"]
    }
  ],
  "review_checkpoints": ["第 3 天复盘 KDA", "第 7 天对比胜率变化"]
}
```

### 9.3 计划模板示例

```text
Day 1: 补刀与经济
- 训练营补刀 10 分钟。
- 实战目标：10 分钟经济不低于队伍第二。

Day 2: 小地图观察
- 每 5 秒看一次小地图。
- 敌方打野消失时停止压线。

Day 3: 团战站位
- 团战前确认敌方刺客位置。
- 不先手进草，不越过前排输出。

Day 4: 逆风局少死
- 逆风只吃安全线。
- 没有视野不单带过河道。

Day 5: 英雄池固定
- 只使用 2 个主练英雄。
- 复盘每个英雄的死亡原因。

Day 6: 版本适应
- 尝试 1 个当前版本强势英雄。
- 对比原主力英雄的输出和生存表现。

Day 7: 综合复盘
- 对比本周胜率、KDA、死亡次数。
- 更新下一阶段训练重点。
```

## 10. Agent 评估体系

### 10.1 业务效果指标

- 胜率变化：训练前后最近 20 场胜率差异。
- KDA 变化：平均 KDA 是否提升。
- 段位变化：是否达到阶段目标。
- 死亡次数变化：平均死亡是否下降。
- 任务完成率：用户训练计划打卡完成比例。
- 英雄熟练度变化：推荐英雄使用场次与胜率。

### 10.2 Agent 质量指标

- 任务拆解正确率：Planner 是否正确识别用户需求。
- Tool 调用成功率：工具调用是否正确且返回有效数据。
- RAG 命中率：检索内容是否与问题相关。
- 建议可执行性：建议是否具体、可操作、可验证。
- 个性化程度：是否利用 Memory 生成个性化建议。
- 平均响应时间：端到端执行耗时。

### 10.3 LangSmith 监控指标

- `planner_task_accuracy`
- `tool_call_success_rate`
- `rag_hit_rate`
- `memory_update_success_rate`
- `avg_response_latency`
- `final_answer_quality_score`
- `training_plan_completion_rate`

## 11. LangSmith 接入规格

需要追踪：

- 每次用户请求的输入与最终输出。
- Planner Agent 生成的任务列表。
- Router 决策路径。
- 每个 Tool 的输入、输出、耗时与异常。
- RAG 检索 query、top_k 文档和命中分数。
- Memory 读取与更新内容。
- Response Agent 最终汇总结果。
- 用户反馈与后续战绩变化。

建议环境变量：

```text
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=gamecoach-ai
OPENAI_API_KEY=your_openai_key
```

## 12. 非功能需求

### 12.1 可扩展性

- 工具层需要与 Agent 解耦，方便替换 Mock 数据为真实 API。
- 游戏配置需要抽象，避免所有逻辑绑定到单一游戏。
- RAG 文档加载、切分、索引和检索需要模块化。

### 12.2 可测试性

- Planner 输出必须是结构化 JSON，方便断言。
- 每个 Tool 需要有独立单元测试。
- MatchAnalysis Agent 的统计逻辑需要使用固定 Mock 数据测试。
- Workflow 需要测试常见用户意图路径。

### 12.3 稳定性

- Tool 调用失败时需要返回降级建议。
- 用户缺少 player_id 时，系统应主动询问或使用 Demo 数据。
- RAG 无命中时，不应编造引用，应提示资料不足。
- Memory 写入失败不应影响主回答生成。

### 12.4 安全与隐私

- 玩家 ID、历史战绩和成长记录属于用户数据。
- 本地开发阶段可使用 JSON 存储，生产环境应切换到数据库。
- 日志中避免记录敏感账号凭证。
- `.env` 不得提交到仓库。

## 13. 推荐技术栈

- Python: `3.11`
- Agent 编排: `langgraph`
- LLM 接入: `langchain-openai`
- Trace 与评估: `langsmith`
- 数据校验: `pydantic`
- RAG: `langchain`, `faiss-cpu` 或 `chromadb`
- API 服务: `fastapi`
- CLI Demo: `typer` 或 `argparse`
- 测试: `pytest`
- 配置管理: `python-dotenv`, `pydantic-settings`
- 代码质量: `ruff`, `mypy`

## 14. 里程碑计划

### Milestone 1: 项目骨架

- 初始化 Python 3.11 项目。
- 配置 `pyproject.toml`。
- 建立 `src/gamecoach` 目录。
- 定义 `GameCoachState`。
- 准备 Mock 数据。

### Milestone 2: LangGraph MVP

- 实现 Planner Agent。
- 实现 Task Router。
- 实现 MatchAnalysis Agent。
- 实现 Response Agent。
- 跑通一次完整工作流。

### Milestone 3: Tool Calling

- 实现战绩查询 Tool。
- 实现英雄数据库 Tool。
- 实现版本信息 Tool。
- 增加 Tool 错误处理。
- 编写工具单元测试。

### Milestone 4: Memory 与 RAG

- 实现 Player Memory JSON 存储。
- 接入攻略文档加载与向量检索。
- 实现 RAG Agent。
- 将 Memory 和 RAG 结果接入 Strategy Agent。

### Milestone 5: 评估与 LangSmith

- 接入 LangSmith Trace。
- 增加 Agent 执行指标。
- 实现基础评估脚本。
- 输出一次 Demo 评估报告。

### Milestone 6: Demo 展示

- 提供 CLI 或 Web Demo。
- 准备 3-5 个典型用户问题。
- 输出简历项目说明与技术亮点。

## 15. 简历描述建议

项目名：

```text
GameCoach AI: 基于 LangGraph 的游戏成长教练 Agent
```

项目描述：

```text
基于 LangGraph 构建多 Agent 游戏成长教练系统，面向玩家战绩分析、英雄推荐、出装建议、攻略检索和训练计划生成等场景。系统通过 Planner Agent 拆解用户意图，由 Task Router 调度 MatchAnalysis、Strategy、Build、RAG 与 Memory 等专用 Agent，并结合 Tool Calling 获取战绩、英雄数据库和版本信息。项目引入 Player Memory 记录玩家英雄偏好、能力短板和成长目标，使用 RAG 检索职业攻略与版本指南，最终生成个性化上分建议和阶段训练计划。通过 LangSmith 监控任务拆解正确率、工具调用成功率、RAG 命中率与平均响应时间，并设计胜率、KDA、段位变化和任务完成率等指标评估建议效果。
```

技术关键词：

```text
LangGraph, Multi-Agent, Planning, Tool Calling, RAG, Memory, LangSmith, Python 3.11
```

## 16. 验收标准

MVP 完成时应满足：

- 用户输入一句自然语言问题后，系统可以生成完整回答。
- Planner 能输出结构化任务拆解。
- Router 能根据任务类型调度至少 3 个 Agent。
- 战绩查询、英雄数据库、版本信息至少使用 Mock Tool 跑通。
- Memory 能读取和更新玩家画像。
- RAG 能从本地攻略文档中检索相关内容。
- Response Agent 输出包含诊断、数据依据、推荐和训练计划。
- LangSmith 可以看到一次完整 Trace。
- 至少包含 5 个单元测试和 1 个端到端工作流测试。

