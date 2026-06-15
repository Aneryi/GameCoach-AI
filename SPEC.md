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

### 5.4 执行模式：Plan-and-Execute 与 ReAct 的选择

当前系统采用 **Plan-and-Execute（先规划后执行）** 模式——Planner 一次性生成完整任务列表，Router 逐个分发执行。这与 ReAct（Reasoning + Acting，边想边做）模式有本质区别：

| | Plan-and-Execute（当前） | ReAct |
| --- | --- | --- |
| 推理时机 | 先规划全局任务列表，再分发执行 | 每步动态决策，Thought → Action → Observation 循环 |
| 工具调用 | Planner 预先指定每个任务需要的工具 | Agent 在循环中自主决定何时调用哪个工具 |
| 灵活性 | 低，执行路径在 Planner 阶段已确定 | 高，能根据中间结果调整方向 |
| 可控性 | 高，任务链路可审计、可打断 | 低，可能出现死循环（需 Guardrails，见 §13.3.3） |
| 适用场景 | 意图明确、任务边界清晰的用户请求 | 探索性检索、信息不完整时需要逐步收敛的场景 |

**本项目适合保持 Plan-and-Execute 为主模式的原因**：
- 大多数玩家请求意图明确（"分析战绩""推荐英雄""生成计划"），不需要边执行边调整方向
- 任务链路需要可审计——Planner 的任务列表是 LangSmith Trace 的关键节点
- Router 的调度决策需要基于完整的任务列表而非单步状态

**适合引入 ReAct 的环节**（后续迭代）：

| Agent | ReAct 适用场景 | 收益 |
| --- | --- | --- |
| `rag_agent` | 首次检索结果不理想时，LLM 改写 query 再检索 | 避免拿着低分结果强行生成 |
| `strategy_agent` | 分析数据 → 查攻略 → 发现新线索 → 针对性检索具体攻略 | 策略建议更有针对性和深度 |
| `build_agent` | 推荐装备后，根据克制关系链反向检查是否需要调整 | 减少"推荐了版本弱势装备"的概率 |

引入 ReAct 时需同时引入对应的 Guardrails（步数限制、重复调用检测、时间预算），详见 §13.3.3。

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

### 6.8 Agent 跨游戏抽象设计

以上 Agent 的角色定义是**跨游戏通用**的，但每个 Agent 内部的知识、Tool 调用和 prompt 随游戏类型切换：

| Agent | 角色（不变） | MOBA 场景 | FPS 场景 | 卡牌游戏场景 |
| --- | --- | --- | --- | --- |
| `MatchAnalysis` | 诊断"哪里出了问题" | 胜率、KDA、参团率、经济 | K/D、爆头率、首杀率、地图胜率 | 胜率、对局回合数、卡组胜率 |
| `Strategy` | 建议"怎么打" | 站位、转线、视野控制 | 架枪点位、道具管理、进点路线 | 起手留牌、资源曲线、对局节奏 |
| `Build` | 建议"带什么" | 出装顺序、符文搭配 | 枪械配件、道具组合 | 卡组构筑、sideboard |
| `Memory` | 记住"玩家是谁" | 偏好英雄、主玩位置 | 偏好武器、主玩地图 | 偏好职业/卡组类型 |

Agent 的**角色不变**（诊断/建议/记忆），但调用的 Tool 和注入的领域知识随游戏类型切换。因此每个 Agent 不应硬编码"射手""KDA"等 MOBA 概念，而应通过游戏配置抽象为"角色""核心指标"。详见 §12.1 可扩展性设计。

### 6.9 Strategy Agent 与 Build Agent 的职责区分

两者不重复，是不同层面的建议：

| 维度 | Strategy（怎么打） | Build（带什么） |
| --- | --- | --- |
| 关注点 | 决策、时机、站位、节奏 | 装备、符文、天赋、构筑 |
| 示例 | "团战站后排，等刺客先手" | "出末世 → 无尽 → 破晓" |
| 示例 | "10 分钟后别单带过河道" | "对面法刺多，逐日换魔女" |
| 示例 | "逆风只吃安全线" | "顺风局先出输出，逆风先出肉" |

类比：Strategy = 驾校教练教你怎么开车（什么时候变道、怎么判断车距），Build = 技师告诉你这辆车用什么轮胎、加什么油。两者互补：先有 Strategy（"我需要容错率高的打法"），再有 Build（"那就出魔女斗篷保命"）。在无装备系统的游戏（如部分 FPS）中，Strategy 可独立工作，Build Agent 可被跳过。

### 6.10 Memory Agent 与 Memory 模块的关系

| | Memory Agent（`agents/memory.py`） | Memory 模块（`memory/`） |
| --- | --- | --- |
| 定位 | 决策层："大脑" | 存储层："数据库" |
| 职责 | 从对话和分析结果中提取长期偏好；判断是否更新目标、弱点、英雄池 | JSON 文件读写、Schema 定义、CRUD 接口 |
| 调用关系 | 调用 Memory 模块的存取接口 | 被 Memory Agent 调用，不关心业务含义 |

类比：Memory Agent 是图书管理员（决定什么信息该放在哪个分类），Memory 模块是书架（物理存储）。分离后，Memory 模块可独立切换存储后端（JSON → PostgreSQL），不影响 Agent 的决策逻辑。

Memory 内部分为两层以避免污染（详见 §17 模型污染防护）：
- `confirmed`：高置信条目，参与 Planner 和 Strategy Agent 的决策。
- `pending`：暂存观察条目（单次低置信信号），不参与决策，积累足够证据后升级为 confirmed。

### 6.11 Strategy Agent 与训练计划的关系

Strategy Agent 负责生成阶段性的训练方向（如"先用 2 个容错英雄打 20 场"），而具体的每日训练计划（Day 1 补刀、Day 2 小地图…）可以由 Strategy Agent 内部生成，也可以通过独立的 TrainingPlan Agent 生成。MVP 阶段合并在 Strategy Agent 中，进阶阶段可拆分。

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

### 7.5 规则校验引擎（通用 → 游戏配置驱动）

规则校验引擎负责在 LLM 输出之后、返回用户之前，自动检测输出中的确定性错误。引擎本身是**完全通用的纯代码**（零 LLM 调用），不包含任何游戏领域知识。每个游戏的校验规则由对应的 GameTool 自带，引擎只负责加载并执行。

**设计原则**：能靠代码解决的问题绝不让 LLM 参与。

**GameTool 抽象接口**：

```python
class GameTool(Protocol):
    """游戏 Tool 的抽象接口——携带数据 + 校验规则 + 矛盾检测模式。"""

    def get_data(self, params: dict) -> dict: ...
    def get_consistency_rules(self) -> list[ConsistencyRule]: ...
    def get_contradiction_patterns(self) -> list[ContradictionPattern]: ...
```

**MOBA 示例**：

```python
class MOBAMatchHistoryTool(GameTool):
    def get_consistency_rules(self) -> list[ConsistencyRule]:
        return [
            ConsistencyRule(
                name="win_rate_calculation",
                check="metrics.win_rate == wins / total_matches", tolerance=0.01,
            ),
            ConsistencyRule(
                name="kda_calculation",
                check="metrics.avg_kda == (total_kills + total_assists) / max(total_deaths, 1)", tolerance=0.1,
            ),
        ]

    def get_contradiction_patterns(self) -> list[ContradictionPattern]:
        return [
            ContradictionPattern(
                name="high_death_no_risk",
                condition="analysis.avg_deaths >= 6",
                forbidden_in_advice=["越塔", "先手开团", "主动找架打", "冲进去"],
            ),
        ]
```

**FPS 示例**：

```python
class FPSMatchHistoryTool(GameTool):
    def get_consistency_rules(self) -> list[ConsistencyRule]:
        return [
            ConsistencyRule(name="kd_calculation",
                check="metrics.kd_ratio == total_kills / max(total_deaths, 1)", tolerance=0.1),
        ]

    def get_contradiction_patterns(self) -> list[ContradictionPattern]:
        return [
            ContradictionPattern(
                name="low_kd_no_aggression",
                condition="analysis.kd_ratio < 1.0",
                forbidden_in_advice=["rush", "push alone", "peek aggressively"],
            ),
        ]
```

**切换游戏 = 实现新的 GameTool 子类，引擎代码一行不改。**

规则引擎执行的检查项（全部零 LLM 成本）：

| 检查项 | 方法 | 示例 |
| --- | --- | --- |
| 结构完整性 | 必填字段是否存在 | final_response 长度 >= 50 字符 |
| 数据一致性 | 独立重算对比 | win_rate == wins / total |
| 引用真实性 | 检查 cited_id 是否在 RAG 检索结果中 | 模型引用的 doc_id 是否真实存在 |
| 数值自洽 | 建议中的数值是否和数据矛盾 | 回复说"死亡 6.8"但实际 avg_deaths = 4.1 |
| 关键词矛盾 | 模式匹配禁止词 | 数据说"死亡偏高"但建议含"越塔强杀" |
| 引用篡改 | 编辑距离检测 | 模型输出的"引用原文"和实际原文的相似度 < 0.8 |

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

## 9. RAG 准确性保障与幻觉防护

### 9.1 攻略准确性：四层防护

攻略准确性问题的根源在于信源质量和语义匹配的鸿沟，需分层解决：

**第一层：信源准入（入库前把关）**

```json
{
  "source": "marksman_teamfight_guide.md",
  "author": "某职业战队教练",
  "source_type": "pro_coach",
  "patch_version": "2026.06",
  "verified": true,
  "expires_at": "2026-09-01"
}
```

检索时按 `source_type` 加权排序：`pro_coach > high_elo_player > community_guide`。黑名单来源（匿名论坛、无法验证来源、过时版本）直接拒绝入库。

**第二层：检索质量控制（检索时把关）**

```text
Query → Embedding Search (top_k=20)
     → Version Filter（版本不匹配的踢掉）
     → Source Authority Rerank（信源权重加成）
     → Relevance Threshold（score < 0.6 的直接丢弃）
     → Context Compression（只保留和 query 最相关的段落）
     → 返回 top_k=3
```

Rerank 阶段：`最终分 = 语义分 × 信源权威分 × 版本时效分`。

**第三层：多源交叉验证**

如果多个独立来源对同一问题给出相同结论，可信度更高。评分规则：

| 支持来源数 | 矛盾来源数 | 结论 |
| --- | --- | --- |
| >= 2 | 0 | `confirmed` — 高置信，正常输出 |
| >= 1 | 0 | `plausible` — 单源支持，标注"参考" |
| >= 1 | >= 1 | `disputed` — 存在冲突，不应输出 |
| 0 | 0 | `insufficient` — 无足够证据 |

交叉验证只对高风险的具体战术建议（如"射手第一件出 X"）执行。

**第四层：时效性管理**

游戏攻略最大的准确性杀手是版本过期：
- 每个文档入库时记录 `patch_version`
- 每次版本更新时，上一版本攻略标记为 `possibly_stale`
- 版本跨度 > 2 个赛季的攻略自动归档，不再检索

### 9.2 幻觉分类与防护

RAG 系统中的幻觉分四类，解法不同。防护体系分在线（实时）和离线（异步批量）两层。

**在线防线（每次请求，100% 覆盖，零/极低成本）**：

```text
┌─ 在线（实时，每次请求都跑）────────────────────────────┐
│                                                       │
│  ① Prompt 约束：强制引证、禁止编造来源                    │  成本: $0
│  ② 结构化输出：每句事实绑定 source_doc + source_snippet  │  成本: $0
│  ③ 规则校验：引用真实性、数据一致性、逻辑自洽              │  成本: $0
│  ④ Embedding Grounding：检测无源事实                     │  成本: ~$0.00001
│                                                       │
│  → 通过这四层的回复直接返回用户，不阻塞                    │
└───────────────────────────────────────────────────────┘
```

**幻觉类型 A：模型无视检索结果，自己编**

```
检索到的攻略："射手应该站在前排后方 600-800 距离输出"
模型输出：    "射手应该主动绕后切对面后排"  ← 完全编的
```

解法：强制引证 + prompt 约束。System Prompt 要求每个建议引用至少一个来源文档。Output Schema 强制每个事实绑定单一来源：

```python
class AdviceItem(BaseModel):
    claim: str                          # "后羿第一件出急速战靴"
    source_doc_id: str                  # "hero_build_guide_houyi.md"
    source_snippet: str                 # "后羿首件推荐急速战靴，提升攻速..."
    claim_type: Literal["fact", "opinion", "conditional"]
```

规则校验：`claim_type == "fact"` 但没有 `source_doc_id` → 直接拒绝。

**幻觉类型 B：检索漂移（检索到不相关内容）**

解法：Answer Grounding 检查——在输出前检查回答中的关键主张是否能追溯到上下文文档：

```python
def check_grounding(answer: str, context_docs: list[str]) -> GroundingScore:
    claims = extract_claims(answer)
    grounded = sum(1 for c in claims if any(is_supported_by(c, d) for d in context_docs))
    return grounded / len(claims)  # 引证覆盖率
```

覆盖率 < 0.6 → 降级输出"当前攻略库中未找到相关内容"。

**幻觉类型 C：模型曲解原文（最难自动解决）**

```
攻略原文："逆风局射手不要单带过河道，应抱团清线"
模型曲解为："逆风局射手应该带线牵制"  ← 理解反了
```

解法：要求模型先引用原文再解读（"原文 → 解读"并排），而非直接给结论，使曲解更容易被后续审查发现。

**幻觉类型 D：模型编造引用来源**

```
模型输出："根据《王者荣耀射手进阶指南》第3章..."  ← 这本书不存在
```

解法：只允许引用检索返回的文档 ID 列表，Prompt 中明确标注可引用范围，规则层检查所有 cited_id 是否在检索结果中。

### 9.3 LLM-as-Judge 的成本优化

全量 LLM 校验的成本等同于把所有输出重新生成一遍，不可持续。优化策略：

**策略一：Embedding Grounding 替代 LLM（性价比最高）**

```python
def embedding_grounding_check(advice_text: str, retrieved_docs: list[str]) -> GroundingScore:
    sentences = split_into_sentences(advice_text)
    for sent in sentences:
        sent_emb = embed(sent)
        best_match = max(cosine_sim(sent_emb, doc_emb) for doc_emb in doc_embeddings)
        if best_match < 0.6:
            mark_as_ungrounded(sent)
```

| 方法 | 单次成本 | 延迟 |
| --- | --- | --- |
| LLM-as-Judge (GPT-4o) | ~$0.003 | ~2s |
| LLM-as-Judge (GPT-4o-mini) | ~$0.0003 | ~1s |
| **Embedding Grounding** | **~$0.00001** | **~50ms** |

**策略二：离线批量替代在线实时**

不是"生成 → 校验通过 → 返回用户"，而是"生成 → 返回用户（规则层已拦截确定性错误）→ 异步写入日志 → 每日批量跑 LLM 评分 → 输出质量日报 → 人工修 prompt/知识库"。

**策略三：风险分层触发**

```python
risk_score = 0.4 * static_risk + 0.4 * dynamic_risk + 0.2 * query_novelty

# static_risk: 类型固有风险（人工标注），build_recommendation=0.9, match_analysis=0.3
# dynamic_risk: 历史质量（离线评分统计，自动更新）
# query_novelty: 查询新颖度（embedding 和最接近已知 query 的距离）

if risk_score >= 0.7:    → 全量 LLM 校验
elif risk_score >= 0.4:  → Embedding grounding check
else:                    → 仅规则校验
```

`dynamic_risk` 从离线评分数据自动更新：某类型实际评分持续很低 → 风险分自动升高 → 增加校验频率。系统据此自动修正 static_risk 的初始标注偏差。

### 9.4 RAG 文档分片策略

游戏攻略文档结构差异极大，一种分片策略不能覆盖所有。采用**按文档类型混合分片**方案：

| 文档类型 | 特点 | 推荐策略 | 避免策略 |
| --- | --- | --- | --- |
| 官方版本公告 | 结构化短文本，按条目列出 | 按条目切分（以 `###` 标题或列表项为边界），200-500 token/条 | 固定窗口——会切断关联条目 |
| 英雄/职业攻略 | 半结构化：属性表 + 技能说明 + 出装推荐 + 对局技巧 | 按章节标题切分（`##`），保留表格完整性 | 语义窗口——表格的 embedding 语义弱，容易丢 |
| 职业复盘/教学 | 长文叙事，逻辑流强 | 语义窗口（512 token，overlap 128），按段落边界调整 | 固定大小——会切断推理链 |
| 装备/符文解析 | 短条目，每条独立 | 最小单元切分（每条装备一个 chunk），保留完整字段 | 任何切分——应以结构化 JSON 存而非向量检索 |

**实现**：

```python
class ChunkingStrategy:
    STRATEGIES = {
        "patch_notes":    SectionalChunker(heading_level=3, min_tokens=200, max_tokens=800),
        "hero_guide":     SectionalChunker(heading_level=2, min_tokens=300, max_tokens=1200),
        "pro_review":     SemanticWindowChunker(window=512, overlap=128),
        "build_analysis": AtomicChunker(preserve_tables=True),
    }

    def chunk(self, doc: Document) -> list[Chunk]:
        strategy = self.STRATEGIES.get(doc.metadata["doc_type"])
        return strategy.chunk(doc)
```

**四个关键决策**：

1. **不用固定大小切分**。Markdown 标题是天然语义边界——按 `##`/`###` 切比按 token 计数可靠。表格（出装表、克制表）被切断会完全丢失信息。

2. **Overlap 不是越大越好**。语义窗口 overlap 只需 128 token 保证不丢失跨窗口逻辑链。过大 overlap 让同一段信息出现在多个 chunk 中，降低检索多样性。

3. **结构化数据不进向量库**。英雄属性表、装备数值等精确查询走 Tool Calling（`hero_database_tool`），而非向量检索。向量检索适合"射手怎么站位"这类开放问题，不适合"后羿攻击力多少"这类精确查询。

4. **入库时标记，检索时过滤**。每个 chunk 携带 `doc_type` 和 `section_title` 元数据。检索时：用户问"当前版本推荐出装"→ 过滤 `doc_type=patch_notes` 优先，`doc_type=pro_review` 作为补充。

## 10. 成长计划生成规格

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

## 11. Agent 评估体系

### 11.1 评估的两层模型

评估体系分两层，分别回答不同的问题：

**第一层：Agent 质量（建议本身好不好？）——独立评估，不依赖玩家执行结果**

| 指标 | 类型 | 评估方式 |
| --- | --- | --- |
| Tool 调用成功率 | 硬指标（可自动计算） | 工具是否返回有效数据、有无异常、耗时 |
| RAG 命中率 | 硬指标 | 检索文档 score 是否 > 阈值、top_k 中相关文档占比 |
| 平均响应时间 | 硬指标 | LangSmith Trace 端到端耗时 |
| 任务拆解正确率 | 软指标（需裁判） | 标注数据对比 或 LLM-as-Judge |
| 建议可执行性 | 软指标 | LLM 评分："是否具体、可操作、可验证？1-5" |
| 个性化程度 | 软指标 | LLM 评分："是否利用了玩家历史画像？1-5" |

软指标的评估不依赖玩家执行结果，本质是"静态审查"——检查建议是否数据自洽、领域合理、具体可操作、与玩家画像匹配。

**第二层：业务效果（建议真的帮到玩家了吗？）——对比训练前后数据**

- 胜率变化：训练前后最近 20 场胜率差异。
- KDA 变化：平均 KDA 是否提升。
- 段位变化：是否达到阶段目标。
- 死亡次数变化：平均死亡是否下降。
- 任务完成率：用户训练计划打卡完成比例。
- 英雄熟练度变化：推荐英雄使用场次与胜率。

### 11.2 因果归因：区分"系统不行"还是"玩家菜"

核心矛盾：如果只看最终结果（胜率变了没），系统会在玩家不执行时错误地给自己差评。

解法：**把评估链路拆成三段，逐段归因**。

```text
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  建议质量      │ ──→ │  执行质量      │ ──→ │  效果质量      │
│  (系统负责)    │     │  (玩家负责)    │     │  (双方共同)    │
└──────────────┘     └──────────────┘     └──────────────┘
      ↓                     ↓                     ↓
  拆解对不对？           打卡了没？             胜率变了没？
  建议合不合理？         练了多久？             KDA 变了没？
  数据依据准不准？        完成质量如何？         段位动了没？
```

**效果质量指标必须被执行质量加权，不作为系统质量的直接判据。**

归因矩阵：

| 建议质量 | 玩家执行 | 战绩提升 | 归因结论 |
| --- | --- | --- | --- |
| ✅ 好 | ✅ 执行了 | ✅ 提升了 | 🎉 系统有效 |
| ✅ 好 | ✅ 执行了 | ❌ 没提升 | ⚠️ 建议方向需调整 / 时间不够 |
| ✅ 好 | ❌ 没执行 | ❌ 没提升 | 👤 玩家问题（不归咎系统） |
| ❌ 差 | ✅ 执行了 | ❌ 没提升 | 🤖 系统问题（建议本身有缺陷） |
| ❌ 差 | ❌ 没执行 | ❌ 没提升 | 🤖👤 双方问题（先修系统，再看玩家） |

### 11.3 建议质量的独立审查（不依赖执行结果）

在生成建议后立即运行，不等待玩家执行：

```python
def judge_advice_quality(state: GameCoachState) -> AdviceQualityReport:
    checks = []

    # 1. 数据一致性：分析结论和建议是否自洽
    if "死亡偏高" in state["match_analysis"]["summary"]:
        risky_advice = ["主动找架打", "越塔强杀", "先手开团"]
        for item in state["strategy"]["action_items"]:
            if any(risk in item for risk in risky_advice):
                checks.append(FAIL("数据表明死亡偏高，但建议中包含高风险动作"))

    # 2. 可操作性：建议是否具体
    for item in state["strategy"]["action_items"]:
        if is_vague(item):  # "提高意识""注意站位"
            checks.append(FAIL(f"建议过于模糊: {item}"))

    # 3. 覆盖度：每个弱点是否都有对应建议
    for weakness in state["match_analysis"]["weaknesses"]:
        if not any(weakness in item for item in state["strategy"]["action_items"]):
            checks.append(FAIL(f"弱点'{weakness}'无对应建议"))

    return {"checks": checks, "pass": all(c["pass"] for c in checks)}
```

### 11.4 离线评分与自动/人工改进分工

评分降低后的处理：

| 问题类型 | 能自动修吗？ | 方式 |
| --- | --- | --- |
| 知识库文档过期 | ⚠️ 半自动 | 自动检测版本不匹配、标记降权；新文档需人工编写 |
| 检索阈值不合理 | ✅ 能 | 根据 RAG 命中率自动调参 |
| Rerank 权重失衡 | ✅ 能 | 根据信源权威性效果自动调权重 |
| Prompt 设计缺陷 | ⚠️ 半自动 | A/B 测试不同 prompt，方向性调整需人工判断 |
| 模型能力退化 | ❌ 不能 | 换模型需人工决策 |
| 游戏版本 meta 变化 | ❌ 不能 | 需人工更新领域知识 |

```text
离线批量评分 → 评分连续 3 天下降 → 触发告警
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
            自动：提高采样率                    人工：查看低分样本
            自动：调整检索阈值                  人工：定位根因
            自动：标记低分文档                  人工：修 prompt/知识库/规则
```

### 11.5 LangSmith 监控指标

- `planner_task_accuracy`
- `tool_call_success_rate`
- `rag_hit_rate`
- `memory_update_success_rate`
- `avg_response_latency`
- `final_answer_quality_score`
- `training_plan_completion_rate`

## 12. LangSmith 接入规格

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

## 13. 非功能需求

### 13.1 可扩展性

- 工具层需要与 Agent 解耦，方便替换 Mock 数据为真实 API。
- 游戏配置需要抽象，避免所有逻辑绑定到单一游戏。
- RAG 文档加载、切分、索引和检索需要模块化。

### 13.2 可测试性

- Planner 输出必须是结构化 JSON，方便断言。
- 每个 Tool 需要有独立单元测试。
- MatchAnalysis Agent 的统计逻辑需要使用固定 Mock 数据测试。
- Workflow 需要测试常见用户意图路径。

### 13.3 稳定性与异常处理

系统需要应对三类运行时异常：节点执行失败、工具调用失败、工具调用死循环。核心原则：**已计算的不丢、不可用的标注、部分可用不等于全部失败。**

#### 13.3.1 节点失败：检查点恢复 + 分级降级

LangGraph 内置 checkpoint 机制在每个节点执行后自动保存 state 快照。节点失败时上游结果保留在最近一次成功的 checkpoint 中。

**分级降级策略**（按节点阻断程度）：

| 节点 | 失败影响 | 降级策略 |
| --- | --- | --- |
| `input_normalizer` | 阻断全部流程 | 用原始输入直接进 planner，跳过标准化 |
| `planner` | 阻断全部流程 | 回退到默认任务模板（根据关键词匹配预设计划） |
| `memory_loader` | 后续 Agent 缺少个性化上下文 | 使用空 memory，标注"未读取历史记录" |
| `match_analysis_agent` | 缺少数据诊断 | 跳过，strategy 仅基于 memory 和 RAG 生成建议，标注"暂未获取最新数据" |
| `rag_agent` | 缺少攻略引用 | 跳过，标注"攻略库暂不可用，以下建议基于通用策略" |
| `response_agent` | 阻断全部流程 | 直接拼接中间结果作为原始回复 |
| `evaluation_logger` | 不影响用户回复 | 静默失败，记录到错误日志 |

```python
try:
    result = graph.invoke(input_state)
except NodeExecutionError as e:
    partial_state = graph.get_state(config)  # 从 checkpoint 恢复上游结果
    degraded_response = generate_degraded_response(partial_state, e.failed_node)
```

**丢弃 vs 重试 vs 降级** 的判断逻辑：

```text
节点失败 → 是否可恢复（重试可能成功）？
              ├── 是 → 重试 1-2 次（指数退避）
              └── 否 → 该节点是否阻断主链路？
                          ├── 是 → 降级输出 + 标注"部分功能暂不可用"
                          └── 否 → 静默跳过，上游结果继续传递
```

#### 13.3.2 工具调用失败：重试 → 降级 → Agent 感知

与节点失败不同，工具是 Agent 调用的外部依赖——Agent 本身没崩，但拿不到数据。

**第一层：重试（仅瞬时故障）**

```python
def with_retry(max_retries=2, base_delay=1.0):
    """指数退避重试：1s → 2s → 放弃。只重试 TimeoutError/ConnectionError。"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except (TimeoutError, ConnectionError) as e:
                    if attempt < max_retries:
                        await asyncio.sleep(base_delay * (2 ** attempt))
            raise last_error
        return wrapper
    return decorator
```

不重试业务错误（参数非法、数据不存在）。

**第二层：降级返回（不抛异常）**

每个 Tool 必须提供降级返回值：

```python
def _degraded_response(self, player_id: str, limit: int) -> dict:
    return {
        "status": "unavailable",
        "reason": "战绩服务暂不可用",
        "matches": [],
        "fallback_note": "以下分析基于历史缓存数据（最近更新：2026-06-10）"
    }
```

**第三层：Agent 感知降级**

Agent 的 System Prompt 约定降级语义：

```text
"工具调用可能返回 status: 'unavailable'。
当遇到此状态时：
1. 不要编造数据。
2. 在回复中明确告知用户'该部分数据暂不可用'。
3. 基于 Memory 和 RAG 等其他可用信息继续生成建议。
4. 不要重复调用同一个已返回 unavailable 的工具。"
```

#### 13.3.3 工具调用死循环：三道 Guardrails

LLM 带 Tool Calling 的 ReAct 循环中最危险的情况——Agent 反复调用同一工具，每次拿到相同结果，但就是不停止。

**死循环的三种形态**：

| 形态 | 症状 | 根因 |
| --- | --- | --- |
| 重复调用 | 连续 N 次调用同一个工具，参数完全相同 | LLM 不满足于返回结果，试图"再查一次" |
| 震荡调用 | 工具 A → 工具 B → 工具 A → 工具 B ... | LLM 在两个工具间来回摇摆 |
| 逃逸调用 | 工具返回不符合预期，LLM 不断换参数重试 | LLM 不相信工具结果 |

**闸门 1：最大步数限制（兜底）**

```python
MAX_AGENT_STEPS = 10

if step_count > MAX_AGENT_STEPS:
    state["errors"].append("agent_reached_step_limit")
    return force_finish(state, reason="step_limit")
```

**闸门 2：重复调用检测（核心）**

```python
class LoopDetector:
    def __init__(self, max_repeats: int = 2):
        self.call_history: list[str] = []

    def record(self, tool_name: str, params: dict) -> bool:
        """返回 False = 检测到死循环"""
        signature = f"{tool_name}:{hashlib.md5(str(sorted(params.items())).encode()).hexdigest()}"
        self.call_history.append(signature)
        recent = self.call_history[-max_repeats:]
        if len(recent) >= max_repeats and len(set(recent)) == 1:
            return False  # 同一调用连续出现
        return True

    def detect_oscillation(self, window: int = 6) -> bool:
        """检测震荡调用（A→B→A→B）"""
        recent = self.call_history[-window:]
        unique = list(dict.fromkeys(recent))
        return len(unique) <= 2 and len(recent) >= 4
```

**闸门 3：时间预算（最后防线）**

```python
class TimeBudget:
    def __init__(self, timeout_seconds: float = 30.0):
        self.deadline = time.time() + timeout_seconds

    def is_expired(self) -> bool:
        return time.time() > self.deadline
```

**强制终止处理**：不丢弃已收集的结果，用已有部分信息拼出降级回复。Response Agent 在末尾追加标注：

```text
---
ℹ️ 本次回答基于部分可用信息生成（检索过程超时）。如需更完整的分析，请稍后重试。
```

#### 13.3.4 异常处理总图

```text
                    异常发生
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
      节点失败     工具调用失败    死循环/超时
          │           │           │
          ▼           ▼           ▼
    checkpoint  重试(指数退避)   Guardrails
    恢复 state  失败→降级返回    强制终止
          │           │           │
          └───────────┼───────────┘
                      ▼
              降级合成最终回复
              （不丢已有结果）
                      │
                      ▼
              标注不可用部分
              提示用户稍后重试
```

核心原则：
1. **已计算的不丢**：checkpoint 保留上游节点结果，死循环终止时已有的工具结果照用。
2. **不可用的标注**：不掩盖、不编造，明确告知用户哪部分数据暂缺。
3. **部分可用 ≠ 全部失败**：一个工具挂了，其他 Agent 继续工作，降级合成总比空白好。

### 13.4 安全与隐私

- 玩家 ID、历史战绩和成长记录属于用户数据。
- 本地开发阶段可使用 JSON 存储，生产环境应切换到数据库。
- 日志中避免记录敏感账号凭证。
- `.env` 不得提交到仓库。

## 14. 推荐技术栈

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

## 15. 里程碑计划

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

## 16. 简历描述建议

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

## 17. 验收标准

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

## 18. 模型污染防护

"模型污染"在此项目中不是指训练数据投毒，而是指**系统中的错误信息被循环放大，污染后续决策链路**。有三条关键污染路径。

### 18.1 污染路径与防护

**路径 1: Memory 污染 → 个性化偏差自激**

```text
玩家某次说了句"我觉得我打野还行"
  → Memory Agent 写入：strengths += "打野"
  → 后续 Planner 看到 memory：这个玩家擅长打野
  → 推荐的英雄和策略都偏向打野
  → 玩家的实际问题是射手位，打野建议全部无效
  → 但系统已经"相信"这个玩家喜欢打野了
```

解法：Memory 写入需要置信度门禁 + 信号衰减 + 来源加权。

```python
class MemoryUpdatePolicy:
    """
    Memory 不是"玩家说了一次就永久写入"——
    需要多次确认、来源加权、过期衰减。
    """

    def should_update(self, key: str, value: str, source: str, evidence_count: int) -> bool:
        source_weight = {
            "match_data":       0.9,   # 战绩数据 → 客观事实，高置信
            "explicit_stated":  0.7,   # 玩家明确说"我只玩射手" → 中高置信
            "inferred":         0.3,   # 从对话推断 → 低置信
        }.get(source, 0.1)

        if source_weight >= 0.9:
            return True                  # 数据来源直接写入
        if source_weight >= 0.7 and evidence_count >= 2:
            return True                  # 玩家说了两次以上
        if source_weight >= 0.3 and evidence_count >= 5:
            return True                  # 推断需要累积多次
        return False                     # 写入 pending，不进 permanent memory
```

Memory 分为两层：
- `confirmed`：高置信条目，参与 Planner 和 Strategy Agent 的决策。
- `pending`：暂存观察条目（单次低置信信号），不参与决策，积累足够证据后升级为 confirmed。

每个 memory 条目有 `decay_at`，长期未被新信号"续约"的条目自动过期。

**路径 2: RAG 知识库污染 → 错误攻略循环引用**

```text
一个低质量攻略进了知识库："后羿第一件出破军，伤害爆炸"
  → 玩家问"后羿怎么出装" → RAG 检索到这个攻略
  → 系统推荐破军首件 → 玩家反馈：输了
  → 如果系统没有反馈闭环，这个错误攻略永远在那
```

解法：文档质量衰减 + 使用反馈闭环。

```python
class DocumentQualityManager:
    def calculate_doc_weight(self, doc: Doc, feedback_log: list[Feedback]) -> float:
        base_weight = doc.source_authority       # 信源权威分（入库时标注）
        version_freshness = self._version_decay(doc.patch_version)
        feedback_score = self._feedback_penalty(doc.id, feedback_log)
        return base_weight * version_freshness * feedback_score

    def _version_decay(self, patch_version: str) -> float:
        """版本每差一个大版本，权重衰减 50%"""
        versions_behind = current_version - patch_version
        return max(0.1, 0.5 ** versions_behind)

    def _feedback_penalty(self, doc_id: str, feedback_log: list[Feedback]) -> float:
        """每 3 个差评，权重降一半"""
        negative_count = sum(1 for f in feedback_log
                             if f.doc_id == doc_id and f.rating == "unhelpful")
        return max(0.1, 0.5 ** (negative_count / 3))
```

**路径 3: Planner 输出污染 → 任务拆解劣化**

Planner 一次错误拆解（如用户问"怎么上分"被错误拆出"先练打野英雄"）会导致整个链路基于错误假设运行。

解法：异常检测——不判断"对不对"，只判断"怪不怪"。

```python
def detect_anomalous_plan(planned_tasks: list[PlannedTask], user_message: str) -> bool:
    historical_pattern = get_task_type_distribution(intent)
    current_pattern = Counter(t["task_type"] for t in planned_tasks)
    if distribution_divergence(current_pattern, historical_pattern) > 0.5:
        trigger_alert("planner_anomaly", ...)
        return True
    return False
```

这不阻止输出（新类型问题可能需要不同的拆解），但会生成告警供后续人工审查。

### 18.2 污染防护总图

```text
                    污染源
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
    Memory 污染    RAG 污染     Planner 污染
        │             │             │
        ▼             ▼             ▼
   置信度门禁      版本衰减      异常检测
   pending/confirmed 反馈降权    模式分布对比
   信号衰减        信源权重      告警不阻断
   来源权重
        │             │             │
        └─────────────┼─────────────┘
                      ▼
              人工审查（离线）
              → 修 memory 存储策略
              → 下架低质量文档
              → 调 Planner prompt
```

核心原则：**不完美信号不进长期存储，不确定的事情不进决策链路，异常不阻断但必须告警。**

## 19. 敏感信息与数据安全

### 19.1 数据敏感性分级

| 数据类型 | 敏感程度 | 泄露风险 |
| --- | --- | --- |
| 玩家 ID / 游戏账号 ID | 🔴 高 | 可关联到真人 |
| 历史战绩（20场对局详情） | 🟡 中 | 可推断水平、习惯、在线时间 |
| 段位 / 排位分 | 🟡 中 | 社交工程素材 |
| 短板、弱点、训练计划 | 🟡 中 | 如果被恶意利用（游戏内针对） |
| 英雄偏好、打法风格 | 🟢 低 | 孤立数据无意义 |

### 19.2 泄露面识别与防护

数据在系统中流经七条路径，每条都可能泄露：

```text
用户输入 player_id / 问题
  │
  ├──→ ① LangGraph State（内存中，Python 对象）
  │      │
  │      ├──→ ② Tool 调用（Mock 阶段无风险，真实阶段需 ID 映射）
  │      │
  │      ├──→ ③ LLM API（OpenAI / Claude）← 最高风险
  │      │
  │      ├──→ ④ LangSmith Trace ← 次高风险
  │      │
  │      ├──→ ⑤ Memory JSON 文件（本地磁盘）
  │      │
  │      └──→ ⑥ 日志 / print 输出
  │
  └──→ ⑦ 前端 / CLI 展示
```

**③ LLM API（最大泄露面）**：你无法控制 API 提供商如何处理数据，只能控制你发什么。

```python
class LLMSanitizer:
    """
    在发给 LLM 之前，清洗 state 中的敏感字段。
    LLM 不需要知道真实的 player_id 或账号信息。
    """

    SANITIZE_FIELDS = {
        "player_id": "[PLAYER_ANON]",
        "account_name": "[REDACTED]",
        "real_game_id": "[REDACTED]",
    }

    ALLOW_FIELDS = {
        # 这些可以发给 LLM，用于生成建议——足够且不泄露身份
        "favorite_heroes", "main_roles", "weaknesses",
        "goals", "rank", "match_metrics", "hero_win_rates",
    }

    def sanitize(self, state: GameCoachState) -> GameCoachState:
        clean = {}
        for key, value in state.items():
            if key in self.SANITIZE_FIELDS:
                clean[key] = self.SANITIZE_FIELDS[key]
            elif key in self.ALLOW_FIELDS:
                clean[key] = value
        return clean
```

**关键设计**：LLM 看到的"玩家"是一个匿名实体——有段位、有英雄偏好、有战绩指标，但不知道是谁。这足够生成建议，且零 PII 泄露。

**② Tool 调用**：真实 API 接入时，Tool 请求中不应带真实 ID，用内部匿名 ID 映射：

```python
class MatchHistoryTool:
    def __init__(self):
        self._id_mapping = IdMappingStore()  # player_001 → 真实游戏ID 的映射表

    def query(self, internal_player_id: str) -> dict:
        real_game_id = self._id_mapping.resolve(internal_player_id)
        return call_external_api(real_game_id)
        # 返回结果中不包含真实游戏 ID
```

**④ LangSmith Trace**：利用 Callback 机制或在 state 进入 graph 之前清洗——LangSmith 只看到干净版本。

**⑥ 日志**：

```python
# 禁止：
print(f"玩家 {player_id} 的战绩: {match_data}")

# 正确：
logger.info("match_analysis_complete", extra={
    "player_id_hash": hash(player_id),  # 哈希不可逆
    "match_count": len(matches),
    # 不记录任何对局细节
})
```

### 19.3 环境与存储安全

| 环境 | 策略 |
| --- | --- |
| 本地开发 | 全部 Mock 数据（`player_001`、`后羿`、`铂金 I`），不存在真正的敏感信息 |
| `data/memory/` | `.gitignore` 必须包含 |
| `.env` | 不得提交到仓库 |
| 生产环境 | PostgreSQL + 字段级加密；player_id 列加密存储；访问需要鉴权 |

### 19.4 设计原则

**数据最小化**：LLM 只收到生成建议所需的数据，不包含可关联到真人的标识符。系统设计时在每个外部接口（LLM API、LangSmith、日志）都做 PII 清洗，而非事后补救。

