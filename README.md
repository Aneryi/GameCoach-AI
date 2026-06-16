# GameCoach AI

基于 LangGraph 的游戏成长教练 Agent。

GameCoach AI 面向希望提升游戏水平的玩家，提供战绩分析、英雄推荐、出装建议、攻略检索和训练计划生成能力。系统通过多 Agent 架构将用户的自然语言问题拆解为可执行任务，并结合战绩数据、英雄数据库、版本信息、攻略 RAG 和玩家长期记忆，生成个性化成长建议。

## 项目定位

该项目对标游戏公司中 AI Agent、玩家成长系统、智能推荐、游戏数据分析和 RAG 应用相关岗位，重点展示：

- LangGraph 多 Agent 工作流编排。
- Planner Agent 任务拆解。
- Tool Calling 接入结构化游戏数据。
- RAG 检索攻略、版本指南和职业选手打法。
- Player Memory 记录玩家偏好、短板和目标。
- LangSmith 监控 Agent 执行质量与响应性能。
- 基于胜率、KDA、段位变化和任务完成率的效果评估。

## 核心能力

玩家可以输入：

```text
我最近胜率很低，怎么上分？
```

系统会自动拆解为：

```text
1. 分析最近战绩
2. 分析英雄池与常用位置
3. 结合当前版本推荐英雄
4. 检索相关攻略
5. 生成训练计划
```

最终输出：

```text
你的主要问题是中期死亡过多和团战站位靠前。

建议：
1. 暂时主练 2 个容错率较高的英雄。
2. 10 分钟后减少无视野单带。
3. 团战等待敌方刺客露头后再进场输出。
4. 未来 7 天重点训练补刀、地图观察和团战站位。
```

## 系统架构

```text
User
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
LangSmith / Evaluation
```

## 核心模块

### Planner Agent

理解用户问题并拆解任务，例如将“我最近胜率很低，怎么上分？”拆解为战绩分析、英雄池分析、版本推荐和训练计划生成。

### Tool Calling

系统规划以下工具：

- `match_history_tool`: 查询最近 20 场战绩、KDA、胜率、英雄表现。
- `hero_database_tool`: 查询英雄属性、定位、难度、克制关系。
- `patch_meta_tool`: 查询当前版本强势英雄、装备改动和版本趋势。
- `guide_rag_tool`: 检索职业攻略、版本指南和英雄教学内容。

### Player Memory

记录玩家长期画像：

```json
{
  "favorite_heroes": ["后羿", "狄仁杰"],
  "main_roles": ["damage"],
  "weaknesses": ["团战意识差", "中期容易掉点"],
  "goals": ["上钻石"]
}
```

### 成长计划生成

根据玩家问题生成阶段训练计划：

```text
Day 1: 补刀与经济训练
Day 2: 小地图观察训练
Day 3: 团战站位训练
Day 4: 逆风局少死训练
Day 5: 英雄池固定训练
Day 6: 版本强势英雄适应
Day 7: 综合复盘
```

### Agent 评估

从业务效果和 Agent 质量两个层面评估：

- 胜率变化。
- KDA 变化。
- 段位变化。
- 任务完成率。
- Planner 任务拆解正确率。
- Tool 调用成功率。
- RAG 命中率。
- 平均响应时间。

## 技术栈

- Python 3.11
- LangGraph (条件路由 DAG)
- LangChain (LangChain Tool 封装)
- LangSmith (Trace 监控)
- Pydantic (结构化输出校验)
- FAISS (向量检索)
- DashScope (阿里云 text-embedding-v2)
- FastAPI (REST API 服务)
- Pytest (17 个测试)

## 设计亮点

### Agent 跨游戏抽象

所有 Agent 的角色定义是**跨游戏通用**的（诊断/建议/记忆），但每个 Agent 内部的知识、Tool 调用和 prompt 随游戏类型切换。通过 GameTool 抽象接口，每个游戏 Tool 自带数据 + 校验规则 + 矛盾检测模式，切换游戏只需实现新的 GameTool 子类，引擎代码一行不改。

### Strategy 与 Build 解耦

Strategy 回答"怎么打"（站位、时机、节奏），Build 回答"带什么"（装备、构筑）。两者是不同层面的建议，互补而非重复。在无装备系统的游戏中，Build Agent 可被跳过，Strategy 独立工作。

### Memory 双层架构

Memory 分为 `confirmed`（高置信，参与决策）和 `pending`（暂存观察，不参与决策）两层，防止低质量信号污染长期画像。写入策略基于来源权重（数据 > 明确陈述 > 推断）和证据累积次数，条目有衰减机制。

### 幻觉分层防护

在线防线：Prompt 约束 + 结构化输出（每句事实绑定 source）+ 规则校验（纯代码，零成本，100% 覆盖）+ Embedding Grounding（检测无源事实，几乎零成本）。离线防线：批量 LLM 评分 → 趋势监控 → 定向深查。LLM 审核不在线做门禁，而是离线做趋势发现。

### 评估因果归因

区分"系统不行"和"玩家菜"：建议质量独立评估（不看执行结果），效果质量被执行质量加权。只有当建议质量独立通过、玩家也执行了、但战绩仍不提升时，才认为系统存在问题。

### 模型污染防护

三条污染路径（Memory → 置信度门禁 + 信号衰减；RAG → 版本衰减 + 反馈降权；Planner → 异常检测 + 告警不阻断）各自有对应的防护机制。核心原则：不完美信号不进长期存储，不确定的事情不进决策链路。

### 数据最小化

LLM 只收到生成建议所需的数据（英雄偏好、战绩指标、段位），不包含可关联到真人的标识符。每个外部接口（LLM API、LangSmith、日志）在数据流出前做 PII 清洗。

详细设计请查看 [SPEC.md](./SPEC.md)。

## 快速启动

```bash
# 安装依赖
pip install -e .[dev]

# CLI 模式：运行默认演示
python -m gamecoach.main

# CLI 模式：自定义问题
python -m gamecoach.main --message "我适合练什么英雄？"

# Web API 模式：启动服务
uvicorn gamecoach.api:app --host 127.0.0.1 --port 8000

# 然后访问 http://127.0.0.1:8000/docs 查看 Swagger UI
# POST /coach {"user_message": "我最近胜率很低，怎么上分？"}
```

## 文档

完整项目规格请查看 [SPEC.md](./SPEC.md)。

面试考察问答请查看 [INTERVIEW_QA.md](./INTERVIEW_QA.md)。

