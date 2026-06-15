# GameCoach AI 项目周期表

## 阶段 1: 项目骨架与可视化 Graph

难度：入门

周期：1-2 天

目标：

- 初始化 Python 3.11 项目结构。
- 配置 `pyproject.toml`、`.env.example`、`langgraph.json`。
- 定义 LangGraph 状态 `GameCoachState`。
- 实现可在 LangSmith Studio 中加载的最小 Graph。
- 使用 Mock 数据跑通一次玩家成长建议流程。

需要完成：

- 建立 `src/gamecoach` 包。
- 建立 `graph/state.py`、`graph/nodes.py`、`graph/workflow.py`。
- 建立 Mock 战绩、英雄、版本数据。
- 实现基础节点：
  - `input_normalizer`
  - `planner`
  - `memory_loader`
  - `match_analysis_agent`
  - `hero_recommendation_agent`
  - `strategy_agent`
  - `response_agent`
  - `evaluation_logger`
- 输出最终 `final_response`。
- 编写 LangSmith Studio 测试文档。

验收标准：

- LangSmith Studio 能识别 `gamecoach` graph。
- 输入一句用户问题后，状态中能看到任务拆解、战绩分析、推荐英雄、策略建议和最终回复。
- 不依赖真实游戏 API，不依赖真实 RAG，不依赖真实 LLM。

## 阶段 2: Planner 与 Router 强化

难度：基础

周期：2-3 天

目标：

- 将 Planner 从规则版升级为 LLM 结构化输出。
- 根据任务类型动态路由不同 Agent。
- 支持多类用户意图。

需要完成：

- 使用 Pydantic 定义 `PlannedTask` 输出模型。
- Planner 支持：
  - 战绩分析
  - 英雄推荐
  - 出装推荐
  - 攻略检索
  - 成长计划
- Router 根据任务列表选择执行路径。
- 增加异常兜底：Planner 输出错误时回退到默认计划。

验收标准：

- 至少 5 类用户问题能得到不同任务拆解。
- Studio Trace 中能清楚看到 Planner 到 Router 的决策链路。

## 阶段 3: Tool Calling 与数据分析

难度：中等

周期：3-5 天

目标：

- 将 Mock 数据封装为工具。
- 增强战绩分析逻辑。
- 输出更可靠的数据依据。

需要完成：

- 实现 `match_history_tool`。
- 实现 `hero_database_tool`。
- 实现 `patch_meta_tool`。
- 统计最近 N 场：
  - 胜率
  - KDA
  - 平均死亡
  - 平均参团率
  - 英雄胜率
  - 位置表现
- 支持工具调用失败时的降级输出。
- 设计 GameTool 抽象接口：每个 Tool 自带校验规则和矛盾检测模式，使规则引擎代码与具体游戏解耦。
- 实现规则校验引擎：在线实时检查引用真实性、数据一致性、数值自洽（纯代码，零 LLM 成本，100% 覆盖）。

验收标准：

- 每个 Tool 有独立单元测试。
- MatchAnalysis Agent 可以解释玩家短板来源。
- LangSmith 中能看到工具输入、输出和耗时。
- 规则引擎可检测出编造引用、数值计算错误、数据与建议矛盾三种确定性问题。

## 阶段 4: Memory 与个性化

难度：中等

周期：3-4 天

目标：

- 建立 Player Memory 存储。
- 让系统根据玩家长期画像生成个性化建议。

需要完成：

- 实现 JSON Memory Store。
- 支持读取玩家常用英雄、主玩位置、目标、弱点。
- 从用户输入和分析结果中提取新记忆。
- 将 Memory 注入 Planner 和 Strategy Agent。
- 实现 Memory 双层架构：`confirmed`（高置信，参与决策）+ `pending`（暂存，不参与决策），防止低质量信号污染长期画像。
- Memory 写入策略：来源加权（match_data 0.9 > explicit_stated 0.7 > inferred 0.3）+ 证据累积次数 + 衰减机制。

验收标准：

- 同一玩家多次请求时，建议能体现历史偏好。
- Memory 更新失败不影响最终回答。
- 单次低置信信号写入 pending，不污染 confirmed 层。

## 阶段 5: 攻略 RAG

难度：进阶

周期：4-6 天

目标：

- 接入本地攻略知识库。
- 将职业攻略、版本指南转化为可执行建议。

需要完成：

- 建立 `data/guides` 攻略文档。
- 实现文档加载、切分、索引。
- 使用 FAISS 或 Chroma 做向量检索。
- 实现 Query Rewrite 与 Context Compression。
- Response Agent 输出引用来源。
- 实现 RAG 准确性四层防护：信源准入（source_type 加权）、检索质量控制（版本过滤 + 信源 Rerank）、多源交叉验证、时效性管理（版本衰减）。
- 实现幻觉防护在线防线：结构化输出（每句事实绑定 source_doc + source_snippet）、Embedding Grounding 检查（检测无源事实）、强制引证 Prompt 约束。
- 实现 Prompt 约束：要求模型”先引用原文，再给出解读”，禁止编造来源。

验收标准：

- 输入”射手团战站位”能检索到相关攻略。
- RAG 无命中时不编造来源。
- LangSmith 中记录 query、top_k 文档和命中分数。
- 编造的文档引用在规则层被拦截。
- 版本跨度 > 2 个赛季的攻略自动降权/归档。

## 阶段 6: 成长计划与评估闭环

难度：进阶

周期：4-5 天

目标：

- 生成更细化的训练计划。
- 跟踪计划完成率和战绩变化。

需要完成：

- 实现 3 天、7 天、14 天训练计划模板。
- 支持每日训练目标和成功标准。
- 实现评估两层模型：Agent 质量（独立评估建议本身，不看执行结果）+ 业务效果（对比训练前后战绩）。
- 实现因果归因矩阵：通过"建议质量 → 执行质量 → 效果质量"三段链路，区分系统问题和玩家执行问题。
- 实现 LLM-as-Judge 离线评分：建议可执行性、数据自洽性、个性化程度打分。
- 实现风险分层触发：static_risk（类型固有风险）+ dynamic_risk（历史质量，自动更新）+ query_novelty（查询新颖度）三维决定校验深度。
- 输出评估报告。

验收标准：

- 训练计划具体到每日任务。
- 可以对比训练前后 Mock 战绩变化。
- 建议质量评分和业务效果评分独立计算。
- 玩家未执行但建议合理 → 不归咎为系统问题。

## 阶段 7: Demo 与简历包装

难度：综合

周期：2-4 天

目标：

- 做成可展示项目。
- 输出简历项目描述、演示问题和技术亮点。

需要完成：

- CLI Demo 或 FastAPI Demo。
- 准备 5 个典型测试问题。
- 准备项目截图或 Studio Trace 截图。
- 整理简历描述。
- 实现 LLMSanitizer：在发给 LLM API 和 LangSmith 前清洗 player_id 等敏感字段。
- 配置 `.gitignore` 排除 `data/memory/` 和 `.env`。
- 在代码中体现数据最小化设计意识（即使 Mock 阶段数据全是假的）。

验收标准：

- 可以完整演示一次从用户问题到成长建议的 Agent 流程。
- 简历描述能体现 LangGraph、Tool Calling、RAG、Memory、LangSmith。
- LLM 调用不传输可关联到真人的标识符。

