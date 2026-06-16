# GameCoach AI — 基于 LangGraph 的多 Agent 游戏成长教练系统

使用 Python + LangGraph + LangChain 开发游戏成长教练 AI 原型，探索多 Agent 编排、RAG
攻略检索、用户长期记忆与评估归因体系在个性化推荐场景中的应用。项目背景是玩家个人战绩数据与攻略信息碎片化，通用大模型缺乏对用户长期画像的持续记忆，且难以将模糊的"如何提升"类问题拆解为可执行的阶段性训练计划。

系统基于 LangGraph 构建 Planner → Task Router → 多专用 Agent（MatchAnalysis / Character Recommendation / Strategy / Build / RAG / Memory）→ Response Agent 的条件路由 DAG：Planner 将用户自然语言意图拆解为结构化任务列表（LLM JSON 输出 + Pydantic 校验 + 规则 fallback 三层容错），Router 根据任务类型动态跳过不需要的节点，避免全链路串行执行。MatchAnalysis Agent 调用战绩查询、角色数据库、版本信息等 LangChain @tool 工具，计算胜率/KDA/死亡/参团等核心指标并定位能力短板；RAG Agent 基于 FAISS 向量库 + DashScope Embedding 检索攻略文档，检索结果注入 Strategy Agent 的 prompt 以生成有据可依的策略建议；Build Agent 结合角色推荐和装备数据库给出适配出装方案。引入 User Memory 读写机制，分析结果中的新弱点自动回写玩家画像，使下次请求的策略建议具备个性化上下文。

在工程实践中解决了多个实际问题上线的常见坑：(1) DeepSeek API 不支持 response_format 结构化输出，改用 JSON prompt + 正则提取 + Pydantic 后校验的通用方案，兼容任意 LLM；(2) LangGraph 条件边函数无法修改 state（传入的是快照副本），将路由决策预计算移至 Planner 节点返回，避免 routing_decisions 静默丢失；(3) HuggingFace 模型下载在境内网络不可达，切换至阿里 DashScope text-embedding-v2 API；(4) 全链路 LLM 降级设计——DEEPSEEK_API_KEY 未配置时，Planner 回退关键词规则匹配、Strategy 回退硬编码模板、Response 回退 f-string 拼装，系统仍产出结构化建议而非崩溃；(5) Windows GBK 终端输出 emoji 导致 UnicodeEncodeError，统一改用 ASCII 标记；(6) FAISS 索引加载失败时 retrieve() 静默返回空列表，通过在各层增加 status 字段（"ok"/"unavailable"）区分"无结果"和"不可用"，避免上层误判。

在评估体系上，通过 LangSmith SDK 的 create_feedback 接口上报 planner_task_count / degraded_nodes / rag_hits / response_length / health 五个自定义指标，配合 LangChain 自动 Tracing 实现全链路可观测。离线评估通过 evaluation/metrics.py 计算执行健康度、数据完整性、输出质量三维评分。接入 LangSmith 全链路追踪与监控。

该项目实现了一个从意图理解到周期训练计划生成的完整 Agent 闭环，加深了对 LangGraph 条件路由与状态管理、RAG 语义检索的工程落地、LLM 容错与降级设计、以及 Agent 可观测性评估的工程实践理解。
