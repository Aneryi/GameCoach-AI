# GameCoach AI — 面试考察问答

## 一、架构设计

### Q1: 为什么用 Plan-and-Execute 而非 ReAct？Planner 拆错任务怎么办？

**为什么选 Plan-and-Execute：**

这个项目的核心场景是"玩家的提升诉求"——意图相对明确（分析战绩、推荐英雄、生成计划），不需要边执行边探索方向。Plan-and-Execute 带来三个 ReAct 没有的好处：

1. **链路可审计**：Planner 生成的任务列表是 LangSmith Trace 的关键节点，面试官/同事能看清"系统为什么调用了这个工具"。ReAct 的 Thought-Action-Observation 循环对调试者不友好。
2. **Router 需要全局视角**：如果 match_analysis 和 memory_lookup 可以并行（两者无依赖），Plan-and-Execute 能提前识别。ReAct 天然串行，浪费延迟。
3. **可控性强**：任务列表可以被 Human-in-the-loop 审核后再执行，ReAct 每一步打断的成本太高。

**Planner 拆错了怎么办：**

三层兜底，不是零保障：

- **规则回退**（当前 MVP）：如果 Planner（LLM）输出非法 JSON 或任务类型不在预定义列表中 → 回退到关键词匹配的默认计划模板。比如用户消息含"胜率"/"上分"→ 默认拆为 [match_analysis, hero_recommendation, training_plan]。
- **异常检测**（进阶）：不判断"对不对"，只判断"怪不怪"。如果当前拆解和历史同 intent 的任务类型分布差异 > 阈值 → 标记异常但不阻断，LangSmith 中生成告警供人工审查。详见 SPEC §18.1 污染路径 3。
- **结构化输出约束**（防线）：Planner 的输出 schema 是 Pydantic 强约束的，task_type 只能是枚举值，required_tools 只能是已注册的 tool name。LLM 不能"发明"一个不存在的任务类型。

诚实的局限：这三层能防"格式错误"和"明显异常"，但防不了"拆得不够好"——比如用户其实需要 build_recommendation，但 Planner 没拆出来。这是 Plan-and-Execute 的结构性代价，ReAct 在这个场景确实更灵活。

---

### Q2: Agent 间的数据依赖怎么处理？为什么全是串行？

**当前设计是故意简化的**。五个 Agent 确实存在数据依赖关系：

```text
memory_loader ──→ match_analysis ──→ strategy ──→ response
                       │                 ↑
                       └──→ rag ────────┘
                       └──→ build ──────┘
```

- `match_analysis` 输出"死亡偏高、参团率低"→ `rag` 应该用这些结论改写 query（"射手 死亡偏高 参团 站位"），而非用原始用户 query 检索
- `memory`（玩家偏好英雄）→ `build` 的推荐应该过滤玩家不会的英雄
- `rag` 检索到的打法 → `strategy` 应该引用而非凭空生成

**当前 MVP 为什么串行：**

阶段 1 的目标是"跑通一条完整链路"，不是"实现最优调度"。LangGraph 的 `add_edge` 串行链路是故意为之——先验证每个节点的输入输出在 state 中传递正确，再引入条件路由和并行。

**如果要改进：**

```python
# 当前（串行）：
workflow.add_edge("match_analysis_agent", "hero_recommendation_agent")

# 改进后（条件路由 + 并行）：
workflow.add_conditional_edges(
    "planner",
    route_by_tasks,  # 根据任务列表决定执行顺序
    {
        "memory_first": "memory_loader",
        "analysis_first": "match_analysis_agent",
    }
)
# match_analysis 和 memory 并行（无依赖），strategy 等两者都完成
```

LangGraph 的 `Send` API 支持 fan-out：Planner 拆出 5 个任务 → 每个任务通过 `Send` 发给对应 Agent → 所有 Agent 并行执行 → 结果汇总到 response_agent。但这要求 Agent 间确实无数据依赖，实际不是这样。

---

### Q3: Strategy 和 Build 结果矛盾怎么办？

**这是真实存在的问题**。比如 Build Agent 推荐纯输出装（高攻低防），Strategy Agent 建议"团战站后排保命"——两者不矛盾。但如果 Build 推荐"全肉装冲阵型"，Strategy 建议"站后排输出"——矛盾了。

**当前设计中的处理**：

Response Agent 汇总时有隐式的优先级：数据诊断（MatchAnalysis）> 策略建议（Strategy）> 装备推荐（Build）。Build 应该服务于 Strategy，而非反过来。Response Agent 的 prompt 中应明确：

```text
"Strategy 和 Build 的建议如果存在矛盾，以 Strategy 为准。
Build 的装备推荐必须兼容 Strategy 的打法建议。
如果 Build Agent 推荐的装备组合与 Strategy Agent 描述的打法场景不一致，
请以 Strategy 为准调整装备推荐，或标注'此装备组合适用于以下替代打法...'。"
```

**更根本的解法**（后续迭代）：不要让 Strategy 和 Build 独立生成再汇总——让 Build Agent 接收 Strategy Agent 的输出作为输入。Build 的 prompt 变成："基于以下打法策略（Strategy 输出），推荐适配的装备组合"。这样 Build 天然受 Strategy 约束，矛盾在生成阶段就被消除。

---

## 二、LangGraph 工程实践

### Q4: Checkpoint 具体怎么用？恢复后怎么跳过已成功的节点？

**Checkpoint 机制**：

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

# 每次调用带 thread_id，LangGraph 自动在每个节点后保存 state
config = {"configurable": {"thread_id": "user_session_123"}}

try:
    result = graph.invoke(input_state, config)
except Exception:
    # 获取最近的 checkpoint state——上游节点结果都在
    last_state = graph.get_state(config)
    # last_state.values → 最后一个成功节点之后的 state
    # last_state.next → 下一个待执行的节点（空表示已完成或失败）
```

**关键细节**：
- Checkpoint 存的是完整的 `GameCoachState` 快照，不只是"执行到哪个节点"
- 恢复时拿到的是上一个节点**输出后**的 state，即失败节点的输入已经就绪
- `graph.get_state(config).next` 告诉你"下一步该执行哪个节点"——空值说明执行完了或异常终止

**跳过已成功节点**：

LangGraph 没有原生的"跳过节点重放"功能。恢复策略不是"继续执行"，而是：

```python
# 方案 1：从 checkpoint state 重新 invoke（从头开始但有 state）
# 节点函数检测到自己的输出已存在于 state → 跳过执行
def match_analysis_agent(state: GameCoachState) -> GameCoachState:
    if state.get("match_analysis"):  # 已有结果，上一轮已成功
        return {}  # 返回空更新，不重新执行
    # ... 正常执行 ...

# 方案 2：用 LangGraph 的 interrupt 机制
# 在关键节点后设置 interrupt，失败时从该节点之后恢复
workflow.add_node("match_analysis_agent", match_analysis_agent)
# 编译后通过 graph.update_state() 手动设置 state 并指定恢复节点
```

诚实的局限：LangGraph 的 checkpoint 主要设计用于"人机交互中断后恢复"（如 Human-in-the-loop），不是为"异常重试"设计的。节点失败的恢复更多依赖节点函数内部的 try/except + 降级返回，而非 graph 层面的重放。

---

### Q5: GameCoachState 臃肿怎么办？

**这是一个真实的工程问题**。当前 state 已经 15+ 字段，随 Agent 增多会持续膨胀。

**方案一：只传递必要字段（当前可用）**

节点函数返回 `dict` 而非完整 state，LangGraph 会自动做 partial update：

```python
def match_analysis_agent(state: GameCoachState) -> dict:
    # 只读取需要的字段
    player_id = state["player_id"]
    # 只返回自己产生的字段
    return {"match_analysis": analysis, "match_data": data}
    # 不碰 hero_recommendations、training_plan 等无关字段
```

这不能解决 state schema 本身的臃肿，但能减少节点间的耦合。

**方案二：LangGraph Subgraph（推荐）**

将相关 Agent 封进子图，每个子图有独立的 state schema：

```python
# 主图 state：只保留跨子图需要传递的核心字段
class MasterState(TypedDict):
    user_message: str
    player_id: str
    planned_tasks: list[PlannedTask]
    final_response: str

# 分析子图 state：战绩分析 + 英雄推荐相关字段
class AnalysisSubState(TypedDict):
    player_id: str
    match_data: dict
    match_analysis: dict
    hero_recommendations: list[dict]

# 策略子图 state：策略 + 构建相关字段
class StrategySubState(TypedDict):
    match_analysis: dict
    memory: dict
    strategy: dict
    build_recommendations: list[dict]

analysis_subgraph = StateGraph(AnalysisSubState)
# ... 添加 match_analysis_agent, hero_recommendation_agent ...

strategy_subgraph = StateGraph(StrategySubState)
# ... 添加 strategy_agent, build_agent ...

# 主图引用子图
master_workflow.add_node("analysis_subgraph", analysis_subgraph.compile())
master_workflow.add_node("strategy_subgraph", strategy_subgraph.compile())
```

好处：每个子图只关心自己的字段，state schema 按业务边界拆分，Agent 间字段不会互相污染。

**方案三：按需使用 Channel（过度工程，不建议 MVP）**

LangGraph 支持自定义 state channel，可以实现字段级读写控制。但对于这个项目的规模，subgraph 已经足够。

---

### Q6: Tool 的 description 怎么写？

**这是 Tool Calling 实践中最重要的经验**——description 好坏直接影响调用准确率。

**原则**：

```python
# ❌ 差：描述太泛，LLM 不知道什么时候用
@tool
def match_history_tool(player_id: str, limit: int, game: str) -> dict:
    """查询玩家战绩数据。"""

# ✅ 好：明确触发条件、返回内容、适用场景、限制
@tool
def match_history_tool(player_id: str, limit: int = 20, game: str = "moba") -> dict:
    """
    查询玩家最近 N 场对局的详细数据。

    使用时机：用户问到"最近战绩""胜率""表现如何""为什么一直输"时调用。
    不要使用时机：用户只问英雄推荐或出装建议，没有提到战绩分析时。

    返回内容：每局的胜负、KDA、参团率、经济、时长等字段。
    限制：最多返回最近 50 场。game 参数默认 "moba"。

    如果此工具返回 status='unavailable'，不要重试，
    告知用户数据暂不可用，并基于 Memory 等其他来源继续建议。
    """
```

**关键要素**：
1. **When to use / When NOT to use**：比"这个工具做什么"更重要。LLM 的 Tool Calling 主要问题是**滥用**（不该调的时候调），不是遗漏。
2. **返回格式提示**：不是完整 schema，但告知关键字段名（如 `matches`、`status`），帮助 LLM 理解返回值。
3. **失败语义**：明确告诉 LLM 遇到 `unavailable` 时怎么做——不要重复调用。
4. **参数默认值和约束**：`limit=20, max=50` 写在 description 里，比只靠 schema 更可靠。

**测试方法**：准备 30 条用户输入，人工标注每条应该调用哪些 Tool → 跑一遍看准确率。真正的坑在"LLM 不调"和"LLM 乱调"的边界 case。

---

## 三、幻觉控制

### Q7: 规则标注是把人工成本转移了——这个批评怎么回应？

**诚实的回答：是的，zero-LLM-cost ≠ zero-human-cost。**

但转移不等于坏事——关键看**转移给了谁**和**复用率**：

1. **规则标注是一次性成本**。为一个游戏标注的 `contradiction_patterns`（如"死亡偏高时禁止建议越塔"）在成千上万次用户请求中复用。标注 30 条规则 → 覆盖 100 万次对话 → 每次对话的边际人工成本趋近于零。

2. **标注者是领域专家而非 AI 工程师**。规则标注不需要写代码——游戏教练可以标注"当数据表明显然 X 时，不应该建议 Y"。这比让教练去调 LLM prompt 可行得多。

3. **规则引擎本身不需要标注也能工作**。引用真实性检查（cited_id 是否在检索结果中）、数值自洽（win_rate 是否 = wins/total）是真正的零标注。只有 contradiction_patterns 需要领域知识——而且这些规则可以渐进式添加。

**真正的设计权衡**：规则校验的覆盖率（能发现多少种幻觉）vs 标注维护成本（每种游戏、每种语言要维护多少规则）。答案是：通用检查（引用真实性、数值自洽）零人工成本覆盖了最蠢的幻觉类型；领域特定检查（矛盾检测）是可选增强，成本由使用方承担。

---

### Q8: Embedding Grounding 的误判率——近义假话怎么区分？

**确实区分不了**。"后羿出破军伤害爆炸"和"后羿出末世伤害稳定"的 embedding 可能很接近——两者都是"后羿 + 装备 + 效果描述"的语义结构。

**Embedding Grounding 擅长发现的**：
- 编造完全不存在的概念："后羿出星空之杖"（装备不存在，向量和任何文档都不接近）
- 跨领域缝合："后羿学打野刀出门"（把打野概念嫁接到射手，向量偏离射手装备文档）

**Embedding Grounding 发现不了的**：
- 数值篡改："后羿出末世 + 200 攻击力"（实际上末世 +60，但语义结构对）
- 近义替换：上面那个例子

**所以 Embedding Grounding 是防线之一，不是唯一防线**：

```text
Embedding Grounding（查"像不像"）→ 筛掉明显的编造
    +
结构化输出强制绑定 source_snippet → "你说的这句引用在哪篇攻略的哪一行？"
    +
规则校验（编辑距离对比引用原文） → "你引用的原文真的是原文吗？"
```

三层叠加的效果：Embedding 筛掉"看起来完全不像"的，source_snippet 绑定迫使模型自己交代来源，编辑距离检测筛掉"看起来像但被篡改了的"。近义假话会穿过 embedding 层，但会在编辑距离检测层被拦截——因为模型输出的"引用原文"和实际原文的编辑距离会很大。

**误判率的诚实回答**：没测过（Mock 数据阶段）。生产环境需要标注一批"已知幻觉"的测试集来算召回率。

---

### Q9: 离线评分没阻止坏回复——评估到底帮了什么？

**诚实的回答：当前评估体系是"发现问题的雷达"，不是"拦截问题的盾牌"。**

两者分工：

```text
在线防线（盾牌）：
  Prompt 约束 + 结构化输出 + 规则校验 + Embedding Grounding
  → 在回复返回给用户之前，拦截确定性错误
  → 成本：几乎为零
  → 覆盖：可自动判断的错误（引用造假、数值矛盾、格式错误）

离线评分（雷达）：
  LLM-as-Judge 评分 → 趋势监控 → 定向深查
  → 不拦截单条回复，但发现系统性问题
  → 成本：有，但批量 + 便宜模型
  → 覆盖：需要语义判断的软质量（建议是否真的可执行、个性化是否到位）
```

**离线评分的真正价值**：
- **发现趋势性退化**：这周的"建议可执行性"评分从 4.2 降到 3.5 → 可能是知识库过时或 prompt 被意外修改
- **找到问题类型**：不是所有回复都差——是 build_recommendation 类型评分骤降，match_analysis 类型正常 → 定位到 Build Agent 或装备数据问题
- **为人工改进提供优先级**：一个月 1000 条回复，人工不可能全看。离线评分挑出最低分的 20 条 → 人工只需看这 20 条

---

## 四、评估体系

### Q10: 执行数据采集不到怎么办？玩家可能撒谎。

**这是评估体系最大的数据依赖漏洞。诚实承认。**

三种采集方式，可行性递减：

| 方式 | 可靠性 | 可行性 | 适用场景 |
| --- | --- | --- | --- |
| 自动采集（游戏 API） | 高 | 低——大多数游戏不开放 API | 理想情况，Demo 用 Mock 数据模拟 |
| 玩家自报（打卡/问卷） | 中——可能夸大或遗漏 | 高 | MVP 阶段可行 |
| 行为推断（从后续战绩变化反推） | 低——无法区分"练了没用"和"根本没练" | 中 | 作为辅助信号 |

**最务实的办法**：

在 Demo 阶段坦诚标注数据假设，面试时可以主动说：

> "评估体系的设计逻辑是完整的——建议质量 → 执行质量 → 效果质量三段归因。但执行质量的数据采集是薄弱环节：当前依赖玩家自报打卡，可靠性有限。如果接入真实游戏 API，战绩数据和训练量可以自动采集，归因链条会完整很多。目前的 Mock 数据阶段，这个链路是假设闭环的。"

面试官想听的不是"我完美解决了"，而是"我知道问题在哪，以及如果有资源我会怎么解决"。

---

### Q11: LLM-as-Judge 的可信度——裁判自己也会判错吧？

**会。LLM-as-Judge 的评分不是真相，是参考信号。**

**可信度取决于场景**：

| 场景 | 可信度 | 原因 |
| --- | --- | --- |
| 判断"建议是否具体可操作" | 高 | "注意站位"vs"站在后排 600 距离输出"——LLM 识别具体性比人稳定 |
| 判断"回复是否自洽" | 中高 | 对比数据结论和策略建议是否有矛盾——LLM 能做文本对比 |
| 判断"策略是否符合游戏常识" | 中 | 依赖 LLM 训练数据中的领域知识，可能过时或错误 |
| 判断"个性化是否到位" | 中低 | 需要理解玩家历史画像和当前建议的匹配度，容易漏判 |

**校准方法**：

```python
# 对 20% 的 LLM 评分结果做人工复核，计算 LLM vs 人工的偏差
def calibrate_llm_judge(llm_scores: list[float], human_scores: list[float]) -> float:
    """如果 LLM 系统性偏高 0.5 分，对所有 LLM 评分减 0.5 做校准"""
    bias = mean(llm_scores) - mean(human_scores)
    return bias
```

**触发人工复核的条件**：
- LLM 评分 < 2.5/5（严重低分）
- LLM 评分 > 4.8/5（可能漏判）
- LLM 评分和人类标注的历史偏差 > 0.5

---

## 五、工程务实

### Q12: 端到端延迟大概多少？最大瓶颈在哪？

**预估（Mock 阶段未实测）**：

```text
input_normalizer    →  < 1ms   （纯字符串处理）
planner (LLM)       →  1-2s    （结构化输出，单次 LLM 调用）
memory_loader       →  < 10ms  （JSON 文件读取）
match_analysis      →  < 5ms   （纯数值计算，Mock 数据）
hero_recommendation →  < 5ms   （数据匹配）
strategy (LLM)      →  1-2s    （生成策略，单次 LLM 调用）
rag_agent (LLM+检索) →  0.5-1s （向量检索 <50ms + LLM 生成）
response (LLM)      →  1-2s    （汇总生成）
evaluation_logger   →  < 1ms   （纯计算）
                            ─────
                      总计：4-7s
```

**最大瓶颈**：LLM 调用占了 80-90% 的时间。当前串行架构下，planner → strategy → rag → response 四次 LLM 调用是串行的。

**改进方向**：
- planner 和 memory_loader 并行（memory 不依赖 planner 结果）
- match_analysis + hero_recommendation + rag 并行（三者无相互依赖）
- 用更快的模型做 planner（如 GPT-4o-mini），省下的时间给 strategy 用 Claude Sonnet

并行后预估可降到 2-3s。

---

### Q13: pending → confirmed 的阈值怎么定？

**不是写死的，是按来源类型配置的**：

```python
UPGRADE_THRESHOLDS = {
    # (来源权重, 最少证据次数)
    "match_data":       (0.9, 1),   # 数据来源：出现 1 次即升级
    "explicit_stated":  (0.7, 2),   # 玩家明说：需要说 2 次
    "inferred":         (0.3, 5),   # 系统推断：需要 5 次独立推断
}

def should_upgrade(entry: PendingEntry) -> bool:
    threshold_weight, threshold_count = UPGRADE_THRESHOLDS[entry.source]
    # 来源权重达标 AND 证据次数达标
    return entry.source_weight >= threshold_weight and entry.count >= threshold_count
```

**不同游戏的阈值应该一样吗？** 不一样。FPS 游戏中"玩家偏好武器"的信号比 MOBA 中"玩家偏好英雄"更稳定（武器选择变化比英雄慢）。阈值应该放在游戏配置中，而非写死在 Memory 模块。

---

### Q14: 真实 API 接入，Tool 层抽象够用吗？

**不够——当前 Tool 抽象只覆盖了"正常返回"的情况。**

真实 API 带来的额外需求：

```python
class ProductionTool(GameTool):
    """真实 API 接入时的 Tool 基类"""

    def validate_input(self, params: dict) -> ValidationResult:
        """入参校验：player_id 格式、limit 范围。坏参数不应发到外部 API。"""

    def normalize_response(self, raw: dict) -> dict:
        """响应归一化：外部 API 返回格式可能随版本变化，在此层统一为内部 schema。"""

    def rate_limit_check(self) -> bool:
        """限流检查：API 有 QPS 限制，超限时排队或降级。"""

    def cache_lookup(self, params: dict) -> dict | None:
        """缓存检查：相同 player_id + limit 的查询 5 分钟内不重复请求。"""
```

这些都是 Mock 阶段不需要、但生产环境第一天就会遇到的问题。当前的 GameTool 抽象接口已经把 `get_data`、`get_consistency_rules`、`get_contradiction_patterns` 标准化了——`validate_input`、`normalize_response`、`rate_limit_check` 应该作为下一个版本的接口扩展。

---

## 六、反思与开放题

### Q15: 如果重来一次，哪个设计决策会做不同选择？

**Agent 拆分太细了。**

当前 5 个 Agent（match_analysis / strategy / build / rag / memory），每个都很"薄"——memory_loader 只是读一个 JSON 文件，hero_recommendation 只是匹配两个列表。这些逻辑不需要独立的 Agent 节点，合并为 3 个更合适：

```text
当前：                          重来：
match_analysis_agent            analysis_agent（战绩分析 + 数据诊断）
hero_recommendation_agent  →    recommendation_agent（英雄推荐 + 装备推荐 + 攻略检索）
strategy_agent                  strategy_agent（打法策略 + 训练计划）
rag_agent                  →
build_agent                →
memory_loader                   memory 作为以上所有 Agent 的共享上下文层，而非独立 Agent
```

拆太细的代价：
- 每个 Agent 节点都是一次潜在的 LLM 调用，延迟累加
- Graph 节点太多，LangSmith Trace 看起来很"壮观"但大部分节点做的事情太少
- Agent 间的数据传递依赖 state 字段，字段数膨胀

**但拆得细也有一个好处**（这也是一开始选择细拆的原因）：作为面试项目，细拆能清晰展示 LangGraph 的多节点编排能力。重来一次，我会在"展示清晰度"和"工程务实"之间取一个折中——合并 match_analysis + hero_recommendation，保留 strategy/training_plan 独立，把 memory 降级为上下文层。

---

### Q16: 最让你觉得"做对了"的设计是哪个？

**Memory 的 confirmed/pending 双层架构。**

原因：
1. **这个设计解决了真实问题**——不是"玩家说了一次就永远记住"，而是"多次确认才入长期记忆"。这在所有个性化系统里都适用，不分游戏类型。
2. **成本极低**——本质上只是一个写入前的判断逻辑（来源权重 + 证据次数 + 衰减），不需要额外存储或 LLM 调用。
3. **防止了最难排查的 bug**——"系统越来越不准但不知道为什么"。没有双层架构，低质量信号会缓慢污染 Memory，导致所有基于 Memory 的决策越来越歪，而且很难定位根因。有了 confirmed/pending 分层，至少能回溯"这条信息是什么时候、基于什么证据写入的"。
4. **这个设计是自发想到的**，不是抄的教程。在讨论模型污染的时候突然意识到"Memory Agent 的写入策略本身就是污染源"，然后推导出需要门禁机制。

---

## 七、RAG 分片策略

### Q17: 游戏攻略文档用什么分片策略？

**不能用一种策略切所有文档。** 游戏攻略的内部结构差异极大：

| 文档类型 | 特点 | 推荐策略 | 避免策略 |
| --- | --- | --- | --- |
| 官方版本公告 | 结构化短文本，按条目列出 | **按条目切分**（以 `###` 标题或列表项为边界），每条约 200-500 token | 固定窗口——会切断关联条目 |
| 英雄攻略 | 半结构化：属性表 + 技能说明 + 出装 + 技巧 | **按章节标题切分**（`##`），保留表格完整性 | 语义窗口——表格的 embedding 语义弱，容易丢 |
| 职业复盘/教学 | 长文叙事，逻辑流强 | **语义窗口**（512 token，overlap 128），按段落边界调整 | 固定大小——会切断推理链 |
| 装备/符文解析 | 短条目，每条独立 | **最小单元切分**（每条装备一个 chunk），保留完整字段 | 任何切分——应该以结构化 JSON 存而非向量检索 |

**实现**：

```python
class ChunkingStrategy:
    STRATEGIES = {
        "patch_notes":    SectionalChunker(heading_level=3, min=200, max=800),
        "hero_guide":     SectionalChunker(heading_level=2, min=300, max=1200),
        "pro_review":     SemanticWindowChunker(window=512, overlap=128),
        "build_analysis": AtomicChunker(preserve_tables=True),
    }

    def chunk(self, doc: Document) -> list[Chunk]:
        strategy = self.STRATEGIES.get(doc.metadata["doc_type"])
        return strategy.chunk(doc)
```

**四个关键决策**：

1. **不用固定大小切分**。Markdown 标题是天然的语义边界——按 `##` / `###` 切比按 token 数切可靠得多。表格（出装表、克制表）被切断会完全丢失信息。

2. **Overlap 不是越大越好**。语义窗口的 overlap 只需保证不丢失跨窗口的逻辑链——128 token 足够覆盖一个推理步骤的上下文。过大 overlap 让同一段信息出现在多个 chunk，降低检索多样性。

3. **结构化数据不进向量库**。英雄属性表、装备数值这类精确查询走 Tool Calling（`hero_database_tool`），而非向量检索。向量检索适合"射手怎么站位"这类开放问题，不适合"后羿攻击力多少"。

4. **入库时标记，检索时过滤**。每个 chunk 携带 `doc_type` 和 `section_title` 元数据。检索时：用户问"当前版本推荐出装"→ 过滤 `doc_type=patch_notes` 优先，`doc_type=pro_review` 作为补充。
