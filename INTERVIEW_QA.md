# GameCoach AI — 面试考察问答

## 一、架构设计

### Q1: 为什么用 Plan-and-Execute 而非 ReAct？Planner 拆错任务怎么办？

**为什么选 Plan-and-Execute：**

这个项目的核心场景是"玩家的提升诉求"——意图相对明确（分析战绩、推荐角色、生成计划），不需要边执行边探索方向。Plan-and-Execute 带来三个 ReAct 没有的好处：

1. **链路可审计**：Planner 生成的任务列表是 LangSmith Trace 的关键节点，面试官/同事能看清"系统为什么调用了这个工具"。ReAct 的 Thought-Action-Observation 循环对调试者不友好。
2. **Router 需要全局视角**：如果 match_analysis 和 memory_lookup 可以并行（两者无依赖），Plan-and-Execute 能提前识别。ReAct 天然串行，浪费延迟。
3. **可控性强**：任务列表可以被 Human-in-the-loop 审核后再执行，ReAct 每一步打断的成本太高。

**Planner 拆错了怎么办：**

三层兜底，不是零保障：

- **规则回退**（当前 MVP）：如果 Planner（LLM）输出非法 JSON 或任务类型不在预定义列表中 → 回退到关键词匹配的默认计划模板。比如用户消息含"胜率"/"上分"→ 默认拆为 [match_analysis, character_recommendation, training_plan]。
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

- `match_analysis` 输出"死亡偏高、参团率低"→ `rag` 应该用这些结论改写 query（"damage role 死亡偏高 参团 站位"），而非用原始用户 query 检索
- `memory`（玩家偏好角色）→ `build` 的推荐应该过滤玩家不会的角色
- `rag` 检索到的打法 → `strategy` 应该引用而非凭空生成

**当前 MVP 为什么串行：**

阶段 1 的目标是"跑通一条完整链路"，不是"实现最优调度"。LangGraph 的 `add_edge` 串行链路是故意为之——先验证每个节点的输入输出在 state 中传递正确，再引入条件路由和并行。

**如果要改进：**

```python
# 当前（串行）：
workflow.add_edge("match_analysis_agent", "character_recommendation_agent")

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
    # 不碰 character_recommendations、training_plan 等无关字段
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

# 分析子图 state：战绩分析 + 角色推荐相关字段
class AnalysisSubState(TypedDict):
    player_id: str
    match_data: dict
    match_analysis: dict
    character_recommendations: list[dict]

# 策略子图 state：策略 + 构建相关字段
class StrategySubState(TypedDict):
    match_analysis: dict
    memory: dict
    strategy: dict
    build_recommendations: list[dict]

analysis_subgraph = StateGraph(AnalysisSubState)
# ... 添加 match_analysis_agent, character_recommendation_agent ...

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
def match_history_tool(player_id: str, limit: int = 20, game: str = "general") -> dict:
    """
    查询玩家最近 N 场对局的详细数据。

    使用时机：用户问到"最近战绩""胜率""表现如何""为什么一直输"时调用。
    不要使用时机：用户只问角色推荐或出装建议，没有提到战绩分析时。

    返回内容：每局的胜负、KDA、参团率、经济、时长等字段。
    限制：最多返回最近 50 场。game 参数默认 "general"。

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

**确实区分不了**。"Alpha出破军伤害爆炸"和"Alpha出Core_Item伤害稳定"的 embedding 可能很接近——两者都是"Alpha + 装备 + 效果描述"的语义结构。

**Embedding Grounding 擅长发现的**：
- 编造完全不存在的概念："Alpha出星空之杖"（装备不存在，向量和任何文档都不接近）
- 跨领域缝合："Alpha学off-role刀出门"（把off-role概念嫁接到damage role，向量偏离damage role装备文档）

**Embedding Grounding 发现不了的**：
- 数值篡改："Alpha出Core_Item + 200 攻击力"（实际上Core_Item +60，但语义结构对）
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
character_recommendation →  < 5ms   （数据匹配）
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
- match_analysis + character_recommendation + rag 并行（三者无相互依赖）
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

**不同游戏的阈值应该一样吗？** 不一样。FPS 游戏中"玩家偏好武器"的信号比 MOBA 中"玩家偏好角色"更稳定（武器选择变化比角色慢）。阈值应该放在游戏配置中，而非写死在 Memory 模块。

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

当前 5 个 Agent（match_analysis / strategy / build / rag / memory），每个都很"薄"——memory_loader 只是读一个 JSON 文件，character_recommendation 只是匹配两个列表。这些逻辑不需要独立的 Agent 节点，合并为 3 个更合适：

```text
当前：                          重来：
match_analysis_agent            analysis_agent（战绩分析 + 数据诊断）
character_recommendation_agent  →    recommendation_agent（角色推荐 + 装备推荐 + 攻略检索）
strategy_agent                  strategy_agent（打法策略 + 训练计划）
rag_agent                  →
build_agent                →
memory_loader                   memory 作为以上所有 Agent 的共享上下文层，而非独立 Agent
```

拆太细的代价：
- 每个 Agent 节点都是一次潜在的 LLM 调用，延迟累加
- Graph 节点太多，LangSmith Trace 看起来很"壮观"但大部分节点做的事情太少
- Agent 间的数据传递依赖 state 字段，字段数膨胀

**但拆得细也有一个好处**（这也是一开始选择细拆的原因）：作为面试项目，细拆能清晰展示 LangGraph 的多节点编排能力。重来一次，我会在"展示清晰度"和"工程务实"之间取一个折中——合并 match_analysis + character_recommendation，保留 strategy/training_plan 独立，把 memory 降级为上下文层。

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
| 角色攻略 | 半结构化：属性表 + 技能说明 + 出装 + 技巧 | **按章节标题切分**（`##`），保留表格完整性 | 语义窗口——表格的 embedding 语义弱，容易丢 |
| 职业复盘/教学 | 长文叙事，逻辑流强 | **语义窗口**（512 token，overlap 128），按段落边界调整 | 固定大小——会切断推理链 |
| 装备/物品解析 | 短条目，每条独立 | **最小单元切分**（每条装备一个 chunk），保留完整字段 | 任何切分——应该以结构化 JSON 存而非向量检索 |

**实现**：

```python
class ChunkingStrategy:
    STRATEGIES = {
        "patch_notes":    SectionalChunker(heading_level=3, min=200, max=800),
        "character_guide": SectionalChunker(heading_level=2, min=300, max=1200),
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

3. **结构化数据不进向量库**。角色属性表、装备数值这类精确查询走 Tool Calling（`character_database_tool`），而非向量检索。向量检索适合"damage role怎么站位"这类开放问题。

4. **入库时标记，检索时过滤**。每个 chunk 携带 `doc_type` 和 `section_title` 元数据。检索时：用户问"当前版本推荐出装"→ 过滤 `doc_type=patch_notes` 优先，`doc_type=pro_review` 作为补充。

---

## 八、代码实现细节

### Q18: 条件路由是怎么实现的？planned_tasks 如何转化为实际执行路径？

**核心机制**：Planner 生成的任务列表通过 `TASK_NODE_MAP` 映射为节点名，Router 按 `EXECUTION_ORDER` 依次检查每个节点是否需要执行。

```python
# router.py — 任务类型到节点的映射
TASK_NODE_MAP = {
    "match_analysis": "match_analysis_agent",
    "character_recommendation": "character_recommendation_agent",
    "build_recommendation": "build_agent",
    "rag_lookup": "rag_agent",
    "memory_lookup": "memory_loader",
    "training_plan": "strategy_agent",
}

# DAG 中的固定执行顺序
EXECUTION_ORDER = [
    "memory_loader",
    "match_analysis_agent",
    "character_recommendation_agent",
    "rag_agent",
    "build_agent",
]

# 条件路由函数示例
def route_after_planner(state: GameCoachState) -> str:
    required = _collect_required_nodes(state)  # 从 planned_tasks 提取需要的节点
    if not required:
        return "strategy_agent"  # 全部跳过 → 直接到策略合成
    if "memory_loader" in required:
        return "memory_loader"   # 有任务需要 → 先加载玩家画像
    return required[0]
```

**关键设计决策：为什么 routing_decisions 在 planner 节点中计算而非条件边函数？**

这是一个实际踩过的坑。LangGraph 的条件边函数（如 `route_after_planner`）**不能修改 state**——它们只能读取 state 并返回下一个节点名。如果在条件边函数中写 `state["routing_decisions"] = ...`，这个修改不会被持久化。

解决方案：在 `planner` 节点中预计算路由决策并写入返回的 dict：

```python
def planner(state: GameCoachState) -> GameCoachState:
    result = create_llm_planner(state)  # LLM 生成任务
    # 预计算路由决策（条件边只能读 state，不能写）
    planned = result.get("planned_tasks", [])
    required = {TASK_NODE_MAP[t["task_type"]] for t in planned}
    decisions = {n: ("execute" if n in required else "skip") for n in EXECUTION_ORDER}
    result["routing_decisions"] = decisions
    return result
```

---

### Q19: LLM 结构化输出失败了怎么办？JSON 解析是怎么做的？

**DeepSeek 不支持 `with_structured_output`**（`response_format: json_object` 不可用）。实际做法是：在 prompt 中要求 LLM 输出 JSON，然后手动提取和解析。

```python
def _extract_json(text: str) -> str:
    # 尝试匹配 ```json ... ``` 代码块
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    # 尝试匹配裸 JSON { ... }
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return match.group(0).strip()
    return text.strip()
```

**三层容错**：

```python
try:
    result = llm.invoke(prompt)
    text = result.content
    json_str = _extract_json(text)
    data = json.loads(json_str)          # 第1层：JSON 解析
    validated = PlannedTaskList(**data)   # 第2层：Pydantic 校验
    return validated
except json.JSONDecodeError:
    # LLM 输出了非 JSON → 回退规则版
    return _fallback_planner(state)
except ValidationError:
    # JSON 格式对但字段类型/值不合法 → 回退
    return _fallback_planner(state)
except Exception:
    # 网络错误、API 限流等 → 回退
    return _fallback_planner(state)
```

**面试要点**：不是所有 LLM 都支持 structured output。JSON prompt + Pydantic 后校验是更通用的方案，兼容任何 LLM API。

---

### Q20: 为什么 Embedding 从 HuggingFace 换成了 DashScope？

**踩过的坑**：最初用 `all-MiniLM-L6-v2` 做本地 embedding。`sentence-transformers` 首次加载时会从 huggingface.co 下载模型（~80MB），在境内服务器上 `getaddrinfo failed`——huggingface.co 被墙。

切换到达摩盘 DashScope (`text-embedding-v2`)：
- API 调用，不需要本地下载模型
- 不需要科学上网，国内网络直连
- 通过 `DASHSCOPE_API_KEY` 环境变量配置

```python
# config/llm.py
def get_embeddings():
    global _DASH_SCOPE_EMBEDDINGS
    if _DASH_SCOPE_EMBEDDINGS is None:
        from langchain_community.embeddings import DashScopeEmbeddings
        settings = get_settings()
        if not settings.dashscope_api_key:
            return None  # 返回 None，调用方（RAG）检测后返回空结果
        _DASH_SCOPE_EMBEDDINGS = DashScopeEmbeddings(
            model="text-embedding-v2",
            dashscope_api_key=settings.dashscope_api_key,
        )
    return _DASH_SCOPE_EMBEDDINGS
```

**通用教训**：做 embedding 的技术选型时，先确认目标部署环境的网络可达性。境内部署 → 优先国内云厂商的 embedding API（阿里/百度/讯飞），境外部署 → OpenAI/HuggingFace。

---

## 九、线上常见坑与解决方案

### Q21: 工具调用失败导致的死循环怎么防？

**问题场景**：如果工具每次返回 `status: "unavailable"`，Agent 可能不断重试——每次调用失败 → LLM 决定"再试一次" → 又失败 → 无限循环。

**本项目的防循环策略**（Plan-and-Execute 模式下天然免疫）：

因为是 Planner 先生成完整任务列表再执行，而非 ReAct 循环——不存在 LLM 自主决定"再调一次工具"的路径。每个工具只被调用一次。

但如果引入 ReAct（比如 rag_agent 检索不理想时改写 query 重试），需要加：

```python
MAX_RETRIES = 3
MAX_STEPS = 10          # Agent 总步数上限
MAX_TIME_BUDGET = 30    # 超时秒数

class AgentLoopGuard:
    def __init__(self):
        self.steps = 0
        self.tool_calls = {}  # tool_name → call_count

    def check(self, tool_name: str) -> bool:
        self.steps += 1
        if self.steps > MAX_STEPS:
            raise LoopLimitExceeded("Agent step limit reached")
        self.tool_calls[tool_name] = self.tool_calls.get(tool_name, 0) + 1
        if self.tool_calls[tool_name] > MAX_RETRIES:
            raise LoopLimitExceeded(f"{tool_name} called {MAX_RETRIES} times")
        return True
```

**当前项目的防护**：每个 Tool 返回 `{"status": "unavailable", "reason": "..."}`，下游 Agent 的 fallback 逻辑检测到 `unavailable` 后直接使用降级输出，不重试。

---

### Q22: LLM Key 没配怎么办？系统能跑吗？

**能跑。** 这是从 Stage 1 就坚持的设计原则——LLM 是增强而非必需。

每个 LLM 调用点都有 fallback：

```python
# planner — LLM 不可用时用关键词匹配规则
def create_llm_planner(state):
    llm = get_chat_model()
    if llm is None:
        return _fallback_planner(state)  # 规则版：关键词 → 默认4任务

# strategy — LLM 不可用时用硬编码策略模板
def create_llm_strategy(state):
    llm = get_chat_model()
    if llm is None:
        return _fallback_strategy(state)  # 硬编码：3天计划 + 通用建议

# response — LLM 不可用时用字符串模板拼装
def create_llm_response(state):
    llm = get_chat_model()
    if llm is None:
        return _fallback_response(state)  # 模板版：f-string 拼装各模块输出
```

`get_chat_model()` 在 `DEEPSEEK_API_KEY` 未配置时返回 `None`（不抛异常）：

```python
def get_chat_model(temperature=0.2) -> Optional[ChatOpenAI]:
    settings = get_settings()
    if not settings.deepseek_api_key:
        logger.warning("DEEPSEEK_API_KEY 未配置，LLM 调用将使用 fallback 模式。")
        return None
    return ChatOpenAI(...)
```

**实测**：没有 LLM 时，系统仍能输出带数据依据的结构化建议——只是建议是模板化的而非个性化的。有 LLM 时，同一条用户输入可以生成针对性的策略和动态天数的训练计划。

---

### Q23: 条件边函数修改 state 不生效——怎么发现的？

**问题**：最初在 `route_after_planner`（条件边函数）中直接写 `state["routing_decisions"] = decisions`，测试中 `routing_decisions` 始终为 `{}`。

**原因**：LangGraph 传给条件边函数的 state 是**快照副本**，函数内的修改不会被写回。条件边函数只能返回字符串（下一个节点名），不能返回 state 更新。

**调试过程**：
1. 加 `print(state["routing_decisions"])` 在条件边函数末尾 — 有值
2. 在下一个节点的 `state.get("routing_decisions")` — 是 `{}`
3. 查 LangGraph 源码确认：条件边函数返回值是 `str`，不参与 state merge

**修复**：将 `routing_decisions` 的计算移到 `planner` 节点（正常节点，可以返回 dict 更新 state）：

```python
# nodes.py — planner 节点返回 routing_decisions
def planner(state: GameCoachState) -> GameCoachState:
    result = create_llm_planner(state)
    # 在这里计算路由决策（节点可以改 state，条件边不行）
    decisions = _compute_routing(result["planned_tasks"])
    result["routing_decisions"] = decisions
    return result
```

**教训**：LangGraph 中只有节点函数能修改 state。条件边、Map-Reduce 的 mapper 都不能。这是 LangGraph 的核心概念，但文档不够显眼，容易踩坑。

---

### Q24: Windows GBK 终端输出中文/Emoji 崩溃

**问题**：`print("📊 执行路径: ...")` 在 Windows 终端报 `UnicodeEncodeError: 'gbk' codec can't encode character '\U0001f4ca'`。

**原因**：Windows 中文版终端默认编码是 GBK，不支持 emoji 和部分 Unicode 字符。

**修复**：
1. 所有 `print()` 输出移除 emoji，用 ASCII 替代：`📊` → `[Path]`、`⚠` → `[!]`、`✅` → `OK:`
2. 日志用英文（`logging` 模块输出到 stderr，不受 GBK 影响）
3. `final_response` 中包含的中文由 LLM 生成，通过 FastAPI JSON 响应返回时不受影响

**通用方案**（如果必须输出 Unicode）：
```python
import sys
sys.stdout.reconfigure(encoding='utf-8')  # Python 3.7+
```

或在终端执行前设置：`set PYTHONIOENCODING=utf-8`

---

### Q25: FAISS 索引加载失败导致 RAG 静默返回空

**问题**：第一次运行时 RAG 检索始终返回 `[]`，没有报错。

**排查链**：
1. `rag_agent` 节点 → `docs = retrieve(query)` → `[]`
2. `retriever.py` → `get_or_build_index()` 抛异常被 catch
3. `indexer.py` → `get_embeddings()` 返回 `None`（DashScope key 未配）
4. 但异常在 `retriever` 层被 `except Exception: return []` 吞掉了

**修复**：在 `indexer.py` 的 `_build_new_index` 中加显式检查：

```python
embeddings = get_embeddings()
if embeddings is None:
    raise RuntimeError("Embedding 模型不可用（DASHSCOPE_API_KEY 未配置）")
```

同时 `guide_rag_tool` 返回中加 `status: "unavailable"` 让上游知道不是"没搜到"而是"搜不了"。

**教训**：`except Exception: return []` 是双刃剑——保证了系统不崩溃，但也掩盖了配置错误。应该在工具返回中用 `status` 字段区分"无结果"和"不可用"。

---

### Q26: .env 没有自动加载——为什么需要显式调 load_dotenv？

**问题**：`python-dotenv` 已在依赖中，但 `os.getenv("DEEPSEEK_API_KEY")` 始终返回 `None`。

**原因**：`load_dotenv()` 不是自动调用的——需要显式执行。最初 `settings.py` 只调了 `os.getenv()`，没调 `load_dotenv()`。

**修复**：
```python
# config/settings.py — 在模块加载时自动调用
from dotenv import load_dotenv

_env_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env")
load_dotenv(_env_path, override=False)  # override=False: 系统环境变量优先
```

**教训**：新成员 clone 项目后，按照 README 配了 `.env` 但运行报错——最容易的诊断就是"你是不是没调 load_dotenv"。把这个调用放在 settings 模块顶层，对所有 import settings 的代码透明。

---

### Q27: Planner 输出 JSON 格式不稳定——DeepSeek 有时返回 markdown 包裹的 JSON

**观察**：同样的 prompt，DeepSeek 有时返回裸 JSON，有时返回 ` ```json { ... } ``` `。

**解决方案**：`_extract_json()` 函数先尝试匹配 markdown 代码块，再尝试匹配裸 JSON：

```python
def _extract_json(text: str) -> str:
    # 优先：```json ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    # 兜底：匹配第一个 { ... }
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return match.group(0).strip()
    return text.strip()
```

**为什么还要 `\{[\s\S]*\}` 兜底**：LLM 有时会在 JSON 前后加一两句解释（"以下是任务拆解：\n{...}"），正则只摘 JSON 部分。

**教训**：不要假设 LLM 的输出格式是稳定的。JSON 提取 + Pydantic 校验 = 双重保险。
