# GameCoach AI 测试文档

## 1. 测试目标

验证代码是否能在 LangSmith Studio / LangGraph 本地开发服务中加载，并完成一次从用户问题到最终成长建议的完整工作流。

## 2. 环境准备

要求：
- Python 3.11
- LangGraph CLI
- LangSmith API Key

创建虚拟环境：
```powershell
python -m venv .venv
```

安装依赖：
```powershell
.venv\Scripts\python.exe -m pip install -e .[dev]
```

在 `.env` 中填入：
```text
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=你的 LangSmith API Key
LANGCHAIN_PROJECT=gamecoach-ai
DEEPSEEK_API_KEY=你的 DeepSeek API Key
DEEPSEEK_MODEL=deepseek-chat
DASHSCOPE_API_KEY=你的阿里云 DashScope API Key
```

## 3. 启动 LangGraph 开发服务

在项目根目录执行：
```powershell
.venv\Scripts\langgraph.exe dev
```

预期：控制台识别 `langgraph.json`，加载 `gamecoach` graph。

## 4. LangSmith Studio 测试步骤

1. 打开 LangSmith Studio
2. 选择本地运行的 `gamecoach` graph
3. 创建一次新的 Thread
4. 输入以下 JSON：

```json
{
  "user_message": "我最近胜率很低，怎么上分？",
  "player_id": "player_001"
}
```

5. 运行 graph
6. 观察每个节点的输入、输出和状态变化

## 5. 预期节点执行顺序

```text
input_normalizer
planner
memory_loader
match_analysis_agent
character_recommendation_agent
rag_agent
build_agent
strategy_agent
response_agent
evaluation_logger
```

## 6. 预期状态结果

### `planner`

预期输出：
- `intent` 为 `improve_performance`。
- `planned_tasks` 至少包含：
  - `memory_lookup`
  - `match_analysis`
  - `character_recommendation`
  - `strategy_generation`

### `memory_loader`

预期输出：
- `memory.player_id` 为 `player_001`。
- `memory.favorite_characters` 包含 `Delta`、`Alpha`。
- `memory.goals` 包含 `reach Diamond rank`。

### `match_analysis_agent`

预期输出：
- `match_analysis.metrics.matches` 为 `20`。
- `match_analysis.metrics.win_rate` 约为 `0.35`。
- `match_analysis.metrics.avg_deaths` 大于 `6`。
- `match_analysis.weaknesses` 包含死亡偏高或参团不足相关诊断。

### `character_recommendation_agent`

预期输出：
- `character_recommendations` 包含 `Alpha`、`Bravo`、`Charlie`。
- 每个推荐角色包含 `fit_reasons`。

### `strategy_agent`

预期输出：
- `strategy.priorities` 包含改进优先级列表。
- `training_plan.duration_days` 为 `3`。
- `training_plan.daily_tasks` 包含训练任务。

### `response_agent`

预期输出：
- `final_response` 包含：
  - `结论`
  - `数据依据`
  - `优先改进`
  - `推荐角色`
  - `训练计划`

### `evaluation_logger`

预期输出：
- `metrics.planner_task_count` 为 `4`。
- `metrics.tool_call_success_rate` 为 `1.0`。
- `metrics.has_match_analysis` 为 `true`。
- `metrics.has_character_recommendations` 为 `true`。
- `metrics.has_training_plan` 为 `true`。

## 7. 本地 CLI 测试

```powershell
.venv\Scripts\python.exe -m gamecoach.main
```

预期：终端输出一段中文成长建议，包含胜率、KDA、死亡、参团率、推荐角色和训练计划。

## 8. 常见问题

### 找不到 `gamecoach` 包

确认已执行 `pip install -e .[dev]`，或确认当前目录是项目根目录。

### Studio 无法加载 graph

检查：
- `langgraph.json` 是否在项目根目录
- `graphs.gamecoach` 是否指向 `./src/gamecoach/graph/workflow.py:graph`
- Python 版本是否为 3.11
- 是否安装了 `langgraph-cli[inmem]`

### 回复是英文不是中文

`.env` 中 DEEPSEEK_API_KEY 已配置时，LLM 会根据用户输入语言自动选择回复语言（中文输入 → 中文回复）。如果 LLM Key 未配置，系统在 fallback 模式下也会自动检测用户输入语言。

### 看不到 LangSmith Trace

检查 `.env` 中 `LANGCHAIN_TRACING_V2=true` 和 `LANGCHAIN_API_KEY` 是否正确。
