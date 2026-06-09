# GameCoach AI 第一阶段测试文档

## 1. 测试目标

验证第一阶段代码是否能在 LangSmith Studio / LangGraph 本地开发服务中加载，并完成一次从用户问题到最终成长建议的基础工作流。

第一阶段不测试真实 LLM、真实游戏 API、真实 RAG，只测试 LangGraph 结构、状态传递、Mock 数据分析和最终回答生成。

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

如果你已经在系统环境变量中配置了以下变量，可以跳过 `.env` 文件：

```text
DEEPSEEK_API_KEY
LANGCHAIN_API_KEY
LANGCHAIN_TRACING_V2
LANGCHAIN_PROJECT
```

如果没有配置系统环境变量，可以复制示例文件：

```powershell
Copy-Item .env.example .env
```

在 `.env` 中填入：

```text
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=你的 LangSmith API Key
LANGCHAIN_PROJECT=gamecoach-ai
DEEPSEEK_API_KEY=你的 DeepSeek API Key
DEEPSEEK_MODEL=deepseek-chat
```

## 3. 启动 LangGraph 开发服务

在项目根目录执行：

```powershell
.venv\Scripts\langgraph.exe dev
```

预期结果：

- 控制台能识别 `langgraph.json`。
- 控制台能加载 `gamecoach` graph。
- 输出 LangGraph API / Studio 相关访问地址。

## 4. LangSmith Studio 测试步骤

1. 打开 LangSmith Studio。
2. 选择本地运行的 `gamecoach` graph。
3. 创建一次新的 Thread。
4. 输入以下 JSON：

```json
{
  "user_message": "我最近胜率很低，玩射手总是团战暴毙，怎么上分？",
  "player_id": "player_001",
  "game": "moba"
}
```

5. 运行 graph。
6. 观察每个节点的输入、输出和状态变化。

## 5. 预期节点执行顺序

```text
input_normalizer
planner
memory_loader
match_analysis_agent
hero_recommendation_agent
strategy_agent
response_agent
evaluation_logger
```

## 6. 预期状态结果

### `planner`

预期输出：

- `intent` 为 `improve_rank`。
- `planned_tasks` 至少包含：
  - `memory_lookup`
  - `match_analysis`
  - `hero_recommendation`
  - `strategy_generation`

### `memory_loader`

预期输出：

- `memory.player_id` 为 `player_001`。
- `memory.favorite_heroes` 包含 `后羿`、`狄仁杰`。
- `memory.goals` 包含 `上钻石`。

### `match_analysis_agent`

预期输出：

- `match_analysis.metrics.matches` 为 `20`。
- `match_analysis.metrics.win_rate` 约为 `0.35`。
- `match_analysis.metrics.avg_deaths` 大于 `6`。
- `match_analysis.weaknesses` 包含死亡偏高或参团不足相关诊断。

### `hero_recommendation_agent`

预期输出：

- `hero_recommendations` 包含 `狄仁杰`、`孙尚香`、`戈娅`。
- 每个推荐英雄包含 `fit_reasons`。

### `strategy_agent`

预期输出：

- `strategy.priorities` 包含 `少死`、`固定英雄池`、`提高中期参团率`。
- `training_plan.duration_days` 为 `3`。
- `training_plan.daily_tasks` 包含 3 天训练任务。

### `response_agent`

预期输出：

- `final_response` 包含：
  - `结论`
  - `数据依据`
  - `优先改进`
  - `推荐英雄`
  - `3 天训练计划`

### `evaluation_logger`

预期输出：

- `metrics.planner_task_count` 为 `4`。
- `metrics.tool_call_success_rate` 为 `1.0`。
- `metrics.has_match_analysis` 为 `true`。
- `metrics.has_hero_recommendations` 为 `true`。
- `metrics.has_training_plan` 为 `true`。

## 7. 本地 CLI 测试

如果不使用 Studio，也可以运行：

```powershell
.venv\Scripts\python.exe -m gamecoach.main
```

预期结果：

- 终端输出一段中文成长建议。
- 内容包含胜率、KDA、死亡、参团率、推荐英雄和 3 天训练计划。

## 8. 常见问题

### 找不到 `gamecoach` 包

确认已经执行：

```powershell
.venv\Scripts\python.exe -m pip install -e .[dev]
```

或确认当前目录是项目根目录。

### Studio 无法加载 graph

检查：

- `langgraph.json` 是否在项目根目录。
- `graphs.gamecoach` 是否指向 `./src/gamecoach/graph/workflow.py:graph`。
- Python 版本是否为 3.11。
- 是否安装了 `langgraph-cli[inmem]`。

### 看不到 LangSmith Trace

检查：

- `.env` 中是否设置 `LANGCHAIN_TRACING_V2=true`。
- `LANGCHAIN_API_KEY` 是否正确。
- `LANGCHAIN_PROJECT` 是否为 `gamecoach-ai`。

如果这些变量已经配置在系统环境变量中，不需要再写入 `.env`。

### DeepSeek Key 是否必须写入 `.env`

不必须。项目会优先读取系统环境变量中的 `DEEPSEEK_API_KEY`。

`.env` 只适合本地临时开发，不建议提交真实 Key。
