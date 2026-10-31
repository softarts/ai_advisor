# Strands AI Advisor

基于 [Strands Agents SDK](https://strandsagents.com/) 的 AI Advisor 实现，与 `advisor/`（LangGraph 版）提供相同的公共接口，但底层架构完全不同。

---

## 1. Strands SDK 架构概述

### Agent（代理）

`strands.Agent` 是核心类。它封装了模型调用、工具执行、多轮历史管理的完整 agentic loop。创建时指定模型、工具列表和系统提示，之后直接 `agent("用户输入")` 即可驱动一轮对话。

```python
from strands import Agent
agent = Agent(model=model, tools=[...], system_prompt="...")
result = agent("你好")          # AgentResult
text   = str(result)            # 提取文本
```

### Tool（工具）

Strands 工具是被 `@tool` 装饰器修饰的 Python 函数，或实现了 `AgentTool` 协议的对象。装饰器从函数的类型注解、docstring、以及可选的 `inputSchema` 参数提取名称、描述和 JSON Schema，无需另外注册。

```python
from strands import tool

@tool(name="my_tool", description="描述", inputSchema={"json": {...}})
def my_tool(**kwargs):
    ...
```

### Session Manager

`strands.session.SessionManager` 可将对话历史持久化到外部存储（如 DynamoDB）。本实现未使用 SessionManager——改为在 `StrandsAdvisor._agents` 字典里为每个 `thread_id` 保留一个 `Agent` 实例，利用其内置的 `agent.messages` 列表来保持多轮历史，进程退出后历史自然消失（与 LangGraph `MemorySaver` 行为一致）。

### Model Provider

`strands.models.openai.OpenAIModel` 对接任何 OpenAI 兼容接口。通过 `client_args` 传入 `base_url` 和 `api_key`，通过 `model_id` 指定模型名称。

```python
from strands.models.openai import OpenAIModel
model = OpenAIModel(
    client_args={"api_key": "...", "base_url": "https://..."},
    model_id="your-model",
)
```

### Intervention（干预）

Strands 支持在 agentic loop 的不同阶段注入 `InterventionHandler`，可在模型调用前后修改消息、工具结果或终止循环。本实现暂未使用 intervention，output review 在 `agent()` 返回后用 Python 代码完成。

---

## 2. 本实现架构

| 维度 | LangGraph 版（`advisor/`） | Strands 版（`strands/`） |
|------|--------------------------|------------------------|
| 核心执行单元 | `StateGraph` 节点/图 | `Agent` agentic loop |
| 历史管理 | `MemorySaver`（checkpoint 存 SQLite/内存） | `agent.messages`（每线程一个 `Agent` 实例） |
| 工具注册 | `langchain_core` tool schema，`httpx` 直调 LLM | `@tool` 装饰器，`OpenAIModel` |
| 工具包装 | `Tool.api_schema()` → 直接发 HTTP | `make_strands_tool()` → `DecoratedFunctionTool` |
| 系统提示传递 | 每次 LLM 调用时注入 `messages[0]` | `Agent(system_prompt=...)` 构造参数 |
| Output Review | `_output_review` 节点，条件边触发 repair | `_run_output_review()` 方法，repair 时临时创建无工具 `Agent` |
| 多轮对话 | `thread_id` → `configurable` → `MemorySaver` | `thread_id` → `_agents[thread_id]`（`Agent` 实例字典）|
| 长期记忆 | `InMemoryStore` | `_memory` 字典（内存） |
| LLM 供应商 | `httpx` 直接调用 OpenAI 兼容端点 | `strands.models.openai.OpenAIModel` |

---

## 3. 共用模块说明

以下模块来自现有实现，Strands 版直接 import，**不做任何修改**：

| 模块 | 用途 |
|------|------|
| `pa_analytics.Store` | 数据库访问：导入报告、查询数据 |
| `pa_analytics.registry` | 工具规格定义（schema、函数） |
| `advisor.tools` | `Tool`、`create_tools`、`execute`（schema 验证 + 执行） |
| `advisor.router` | 意图分类、模板加载、`INTENT_PROFILES` |
| `advisor.output_review` | `check_numeric_provenance`、`check_product_policy`、`CheckContext` |
| `advisor.checklist_config` | `GLOBAL_CHECKLIST`、`INTENT_CHECKLISTS`、`GLOBAL_CHECKS` |
| `advisor.llm_settings` | Token 预算和超时常量 |
| `prompts/` | Compose 模板、Router 提示、Output Review 模板 |

---

## 4. 安装依赖

```bash
pip install 'strands-agents[openai]'
# 其他必要依赖（已包含在 requirements.txt）
pip install prompt_toolkit pycountry httpx
```

> 推荐使用项目 venv：
> ```bash
> /path/to/.venv-py314/bin/pip install 'strands-agents[openai]'
> ```

---

## 5. 环境变量

| 变量 | 说明 |
|------|------|
| `LLM_BASE_URL` | OpenAI 兼容 API 地址，如 `https://api.openai.com/v1` |
| `LLM_API_KEY` | API 密钥 |
| `LLM_MODEL` | 模型名称，如 `gpt-4o` |
| `APP_TIMEZONE` | 时区，默认 `Asia/Singapore` |

在项目根目录创建 `.env` 文件：

```
LLM_BASE_URL=https://your-api-endpoint/v1
LLM_API_KEY=your-key
LLM_MODEL=your-model-name
APP_TIMEZONE=Asia/Shanghai
```

---

## 6. 运行方式

```bash
# 从项目根目录
python -m strands --db data/pa.db --raw-dir data/raw

# 指定用户 ID
python -m strands --db data/pa.db --raw-dir data/raw --user-id alice
```

REPL 支持以下命令：

| 命令 | 说明 |
|------|------|
| `/import-report [路径]` | 导入报告文件 |
| `/history` | 查看本轮对话历史 |
| `/clear` | 清空本轮对话（新建 thread_id） |
| `/remember 键=值` | 保存长期记忆 |
| `/recall` | 查看所有长期记忆 |
| `/help` | 显示帮助 |
| `/exit` 或 `/quit` | 退出 |

---

## 7. 测试

```bash
# 从项目根目录运行
pytest strands/tests/ -x -v

# 使用项目 venv
/path/to/.venv-py314/bin/pytest strands/tests/ -x -v
```

测试不需要 LLM API Key，所有 LLM 调用均被 mock。当 `strands-agents` 未安装时，相关测试自动跳过（`pytest.mark.skipif`）。

---

## 8. 目录结构

```
strands/
├── __init__.py          # 包声明
├── __main__.py          # REPL 入口（python -m strands）
├── agent.py             # StrandsAdvisor 核心实现
├── model_provider.py    # OpenAIModel 工厂
├── prompts.py           # 从 advisor.router 再导出的提示助手
├── tools.py             # advisor.tools.Tool → Strands @tool 包装
├── tests/
│   ├── test_agent.py    # StrandsAdvisor 单元测试
│   ├── test_tools.py    # 工具包装测试
│   └── test_output_review.py  # Output review 逻辑测试
└── README.md            # 本文件
```
