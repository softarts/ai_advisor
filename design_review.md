# SA Advisor (Strands SDK) 设计与最佳实践审查报告

## 1. 架构总览

`sa_advisor` 是基于 Strands Agents SDK 构建的高级投资顾问模块，与系统原有 `advisor`（LangGraph 架构）保持一致的公共接口，可无缝替换执行。

核心组成模块：
- `sa_advisor/agent.py`: `StrandsAdvisor` 核心调度类，负责意图分发、Agent 实例管理、会话历史维持与输出复核修复。
- `sa_advisor/model_provider.py`: 统一封装基于 `OpenAIModel` 的模型创建与参数配置。
- `sa_advisor/router.py`: 意图路由分类器，利用 Strands 的 `structured_output_model` 特性实现快速免循环结构化分类。
- `sa_advisor/tools.py`: 将 `common.tools.Tool` 动态适配封装为 Strands `@tool` 规范。
- `sa_advisor/eval/runner.py`: 适配 Strands 消息结构的评测与合规检查运行器。

---

## 2. 对照 Strands SDK 最佳实践审查

### 2.1 优秀实践 (Strengths & Alignment)

1. **原生的结构化输出与零 Tool-Use 意图分类 (`router.py`)**：
   - 充分利用了 Strands 的 `agent(text, structured_output_model=IntentResponse)` 机制。
   - 结构化输出直接走模型约束解码/解析逻辑，不触发多轮 agentic tool 循环，既保证了分类精度与稳定性，又降低了延迟和 Token 开销。

2. **工具签名的反射构造与多线程隔离安全 (`tools.py`)**：
   - **参数内省对齐**：Strands SDK 的 `@tool` 内部验证依赖函数的显式参数签名（`inspect.Signature`），而不是简单的通用 `**kwargs`。`tools.py` 通过解析 JSON Schema 动态生成参数签名与默认值，保证了 Strands Pydantic 校验层可以精确内省。
   - **多线程 SQLite 安全**：Strands 内部执行同步工具时会通过 `asyncio.to_thread` 分发给线程池。`tools.py` 在每次工具执行时为当前工作线程创建独立的 `Store` 实例，彻底避免了 SQLite 跨线程使用导致的崩溃问题。

3. **单会话单 Agent 实例与原生消息追踪 (`agent.py`)**：
   - 采用 `_agents: dict[str, Agent]` 对每个 `thread_id` 维护单个 Agent 实例。
   - 完整复用了 Strands 内部的 `agent.messages` 原生会话历史，无需外部手写消息拼接逻辑。
   - 在多轮交互中若意图发生变化，支持原地更新 `system_prompt` 与当前意图允许的工具集注册表（`ToolRegistry`），既保留了多轮上下文，又避免跨意图的工具越权。

4. **防护机制与执行边界控制**：
   - 在 `ask()` 与 `_run_output_review()` 中增加了 `MAX_EXECUTION_TIME`（180s）与 `MAX_REPAIR_ATTEMPTS`（2次），防止因输出复核不满足或 LLM 异常进入死循环。

---

### 2.2 潜在优化与演进建议 (Recommendations)

1. **异步原生化流转 (Async-native API)**：
   - 当前 `ask()` 统一走同步调用 `agent(text)`，内部通过 `run_async` 包装 `ThreadPoolExecutor + asyncio.run()`。
   - *建议*：在未来的服务化演进中，可提供 `ask_async(...)` 方法直接基于 `agent.invoke_async(...)`，降低线程上下文切换开销，避免与外部主事件循环的协调问题。

2. **私有属性依赖的稳定性考虑**：
   - 在 `_get_or_create_agent` 中动态替换 `system_prompt` 与 `tools` 时，触碰了 SDK 的私有属性：
     - `agent._system_prompt_content`
     - `agent._tools = ToolRegistry(...)`
   - *建议*：由于 Strands SDK 在小版本迭代中可能会重构内部属性命名，建议在单元测试中增加针对 SDK 升级的契约测试（Contract Tests），或者在必要时通过重新构建轻量级 Agent 实例（携带现有 `messages`）进行标准重置。

3. **Python 3.14 运行环境下的 HTTP 异步生成器兼容**：
   - 见 `handoff_httpx.md`，底层 `httpcore2` 在 Python 3.14 的生成器终结流程中存在已知缺陷，建议将补丁作为标准前置步骤纳入 `sa_advisor` 的初始化链路中。

---

## 3. 审查结论

`sa_advisor` 整体实现架构清晰，与 Strands SDK 的核心概念（Agent、Tool Registry、Structured Output、Event Loop）高度吻合，尤其在动态参数签名生成和多线程工具隔离方面展现了成熟的工程质量。在合规与稳健性层面上完整实现了数字溯源复核和超时熔断，是一个高质量的 SDK 落地实现。
