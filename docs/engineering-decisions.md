# Engineering decisions and resolved defects

This file records behavior changes and their engineering rationale. Git remains
the detailed change history.

## 2026-08-25 — Replace overlapping tools and model-named dependencies

**Problem:** Class-profile and long-memory reads overlapped, public search had no
trusted source boundary, and `needs=["model_generated_name"]` could fail only
because two model-generated labels differed. Observation and educational-record
writes also lacked a complete scoped approval path.

**Decision:** Register eight domain tools with trusted teacher/class scope. Use
one RAG tool with standard/deep modes, keep teacher preferences in automatic
context, and handle dependent work as separate Main decisions. Only two
independent multi-step research tasks may use bounded read-only Workers. Writes
freeze validated arguments in PostgreSQL and require an atomic teacher approval.

**Why:** Tool names now describe non-overlapping business capabilities. Code
enforces identity, schema, permission, concurrency and approval invariants while
the model decides only which valid action is useful next.

## 2026-08-08 — Replace fixed specialists with one bounded Main ReAct loop

**Problem:** Planning、Policy、Documentation 和 Family 的顶层固定分支让跨领域
任务难以组合。例如活动计划通常同时需要班级上下文、EYLF 和外部环境信息。

**Decision:** 新生产 Runtime 只运行 Main ReAct。Main 每轮决定当前一个调用或
当前独立批次；普通调用可并发，只有多个独立深度研究才使用固定 Worker。所有
Worker 只返回 Observation，最后草稿只能由 Main 生成。

**Why:** 保留 ReAct 的观察后再决定能力，同时让真正独立的 I/O 获得并发收益。
代码注册表、参数校验、权限、冲突规则和预算保证安全，不把正确性寄托在 Prompt。

## 2026-08-25 — Replace semantic regex routing with a structured task contract

**Problem:** 用 `activity` / `save` 等关键词决定安全检查或写入意图，会漏掉
同义表达、多语言表达和否定句。

**Decision:** Main 的结构化决策显式返回 `task_type`，但它只作为日志和测评元数据，
不在运行内锁定，也不用于 Tool 授权或执行路由。Validator 仅使用 Tool 权限、
Observation、冲突规则和预算来验证执行。

活动方案的安全检查从任务分类中拆出，使用专门的
`requires_activity_safety` 信号保护最终输出，不扩散成通用任务路由。
显式写入请求不再由正则强制，真正的写操作仍受冻结参数和人工审批保护。

**Why:** 这将语义理解和逐步决策留给 Main，同时保留低成本的任务分布观测；
确定性代码只负责权限、审批和运行预算。

## 2026-08-09 — Migrate the production path to PostgreSQL-only async I/O

**Problem:** 生产 API 原先通过 `asyncio.to_thread(graph.invoke)` 包裹同步 Store、
同步 `PostgresSaver` 和同步图，长期高并发会让每个运行占用工作线程。

**Decision:** 生产只支持 PostgreSQL，使用异步 SQLAlchemy Store、
`AsyncPostgresSaver`、异步图节点、`graph.ainvoke/aget_state`、异步模型 HTTP、外部
HTTP 和 API/SSE 数据访问。删除生产 SQLite fallback 和同步 checkpoint wiring。

**Why:** I/O 等待不再占用 Web 工作线程；数据库连接池、事务、唯一约束、幂等和
同 session 串行继续保证并发安全。

Chroma PersistentClient、BM25 和可选 Cross-Encoder 只提供同步本地接口，因此
异步 Retriever 将这些调用 offload 到受控工作线程；Embedding 与 MCP 网络 I/O
使用原生 async client。这里的目标是 Web 事件循环不被阻塞，而不是把纯同步依赖
伪装成原生 async。

## 2026-08-02 — Recover from an early Planning final answer

**Observed behavior:** the Planning model loaded `activity_planning` and then
returned a final answer before calling the Skill's required
the class-context and knowledge-retrieval tools. The code-level guard
correctly rejected the unsupported answer, but it ended the whole request with
`skill_requirements_missing`.

**Decision:** required-tool enforcement remains deterministic, but a premature
final answer is now recoverable. The executor appends a
`skill_requirements_check` observation containing the exact missing tool names
and routes back to the Agent. The Agent must call the missing tools before a
later final answer can pass. Repeated refusal still ends safely at the existing
step budget.

**Why:** model instructions improve behavior but do not guarantee it. Code
continues to own the invariant, while the ordinary model mistake no longer
causes an avoidable user-visible failure.

## 2026-08-02 — Treat greeting-only messages as clarification

**Observed behavior:** a bare greeting such as `hi` could be over-interpreted
by the model router as an educator task and enter Activity Planning.

**Decision:** a small deterministic pre-router recognises greeting-only input
and returns `unknown` with `needs_clarification=true`. A greeting followed by a
real request still goes to the model router. The clarification path now creates
a public non-draft assistant response, allowing API and web clients to display
the question normally.

**Why:** greetings contain no task evidence. The system should ask what the
teacher wants rather than spend model/tool calls on a guessed execution path.
