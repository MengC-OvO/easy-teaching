# Engineering decisions and resolved defects

This file records behavior changes that are useful during later review and
interview preparation. Git remains the detailed change history.

## 2026-08-08 — Replace fixed specialists with one bounded Main ReAct loop

**Problem:** Planning、Policy、Documentation 和 Family 的顶层固定分支让跨领域
任务难以组合。例如活动计划通常同时需要班级上下文、EYLF 和外部环境信息。

**Decision:** 新生产 Runtime 只运行 Main ReAct。Main 每轮决定当前一个调用或
当前独立批次；普通调用可并发，只有多个独立深度研究才使用固定 Worker。所有
Worker 只返回 Observation，最后草稿只能由 Main 生成。

**Why:** 保留 ReAct 的观察后再决定能力，同时让真正独立的 I/O 获得并发收益。
代码注册表、依赖校验和预算保证安全，不把正确性寄托在 Prompt。

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
`get_class_profile` and `align_to_eylf_outcomes` tools. The code-level guard
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
teacher wants rather than spend model/tool calls on a guessed workflow.
