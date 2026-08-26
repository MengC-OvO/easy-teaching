# Architecture

EasyTeaching 的生产 Runtime 只使用一条统一 Main ReAct 主线。旧 Intent Router、固定
Specialist、Skill、审批图和同步单 Tool ReAct 已删除。

Main 可在结构化决策中产生 `task_type`，但它只用于日志和测评。图不锁定
该类型，也不使用它控制 Tool 权限或执行路径；每一轮仍由 Main 根据当前
请求、上下文和 Observation 决定下一步。

活动方案的强制安全检查使用独立的 `requires_activity_safety` 布尔不变量，
避免把完整任务分类表变成路由器。它只表达当前回答是否必须先具有安全检查
Observation，不决定其他 Tool 或 Worker 的选择。

Tool 的重复控制来自注册时的生命周期元数据，而不是请求关键词。安全检查在每个
用户请求中最多成功执行两次：一次初检和一次修改后复检；完全相同的参数仍只执行
一次。最终活动文本还必须与某次实际检查的内容指纹一致。新的教师消息会开始新一
轮并重置计数，因此老师后续继续修改不受上一轮次数影响。

完成契约不是由 Main 自报。每个需要审批的 Tool 在 Registry 注册高置信度操作
别名，运行时用统一策略从当前请求确定必须完成的 Tool 名称，并写入本轮 State。
Main 仍可自由选择中间 ReAct 步骤，但在对应 Tool 生成冻结审批预览前不能以普通
草稿结束。新增保存、发送或发布能力时只需注册别名，不修改 Graph 特判；澄清
缺失字段仍然允许。更严格的生产客户端也可在入口直接提供结构化操作意图。

完整生成草稿保存在 `conversation_run_results`。每次 Main 决策前，ContextManager
从 PostgreSQL 投影最近八个可复用草稿的稳定编号、相对版本、可读标题和
`source_request_id`，以及最近已批准保存返回的 `record_id`。因此“保存这个”或
“保存上一个版本”由 Main 选择明确的可信 ID，保存 Tool 在审批前取回并冻结完整
内容；“导出它”可直接使用最近保存的记录 ID。长草稿不再依赖最近对话中最多
800 字符的文本副本，澄清与错误消息也不会进入可复用产物列表。
读取完整草稿成功后，Graph 会从所有成功 Observation 构造受信任的
`loaded_draft_references` 映射。A、B 等多个版本都保留独立的 result key、
`source_request_id` 和标题，不再用单一“当前草稿”覆盖前一个选择。

## Request lifecycle

```mermaid
sequenceDiagram
    participant UI as Web workspace
    participant API as FastAPI
    participant DB as PostgreSQL
    participant G as Main ReAct LangGraph
    participant C as Tool / Worker

    UI->>API: POST session message
    API->>DB: create idempotent run + run_started
    API-->>UI: 202 accepted
    API->>G: await graph.ainvoke
    loop one current decision per turn
        G->>G: Main selects current call or batch
        G->>G: validate name, schema, permission, concurrency and budget
        G->>C: one Tool / concurrent Tools / Worker fan-out
        C-->>G: immutable Observation
    end
    G->>G: Main creates final teacher draft
    G->>DB: checkpoint, context and scoped memory
    API->>DB: persist public draft + ordered SSE events
    UI->>API: replay SSE and fetch draft
```

Main 不预先生成完整执行计划。每一轮只选择当前一个调用或当前可以安全并行的
批次；有依赖的调用等待 Observation 后由下一轮重新决定。

## Production graph

```text
initialize
  → main_react
  → validate_decision
      ├─ single_tool
      ├─ parallel_tools
      ├─ run_worker (LangGraph Send fan-out)
      ├─ decision_feedback
      ├─ clarification
      ├─ prepare_approval
      └─ finalize_draft

single/parallel/worker/feedback
  → merge_observations
  → main_react

finalize_draft/clarification/prepare_approval
  → context_update
  → long_memory_update
  → END
```

只有 Main 可以产生教师可见草稿。Tool、MCP 适配器和 Worker 只能返回
Observation，不能直接写主 State 或执行业务副作用。

## Execution choices

| 当前需要 | 执行方式 |
| --- | --- |
| 一个普通或单一深度任务 | Main 跨多轮调用一个 Tool |
| 多个独立简单查询 | `parallel_tools` 内异步并发 |
| 多个独立深度研究 | 固定 Worker Profile 并行 fan-out |
| 有依赖或独立性不确定 | 先执行一个前置调用，下一轮继续 |
| 信息足够 | 结束循环并生成草稿 |

两个 Worker Profile 分别限制为课程/安全研究，以及 teacher-scoped 班级和记录
研究。它们有独立模型上下文、固定只读工具白名单和最大 3 轮预算。

## Code boundaries

### `app/api`

负责认证、session ownership、幂等、session busy、公开结果、SSE replay 和恢复。
生产 Runtime 只接受 PostgreSQL，使用 `AsyncPostgresSaver`、异步 SQLAlchemy
Store 和 `await graph.ainvoke()`。API、恢复、SSE 查询、模型 HTTP 和外部 HTTP
工具均通过原生 `await`，不再把整张图放入工作线程。

### `app/workflows/main_react_graph.py`

负责节点、条件边、Worker `Send`、Observation reducer、安全合并和循环预算。
并行分支不共享可变字典，只向 `pending_observations` 追加；唯一 merge 节点更新
主 Observation map。

### `app/agents`

`MainReActAgent` 只提出结构化当前决定；`MainDecisionValidator` 用代码重做权限、
隐私字段、并发安全和重复调用检查；`BoundedWorkerRunner` 在小上下文内运行
受限 ReAct。

### `app/tools`

`ToolRegistry` 是真实执行边界。每个 Tool 包含 Pydantic 输入/输出、领域、风险、
权限、超时和并发标记。可信 `teacher_id/class_id` 由图注入，模型参数不能扩大
本地读取范围。

### `app/services`

保留 SQLAlchemy Store、模型 Provider/retry、Chroma + BM25 hybrid RAG、reranker、
引用、短上下文压缩和长期记忆。它们是普通 Python 模块，不被隐藏到图框架中。

## State and persistence

| State | Location | Purpose |
| --- | --- | --- |
| Graph checkpoint | LangGraph AsyncPostgresSaver | 恢复精确节点和同一 thread |
| API session/run/event | SQLAlchemy PostgreSQL | 幂等、状态和 SSE replay |
| Current ReAct state | GraphState | 当前决定、Observation、预算和 trace |
| Short context | checkpointed GraphState | 有界 recent turns 和摘要 |
| Long-term memory | SQLAlchemy PostgreSQL | teacher/class scoped durable memory |
| Knowledge index | Chroma + BM25 | 政策与 EYLF 检索 |

同一个 checkpoint thread 会跨消息保留 context，但 `initialize` 为每个新
`request_id` 重置临时 Observation、计数和重复调用记录，并记录本轮 trace/citation
起点，避免 SSE 重放上一轮运行数据。

## Current safety boundary

生产主图负责生成草稿和准备受控操作，但模型生成阶段不会直接写数据库或上传
文件。`save_observation`、`save_educational_record`、`export_records` 和
`upload_export_to_google_drive` 会先冻结已经验证的参数并进入
`waiting_for_approval`；只有教师批准后，API 才原子执行同一份冻结参数。重复或
并发批准不会执行两次。系统不提供真实消息发送能力。

本地隐私 Gateway 已接入但默认关闭。最终端到端测评未通过 release gate，因此
真实儿童与家庭数据仍不在当前项目范围内；在模型按生产预脱敏格式重新训练并通过
独立测评前，只能使用合成数据或彻底脱敏数据。
