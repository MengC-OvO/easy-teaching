# Engineering decisions and resolved defects

This file records behavior changes that are useful during later review and
interview preparation. Git remains the detailed change history.

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
