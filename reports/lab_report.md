# Day 08 Lab Report

## 1. Team / student

- Name: Do Thi Thanh Binh
- Repo/commit: 2A202600717-DoThiThanhBinh-Track3-Day8
- Date: 2026-06-29

## 2. Architecture

The graph is a **LangGraph StateGraph** built around a support-ticket intent pipeline:

```
START → intake → classify ──┬─ simple       → answer → finalize → END
                            ├─ tool         → tool → evaluate ──┬─ success   → answer → finalize → END
                            │                                   └─ needs_retry → retry ──┬─ within limit → tool
                            │                                                            └─ at limit    → dead_letter → finalize → END
                            ├─ missing_info → clarify → finalize → END
                            ├─ risky        → risky_action → approval ──┬─ approved → tool → ...
                            │                                           └─ rejected → clarify → finalize → END
                            └─ error        → retry ──┬─ within limit → tool
                                                      └─ at limit    → dead_letter → finalize → END
```

**Key design decisions:**
- `classify_node` uses OpenAI `gpt-4o-mini` with `.with_structured_output(ClassificationOutput)` for reliable enum routing.
- `answer_node` uses the same LLM grounded in `tool_results` and `approval` context.
- `evaluate_node` acts as an LLM-as-judge for tool result quality, enabling the retry loop.
- `approval_node` defaults to mock approval; set `LANGGRAPH_INTERRUPT=true` for real HITL.
- All paths converge at `finalize → END` ensuring no hanging branches.

## 3. State schema

| Field | Reducer | Why |
|---|---|---|
| query | overwrite | normalized once by intake |
| route | overwrite | latest classification only |
| risk_level | overwrite | latest assessment only |
| attempt | overwrite | monotonically increasing counter |
| max_attempts | overwrite | set from scenario config |
| final_answer | overwrite | last generated answer |
| evaluation_result | overwrite | latest evaluate verdict |
| pending_question | overwrite | latest clarification question |
| proposed_action | overwrite | staged risky action description |
| approval | overwrite | HITL decision dict |
| messages | append (add) | full conversation audit trail |
| tool_results | append (add) | all tool outputs including retries |
| errors | append (add) | all error messages for debugging |
| events | append (add) | audit log — one entry per node |

## 4. Scenario results

**Summary:** 7 scenarios | success_rate=100% | avg_nodes=6.4 | total_retries=3 | total_interrupts=2

| Scenario | Expected | Actual | Success | Retries | Interrupts |
|---|---|---|---:|---:|---:|
| S01_simple | simple | simple | ✓ | 0 | 0 |
| S02_tool | tool | tool | ✓ | 0 | 0 |
| S03_missing | missing_info | missing_info | ✓ | 0 | 0 |
| S04_risky | risky | risky | ✓ | 0 | 1 |
| S05_error | error | error | ✓ | 2 | 0 |
| S06_delete | risky | risky | ✓ | 0 | 1 |
| S07_dead_letter | error | error | ✓ | 1 | 0 |

## 5. Failure analysis

1. **Retry / tool failure (S05, S07):** Error-route scenarios simulate transient tool failures. The `evaluate_node` (LLM-as-judge) detects `ERROR` in tool output and returns `needs_retry`. The `retry_or_fallback_node` increments `attempt`; `route_after_retry` loops back to `tool` until `attempt >= max_attempts`, then diverts to `dead_letter`. S07 sets `max_attempts=1` so it immediately exhausts retries and lands in dead_letter.

2. **Risky action without approval (S04, S06):** Queries with side-effects (refund, delete account) are classified as `risky` by the LLM. They pass through `risky_action_node` (which prepares a description) then `approval_node` (mock approves by default). Without the approval gate, the risky action could execute immediately — the HITL node is the safety checkpoint before the tool runs.

## 6. Persistence / recovery evidence

- **Checkpointer:** `SqliteSaver` backed by `outputs/checkpoints.db` with WAL journal mode.
- **thread_id:** Every scenario run uses `thread-<scenario_id>` as the thread ID, making checkpoints queryable per run.
- **State history:** After `make run-scenarios`, each scenario's full state history is persisted in SQLite and can be replayed with `graph.get_state_history(config={'configurable': {'thread_id': '...'}})` for time-travel debugging.
- **Crash-resume:** If the process is killed mid-run, re-invoking with the same `thread_id` resumes from the last checkpoint.

## 7. Extension work

- **SQLite persistence:** Full `SqliteSaver` implementation with WAL mode in `persistence.py`.
- **LLM-as-judge evaluation:** `evaluate_node` uses `gpt-4o-mini` to assess tool result quality rather than a simple string heuristic.
- **Structured output classification:** `classify_node` uses `.with_structured_output(ClassificationOutput)` for reliable routing.

## 8. Improvement plan

If given one more day, priorities would be:
1. **Parallel fan-out** — use `Send()` to invoke multiple tools concurrently for `tool`-route queries, reducing latency.
2. **Real HITL** — wire `interrupt()` with a Streamlit UI for approve/reject on risky actions.
3. **Streaming** — surface intermediate node outputs via `graph.astream_events()` for real-time UX.
4. **Tracing** — integrate LangSmith for production observability of LLM calls and graph state.
