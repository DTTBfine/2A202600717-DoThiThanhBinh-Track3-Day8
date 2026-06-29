# Day 08 Lab — LangGraph Agentic Orchestration

Build a production-style LangGraph workflow for a support-ticket agent with state management, conditional routing, retry loops, human-in-the-loop approval, persistence, and metrics.

---

## Kết quả đạt được

| Metric | Kết quả |
|---|---|
| Unit tests | **25 / 25 passed** |
| Scenarios | **7 / 7 — success_rate 100%** |
| LLM | OpenAI `gpt-4o-mini` |
| Persistence | SQLite WAL (`outputs/checkpoints.db`) |
| UI | Streamlit — `app.py` |
| Logs | `outputs/logs/langchain.log` |

---

## Graph Topology

```
START
  │
  ▼
intake ──► classify
              │
              ├─[simple] ──────────────────────────────────► answer ──► finalize ──► END
              │
              ├─[tool] ────────────────► tool ──► evaluate
              │                                       │
              │                               [success]├──────────────► answer ──► finalize ──► END
              │                                        │
              │                             [needs_retry]└──► retry
              │                                                  │
              │                                    [attempt < max]├──► tool  (retry loop)
              │                                                   │
              │                                    [attempt ≥ max]└──► dead_letter ──► finalize ──► END
              │
              ├─[missing_info] ─────────────────────────────► clarify ──► finalize ──► END
              │
              ├─[risky] ──► risky_action ──► approval
              │                                  │
              │                        [approved]├──► tool ──► evaluate ──► ...
              │                                  │
              │                        [rejected]└──► clarify ──► finalize ──► END
              │
              └─[error] ──► retry
                                │
                   [attempt < max]├──► tool ──► evaluate ──► ...
                                  │
                   [attempt ≥ max]└──► dead_letter ──► finalize ──► END
```

### Nodes (11)

| Node | Vai trò | LLM |
|---|---|---|
| `intake` | Normalize query | — |
| `classify` | Phân loại intent → route | ✅ structured output |
| `tool` | Mock tool call, simulate error | — |
| `evaluate` | Đánh giá tool result | ✅ LLM-as-judge |
| `answer` | Sinh câu trả lời cuối | ✅ grounded |
| `clarify` | Hỏi làm rõ khi thiếu thông tin | ✅ |
| `risky_action` | Mô tả action cần approval | ✅ |
| `approval` | HITL gate (mock / real interrupt) | — |
| `retry` | Tăng attempt counter | — |
| `dead_letter` | Xử lý khi hết retry | — |
| `finalize` | Emit audit event, tất cả routes kết thúc ở đây | — |

### Routing logic

| Function | Từ node | Điều kiện | Đến node |
|---|---|---|---|
| `route_after_classify` | classify | route = simple | answer |
| | | route = tool | tool |
| | | route = missing_info | clarify |
| | | route = risky | risky_action |
| | | route = error | retry |
| `route_after_evaluate` | evaluate | evaluation_result = success | answer |
| | | evaluation_result = needs_retry | retry |
| `route_after_retry` | retry | attempt < max_attempts | tool |
| | | attempt ≥ max_attempts | dead_letter |
| `route_after_approval` | approval | approved = True | tool |
| | | approved = False | clarify |

---

## Cài đặt & chạy

### 1. Cài môi trường

```bash
python3.11 -m venv .venv
source .venv/bin/activate

pip install -e '.[dev]'
pip install langchain-openai          # hoặc langchain-anthropic / langchain-google-genai
pip install langgraph-checkpoint-sqlite
```

### 2. Cấu hình API key

```bash
cp .env.example .env
# Mở .env, điền key vào:
# OPENAI_API_KEY=sk-...
# hoặc ANTHROPIC_API_KEY=sk-ant-...
# hoặc GEMINI_API_KEY=AIza...
```

### 3. Chạy tests

```bash
# Unit tests (không cần API key)
pytest tests/test_state.py tests/test_routing.py tests/test_metrics.py -v

# Integration tests (cần API key)
pytest tests/test_graph_smoke.py -v

# Toàn bộ
pytest -v
```

### 4. Chạy scenarios và validate

```bash
make run-scenarios    # → outputs/metrics.json
make grade-local      # → Metrics valid. success_rate=100.00%
```

### 5. Chạy Streamlit UI

```bash
source .venv/bin/activate
streamlit run app.py --server.port 8502
# Mở trình duyệt: http://localhost:8502
```

UI hỗ trợ:
- 7 preset scenarios + custom query
- Route badge màu sắc, metrics row, audit trail
- Tab **LangChain Logs** — xem LLM call theo thời gian thực
- Download `outputs/logs/langchain.log`
- Toggle checkpointer memory / SQLite

### 6. Make commands

| Command | Tác dụng |
|---|---|
| `make install` | Cài project + dev dependencies |
| `make test` | Chạy pytest |
| `make lint` | Ruff linter |
| `make typecheck` | Mypy type checker |
| `make run-scenarios` | Chạy 7 scenarios → `outputs/metrics.json` |
| `make grade-local` | Validate metrics JSON schema |
| `make clean` | Xóa cache và generated files |

---

## Cấu trúc outputs

```
outputs/
├── metrics.json          ← kết quả grading (success_rate, per-scenario)
├── checkpoints.db        ← SQLite persistence (WAL mode)
└── logs/
    └── langchain.log     ← mọi LLM call: model, prompt, response, tokens, latency
```

---

## How you will be graded

| Category | Points | What we look for |
|---|---:|---|
| Architecture & state schema | 15 | Typed state with correct reducers, student-added fields, lean serializable state |
| Graph construction & wiring | 15 | All nodes registered, edges correct, conditional edges work, graph compiles |
| LLM integration | 15 | classify_node + answer_node use real LLM calls (structured output, grounded generation) |
| Graph behavior | 20 | All scenario routes correct, bounded retry loop, HITL approval path, all routes terminate |
| Persistence & recovery | 10 | Checkpointer wired, thread_id per run, state history or crash-resume evidence |
| Metrics & tests | 15 | `metrics.json` valid, scenario coverage, tests pass, meaningful counts |
| Report & demo | 10 | Architecture explanation, metrics table, failure analysis, improvement ideas |

---

## Understanding `scenarios.jsonl`

The file `data/sample/scenarios.jsonl` contains **7 sample scenarios**:

```jsonl
{"id":"S01_simple",      "query":"How do I reset my password?",                          "expected_route":"simple"}
{"id":"S02_tool",        "query":"Please lookup order status for order 12345",            "expected_route":"tool"}
{"id":"S03_missing",     "query":"Can you fix it?",                                      "expected_route":"missing_info"}
{"id":"S04_risky",       "query":"Refund this customer and send confirmation email",      "expected_route":"risky"}
{"id":"S05_error",       "query":"Timeout failure while processing request",              "expected_route":"error"}
{"id":"S06_delete",      "query":"Delete customer account after support verification",    "expected_route":"risky"}
{"id":"S07_dead_letter", "query":"System failure cannot recover after multiple attempts", "expected_route":"error", "max_attempts":1}
```

**Route priority:** risky > tool > missing_info > error > simple

---

## Common pitfalls

1. **Missing state fields** — Add `evaluation_result`, `pending_question`, `proposed_action`, `approval` to `AgentState`.
2. **LLM structured output** — Use `.with_structured_output(YourModel)`. Raw text parsing breaks on hidden scenarios.
3. **Unbounded retry** — Check `attempt >= max_attempts` (not `==`) to handle edge cases.
4. **Graph wiring** — Every path must end at `finalize → END`.
5. **SqliteSaver API** — Use `SqliteSaver(conn=sqlite3.connect(...))`, not `from_conn_string()`.
6. **API key not set** — Check `.env` and use `python-dotenv` or export manually.
