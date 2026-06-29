"""Streamlit UI for testing the LangGraph support-ticket agent."""

from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path

# ── ensure src/ is importable when running with `.venv/bin/streamlit run app.py`
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="LangGraph Agent Lab",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ───────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    .route-badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
    }
    .route-simple      { background:#d1fae5; color:#065f46; }
    .route-tool        { background:#dbeafe; color:#1e40af; }
    .route-missing_info{ background:#fef3c7; color:#92400e; }
    .route-risky       { background:#fee2e2; color:#991b1b; }
    .route-error       { background:#f3e8ff; color:#6b21a8; }
    .route-dead_letter { background:#f1f5f9; color:#475569; }
    .event-node {
        font-family: monospace;
        font-size: 0.78rem;
        padding: 2px 8px;
        border-left: 3px solid #6366f1;
        margin: 2px 0;
        background: #f8f9ff;
    }
    .log-line {
        font-family: monospace;
        font-size: 0.75rem;
        padding: 2px 6px;
        margin: 1px 0;
        border-radius: 3px;
        white-space: pre-wrap;
        word-break: break-all;
    }
    .log-start  { background:#eff6ff; border-left:3px solid #3b82f6; }
    .log-end    { background:#f0fdf4; border-left:3px solid #22c55e; }
    .log-error  { background:#fef2f2; border-left:3px solid #ef4444; }
    .log-chain  { background:#faf5ff; border-left:3px solid #a855f7; }
    .log-other  { background:#f8fafc; border-left:3px solid #94a3b8; }
    .stAlert { border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

LOG_FILE = Path("outputs/logs/langchain.log")

# ── sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🤖 LangGraph Agent")
    st.caption("Day 08 — Support Ticket Lab")

    st.divider()
    st.subheader("⚡ Preset Scenarios")

    PRESETS = {
        "S01 — Simple": "How do I reset my password?",
        "S02 — Tool lookup": "Please lookup order status for order 12345",
        "S03 — Missing info": "Can you fix it?",
        "S04 — Risky (refund)": "Refund this customer and send confirmation email",
        "S05 — Error + retry": "Timeout failure while processing request",
        "S06 — Risky (delete)": "Delete customer account after support verification",
        "S07 — Dead letter": "System failure cannot recover after multiple attempts",
    }

    chosen_preset = st.selectbox(
        "Load preset",
        options=["(custom)"] + list(PRESETS.keys()),
        key="preset_select",
    )

    st.divider()
    st.subheader("⚙️ Settings")

    checkpointer_kind = st.radio(
        "Checkpointer",
        ["memory", "sqlite"],
        horizontal=True,
        help="SQLite persists state history to outputs/checkpoints.db",
    )

    show_raw_state = st.toggle("Show raw state", value=False)
    show_logs = st.toggle("Show LangChain logs", value=True)

    st.divider()
    st.markdown(
        "**Routes:** `simple` `tool` `missing_info` `risky` `error`\n\n"
        "Priority: **risky** > tool > missing_info > error > simple"
    )

# ── main content ──────────────────────────────────────────────────────────────

st.title("🎫 Support Ticket Agent")
st.caption("Powered by LangGraph + GPT-4o-mini — all 5 routes supported")

# ── query input ───────────────────────────────────────────────────────────────

default_query = PRESETS.get(chosen_preset, "") if chosen_preset != "(custom)" else ""

col_input, col_btn = st.columns([5, 1])
with col_input:
    query = st.text_area(
        "Support ticket / query",
        value=default_query,
        height=80,
        placeholder="Type a customer support request…",
        key="query_input",
        label_visibility="collapsed",
    )
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("▶ Run", type="primary", use_container_width=True)

# ── helper: route badge ───────────────────────────────────────────────────────

ROUTE_LABELS = {
    "simple": "✅ Simple",
    "tool": "🔧 Tool",
    "missing_info": "❓ Missing Info",
    "risky": "⚠️ Risky",
    "error": "🔴 Error",
    "dead_letter": "💀 Dead Letter",
}

def route_badge(route: str) -> str:
    label = ROUTE_LABELS.get(route, route)
    return f'<span class="route-badge route-{route}">{label}</span>'


# ── log reader ────────────────────────────────────────────────────────────────

def read_log_tail(n: int = 200) -> list[str]:
    """Return last n lines from langchain.log."""
    if not LOG_FILE.exists():
        return []
    lines = LOG_FILE.read_text(encoding="utf-8").splitlines()
    return lines[-n:]


def read_log_since(byte_pos: int) -> tuple[list[str], int]:
    """Return new lines added after byte_pos, and new file size."""
    if not LOG_FILE.exists():
        return [], 0
    size = LOG_FILE.stat().st_size
    if size <= byte_pos:
        return [], size
    with LOG_FILE.open("rb") as f:
        f.seek(byte_pos)
        new_bytes = f.read()
    lines = new_bytes.decode("utf-8", errors="replace").splitlines()
    return [l for l in lines if l.strip()], size


def render_log_line(raw: str) -> None:
    """Parse a JSON log line and render with color coding."""
    # strip timestamp prefix: "2026-06-29 10:00:00 [DEBUG] langchain_agent_lab — {...}"
    try:
        json_start = raw.index("{")
        payload = json.loads(raw[json_start:])
        event = payload.get("event", "")
        node = payload.get("node", "")
        msg = payload.get("msg", raw)

        if "start" in event:
            css = "log-start"
            preview = payload.get("messages_preview") or payload.get("prompt_preview", "")
            if isinstance(preview, list):
                preview = " | ".join(str(p) for p in preview)
            detail = f"  ↳ {str(preview)[:180]}" if preview else ""
            label = f"[{node}] ▶ {msg}{detail}"
        elif "end" in event:
            css = "log-end"
            tokens = payload.get("total_tokens")
            latency = payload.get("latency_ms")
            resp = payload.get("response_preview", "")
            label = (
                f"[{node}] ✓ {msg}"
                + (f"  tokens={tokens}" if tokens else "")
                + (f"  {latency}ms" if latency else "")
                + (f"\n  ↳ {str(resp)[:200]}" if resp else "")
            )
        elif "error" in event:
            css = "log-error"
            label = f"[{node}] ✗ {msg}"
        elif "chain" in event:
            css = "log-chain"
            label = f"[{node}] ⛓ {msg}"
        else:
            css = "log-other"
            label = f"[{node}] {msg}"

        st.markdown(
            f'<div class="log-line {css}">{label}</div>',
            unsafe_allow_html=True,
        )
    except (ValueError, KeyError, json.JSONDecodeError):
        # fallback: plain text
        st.markdown(
            f'<div class="log-line log-other">{raw}</div>',
            unsafe_allow_html=True,
        )


# ── run graph ─────────────────────────────────────────────────────────────────

if run_btn and query.strip():
    try:
        from langgraph_agent_lab.graph import build_graph
        from langgraph_agent_lab.persistence import build_checkpointer
        from langgraph_agent_lab.state import Route, Scenario, initial_state
    except Exception as exc:
        st.error(f"Import error: {exc}")
        st.stop()

    scenario_id = f"ui-{uuid.uuid4().hex[:8]}"
    scenario = Scenario(id=scenario_id, query=query.strip(), expected_route=Route.SIMPLE)
    state = initial_state(scenario)

    try:
        checkpointer = build_checkpointer(checkpointer_kind)
        graph = build_graph(checkpointer=checkpointer)
    except Exception as exc:
        st.error(f"Graph build error: {exc}")
        st.stop()

    run_config = {"configurable": {"thread_id": state["thread_id"]}}

    # record log position before run
    log_pos_before = LOG_FILE.stat().st_size if LOG_FILE.exists() else 0

    progress_area = st.empty()
    with progress_area.container():
        with st.spinner("Running graph…"):
            t_start = time.perf_counter()
            try:
                final_state = graph.invoke(state, config=run_config)
            except Exception as exc:
                st.error(f"Graph execution error: {exc}")
                st.stop()
            elapsed = time.perf_counter() - t_start

    progress_area.empty()

    # ── results ─────────────────────────────────────────────────────────────

    actual_route = final_state.get("route", "unknown")
    final_answer = final_state.get("final_answer")
    pending_q = final_state.get("pending_question")
    approval = final_state.get("approval")
    events: list[dict] = final_state.get("events") or []
    tool_results: list[str] = final_state.get("tool_results") or []
    errors: list[str] = final_state.get("errors") or []
    attempt = final_state.get("attempt", 0)

    # ── top metrics row ──────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Route", ROUTE_LABELS.get(actual_route, actual_route))
    m2.metric("Nodes visited", len(events))
    m3.metric("Retries", attempt)
    m4.metric("Elapsed", f"{elapsed:.2f}s")

    st.markdown(
        f"**Classified as:** {route_badge(actual_route)} &nbsp; "
        f"risk_level=`{final_state.get('risk_level', '?')}`",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── tabs: Response / Audit Trail / LangChain Logs ────────────────────────
    tab_response, tab_audit, tab_logs = st.tabs(["💬 Agent Response", "📋 Audit Trail", "🪵 LangChain Logs"])

    # ── TAB 1: Response ──────────────────────────────────────────────────────
    with tab_response:
        if actual_route == "missing_info" and pending_q:
            st.warning(f"**Clarification needed:**\n\n{pending_q}")
            if final_answer:
                st.info(final_answer)

        elif actual_route == "dead_letter":
            st.error(final_answer or "Request could not be completed.")

        elif actual_route == "risky" and approval:
            approved = approval.get("approved", False)
            reviewer = approval.get("reviewer", "?")
            comment = approval.get("comment", "")
            proposed = final_state.get("proposed_action", "")

            if proposed:
                st.warning(f"**Proposed action:**\n\n{proposed}")
            if approved:
                st.success(f"✅ **Approved** by `{reviewer}`  \n_{comment}_")
            else:
                st.error(f"❌ **Rejected** by `{reviewer}`  \n_{comment}_")
            if final_answer:
                st.info(f"**Final response:**\n\n{final_answer}")

        elif final_answer:
            st.success(final_answer)
        else:
            st.info("No response generated.")

        if tool_results:
            with st.expander(f"🔧 Tool results ({len(tool_results)})"):
                for i, tr in enumerate(tool_results, 1):
                    is_error = "ERROR" in tr.upper()
                    (st.error if is_error else st.success)(f"**Call {i}:** {tr}")

        if errors:
            with st.expander(f"⚠️ Error log ({len(errors)})"):
                for e in errors:
                    st.text(e)

    # ── TAB 2: Audit Trail ───────────────────────────────────────────────────
    with tab_audit:
        ICONS = {
            "intake": "📥", "classify": "🏷️", "tool": "🔧", "evaluate": "🔍",
            "answer": "💬", "clarify": "❓", "risky_action": "⚠️", "approval": "✅",
            "retry": "🔄", "dead_letter": "💀", "finalize": "🏁",
        }
        for ev in events:
            node = ev.get("node", "?")
            etype = ev.get("event_type", "")
            msg = ev.get("message", "")
            meta = ev.get("metadata", {})
            icon = ICONS.get(node, "•")
            color = "#ef4444" if etype in ("error", "failed") else "#6366f1"
            st.markdown(
                f'<div class="event-node" style="border-color:{color}">'
                f"{icon} <b>{node}</b> — {msg}"
                + (f" <code>{meta}</code>" if meta and node != "finalize" else "")
                + "</div>",
                unsafe_allow_html=True,
            )
        if checkpointer_kind == "sqlite":
            st.caption(f"💾 Persisted: `thread_id={state['thread_id']}`")

    # ── TAB 3: LangChain Logs ────────────────────────────────────────────────
    with tab_logs:
        new_lines, _ = read_log_since(log_pos_before)

        if not new_lines:
            st.info("Không có log mới. Hãy đảm bảo `show_logs` được bật.")
        else:
            llm_calls = [l for l in new_lines if '"event": "chat_model_start"' in l or '"event": "llm_start"' in l]
            st.caption(
                f"📄 `outputs/langchain.log` — **{len(new_lines)} dòng mới** "
                f"({len(llm_calls)} LLM calls trong lần chạy này)"
            )
            col_dl, col_full = st.columns([2, 1])
            with col_dl:
                st.download_button(
                    "⬇️ Tải file log đầy đủ",
                    data=LOG_FILE.read_bytes() if LOG_FILE.exists() else b"",
                    file_name="langchain.log",
                    mime="text/plain",
                    use_container_width=True,
                )
            with col_full:
                view_all = st.toggle("Xem tất cả log cũ", value=False)

            st.markdown("---")
            lines_to_show = read_log_tail(500) if view_all else new_lines
            for raw in lines_to_show:
                render_log_line(raw)

    # ── raw state ────────────────────────────────────────────────────────────
    if show_raw_state:
        st.divider()
        st.subheader("🛠 Raw State")
        display_state = {k: v for k, v in final_state.items() if k != "events"}
        st.json(display_state)
        with st.expander("Events JSON"):
            st.json(events)

elif run_btn and not query.strip():
    st.warning("Please enter a query first.")

# ── empty state hint ─────────────────────────────────────────────────────────

if not run_btn:
    st.info(
        "👆 Type a support query above or pick a preset from the sidebar, then click **▶ Run**.\n\n"
        "The agent will classify the intent, route through the correct nodes, and return a response."
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Simple route**\n> How do I reset my password?\n\nDirect answer from LLM.")
    with col2:
        st.markdown("**Risky route (HITL)**\n> Refund this customer\n\nGoes through approval before acting.")
    with col3:
        st.markdown("**Error + retry**\n> Timeout failure while processing\n\nRetries up to `max_attempts` times.")

# ── sidebar: log preview (last 20 lines always visible) ──────────────────────

if show_logs and LOG_FILE.exists():
    with st.sidebar:
        st.divider()
        st.subheader("🪵 Log gần nhất")
        tail = read_log_tail(20)
        if tail:
            log_text = "\n".join(tail)
            st.text_area("outputs/langchain.log", value=log_text, height=200, label_visibility="collapsed")
        else:
            st.caption("Log trống — chạy một query để xem.")
