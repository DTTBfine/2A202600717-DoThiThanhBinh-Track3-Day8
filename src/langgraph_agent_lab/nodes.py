"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update dict.
Do NOT mutate input state — return new values only.
"""

from __future__ import annotations

import os
import time

from dotenv import load_dotenv
from pydantic import BaseModel

from .llm import get_llm
from .logging_config import get_llm_callbacks, setup_logging
from .state import AgentState, make_event

load_dotenv()
setup_logging()


# ── Pydantic schema for structured classification output ──────────────────────

class ClassificationOutput(BaseModel):
    route: str   # one of: simple | tool | missing_info | risky | error
    risk_level: str  # "high" for risky, "low" otherwise
    reasoning: str


_CLASSIFY_SYSTEM = """You are a support-ticket intent classifier.

Classify the user query into EXACTLY ONE route using this priority order:
1. risky   — actions with side-effects: refunds, deletions, sending emails, cancellations, account changes
2. tool    — information lookups: order status, tracking, search, data retrieval
3. missing_info — vague/incomplete queries lacking actionable context (e.g. "fix it", "help me")
4. error   — system failures: timeouts, crashes, service unavailable, repeated failures
5. simple  — general questions answerable without tools or risky actions

Return JSON with:
- route: one of simple | tool | missing_info | risky | error
- risk_level: "high" if route is risky, else "low"
- reasoning: one sentence explaining why
"""

_ANSWER_SYSTEM = """You are a helpful customer support agent.
Generate a concise, professional response grounded in the provided context.
Do not hallucinate order details or account information not present in tool_results.
Keep the response under 3 sentences."""


# ─── EXAMPLE: working node (provided for reference) ──────────────────────────

def intake_node(state: AgentState) -> dict:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


# ─── classify_node ────────────────────────────────────────────────────────────

def classify_node(state: AgentState) -> dict:
    """Classify the query using LLM structured output."""
    t0 = time.monotonic()
    callbacks = get_llm_callbacks("classify")
    llm = get_llm(temperature=0.0)
    structured = llm.with_structured_output(ClassificationOutput)
    query = state.get("query", "")
    result: ClassificationOutput = structured.invoke(
        [
            {"role": "system", "content": _CLASSIFY_SYSTEM},
            {"role": "user", "content": query},
        ],
        config={"callbacks": callbacks},
    )
    latency = int((time.monotonic() - t0) * 1000)
    return {
        "route": result.route,
        "risk_level": result.risk_level,
        "messages": [f"classify:{result.route}"],
        "events": [
            make_event(
                "classify",
                "completed",
                f"route={result.route} risk={result.risk_level}",
                latency_ms=latency,
                reasoning=result.reasoning,
            )
        ],
    }


# ─── tool_node ────────────────────────────────────────────────────────────────

def tool_node(state: AgentState) -> dict:
    """Execute a mock tool call with transient failure simulation."""
    route = state.get("route", "")
    attempt = state.get("attempt", 0)

    # Simulate transient error for error-route scenarios on early attempts
    if route == "error" and attempt < 2:
        result = f"ERROR: tool call failed on attempt {attempt} (transient timeout)"
        return {
            "tool_results": [result],
            "events": [make_event("tool", "error", result, attempt=attempt)],
        }

    query = state.get("query", "")
    result = f"TOOL_RESULT: Successfully retrieved data for '{query[:60]}' on attempt {attempt}"
    return {
        "tool_results": [result],
        "events": [make_event("tool", "completed", result, attempt=attempt)],
    }


# ─── evaluate_node ───────────────────────────────────────────────────────────

def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — LLM-as-judge for quality gate."""
    tool_results = state.get("tool_results") or []
    latest = tool_results[-1] if tool_results else ""

    # LLM-as-judge: use LLM to determine if result is satisfactory
    llm = get_llm(temperature=0.0).with_config(callbacks=get_llm_callbacks("evaluate"))
    prompt = (
        f"Evaluate this tool result. Reply with ONLY 'success' or 'needs_retry'.\n\n"
        f"Tool result: {latest}\n\n"
        f"Reply 'needs_retry' if the result contains ERROR or failure indicators. "
        f"Reply 'success' otherwise."
    )
    response = llm.invoke(prompt)
    verdict_text = response.content.strip().lower()
    verdict = "needs_retry" if "needs_retry" in verdict_text or "error" in latest.upper() else "success"

    return {
        "evaluation_result": verdict,
        "events": [make_event("evaluate", "completed", f"evaluation={verdict}", tool_result_preview=latest[:80])],
    }


# ─── answer_node ─────────────────────────────────────────────────────────────

def answer_node(state: AgentState) -> dict:
    """Generate a final response using LLM grounded in context."""
    t0 = time.monotonic()
    llm = get_llm(temperature=0.3).with_config(callbacks=get_llm_callbacks("answer"))
    query = state.get("query", "")
    tool_results = state.get("tool_results") or []
    approval = state.get("approval")

    context_parts = [f"User query: {query}"]
    if tool_results:
        context_parts.append(f"Tool results: {'; '.join(tool_results[-3:])}")
    if approval:
        approved = approval.get("approved", False)
        context_parts.append(f"Action approval: {'Approved' if approved else 'Rejected'} by {approval.get('reviewer', 'reviewer')}")

    context = "\n".join(context_parts)
    response = llm.invoke(
        [
            {"role": "system", "content": _ANSWER_SYSTEM},
            {"role": "user", "content": context},
        ]
    )
    answer = response.content.strip()
    latency = int((time.monotonic() - t0) * 1000)
    return {
        "final_answer": answer,
        "events": [make_event("answer", "completed", "response generated", latency_ms=latency)],
    }


# ─── ask_clarification_node ───────────────────────────────────────────────────

def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating."""
    query = state.get("query", "")
    llm = get_llm(temperature=0.2).with_config(callbacks=get_llm_callbacks("clarify"))
    prompt = (
        f"The user sent a vague support request: '{query}'\n\n"
        f"Generate ONE specific clarification question to get the missing information needed to help them. "
        f"Be concise and direct."
    )
    response = llm.invoke(prompt)
    question = response.content.strip()
    clarification_msg = f"To help you better, could you please clarify: {question}"
    return {
        "pending_question": question,
        "final_answer": clarification_msg,
        "events": [make_event("clarify", "completed", "clarification requested", question=question[:80])],
    }


# ─── risky_action_node ────────────────────────────────────────────────────────

def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for human approval."""
    query = state.get("query", "")
    llm = get_llm(temperature=0.0).with_config(callbacks=get_llm_callbacks("risky_action"))
    prompt = (
        f"A support agent wants to perform this action: '{query}'\n\n"
        f"Describe in ONE sentence what will happen and why it requires human approval."
    )
    response = llm.invoke(prompt)
    proposed = response.content.strip()
    return {
        "proposed_action": proposed,
        "events": [make_event("risky_action", "completed", "action staged for approval", action=proposed[:80])],
    }


# ─── approval_node ───────────────────────────────────────────────────────────

def approval_node(state: AgentState) -> dict:
    """Human-in-the-loop approval step (mock by default)."""
    use_interrupt = os.getenv("LANGGRAPH_INTERRUPT", "").lower() == "true"
    if use_interrupt:
        from langgraph.types import interrupt  # type: ignore[import]
        decision = interrupt({"proposed_action": state.get("proposed_action", "")})
        approved = bool(decision.get("approved", False))
        reviewer = decision.get("reviewer", "human-reviewer")
        comment = decision.get("comment", "")
    else:
        # Default: mock approval (approved=True) so tests and CI run offline
        approved = True
        reviewer = "mock-reviewer"
        comment = "Auto-approved in mock mode"

    approval_dict = {"approved": approved, "reviewer": reviewer, "comment": comment}
    return {
        "approval": approval_dict,
        "events": [make_event("approval", "completed", f"approved={approved}", reviewer=reviewer)],
    }


# ─── retry_or_fallback_node ──────────────────────────────────────────────────

def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt and increment counter."""
    attempt = state.get("attempt", 0) + 1
    error_msg = f"Attempt {attempt} failed — scheduling retry"
    return {
        "attempt": attempt,
        "errors": [error_msg],
        "events": [make_event("retry", "retry", error_msg, attempt=attempt)],
    }


# ─── dead_letter_node ────────────────────────────────────────────────────────

def dead_letter_node(state: AgentState) -> dict:
    """Handle unresolvable failures after max retries exceeded."""
    attempt = state.get("attempt", 0)
    max_attempts = state.get("max_attempts", 3)
    query = state.get("query", "")
    msg = (
        f"We were unable to complete your request after {attempt} attempts. "
        f"Your ticket has been escalated to our engineering team for manual review. "
        f"Original request: '{query[:80]}'"
    )
    return {
        "final_answer": msg,
        "events": [make_event("dead_letter", "failed", f"exhausted {attempt}/{max_attempts} attempts")],
    }


# ─── finalize_node ───────────────────────────────────────────────────────────

def finalize_node(state: AgentState) -> dict:
    """Emit a final audit event. All routes must pass through here before END."""
    route = state.get("route", "unknown")
    answer = state.get("final_answer") or state.get("pending_question") or ""
    return {
        "events": [
            make_event(
                "finalize",
                "completed",
                "workflow finished",
                route=route,
                has_answer=bool(answer),
            )
        ]
    }
