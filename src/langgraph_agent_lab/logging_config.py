"""LangChain logging — captures every LLM call to outputs/langchain.log.

Usage:
    from langgraph_agent_lab.logging_config import get_llm_callbacks, setup_logging
    setup_logging()
    callbacks = get_llm_callbacks()
    llm = get_llm().with_config(callbacks=callbacks)
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Union
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, LLMResult

# ── log file path ─────────────────────────────────────────────────────────────
LOG_DIR = Path("outputs/logs")
LOG_FILE = LOG_DIR / "langchain.log"

# ── Python std logger ─────────────────────────────────────────────────────────
logger = logging.getLogger("langchain_agent_lab")


def setup_logging(level: str | None = None) -> None:
    """Initialize file + console logging. Call once at app startup."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log_level = getattr(logging, (level or os.getenv("LOG_LEVEL", "INFO")).upper(), logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler — all LangChain events
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Console handler — INFO and above
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(fmt)

    root = logging.getLogger("langchain_agent_lab")
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.addHandler(fh)
    root.addHandler(ch)
    root.propagate = False


# ── Custom callback handler ────────────────────────────────────────────────────

class LangChainFileLogger(BaseCallbackHandler):
    """Writes every LLM call (request + response + latency) to langchain.log."""

    def __init__(self, node_name: str = "unknown") -> None:
        super().__init__()
        self.node_name = node_name
        self._t_start: dict[UUID, float] = {}

    def _log(self, level: str, msg: str, **extra: Any) -> None:
        record = {"node": self.node_name, "event": level, "msg": msg, **extra}
        logger.debug(json.dumps(record, ensure_ascii=False, default=str))

    # ── LLM events ────────────────────────────────────────────────────────────

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._t_start[run_id] = time.monotonic()
        model = serialized.get("kwargs", {}).get("model_name") or serialized.get("id", ["?"])[-1]
        self._log(
            "llm_start",
            f"LLM call started — model={model}",
            run_id=str(run_id),
            model=model,
            prompt_preview=prompts[0][:200] if prompts else "",
        )

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._t_start[run_id] = time.monotonic()
        model = (
            serialized.get("kwargs", {}).get("model_name")
            or serialized.get("kwargs", {}).get("model")
            or serialized.get("id", ["?"])[-1]
        )
        flat = [m.content[:150] for batch in messages for m in batch]
        self._log(
            "chat_model_start",
            f"Chat model call — model={model} messages={len(flat)}",
            run_id=str(run_id),
            model=model,
            messages_preview=flat,
        )

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        latency_ms = int((time.monotonic() - self._t_start.pop(run_id, time.monotonic())) * 1000)
        gen = response.generations
        text = ""
        if gen and gen[0]:
            g = gen[0][0]
            text = g.text if hasattr(g, "text") else (g.message.content if isinstance(g, ChatGeneration) else "")

        usage = response.llm_output or {}
        token_info = usage.get("token_usage", {})
        self._log(
            "llm_end",
            f"LLM call finished — latency={latency_ms}ms",
            run_id=str(run_id),
            latency_ms=latency_ms,
            response_preview=str(text)[:300],
            total_tokens=token_info.get("total_tokens"),
            prompt_tokens=token_info.get("prompt_tokens"),
            completion_tokens=token_info.get("completion_tokens"),
        )

    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        latency_ms = int((time.monotonic() - self._t_start.pop(run_id, time.monotonic())) * 1000)
        self._log(
            "llm_error",
            f"LLM call FAILED — {type(error).__name__}: {error}",
            run_id=str(run_id),
            latency_ms=latency_ms,
            error=str(error),
        )
        logger.error("[%s] LLM error after %dms: %s", self.node_name, latency_ms, error)

    # ── Chain events (optional verbose) ───────────────────────────────────────

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        name = (serialized or {}).get("id", ["chain"])[-1] if serialized else "chain"
        self._log("chain_start", f"Chain started — {name}", run_id=str(run_id))

    def on_chain_end(self, outputs: dict[str, Any], *, run_id: UUID, **kwargs: Any) -> None:
        self._log("chain_end", "Chain finished", run_id=str(run_id))


# ── Factory ───────────────────────────────────────────────────────────────────

def get_llm_callbacks(node_name: str = "unknown") -> list[BaseCallbackHandler]:
    """Return callback list to pass into get_llm().with_config(callbacks=...)."""
    return [LangChainFileLogger(node_name=node_name)]
