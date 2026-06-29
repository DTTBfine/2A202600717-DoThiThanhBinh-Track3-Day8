"""Checkpointer adapter."""

from __future__ import annotations

from typing import Any


def build_checkpointer(kind: str = "memory", database_url: str | None = None) -> Any | None:
    """Return a LangGraph checkpointer.

    Supports: memory (default), sqlite, postgres, none.
    SQLite uses WAL mode for concurrent read safety.
    """
    if kind == "none":
        return None

    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver  # noqa: PLC0415
        return MemorySaver()

    if kind == "sqlite":
        import sqlite3  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: PLC0415

        db_path = database_url or "outputs/checkpoints.db"
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.commit()
        return SqliteSaver(conn)

    if kind == "postgres":
        raise NotImplementedError(
            "TODO(student): implement Postgres checkpointer (optional extension)"
        )

    raise ValueError(f"Unknown checkpointer kind: {kind}")
