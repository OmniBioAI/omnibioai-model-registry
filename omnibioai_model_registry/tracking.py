"""
Pure DB read/write functions for MLflow-style run tracking.

Every function accepts `conn` as its first argument (connection-injection pattern)
so they can be exercised in unit tests with a real or mock connection without
touching module-level state.

All timestamps are stored as naive UTC datetimes (MySQL DATETIME has no tz).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .errors import ModelNotFound


# --------------------------------------------------------------------------- #
# Internal helpers                                                              #
# --------------------------------------------------------------------------- #

def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _ensure_run(conn, run_id: str, task: str, model_name: str, actor: Optional[str] = None) -> None:
    """INSERT IGNORE — idempotent; safe to call before every log mutation."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT IGNORE INTO omr_runs (run_id, task, model_name, status, started_at, actor)
            VALUES (%s, %s, %s, 'running', %s, %s)
            """,
            (run_id, task, model_name, _now_utc(), actor),
        )


# --------------------------------------------------------------------------- #
# Run lifecycle                                                                 #
# --------------------------------------------------------------------------- #

def create_run(
    conn,
    task: str,
    model_name: str,
    run_id: str,
    actor: Optional[str] = None,
) -> dict:
    """Ensure the run row exists and return its current state."""
    _ensure_run(conn, run_id, task, model_name, actor)
    return get_run(conn, run_id)


def finish_run(conn, run_id: str) -> None:
    """Mark a run as finished. Raises ModelNotFound if run_id is unknown."""
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE omr_runs SET status = 'finished', finished_at = %s WHERE run_id = %s",
            (_now_utc(), run_id),
        )
        if cur.rowcount == 0:
            raise ModelNotFound(f"Run not found: {run_id}")


# --------------------------------------------------------------------------- #
# Params                                                                        #
# --------------------------------------------------------------------------- #

def log_param(
    conn,
    run_id: str,
    key: str,
    value: Any,
    *,
    task: str = "",
    model_name: str = "",
) -> None:
    _ensure_run(conn, run_id, task, model_name)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO omr_params (run_id, key_name, value_text)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE value_text = VALUES(value_text)
            """,
            (run_id, key, json.dumps(value)),
        )


def log_params(
    conn,
    run_id: str,
    params: Dict[str, Any],
    *,
    task: str = "",
    model_name: str = "",
) -> None:
    for k, v in params.items():
        log_param(conn, run_id, k, v, task=task, model_name=model_name)


# --------------------------------------------------------------------------- #
# Metrics                                                                       #
# --------------------------------------------------------------------------- #

def log_metric(
    conn,
    run_id: str,
    key: str,
    value: float,
    step: int = 0,
    *,
    task: str = "",
    model_name: str = "",
) -> None:
    _ensure_run(conn, run_id, task, model_name)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO omr_metrics (run_id, key_name, value, step, ts_utc)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (run_id, key, value, step, _now_utc()),
        )


def log_metrics(
    conn,
    run_id: str,
    metrics: Dict[str, float],
    step: int = 0,
    *,
    task: str = "",
    model_name: str = "",
) -> None:
    for k, v in metrics.items():
        log_metric(conn, run_id, k, v, step, task=task, model_name=model_name)


# --------------------------------------------------------------------------- #
# Tags (run-level)                                                              #
# --------------------------------------------------------------------------- #

def set_tag(
    conn,
    run_id: str,
    key: str,
    value: str,
    *,
    task: str = "",
    model_name: str = "",
) -> None:
    _ensure_run(conn, run_id, task, model_name)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO omr_tags (run_id, key_name, value_text)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE value_text = VALUES(value_text)
            """,
            (run_id, key, value),
        )


def set_tags(
    conn,
    run_id: str,
    tags: Dict[str, str],
    *,
    task: str = "",
    model_name: str = "",
) -> None:
    for k, v in tags.items():
        set_tag(conn, run_id, k, v, task=task, model_name=model_name)


# --------------------------------------------------------------------------- #
# Run query                                                                     #
# --------------------------------------------------------------------------- #

def get_run(conn, run_id: str) -> dict:
    """
    Return a full run snapshot: metadata + params + tags + metrics_summary.

    metrics_summary keeps the latest value (highest step, then latest insert)
    for each key — same semantics as MLflow's RunData.metrics.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM omr_runs WHERE run_id = %s", (run_id,))
        row = cur.fetchone()
        if row is None:
            raise ModelNotFound(f"Run not found: {run_id}")

        cur.execute(
            "SELECT key_name, value_text FROM omr_params WHERE run_id = %s",
            (run_id,),
        )
        params: Dict[str, Any] = {}
        for r in cur.fetchall():
            try:
                params[r["key_name"]] = json.loads(r["value_text"])
            except (json.JSONDecodeError, TypeError):
                params[r["key_name"]] = r["value_text"]

        cur.execute(
            "SELECT key_name, value_text FROM omr_tags WHERE run_id = %s",
            (run_id,),
        )
        tags: Dict[str, str] = {r["key_name"]: r["value_text"] for r in cur.fetchall()}

        # Highest-step value per key (latest-insert tie-breaker via id DESC)
        cur.execute(
            """
            SELECT key_name, value, step
            FROM omr_metrics
            WHERE run_id = %s
            ORDER BY step DESC, id DESC
            """,
            (run_id,),
        )
        metrics_summary: Dict[str, float] = {}
        for r in cur.fetchall():
            if r["key_name"] not in metrics_summary:
                metrics_summary[r["key_name"]] = r["value"]

    started = row["started_at"]
    finished = row["finished_at"]
    return {
        "run_id": row["run_id"],
        "task": row["task"],
        "model_name": row["model_name"],
        "status": row["status"],
        "started_at": started.isoformat() if started else None,
        "finished_at": finished.isoformat() if finished else None,
        "actor": row["actor"],
        "params": params,
        "tags": tags,
        "metrics_summary": metrics_summary,
    }


def get_metric_history(conn, run_id: str, key: str) -> List[dict]:
    """Return all (value, step, ts_utc) rows for a metric key, ordered by step ASC."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT value, step, ts_utc
            FROM omr_metrics
            WHERE run_id = %s AND key_name = %s
            ORDER BY step ASC, id ASC
            """,
            (run_id, key),
        )
        return [
            {
                "value": r["value"],
                "step": r["step"],
                "ts_utc": r["ts_utc"].isoformat() if r["ts_utc"] else None,
            }
            for r in cur.fetchall()
        ]


def list_runs(conn, task: str, model_name: str) -> List[dict]:
    """List all runs for a (task, model_name), newest first."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT run_id, status, started_at, finished_at
            FROM omr_runs
            WHERE task = %s AND model_name = %s
            ORDER BY started_at DESC
            """,
            (task, model_name),
        )
        return [
            {
                "run_id": r["run_id"],
                "status": r["status"],
                "started_at": r["started_at"].isoformat() if r["started_at"] else None,
                "finished_at": r["finished_at"].isoformat() if r["finished_at"] else None,
            }
            for r in cur.fetchall()
        ]


# --------------------------------------------------------------------------- #
# Version-level tags                                                            #
# --------------------------------------------------------------------------- #

def set_version_tag(
    conn,
    task: str,
    model_name: str,
    version: str,
    key: str,
    value: str,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO omr_version_tags (task, model_name, version, key_name, value_text)
            VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE value_text = VALUES(value_text)
            """,
            (task, model_name, version, key, value),
        )


def get_version_tags(conn, task: str, model_name: str, version: str) -> Dict[str, str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT key_name, value_text
            FROM omr_version_tags
            WHERE task = %s AND model_name = %s AND version = %s
            """,
            (task, model_name, version),
        )
        return {r["key_name"]: r["value_text"] for r in cur.fetchall()}
