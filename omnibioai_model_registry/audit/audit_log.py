# File: omnibioai_model_registry/audit/audit_log.py
"""
OmniBioAI omnibioai_model_registry.audit.audit_log.

Purpose:
    Defines PromotionEvent, append_promotion_event and now_utc_iso for omnibioai_model_registry.audit.audit_log.

Author:
    Manish Kumar <manish@omnibioai.org>
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class PromotionEvent:
    task: str
    model_name: str
    alias: str
    version: str
    actor: str | None
    reason: str | None
    ts_utc: str


def append_promotion_event(log_path: Path, event: PromotionEvent) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event.__dict__, ensure_ascii=False) + "\n")


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
