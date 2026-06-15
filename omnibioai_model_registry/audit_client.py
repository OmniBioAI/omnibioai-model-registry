from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any


class AuditClient:
    def __init__(self, audit_url: str) -> None:
        self._audit_url = audit_url.rstrip("/") if audit_url else ""

    def log_event(
        self,
        action: str,
        actor: str,
        resource: str,
        task: str = "",
        model_name: str = "",
        version: str = "",
        metadata: dict = {},
    ) -> None:
        """Fire-and-forget. Never raises."""
        if not self._audit_url:
            return
        payload: dict[str, Any] = {
            "service": "model-registry",
            "action": action,
            "actor": actor,
            "resource": resource,
            "task": task,
            "model_name": model_name,
            "version": version,
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata,
        }
        t = threading.Thread(target=self._send, args=(payload,), daemon=True)
        t.start()

    def _send(self, payload: dict) -> None:
        """Called in daemon thread. Silently swallows all errors."""
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self._audit_url + "/events",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5):
                pass
        except Exception as exc:
            print(
                f"[AuditClient] warning: failed to send audit event: {exc}",
                file=sys.stderr,
            )
