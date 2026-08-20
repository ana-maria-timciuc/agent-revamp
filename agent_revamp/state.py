"""Conversation cache / state manager — JSON-file-backed session persistence.

One JSON file per session in STATE_DIR, following the "by JSON template" shape:

    {
        "session_id": "...",
        "created_at": "ISO timestamp",
        "updated_at": "ISO timestamp",
        "model": "gpt-5",
        "messages": [{role, content, ...}, ...]
    }

Writes are atomic (temp file + rename). Corrupt or missing files are treated
as absent rather than raising, so a bad state file never blocks a session.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _session_path(state_dir: Path, session_id: str) -> Path:
    safe_id = Path(session_id).name
    return state_dir / f"{safe_id}.json"


class SessionStore:
    def __init__(self, state_dir: str | Path = "state/"):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def list_sessions(self) -> list[dict]:
        sessions = []
        for path in sorted(self.state_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            data = self.load(path.stem)
            if data:
                sessions.append(
                    {
                        "session_id": data.get("session_id", path.stem),
                        "created_at": data.get("created_at", ""),
                        "updated_at": data.get("updated_at", ""),
                        "model": data.get("model", ""),
                        "message_count": len(data.get("messages", [])),
                    }
                )
        return sessions

    def load(self, session_id: str) -> dict | None:
        path = _session_path(self.state_dir, session_id)
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict) or "messages" not in data:
                raise ValueError("missing 'messages' key")
            return data
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            logger.warning("Ignoring corrupt session file %s: %s", path, exc)
            return None

    def save(self, session_id: str, messages: list[dict], model: str, created_at: str | None = None) -> None:
        data = {
            "session_id": session_id,
            "created_at": created_at or _now(),
            "updated_at": _now(),
            "model": model,
            "messages": messages,
        }
        path = _session_path(self.state_dir, session_id)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    def delete(self, session_id: str) -> bool:
        path = _session_path(self.state_dir, session_id)
        if path.exists():
            path.unlink()
            return True
        return False
