from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT_STR = str(PROJECT_ROOT)

if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

import config
from utils.llm_usage import _get_langfuse_client


logger = logging.getLogger(__name__)


def _backend_log_path() -> Path:
    """Generate log path with date as filename: logs/YYYY-MM-DD.jsonl"""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    logs_dir = Path(config.PROJECT_ROOT) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir / f"backend_events_{today}.jsonl"


def _sanitize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = dict(payload)

    if not config.BACKEND_LOG_INCLUDE_CONTENT:
        for key in (
            "prompt_text",
            "response_text",
            "rendered_prompt",
            "raw_response",
            "parsed_response",
        ):
            if key in sanitized:
                value = sanitized[key]
                if isinstance(value, str):
                    sanitized[key] = {
                        "captured": False,
                        "chars": len(value),
                    }
                else:
                    sanitized[key] = {
                        "captured": False,
                        "type": type(value).__name__,
                    }

    return sanitized


def _event_trace_context(event: Dict[str, Any]) -> Dict[str, str] | None:
    client = _get_langfuse_client()
    session_id = event.get("session_id")

    if client is None or not session_id or not hasattr(client, "create_trace_id"):
        return None

    try:
        return {"trace_id": client.create_trace_id(seed=str(session_id))}
    except Exception:
        logger.exception("Failed to create Langfuse trace context")
        return None


def _emit_backend_event_to_langfuse(event: Dict[str, Any]) -> None:
    client = _get_langfuse_client()
    if client is None:
        return

    try:
        metadata = {
            "timestamp_utc": event.get("timestamp_utc"),
            "session_id": event.get("session_id"),
        }
        event_input = {}
        event_output = {}

        for key, value in event.items():
            if key in {"event_type", "timestamp_utc", "session_id"}:
                continue

            if key in {"prompt_text", "rendered_prompt", "raw_query"}:
                event_input[key] = value
            elif key in {"response_text", "raw_response", "parsed_response"}:
                event_output[key] = value
            else:
                metadata[key] = value

        trace_context = _event_trace_context(event)
        client.create_event(
            trace_context=trace_context,
            name=event["event_type"],
            input=event_input or None,
            output=event_output or None,
            metadata=metadata,
        )
        client.flush()
    except Exception:
        logger.exception("Failed to emit backend event to Langfuse")


def log_backend_event(event_type: str, **payload: Any) -> Dict[str, Any]:
    event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        **_sanitize_payload(payload),
    }

    path = _backend_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=True, default=str) + "\n")

    _emit_backend_event_to_langfuse(event)
    logger.debug("Backend event logged: %s", event_type)
    return event
