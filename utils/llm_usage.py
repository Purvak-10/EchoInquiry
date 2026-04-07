from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT_STR = str(PROJECT_ROOT)

if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

import config


logger = logging.getLogger(__name__)


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_usage(raw_usage: Dict[str, Any] | None) -> Dict[str, Optional[int]]:
    if not isinstance(raw_usage, dict):
        return {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
        }

    input_tokens = _safe_int(raw_usage.get("input_tokens"))
    output_tokens = _safe_int(raw_usage.get("output_tokens"))
    total_tokens = _safe_int(raw_usage.get("total_tokens"))

    if total_tokens is None and (
        input_tokens is not None or output_tokens is not None
    ):
        total_tokens = (input_tokens or 0) + (output_tokens or 0)

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _usage_log_path() -> Path:
    """Generate log path with date as filename: logs/YYYY-MM-DD_llm_usage.jsonl"""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    logs_dir = Path(config.PROJECT_ROOT) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir / f"llm_usage_{today}.jsonl"

def _append_jsonl(record: Dict[str, Any]) -> None:
    path = _usage_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True, default=str) + "\n")


_langfuse_client = None
_langfuse_disabled_logged = False


def _get_langfuse_client():
    global _langfuse_client
    global _langfuse_disabled_logged

    if not config.LANGFUSE_ENABLED:
        return None

    if not config.LANGFUSE_PUBLIC_KEY or not config.LANGFUSE_SECRET_KEY:
        if not _langfuse_disabled_logged:
            logger.warning(
                "LANGFUSE_ENABLED is true, but credentials are missing."
            )
            _langfuse_disabled_logged = True
        return None

    if _langfuse_client is not None:
        return _langfuse_client

    try:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            public_key=config.LANGFUSE_PUBLIC_KEY,
            secret_key=config.LANGFUSE_SECRET_KEY,
            host=config.LANGFUSE_HOST,
        )
        return _langfuse_client
    except Exception:
        if not _langfuse_disabled_logged:
            logger.exception("Langfuse client initialization failed")
            _langfuse_disabled_logged = True
        return None


def _session_trace_context(client, session_id: str | None):
    if not session_id or not hasattr(client, "create_trace_id"):
        return None

    try:
        return {"trace_id": client.create_trace_id(seed=str(session_id))}
    except Exception:
        logger.exception("Failed to create Langfuse session trace context")
        return None


def _emit_langfuse(
    event: Dict[str, Any],
    prompt_text: str,
    response_text: str,
) -> None:
    client = _get_langfuse_client()
    if client is None:
        return

    try:
        usage_details = {
            "prompt_tokens": event["input_tokens"] or 0,
            "completion_tokens": event["output_tokens"] or 0,
            "total_tokens": event["total_tokens"] or 0,
        }
        metadata = {
            "backend": event["backend"],
            "session_id": event["session_id"],
            "attempt": event["attempt"],
            "duration_seconds": event["duration_seconds"],
            "max_tokens": event["max_tokens"],
            "prompt_chars": event["prompt_chars"],
            "response_chars": event["response_chars"],
        }
        generation_input = (
            prompt_text
            if config.LANGFUSE_CAPTURE_CONTENT
            else {
                "session_id": event["session_id"],
                "step_name": event["step_name"],
                "prompt_chars": event["prompt_chars"],
            }
        )
        generation_output = (
            response_text
            if config.LANGFUSE_CAPTURE_CONTENT
            else {
                "response_chars": event["response_chars"],
                "captured_content": False,
            }
        )
        trace_context = _session_trace_context(client, event["session_id"])
        end_time = datetime.now(timezone.utc)
        duration_seconds = float(event.get("duration_seconds") or 0.0)
        start_time = end_time - timedelta(seconds=max(duration_seconds, 0.0))

        # Newer Langfuse Python SDKs use observation/generation spans.
        if hasattr(client, "start_observation"):
            obs = client.start_observation(
                trace_context=trace_context,
                name=event["step_name"],
                as_type="generation",
                input=generation_input,
                output=generation_output,
                metadata=metadata,
                model=event["model_id"] or event["backend"],
                model_parameters={
                    "max_tokens": event["max_tokens"],
                    "backend": event["backend"],
                },
                usage_details=usage_details,
            )
            # The SDK records start_time when the observation is created. Since we
            # only learn about the call after it completes, backfill start/end.
            obs.update(start_time=start_time)
            obs.end(end_time=int(end_time.timestamp() * 1000))
            client.flush()
            return

        # Backward compatibility for older SDKs that expose a trace API.
        if hasattr(client, "trace"):
            trace = client.trace(
                trace_context=trace_context,
                name=event["step_name"],
                session_id=event["session_id"],
                metadata=metadata,
            )

            trace.generation(
                name=event["step_name"],
                model=event["model_id"] or event["backend"],
                input=generation_input,
                output=generation_output,
                usage={
                    "input": event["input_tokens"],
                    "output": event["output_tokens"],
                    "total": event["total_tokens"],
                },
                metadata=metadata,
            )

            client.flush()
            return

        logger.warning(
            "Langfuse client does not expose a supported tracing API."
        )
    except Exception:
        logger.exception("Langfuse emission failed")


def record_token_usage(
    *,
    session_id: str,
    step_name: str,
    backend: str,
    model_id: str | None,
    usage: Dict[str, Any] | None,
    max_tokens: int,
    attempt: int,
    duration_seconds: float,
    prompt_text: str,
    response_text: str,
) -> Dict[str, Any]:
    normalized = normalize_usage(usage)
    now = datetime.now().astimezone()

    event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "local_date": now.date().isoformat(),
        "session_id": session_id,
        "step_name": step_name,
        "backend": backend,
        "model_id": model_id,
        "attempt": attempt,
        "max_tokens": max_tokens,
        "duration_seconds": round(duration_seconds, 3),
        "prompt_chars": len(prompt_text),
        "response_chars": len(response_text),
        "input_tokens": normalized["input_tokens"],
        "output_tokens": normalized["output_tokens"],
        "total_tokens": normalized["total_tokens"],
    }

    _append_jsonl(event)
    _emit_langfuse(event, prompt_text, response_text)
    return event


def get_daily_summary(target_date: str | None = None) -> Dict[str, Any]:
    target_date = target_date or datetime.now().astimezone().date().isoformat()
    path = _usage_log_path()

    summary = {
        "date": target_date,
        "calls": 0,
        "calls_missing_usage": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "by_step": {},
        "by_model": {},
    }

    if not path.exists():
        return summary

    by_step: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"calls": 0, "total_tokens": 0}
    )
    by_model: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"calls": 0, "total_tokens": 0}
    )

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            if event.get("local_date") != target_date:
                continue

            summary["calls"] += 1

            total_tokens = _safe_int(event.get("total_tokens"))
            input_tokens = _safe_int(event.get("input_tokens"))
            output_tokens = _safe_int(event.get("output_tokens"))

            if total_tokens is None:
                summary["calls_missing_usage"] += 1
            else:
                summary["input_tokens"] += input_tokens or 0
                summary["output_tokens"] += output_tokens or 0
                summary["total_tokens"] += total_tokens

            step_name = event.get("step_name") or "unknown"
            model_id = event.get("model_id") or event.get("backend") or "unknown"

            by_step[step_name]["calls"] += 1
            by_step[step_name]["total_tokens"] += total_tokens or 0

            by_model[model_id]["calls"] += 1
            by_model[model_id]["total_tokens"] += total_tokens or 0

    summary["by_step"] = dict(
        sorted(
            by_step.items(),
            key=lambda item: item[1]["total_tokens"],
            reverse=True,
        )
    )
    summary["by_model"] = dict(
        sorted(
            by_model.items(),
            key=lambda item: item[1]["total_tokens"],
            reverse=True,
        )
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show daily LLM token usage totals."
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Local date in YYYY-MM-DD format. Defaults to today.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the summary as JSON.",
    )
    args = parser.parse_args()

    summary = get_daily_summary(args.date)

    if args.json:
        print(json.dumps(summary, indent=2))
        return

    print(f"Date: {summary['date']}")
    print(f"Calls: {summary['calls']}")
    print(f"Calls missing usage: {summary['calls_missing_usage']}")
    print(f"Input tokens: {summary['input_tokens']}")
    print(f"Output tokens: {summary['output_tokens']}")
    print(f"Total tokens: {summary['total_tokens']}")

    if summary["by_step"]:
        print("\nTop steps:")
        for step_name, data in summary["by_step"].items():
            print(
                f"- {step_name}: {data['calls']} calls, "
                f"{data['total_tokens']} tokens"
            )

    if summary["by_model"]:
        print("\nBy model:")
        for model_id, data in summary["by_model"].items():
            print(
                f"- {model_id}: {data['calls']} calls, "
                f"{data['total_tokens']} tokens"
            )


if __name__ == "__main__":
    main()
