import json
import time
import logging

import config
from aws.llm_client import OllamaClient
from utils.backend_logging import log_backend_event
from utils.llm_usage import record_token_usage


logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Singleton
# ---------------------------------------------------------

_client_instances = {}


def get_llm_client(backend: str | None = None):
    # Always force Ollama
    backend = "ollama"

    if backend not in _client_instances:
        _client_instances[backend] = OllamaClient()

    return _client_instances[backend]


def _invoke_client_with_metadata(
    client,
    backend: str,
    prompt: str,
    max_tokens: int,
    *,
    session_id: str | None = None,
    step_name: str | None = None,
    json_mode: bool = False,
):
    log_backend_event(
        "llm_invoke_attempt_started",
        session_id=session_id,
        step_name=step_name,
        backend=backend,
        model_id=getattr(client, "model_id", backend),
        max_tokens=max_tokens,
        prompt_text=prompt,
    )

    if hasattr(client, "invoke_with_metadata"):
        result = client.invoke_with_metadata(
            prompt,
            max_tokens,
            json_mode=json_mode,
        )
        if isinstance(result, dict):
            result.setdefault("backend", backend)
            result.setdefault("model_id", getattr(client, "model_id", backend))
            result.setdefault("usage", None)
            log_backend_event(
                "llm_invoke_attempt_completed",
                session_id=session_id,
                step_name=step_name,
                backend=result.get("backend", backend),
                model_id=result.get("model_id"),
                max_tokens=max_tokens,
                prompt_text=prompt,
                response_text=result.get("content", ""),
                usage=result.get("usage"),
            )
            return result

    result = {
        "content": client.invoke(prompt, max_tokens),
        "usage": None,
        "model_id": getattr(client, "model_id", backend),
        "backend": backend,
    }
    log_backend_event(
        "llm_invoke_attempt_completed",
        session_id=session_id,
        step_name=step_name,
        backend=result["backend"],
        model_id=result["model_id"],
        max_tokens=max_tokens,
        prompt_text=prompt,
        response_text=result["content"],
        usage=result["usage"],
    )
    return result


def _invoke_with_failover(
    prompt: str,
    max_tokens: int = 2048,
    *,
    session_id: str | None = None,
    step_name: str | None = None,
    json_mode: bool = False,
) -> dict:
    primary_backend = "ollama"
    primary_client = get_llm_client(primary_backend)

    return _invoke_client_with_metadata(
        primary_client,
        primary_backend,
        prompt,
        max_tokens,
        session_id=session_id,
        step_name=step_name,
        json_mode=json_mode,
    )


# ---------------------------------------------------------
# Logging helper
# ---------------------------------------------------------

def _log_backend(level: str, message: str, **kwargs):
    """
    Structured log for backend events.
    Logs to standard logger.
    """
    log_payload = {
        "level": level,
        "message": message,
        **kwargs,
    }

    if level == "ERROR":
        logger.error(json.dumps(log_payload))
    else:
        logger.info(json.dumps(log_payload))


# ---------------------------------------------------------
# LLM CALL WITH RETRY
# ---------------------------------------------------------

def llm_call_with_retry(
    prompt: str,
    session_id: str,
    step_name: str,
    parse_fn=json.loads,
    fallback=None,
    max_tokens: int = 2048,
    json_mode: bool = True,
):
    start_time = time.time()
    log_backend_event(
        "llm_call_started",
        session_id=session_id,
        step_name=step_name,
        configured_backend=config.LLM_BACKEND,
        max_tokens=max_tokens,
        prompt_text=prompt,
    )

    for attempt in range(1, config.LLM_MAX_RETRIES + 1):
        try:
            response = _invoke_with_failover(
                prompt,
                max_tokens,
                session_id=session_id,
                step_name=step_name,
                json_mode=json_mode,
            )
            raw = response["content"]
            elapsed = round(time.time() - start_time, 2)

            usage_event = record_token_usage(
                session_id=session_id,
                step_name=step_name,
                backend=response.get("backend", config.LLM_BACKEND),
                model_id=response.get("model_id"),
                usage=response.get("usage"),
                max_tokens=max_tokens,
                attempt=attempt,
                duration_seconds=elapsed,
                prompt_text=prompt,
                response_text=raw,
            )

            content = raw.strip()

            # strip markdown fences
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1])
                if content.startswith("json"):
                    content = content[4:].strip()

            result = parse_fn(content)
            log_backend_event(
                "llm_call_succeeded",
                session_id=session_id,
                step_name=step_name,
                attempt=attempt,
                backend=response.get("backend", config.LLM_BACKEND),
                model_id=response.get("model_id"),
                max_tokens=max_tokens,
                duration_seconds=elapsed,
                usage=usage_event,
                prompt_text=prompt,
                response_text=raw,
                parsed_response=result,
            )

            _log_backend(
                "INFO",
                "LLM call success",
                session_id=session_id,
                step=step_name,
                attempt=attempt,
                duration_seconds=elapsed,
                backend=response.get("backend", config.LLM_BACKEND),
                model_id=response.get("model_id"),
                input_tokens=usage_event.get("input_tokens"),
                output_tokens=usage_event.get("output_tokens"),
                total_tokens=usage_event.get("total_tokens"),
            )

            return result

        except Exception as e:
            log_backend_event(
                "llm_call_attempt_failed",
                session_id=session_id,
                step_name=step_name,
                attempt=attempt,
                configured_backend=config.LLM_BACKEND,
                max_tokens=max_tokens,
                prompt_text=prompt,
                error=str(e),
            )
            logging.warning(
                f"LLM attempt {attempt}/{config.LLM_MAX_RETRIES} "
                f"failed at {step_name}: {e}"
            )

            if attempt < config.LLM_MAX_RETRIES:
                time.sleep(config.LLM_RETRY_DELAY_SECONDS)
            else:
                _log_backend(
                    "ERROR",
                    "All LLM retries failed",
                    session_id=session_id,
                    step=step_name,
                    error=str(e),
                )
                log_backend_event(
                    "llm_call_failed_final",
                    session_id=session_id,
                    step_name=step_name,
                    configured_backend=config.LLM_BACKEND,
                    max_tokens=max_tokens,
                    prompt_text=prompt,
                    error=str(e),
                    fallback=fallback,
                )

    return fallback


# ---------------------------------------------------------
# STREAMING HELPER
# ---------------------------------------------------------

def _fallback_backend(primary: str) -> str | None:
    """No fallback — Ollama is the only backend."""
    return None


def llm_stream(prompt: str):
    """Used by output generator agent."""
    client = get_llm_client("ollama")
    yield from client.invoke_streaming(prompt)
