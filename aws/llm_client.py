import json
import time
import logging
from typing import Any, Dict, Optional

import config

logger = logging.getLogger(__name__)

class OllamaClient:
    """Drop-in replacement for LLM API backends using local Ollama."""

    def __init__(self):
        from langchain_ollama import ChatOllama
        self.llm = ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url="http://localhost:11434",
            temperature=0.1,
            client_kwargs={"timeout": config.OLLAMA_REQUEST_TIMEOUT_SECONDS},
            sync_client_kwargs={"timeout": config.OLLAMA_REQUEST_TIMEOUT_SECONDS},
        )
        self.model_id = config.OLLAMA_MODEL

    def _extract_usage(self, response) -> Optional[Dict[str, int]]:
        usage = getattr(response, "usage_metadata", None)
        if isinstance(usage, dict):
            input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens")
            output_tokens = usage.get("output_tokens") or usage.get("completion_tokens")
            total_tokens = usage.get("total_tokens")

            if total_tokens is None and (
                input_tokens is not None or output_tokens is not None
            ):
                total_tokens = (input_tokens or 0) + (output_tokens or 0)

            if (
                input_tokens is not None
                or output_tokens is not None
                or total_tokens is not None
            ):
                return {
                    "input_tokens": int(input_tokens or 0),
                    "output_tokens": int(output_tokens or 0),
                    "total_tokens": int(total_tokens or 0),
                }

        response_metadata = getattr(response, "response_metadata", {}) or {}
        input_tokens = response_metadata.get("prompt_eval_count")
        output_tokens = response_metadata.get("eval_count")

        if input_tokens is None and output_tokens is None:
            return None

        return {
            "input_tokens": int(input_tokens or 0),
            "output_tokens": int(output_tokens or 0),
            "total_tokens": int((input_tokens or 0) + (output_tokens or 0)),
        }

    @staticmethod
    def _predict_options(max_tokens: int) -> Dict[str, int]:
        try:
            value = int(max_tokens)
        except Exception:
            value = 512
        # Avoid pathological long generations that can stall the pipeline.
        value = max(32, min(value, 1024))
        return {"num_predict": value}

    def invoke(self, prompt: str, max_tokens: int = 2048) -> str:
        return self.invoke_with_metadata(prompt, max_tokens)["content"]

    def invoke_with_metadata(
        self,
        prompt: str,
        max_tokens: int = 2048,
        *,
        json_mode: bool = False,
    ) -> Dict[str, Any]:
        invoke_kwargs: Dict[str, Any] = {
            "options": self._predict_options(max_tokens),
        }
        if json_mode:
            invoke_kwargs["format"] = "json"

        response = self.llm.invoke(prompt, **invoke_kwargs)
        return {
            "content": response.content,
            "usage": self._extract_usage(response),
            "model_id": self.model_id,
            "backend": "ollama",
        }

    def invoke_streaming(
        self,
        prompt: str,
        max_tokens: int = 2048,
        *,
        json_mode: bool = False,
    ):
        stream_kwargs: Dict[str, Any] = {
            "options": self._predict_options(max_tokens),
        }
        if json_mode:
            stream_kwargs["format"] = "json"

        for chunk in self.llm.stream(prompt, **stream_kwargs):
            if chunk.content:
                yield chunk.content


# Alias for backwards compatibility
LLMClient = OllamaClient
