import json
import logging
import re
import time
from itertools import combinations
from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

import config
from aws.dynamodb_client import DynamoDBClient
from prompts.contradiction_prompt import (
    CLAIM_EXTRACTION_PROMPT,
    CONTRADICTION_ANALYSIS_PROMPT,
)
from utils.backend_logging import log_backend_event
from utils.llm_helpers import llm_call_with_retry


logger = logging.getLogger(__name__)


def _safe_json_parse(response: str, fallback: Dict = None) -> Dict:
    """
    Safely parse JSON from LLM response with aggressive cleanup.
    Handles markdown blocks, unescaped newlines, smart quotes, and malformed JSON.
    """
    if fallback is None:
        fallback = {}
    
    try:
        if isinstance(response, dict):
            return response
        
        if not isinstance(response, str):
            return fallback
        
        response = response.strip()
        
        # Remove markdown code blocks
        if "```json" in response:
            response = response.split("```json")[1]
        if "```" in response:
            response = response.split("```")[0]
        
        response = response.strip()
        
        # Find JSON boundaries (first { and last })
        start = response.find("{")
        end = response.rfind("}")
        
        if start == -1 or end == -1:
            return fallback
        
        json_str = response[start:end+1]
        
        # Fix smart quotes
        json_str = json_str.replace(""", '"').replace(""", '"')
        json_str = json_str.replace("'", "'").replace("'", "'")
        
        # Remove literal newlines and carriage returns
        json_str = json_str.replace('\\n', ' ').replace('\\r', ' ')
        
        # Try parsing
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # If still fails, try removing all newlines
            json_str = json_str.replace('\n', ' ').replace('\r', ' ')
            return json.loads(json_str)
    
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error: {e}")
        return fallback
    except Exception as e:
        logger.warning(f"Unexpected error in JSON parsing: {e}")
        return fallback


def _clean_source_content(text: str) -> str:
    cleaned = re.sub(r"<[^>]+>", " ", text or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[: config.CONTRADICTION_SOURCE_CONTENT_MAX_CHARS]


def extract_claims(source: Dict, session_id: str) -> List[Dict]:
    source_title = source.get("title", "")
    raw_content = (
        source.get("full_text_snippet")
        or source.get("abstract")
        or source.get("content")
        or ""
    )
    source_content = _clean_source_content(raw_content)

    try:
        prompt = CLAIM_EXTRACTION_PROMPT.format(
            source_title=source_title,
            source_content=source_content,
        )
        log_backend_event(
            "contradiction_claim_extraction_prompt",
            session_id=session_id,
            source_title=source_title,
            rendered_prompt=prompt,
        )

        response = llm_call_with_retry(
            prompt=prompt,
            session_id=session_id,
            step_name="contradiction_claim_extraction",
            fallback={"claims": []},
            max_tokens=768,
        )
        
        # Use safe JSON parser
        parsed = _safe_json_parse(response, fallback={"claims": []})
        claims = parsed.get("claims", [])
        if not isinstance(claims, list):
            claims = []
        claims = claims[: config.CONTRADICTION_MAX_CLAIMS_PER_SOURCE]

        log_backend_event(
            "contradiction_claim_extraction_completed",
            session_id=session_id,
            source_title=source_title,
            claims=claims,
        )
        return claims

    except Exception as e:
        logger.warning(f"Claim extraction failed: {e}")
        log_backend_event(
            "contradiction_claim_extraction_failed",
            session_id=session_id,
            source_title=source_title,
            error=str(e),
        )
        return []


def analyze_contradiction(
    claim_a: Dict,
    source_a: Dict,
    claim_b: Dict,
    source_b: Dict,
    session_id: str,
) -> Optional[Dict]:
    try:
        prompt = CONTRADICTION_ANALYSIS_PROMPT.format(
            claim_a=claim_a["claim_text"],
            source_a_title=source_a.get("title", ""),
            claim_b=claim_b["claim_text"],
            source_b_title=source_b.get("title", ""),
        )
        log_backend_event(
            "contradiction_analysis_prompt",
            session_id=session_id,
            source_a_title=source_a.get("title", ""),
            source_b_title=source_b.get("title", ""),
            rendered_prompt=prompt,
        )

        response = llm_call_with_retry(
            prompt=prompt,
            session_id=session_id,
            step_name="contradiction_analysis",
            fallback={"is_contradiction": False},
            max_tokens=512,
        )
        
        # Use safe JSON parser
        parsed = _safe_json_parse(response, fallback={"is_contradiction": False})

        if parsed.get("is_contradiction"):
            result = {
                "claim_a": claim_a,
                "source_a": source_a.get("title"),
                "claim_b": claim_b,
                "source_b": source_b.get("title"),
                "severity": parsed.get("severity", "none"),
                "contradiction_type": parsed.get("contradiction_type", "none"),
                "explanation": parsed.get("explanation", ""),
                "resolution_hint": parsed.get("resolution_hint", ""),
            }
            log_backend_event(
                "contradiction_analysis_completed",
                session_id=session_id,
                contradiction=result,
            )
            return result

    except Exception as e:
        logger.warning(f"Contradiction analysis failed: {e}")
        log_backend_event(
            "contradiction_analysis_failed",
            session_id=session_id,
            error=str(e),
        )

    return None


def severity_rank(severity: str) -> int:
    return {"high": 3, "medium": 2, "low": 1, "none": 0}.get(severity, 0)


def contradiction_detector_node(state: dict) -> dict:
    """
    Detect contradictions between sources.
    Currently simplified to avoid timeout issues - can be re-enabled after optimization.
    """
    sources = state.get("retrieved_sources", [])
    session_id = state.get("session_id", "unknown")

    if not sources or len(sources) < 3:
        # Skip with insufficient sources
        return {
            "contradictions": [],
            "current_step": "contradictions_detected",
        }

    # TEMPORARILY DISABLED: Contradiction detection is too slow
    # It will be re-enabled after optimization
    logger.info("Contradiction detection temporarily disabled (performance optimization in progress)")
    
    log_backend_event(
        "contradiction_detector_skipped",
        session_id=session_id,
        reason="performance_optimization",
        source_count=len(sources),
    )

    return {
        "contradictions": [],
        "current_step": "contradictions_detected",
    }
