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
_embedder = None


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


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return _embedder


def _source_identifier(source: Dict) -> str:
    return str(
        source.get("source_id")
        or source.get("doi")
        or source.get("url")
        or source.get("title")
        or ""
    )


def _claim_identifier(claim: Dict, index: int) -> str:
    return str(claim.get("claim_id") or f"claim_{index + 1}")


def _claim_text(claim: Dict) -> str:
    return str(claim.get("claim_text") or "").strip()


def _token_set(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9]+", (text or "").lower())
        if len(token) >= 4
    }


def _polarity_score(text: str) -> int:
    positive_terms = {
        "increase", "increases", "improve", "improves", "improved",
        "higher", "supports", "benefit", "benefits", "enhance",
        "enhances", "more", "greater", "effective",
    }
    negative_terms = {
        "decrease", "decreases", "decreased", "reduce", "reduces",
        "reduced", "lower", "worse", "impair", "impairs", "impaired",
        "no", "not", "none", "without", "lack", "lacks", "ineffective",
    }
    tokens = _token_set(text)
    positive_hits = len(tokens & positive_terms)
    negative_hits = len(tokens & negative_terms)
    if positive_hits > negative_hits:
        return 1
    if negative_hits > positive_hits:
        return -1
    return 0


def _cosine_similarity(vector_a, vector_b) -> float:
    array_a = np.asarray(vector_a, dtype=float)
    array_b = np.asarray(vector_b, dtype=float)
    denom = np.linalg.norm(array_a) * np.linalg.norm(array_b)
    if denom == 0:
        return 0.0
    return float(np.dot(array_a, array_b) / denom)


def _build_claim_entries(sources: List[Dict], session_id: str, started_at: float) -> List[Dict]:
    claim_entries: List[Dict] = []

    for source in sources:
        if (time.monotonic() - started_at) > config.CONTRADICTION_MAX_STAGE_SECONDS:
            break

        claims = extract_claims(source, session_id)
        for index, claim in enumerate(claims):
            claim_text = _claim_text(claim)
            if not claim_text:
                continue
            claim_entries.append(
                {
                    "claim_id": _claim_identifier(claim, index),
                    "claim": claim,
                    "source": source,
                    "source_id": _source_identifier(source),
                    "claim_text": claim_text,
                    "token_set": _token_set(claim_text),
                    "polarity": _polarity_score(claim_text),
                }
            )

    return claim_entries


def _rank_candidate_pairs(claim_entries: List[Dict], started_at: float) -> List[Dict]:
    if len(claim_entries) < 2:
        return []

    embeddings = _get_embedder().encode(
        [entry["claim_text"] for entry in claim_entries]
    )

    candidate_pairs: List[Dict] = []
    for left_index, right_index in combinations(range(len(claim_entries)), 2):
        if (time.monotonic() - started_at) > config.CONTRADICTION_MAX_STAGE_SECONDS:
            break

        left = claim_entries[left_index]
        right = claim_entries[right_index]

        if left["source_id"] == right["source_id"]:
            continue

        similarity = _cosine_similarity(embeddings[left_index], embeddings[right_index])
        if similarity < config.CONTRADICTION_SIMILARITY_THRESHOLD:
            continue

        term_overlap = len(left["token_set"] & right["token_set"])
        polarity_conflict = (
            left["polarity"] != 0
            and right["polarity"] != 0
            and left["polarity"] != right["polarity"]
        )

        score = similarity
        if term_overlap:
            score += min(term_overlap, 3) * 0.03
        if polarity_conflict:
            score += 0.08

        candidate_pairs.append(
            {
                "score": score,
                "similarity": similarity,
                "term_overlap": term_overlap,
                "polarity_conflict": polarity_conflict,
                "left": left,
                "right": right,
            }
        )

    candidate_pairs.sort(key=lambda item: item["score"], reverse=True)
    return candidate_pairs[: config.CONTRADICTION_MAX_CANDIDATE_PAIRS]


def contradiction_detector_node(state: dict) -> dict:
    """
    Detect contradictions between high-credibility sources with bounded
    claim extraction, embedding-based candidate pruning, and capped LLM checks.
    """
    sources = state.get("retrieved_sources", [])
    session_id = state.get("session_id", "unknown")
    fast_mode = bool(state.get("fast_mode"))

    if fast_mode or not sources or len(sources) < 3:
        return {
            "contradictions": [],
            "current_step": "contradictions_detected",
        }

    started_at = time.monotonic()

    try:
        ranked_sources = sorted(
            sources,
            key=lambda source: source.get("credibility_score", 0),
            reverse=True,
        )[: config.CONTRADICTION_MAX_SOURCES]

        claim_entries = _build_claim_entries(ranked_sources, session_id, started_at)
        if len(claim_entries) < 2:
            log_backend_event(
                "contradiction_detector_completed",
                session_id=session_id,
                source_count=len(ranked_sources),
                claim_count=len(claim_entries),
                contradiction_count=0,
                reason="insufficient_claims",
            )
            return {
                "contradictions": [],
                "current_step": "contradictions_detected",
            }

        candidate_pairs = _rank_candidate_pairs(claim_entries, started_at)
        contradictions: List[Dict] = []
        seen_pairs = set()

        for candidate in candidate_pairs[: config.CONTRADICTION_MAX_ANALYSIS_PAIRS]:
            if (time.monotonic() - started_at) > config.CONTRADICTION_MAX_STAGE_SECONDS:
                break

            left = candidate["left"]
            right = candidate["right"]
            pair_key = tuple(
                sorted(
                    [
                        (left["source_id"], left["claim_text"]),
                        (right["source_id"], right["claim_text"]),
                    ]
                )
            )
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            contradiction = analyze_contradiction(
                left["claim"],
                left["source"],
                right["claim"],
                right["source"],
                session_id,
            )
            if contradiction:
                contradictions.append(contradiction)

        contradictions.sort(
            key=lambda item: severity_rank(item.get("severity", "none")),
            reverse=True,
        )

        log_backend_event(
            "contradiction_detector_completed",
            session_id=session_id,
            source_count=len(ranked_sources),
            claim_count=len(claim_entries),
            candidate_pair_count=len(candidate_pairs),
            contradiction_count=len(contradictions),
            duration_seconds=round(time.monotonic() - started_at, 2),
        )

    except Exception as error:
        logger.warning(f"Contradiction detector failed: {error}")
        log_backend_event(
            "contradiction_detector_failed",
            session_id=session_id,
            error=str(error),
        )
        contradictions = []

    return {
        "contradictions": contradictions,
        "current_step": "contradictions_detected",
    }
