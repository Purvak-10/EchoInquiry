import json
from typing import Dict, Any, List

from prompts.synthesis_prompt import SYNTHESIS_PROMPT
from utils.backend_logging import log_backend_event
from utils.llm_helpers import llm_call_with_retry


def _prepare_top_sources(sources: List[Dict]) -> List[Dict]:
    """
    Select and compress top 10 sources.
    """
    if not sources:
        return []

    # Sort by credibility_score (descending)
    sorted_sources = sorted(
        sources,
        key=lambda x: x.get("credibility_score", 0.0),
        reverse=True
    )

    top_sources = []
    for s in sorted_sources[:10]:
        top_sources.append({
            "title": s.get("title"),
            "abstract": (s.get("abstract") or "")[:300],
            "full_text_snippet": (s.get("full_text_snippet") or "")[:500],
            "credibility_score": round(s.get("credibility_score", 0.0), 3),
            "year": s.get("year"),
            "url": s.get("url", ""),
        })

    return top_sources


def _sources_are_relevant(sources: List[Dict], keywords: List[str]) -> bool:
    """
    Return False when none of the top sources contain ANY query keyword.
    This catches cases where the retriever only found off-topic papers.
    """
    if not sources or not keywords:
        return bool(sources)

    meaningful = [k.lower() for k in keywords if len(k) > 3]
    if not meaningful:
        return True

    for src in sources[:10]:
        text = (
            (src.get("title") or "") + " " +
            (src.get("abstract") or "") + " " +
            (src.get("full_text_snippet") or "")
        ).lower()
        if any(kw in text for kw in meaningful):
            return True
    return False


def _prepare_hypotheses(hypotheses: List[Dict]) -> List[Dict]:
    """
    Normalize hypothesis structure.
    """
    if not hypotheses:
        return []

    formatted = []
    for h in hypotheses:
        formatted.append({
            "statement": h.get("statement"),
            "status": h.get("status"),
            "confidence_posterior": h.get("confidence_posterior"),
            "verdict": h.get("verdict")
        })

    return formatted


def _prepare_contradictions(contradictions: List[Dict]) -> List[Dict]:
    """
    Select top 5 contradictions by severity.
    """
    if not contradictions:
        return []

    sorted_contra = sorted(
        contradictions,
        key=lambda x: x.get("severity", 0),
        reverse=True
    )

    top = []
    for c in sorted_contra[:5]:
        top.append({
            "claim_a": c.get("claim_a"),
            "claim_b": c.get("claim_b"),
            "severity": c.get("severity"),
            "explanation": c.get("explanation")
        })

    return top


def _safe_json_loads(text: str) -> Dict[str, Any]:
    """
    Robust JSON parser with fallback.
    """
    try:
        return json.loads(text)
    except Exception:
        return {}


def _clamp_confidence(score: float) -> float:
    """
    Ensure confidence_score is within [0,1]
    """
    try:
        return max(0.0, min(1.0, float(score)))
    except Exception:
        return 0.0


def _fallback_response() -> Dict[str, Any]:
    """
    Low-confidence fallback when synthesis fails.
    """
    return {
        "consensus": "Insufficient evidence to form a reliable conclusion.",
        "confidence_score": 0.2,
        "evidence_weight_map": {
            "strong_support": [],
            "weak_support": [],
            "contested": ["Conflicting or insufficient data across sources"],
            "unsupported": []
        },
        "key_findings": [],
        "outliers": [],
        "limitations": ["Limited or low-quality data", "High uncertainty"],
        "research_gaps": ["More high-quality studies needed"]
    }


def synthesis_engine_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: Synthesis Engine
    """

    try:
        hypotheses = state.get("hypotheses", [])
        sources = state.get("retrieved_sources", [])
        contradictions = state.get("contradictions", [])
        parsed_query = state.get("parsed_query", {})
        session_id = state.get("session_id")

        core_question = parsed_query.get("core_question", "")
        keywords = parsed_query.get("keywords", [])

        # ---- Guard: no sources or all sources irrelevant ----
        if not sources or not _sources_are_relevant(sources, keywords):
            log_backend_event(
                "synthesis_engine_no_relevant_sources",
                session_id=session_id,
                source_count=len(sources),
                core_question=core_question,
            )
            fallback = _fallback_response()
            fallback["consensus"] = (
                f"No relevant sources were found to answer: '{core_question}'. "
                "The retriever did not return results that match this query. "
                "This may be a general-knowledge or pop-culture question not well "
                "indexed by academic databases. Please try rephrasing or adding more context."
            )
            fallback["confidence_score"] = 0.1
            return {
                "synthesis": fallback,
                "current_step": "synthesis_complete",
            }

        # ---- Prepare inputs ----
        top_sources = _prepare_top_sources(sources)
        formatted_hypotheses = _prepare_hypotheses(hypotheses)
        top_contradictions = _prepare_contradictions(contradictions)

        # ---- Build prompt ----
        prompt = SYNTHESIS_PROMPT.format(
            core_question=core_question,
            hypotheses_json=json.dumps(formatted_hypotheses, indent=2),
            top_sources_json=json.dumps(top_sources, indent=2),
            contradictions_json=json.dumps(top_contradictions, indent=2)
        )
        log_backend_event(
            "synthesis_engine_rendered_prompt",
            session_id=session_id,
            core_question=core_question,
            rendered_prompt=prompt,
        )

        # ---- LLM call ----
        response = llm_call_with_retry(
            prompt=prompt,
            step_name="synthesis_engine",
            session_id=session_id,
            fallback=_fallback_response(),
        )

        result = _safe_json_loads(response) if isinstance(response, str) else response

        # ---- Validate & normalize ----
        if not result:
            result = _fallback_response()

        result["confidence_score"] = _clamp_confidence(
            result.get("confidence_score", 0.0)
        )

        # Ensure required keys exist
        result.setdefault("consensus", "")
        result.setdefault("evidence_weight_map", {})
        result.setdefault("key_findings", [])
        result.setdefault("outliers", [])
        result.setdefault("limitations", [])
        result.setdefault("research_gaps", [])
        log_backend_event(
            "synthesis_engine_completed",
            session_id=session_id,
            synthesis=result,
        )

        return {
            "synthesis": result,
            "current_step": "synthesis_complete"
        }

    except Exception as e:
        log_backend_event(
            "synthesis_engine_failed",
            session_id=state.get("session_id", "unknown"),
            error=str(e),
        )
        return {
            "synthesis": _fallback_response(),
            "current_step": "synthesis_complete",
            "error": str(e)
        }
