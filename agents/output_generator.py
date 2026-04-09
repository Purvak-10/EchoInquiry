import json
import logging
import re
import uuid
from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple

from aws.dynamodb_client import DynamoDBClient
from aws.s3_client import S3Client
from prompts.output_prompt import OUTPUT_PROMPT
from utils.backend_logging import log_backend_event
from utils.llm_helpers import llm_call_with_retry, llm_stream

logger = logging.getLogger(__name__)


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "which",
    "who",
    "why",
    "with",
}

_GENERIC_QUERY_TERMS = {
    "important",
    "better",
    "best",
    "compare",
    "comparison",
    "versus",
    "between",
    "more",
    "less",
}

_QUERY_TERM_ALIASES = {
    "food": ["diet", "dietary", "nutrition", "nutritional"],
    "diet": ["food", "dietary", "nutrition", "nutritional"],
    "nutrition": ["food", "diet", "dietary", "nutritional"],
    "exercise": ["physical activity", "fitness", "training"],
    "sleep": ["sleep quality", "sleep deprivation", "circadian"],
}

_ALLOWED_HYPOTHESIS_VERDICTS = {"supported", "weakly_supported", "contested", "unsupported"}
_ALLOWED_CONTRADICTION_SEVERITIES = {"low", "medium", "high"}

_PLACEHOLDER_PHRASES = [
    "[topic]",
    "[field]",
    "[outcome]",
    "title of source",
    "authors' names",
    "doi of the source",
    "url of the source",
    "string",
]


# ---------------------------
# Helper functions
# ---------------------------

def _safe_json_dumps(data: Any) -> str:
    try:
        return json.dumps(data, ensure_ascii=False)
    except Exception:
        return json.dumps({"error": "serialization_failed"})


def _strip_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _as_text(value: Any, max_chars: int = 1000) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    elif isinstance(value, (int, float, bool)):
        text = str(value)
    elif isinstance(value, dict):
        for key in ("statement", "summary", "content", "heading", "title", "question", "id"):
            if value.get(key):
                text = str(value.get(key))
                break
        else:
            text = _safe_json_dumps(value)
    elif isinstance(value, list):
        text = "; ".join(_as_text(item, max_chars=200) for item in value if item is not None)
    else:
        text = str(value)

    text = _strip_html(text)
    if max_chars > 0:
        return text[:max_chars].strip()
    return text.strip()


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _extract_query_terms(core_question: str, parsed_query: Dict[str, Any]) -> List[str]:
    terms: List[str] = []

    for token in _tokenize(core_question):
        if (
            len(token) >= 4
            and token not in _STOPWORDS
            and token not in _GENERIC_QUERY_TERMS
        ):
            terms.append(token)
            terms.extend(_QUERY_TERM_ALIASES.get(token, []))

    keywords = parsed_query.get("keywords")
    if isinstance(keywords, list):
        for kw in keywords:
            for token in _tokenize(_as_text(kw, max_chars=120)):
                if (
                    len(token) >= 4
                    and token not in _STOPWORDS
                    and token not in _GENERIC_QUERY_TERMS
                ):
                    terms.append(token)
                    terms.extend(_QUERY_TERM_ALIASES.get(token, []))

    # Deduplicate while preserving order.
    seen = set()
    unique_terms = []
    for term in terms:
        if term in seen:
            continue
        seen.add(term)
        unique_terms.append(term)

    return unique_terms[:10]


def _source_identifier(source: Dict[str, Any], idx: int) -> str:
    return str(
        source.get("source_id")
        or source.get("doi")
        or source.get("url")
        or source.get("title")
        or f"source_{idx + 1}"
    )


def _score_source_relevance(source: Dict[str, Any], query_terms: List[str]) -> int:
    if not query_terms:
        return 0
    haystack = " ".join(
        [
            _as_text(source.get("title"), max_chars=200),
            _as_text(source.get("abstract"), max_chars=600),
            _as_text(source.get("full_text_snippet"), max_chars=600),
            _as_text(source.get("journal"), max_chars=120),
        ]
    ).lower()
    return sum(1 for term in query_terms if term in haystack)


def _sanitize_source_for_prompt(source: Dict[str, Any], idx: int, query_terms: List[str]) -> Dict[str, Any]:
    source_id = _source_identifier(source, idx)
    title = _as_text(source.get("title"), max_chars=220)
    abstract = _as_text(source.get("abstract") or source.get("full_text_snippet"), max_chars=900)

    return {
        "source_id": source_id,
        "title": title,
        "authors": _as_text(source.get("authors"), max_chars=180),
        "year": source.get("year") or "",
        "doi": _as_text(source.get("doi"), max_chars=140),
        "url": _as_text(source.get("url"), max_chars=220),
        "journal": _as_text(source.get("journal"), max_chars=140),
        "abstract": abstract,
        "citation_count": int(source.get("citation_count") or 0),
        "credibility_score": round(float(source.get("credibility_score") or 0.0), 3),
        "retraction_status": _as_text(source.get("retraction_status"), max_chars=24) or "unknown",
        "relevance_score": _score_source_relevance(source, query_terms),
    }


def _sanitize_hypothesis_for_prompt(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
    verdict = _normalize_hypothesis_verdict(
        item.get("status"),
        item.get("verdict"),
    )

    confidence = item.get("confidence_posterior")
    try:
        confidence = round(float(confidence), 3)
    except Exception:
        confidence = None

    return {
        "id": _as_text(item.get("id"), max_chars=40) or f"h{idx + 1}",
        "statement": _as_text(item.get("statement"), max_chars=320),
        "verdict": verdict,
        "summary": _summarize_hypothesis(item, max_chars=260),
        "confidence_posterior": confidence,
    }


def _sanitize_contradiction_for_prompt(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
    severity_raw = _as_text(item.get("severity"), max_chars=16).lower()
    severity = severity_raw if severity_raw in _ALLOWED_CONTRADICTION_SEVERITIES else "low"

    return {
        "id": f"c{idx + 1}",
        "summary": _as_text(item.get("summary") or item.get("explanation"), max_chars=320),
        "severity": severity,
        "action": _as_text(item.get("action") or item.get("resolution_hint"), max_chars=220),
    }


def _prepare_inputs(state: Dict[str, Any]) -> Dict[str, Any]:
    synthesis = state.get("synthesis", {}) or {}
    hypotheses = state.get("hypotheses", []) or []
    contradictions = state.get("contradictions", []) or []
    sources = state.get("retrieved_sources", []) or []
    parsed_query = state.get("parsed_query", {}) or {}

    core_question = parsed_query.get("core_question", state.get("raw_query", ""))
    query_terms = _extract_query_terms(core_question, parsed_query)

    sanitized_sources = [
        _sanitize_source_for_prompt(source, idx, query_terms)
        for idx, source in enumerate(sources)
        if isinstance(source, dict)
    ]
    top_sources = sorted(
        sanitized_sources,
        key=lambda x: (
            x.get("relevance_score", 0),
            x.get("credibility_score", 0),
            x.get("citation_count", 0),
        ),
        reverse=True,
    )[:10]

    sanitized_hypotheses = [
        _sanitize_hypothesis_for_prompt(item, idx)
        for idx, item in enumerate(hypotheses)
        if isinstance(item, dict)
    ][:8]

    sanitized_contradictions = [
        _sanitize_contradiction_for_prompt(item, idx)
        for idx, item in enumerate(contradictions)
        if isinstance(item, dict)
    ]
    top_contradictions = sorted(
        sanitized_contradictions,
        key=lambda x: {"high": 3, "medium": 2, "low": 1}.get(x.get("severity", "low"), 0),
        reverse=True,
    )[:5]

    return {
        "core_question": core_question,
        "query_terms": query_terms,
        "keywords": parsed_query.get("keywords", []),
        "synthesis": synthesis,
        "hypotheses": sanitized_hypotheses,
        "hypotheses_verdict": sanitized_hypotheses,
        "contradictions": top_contradictions,
        "retrieved_sources": sanitized_sources,
        "top_sources": top_sources,
        "synthesis_json": _safe_json_dumps(synthesis),
        "hypotheses_json": _safe_json_dumps(sanitized_hypotheses),
        "contradictions_json": _safe_json_dumps(top_contradictions),
        "top_sources_json": _safe_json_dumps(top_sources),
        "output_format": parsed_query.get("output_format", "research_brief"),
    }


def _ensure_generated_at(report: Dict[str, Any]) -> Dict[str, Any]:
    if "generated_at" not in report or not report["generated_at"]:
        report["generated_at"] = datetime.utcnow().isoformat()
    return report


def _clamp_confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _calculate_final_confidence(prepared: Dict[str, Any]) -> float:
    """
    Calculate final confidence score based on:
    - Source quality (credibility scores) — 40%
    - Evidence consistency (contradiction detection) — 30%
    - Hypothesis support distribution — 20%
    - Synthesis confidence — 10%
    """
    try:
        sources = prepared.get("retrieved_sources", [])
        contradictions = prepared.get("contradictions", [])
        synthesis = prepared.get("synthesis", {})
        
        # ===== SOURCE QUALITY COMPONENT (40%) =====
        if sources:
            credibility_scores = [s.get("credibility_score", 0.5) for s in sources]
            avg_credibility = sum(credibility_scores) / len(credibility_scores)
            # Boost: if we have decent sources, be more confident
            source_confidence = min(avg_credibility * 1.2, 1.0)
        else:
            source_confidence = 0.3
        
        # ===== CONSISTENCY COMPONENT (30%) =====
        if not contradictions or len(contradictions) == 0:
            consistency_confidence = 0.9  # High confidence when no contradictions
        elif len(contradictions) <= 2:
            consistency_confidence = 0.7
        else:
            consistency_confidence = 0.5
        
        # ===== HYPOTHESIS SUPPORT COMPONENT (20%) =====
        hypotheses_verdict = prepared.get("hypotheses_verdict", [])
        if hypotheses_verdict:
            verdicts_list = [h.get("verdict", "").lower() for h in hypotheses_verdict]
            supported = sum(1 for v in verdicts_list if "support" in v)
            contested = sum(1 for v in verdicts_list if "contest" in v)
            total = len(verdicts_list)
            
            if total > 0:
                support_ratio = supported / total
                
                if support_ratio >= 0.7:
                    hypothesis_confidence = 0.85
                elif support_ratio >= 0.5:
                    hypothesis_confidence = 0.65
                elif support_ratio >= 0.3:
                    hypothesis_confidence = 0.50
                else:
                    hypothesis_confidence = 0.35
            else:
                hypothesis_confidence = 0.5
        else:
            hypothesis_confidence = 0.5
        
        # ===== SYNTHESIS CONFIDENCE COMPONENT (10%) =====
        synthesis_conf_level = synthesis.get("confidence_level", "MODERATE").upper()
        
        if synthesis_conf_level == "HIGH":
            synthesis_confidence = 0.9
        elif synthesis_conf_level == "MODERATE":
            synthesis_confidence = 0.65  # Default to reasonable confidence
        else:
            synthesis_confidence = 0.4
        
        # ===== FINAL CALCULATION =====
        final_confidence = (
            source_confidence * 0.40 +
            consistency_confidence * 0.30 +
            hypothesis_confidence * 0.20 +
            synthesis_confidence * 0.10
        )
        
        # Ensure valid range [0.1, 1.0] - minimum 10% confidence for any result
        final_confidence = max(0.1, min(1.0, final_confidence))
        
        return final_confidence
        
    except Exception as e:
        logger.error(f"Error calculating confidence: {e}")
        return 0.5  # Default to medium confidence on error


def _iter_text_values(report: Dict[str, Any]) -> Iterable[str]:
    yield _as_text(report.get("title"), max_chars=500)
    yield _as_text(report.get("executive_summary"), max_chars=3000)

    sections = report.get("sections") if isinstance(report.get("sections"), list) else []
    for section in sections:
        if not isinstance(section, dict):
            continue
        yield _as_text(section.get("heading"), max_chars=500)
        yield _as_text(section.get("content"), max_chars=3000)

    for key in ("key_conclusions", "research_gaps", "follow_up_questions"):
        items = report.get(key) if isinstance(report.get(key), list) else []
        for item in items:
            yield _as_text(item, max_chars=700)


def _contains_placeholders(text: str) -> bool:
    if not text:
        return False

    lower = text.lower()
    if any(phrase in lower for phrase in _PLACEHOLDER_PHRASES):
        return True

    # Avoid unresolved template chunks like [entity_name].
    if re.search(r"\[[a-zA-Z_\- ]{2,}\]", text):
        return True

    return False


def _normalize_hypothesis_verdict(status: Any, verdict: Any) -> str:
    status_text = _as_text(status, max_chars=40).lower()
    verdict_text = _as_text(verdict, max_chars=40).lower()

    if status_text in _ALLOWED_HYPOTHESIS_VERDICTS:
        return status_text
    if status_text == "supported":
        return "supported"
    if status_text in {"partially_supported", "weakly_supported"}:
        return "weakly_supported"
    if status_text in {"falsified", "refuted", "unsupported"}:
        return "unsupported"
    if status_text in {"insufficient_evidence", "mixed", "unverified"}:
        return "contested"
    if verdict_text in _ALLOWED_HYPOTHESIS_VERDICTS:
        return verdict_text
    return "contested"


def _looks_structured(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    return (
        (stripped.startswith("{") and stripped.endswith("}"))
        or (stripped.startswith("[") and stripped.endswith("]"))
        or '"claim' in stripped
        or '"testable' in stripped
        or '"support' in stripped
    )


def _summarize_hypothesis(item: Dict[str, Any], max_chars: int = 280) -> str:
    raw_summary = item.get("summary")
    if isinstance(raw_summary, str) and raw_summary.strip() and not _looks_structured(raw_summary):
        return _as_text(raw_summary, max_chars=max_chars)

    if isinstance(raw_summary, dict):
        for key in ("summary", "verdict", "conclusion", "statement"):
            value = _as_text(raw_summary.get(key), max_chars=max_chars)
            if value:
                return value

    verdict_sentence = _as_text(item.get("verdict"), max_chars=max_chars)
    if verdict_sentence and verdict_sentence.lower() not in _ALLOWED_HYPOTHESIS_VERDICTS:
        return verdict_sentence

    supporting = item.get("supporting_evidence")
    opposing = item.get("opposing_evidence")
    supporting_text = _as_text(supporting, max_chars=160)
    opposing_text = _as_text(opposing, max_chars=160)

    verdict_bucket = _normalize_hypothesis_verdict(
        item.get("status"),
        item.get("verdict"),
    )

    if verdict_bucket == "supported":
        summary = "Retrieved evidence mostly supports this hypothesis."
    elif verdict_bucket == "weakly_supported":
        summary = "Retrieved evidence leans toward support, but remains mixed."
    elif verdict_bucket == "unsupported":
        summary = "Retrieved evidence weighs against this hypothesis."
    else:
        summary = "Retrieved evidence is mixed or insufficient for a clear verdict."

    if supporting_text:
        summary += f" Supporting evidence: {supporting_text}"
    elif opposing_text:
        summary += f" Opposing evidence: {opposing_text}"

    return _as_text(summary, max_chars=max_chars)


def _is_comparison_query(core_question: str, output_format: str) -> bool:
    lower_question = (core_question or "").lower()
    return (
        output_format == "comparison"
        or "compare" in lower_question
        or " versus " in lower_question
        or " vs " in lower_question
        or " more important " in lower_question
        or " or " in lower_question
    )


def _extract_comparison_entities(core_question: str, keywords: List[Any]) -> List[str]:
    entities: List[str] = []

    if isinstance(keywords, list):
        for keyword in keywords:
            cleaned = _as_text(keyword, max_chars=80).strip().lower()
            if not cleaned:
                continue
            if any(token in _GENERIC_QUERY_TERMS for token in _tokenize(cleaned)):
                continue
            entities.append(cleaned)

    if not entities:
        question = (core_question or "").lower()
        question = question.replace(" versus ", " or ").replace(" vs ", " or ")
        parts = re.split(r",|\bor\b", question)
        for part in parts:
            cleaned = part.strip()
            for prefix in (
                "what is more important",
                "which is more important",
                "which matters more",
                "compare",
                "between",
            ):
                cleaned = cleaned.replace(prefix, " ")
            tokens = [
                token for token in _tokenize(cleaned)
                if token not in _STOPWORDS and token not in _GENERIC_QUERY_TERMS
            ]
            if tokens:
                entities.append(" ".join(tokens[:2]))

    seen = set()
    unique_entities = []
    for entity in entities:
        key = entity.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique_entities.append(key)

    return unique_entities[:4]


def _entity_aliases(entity: str) -> List[str]:
    aliases = [entity]
    for token in _tokenize(entity):
        aliases.extend(_QUERY_TERM_ALIASES.get(token, []))
    return list(dict.fromkeys(aliases))


def _format_entity_list(entities: List[str]) -> str:
    labels = [entity.replace("_", " ") for entity in entities if entity]
    if not labels:
        return "the compared factors"
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return f"{', '.join(labels[:-1])}, and {labels[-1]}"


def _comparison_entity_coverage(top_sources: List[Dict[str, Any]], entities: List[str]) -> List[Tuple[str, float, int]]:
    coverage: List[Tuple[str, float, int]] = []
    for entity in entities:
        aliases = _entity_aliases(entity)
        score = 0.0
        mentions = 0
        for source in top_sources:
            haystack = " ".join(
                [
                    _as_text(source.get("title"), max_chars=220),
                    _as_text(source.get("abstract"), max_chars=900),
                ]
            ).lower()
            matched = False
            for alias in aliases:
                if alias.lower() in haystack:
                    matched = True
                    break
            if matched:
                mentions += 1
                score += 1.0 + float(source.get("credibility_score") or 0.0)
        coverage.append((entity, round(score, 3), mentions))
    coverage.sort(key=lambda item: (item[1], item[2]), reverse=True)
    return coverage


def _normalize_report_shape(report: Dict[str, Any], prepared: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {
        "title": _as_text(report.get("title"), max_chars=220),
        "executive_summary": _as_text(report.get("executive_summary"), max_chars=2200),
        "sections": [],
        "key_conclusions": [],
        "hypotheses_verdict": [],
        "contradictions_flagged": [],
        "research_gaps": [],
        "follow_up_questions": [],
        "citations": [],
        "confidence_overall": _clamp_confidence(report.get("confidence_overall", 0.0)),
        "generated_at": _as_text(report.get("generated_at"), max_chars=100),
    }

    # ---- Fallback: use synthesis confidence if report confidence is missing or 0 ----
    if normalized["confidence_overall"] == 0.0:
        synthesis_confidence = _clamp_confidence(
            prepared.get("synthesis", {}).get("confidence_score", 0.0)
        )
        if synthesis_confidence > 0.0:
            normalized["confidence_overall"] = synthesis_confidence

    if not normalized["executive_summary"]:
        consensus = _as_text(prepared.get("synthesis", {}).get("consensus"), max_chars=1000)
        normalized["executive_summary"] = consensus or "Limited evidence supports a provisional summary only."

    sections = report.get("sections")
    if isinstance(sections, list):
        for idx, section in enumerate(sections[:8]):
            if isinstance(section, dict):
                heading = _as_text(section.get("heading"), max_chars=120) or f"Section {idx + 1}"
                content = _as_text(section.get("content"), max_chars=1800)
                supporting = section.get("supporting_source_ids")
                if not isinstance(supporting, list):
                    supporting = []
                supporting_ids = [
                    _as_text(source_id, max_chars=120)
                    for source_id in supporting
                    if _as_text(source_id, max_chars=120)
                ][:8]
                normalized["sections"].append(
                    {
                        "heading": heading,
                        "content": content,
                        "supporting_source_ids": supporting_ids,
                    }
                )

    key_conclusions = report.get("key_conclusions")
    if isinstance(key_conclusions, list):
        for item in key_conclusions[:10]:
            text = _as_text(item, max_chars=360)
            if text:
                normalized["key_conclusions"].append(text)

    hypotheses_verdict = report.get("hypotheses_verdict")
    if isinstance(hypotheses_verdict, list):
        for idx, item in enumerate(hypotheses_verdict[:8]):
            if not isinstance(item, dict):
                continue
            verdict_raw = _as_text(item.get("verdict"), max_chars=32).lower()
            verdict = verdict_raw if verdict_raw in _ALLOWED_HYPOTHESIS_VERDICTS else "contested"
            normalized["hypotheses_verdict"].append(
                {
                    "id": _as_text(item.get("id"), max_chars=40) or f"h{idx + 1}",
                    "statement": _as_text(item.get("statement"), max_chars=320),
                    "verdict": _normalize_hypothesis_verdict(item.get("status"), verdict),
                    "summary": _summarize_hypothesis(item, max_chars=300),
                }
            )

    contradictions_flagged = report.get("contradictions_flagged")
    if isinstance(contradictions_flagged, list):
        for item in contradictions_flagged[:8]:
            if not isinstance(item, dict):
                continue
            severity_raw = _as_text(item.get("severity"), max_chars=16).lower()
            severity = severity_raw if severity_raw in _ALLOWED_CONTRADICTION_SEVERITIES else "low"
            normalized["contradictions_flagged"].append(
                {
                    "summary": _as_text(item.get("summary"), max_chars=360),
                    "severity": severity,
                    "action": _as_text(item.get("action"), max_chars=280),
                }
            )

    research_gaps = report.get("research_gaps")
    if isinstance(research_gaps, list):
        normalized["research_gaps"] = [
            _as_text(item, max_chars=260)
            for item in research_gaps[:10]
            if _as_text(item, max_chars=260)
        ]

    follow_up_questions = report.get("follow_up_questions")
    if isinstance(follow_up_questions, list):
        normalized["follow_up_questions"] = [
            _as_text(item, max_chars=260)
            for item in follow_up_questions[:10]
            if _as_text(item, max_chars=260)
        ]

    citations = report.get("citations")
    if isinstance(citations, list):
        for idx, citation in enumerate(citations[:20]):
            if not isinstance(citation, dict):
                continue
            source_id = _as_text(citation.get("source_id"), max_chars=120)
            normalized["citations"].append(
                {
                    "source_id": source_id or f"source_{idx + 1}",
                    "title": _as_text(citation.get("title"), max_chars=220),
                    "authors": _as_text(citation.get("authors"), max_chars=220),
                    "year": _as_text(citation.get("year"), max_chars=20),
                    "doi": _as_text(citation.get("doi"), max_chars=120),
                    "url": _as_text(citation.get("url"), max_chars=220),
                }
            )

    if not normalized["citations"]:
        for source in prepared.get("top_sources", [])[:12]:
            normalized["citations"].append(
                {
                    "source_id": _as_text(source.get("source_id"), max_chars=120),
                    "title": _as_text(source.get("title"), max_chars=220),
                    "authors": _as_text(source.get("authors"), max_chars=220),
                    "year": _as_text(source.get("year"), max_chars=20),
                    "doi": _as_text(source.get("doi"), max_chars=120),
                    "url": _as_text(source.get("url"), max_chars=220),
                }
            )

    normalized = _ensure_generated_at(normalized)
    return normalized


def _validate_report(report: Dict[str, Any], prepared: Dict[str, Any]) -> List[str]:
    errors: List[str] = []

    if not report.get("executive_summary"):
        errors.append("missing_executive_summary")

    if not report.get("sections"):
        errors.append("missing_sections")

    if not report.get("key_conclusions"):
        errors.append("missing_key_conclusions")

    text_blob = "\n".join(_iter_text_values(report))
    if _contains_placeholders(text_blob):
        errors.append("template_placeholders_detected")

    query_terms = prepared.get("query_terms", [])
    if query_terms:
        lower_blob = text_blob.lower()
        if not any(term in lower_blob for term in query_terms):
            errors.append("report_not_grounded_to_query_terms")

    if _is_comparison_query(
        prepared.get("core_question", ""),
        prepared.get("output_format", ""),
    ):
        executive_lower = _as_text(report.get("executive_summary"), max_chars=1200).lower()
        entities = _extract_comparison_entities(
            prepared.get("core_question", ""),
            prepared.get("keywords", []),
        )
        mentioned = [
            entity for entity in entities
            if any(alias.lower() in executive_lower for alias in _entity_aliases(entity))
        ]
        comparison_cues = (
            "compare",
            "comparison",
            "ranking",
            "more important",
            "head-to-head",
            "single universal ranking",
            "balanced enough",
        )
        if len(mentioned) < min(2, len(entities)) or not any(cue in executive_lower for cue in comparison_cues):
            errors.append("comparison_question_not_answered_directly")

    citations = report.get("citations") if isinstance(report.get("citations"), list) else []
    if not citations:
        errors.append("missing_citations")
    else:
        joined_citations = " ".join(
            _as_text(citation.get("title"), max_chars=180)
            for citation in citations
            if isinstance(citation, dict)
        ).lower()
        if _contains_placeholders(joined_citations):
            errors.append("placeholder_citations_detected")

    return errors


def _build_grounded_report(prepared: Dict[str, Any], quality_errors: List[str]) -> Dict[str, Any]:
    core_question = _as_text(prepared.get("core_question"), max_chars=260) or "Research question"
    synthesis = prepared.get("synthesis", {}) if isinstance(prepared.get("synthesis"), dict) else {}
    top_sources = prepared.get("top_sources", [])
    hypotheses = prepared.get("hypotheses", [])
    contradictions = prepared.get("contradictions", [])

    output_format = _as_text(prepared.get("output_format"), max_chars=60).lower() or "research_brief"
    prefix_map = {
        "literature_review": "Literature Review",
        "research_brief": "Research Brief",
        "comparison": "Comparative Review",
        "summary": "Evidence Summary",
    }
    title = f"{prefix_map.get(output_format, 'Research Brief')}: {core_question}"

    consensus = _as_text(synthesis.get("consensus"), max_chars=800)
    key_findings = synthesis.get("key_findings") if isinstance(synthesis.get("key_findings"), list) else []
    finding_texts = [_as_text(item, max_chars=260) for item in key_findings if _as_text(item, max_chars=260)]

    if consensus:
        executive_summary = f"Evidence suggests: {consensus}"
    elif finding_texts:
        executive_summary = f"Data indicates: {finding_texts[0]}"
    else:
        executive_summary = (
            f"Limited evidence supports a provisional answer for '{core_question}'. "
            "Additional high-quality sources are needed for stronger certainty."
        )

    source_ids = [_as_text(source.get("source_id"), max_chars=120) for source in top_sources[:6]]
    source_ids = [source_id for source_id in source_ids if source_id]

    source_highlights = []
    for source in top_sources[:4]:
        title_part = _as_text(source.get("title"), max_chars=120) or "Untitled source"
        year_part = _as_text(source.get("year"), max_chars=20) or "n/a"
        credibility = source.get("credibility_score", 0.0)
        source_highlights.append(
            f"{title_part} (year: {year_part}, credibility: {credibility})"
        )

    if source_highlights:
        evidence_section = (
            "Data indicates the strongest evidence comes from: "
            + "; ".join(source_highlights)
            + "."
        )
    else:
        evidence_section = "Limited evidence supports a source-grounded conclusion at this time."

    hypotheses_verdict = []
    for idx, hypothesis in enumerate(hypotheses[:5]):
        verdict = _normalize_hypothesis_verdict(
            hypothesis.get("status"),
            hypothesis.get("verdict"),
        )
        hypotheses_verdict.append(
            {
                "id": _as_text(hypothesis.get("id"), max_chars=40) or f"h{idx + 1}",
                "statement": _as_text(hypothesis.get("statement"), max_chars=300),
                "verdict": verdict,
                "summary": _summarize_hypothesis(hypothesis, max_chars=280),
            }
        )

    contradictions_flagged = []
    for item in contradictions[:5]:
        severity_raw = _as_text(item.get("severity"), max_chars=16).lower()
        severity = severity_raw if severity_raw in _ALLOWED_CONTRADICTION_SEVERITIES else "low"
        contradictions_flagged.append(
            {
                "summary": _as_text(item.get("summary"), max_chars=300),
                "severity": severity,
                "action": _as_text(item.get("action"), max_chars=240) or "Review conflicting claims before final decisions.",
            }
        )

    limitations = synthesis.get("limitations") if isinstance(synthesis.get("limitations"), list) else []
    limitation_texts = [_as_text(item, max_chars=220) for item in limitations if _as_text(item, max_chars=220)]
    gap_items = synthesis.get("research_gaps") if isinstance(synthesis.get("research_gaps"), list) else []
    research_gaps = [_as_text(item, max_chars=220) for item in gap_items if _as_text(item, max_chars=220)]
    if not research_gaps and limitation_texts:
        research_gaps = limitation_texts[:4]

    sub_questions = prepared.get("synthesis", {}).get("follow_up_questions")
    follow_up_questions: List[str] = []
    if isinstance(sub_questions, list):
        follow_up_questions = [
            _as_text(item, max_chars=220)
            for item in sub_questions
            if _as_text(item, max_chars=220)
        ][:5]

    if not follow_up_questions:
        follow_up_questions = [
            f"Which high-quality sources most directly quantify '{core_question}'?",
            "What additional recent studies could change the current conclusion?",
        ]

    key_conclusions = finding_texts[:5]

    if _is_comparison_query(core_question, output_format):
        entities = _extract_comparison_entities(core_question, prepared.get("keywords", []))
        coverage = _comparison_entity_coverage(top_sources, entities)
        represented = [item for item in coverage if item[1] > 0]
        entity_list = _format_entity_list(entities)

        if len(represented) >= 2:
            strongest_entity = represented[0][0]
            strongest_score = represented[0][1]
            second_score = represented[1][1]
            if strongest_score <= second_score * 1.2:
                executive_summary = (
                    f"Evidence suggests the retrieved literature does not support a single universal ranking among {entity_list}. "
                    "These factors appear complementary, and this run does not contain enough direct head-to-head evidence to conclude one is categorically more important than the others."
                )
            else:
                executive_summary = (
                    f"Evidence suggests the retrieved literature does not support a single universal ranking among {entity_list}. "
                    f"In this run, {strongest_entity} is the most represented factor in the retrieved evidence, "
                    "but the source mix is too unbalanced to conclude it is categorically more important overall."
                )
            coverage_text = "; ".join(
                f"{entity} appears in {mentions} high-ranking source(s)"
                for entity, _, mentions in coverage
                if mentions > 0
            )
            evidence_section = f"Coverage in the retrieved evidence is: {coverage_text}."
            key_conclusions = [
                f"The retrieved evidence does not justify a single universal ranking among {entity_list}.",
                "The factors are better treated as complementary health behaviors than as interchangeable substitutes.",
            ]
            if represented:
                weakest = [entity for entity, _, mentions in coverage if mentions == 0]
                if weakest:
                    key_conclusions.append(
                        f"This run contains thinner direct evidence for {_format_entity_list(weakest)} than for the other compared factors."
                    )
            research_gaps = [
                f"More direct comparative studies are needed to quantify the relative contribution of {entity_list} within the same population and outcome window."
            ] + research_gaps[:3]
            follow_up_questions = [
                f"Which studies directly compare the relative effects of {entity_list} on the same health outcome?",
                f"How do the effects of {entity_list} differ by age, baseline risk, or chronic disease status?",
            ]
        elif entities:
            executive_summary = (
                f"Limited evidence supports a cautious answer: this run does not contain balanced comparative evidence to rank {entity_list} reliably."
            )
            key_conclusions = [
                f"The retrieved sources do not provide a balanced enough basis to rank {entity_list} confidently.",
            ]

    if not key_conclusions:
        key_conclusions = [executive_summary]

    if quality_errors:
        # Keep this human-readable and transparent inside report body.
        key_conclusions.append(
            "Automated quality guard replaced a low-quality draft to keep this report grounded to retrieved evidence."
        )

    citations = []
    for idx, source in enumerate(top_sources[:15]):
        citations.append(
            {
                "source_id": _as_text(source.get("source_id"), max_chars=120) or f"source_{idx + 1}",
                "title": _as_text(source.get("title"), max_chars=220),
                "authors": _as_text(source.get("authors"), max_chars=220),
                "year": _as_text(source.get("year"), max_chars=20),
                "doi": _as_text(source.get("doi"), max_chars=120),
                "url": _as_text(source.get("url"), max_chars=220),
            }
        )

    # Calculate improved confidence score based on multiple factors
    confidence = _calculate_final_confidence(prepared)

    report = {
        "title": title,
        "executive_summary": executive_summary,
        "sections": [
            {
                "heading": "Evidence Snapshot",
                "content": evidence_section,
                "supporting_source_ids": source_ids,
            },
            {
                "heading": "Hypothesis Assessment",
                "content": (
                    "Evidence suggests the hypothesis set remains mixed; "
                    "see verdict details for which claims are better supported."
                ),
                "supporting_source_ids": source_ids,
            },
            {
                "heading": "Uncertainty and Research Gaps",
                "content": (
                    "; ".join(research_gaps[:4])
                    if research_gaps
                    else "Limited evidence supports a confident final claim; additional targeted studies are needed."
                ),
                "supporting_source_ids": source_ids,
            },
        ],
        "key_conclusions": key_conclusions,
        "hypotheses_verdict": hypotheses_verdict,
        "contradictions_flagged": contradictions_flagged,
        "research_gaps": research_gaps,
        "follow_up_questions": follow_up_questions,
        "citations": citations,
        "confidence_overall": confidence,
        "generated_at": datetime.utcnow().isoformat(),
    }

    return report


def _save_outputs(state: Dict[str, Any], report: Dict[str, Any]) -> Tuple[str, Any, Any]:
    session_id = state.get("session_id", str(uuid.uuid4()))
    living_doc_id = str(uuid.uuid4())

    if state.get("fast_mode"):
        return living_doc_id, None, None

    dynamo = DynamoDBClient()
    s3 = S3Client()

    query_hint = (
        state.get("raw_query")
        or state.get("parsed_query", {}).get("core_question")
        or report.get("title")
    )
    slug = s3.build_report_slug(report, query_hint=query_hint, session_id=session_id)
    s3_json_uri = s3.save_report(session_id, report, slug=slug)
    s3_text_uri = s3.save_report_as_text(session_id, report, slug=slug)
    dynamo.save_session_report(session_id, report, s3_json_uri)

    return living_doc_id, s3_json_uri, s3_text_uri


def _render_text_version(report: Dict[str, Any]) -> str:
    parts = []

    parts.append(f"# {report.get('title', '')}\n")
    parts.append(f"## Executive Summary\n{report.get('executive_summary', '')}\n")

    for section in report.get("sections", []):
        parts.append(f"## {section.get('heading')}\n{section.get('content')}\n")

    parts.append("## Key Conclusions")
    for conclusion in report.get("key_conclusions", []):
        parts.append(f"- {conclusion}")

    return "\n".join(parts)


# ---------------------------
# FUNCTION 1: STANDARD NODE
# ---------------------------

def output_generator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        session_id = state.get("session_id", "unknown")
        prepared = _prepare_inputs(state)

        rendered_prompt = OUTPUT_PROMPT.format(**prepared)
        log_backend_event(
            "output_generator_rendered_prompt",
            session_id=session_id,
            rendered_prompt=rendered_prompt,
        )

        # Use synthesis confidence as fallback, not hardcoded 0.0
        synthesis = state.get("synthesis", {}) or {}
        synthesis_confidence = _clamp_confidence(synthesis.get("confidence_score", 0.0))
        
        llm_response = llm_call_with_retry(
            prompt=rendered_prompt,
            step_name="output_generator",
            session_id=session_id,
            fallback={
                "title": prepared["core_question"] or "Research Report",
                "executive_summary": "Insufficient evidence to generate a full report.",
                "sections": [],
                "key_conclusions": [],
                "hypotheses_verdict": [],
                "contradictions_flagged": [],
                "research_gaps": [],
                "follow_up_questions": [],
                "citations": [],
                "confidence_overall": synthesis_confidence,
                "generated_at": "",
            },
        )

        parsed = json.loads(llm_response) if isinstance(llm_response, str) else llm_response
        if not isinstance(parsed, dict):
            parsed = {}

        report = _normalize_report_shape(parsed, prepared)
        quality_errors = _validate_report(report, prepared)

        if quality_errors:
            log_backend_event(
                "output_generator_quality_guard_triggered",
                session_id=session_id,
                quality_errors=quality_errors,
                candidate_report=report,
            )
            report = _build_grounded_report(prepared, quality_errors)
            report = _normalize_report_shape(report, prepared)

        report = _ensure_generated_at(report)

        living_doc_id, s3_json_uri, _ = _save_outputs(state, report)
        log_backend_event(
            "output_generator_completed",
            session_id=session_id,
            final_report=report,
            living_doc_id=living_doc_id,
            s3_report_uri=s3_json_uri,
        )

        logger.info("Output Generator completed successfully")

        return {
            "final_report": report,
            "living_doc_id": living_doc_id,
            "s3_report_uri": s3_json_uri,
            "current_step": "report_complete",
        }

    except Exception as e:
        logger.exception("Output Generator failed")
        log_backend_event(
            "output_generator_failed",
            session_id=state.get("session_id", "unknown"),
            error=str(e),
        )

        return {
            "error": str(e),
            "current_step": "output_generation_failed",
        }


# ---------------------------
# FUNCTION 2: STREAMING NODE
# ---------------------------

def output_generator_stream(state: Dict[str, Any]):
    full_text = ""

    try:
        session_id = state.get("session_id", "unknown")
        prepared = _prepare_inputs(state)
        rendered_prompt = OUTPUT_PROMPT.format(**prepared)
        log_backend_event(
            "output_generator_stream_rendered_prompt",
            session_id=session_id,
            rendered_prompt=rendered_prompt,
        )

        for chunk in llm_stream(rendered_prompt):
            full_text += chunk
            yield chunk

        parsed = json.loads(full_text)
        if not isinstance(parsed, dict):
            parsed = {}

        report = _normalize_report_shape(parsed, prepared)
        quality_errors = _validate_report(report, prepared)
        if quality_errors:
            log_backend_event(
                "output_generator_stream_quality_guard_triggered",
                session_id=session_id,
                quality_errors=quality_errors,
                candidate_report=report,
            )
            report = _build_grounded_report(prepared, quality_errors)
            report = _normalize_report_shape(report, prepared)

        report = _ensure_generated_at(report)

        living_doc_id, s3_json_uri, _ = _save_outputs(state, report)
        log_backend_event(
            "output_generator_stream_completed",
            session_id=session_id,
            final_report=report,
            living_doc_id=living_doc_id,
            s3_report_uri=s3_json_uri,
        )

        yield {
            "__DONE__": True,
            "final_report": report,
            "living_doc_id": living_doc_id,
            "s3_report_uri": s3_json_uri,
        }

    except Exception as e:
        logger.exception("Streaming Output Generator failed")
        log_backend_event(
            "output_generator_stream_failed",
            session_id=state.get("session_id", "unknown"),
            error=str(e),
        )

        yield {
            "__DONE__": True,
            "error": str(e),
        }
