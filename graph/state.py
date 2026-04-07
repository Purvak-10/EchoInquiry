from typing import TypedDict, List, Dict, Optional, Any


class ResearchState(TypedDict):
    # --- session ---
    raw_query: str
    session_id: str

    # --- parsed query ---
    parsed_query: Dict[str, Any]

    # --- research planning ---
    research_plan: Dict[str, Any]

    # --- hypothesis engine ---
    hypotheses: List[Dict[str, Any]]

    # --- retrieval layer ---
    retrieved_sources: List[Dict[str, Any]]

    # --- contradiction detection ---
    contradictions: List[Dict[str, Any]]

    # --- synthesis layer ---
    synthesis: Dict[str, Any]

    # --- final reporting ---
    final_report: Dict[str, Any]

    # --- persistence / infra ---
    living_doc_id: Optional[str]
    source_registry_entries: List[Dict[str, Any]]
    s3_report_uri: Optional[str]

    # --- runtime ---
    error_log: List[str]
    current_step: str
    iteration_count: int
    fast_mode: bool


def _default_parsed_query() -> Dict[str, Any]:
    return {
        "intent": "",
        "domain": "",
        "scope": "",
        "core_question": "",
        "sub_questions": [],
        "ambiguities": [],
        "keywords": [],
        "time_range": "",
        "output_format": "",
    }


def _default_research_plan() -> Dict[str, Any]:
    return {
        "task_graph": [],
        "estimated_depth": 0,
        "recommended_hypothesis_count": 0,
        "search_strategy": "",
    }


def _default_synthesis() -> Dict[str, Any]:
    return {
        "consensus": "",
        "confidence_score": 0.0,
        "evidence_weight_map": {},
        "key_findings": [],
        "outliers": [],
        "limitations": [],
        "research_gaps": [],
    }


def _default_final_report() -> Dict[str, Any]:
    return {
        "title": "",
        "executive_summary": "",
        "sections": [],
        "key_conclusions": [],
        "hypotheses_verdict": [],
        "contradictions_flagged": [],
        "research_gaps": [],
        "follow_up_questions": [],
        "citations": [],
        "confidence_overall": 0.0,
        "generated_at": "",
    }


def make_initial_state(raw_query: str, session_id: str) -> ResearchState:
    """
    Create a clean initial research state.
    Safe for LangGraph checkpointing + AWS persistence.
    """

    return ResearchState(
        raw_query=raw_query,
        session_id=session_id,
        parsed_query=_default_parsed_query(),
        research_plan=_default_research_plan(),
        hypotheses=[],
        retrieved_sources=[],
        contradictions=[],
        synthesis=_default_synthesis(),
        final_report=_default_final_report(),
        living_doc_id=None,
        source_registry_entries=[],
        s3_report_uri=None,
        error_log=[],
        current_step="not_started",
        iteration_count=0,
        fast_mode=False,
    )
