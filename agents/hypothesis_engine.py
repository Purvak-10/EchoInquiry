import json
import logging

from prompts.hypothesis_prompt import (
    HYPOTHESIS_EVALUATION_PROMPT,
    HYPOTHESIS_GENERATION_PROMPT,
)
from utils.backend_logging import log_backend_event
from utils.llm_helpers import llm_call_with_retry

logger = logging.getLogger(__name__)


# =========================================================
# FUNCTION 1: Hypothesis Generation Node (PRE-RETRIEVAL)
# =========================================================

def hypothesis_generation_node(state: dict) -> dict:
    """
    Runs BEFORE retrieval.
    Generates falsifiable hypotheses from parsed query.
    """

    parsed_query = state.get("parsed_query", {})
    session_id = state.get("session_id", "unknown")

    core_question = (
        parsed_query.get("core_question")
        or parsed_query.get("query")
        or state.get("raw_query", "")
    )
    domain = parsed_query.get("domain", "general")
    sub_questions = parsed_query.get("sub_questions", [])
    hypothesis_count = parsed_query.get("hypothesis_count", 3)

    try:
        prompt = HYPOTHESIS_GENERATION_PROMPT.format(
            core_question=core_question,
            domain=domain,
            sub_questions=json.dumps(sub_questions),
            hypothesis_count=hypothesis_count
        )
        log_backend_event(
            "hypothesis_generation_rendered_prompt",
            session_id=session_id,
            parsed_query=parsed_query,
            rendered_prompt=prompt,
        )

        response = llm_call_with_retry(
            prompt=prompt,
            step_name="hypothesis_generation",
            session_id=session_id,
            fallback={
                "hypotheses": [{
                    "id": "h1",
                    "statement": f"No significant relationship exists for: {core_question}",
                    "mechanism": "Null hypothesis assumes no causal mechanism.",
                    "predicted_evidence": "No statistically significant pattern observed.",
                    "falsification_criteria": "Consistent statistically significant evidence observed.",
                    "confidence_prior": 0.5,
                }]
            },
        )

        parsed = json.loads(response) if isinstance(response, str) else response
        hypotheses = parsed.get("hypotheses", [])

        # Enrich hypotheses with evaluation placeholders
        for h in hypotheses:
            h["status"] = "unverified"
            h["confidence_posterior"] = None
            h["supporting_evidence"] = []
            h["opposing_evidence"] = []
            h["verdict"] = ""

        log_backend_event(
            "hypothesis_generation_completed",
            session_id=session_id,
            parsed_query=parsed_query,
            hypotheses=hypotheses,
        )

        return {
            "hypotheses": hypotheses,
            "current_step": "hypotheses_generated"
        }

    except Exception as e:
        logger.error(f"[HypothesisGeneration] Failed: {e}")
        log_backend_event(
            "hypothesis_generation_failed",
            session_id=session_id,
            parsed_query=parsed_query,
            error=str(e),
        )

        # Fallback hypothesis
        fallback = [{
            "id": "h1",
            "statement": f"No significant relationship exists for: {core_question}",
            "mechanism": "Null hypothesis assumes no causal mechanism",
            "predicted_evidence": "No statistically significant pattern observed",
            "falsification_criteria": "Consistent statistically significant evidence observed",
            "confidence_prior": 0.5,
            "status": "unverified",
            "confidence_posterior": None,
            "supporting_evidence": [],
            "opposing_evidence": [],
            "verdict": ""
        }]

        return {
            "hypotheses": fallback,
            "current_step": "hypotheses_generated"
        }


# =========================================================
# FUNCTION 2: Hypothesis Evaluation Node (POST-RETRIEVAL)
# =========================================================

def hypothesis_evaluation_node(state: dict) -> dict:
    """
    Runs AFTER credibility scoring.
    Evaluates hypotheses against retrieved evidence.
    """

    hypotheses = state.get("hypotheses", [])
    sources = state.get("retrieved_sources", [])
    session_id = state.get("session_id", "unknown")

    try:
        # Sort by credibility_score (descending)
        top_sources = sorted(
            sources,
            key=lambda x: x.get("credibility_score", 0),
            reverse=True
        )[:10]

        # Build evidence string
        evidence_blocks = []
        for s in top_sources:
            block = {
                "title": s.get("title", ""),
                "abstract": s.get("abstract", ""),
                "snippet": s.get("full_text_snippet", ""),
                "credibility": s.get("credibility_score", 0)
            }
            evidence_blocks.append(block)

        prompt = HYPOTHESIS_EVALUATION_PROMPT.format(
            hypothesis_json=json.dumps(hypotheses),
            source_evidence_json=json.dumps(evidence_blocks)
        )
        log_backend_event(
            "hypothesis_evaluation_rendered_prompt",
            session_id=session_id,
            hypotheses=hypotheses,
            top_sources=evidence_blocks,
            rendered_prompt=prompt,
        )

        response = llm_call_with_retry(
            prompt=prompt,
            step_name="hypothesis_evaluation",
            session_id=session_id,
            fallback={"evaluations": []},
        )

        parsed = json.loads(response) if isinstance(response, str) else response
        evaluations = parsed.get("evaluations", [])

        # Map evaluations by id
        eval_map = {e["id"]: e for e in evaluations}

        # Merge results into hypotheses
        for h in hypotheses:
            eid = h.get("id")
            if eid in eval_map:
                e = eval_map[eid]
                h["status"] = e.get("status", "insufficient_evidence")
                h["confidence_posterior"] = e.get("confidence_posterior", 0.0)
                h["supporting_evidence"] = e.get("supporting_evidence", [])
                h["opposing_evidence"] = e.get("opposing_evidence", [])
                h["verdict"] = e.get("verdict", "")

        log_backend_event(
            "hypothesis_evaluation_completed",
            session_id=session_id,
            hypotheses=hypotheses,
            evaluations=evaluations,
        )

        return {
            "hypotheses": hypotheses,
            "current_step": "hypotheses_evaluated"
        }

    except Exception as e:
        logger.error(f"[HypothesisEvaluation] Failed: {e}")
        log_backend_event(
            "hypothesis_evaluation_failed",
            session_id=session_id,
            hypotheses=hypotheses,
            source_count=len(sources),
            error=str(e),
        )

        # Fallback: mark all as insufficient
        for h in hypotheses:
            h["status"] = "insufficient_evidence"
            h["confidence_posterior"] = 0.0
            h["verdict"] = "Insufficient evidence to evaluate"

        return {
            "hypotheses": hypotheses,
            "current_step": "hypotheses_evaluated"
        }
