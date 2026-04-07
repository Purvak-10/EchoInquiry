import json
import logging
from datetime import datetime

from prompts.query_parser_prompt import QUERY_PARSER_PROMPT
from utils.backend_logging import log_backend_event
from utils.llm_helpers import llm_call_with_retry

logger = logging.getLogger(__name__)


def query_parser_node(state):
    """
    LangGraph node — Query Parser (FIRST node)

    Input:
        state["raw_query"]
        state["session_id"]

    Output:
        {
            "parsed_query": dict,
            "current_step": "query_parsed"
        }
    """

    raw_query = state["raw_query"]
    session_id = state["session_id"]

    rendered_prompt = QUERY_PARSER_PROMPT.replace("{raw_query}", raw_query)
    log_backend_event(
        "query_parser_rendered_prompt",
        session_id=session_id,
        raw_query=raw_query,
        rendered_prompt=rendered_prompt,
    )

    fallback = {
        "intent": "explore",
        "domain": "general",
        "scope": "medium",
        "core_question": raw_query,
        "sub_questions": [],
        "ambiguities": [],
        "keywords": raw_query.split()[:5],
        "time_range": "any",
        "output_format": "summary",
    }

    result = llm_call_with_retry(
        prompt=rendered_prompt,
        session_id=session_id,
        step_name="query_parser",
        fallback=fallback,
    )

    if isinstance(result, str):
        try:
            result = json.loads(result)
        except Exception:
            logger.exception("Query parser JSON parse failed")
            result = fallback

    logger.info(
        "QUERY_PARSED",
        extra={
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "parsed_query": result,
        },
    )
    log_backend_event(
        "query_parser_completed",
        session_id=session_id,
        raw_query=raw_query,
        parsed_response=result,
    )

    return {
        "parsed_query": result,
        "current_step": "query_parsed",
    }
