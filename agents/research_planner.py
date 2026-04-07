import json
import logging
from typing import Dict, List, Any

from prompts.planner_prompt import PLANNER_PROMPT
from utils.backend_logging import log_backend_event
from utils.llm_helpers import llm_call_with_retry

logger = logging.getLogger(__name__)


# -----------------------------
# Priority Task Queue
# -----------------------------
class PriorityTaskQueue:
    def __init__(self):
        self.tasks: List[Dict] = []
        self.completed = set()

    def push(self, task: Dict):
        self.tasks.append(task)
        self.tasks.sort(key=lambda x: x.get("priority", 999))

    def pop(self) -> Dict:
        if self.is_empty():
            return None
        return self.tasks.pop(0)

    def mark_done(self, task_id: str):
        self.completed.add(task_id)

    def is_empty(self) -> bool:
        return len(self.tasks) == 0

    def __len__(self):
        return len(self.tasks)


# -----------------------------
# Helper: Fallback Plan
# -----------------------------
def _fallback_plan(parsed_query: Dict) -> Dict:
    keywords = parsed_query.get("keywords", []) or ["general research"]

    return {
        "task_graph": [
            {
                "task_id": "t1",
                "task_type": "search_web",
                "description": "General search for query",
                "depends_on": [],
                "priority": 1,
                "keywords": keywords,
                "target_sources": ["web"]
            }
        ],
        "estimated_depth": "shallow",
        "recommended_hypothesis_count": 1,
        "search_strategy": "breadth_first"
    }


# -----------------------------
# Helper: Validation & Sorting
# -----------------------------
def _process_plan(plan: Dict) -> Dict:
    task_graph = plan.get("task_graph", [])

    # Ensure valid structure
    if not isinstance(task_graph, list) or not task_graph:
        raise ValueError("Invalid task_graph")

    # Normalize + sort
    for task in task_graph:
        task.setdefault("depends_on", [])
        task.setdefault("priority", 999)
        task.setdefault("keywords", [])
        task.setdefault("target_sources", ["web"])

    task_graph.sort(key=lambda x: x["priority"])
    plan["task_graph"] = task_graph

    return plan


# -----------------------------
# MAIN NODE
# -----------------------------
def research_planner_node(state: Dict) -> Dict:
    """
    LangGraph Node: Research Planner
    """

    parsed_query = state.get("parsed_query", {})
    session_id = state.get("session_id")

    try:
        # 1. Serialize input
        parsed_query_json = json.dumps(parsed_query)

        # 2. Build prompt
        prompt = PLANNER_PROMPT.format(parsed_query_json=parsed_query_json)
        log_backend_event(
            "research_planner_rendered_prompt",
            session_id=session_id,
            parsed_query=parsed_query,
            rendered_prompt=prompt,
        )

        # 3. Call LLM
        response = llm_call_with_retry(
            prompt=prompt,
            session_id=session_id,
            step_name="research_planner",
            fallback=json.dumps(_fallback_plan(parsed_query)),
            max_tokens=1500
        )

        # 4. Parse JSON
        if isinstance(response, str):
            plan = json.loads(response)
        else:
            plan = response

        # 5. Validate + sort
        plan = _process_plan(plan)

        # 6. Logging
        num_tasks = len(plan["task_graph"])
        depth = plan.get("estimated_depth", "unknown")

        logger.info(f"[Session {session_id}] Plan: {num_tasks} tasks, depth: {depth}")
        log_backend_event(
            "research_planner_completed",
            session_id=session_id,
            parsed_query=parsed_query,
            research_plan=plan,
        )

    except Exception as e:
        logger.error(f"[Session {session_id}] Planner failed: {str(e)}")
        log_backend_event(
            "research_planner_failed",
            session_id=session_id,
            parsed_query=parsed_query,
            error=str(e),
        )

        # Fallback plan
        plan = _fallback_plan(parsed_query)

        logger.info(f"[Session {session_id}] Fallback plan used")

    return {
        "research_plan": plan,
        "current_step": "planning_complete"
    }
