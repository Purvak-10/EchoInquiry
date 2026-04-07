import uuid
from langgraph.graph import StateGraph, END

from graph.state import ResearchState, make_initial_state
from agents.query_parser import query_parser_node
from agents.research_planner import research_planner_node
from agents.hypothesis_engine import (
    hypothesis_generation_node,
    hypothesis_evaluation_node,
)
from agents.retriever import retriever_node
from agents.credibility_scorer import credibility_scorer_node
from agents.contradiction_detector import contradiction_detector_node
from agents.synthesis_engine import synthesis_engine_node
from agents.output_generator import output_generator_node

from memory.source_registry import SourceRegistry
from memory.vector_store import VectorStore
from memory.knowledge_graph import KnowledgeGraph
from aws.dynamodb_client import DynamoDBClient
import config


def build_graph():
    graph = StateGraph(ResearchState)
    graph.add_node("query_parser", query_parser_node)
    graph.add_node("research_planner", research_planner_node)
    graph.add_node("hypothesis_generation", hypothesis_generation_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("credibility_scorer", credibility_scorer_node)
    graph.add_node("hypothesis_evaluation", hypothesis_evaluation_node)
    graph.add_node("contradiction_detector", contradiction_detector_node)
    graph.add_node("synthesis_engine", synthesis_engine_node)
    graph.add_node("output_generator", output_generator_node)

    graph.set_entry_point("query_parser")
    graph.add_edge("query_parser", "research_planner")
    graph.add_edge("research_planner", "hypothesis_generation")
    graph.add_edge("hypothesis_generation", "retriever")
    graph.add_edge("retriever", "credibility_scorer")
    graph.add_edge("credibility_scorer", "hypothesis_evaluation")
    graph.add_edge("hypothesis_evaluation", "contradiction_detector")
    graph.add_edge("contradiction_detector", "synthesis_engine")
    graph.add_edge("synthesis_engine", "output_generator")
    graph.add_edge("output_generator", END)
    return graph.compile()


app = build_graph()


def _allocate_session_id(fast_mode: bool) -> str:
    if fast_mode:
        return str(uuid.uuid4())
    try:
        return DynamoDBClient().allocate_session_id(prefix="session")
    except Exception:
        return str(uuid.uuid4())


def _run_post_pipeline(state: ResearchState):
    session_id = state["session_id"]
    report = state.get("final_report", {})
    sources = state.get("retrieved_sources", [])
    hypotheses = state.get("hypotheses", [])
    contradictions = state.get("contradictions", [])

    SourceRegistry().register_all_from_report(
        session_id=session_id,
        final_report=report,
        sources=sources,
    )

    vector_sources = []
    for source in sources:
        text = (
            source.get("full_text_snippet")
            or source.get("abstract")
            or source.get("title")
            or ""
        )
        source_id = (
            source.get("source_id")
            or source.get("id")
            or source.get("doi")
            or source.get("title")
        )
        if not source_id or not text:
            continue
        vector_sources.append({
            "id": str(source_id),
            "text": text,
            "metadata": {
                "title": source.get("title", ""),
                "doi": source.get("doi", ""),
                "year": source.get("year"),
                "credibility_score": source.get("credibility_score", 0.0),
            },
        })

    if vector_sources:
        VectorStore().add_sources_batch(sources=vector_sources, session_id=session_id)

    kg = KnowledgeGraph()
    kg.add_session_node(session_id=session_id, query=state.get("raw_query", ""))
    for source in sources:
        kg.add_source_node(source, session_id=session_id)
    kg.save()

    db = DynamoDBClient()
    db.save_all_hypotheses(hypotheses, session_id)
    for c in contradictions:
        db.save_contradiction(c, session_id)


def run_research(query: str, fast_mode: bool = False) -> dict:
    session_id = _allocate_session_id(fast_mode)
    initial_state = make_initial_state(query, session_id)
    initial_state["fast_mode"] = fast_mode
    original_langfuse_enabled = config.LANGFUSE_ENABLED

    if not fast_mode:
        try:
            DynamoDBClient().create_session(session_id, query)
        except Exception:
            pass

    if fast_mode:
        config.LANGFUSE_ENABLED = False
    try:
        result = app.invoke(initial_state)
    finally:
        config.LANGFUSE_ENABLED = original_langfuse_enabled

    if not fast_mode:
        try:
            _run_post_pipeline(result)
        except Exception:
            pass

    return result.get("final_report", result)


def stream_research(query: str, fast_mode: bool = False):
    session_id = _allocate_session_id(fast_mode)
    initial_state = make_initial_state(query, session_id)
    initial_state["fast_mode"] = fast_mode
    original_langfuse_enabled = config.LANGFUSE_ENABLED

    if not fast_mode:
        DynamoDBClient().create_session(session_id, query)

    final_state = None
    current_state = dict(initial_state)

    if fast_mode:
        config.LANGFUSE_ENABLED = False
    try:
        for event in app.stream(initial_state, stream_mode="updates"):
            node_name = list(event.keys())[0]
            node_payload = event[node_name]
            current_state.update(node_payload)
            yield {
                "node": node_name,
                "step": node_payload.get("current_step", node_name),
                "session_id": session_id,
                "payload": node_payload,
            }
            final_state = current_state
    finally:
        config.LANGFUSE_ENABLED = original_langfuse_enabled

    if final_state:
        final_state["session_id"] = session_id
        if not fast_mode:
            try:
                _run_post_pipeline(final_state)
            except Exception as e:
                yield {
                    "node": "post_pipeline_warning",
                    "step": "post_pipeline_warning",
                    "session_id": session_id,
                    "payload": {"warning": str(e)},
                }

        yield {
            "node": "__done__",
            "session_id": session_id,
            "final_report": final_state.get("final_report"),
            "s3_uri": final_state.get("s3_report_uri"),
        }
