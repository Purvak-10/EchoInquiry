"""
Multi-Agent Research System

Specialized agents for research pipeline (LangGraph based):
- query_parser_node: Parse and structure research queries
- research_planner_node: Design research methodology
- hypothesis_generation_node: Generate candidate hypotheses
- hypothesis_evaluation_node: Test hypotheses against evidence
- retriever_node: Find and retrieve sources
- credibility_scorer_node: Evaluate source quality
- contradiction_detector_node: Detect conflicting claims
- synthesis_engine_node: Synthesize findings
- output_generator_node: Generate final report
"""

from agents.followup_agent import FollowupAgent

__all__ = [
    "FollowupAgent",
]

__version__ = "2.0.0"
__description__ = "Multi-agent research system with specialized agents"
