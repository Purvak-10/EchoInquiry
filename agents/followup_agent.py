import json
from typing import List, Dict, Generator

from aws.dynamodb_client import DynamoDBClient
from utils.llm_helpers import get_llm_client
from prompts.followup_prompt import FOLLOWUP_SYSTEM_PROMPT


class FollowupAgent:
    def __init__(self, session_id: str, report_data: dict = None):
        self.session_id = session_id
        self.db = DynamoDBClient()
        self.llm = get_llm_client()
        self.context = self._build_context(report_data)

    def _fetch_session_data(self):
        try:
            session = self.db.get_session(self.session_id)
        except Exception:
            session = None

        if not session:
            return None

        report_raw = session.get("final_report_json", {})
        if isinstance(report_raw, str):
            try:
                report = json.loads(report_raw)
            except Exception:
                report = {}
        else:
            report = report_raw or {}

        raw_query = session.get("raw_query", "")
        try:
            sources = self.db.get_sources_for_session(self.session_id)
            hypotheses = self.db.get_hypotheses_for_session(self.session_id)
            contradictions = self.db.get_contradictions_for_session(self.session_id)
        except Exception:
            sources = report.get("citations", [])
            hypotheses = report.get("hypotheses_verdict", [])
            contradictions = report.get("contradictions_flagged", [])

        return report, raw_query, sources, hypotheses, contradictions

    def _build_context(self, report_data: dict = None) -> str:
        report = {}
        raw_query = "Unknown Query"
        sources = []
        hypotheses = []
        contradictions = []

        if report_data:
            report = report_data
            raw_query = (
                report.get("raw_query")
                or report.get("query")
                or report.get("core_question")
                or report.get("title")
                or "Unknown Query"
            )
            sources = report.get("citations", [])
            hypotheses = report.get("hypotheses_verdict", [])
            contradictions = report.get("contradictions_flagged", [])
        else:
            session_data = self._fetch_session_data()
            if not session_data:
                return f"No session data found for ID: {self.session_id}"

            report, raw_query, sources, hypotheses, contradictions = session_data

        # Sort and take top 8 sources by credibility_score
        sources = sorted(
            sources,
            key=lambda x: x.get("credibility_score", 0),
            reverse=True
        )[:8]

        # --- Extract report fields safely ---
        executive_summary = report.get("executive_summary", "")
        key_conclusions = report.get("key_conclusions", [])
        research_gaps = report.get("research_gaps", [])

        # --- Format sections ---
        conclusions_str = "\n".join(
            [f"{i+1}. {c}" for i, c in enumerate(key_conclusions)]
        )

        hypotheses_str = "\n".join([
            f"{h.get('hypothesis_id')} | {h.get('status')} | {h.get('verdict')}"
            for h in hypotheses
        ])

        contradictions_str = "\n".join([
            f"{c.get('severity')} | {c.get('explanation')}"
            for c in contradictions
        ])

        sources_str = "\n".join([
            f"{s.get('title')} | {s.get('credibility_score')} | {s.get('doi', 'N/A')}"
            for s in sources
        ])

        gaps_str = "\n".join([f"- {gap}" for gap in research_gaps])

        # --- Final structured context ---
        context = f"""
RESEARCH QUESTION:
{raw_query}

EXECUTIVE SUMMARY:
{executive_summary}

KEY CONCLUSIONS:
{conclusions_str}

HYPOTHESES:
{hypotheses_str}

CONTRADICTIONS:
{contradictions_str}

TOP 8 SOURCES:
{sources_str}

RESEARCH GAPS:
{gaps_str}
"""
        return context.strip()

    def _format_chat_history(self, chat_history: List[Dict]) -> str:
        formatted = []
        for turn in chat_history:
            role = turn.get("role")
            content = turn.get("content", "")

            if role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")

        return "\n".join(formatted)

    def _build_prompt(self, question: str, chat_history: List[Dict]) -> str:
        history_str = self._format_chat_history(chat_history)

        prompt = f"""
[System Instructions]
{FOLLOWUP_SYSTEM_PROMPT}

[Research Context]
{self.context}

[Conversation So Far]
{history_str}

User: {question}
Assistant:
"""
        return prompt.strip()

    def answer_streaming(
        self,
        question: str,
        chat_history: List[Dict]
    ) -> Generator[str, None, None]:
        prompt = self._build_prompt(question, chat_history)

        for chunk in self.llm.invoke_streaming(
            prompt,
            max_tokens=1024
        ):
            yield chunk

    def answer(
        self,
        question: str,
        chat_history: List[Dict]
    ) -> str:
        prompt = self._build_prompt(question, chat_history)

        return self.llm.invoke(
            prompt,
            max_tokens=1024
        )
