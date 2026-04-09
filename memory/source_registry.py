import uuid
import hashlib
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any

import config
from aws.dynamodb_client import DynamoDBClient

logger = logging.getLogger(__name__)


class SourceRegistry:
    """
    Stores and manages sources used in generated reports.
    Enables 'living document' behavior via periodic re-checking.
    """

    def __init__(self):
        self.db = DynamoDBClient()

    # -------------------------
    # Core Helpers
    # -------------------------

    def _generate_source_id(self) -> str:
        return str(uuid.uuid4())

    def _hash_content(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _now(self) -> datetime:
        return datetime.utcnow()

    # -------------------------
    # Register Single Source
    # -------------------------

    def register_source(
        self,
        session_id: str,
        claim_text: str,
        claim_weight: float,
        source: Dict[str, Any]
    ) -> str:
        """
        Registers a single source tied to a claim.
        """

        source_id = source.get("source_id") or self._generate_source_id()
        now = self._now()

        entry = {
            "source_id": source_id,
            "session_id": session_id,

            # Claim info
            "claim_text": claim_text,
            "claim_weight": claim_weight,

            # Source metadata
            "title": source.get("title"),
            "url": source.get("url"),
            "authors": source.get("authors"),
            "published_at": source.get("published_at"),
            "journal": source.get("journal"),
            "doi": source.get("doi"),

            # Content tracking
            "content_hash": self._hash_content(
                source.get("content")
                or source.get("full_text_snippet")
                or source.get("abstract")
                or source.get("title", "")
            ),

            # Credibility
            "credibility_score": source.get("credibility_score"),

            # Retraction tracking
            "retraction_status": "unknown",

            # Living doc system
            "check_frequency_days": config.LIVING_DOC_CHECK_INTERVAL_DAYS,
            "next_check_at": (
                now + timedelta(days=config.LIVING_DOC_CHECK_INTERVAL_DAYS)
            ).isoformat(),

            # Audit
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        }

        self.db.save_source(entry)

        return entry["source_id"]

    # -------------------------
    # Bulk Registration
    # -------------------------

    def register_all_from_report(
        self,
        session_id: str,
        final_report: Dict[str, Any],
        sources: List[Dict[str, Any]]
    ):
        """
        Registers all sources referenced in the final report.
        """

        source_map = {
            s.get("source_id"): s for s in sources if s.get("source_id")
        }
        citations = final_report.get("citations", [])

        count = 0

        key_conclusions = set(final_report.get("key_conclusions", []))
        sections = final_report.get("sections", [])

        for citation in citations:
            source_id = citation.get("source_id")
            claim_text = (
                citation.get("claim_text")
                or citation.get("title")
                or final_report.get("title", "")
            )

            if source_id not in source_map:
                continue

            # -------------------------
            # Determine claim weight
            # -------------------------
            if claim_text in key_conclusions:
                claim_weight = 0.9
            elif any(
                claim_text in section.get("content", "")
                for section in sections
            ):
                claim_weight = 0.7
            else:
                claim_weight = 0.5

            self.register_source(
                session_id=session_id,
                claim_text=claim_text,
                claim_weight=claim_weight,
                source=source_map[source_id]
            )

            count += 1

        logger.info(f"Registered {count} sources")

    # -------------------------
    # Recheck Pipeline
    # -------------------------

    def get_sources_due_for_recheck(self) -> List[Dict[str, Any]]:
        """
        Returns sources where next_check_at <= now
        """
        return self.db.get_sources_due_for_recheck()

    def update_after_recheck(
        self,
        source_id: str,
        session_id: str,
        new_hash: str,
        retraction_status: str,
        next_check_at: datetime
    ):
        """
        Updates source after revalidation check
        """

        self.db.update_source_after_recheck(
            source_id=source_id,
            session_id=session_id,
            new_hash=new_hash,
            retraction_status=retraction_status,
            next_check_at=next_check_at.isoformat(),
        )

    # -------------------------
    # Retrieval
    # -------------------------

    def get_all_for_session(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Returns all sources linked to a session
        """
        return self.db.get_sources_for_session(session_id)

    def get_alerts(self) -> List[Dict[str, Any]]:
        """
        Returns all open living document alerts.
        """
        return self.db.get_open_alerts()

    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """
        Returns all registered research sessions.
        """
        return self.db.list_sessions()
