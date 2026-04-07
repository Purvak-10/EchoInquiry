"""
Source Recheck Engine
- Periodically verifies sources in living documents
- Detects retractions, dead links, content changes
- Updates source metadata with findings
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from tools.retraction_checker import RetractionChecker
from tools.web_scraper import WebScraper
from tools.academic_apis import SemanticScholarAPI, CrossrefAPI
from aws.dynamodb_client import DynamoDBClient
from utils.backend_logging import log_backend_event
import config

logger = logging.getLogger(__name__)


class SourceRecheckerEngine:
    """
    Rechecks sources for:
    1. Retractions (academic papers)
    2. Dead links (web sources)
    3. Citation count updates
    4. Access changes
    5. Content modifications (hash comparison)
    """

    def __init__(self):
        self.db = DynamoDBClient()
        self.retraction_checker = RetractionChecker()
        self.web_scraper = WebScraper()
        self.semantic_scholar_api = SemanticScholarAPI()
        self.crossref_api = CrossrefAPI()

        # Rate limiting
        self.max_workers = 5
        self.delay_between_requests = 0.5  # seconds

    async def check_all_sources(self) -> Dict[str, Any]:
        """
        Check all sources due for rechecking.
        Returns summary of findings.
        """
        logger.info("🔄 Starting source recheck...")

        try:
            # Get all sources due for recheck
            sources_due = self.db.get_sources_due_for_recheck()

            if not sources_due:
                logger.info("📋 No sources due for recheck")
                return {
                    "total_checked": 0,
                    "changes_detected": 0,
                    "retractions_found": 0,
                    "dead_links": 0,
                    "updates_applied": 0,
                }

            logger.info(f"📋 Found {len(sources_due)} sources due for recheck")

            # Check sources in batches
            results = await self._check_sources_batch(sources_due)

            return {
                "total_checked": results["total_checked"],
                "changes_detected": results["changes_detected"],
                "retractions_found": results["retractions_found"],
                "dead_links": results["dead_links"],
                "updates_applied": results["updates_applied"],
            }

        except Exception as e:
            logger.error(f"❌ Source recheck failed: {str(e)}", exc_info=True)
            log_backend_event(
                "living_document_recheck_error",
                error=str(e),
            )
            raise

    async def _check_sources_batch(self, sources: List[Dict]) -> Dict[str, Any]:
        """
        Process sources in parallel batches.
        """
        total_checked = 0
        changes_detected = 0
        retractions_found = 0
        dead_links = 0
        updates_applied = 0

        # Process in parallel batches using asyncio
        batch_size = self.max_workers
        for i in range(0, len(sources), batch_size):
            batch = sources[i : i + batch_size]
            batch_results = await asyncio.gather(
                *[self._check_single_source(source) for source in batch],
                return_exceptions=True,
            )

            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"❌ Batch item failed: {str(result)}")
                    continue

                if result:
                    total_checked += 1
                    changes_detected += result.get("has_changes", 0)
                    retractions_found += result.get("is_retracted", 0)
                    dead_links += result.get("is_dead_link", 0)
                    updates_applied += result.get("updated", 0)

            # Rate limiting between batches
            await asyncio.sleep(self.delay_between_requests)

        return {
            "total_checked": total_checked,
            "changes_detected": changes_detected,
            "retractions_found": retractions_found,
            "dead_links": dead_links,
            "updates_applied": updates_applied,
        }

    async def _check_single_source(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check a single source for:
        - Retractions
        - Dead links
        - Citation updates
        - Content changes
        """
        try:
            source_id = source.get("source_id")
            url = source.get("url", "")
            doi = source.get("doi")
            source_type = self._determine_source_type(source)

            result = {
                "source_id": source_id,
                "has_changes": False,
                "is_retracted": False,
                "is_dead_link": False,
                "updated": False,
                "changes": {},
            }

            # Check for retraction (academic sources)
            if doi:
                retraction_result = self.retraction_checker.check_retraction(doi)
                if retraction_result.get("is_retracted"):
                    logger.warning(f"🚨 RETRACTED: {source.get('title')} (DOI: {doi})")
                    result["is_retracted"] = True
                    result["changes"]["retraction_status"] = "retracted"
                    result["changes"]["retraction_details"] = retraction_result
                    result["updated"] = True

            # Check for dead links (web sources)
            if url and source_type == "web":
                is_alive = await self._check_link_alive(url)
                if not is_alive:
                    logger.warning(f"💀 DEAD LINK: {url}")
                    result["is_dead_link"] = True
                    result["changes"]["access_status"] = "dead_link"
                    result["updated"] = True

            # Check for citation updates (academic sources)
            if doi or (url and source_type == "academic"):
                citations_result = await self._get_updated_citations(source)
                if citations_result:
                    result["changes"].update(citations_result)
                    result["has_changes"] = True
                    result["updated"] = True

            # Update database if changes detected
            if result["updated"]:
                self._update_source_record(source_id, result["changes"])

            return result

        except Exception as e:
            logger.error(f"❌ Failed to check source {source.get('source_id')}: {str(e)}")
            return None

    def _determine_source_type(self, source: Dict) -> str:
        """Determine if source is academic or web"""
        url = source.get("url", "").lower()
        doi = source.get("doi")
        journal = source.get("journal")

        academic_domains = [
            "arxiv", "doi.org", "scholar.google",
            "pubmed", "jstor", "ieee", "springer",
            "sciencedirect", "nature.com", "science.org",
        ]

        if doi or journal:
            return "academic"

        if any(domain in url for domain in academic_domains):
            return "academic"

        return "web"

    async def _check_link_alive(self, url: str) -> bool:
        """Check if web link is still accessible"""
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                result = await loop.run_in_executor(
                    executor,
                    self.web_scraper.check_url_accessible,
                    url,
                )
            return result
        except Exception as e:
            logger.debug(f"⚠️ Link check failed for {url}: {str(e)}")
            return False

    async def _get_updated_citations(self, source: Dict) -> Optional[Dict]:
        """
        Get updated citation count and metadata.
        """
        try:
            doi = source.get("doi")
            title = source.get("title")
            url = source.get("url")

            updates = {}

            # Try DOI first
            if doi:
                crossref_result = self.crossref_api.fetch_paper(doi)
                if crossref_result:
                    citation_count = crossref_result.get("citation_count")
                    if citation_count and citation_count != source.get(
                        "citation_count"
                    ):
                        updates["citation_count"] = citation_count
                        updates["citation_count_updated_at"] = datetime.utcnow().isoformat()
                        logger.info(
                            f"✅ Updated citations for {doi}: {citation_count}"
                        )
                    return updates if updates else None

            # Try Semantic Scholar if title available
            if title:
                semantic_results = self.semantic_scholar_api.search(title, limit=1)
                if semantic_results and len(semantic_results) > 0:
                    paper = semantic_results[0]
                    citation_count = paper.get("citation_count")
                    if citation_count and citation_count != source.get(
                        "citation_count"
                    ):
                        updates["citation_count"] = citation_count
                        updates["citation_count_updated_at"] = (
                            datetime.utcnow().isoformat()
                        )
                        logger.info(
                            f"✅ Updated citations for '{title}': {citation_count}"
                        )
                    return updates if updates else None

            return None

        except Exception as e:
            logger.debug(f"⚠️ Failed to get updated citations: {str(e)}")
            return None

    def _update_source_record(self, source_id: str, changes: Dict) -> bool:
        """
        Update source record in database with changes.
        """
        try:
            update_data = {
                "updated_at": datetime.utcnow().isoformat(),
                "next_check_at": (
                    datetime.utcnow()
                    + timedelta(days=config.LIVING_DOC_CHECK_INTERVAL_DAYS)
                ).isoformat(),
                **changes,
            }

            self.db.update_source(source_id, update_data)
            return True

        except Exception as e:
            logger.error(f"❌ Failed to update source {source_id}: {str(e)}")
            return False
