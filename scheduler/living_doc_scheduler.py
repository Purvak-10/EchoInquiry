"""
Living Document Scheduler
- Automatically checks sources for updates, retractions, dead links
- Runs every 30 days
- Stores results in DynamoDB
- Alerts on changes
- Rechecks citation counts, dead links, retractions
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from living_document.recheck_engine import SourceRecheckerEngine
from utils.backend_logging import log_backend_event

logger = logging.getLogger(__name__)


class LivingDocumentScheduler:
    """Manages background scheduler for living document updates"""

    def __init__(self):
        self.scheduler: Optional[BackgroundScheduler] = None
        self.recheck_engine = SourceRecheckerEngine()
        self.is_running = False
        self.last_check_at: Optional[datetime] = None
        self.next_check_at: Optional[datetime] = None

    def start(self):
        """Start the background scheduler"""
        if self.is_running:
            logger.warning("⚠️ Scheduler already running, skipping start")
            return

        try:
            self.scheduler = BackgroundScheduler()

            # Schedule living document check every 30 days
            self.scheduler.add_job(
                func=self._check_living_documents,
                trigger=IntervalTrigger(days=30),
                id="living_doc_check",
                name="Living Document Source Recheck (30-day cycle)",
                replace_existing=True,
                max_instances=1,  # Prevent concurrent execution
            )

            self.scheduler.start()
            self.is_running = True
            self.next_check_at = datetime.utcnow() + timedelta(days=30)

            logger.info("✅ Living Document Scheduler STARTED")
            logger.info("   - Trigger: Every 30 days")
            logger.info("   - Job: Recheck all sources for retractions, dead links, citations")
            logger.info("   - Storage: DynamoDB")
            logger.info("   - Alerts: Enabled for retractions and dead links")
            logger.info(f"   - Next check: {self.next_check_at.isoformat()}")

            log_backend_event(
                "scheduler_started",
                type="living_document_30_day_cycle",
                next_check=self.next_check_at.isoformat(),
            )

        except Exception as e:
            logger.error(f"❌ Failed to start scheduler: {str(e)}")
            self.is_running = False
            log_backend_event(
                "scheduler_start_failed",
                error=str(e),
            )
            raise

    def stop(self):
        """Stop the background scheduler"""
        if not self.is_running or not self.scheduler:
            logger.warning("⚠️ Scheduler not running, skipping stop")
            return

        try:
            self.scheduler.shutdown(wait=True)
            self.is_running = False
            self.next_check_at = None
            logger.info("🛑 Living Document Scheduler STOPPED")
            log_backend_event("scheduler_stopped", type="living_document")
        except Exception as e:
            logger.error(f"❌ Failed to stop scheduler: {str(e)}")
            log_backend_event(
                "scheduler_stop_failed",
                error=str(e),
            )

    def _check_living_documents(self):
        """Periodic job: Recheck all living document sources"""
        try:
            logger.info("=" * 80)
            logger.info("🔄 LIVING DOCUMENT RECHECK CYCLE STARTED")
            logger.info(f"   Timestamp: {datetime.utcnow().isoformat()}")
            logger.info("=" * 80)

            self.last_check_at = datetime.utcnow()

            # Run async function in sync context
            asyncio.run(self._run_recheck())

            self.next_check_at = datetime.utcnow() + timedelta(days=30)

            logger.info("=" * 80)
            logger.info("✅ LIVING DOCUMENT RECHECK CYCLE COMPLETED")
            logger.info(f"   Next check: {self.next_check_at.isoformat()}")
            logger.info("=" * 80)

            log_backend_event(
                "living_document_recheck_completed",
                next_check=self.next_check_at.isoformat(),
            )

        except Exception as e:
            logger.error(
                f"❌ Living document recheck failed: {str(e)}", exc_info=True
            )
            log_backend_event(
                "living_document_recheck_failed",
                error=str(e),
            )

    async def _run_recheck(self):
        """Run the recheck engine to verify all sources"""
        try:
            # Check all sources due for rechecking
            results = await self.recheck_engine.check_all_sources()

            if results:
                logger.info(
                    f"   Sources checked: {results.get('total_checked', 0)}"
                )
                logger.info(
                    f"   Changes detected: {results.get('changes_detected', 0)}"
                )
                logger.info(
                    f"   Retractions found: {results.get('retractions_found', 0)}"
                )
                logger.info(f"   Dead links: {results.get('dead_links', 0)}")
                logger.info(
                    f"   Updates applied: {results.get('updates_applied', 0)}"
                )

                if results.get("retractions_found", 0) > 0:
                    logger.error(
                        "🚨 RETRACTIONS DETECTED - High priority alerts created"
                    )
                if results.get("dead_links", 0) > 0:
                    logger.warning("⚠️ DEAD LINKS DETECTED - Source access issues")
                if results.get("changes_detected", 0) > 0:
                    logger.info("ℹ️ SOURCE CHANGES DETECTED - Information updated")
            else:
                logger.info("   No sources due for recheck")

        except Exception as e:
            logger.error(f"❌ Recheck engine error: {str(e)}", exc_info=True)
            raise

    def trigger_manual_check(self):
        """Manually trigger a living document recheck (for testing/maintenance)"""
        logger.info("🔧 MANUAL LIVING DOCUMENT RECHECK TRIGGERED")
        try:
            asyncio.run(self._run_recheck())
            logger.info("✅ Manual recheck completed successfully")
            log_backend_event(
                "living_document_manual_check",
                status="success",
            )
            return {"status": "success", "message": "Manual recheck completed"}
        except Exception as e:
            logger.error(f"❌ Manual recheck failed: {str(e)}")
            log_backend_event(
                "living_document_manual_check",
                status="failed",
                error=str(e),
            )
            return {"status": "error", "message": str(e)}

    def get_status(self) -> dict:
        """Get scheduler status and next check time"""
        if not self.scheduler:
            return {
                "status": "not_initialized",
                "running": False,
                "last_check_at": None,
                "next_check_at": None,
                "jobs": [],
            }

        jobs_info = []
        if self.scheduler.get_jobs():
            for job in self.scheduler.get_jobs():
                next_run = job.next_run_time.isoformat() if job.next_run_time else None
                jobs_info.append(
                    {
                        "id": job.id,
                        "name": job.name,
                        "trigger": str(job.trigger),
                        "next_run": next_run,
                    }
                )

        return {
            "status": "running" if self.is_running else "stopped",
            "running": self.is_running,
            "last_check_at": self.last_check_at.isoformat()
            if self.last_check_at
            else None,
            "next_check_at": self.next_check_at.isoformat()
            if self.next_check_at
            else None,
            "jobs": jobs_info,
            "timestamp": datetime.utcnow().isoformat(),
        }


# Global scheduler instance
scheduler_manager = LivingDocumentScheduler()
