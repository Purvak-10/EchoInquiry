"""
FastAPI Entry Point for Research Agent with Living Document Scheduler
- Manages lifecycle of scheduler
- Exposes health check and scheduler status endpoints
- Ready for production deployment
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from scheduler.living_doc_scheduler import scheduler_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ========== LIFECYCLE MANAGEMENT ==========

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app startup and shutdown with scheduler lifecycle"""
    
    # ===== STARTUP =====
    logger.info("=" * 80)
    logger.info("🚀 RESEARCH AGENT STARTUP")
    logger.info("=" * 80)
    
    try:
        # Initialize API (orchestrator agent, etc.)
        logger.info("Initializing API components...")
        
        # Start the scheduler for living document checks
        logger.info("Starting Living Document Scheduler...")
        scheduler_manager.start()
        
        logger.info("=" * 80)
        logger.info("✅ ALL SYSTEMS READY")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"❌ STARTUP FAILED: {str(e)}", exc_info=True)
        raise
    
    yield
    
    # ===== SHUTDOWN =====
    logger.info("=" * 80)
    logger.info("🛑 RESEARCH AGENT SHUTDOWN")
    logger.info("=" * 80)
    
    try:
        logger.info("Stopping Living Document Scheduler...")
        scheduler_manager.stop()
        logger.info("✅ Scheduler stopped cleanly")
    except Exception as e:
        logger.error(f"❌ Shutdown error: {str(e)}", exc_info=True)

# ========== APP INITIALIZATION ==========

# Create main app with scheduler lifecycle
app = FastAPI(
    title="Research Agent API",
    description="Multi-agent research system with living documents",
    version="2.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== BUILT-IN ENDPOINTS ==========

@app.get("/health")
async def health_check():
    """System health check (async endpoint)"""
    return {
        "status": "healthy",
        "component": "main",
        "scheduler": scheduler_manager.get_status(),
    }

def get_health_status():
    """Get synchronous health check status"""
    return {
        "status": "healthy",
        "component": "main",
        "scheduler": scheduler_manager.get_status(),
    }

@app.get("/scheduler/status")
async def scheduler_status():
    """Get detailed scheduler status"""
    return scheduler_manager.get_status()

@app.post("/scheduler/trigger-check")
async def trigger_living_doc_check():
    """Manually trigger a living document check (for testing)"""
    result = scheduler_manager.trigger_manual_check()
    return result

# ========== MAIN ==========

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting Uvicorn server...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload in production
    )
