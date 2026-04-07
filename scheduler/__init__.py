"""Scheduler module for living document checks"""
try:
    from scheduler.living_doc_scheduler import scheduler_manager
    __all__ = ["scheduler_manager"]
except ImportError:
    scheduler_manager = None
    __all__ = ["scheduler_manager"]
