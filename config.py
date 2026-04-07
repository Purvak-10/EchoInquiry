"""
Global configuration for Research Agent
All values are loaded from environment variables with safe defaults.

Uses python-dotenv for local development.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Load local .env if present
load_dotenv()

# Project Root Path
PROJECT_ROOT: Path = Path(__file__).resolve().parent

LLM_BACKEND: str = "ollama"
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
OLLAMA_REQUEST_TIMEOUT_SECONDS: float = float(
    os.getenv("OLLAMA_REQUEST_TIMEOUT_SECONDS", "120")
)
# ============================================================
# AWS CONFIGURATION
# ============================================================

AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1").strip()
AWS_ACCOUNT_ID: str = os.getenv("AWS_ACCOUNT_ID", "").strip()

# ============================================================
# PINECONE
# ============================================================

PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME: str = os.getenv(
    "PINECONE_INDEX_NAME",
    "research-passages",
)

PINECONE_CLOUD: str = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION: str = os.getenv("PINECONE_REGION", "us-east-1")

EMBEDDING_DIMENSION: int = int(
    os.getenv("EMBEDDING_DIMENSION", "384")
)

EMBEDDING_MODEL_NAME: str = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "all-MiniLM-L6-v2",
)


# ============================================================
# DYNAMODB
# ============================================================

TABLE_SESSIONS: str = os.getenv(
    "TABLE_SESSIONS",
    "research-agent-sessions",
)

TABLE_SOURCES: str = os.getenv(
    "TABLE_SOURCES",
    "research-agent-sources",
)

TABLE_HYPOTHESES: str = os.getenv(
    "TABLE_HYPOTHESES",
    "research-agent-hypotheses",
)

TABLE_CONTRADICTIONS: str = os.getenv(
    "TABLE_CONTRADICTIONS",
    "research-agent-contradictions",
)

TABLE_LIVING_DOC_CHECKS: str = os.getenv(
    "TABLE_LIVING_DOC_CHECKS",
    "research-agent-living-doc-checks",
)


# ============================================================
# S3
# ============================================================

S3_BUCKET_REPORTS: str = os.getenv(
    "S3_BUCKET_REPORTS",
    "research-agent-reports",
)

S3_BUCKET_PDFS: str = os.getenv(
    "S3_BUCKET_PDFS",
    "research-agent-pdfs",
)

S3_BUCKET_EXPORTS: str = os.getenv(
    "S3_BUCKET_EXPORTS",
    "research-agent-exports",
)

S3_REPORTS_PREFIX: str = os.getenv(
    "S3_REPORTS_PREFIX",
    "reports/",
)

S3_PDFS_PREFIX: str = os.getenv(
    "S3_PDFS_PREFIX",
    "pdfs/",
)

S3_KNOWLEDGE_GRAPH_KEY: str = os.getenv(
    "S3_KNOWLEDGE_GRAPH_KEY",
    "knowledge-graph/graph.json",
)


# ============================================================
# SECRETS MANAGER
# ============================================================

SECRETS_NAME: str = os.getenv(
    "SECRETS_NAME",
    "research-agent/config",
)


# ============================================================
# AGENT BEHAVIOUR
# ============================================================

MAX_SOURCES_PER_QUERY: int = int(
    os.getenv("MAX_SOURCES_PER_QUERY", "20")
)

HYPOTHESIS_COUNT: int = int(
    os.getenv("HYPOTHESIS_COUNT", "5")
)

CONTRADICTION_SIMILARITY_THRESHOLD: float = float(
    os.getenv("CONTRADICTION_SIMILARITY_THRESHOLD", "0.75")
)

CONTRADICTION_MAX_SOURCES: int = int(
    os.getenv("CONTRADICTION_MAX_SOURCES", "6")
)

CONTRADICTION_MAX_CLAIMS_PER_SOURCE: int = int(
    os.getenv("CONTRADICTION_MAX_CLAIMS_PER_SOURCE", "5")
)

CONTRADICTION_MAX_CANDIDATE_PAIRS: int = int(
    os.getenv("CONTRADICTION_MAX_CANDIDATE_PAIRS", "30")
)

CONTRADICTION_MAX_ANALYSIS_PAIRS: int = int(
    os.getenv("CONTRADICTION_MAX_ANALYSIS_PAIRS", "12")
)

CONTRADICTION_MAX_STAGE_SECONDS: float = float(
    os.getenv("CONTRADICTION_MAX_STAGE_SECONDS", "180")
)

CONTRADICTION_SOURCE_CONTENT_MAX_CHARS: int = int(
    os.getenv("CONTRADICTION_SOURCE_CONTENT_MAX_CHARS", "3000")
)

SEMANTIC_CHANGE_THRESHOLD: float = float(
    os.getenv("SEMANTIC_CHANGE_THRESHOLD", "0.97")
)

LLM_MAX_RETRIES: int = int(
    os.getenv("LLM_MAX_RETRIES", "2")
)

LLM_RETRY_DELAY_SECONDS: float = float(
    os.getenv("LLM_RETRY_DELAY_SECONDS", "1.0")
)

LIVING_DOC_CHECK_INTERVAL_DAYS: int = int(
    os.getenv("LIVING_DOC_CHECK_INTERVAL_DAYS", "30")
)

RETRIEVER_PARALLEL_TIMEOUT_SECONDS: float = float(
    os.getenv("RETRIEVER_PARALLEL_TIMEOUT_SECONDS", "45")
)

RETRIEVER_PDF_ENRICH_MAX_SECONDS: float = float(
    os.getenv("RETRIEVER_PDF_ENRICH_MAX_SECONDS", "60")
)

RETRIEVER_EMBEDDING_MAX_SECONDS: float = float(
    os.getenv("RETRIEVER_EMBEDDING_MAX_SECONDS", "60")
)

RETRIEVER_MAX_PDF_ITEMS: int = int(
    os.getenv("RETRIEVER_MAX_PDF_ITEMS", "8")
)

RETRIEVER_MAX_EMBED_ITEMS: int = int(
    os.getenv("RETRIEVER_MAX_EMBED_ITEMS", "12")
)


# ============================================================
# OBSERVABILITY
# ============================================================

BACKEND_EVENTS_LOG_PATH: str = os.getenv(
    "BACKEND_EVENTS_LOG_PATH",
    "logs/backend_events.jsonl",
)

BACKEND_LOG_INCLUDE_CONTENT: bool = os.getenv(
    "BACKEND_LOG_INCLUDE_CONTENT",
    "false",
).lower() in {"1", "true", "yes", "on"}

TOKEN_USAGE_LOG_PATH: str = os.getenv(
    "TOKEN_USAGE_LOG_PATH",
    "logs/llm_usage.jsonl",
)

LANGFUSE_ENABLED: bool = os.getenv(
    "LANGFUSE_ENABLED",
    "false",
).lower() in {"1", "true", "yes", "on"}

LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST: str = os.getenv(
    "LANGFUSE_HOST",
    "http://localhost:3000",
)

LANGFUSE_CAPTURE_CONTENT: bool = os.getenv(
    "LANGFUSE_CAPTURE_CONTENT",
    "false",
).lower() in {"1", "true", "yes", "on"}


# ============================================================
# MCP
# ============================================================

MCP_SERVER_NAME: str = os.getenv(
    "MCP_SERVER_NAME",
    "research-agent",
)

MCP_SERVER_VERSION: str = os.getenv(
    "MCP_SERVER_VERSION",
    "1.0.0",
)


def get_runtime_config_issues() -> list[str]:
    issues: list[str] = []

    if not OLLAMA_MODEL:
        issues.append("OLLAMA_MODEL is empty.")

    if OLLAMA_REQUEST_TIMEOUT_SECONDS <= 0:
        issues.append("OLLAMA_REQUEST_TIMEOUT_SECONDS must be > 0.")

    if LLM_MAX_RETRIES < 1:
        issues.append("LLM_MAX_RETRIES must be >= 1.")

    if RETRIEVER_PARALLEL_TIMEOUT_SECONDS <= 0:
        issues.append("RETRIEVER_PARALLEL_TIMEOUT_SECONDS must be > 0.")

    if RETRIEVER_PDF_ENRICH_MAX_SECONDS <= 0:
        issues.append("RETRIEVER_PDF_ENRICH_MAX_SECONDS must be > 0.")

    if RETRIEVER_EMBEDDING_MAX_SECONDS <= 0:
        issues.append("RETRIEVER_EMBEDDING_MAX_SECONDS must be > 0.")

    if RETRIEVER_MAX_PDF_ITEMS < 1:
        issues.append("RETRIEVER_MAX_PDF_ITEMS must be >= 1.")

    if RETRIEVER_MAX_EMBED_ITEMS < 1:
        issues.append("RETRIEVER_MAX_EMBED_ITEMS must be >= 1.")

    if CONTRADICTION_MAX_SOURCES < 1:
        issues.append("CONTRADICTION_MAX_SOURCES must be >= 1.")

    if CONTRADICTION_MAX_CLAIMS_PER_SOURCE < 1:
        issues.append("CONTRADICTION_MAX_CLAIMS_PER_SOURCE must be >= 1.")

    if CONTRADICTION_MAX_CANDIDATE_PAIRS < 1:
        issues.append("CONTRADICTION_MAX_CANDIDATE_PAIRS must be >= 1.")

    if CONTRADICTION_MAX_ANALYSIS_PAIRS < 1:
        issues.append("CONTRADICTION_MAX_ANALYSIS_PAIRS must be >= 1.")

    if CONTRADICTION_MAX_STAGE_SECONDS <= 0:
        issues.append("CONTRADICTION_MAX_STAGE_SECONDS must be > 0.")

    if CONTRADICTION_SOURCE_CONTENT_MAX_CHARS < 200:
        issues.append("CONTRADICTION_SOURCE_CONTENT_MAX_CHARS must be >= 200.")

    if LANGFUSE_ENABLED and (not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY):
        issues.append("LANGFUSE_ENABLED is true but Langfuse keys are missing.")

    return issues
