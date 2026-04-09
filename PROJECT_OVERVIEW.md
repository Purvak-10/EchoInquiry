# EchoInquiry — Complete Project Overview

> A multi-agent autonomous research system built with LangGraph that listens to queries, echoes structured plans, and rechecks sources like echoes in a canyon.

---

## Table of Contents

1. [Project Summary](#1-project-summary)
2. [Core Capabilities](#2-core-capabilities)
3. [Technology Stack](#3-technology-stack)
4. [Pipeline Agents](#4-pipeline-agents)
5. [Memory Systems](#5-memory-systems)
6. [External Data Sources](#6-external-data-sources)
7. [AWS Cloud Infrastructure](#7-aws-cloud-infrastructure)
8. [Tools Layer](#8-tools-layer)
9. [Living Document System](#9-living-document-system)
10. [Entry Points](#10-entry-points)
11. [Utilities](#11-utilities)
12. [Prompts](#12-prompts)
13. [Configuration Reference](#13-configuration-reference)
14. [Directory Structure](#14-directory-structure)
15. [Data Flow Summary](#15-data-flow-summary)

---

## 1. Project Summary

The **EchoInquiry** is a fully automated, multi-agent AI system that accepts a plain-language research query and produces a publication-quality structured research report. It is not a simple RAG (retrieval-augmented generation) chatbot — it implements a complete research methodology:

- **Hypothesis-driven research**: generates falsifiable hypotheses *before* searching, then tests them against retrieved evidence.
- **Multi-source retrieval**: queries Semantic Scholar, PubMed, CrossRef, and live web search in parallel, then optionally enriches academic results with Unpaywall/PDF processing.
- **Credibility scoring**: each source is scored on citation count, journal tier, and retraction status.
- **Contradiction detection**: uses sentence embeddings + capped LLM analysis to surface conflicting claims between high-priority sources.
- **Synthesis**: produces a structured report with executive summary, sections, hypothesis verdicts, contradictions, gaps, and citations.
- **Living documents**: background scheduler re-checks tracked report sources every 30 days for retractions, dead links, and citation updates.
- **Follow-up Q&A**: after report generation, users can ask natural-language follow-up questions answered in the context of the report.

---

## 2. Core Capabilities

| Capability | Description |
|---|---|
| Query Understanding | Parses raw natural-language queries into structured intent, domain, scope, sub-questions, and keywords |
| Research Planning | Generates a priority-ordered task graph with search strategies per sub-topic |
| Hypothesis Generation | Produces N falsifiable hypotheses with mechanism, predicted evidence, and falsification criteria |
| Multi-source Retrieval | Parallel search across SemanticScholar, PubMed, CrossRef, and the web, followed by optional Unpaywall/PDF enrichment |
| Credibility Scoring | Citation-count weighting + journal tier + retraction check = per-source credibility score (0–1) |
| Contradiction Detection | Semantic embedding similarity + bounded LLM analysis over ranked candidate claim pairs |
| Hypothesis Evaluation | Post-retrieval LLM verdict on each hypothesis with supporting and opposing evidence |
| Synthesis | Structured LLM synthesis of top-10 credible sources |
| Report Generation | Full structured report persisted to DynamoDB + S3, with optional local JSON/TXT export from the CLI |
| Living Documents | APScheduler 30-day background job rechecks every tracked report source |
| Follow-up Chat | Contextual Q&A loop grounded in the generated report |
| Email Delivery | SMTP or SendGrid email of the full report |
| CLI Interface | Rich terminal UI with live pipeline stream, coloured tables, email prompt |
| REST API | FastAPI server for programmatic access |

---

## 3. Technology Stack

### Core Framework
| Component | Library / Service |
|---|---|
| Agent orchestration | **LangGraph** `StateGraph` (v0.2+) |
| LLM | **Ollama** (local) — default model `llama3.2:1b` |
| LLM client abstraction | `langchain-ollama`, `langchain-core` |
| Async runtime | Python `asyncio` + `ThreadPoolExecutor` |

### AI / ML
| Component | Library |
|---|---|
| Sentence embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`, 384 dims) |
| Vector similarity | `numpy` cosine similarity |
| JSON generation | LLM with structured prompts + aggressive JSON repair |
| Observability | **Langfuse** (LLM traces, cost tracking) |

### Storage
| Component | Service / Library |
|---|---|
| Vector store | **Pinecone** (serverless) |
| Relational-style store | **AWS DynamoDB** (5 tables) |
| File / graph store | **AWS S3** (3 buckets) |
| In-memory graph | **NetworkX** `DiGraph` (serialised to S3 as JSON) |

### APIs & Web
| Component | Library |
|---|---|
| HTTP requests | `requests` |
| Web scraping | `BeautifulSoup4`, `lxml` |
| PDF parsing | **PyMuPDF** (`fitz`) |
| Retry / backoff | `backoff` |

### Infrastructure
| Component | Library / Service |
|---|---|
| REST API | **FastAPI** + **Uvicorn** |
| Background scheduler | **APScheduler** `BackgroundScheduler` |
| AWS SDK | **boto3** / **botocore** |

### Developer Experience
| Component | Library |
|---|---|
| Terminal UI | **Rich** (tables, panels, progress) |
| Environment config | `python-dotenv` |
| Testing | `pytest`, `pytest-mock`, `moto` (AWS mocks) |

---

## 4. Pipeline Agents

The research pipeline is implemented as a LangGraph `StateGraph` with 9 sequential nodes. Every node receives the full `ResearchState` dict and returns a partial update merged back into state.

### 4.1 `query_parser` — `agents/query_parser.py`

**Purpose**: Transform the raw user query string into a structured JSON object that all downstream agents use.

**Input** (`state` key): `raw_query` (str)

**Processing**:
- Renders `QUERY_PARSER_PROMPT` with the raw query
- Calls LLM via `llm_call_with_retry`
- Validates and extracts structured fields

**Output** (`state` key): `parsed_query` (dict) containing:
```
intent          — what the user wants (overview / comparison / deep-dive / ...)
domain          — subject area (medicine / physics / ...)
scope           — breadth of research
core_question   — single distilled research question
sub_questions   — list of subsidiary questions
ambiguities     — unclear aspects requiring assumption
keywords        — search terms
exclude_keywords — terms to avoid in retrieval
time_range      — publication date filter
output_format   — desired report format
is_academic     — whether the query should prefer academic literature
```

---

### 4.2 `research_planner` — `agents/research_planner.py`

**Purpose**: Produce a priority-sorted task graph that guides the retriever and downstream agents on *what to search for* and in *what order*.

**Input**: `parsed_query`

**Processing**:
- Renders `PLANNER_PROMPT` with parsed query fields
- LLM produces a `task_graph` — ordered list of search tasks
- Each task has: `task_id`, `task_type`, `description`, `depends_on`, `priority`, `keywords`, `target_sources`
- Falls back to a single generic web search task if LLM fails
- Validates task structure and sorts tasks by ascending priority

**Output**: `research_plan` (dict) containing:
```
task_graph               — sorted list of search tasks
estimated_depth          — shallow / moderate / deep
recommended_hypothesis_count
search_strategy          — breadth_first / depth_first / targeted
```

---

### 4.3 `hypothesis_generation` — `agents/hypothesis_engine.py` (Node 1 of 2)

**Purpose**: Generate falsifiable scientific hypotheses *before* any evidence is retrieved (pure prior-based reasoning).

**Input**: `parsed_query`

**Processing**:
- Uses `HYPOTHESIS_GENERATION_PROMPT` with core_question, domain, sub_questions, and a hypothesis count that defaults to `3` when absent
- LLM generates N hypotheses
- Each hypothesis is enriched with evaluation placeholders

**Output**: `hypotheses` (list of dicts), each containing:
```
id                    — unique identifier (h1, h2, ...)
statement             — the falsifiable claim
mechanism             — proposed causal mechanism
predicted_evidence    — what evidence would support it
falsification_criteria — what evidence would refute it
confidence_prior      — prior probability (0–1)
status                — "unverified" (placeholder)
confidence_posterior  — null (filled in evaluation step)
supporting_evidence   — [] (filled in evaluation step)
opposing_evidence     — [] (filled in evaluation step)
verdict               — "" (filled in evaluation step)
```

---

### 4.4 `retriever` — `agents/retriever.py`

**Purpose**: Fetch academic sources from multiple search backends in parallel, then deduplicate and optionally enrich results.

**Input**: `parsed_query`, `research_plan`

**Processing**:
- Extracts keywords from parsed_query and task_graph
- Uses `ThreadPoolExecutor(max_workers=4)` for concurrent search workers with a timeout
- Deduplication: first by DOI (exact match), then by title (normalised lowercase)
- Optionally enriches DOI-backed results with Unpaywall PDF discovery, PDF parsing, and embeddings after the initial search phase
- Falls back to simpler keyword extraction if plan is missing

**Sources queried**:
| Source | API Class | What it returns |
|---|---|---|
| Semantic Scholar | `SemanticScholarAPI` | Papers with citation counts, abstracts, DOIs |
| PubMed | `PubMedAPI` | Biomedical literature |
| CrossRef | `CrossrefAPI` | DOI metadata, publication details |
| Web Scraper | `WebScraper` | General web articles and blogs |
| Unpaywall | `UnpaywallAPI` | Post-retrieval open-access PDF discovery for DOI-backed papers |
| PDF Parser | `PDFParser` | Text extracted from downloaded PDFs during enrichment |

**Output**: `retrieved_sources` (list of dicts), each a standardised source object:
```
title, abstract, authors, year, doi, url,
source_api, citation_count, journal, full_text_snippet, s3_pdf_uri
```

---

### 4.5 `credibility_scorer` — `agents/credibility_scorer.py`

**Purpose**: Assign a credibility score to every retrieved source.

**Input**: `retrieved_sources`

**Scoring formula** (applied per source):

| Factor | Weight / Range | Details |
|---|---|---|
| Citation count | 0.2 – 0.9 | log-scaled; 0 citations → 0.2, 1000+ → 0.9 |
| Journal tier | 0.3 – 1.0 | high-tier = Nature/Science/NEJM/Lancet/Cell/JAMA/BMJ/IEEE/ACM/arXiv → 1.0 |
| DOI presence | 0.8 baseline | papers with a DOI get a 0.8 starting score |
| Retraction check | penalty | uses `RetractionChecker` tool; retracted papers scored near 0 |

**Output**: `retrieved_sources` updated with `credibility_score` (float 0–1) on each source.

---

### 4.6 `hypothesis_evaluation` — `agents/hypothesis_engine.py` (Node 2 of 2)

**Purpose**: Evaluate each hypothesis against the retrieved and scored sources.

**Input**: `hypotheses`, `retrieved_sources` (top sources by credibility)

**Processing**:
- For each hypothesis, selects relevant sources via keyword overlap
- Renders `HYPOTHESIS_EVALUATION_PROMPT` with hypothesis + evidence
- LLM produces verdict: `supported` / `refuted` / `inconclusive` / `partially_supported`
- Updates `confidence_posterior`, `supporting_evidence`, `opposing_evidence`, `verdict`

**Output**: `hypotheses` updated with evaluation results.

---

### 4.7 `contradiction_detector` — `agents/contradiction_detector.py`

**Purpose**: Find contradictory claims across the retrieved sources.

**Input**: `retrieved_sources`

**Processing**:
1. **Claim extraction**: LLM extracts factual claims from each source abstract (using `CLAIM_EXTRACTION_PROMPT`)
2. **Semantic similarity**: `SentenceTransformer` encodes claims and scores cross-source similarity
3. **Candidate pruning**: only top-ranked, above-threshold pairs are kept
4. **LLM analysis**: `CONTRADICTION_ANALYSIS_PROMPT` confirms and describes each contradiction
5. Returns bounded contradiction findings to downstream synthesis/output stages

**Output**: `contradictions` (list of dicts), each containing:
```
claim_a, claim_b      — the two contradicting claims
source_a, source_b    — where each claim came from
severity              — low / medium / high
explanation           — why they contradict
resolution_hint       — how to interpret or resolve the conflict
```

---

### 4.8 `synthesis_engine` — `agents/synthesis_engine.py`

**Purpose**: Produce a structured synthesis from the top credible sources.

**Input**: `retrieved_sources` (top 10 by credibility), `parsed_query`, `hypotheses`, `contradictions`

**Processing**:
- Filters sources for keyword relevance against `parsed_query.keywords`
- Selects top-10 by `credibility_score`
- Renders `SYNTHESIS_PROMPT` with all materials
- LLM synthesises a structured response

**Output**: `synthesis` (dict) containing structured research synthesis (thematic sections, key findings, evidence assessment).

---

### 4.9 `output_generator` — `agents/output_generator.py`

**Purpose**: Generate the final structured research report and persist the canonical report artifacts.

**Input**: All state fields — `synthesis`, `hypotheses`, `contradictions`, `retrieved_sources`, `parsed_query`, `research_plan`

**Processing**:
- Renders `OUTPUT_PROMPT` with all upstream results
- Calls the LLM through `llm_call_with_retry`
- Normalises and quality-checks the report; falls back to a grounded local builder when needed
- Persists the report to DynamoDB (`sessions` table)
- Uploads JSON + plain-text report artifacts to S3 (`reports` bucket)
- Local `.json` / `.txt` export is handled later by the CLI when the user opts in

**Output**: `final_report` (dict) containing:
```
title
executive_summary
sections                — list of themed sections with supporting sources
hypotheses_verdict      — evaluated hypotheses
contradictions_flagged  — all contradictions with severity
research_gaps           — identified gaps for future research
citations               — cited sources with title/author/year/DOI-or-URL metadata
confidence_overall      — overall report confidence (0–1)
follow_up_questions     — suggested follow-up questions
```

---

## 5. Memory Systems

### 5.1 Source Registry — `memory/source_registry.py`

Tracks report-cited sources across sessions in DynamoDB.

- **Purpose**: Deduplication across sessions, source reuse, living-document tracking
- **Storage**: DynamoDB `sources` table
- **Key operations**: `register_source()`, `register_all_from_report()`, `get_all_for_session()`, `get_alerts()`

### 5.2 Vector Store — `memory/vector_store.py`

Semantic search over all previously retrieved sources.

- **Backend**: **Pinecone** (serverless, `us-east-1`)
- **Index**: `research-passages` (or `PINECONE_INDEX_NAME` env var)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` — 384-dimensional dense vectors
- **Key operations**: `add_source()`, `add_sources_batch()`, `search()`, `search_across_sessions()`
- **Metadata stored per vector**: `session_id`, `title`, `doi`, `year`, `credibility_score`, `text` (preview)
- Validates embedding dimension matches config at startup and raises if it does not

### 5.3 Knowledge Graph — `memory/knowledge_graph.py`

Directed conceptual graph connecting research entities across sessions.

- **Backend**: **NetworkX** `DiGraph`, serialised as JSON to **AWS S3**
- **Node types**: `concept`, `source`, `claim`, `author`, `session`
- **Edge types**: `supports`, `contradicts`, `from_source`, `authored_by`, `cites`, `related_to`, `appears_in`
- **Key operations**: `add_source_node()`, `add_session_node()`, `get_related_concepts()`, `find_cross_session_connections()`, `save()` / `load()`
- Concepts are extracted from text by filtering stop-words and short tokens (length > 5)
- Serialised graph stored in the exports bucket at the configured `S3_KNOWLEDGE_GRAPH_KEY` (default `knowledge-graph/graph.json`)

---

## 6. External Data Sources

| API | What it provides | Rate limit handling |
|---|---|---|
| **Semantic Scholar** | Papers, citation counts, abstracts, authors | `backoff` with 3 retries, 1–2s delays |
| **PubMed** (NCBI E-utilities) | Biomedical and life-sciences literature | `backoff` with 3 retries |
| **CrossRef** | DOI resolution, metadata, journal info | `backoff` with 3 retries |
| **Unpaywall** | Open-access PDF URLs from DOIs | `backoff` with 3 retries |
| **Web Scraper** | General web articles | `BeautifulSoup4` HTML parsing |
| **CrossRef metadata** | Retraction status for DOIs | `tools/retraction_checker.py` |

All API wrappers return a normalised `_base_source()` dict with consistent field names, making them interchangeable for downstream processing.

---

## 7. AWS Cloud Infrastructure

### DynamoDB Tables

| Table (env var) | Contents |
|---|---|
| `TABLE_SESSIONS` | Per-session metadata, raw query, timestamps, final report JSON |
| `TABLE_SOURCES` | Tracked report sources with credibility and recheck metadata |
| `TABLE_HYPOTHESES` | Per-session hypothesis records with verdicts |
| `TABLE_CONTRADICTIONS` | Detected contradictions with severity |
| `TABLE_LIVING_DOC_CHECKS` | Living-document check records and alert state |

### S3 Buckets

| Bucket (env var) | Contents |
|---|---|
| `S3_BUCKET_REPORTS` | Final research reports as JSON and plain-text |
| `S3_BUCKET_PDFS` | Downloaded PDFs |
| `S3_BUCKET_EXPORTS` | Serialised knowledge graph and export artifacts |

### AWS Region

Default: `us-east-1` (configurable via `AWS_REGION` env var)

---

## 8. Tools Layer

### `tools/academic_apis.py` — Academic API Wrappers

Four API classes, all sharing a `_base_source()` normalised return format:

- **`SemanticScholarAPI`**: queries `/paper/search` endpoint, returns papers with `citation_count`
- **`PubMedAPI`**: uses NCBI E-utilities (`esearch` + `efetch`) for biomedical literature
- **`CrossrefAPI`**: queries `api.crossref.org/works` for DOI metadata
- **`UnpaywallAPI`**: queries `api.unpaywall.org/v2` for open-access PDF links

### `tools/web_scraper.py` — Web Scraper

- HTTP GET with `requests`
- HTML parsing via `BeautifulSoup4`
- Extracts: title, main body text, URL, domain
- Used for grey literature and non-indexed sources

### `tools/pdf_parser.py` — PDF Parser

- Uses **PyMuPDF** (`fitz`) to extract text from PDFs
- Parses already-downloaded PDF bytes into full text, sections, and references
- Returns extracted text blocks and metadata

### `tools/retraction_checker.py` — Retraction Checker

- Checks if a paper (by DOI) has been retracted
- Queries CrossRef metadata for retraction notices
- Returns `is_retracted: bool` — triggers credibility score reduction to near 0

---

## 9. Living Document System

### Purpose

Academic knowledge evolves. Papers get retracted, links die, citation counts change, and new evidence emerges. The living document system ensures research reports stay accurate over time.

### Components

#### `living_document/recheck_engine.py` — `SourceRecheckerEngine`

Performs the actual rechecking of individual sources:

| Check | What it does |
|---|---|
| Retraction check | Re-queries CrossRef for retraction status |
| Dead link check | HTTP `HEAD` request to source URL |
| Citation update | Re-queries CrossRef by DOI, then falls back to Semantic Scholar by title |
| Source record update | Updates `next_check_at`, timestamps, and changed metadata on the source record |

- Processes sources in async batches of up to 5 and uses `ThreadPoolExecutor` only for blocking link checks
- 0.5-second delay between requests (rate limiting)
- Reads sources due for recheck from the DynamoDB `sources` table via `next_check_at`

#### `scheduler/living_doc_scheduler.py` — `LivingDocumentScheduler`

- **Trigger**: Every 30 days (`IntervalTrigger(days=30)`)
- **Backend**: APScheduler `BackgroundScheduler`
- **Single-instance guard**: `max_instances=1` prevents concurrent runs
- **Lifecycle**: started on FastAPI app startup, or manually via CLI scheduler commands
- Logs `last_check_at` and `next_check_at` timestamps

### Scheduler CLI Commands

```bash
python cli.py scheduler start        # start 30-day background scheduler
python cli.py scheduler stop         # stop scheduler
python cli.py scheduler status       # show next/last check times
python cli.py scheduler check-now    # manually trigger an immediate recheck
```

---

## 10. Entry Points

### 10.1 CLI — `cli.py`

The primary interactive interface. Built with ANSI colour codes and Rich library.

**Usage:**
```bash
python cli.py                               # interactive prompt
python cli.py "who created the atom bomb"   # direct query
python cli.py scheduler start               # start scheduler
python cli.py scheduler stop                # stop scheduler
python cli.py scheduler status              # check scheduler
python cli.py scheduler check-now           # manual recheck
```

**Features:**
- Live pipeline stream — prints each node's result as it completes
- Full report display: executive summary, research sections, hypotheses table, conclusions, contradictions table (colour-coded by severity), research gaps, citations table, confidence score, follow-up recommendations
- Email prompt offered after the full report and again at the end of the session
- Follow-up Q&A chat loop using `FollowupAgent`
- Save report as JSON and plain-text files

### 10.2 REST API — `main.py` (FastAPI)

**Base URL**: `http://localhost:8000`

**Lifecycle**: `lifespan` context manager starts `LivingDocumentScheduler` on startup and stops it on shutdown.

**CORS**: enabled for all origins (development mode)

**Key endpoints** (REST API is secondary; CLI is primary entry point):
```
GET  /health                     — health check
GET  /scheduler/status           — scheduler status
POST /scheduler/trigger-check    — manually trigger living document recheck
```

**Note**: Full pipeline execution happens via CLI: `python cli.py "your query"`. The REST API provides health checks and scheduler management only.

---

## 11. Utilities

### `utils/llm_helpers.py`

- `get_llm_client()` — returns configured Ollama LLM instance
- `llm_call_with_retry()` — LLM call with retry logic, fallback value support, and Langfuse tracing
- `llm_stream()` — streaming LLM call for the output generator

### `utils/backend_logging.py`

- `log_backend_event(event_name, **kwargs)` — structured event logging
- Used by every pipeline agent to capture inputs, outputs, and timing

### `utils/llm_usage.py`

- Tracks token consumption per session
- Reports prompt tokens, completion tokens, total tokens

### `utils/email_sender.py`

Supports two email backends, configurable via `EMAIL_BACKEND` env var:

| Backend | Config vars | Notes |
|---|---|---|
| **SMTP** | `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD` | Works with Gmail, custom SMTP |
| **SendGrid** | `SENDGRID_API_KEY`, `SENDGRID_FROM_EMAIL` | Recommended; 100 free emails/day |

- Sends both HTML and plain-text versions of the report
- Beautiful HTML template with styled sections

---

## 12. Prompts

All prompts are in `prompts/`. They are plain Python string constants imported by the agents.

| File | Used by | What it produces |
|---|---|---|
| `query_parser_prompt.py` | `query_parser` | Structured JSON: intent, domain, keywords, sub-questions |
| `planner_prompt.py` | `research_planner` | Task graph JSON with priority-ordered search tasks |
| `hypothesis_prompt.py` | `hypothesis_engine` | Hypotheses JSON + evaluation verdict JSON |
| `synthesis_prompt.py` | `synthesis_engine` | Structured synthesis with themed sections |
| `output_prompt.py` | `output_generator` | Full report JSON with all sections |
| `contradiction_prompt.py` | `contradiction_detector` | Claim extraction + contradiction analysis JSON |
| `followup_prompt.py` | `followup_agent` | Contextual follow-up answer in report context |

---

## 13. Configuration Reference

All configuration is in `config.py`, loaded from environment variables via `os.getenv()` with safe defaults.

### LLM Configuration
| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.2:1b` | Default model |
| `OLLAMA_REQUEST_TIMEOUT_SECONDS` | `120` | Request timeout (seconds) |

### AWS Configuration
| Variable | Default | Description |
|---|---|---|
| `AWS_REGION` | `us-east-1` | AWS region for all services |
| `AWS_ACCESS_KEY_ID` | — | IAM key (or use instance role) |
| `AWS_SECRET_ACCESS_KEY` | — | IAM secret |

### Pinecone Configuration
| Variable | Default | Description |
|---|---|---|
| `PINECONE_API_KEY` | — | Pinecone API key |
| `PINECONE_INDEX_NAME` | `research-passages` | Vector index name |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `EMBEDDING_DIMENSION` | `384` | Vector dimension |

### DynamoDB Table Names
| Variable | Default |
|---|---|
| `TABLE_SESSIONS` | `research-agent-sessions` |
| `TABLE_SOURCES` | `research-agent-sources` |
| `TABLE_HYPOTHESES` | `research-agent-hypotheses` |
| `TABLE_CONTRADICTIONS` | `research-agent-contradictions` |
| `TABLE_LIVING_DOC_CHECKS` | `research-agent-living-doc-checks` |

### S3 Bucket Names
| Variable | Default |
|---|---|
| `S3_BUCKET_REPORTS` | `research-agent-reports` |
| `S3_BUCKET_PDFS` | `research-agent-pdfs` |
| `S3_BUCKET_EXPORTS` | `research-agent-exports` |

### Email Configuration
| Variable | Default | Description |
|---|---|---|
| `EMAIL_BACKEND` | `smtp` | `smtp` or `sendgrid` |
| `SMTP_HOST` | `smtp.gmail.com` | SMTP server |
| `SMTP_PORT` | `587` | SMTP port |
| `SMTP_USER` | — | Sender email |
| `SMTP_PASSWORD` | — | Password / app-password |
| `SENDGRID_API_KEY` | — | SendGrid API key |
| `SENDGRID_FROM_EMAIL` | — | Verified sender email |

### Langfuse Observability
| Variable | Description |
|---|---|
| `LANGFUSE_SECRET_KEY` | Langfuse project secret key |
| `LANGFUSE_PUBLIC_KEY` | Langfuse project public key |
| `LANGFUSE_HOST` | Langfuse server URL |

---

## 14. Directory Structure

```
research_agent/              ← EchoInquiry project root
│
├── cli.py                    ← Primary CLI entry point
├── main.py                   ← FastAPI REST API entry point
├── config.py                 ← All configuration from env vars
├── requirements.txt          ← All Python dependencies
│
├── graph/
│   ├── research_graph.py     ← LangGraph StateGraph definition (9 nodes)
│   └── state.py              ← ResearchState TypedDict
│
├── agents/                   ← Pipeline nodes plus follow-up chat
│   ├── query_parser.py       ← Node 1: parse raw query
│   ├── research_planner.py   ← Node 2: plan search tasks
│   ├── hypothesis_engine.py  ← Node 3 & 6: generate + evaluate hypotheses
│   ├── retriever.py          ← Node 4: multi-source retrieval
│   ├── credibility_scorer.py ← Node 5: score sources
│   ├── contradiction_detector.py ← Node 7: find conflicting claims
│   ├── synthesis_engine.py   ← Node 8: synthesise findings
│   ├── output_generator.py   ← Node 9: generate final report
│   └── followup_agent.py     ← Post-pipeline: Q&A chat
│
├── prompts/                  ← Prompt strings for LLM-backed stages
│   ├── query_parser_prompt.py
│   ├── planner_prompt.py
│   ├── hypothesis_prompt.py
│   ├── synthesis_prompt.py
│   ├── output_prompt.py
│   ├── contradiction_prompt.py
│   └── followup_prompt.py
│
├── memory/                   ← Persistent memory systems
│   ├── source_registry.py    ← DynamoDB source tracking
│   ├── vector_store.py       ← Pinecone vector index
│   └── knowledge_graph.py    ← NetworkX graph → S3
│
├── tools/                    ← External data acquisition
│   ├── academic_apis.py      ← SemanticScholar, PubMed, CrossRef, Unpaywall
│   ├── web_scraper.py        ← BeautifulSoup4 web scraping
│   ├── pdf_parser.py         ← PyMuPDF PDF text extraction
│   └── retraction_checker.py ← CrossRef retraction status
│
├── aws/
│   ├── dynamodb_client.py    ← DynamoDB CRUD operations
│   ├── s3_client.py          ← S3 upload/download/list
│   └── llm_client.py         ← Ollama client wrapper
│
├── living_document/
│   ├── recheck_engine.py     ← Source rechecking logic
│   └── __init__.py
│
├── scheduler/
│   ├── living_doc_scheduler.py ← APScheduler 30-day job
│   └── __init__.py
│
└── utils/
    ├── llm_helpers.py        ← LLM client, retry, streaming
    ├── llm_usage.py          ← Token usage tracking
    ├── backend_logging.py    ← Event logging
    └── email_sender.py       ← SMTP / SendGrid email
```

---

## 15. Data Flow Summary

```
User Query (str)
       │
       ▼
  query_parser        → parsed_query (intent, domain, keywords, sub-questions)
       │
       ▼
  research_planner    → research_plan (priority task graph)
       │
       ▼
  hypothesis_gen      → hypotheses[] (falsifiable, with priors)
       │
       ▼
  retriever           → retrieved_sources[] (from 4 parallel search workers + optional PDF enrichment)
       │
       ▼
  credibility_scorer  → retrieved_sources[] + credibility_score per source
       │
       ▼
  hypothesis_eval     → hypotheses[] updated (verdict, posterior confidence)
       │
       ▼
  contradiction_det   → contradictions[] (semantic similarity + LLM)
       │
       ▼
  synthesis_engine    → synthesis{} (structured from top-10 sources)
       │
       ▼
  output_generator    → final_report{} (full structured report)
       │
       ├──→ DynamoDB (session report)
       ├──→ S3 (report JSON + TXT files)
       ├──→ Pinecone (source embeddings)
       ├──→ NetworkX/S3 (knowledge graph update)
       └──→ DynamoDB (sources + hypotheses + contradictions in post-pipeline persistence)
       │
       ▼
  CLI display / API response / Email delivery
       │
       ▼
  FollowupAgent (optional Q&A)
       │
       ▼
  LivingDocScheduler (30-day background recheck)
```
