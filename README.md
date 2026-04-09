<div align="center">

# 🔬 EchoInquiry

### Autonomous Multi-Agent Academic Research System

*Listens to your query. Echoes back a publication-quality research report.*

> *"Reflects how the system listens to queries, echoes structured plans, and rechecks sources like echoes in a canyon."*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-green?logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Ollama](https://img.shields.io/badge/LLM-Ollama%20%7C%20llama3.2-orange)](https://ollama.com)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Pinecone](https://img.shields.io/badge/VectorDB-Pinecone-purple)](https://www.pinecone.io)
[![AWS](https://img.shields.io/badge/Cloud-AWS%20DynamoDB%20%7C%20S3-orange?logo=amazon-aws&logoColor=white)](https://aws.amazon.com)

</div>

---

## 📖 What Is This?

**EchoInquiry** is a fully autonomous, hypothesis-driven research system. Give it any research question and it:

1. **Understands** your query — parses intent, domain, sub-questions, and keywords
2. **Plans** a search strategy — generates a prioritised task graph
3. **Hypothesises** — creates falsifiable scientific hypotheses *before* searching
4. **Retrieves** — queries Semantic Scholar, PubMed, CrossRef, and the web in parallel, then enriches academic results with Unpaywall and PDF parsing when available
5. **Scores credibility** — citation counts, journal tier, retraction status
6. **Evaluates hypotheses** — tests each hypothesis against retrieved evidence
7. **Detects contradictions** — extracts claims, ranks semantically similar evidence, and runs bounded contradiction analysis
8. **Synthesises** — structured narrative from the top 10 credible sources
9. **Reports** — structured, publication-quality research report
10. **Remembers** — sources referenced in the persisted final report are tracked for 30-day rechecks via a background scheduler

All powered by a local LLM (Ollama) — **no OpenAI API key required**.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 **Hypothesis-Driven Research** | Generates falsifiable hypotheses before searching, then evaluates them against evidence |
| 🔍 **Multi-Source Retrieval** | Queries Semantic Scholar, PubMed, CrossRef, and the web in parallel, then enriches academic results with Unpaywall and PDFs when available |
| ⭐ **Credibility Scoring** | Each source scored 0–1 based on citations, journal tier, and retraction status |
| ⚡ **Contradiction Detection** | Uses claim extraction, sentence embeddings, and capped LLM checks to flag conflicting evidence |
| 📊 **Structured Reports** | Executive summary, sections, hypotheses verdict, citations, and overall confidence |
| 🔄 **Living Documents** | Background scheduler rechecks tracked report sources every 30 days |
| 💬 **Follow-up Q&A** | Chat with your report — grounded follow-up answers |
| 📧 **Email Delivery** | Send reports via SMTP or SendGrid |
| 🖥️ **Rich Terminal UI** | Colour-coded live pipeline stream, tables, and panels |
| 🌐 **REST API** | FastAPI endpoints for health checks and scheduler management |
| 🏠 **Local LLM** | Runs entirely with Ollama — no cloud LLM costs |
| 🗄️ **Cloud Persistence** | AWS DynamoDB (5 tables) + S3 (3 buckets) + Pinecone vector index |

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| [Ollama](https://ollama.com) | Latest |
| AWS account | (for DynamoDB + S3) |
| [Pinecone](https://pinecone.io) account | Free tier OK |

### 1. Clone & Install

```bash
git clone https://github.com/Purvak-10/EchoInquiry.git
cd EchoInquiry
pip install -r requirements.txt
```

### 2. Pull the LLM

```bash
ollama pull llama3.2:1b
```

> Any Ollama model works. Larger models (e.g., `llama3.1:8b`) produce better results.

### 3. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your credentials (see [Configuration](#-configuration) below).

### 4. Run Your First Research

```bash
python cli.py "What are the effects of sleep deprivation on memory consolidation?"
```

That's it. The agent will stream each pipeline step to your terminal and display the full report.

---

## 🖥️ CLI Usage

```bash
# Interactive mode — you'll be prompted for a query
python cli.py

# Pass query directly
python cli.py "impact of climate change on coral reef biodiversity"

# Scheduler commands
python cli.py scheduler start        # start 30-day background source recheck
python cli.py scheduler stop         # stop the scheduler
python cli.py scheduler status       # show next/last check times
python cli.py scheduler check-now    # trigger an immediate recheck
```

### Sample CLI Output

```
╔══════════════════════════════════════════════════════════════╗
║           🔬 ECHOINQUIRY — PIPELINE LIVE STREAM              ║
╚══════════════════════════════════════════════════════════════╝

  ✅ [1/9] Query Parser        → intent: deep-dive | domain: neuroscience
  ✅ [2/9] Research Planner    → 4 search tasks generated
  ✅ [3/9] Hypothesis Gen      → 3 hypotheses formulated
  ✅ [4/9] Retriever           → 47 sources retrieved (31 after dedup)
  ✅ [5/9] Credibility Scorer  → scored 31 sources
  ✅ [6/9] Hypothesis Eval     → 2 supported, 1 inconclusive
  ✅ [7/9] Contradiction Det   → 2 contradictions flagged after bounded analysis
  ✅ [8/9] Synthesis Engine    → synthesis complete
  ✅ [9/9] Output Generator    → report saved

╔══════════════════════════════════════════════════════════════╗
║                     EXECUTIVE SUMMARY                        ║
╠══════════════════════════════════════════════════════════════╣
║  Sleep deprivation significantly impairs memory              ║
║  consolidation, particularly for declarative memory...       ║
╚══════════════════════════════════════════════════════════════╝

  Confidence Score: ████████░░  0.82 / 1.00

┌─────────────────────────────────── HYPOTHESES ────────────────────────────────────┐
│  H1: Sleep deprivation reduces hippocampal activation...     SUPPORTED    (0.87)  │
│  H2: REM sleep is critical for procedural memory...          SUPPORTED    (0.79)  │
│  H3: Caffeine fully compensates for sleep loss...            REFUTED      (0.12)  │
└────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🌐 REST API

Start the API server:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Available Endpoints

#### Health Check
```
GET /health
  Returns: {"status": "healthy", "component": "main", "scheduler": {...}}
```

#### Living Document Scheduler Status
```
GET /scheduler/status
  Returns: Scheduler status with next check time and jobs
```

#### Trigger Manual Living Document Check
```
POST /scheduler/trigger-check
  Returns: {"status": "success", "message": "Manual recheck completed"}
  Purpose: Manually run source rechecking (for testing/maintenance)
```

### Example

```bash
# Check API health
curl -X GET http://localhost:8000/health

# Get scheduler status
curl -X GET http://localhost:8000/scheduler/status

# Trigger a manual living document recheck
curl -X POST http://localhost:8000/scheduler/trigger-check
```

> **Note**: The primary research pipeline is accessed via CLI (`python cli.py`). The REST API currently provides health checks and scheduler management. For full pipeline execution, use the CLI interface.

---

## 📦 Project Structure

```
research_agent/              ← EchoInquiry project root
│
├── cli.py                    ← Rich terminal UI (primary entry point)
├── main.py                   ← FastAPI REST API
├── config.py                 ← All config from environment variables
├── requirements.txt
│
├── graph/
│   ├── research_graph.py     ← LangGraph StateGraph (9 nodes)
│   └── state.py              ← ResearchState TypedDict
│
├── agents/                   ← Pipeline nodes plus follow-up chat logic
│   ├── query_parser.py
│   ├── research_planner.py
│   ├── hypothesis_engine.py  ← Used for both generation & evaluation
│   ├── retriever.py
│   ├── credibility_scorer.py
│   ├── contradiction_detector.py
│   ├── synthesis_engine.py
│   ├── output_generator.py
│   └── followup_agent.py
│
├── prompts/                  ← Prompt templates for LLM-backed stages
├── memory/                   ← Pinecone, DynamoDB registry, NetworkX graph
├── tools/                    ← Academic APIs, web scraper, PDF parser, retraction checker
├── aws/                      ← DynamoDB, S3 clients
├── living_document/          ← Source recheck engine
├── scheduler/                ← APScheduler 30-day job
└── utils/                    ← LLM helpers, email sender, logging
```

---

## 🏗️ Architecture Overview

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                  LANGGRAPH PIPELINE (9 nodes)                │
│                                                             │
│  query_parser → research_planner → hypothesis_generation    │
│       → retriever → credibility_scorer → hypothesis_eval    │
│       → contradiction_detector → synthesis → output_gen     │
└─────────────────────────┬───────────────────────────────────┘
                          │
            ┌─────────────┼──────────────────┐
            ▼             ▼                  ▼
       DynamoDB        Pinecone           S3 + NetworkX
       (5 tables)    (vector index)     (files + graph)
```

**Full architecture details**: see [`ARCHITECTURE.md`](ARCHITECTURE.md)  
**Complete component reference**: see [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md)

---

## ⚙️ Configuration

Copy `.env.example` to `.env` and fill in your values:

```bash
# ── LLM (Ollama) ────────────────────────────────────────────────
OLLAMA_MODEL=llama3.2:1b          # or llama3.1:8b, mistral, etc.
OLLAMA_REQUEST_TIMEOUT_SECONDS=120

# ── AWS ─────────────────────────────────────────────────────────
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_key_here       # standard AWS env var
AWS_SECRET_ACCESS_KEY=your_secret_here

# ── DynamoDB Tables ─────────────────────────────────────────────
TABLE_SESSIONS=research-agent-sessions
TABLE_SOURCES=research-agent-sources
TABLE_HYPOTHESES=research-agent-hypotheses
TABLE_CONTRADICTIONS=research-agent-contradictions
TABLE_LIVING_DOC_CHECKS=research-agent-living-doc-checks

# ── S3 Buckets ──────────────────────────────────────────────────
S3_BUCKET_REPORTS=research-agent-reports
S3_BUCKET_PDFS=research-agent-pdfs
S3_BUCKET_EXPORTS=research-agent-exports

# ── Pinecone ────────────────────────────────────────────────────
PINECONE_API_KEY=your_pinecone_key_here
PINECONE_INDEX_NAME=research-passages

# ── Email (choose one backend) ──────────────────────────────────
EMAIL_BACKEND=smtp                 # or sendgrid
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your@gmail.com
SMTP_PASSWORD=your_app_password

# SENDGRID_API_KEY=your_key         # Alternative: SendGrid
# SENDGRID_FROM_EMAIL=noreply@you.com

# ── Observability (optional) ────────────────────────────────────
LANGFUSE_SECRET_KEY=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com
```

---

## 🔧 AWS Setup

Create the required DynamoDB tables and S3 buckets before first run:

```bash
# Create DynamoDB tables
aws dynamodb create-table --table-name research-agent-sessions \
  --attribute-definitions AttributeName=session_id,AttributeType=S \
  --key-schema AttributeName=session_id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST --region us-east-1

aws dynamodb create-table --table-name research-agent-sources \
  --attribute-definitions AttributeName=source_id,AttributeType=S AttributeName=session_id,AttributeType=S \
  --key-schema AttributeName=source_id,KeyType=HASH \
  --global-secondary-indexes '[{"IndexName":"session-index","KeySchema":[{"AttributeName":"session_id","KeyType":"HASH"}],"Projection":{"ProjectionType":"ALL"}}]' \
  --billing-mode PAY_PER_REQUEST --region us-east-1

aws dynamodb create-table --table-name research-agent-hypotheses \
  --attribute-definitions AttributeName=hypothesis_id,AttributeType=S AttributeName=session_id,AttributeType=S \
  --key-schema AttributeName=hypothesis_id,KeyType=HASH \
  --global-secondary-indexes '[{"IndexName":"session-index","KeySchema":[{"AttributeName":"session_id","KeyType":"HASH"}],"Projection":{"ProjectionType":"ALL"}}]' \
  --billing-mode PAY_PER_REQUEST --region us-east-1

aws dynamodb create-table --table-name research-agent-contradictions \
  --attribute-definitions AttributeName=contradiction_id,AttributeType=S AttributeName=session_id,AttributeType=S \
  --key-schema AttributeName=contradiction_id,KeyType=HASH \
  --global-secondary-indexes '[{"IndexName":"session-index","KeySchema":[{"AttributeName":"session_id","KeyType":"HASH"}],"Projection":{"ProjectionType":"ALL"}}]' \
  --billing-mode PAY_PER_REQUEST --region us-east-1

aws dynamodb create-table --table-name research-agent-living-doc-checks \
  --attribute-definitions AttributeName=check_id,AttributeType=S \
  --key-schema AttributeName=check_id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST --region us-east-1

# Create S3 buckets
aws s3 mb s3://research-agent-reports --region us-east-1
aws s3 mb s3://research-agent-pdfs --region us-east-1
aws s3 mb s3://research-agent-exports --region us-east-1
```

---

## 📊 Report Structure

Every research run produces a structured `final_report` with these sections:

```json
{
  "title": "Research Report: ...",
  "executive_summary": "...",
  "sections": [
    {"heading": "...", "content": "...", "supporting_source_ids": ["source_1"]}
  ],
  "hypotheses_verdict": [
    {"id": "h1", "statement": "...", "verdict": "supported",
     "summary": "..."}
  ],
  "contradictions_flagged": [
    {"summary": "...", "severity": "high", "action": "..."}
  ],
  "research_gaps": ["Gap 1: ...", "Gap 2: ..."],
  "citations": [
    {"source_id": "source_1", "title": "...", "authors": "Author A, Author B",
     "year": "2023", "doi": "...", "url": "..."}
  ],
  "confidence_overall": 0.82,
  "follow_up_questions": ["What is the role of...", "How does..."]
}
```

Reports are also stored as:
- S3 JSON report at `reports/{session_id}/{slug}.json`
- S3 plain-text report at `reports/{session_id}/{slug}.txt`
- Optional local CLI exports such as `research_report_<query>_<timestamp>.json` and `.txt`

---

## 🔄 Living Document System

Research doesn't stand still. The living document system keeps your reports accurate:

```bash
# Start the background scheduler (runs every 30 days automatically)
python cli.py scheduler start

# Manually trigger a recheck right now
python cli.py scheduler check-now

# Check when the next automatic recheck is scheduled
python cli.py scheduler status
```

**What gets rechecked every 30 days:**
- 🔴 **Retraction status** — has the paper been retracted?
- 🔗 **Link health** — is the source URL still alive? (for web sources)
- 📈 **Citation count** — has the paper gained or lost citations?
- 📄 **Stored source metadata** — tracked source records are updated with any detected changes and the next scheduled check

The scheduler runs as a background thread when using the API (`main.py`). In CLI mode, start it explicitly with `python cli.py scheduler start`.

---

## 🤖 Changing the LLM

Any Ollama model works. Larger = better quality, slower:

```bash
# Fast (good for testing)
OLLAMA_MODEL=llama3.2:1b

# Balanced
OLLAMA_MODEL=llama3.1:8b

# High quality
OLLAMA_MODEL=llama3.1:70b

# Code-optimised
OLLAMA_MODEL=deepseek-r1:7b
```

Pull the model first:
```bash
ollama pull llama3.1:8b
```

---

## 📈 Observability with Langfuse

Every LLM call is traced automatically when Langfuse keys are configured:

```bash
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

This gives you:
- Per-session LLM call traces
- Token usage and cost per pipeline step
- Latency breakdowns
- Prompt/completion history

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Agent orchestration** | LangGraph `StateGraph` |
| **LLM** | Ollama (`llama3.2:1b` default) |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| **Vector DB** | Pinecone serverless |
| **Key-value store** | AWS DynamoDB |
| **Object storage** | AWS S3 |
| **Graph DB** | NetworkX (serialised to S3) |
| **Web scraping** | BeautifulSoup4 + lxml |
| **PDF parsing** | PyMuPDF (fitz) |
| **Scheduler** | APScheduler |
| **Terminal UI** | Rich |
| **Observability** | Langfuse |

---

## 📚 Documentation

| Document | Description |
|---|---|
| [`README.md`](README.md) | This file — setup, usage, quick reference |
| [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md) | Complete technical reference for every component |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | System architecture, data flow diagrams, state schema |

---

## 🗺️ Roadmap

- [ ] Streamlit web UI
- [ ] Multi-model support (OpenAI, Anthropic, Bedrock)
- [ ] Configurable pipeline (skip/add nodes)
- [ ] PDF upload support for local document research
- [ ] Graph visualisation for knowledge graph
- [ ] Collaborative sessions (multiple users, shared state)
- [ ] Docker Compose deployment

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Make your changes (do not modify `.env` or AWS credentials)
4. Run any local checks you use for validation
5. Push and open a pull request

## 📄 License

A repository license file has not been added yet.

---

<div align="center">

Built with ❤️ using LangGraph, Ollama, and Python — **EchoInquiry**

</div>
