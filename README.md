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
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

</div>

---

## 📖 What Is This?

**EchoInquiry** is a fully autonomous, hypothesis-driven research system. Give it any research question and it:

1. **Understands** your query — parses intent, domain, sub-questions, and keywords
2. **Plans** a search strategy — generates a prioritised task graph
3. **Hypothesises** — creates falsifiable scientific hypotheses *before* searching
4. **Retrieves** — queries Semantic Scholar, PubMed, CrossRef, Unpaywall, and the web simultaneously
5. **Scores credibility** — citation counts, journal tier, retraction status
6. **Evaluates hypotheses** — tests each hypothesis against retrieved evidence
7. **Detects contradictions** — finds conflicting claims using sentence embeddings + LLM
8. **Synthesises** — structured narrative from the top 10 credible sources
9. **Reports** — structured, publication-quality research report
10. **Remembers** — every source is tracked and rechecked every 30 days via a background scheduler

All powered by a local LLM (Ollama) — **no OpenAI API key required**.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 **Hypothesis-Driven Research** | Generates falsifiable hypotheses before searching, then evaluates them against evidence |
| 🔍 **Multi-Source Retrieval** | Queries 6 sources in parallel: SemanticScholar, PubMed, CrossRef, Unpaywall, web, PDFs |
| ⭐ **Credibility Scoring** | Each source scored 0–1 based on citations, journal tier, and retraction status |
| ⚡ **Contradiction Detection** | Semantic embedding + LLM identifies conflicting claims between sources |
| 📊 **Structured Reports** | Executive summary, research sections, hypotheses verdict, citations, confidence score |
| 🔄 **Living Documents** | Background scheduler rechecks all sources every 30 days |
| 💬 **Follow-up Q&A** | Chat with your report — grounded follow-up answers |
| 📧 **Email Delivery** | Send reports via SMTP or SendGrid |
| 🖥️ **Rich Terminal UI** | Colour-coded live pipeline stream, tables, and panels |
| 🌐 **REST API** | FastAPI server for programmatic access |
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
git clone https://github.com/your-username/EchoInquiry.git
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
  ✅ [7/9] Contradiction Det   → 3 contradictions found
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

### Endpoints

```
POST /research
  Body: {"query": "your research question", "fast_mode": false}
  Returns: {"session_id": "...", "final_report": {...}}

GET /session/{session_id}
  Returns: stored session data and final report from DynamoDB

GET /health
  Returns: {"status": "ok", "scheduler": "running"}
```

### Example

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"query": "CRISPR applications in treating genetic diseases"}'
```

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
├── agents/                   ← One agent per pipeline node
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
├── prompts/                  ← LLM prompts (one per agent)
├── memory/                   ← Pinecone, DynamoDB registry, NetworkX graph
├── tools/                    ← Academic APIs, web scraper, PDF parser, retraction checker
├── aws/                      ← DynamoDB, S3, CloudWatch clients
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
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:1b          # or llama3.1:8b, mistral, etc.
OLLAMA_TEMPERATURE=0.7
OLLAMA_TIMEOUT=120

# ── AWS ─────────────────────────────────────────────────────────
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_key_here
AWS_SECRET_ACCESS_KEY=your_secret_here

# ── DynamoDB Tables ─────────────────────────────────────────────
DYNAMODB_SESSIONS_TABLE=research-sessions
DYNAMODB_SOURCES_TABLE=research-sources
DYNAMODB_HYPOTHESES_TABLE=research-hypotheses
DYNAMODB_CONTRADICTIONS_TABLE=research-contradictions
DYNAMODB_LIVING_DOCS_TABLE=research-living-docs

# ── S3 Buckets ──────────────────────────────────────────────────
S3_REPORTS_BUCKET=research-agent-reports
S3_SOURCES_BUCKET=research-agent-sources
S3_KNOWLEDGE_GRAPHS_BUCKET=research-knowledge-graphs

# ── Pinecone ────────────────────────────────────────────────────
PINECONE_API_KEY=your_pinecone_key_here
PINECONE_INDEX_NAME=research-agent

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
aws dynamodb create-table --table-name research-sessions \
  --attribute-definitions AttributeName=session_id,AttributeType=S \
  --key-schema AttributeName=session_id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST --region us-east-1

aws dynamodb create-table --table-name research-sources \
  --attribute-definitions AttributeName=source_id,AttributeType=S \
  --key-schema AttributeName=source_id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST --region us-east-1

aws dynamodb create-table --table-name research-hypotheses \
  --attribute-definitions AttributeName=hypothesis_id,AttributeType=S \
  --key-schema AttributeName=hypothesis_id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST --region us-east-1

aws dynamodb create-table --table-name research-contradictions \
  --attribute-definitions AttributeName=contradiction_id,AttributeType=S \
  --key-schema AttributeName=contradiction_id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST --region us-east-1

aws dynamodb create-table --table-name research-living-docs \
  --attribute-definitions AttributeName=source_id,AttributeType=S \
  --key-schema AttributeName=source_id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST --region us-east-1

# Create S3 buckets
aws s3 mb s3://research-agent-reports --region us-east-1
aws s3 mb s3://research-agent-sources --region us-east-1
aws s3 mb s3://research-knowledge-graphs --region us-east-1
```

---

## 🧪 Running Tests

```bash
pytest
```

Uses `moto` for AWS service mocking — no real AWS calls during tests.

---

## 📊 Report Structure

Every research run produces a structured `final_report` with these sections:

```json
{
  "title": "Research Report: ...",
  "executive_summary": "...",
  "research_sections": [
    {"heading": "...", "findings": "...", "evidence_strength": "high"}
  ],
  "hypotheses_verdict": [
    {"id": "h1", "statement": "...", "verdict": "supported",
     "confidence_posterior": 0.87, "supporting_evidence": [...]}
  ],
  "contradictions_flagged": [
    {"claim_a": "...", "claim_b": "...", "severity": "high",
     "explanation": "..."}
  ],
  "research_gaps": ["Gap 1: ...", "Gap 2: ..."],
  "citations": [
    {"title": "...", "authors": [...], "year": 2023,
     "doi": "...", "credibility_score": 0.92}
  ],
  "confidence_score": 0.82,
  "followup_recommendations": ["What is the role of...", "How does..."]
}
```

Reports are also saved as:
- `report.json` — full machine-readable report
- `report.txt` — plain-text version for readability
- Uploaded to S3 under `{session_id}/`

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
- 🔗 **Link health** — is the source URL still alive?
- 📈 **Citation count** — has the paper gained or lost citations?
- 🔓 **Open access status** — is a full-text PDF now available?
- 📄 **Content integrity** — has the abstract been modified?

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
| **REST API** | FastAPI + Uvicorn |
| **Scheduler** | APScheduler |
| **Terminal UI** | Rich |
| **Observability** | Langfuse + AWS CloudWatch |
| **Testing** | pytest + moto |

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
4. Run tests: `pytest`
5. Push and open a pull request

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

Built with ❤️ using LangGraph, Ollama, and Python — **EchoInquiry**

</div>
