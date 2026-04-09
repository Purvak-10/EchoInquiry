# EchoInquiry — System Architecture

> A complete technical reference for the data flow, component boundaries, inputs, and outputs of every layer in the EchoInquiry system.

---

## Table of Contents

- [EchoInquiry — System Architecture](#echoinquiry--system-architecture)
  - [Table of Contents](#table-of-contents)
  - [1. High-Level Overview](#1-high-level-overview)
  - [2. LangGraph Pipeline — Full Diagram](#2-langgraph-pipeline--full-diagram)
  - [3. ResearchState — The Shared Data Bus](#3-researchstate--the-shared-data-bus)
  - [4. Node-by-Node Data Flow](#4-node-by-node-data-flow)
    - [Concise Input → Processing → Output per Node](#concise-input--processing--output-per-node)
  - [5. External System Integrations](#5-external-system-integrations)
    - [Academic APIs — Retrieval](#academic-apis--retrieval)
    - [LLM — Ollama (Local)](#llm--ollama-local)
    - [Retraction Checker](#retraction-checker)
  - [6. Memory Layer Architecture](#6-memory-layer-architecture)
    - [Pinecone Vector Store](#pinecone-vector-store)
    - [DynamoDB Schema](#dynamodb-schema)
    - [S3 Bucket Layout](#s3-bucket-layout)
    - [NetworkX Knowledge Graph](#networkx-knowledge-graph)
  - [7. Living Document Cycle](#7-living-document-cycle)
  - [8. Request Lifecycle — CLI Path](#8-request-lifecycle--cli-path)
  - [9. Request Lifecycle — CLI Path (Primary)](#9-request-lifecycle--cli-path-primary)
  - [10. Component Dependency Map](#10-component-dependency-map)
  - [11. State Mutation Table](#11-state-mutation-table)

---

## 1. High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           ECHOINQUIRY SYSTEM                            │
│                                                                         │
│  ┌──────────┐    ┌──────────────────────────────────────────────────┐  │
│  │          │    │              LangGraph Pipeline                  │  │
│  │  User    │───▶│  query_parser → planner → hypotheses → retriever │  │
│  │ (CLI or  │    │  → scorer → hypothesis_eval → contradictions     │  │
│  │  API)    │◀───│  → synthesis → output_generator                  │  │
│  │          │    └──────────────────────────────────────────────────┘  │
│  └──────────┘                         │                                │
│                                       ▼                                │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                        PERSISTENCE LAYER                           │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │ │
│  │  │   Pinecone   │  │  DynamoDB    │  │   S3 + NetworkX Graph    │ │ │
│  │  │ Vector Store │  │  (5 tables)  │  │   (3 buckets)            │ │ │
│  │  └──────────────┘  └──────────────┘  └──────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                       │                                │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    LIVING DOCUMENT SCHEDULER                       │ │
│  │         APScheduler → SourceRecheckerEngine (every 30 days)        │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. LangGraph Pipeline — Full Diagram

```
                    ┌─────────────────────────────┐
                    │      raw_query: str          │
                    │   session_id: str            │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │       NODE 1                  │
                    │     query_parser              │
                    │  agents/query_parser.py       │
                    │                               │
                    │  IN:  raw_query               │
                    │  OUT: parsed_query            │
                    │       {intent, domain,        │
                    │        is_academic, scope,    │
                    │        core_question,         │
                    │        sub_questions,         │
                    │        keywords,              │
                    │        exclude_keywords,      │
                    │        time_range,            │
                    │        output_format}         │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │       NODE 2                  │
                    │    research_planner           │
                    │  agents/research_planner.py   │
                    │                               │
                    │  IN:  parsed_query            │
                    │  OUT: research_plan           │
                    │       {task_graph[],          │
                    │        estimated_depth,       │
                    │        search_strategy}       │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │       NODE 3                  │
                    │   hypothesis_generation       │
                    │  agents/hypothesis_engine.py  │
                    │                               │
                    │  IN:  parsed_query            │
                    │  OUT: hypotheses[]            │
                    │       {id, statement,         │
                    │        mechanism,             │
                    │        predicted_evidence,    │
                    │        falsification_criteria,│
                    │        confidence_prior}      │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │       NODE 4                  │
                    │        retriever              │
                    │    agents/retriever.py        │
                    │                               │
                    │  IN:  parsed_query            │
                    │       research_plan           │
                    │                               │
                    │  ┌────────────────────────┐   │
                    │  │ ThreadPoolExecutor     │   │
                    │  │ ┌──────────────────┐   │   │
                    │  │ │ SemanticScholar  │   │   │
                    │  │ │ PubMed           │   │   │
                    │  │ │ CrossRef         │   │   │
                    │  │ │ Web Search       │   │   │
                    │  │ └──────────────────┘   │   │
                    │  └────────────────────────┘   │
                    │   then optional Unpaywall +   │
                    │   PDF enrichment + embeddings │
                    │                               │
                    │  OUT: retrieved_sources[]     │
                    │       {title, abstract,       │
                    │        authors, year, doi,    │
                    │        url, citation_count,   │
                    │        journal, full_text_    │
                    │        snippet, s3_pdf_uri}   │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │       NODE 5                  │
                    │    credibility_scorer         │
                    │ agents/credibility_scorer.py  │
                    │                               │
                    │  IN:  retrieved_sources[]     │
                    │                               │
                    │  scoring:                     │
                    │   citation_count → 0.2–0.9   │
                    │   journal_tier   → 0.3–1.0   │
                    │   doi_present    → 0.8 base  │
                    │   retraction     → ~0.0       │
                    │                               │
                    │  OUT: retrieved_sources[]     │
                    │       + credibility_score     │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │       NODE 6                  │
                    │   hypothesis_evaluation       │
                    │  agents/hypothesis_engine.py  │
                    │                               │
                    │  IN:  hypotheses[]            │
                    │       retrieved_sources[]     │
                    │                               │
                    │  OUT: hypotheses[] updated    │
                    │       + verdict               │
                    │       + confidence_posterior  │
                    │       + supporting_evidence[] │
                    │       + opposing_evidence[]   │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │       NODE 7                  │
                    │   contradiction_detector      │
                    │ agents/contradiction_         │
                    │         detector.py           │
                    │                               │
                    │  IN:  retrieved_sources[]     │
                    │                               │
                    │  ① CLAIM EXTRACTION (LLM)    │
                    │  ② ENCODE (SentenceTransformer│
                    │  ③ COSINE SIMILARITY matrix  │
                    │  ④ CONTRADICTION ANALYSIS    │
                    │     (LLM on candidate pairs) │
                    │                               │
                    │  OUT: contradictions[]        │
                    │       {claim_a, claim_b,      │
                    │        source_a, source_b,    │
                    │        severity, explanation, │
                    │        resolution_hint}       │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │       NODE 8                  │
                    │     synthesis_engine          │
                    │  agents/synthesis_engine.py   │
                    │                               │
                    │  IN:  retrieved_sources[]     │
                    │       (top-10 by score)       │
                    │       parsed_query            │
                    │       hypotheses[]            │
                    │       contradictions[]        │
                    │                               │
                    │  OUT: synthesis{}             │
                    │       (themed sections,       │
                    │        key findings,          │
                    │        evidence assessment)   │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │       NODE 9                  │
                    │     output_generator          │
                    │  agents/output_generator.py   │
                    │                               │
                    │  IN:  synthesis{}             │
                    │       hypotheses[]            │
                    │       contradictions[]        │
                    │       retrieved_sources[]     │
                    │       parsed_query            │
                    │       research_plan           │
                    │                               │
                    │  OUT: final_report{}          │
                    │       {title,                 │
                    │        executive_summary,     │
                    │        sections[],            │
                    │        key_conclusions[],     │
                    │        hypotheses_verdict[],  │
                    │        contradictions_flagged │
                    │        research_gaps[],       │
                    │        citations[],           │
                    │        confidence_overall,    │
                    │        follow_up_questions,   │
                    │        generated_at}          │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────┴──────────────────────────────────┐
                    │              POST-PIPELINE PERSISTENCE           │
                    │                                                  │
                    │   ┌──────────────┐   ┌──────────────────────┐  │
                    │   │  DynamoDB    │   │   Pinecone           │  │
                    │   │ sessions     │   │  upsert source       │  │
                    │   │ sources      │   │  embeddings          │  │
                    │   │ hypotheses   │   └──────────────────────┘  │
                    │   │ contradictions│                             │
                    │   └──────────────┘   ┌──────────────────────┐  │
                    │                      │  S3                  │  │
                    │   ┌──────────────┐   │  reports/{session}/  │  │
                    │   │ NetworkX     │   │  {slug}.json/.txt    │  │
                    │   │ KnowledgeGraph│  │  knowledge-graph/    │  │
                    │   │ (→ S3)       │   │  graph.json          │  │
                    │   └──────────────┘   └──────────────────────┘  │
                    └─────────────────────────────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │     final_report{}            │
                    │   (returned to caller)        │
                    └──────────────────────────────┘
```

---

## 3. ResearchState — The Shared Data Bus

All 9 nodes communicate exclusively through `ResearchState`, a `TypedDict` defined in `graph/state.py`. Each node receives the full state and returns a partial dict that is merged back.

```
ResearchState
├── raw_query: str                    ← Set by caller before pipeline start
├── session_id: str                   ← Allocated before pipeline start (`sessionN`, UUID fallback)
│
├── parsed_query: Dict                ← Set by NODE 1
│   ├── intent: str
│   ├── domain: str
│   ├── scope: str
│   ├── core_question: str
│   ├── sub_questions: List[str]
│   ├── ambiguities: List[str]
│   ├── keywords: List[str]
│   ├── exclude_keywords: List[str]
│   ├── time_range: str
│   ├── output_format: str
│   └── is_academic: bool
│
├── research_plan: Dict               ← Set by NODE 2
│   ├── task_graph: List[Dict]
│   │   └── each task: {task_id, task_type, description,
│   │                    depends_on, priority, keywords, target_sources}
│   ├── estimated_depth: str
│   ├── recommended_hypothesis_count: int
│   └── search_strategy: str
│
├── hypotheses: List[Dict]            ← Set by NODE 3, updated by NODE 6
│   └── each: {id, statement, mechanism, predicted_evidence,
│               falsification_criteria, confidence_prior,
│               status, confidence_posterior,
│               supporting_evidence, opposing_evidence, verdict}
│
├── retrieved_sources: List[Dict]     ← Set by NODE 4, updated by NODE 5
│   └── each: {title, abstract, authors, year, doi, url,
│               source_api, citation_count, journal,
│               full_text_snippet, s3_pdf_uri, credibility_score}
│
├── contradictions: List[Dict]        ← Set by NODE 7
│   └── each: {claim_a, claim_b, source_a, source_b,
│               severity, explanation, resolution_hint}
│
├── synthesis: Dict                   ← Set by NODE 8
│
├── final_report: Dict                ← Set by NODE 9
│   ├── title: str
│   ├── executive_summary: str
│   ├── sections: List[Dict]
│   ├── key_conclusions: List[str]
│   ├── hypotheses_verdict: List[Dict]
│   ├── contradictions_flagged: List[Dict]
│   ├── research_gaps: List[str]
│   ├── citations: List[Dict]
│   ├── confidence_overall: float
│   ├── follow_up_questions: List[str]
│   └── generated_at: str
│
├── living_doc_id: Optional[str]      ← Set by NODE 9 (report-linked UUID)
├── source_registry_entries: List[Dict] ← Reserved for registry metadata
├── s3_report_uri: Optional[str]      ← Set by NODE 9 (S3 report path)
├── error_log: List[str]              ← Reserved for runtime errors
├── current_step: str                 ← Current pipeline step
├── iteration_count: int              ← Reserved for iterative workflows
└── fast_mode: bool                   ← Disables slower persistence/enrichment steps
```

---

## 4. Node-by-Node Data Flow

### Concise Input → Processing → Output per Node

| Node | Key Input Fields | Core Processing | Key Output Fields |
|---|---|---|---|
| `query_parser` | `raw_query` | LLM prompt → JSON parse | `parsed_query` |
| `research_planner` | `parsed_query` | LLM prompt → validate task structure → priority sort | `research_plan` |
| `hypothesis_generation` | `parsed_query` | LLM prompt → default to 3 hypotheses when count absent → enrich with placeholders | `hypotheses[]` |
| `retriever` | `parsed_query`, `research_plan` | ThreadPoolExecutor → 4 parallel searches → DOI/title dedup → optional PDF/enrichment | `retrieved_sources[]` |
| `credibility_scorer` | `retrieved_sources[]` | citation log-scale + journal tier + retraction check | `retrieved_sources[]` + `.credibility_score` |
| `hypothesis_evaluation` | `hypotheses[]`, `retrieved_sources[]` | keyword match → LLM verdict per hypothesis | `hypotheses[]` + verdict fields |
| `contradiction_detector` | `retrieved_sources[]` | bounded claim extract → ST encode → candidate pruning → capped LLM checks | `contradictions[]` |
| `synthesis_engine` | top-10 sources, all context | keyword filter → LLM synthesis | `synthesis{}` |
| `output_generator` | all state | LLM call → quality guard → DynamoDB session update → S3 report save | `final_report{}` |

---

## 5. External System Integrations

### Academic APIs — Retrieval

```
retriever (Node 4)
       │
       ├──▶ SemanticScholar API
       │      URL: api.semanticscholar.org/graph/v1/paper/search
       │      Query: keywords
       │      Returns: papers with citation_count, abstract, DOI, authors
       │      Retry: backoff (3 attempts, 1–2s delays)
       │
       ├──▶ PubMed / NCBI E-utilities
       │      URL: eutils.ncbi.nlm.nih.gov
       │      Steps: esearch → efetch (XML)
       │      Returns: biomedical literature, MeSH terms
       │      Retry: backoff (3 attempts)
       │
       ├──▶ CrossRef API
       │      URL: api.crossref.org/works
       │      Query: keywords, free-text
       │      Returns: DOI metadata, journal, publisher, year
       │      Retry: backoff (3 attempts)
       │
       ├──▶ Web Scraper
       │      Library: requests + BeautifulSoup4 + lxml
       │      Input: search queries via URL construction
       │      Returns: title, body text, URL
       │
       ├──▶ Unpaywall API (post-retrieval enrichment)
       │      URL: api.unpaywall.org/v2/{doi}
       │      Input: DOIs from retrieved academic results
       │      Returns: open-access PDF URL if available
       │      Retry: backoff (3 attempts)
       │
       └──▶ PDF Parser
              Library: PyMuPDF (fitz)
              Input: PDF URLs (from Unpaywall or web)
              Returns: extracted text blocks
```

### LLM — Ollama (Local)

```
All agents
    │
    └──▶ Ollama Server (localhost:11434)
           Model: llama3.2:1b (default, configurable)
           Interface: langchain-ollama ChatOllama
           Retry wrapper: llm_call_with_retry() (utils/llm_helpers.py)
           Streaming: llm_stream() for output_generator
           Observability: Langfuse traces every call
           System name: EchoInquiry
```

### Retraction Checker

```
credibility_scorer (Node 5)
    │
    └──▶ tools/retraction_checker.py
               │
               └──▶ CrossRef API /works/{doi}
                      Checks: retraction notices in metadata
                      Result: is_retracted: bool
                      Effect: credibility_score → ~0.0
```

---

## 6. Memory Layer Architecture

### Pinecone Vector Store

```
                   ┌─────────────────────────────────────────┐
                   │              PINECONE INDEX              │
                   │     Name: research-passages (default)    │
                   │     Dimensions: 384                       │
                   │     Metric: cosine                        │
                   │     Type: serverless (us-east-1)         │
                   └─────────────────┬───────────────────────┘
                                     │
                   ┌─────────────────┴───────────────────────┐
                   │           memory/vector_store.py         │
                   │                                         │
                   │  WRITE (after pipeline):                 │
                   │    source abstract                       │
                   │      → SentenceTransformer encode        │
                   │      → 384-dim vector                    │
                   │      → upsert with metadata:             │
                   │          {session_id, title, doi,        │
                   │           year, credibility_score, text} │
                   │                                         │
                   │  READ (for similarity search):           │
                   │    query string                          │
                   │      → SentenceTransformer encode        │
                   │      → top-k nearest neighbours         │
                   │      → return source metadata           │
                   └─────────────────────────────────────────┘
```

### DynamoDB Schema

```
research-agent-sessions
  PK: session_id (str)
  Attributes: raw_query, created_at, status,
              final_report_json, s3_report_uri

research-agent-sources
  PK: source_id (str)
  GSI: session-index on session_id
  Attributes: title, abstract, authors, year, doi,
              url, credibility_score, source_api,
              citation_count, journal, next_check_at, content_hash

research-agent-hypotheses
  PK: hypothesis_id (str)
  GSI: session-index on session_id
  Attributes: statement, verdict, confidence_posterior,
              supporting_evidence, opposing_evidence

research-agent-contradictions
  PK: contradiction_id (str)
  GSI: session-index on session_id
  Attributes: claim_a, claim_b, source_a, source_b,
              severity, explanation, resolution_hint

research-agent-living-doc-checks
  PK: check_id (str)
  Attributes: source_id, checked_at, alert_sent, change summary
```

### S3 Bucket Layout

```
research-agent-reports/
  └── reports/{session_id}/
      ├── {slug}.json        ← Full final_report dict
      └── {slug}.txt         ← Plain-text version

research-agent-pdfs/
  └── pdfs/{session_id}/
      └── {session_prefix}_{doi}.pdf  ← Downloaded PDFs when available

research-agent-exports/
  └── knowledge-graph/graph.json      ← Serialised NetworkX DiGraph
```

### NetworkX Knowledge Graph

```
NODE TYPES:
  concept  — extracted from source abstracts (words > 5 chars, not stopwords)
  source   — a retrieved paper (source_id / doi / title-derived node ID)
  claim    — a factual statement extracted by contradiction_detector
  author   — paper author name
  session  — a research session

EDGE TYPES:
  supports      — concept/claim supports hypothesis
  contradicts   — claim contradicts another claim
  from_source   — claim/concept came from this source
  authored_by   — source was authored by person
  cites         — source cites another source
  related_to    — concept related to another concept
  appears_in    — concept appears in session

PERSISTENCE:
  Load: S3 → JSON → nx.node_link_graph()
  Save: nx.node_link_data() → JSON → S3
```

---

## 7. Living Document Cycle

```
                 INITIAL RESEARCH RUN
                         │
                         ▼
            post-pipeline source registry saves
            cited report sources to DynamoDB
            research-agent-sources with next_check_at = now + 30d
                         │
                         │
        ─────────────────────────────────────────
        │            30 DAYS LATER               │
        │                                        │
        ▼                                        │
  APScheduler fires                              │
  LivingDocumentScheduler._check_living_documents│
        │                                        │
        ▼                                        │
  SourceRecheckerEngine.check_all_sources()      │
        │                                        │
        ├──▶ For each source due for recheck:    │
        │         │                              │
        │         ├── Retraction check           │
        │         │   (CrossRef API)             │
        │         │                              │
        │         ├── Dead link check            │
        │         │   (HTTP HEAD request)        │
        │         │                              │
        │         ├── Citation count update      │
        │         │   (CrossRef, then Semantic   │
        │         │    Scholar fallback)         │
        │         │                              │
        │         └── Source record update       │
        │             (next_check_at, metadata)  │
        │                                        │
        ▼                                        │
  Update DynamoDB research-agent-sources         │
  (new status, new next_check_at = +30d)         │
        │                                        │
        └────────────────────────────────────────┘
                   CYCLE REPEATS
```

---

## 8. Request Lifecycle — CLI Path

```
$ python cli.py "impact of sleep on memory consolidation"
         │
         ▼
  cli.py loads .env, configures logging
         │
         ▼
  Allocates session_id (usually sessionN, UUID fallback)
         │
         ▼
  Builds initial ResearchState:
    {raw_query: "impact of sleep...",
     session_id: "abc-123",
     ...all other fields empty}
         │
         ▼
  Calls stream_research(query)
    → research_graph.py allocates state and runs graph.stream(...)
    → LangGraph executes nodes 1–9 sequentially
    → CLI prints each node result as it completes
         │
         ▼
  Receives final_report from state
         │
         ├──▶ Display: Executive Summary panel
         ├──▶ Display: Research Sections
         ├──▶ Display: Hypotheses table (Rich)
         ├──▶ Display: Contradictions table (colour by severity)
         ├──▶ Display: Citations table
         ├──▶ Display: Confidence score
         └──▶ Display: Follow-up recommendations
                  │
                  ▼
           Prompt: "Send report so far by email? (y/n)"
                  │
                  ▼
           Optional local save (.json + .txt)
                  │
                  ▼
           Optional follow-up Q&A loop
           FollowupAgent(session_id, report_data)
             → LLM answers each question grounded in report
             → Loop until user types "exit"
```

---

## 9. Request Lifecycle — CLI Path (Primary)

The CLI is the primary and fully implemented entry point:

```
$ python cli.py "your research question"
         │
         ▼
  cli.py loads .env and pipeline modules
         │
         ▼
  Allocates session_id (usually sessionN, UUID fallback)
         │
         ▼
  Builds ResearchState:
    {raw_query: "...", session_id: "...", ...}
         │
         ▼
  Calls stream_research(query)
    → research_graph.py allocates state and runs graph.stream(...)
    → LangGraph executes 9 nodes sequentially
    → CLI streams each node result to terminal
         │
         ▼
  Receives final_report from state
         │
         ├──▶ Display: Executive Summary, Research Sections, Hypotheses
         ├──▶ Display: Contradictions (colour-coded), Citations
         ├──▶ Display: Confidence score, Follow-up recommendations
         │
         ├──▶ Optional: Email report via SMTP/SendGrid
         ├──▶ Optional: Save JSON/TXT locally
         │
         └──▶ Optional: Enter Follow-up Q&A loop
             FollowupAgent(session_id, report_data)
             → Grounded answers from report context
```

**Note**: The REST API (`main.py`) currently provides only health checks and scheduler management. For full research pipeline execution, use the CLI.

---

## 10. Component Dependency Map

```
cli.py / main.py
    │
    └──▶ graph/research_graph.py
              │
              ├──▶ agents/query_parser.py
              │         └──▶ prompts/query_parser_prompt.py
              │         └──▶ utils/llm_helpers.py
              │
              ├──▶ agents/research_planner.py
              │         └──▶ prompts/planner_prompt.py
              │         └──▶ utils/llm_helpers.py
              │
              ├──▶ agents/hypothesis_engine.py
              │         └──▶ prompts/hypothesis_prompt.py
              │         └──▶ utils/llm_helpers.py
              │
              ├──▶ agents/retriever.py
              │         ├──▶ tools/academic_apis.py
              │         │         └──▶ SemanticScholarAPI
              │         │         └──▶ PubMedAPI
              │         │         └──▶ CrossrefAPI
              │         │         └──▶ UnpaywallAPI
              │         ├──▶ tools/web_scraper.py
              │         └──▶ tools/pdf_parser.py
              │
              ├──▶ agents/credibility_scorer.py
              │         └──▶ tools/retraction_checker.py
              │
              ├──▶ agents/contradiction_detector.py
              │         ├──▶ sentence_transformers
              │         ├──▶ prompts/contradiction_prompt.py
              │         ├──▶ utils/llm_helpers.py
              │         └──▶ aws/dynamodb_client.py
              │
              ├──▶ agents/synthesis_engine.py
              │         ├──▶ prompts/synthesis_prompt.py
              │         └──▶ utils/llm_helpers.py
              │
              ├──▶ agents/output_generator.py
              │         ├──▶ prompts/output_prompt.py
              │         ├──▶ utils/llm_helpers.py
              │         ├──▶ aws/dynamodb_client.py
              │         └──▶ aws/s3_client.py
              │
              └──▶ POST-PIPELINE (in research_graph.py):
                        ├──▶ memory/source_registry.py
                        │         └──▶ aws/dynamodb_client.py
                        ├──▶ memory/vector_store.py
                        │         └──▶ pinecone
                        │         └──▶ sentence_transformers
                        └──▶ memory/knowledge_graph.py
                                  └──▶ networkx
                                  └──▶ aws/s3_client.py
```

---

## 11. State Mutation Table

Which nodes read and write which state fields:

| State Field | Written by | Read by |
|---|---|---|
| `raw_query` | Caller | `query_parser`, `hypothesis_engine`, `output_generator` |
| `session_id` | Caller | All nodes (for logging) |
| `parsed_query` | `query_parser` (Node 1) | `research_planner`, `hypothesis_engine`, `retriever`, `synthesis_engine`, `output_generator` |
| `research_plan` | `research_planner` (Node 2) | `retriever`, `output_generator` |
| `hypotheses` | `hypothesis_engine` (Node 3) | `hypothesis_evaluation`, `synthesis_engine`, `output_generator` |
| `retrieved_sources` | `retriever` (Node 4) | `credibility_scorer`, `hypothesis_evaluation`, `contradiction_detector`, `synthesis_engine`, `output_generator` |
| `credibility_score` (on sources) | `credibility_scorer` (Node 5) | `hypothesis_evaluation`, `synthesis_engine`, `output_generator` |
| `hypotheses` (updated) | `hypothesis_evaluation` (Node 6) | `synthesis_engine`, `output_generator` |
| `contradictions` | `contradiction_detector` (Node 7) | `synthesis_engine`, `output_generator` |
| `synthesis` | `synthesis_engine` (Node 8) | `output_generator` |
| `final_report` | `output_generator` (Node 9) | Caller, CLI display |
| `living_doc_id` | `output_generator` (Node 9) | Caller / downstream integrations |
| `s3_report_uri` | `output_generator` (Node 9) | Caller / downstream integrations |
| `error_log` | Reserved runtime field | Caller / debugging tools |
