import logging
import time
import urllib.parse
import urllib.request
import re
from concurrent.futures import ThreadPoolExecutor, wait
from typing import List, Dict
from urllib.parse import urlparse

from sentence_transformers import SentenceTransformer

import config

from tools.academic_apis import (
    SemanticScholarAPI,
    PubMedAPI,
    CrossrefAPI,
    UnpaywallAPI,
)
from tools.web_scraper import WebScraper
from tools.pdf_parser import PDFParser
from aws.s3_client import S3Client
from utils.backend_logging import log_backend_event


logger = logging.getLogger(__name__)
_embedder = None

# Domains that warrant hitting academic APIs
_ACADEMIC_DOMAINS = {
    "science", "medicine", "biology", "physics", "chemistry",
    "engineering", "psychology", "history", "economics", "law",
    "nutrition", "healthcare", "artificial intelligence healthcare",
}


# -------------------------------
# Helper: Deduplication
# -------------------------------
def deduplicate_sources(sources: List[Dict]) -> List[Dict]:
    doi_map = {}
    final_sources = []

    for src in sources:
        doi = src.get("doi")
        title = (src.get("title") or "").lower()
        abstract = src.get("abstract") or ""

        # ---- PRIMARY: DOI dedup ----
        if doi:
            if doi in doi_map:
                if len(abstract) > len(doi_map[doi].get("abstract", "")):
                    doi_map[doi] = src
            else:
                doi_map[doi] = src
            continue

        # ---- SECONDARY: title overlap ----
        is_duplicate = False
        title_words = set(title.split())

        for existing in final_sources:
            existing_words = set((existing.get("title") or "").lower().split())
            if not existing_words:
                continue

            overlap = len(title_words & existing_words) / max(len(title_words), 1)

            if overlap > 0.8:
                is_duplicate = True
                break

        if not is_duplicate:
            final_sources.append(src)

    final_sources.extend(doi_map.values())
    return final_sources


# -------------------------------
# Worker Functions
# -------------------------------
def semantic_scholar_worker(keywords):
    """Search using a combined query (single call) then fall back per-keyword if empty."""
    api = SemanticScholarAPI()
    combined = " ".join(keywords[:3])  # top-3 keywords → one focused query
    try:
        results = api.search(combined, limit=10)
        if results:
            return results
    except Exception as e:
        logger.warning(f"SemanticScholar combined search failed: {e}")

    # Fallback: try first keyword individually
    if keywords:
        try:
            return api.search(keywords[0], limit=8)
        except Exception as e:
            logger.warning(f"SemanticScholar fallback failed: {e}")
    return []


def pubmed_worker(keywords):
    """Search using a combined query (single call) then fall back per-keyword if empty."""
    api = PubMedAPI()
    combined = " ".join(keywords[:3])
    try:
        results = api.search(combined, limit=8)
        if results:
            return results
    except Exception as e:
        logger.warning(f"PubMed combined search failed: {e}")

    if keywords:
        try:
            return api.search(keywords[0], limit=6)
        except Exception as e:
            logger.warning(f"PubMed fallback failed: {e}")
    return []


def crossref_worker(keywords):
    """Search using a combined query (single call) then fall back per-keyword if empty."""
    api = CrossrefAPI()
    combined = " ".join(keywords[:3])
    try:
        results = api.search(combined, limit=8)
        if results:
            return results
    except Exception as e:
        logger.warning(f"Crossref combined search failed: {e}")

    if keywords:
        try:
            return api.search(keywords[0], limit=5)
        except Exception as e:
            logger.warning(f"Crossref fallback failed: {e}")
    return []


def web_worker(core_question, keywords=None, exclude_keywords=None):
    """
    Search DuckDuckGo HTML for the query, then scrape top 3 result pages (reduced from 5).
    Falls back gracefully if network is unavailable.
    """
    if keywords is None:
        keywords = []
    if exclude_keywords is None:
        exclude_keywords = []
    
    if not core_question or not core_question.strip():
        return []

    results = []

    try:
        # ---- Step 1: DuckDuckGo HTML search ----
        encoded_q = urllib.parse.quote_plus(core_question.strip())
        ddg_url = f"https://html.duckduckgo.com/html/?q={encoded_q}"

        req = urllib.request.Request(
            ddg_url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchAgent/2.0)"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")

        # ---- Step 2: Extract result URLs and snippets from DDG HTML ----
        # DDG HTML results have links like: <a class="result__url" href="...">
        url_pattern = re.compile(
            r'class="result__a"[^>]*href="([^"]+)"[^>]*>([^<]+)</a>',
            re.IGNORECASE,
        )
        snippet_pattern = re.compile(
            r'class="result__snippet"[^>]*>(.*?)</a>',
            re.IGNORECASE | re.DOTALL,
        )

        urls = url_pattern.findall(html)
        snippets = snippet_pattern.findall(html)

        scraper = WebScraper()

        for i, (url, title) in enumerate(urls[:3]):  # top 3 results (REDUCED from 5)
            # DDG sometimes wraps URLs — unwrap if needed
            if url.startswith("//duckduckgo.com/l/?"):
                parsed = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
                url = parsed.get("uddg", [url])[0]

            snippet_text = ""
            if i < len(snippets):
                snippet_text = re.sub(r"<[^>]+>", " ", snippets[i]).strip()

            try:
                fetched = scraper.fetch(url)
                page_text = ""
                if isinstance(fetched, dict) and not fetched.get("error") and fetched.get("text"):
                    page_text = fetched["text"][:3000]
            except Exception:
                page_text = snippet_text

            if not page_text and not snippet_text:
                continue
            
            # Early check: skip if excluded keywords appear in title/snippet
            combined_text = (title + " " + snippet_text).lower()
            if any(excl.lower() in combined_text for excl in exclude_keywords):
                continue

            results.append({
                "source_id": url,
                "title": title.strip() or url,
                "authors": "",
                "year": 0,
                "doi": None,
                "url": url,
                "abstract": snippet_text[:500],
                "full_text_snippet": page_text or snippet_text,
                "journal": None,
                "citation_count": 0,
                "credibility_score": 0.35,
                "retraction_status": "unknown",
                "content_hash": "",
                "content_embedding": [],
                "content_xpath": "//body",
                "source_type": "web",
                "s3_pdf_uri": "",
            })

    except Exception as e:
        logger.warning(f"Web search failed: {e}")

    return results


# -------------------------------
# PDF Fetch + S3 Storage
# -------------------------------
def enrich_with_pdfs(
    sources: List[Dict],
    *,
    session_id: str | None = None,
    query_hint: str | None = None,
) -> List[Dict]:
    scraper = WebScraper()
    unpaywall = UnpaywallAPI()
    parser = PDFParser()
    s3 = S3Client()

    if session_id:
        log_backend_event(
            "retriever_pdf_enrichment_started",
            session_id=session_id,
            source_count=len(sources),
        )

    stage_started = time.monotonic()
    processed = 0
    timed_out = False

    for src in sources:
        if processed >= config.RETRIEVER_MAX_PDF_ITEMS:
            break
        if (time.monotonic() - stage_started) > config.RETRIEVER_PDF_ENRICH_MAX_SECONDS:
            timed_out = True
            break

        processed += 1
        doi = src.get("doi")
        if not doi:
            continue

        try:
            # Skip if already exists
            if s3.pdf_exists(doi, session_id=session_id):
                continue

            url = unpaywall.get_open_access_url(doi)
            if not url:
                continue

            pdf_bytes = scraper.download_pdf_bytes(url)
            if not pdf_bytes:
                continue

            parsed = parser.parse(pdf_bytes)
            text = parsed.get("abstract", "")

            s3_uri = s3.save_pdf(
                pdf_bytes,
                doi,
                session_id=session_id,
                query_hint=query_hint,
                source_hint=src.get("title"),
            )

            src["full_text_snippet"] = text[:1000]
            src["s3_pdf_uri"] = s3_uri

        except Exception as e:
            logger.warning(f"PDF pipeline failed for DOI {doi}: {e}")
            if session_id:
                log_backend_event(
                    "retriever_pdf_enrichment_item_failed",
                    session_id=session_id,
                    doi=doi,
                    error=str(e),
                )

    if session_id:
        log_backend_event(
            "retriever_pdf_enrichment_completed",
            session_id=session_id,
            processed_count=processed,
            timed_out=timed_out,
        )
    return sources


# -------------------------------
# Relevance Filter
# -------------------------------
def _is_relevant(source: Dict, keywords: list, exclude_keywords: list = None) -> bool:
    """
    Drop sources with insufficient overlap against the query keywords.
    Also exclude sources matching exclude_keywords.

    Strategy:
    - First, REJECT if any exclude_keyword appears in title/abstract
    - Web sources (DDG snippets, credibility <= 0.4): split multi-word keyword
      phrases into individual words; require at least 1 individual word hit.
      (Snippet titles are too short to match full phrases.)
    - Academic sources: require that at least one keyword phrase has ≥2 of its
      component words appear in the text (or all words, if the phrase is short).
      This accepts "Norse Vikings sailed to America" for the phrase "norse
      vikings america" (2/3 words hit), while rejecting a diabetes paper that
      only shares "indigenous" and "america" from different unrelated phrases.
    """
    if exclude_keywords is None:
        exclude_keywords = []
    
    if not keywords:
        return True

    text = (
        (source.get("title") or "") + " " + (source.get("abstract") or "")
    ).lower()

    # EXCLUDE: Reject if any exclude_keyword appears in text
    for excl_kw in exclude_keywords:
        if excl_kw.lower() in text:
            return False

    meaningful_kw = [k.lower() for k in keywords if len(k) > 3]
    if not meaningful_kw:
        return True

    is_web = source.get("source_type") == "web" or (source.get("credibility_score", 1.0) <= 0.4)

    if is_web:
        # For web sources, split multi-word phrases into individual words
        # and require at least 1 individual word match
        all_words = set()
        for kw in meaningful_kw:
            for word in kw.split():
                if len(word) > 3:
                    all_words.add(word)
        if not all_words:
            return True
        return any(word in text for word in all_words)

    # Academic sources: for each keyword phrase, count how many of that
    # phrase's words appear in the text. If ANY single phrase has ≥2 of
    # its component words present (or all words for single-word phrases),
    # the paper is relevant.
    for kw in meaningful_kw:
        phrase_words = [w for w in kw.split() if len(w) > 3]
        if not phrase_words:
            continue
        hits = sum(1 for w in phrase_words if w in text)
        if len(phrase_words) == 1 and hits == 1:
            return True
        if len(phrase_words) >= 2 and hits >= 2:
            return True

    return False


# -------------------------------
# Embedding + Hashing
# -------------------------------
def enrich_with_embeddings(sources: List[Dict], *, session_id: str | None = None) -> List[Dict]:
    global _embedder
    if _embedder is None:
        if session_id:
            log_backend_event(
                "retriever_embedding_model_loading",
                session_id=session_id,
                model_name=config.EMBEDDING_MODEL_NAME,
            )
        _embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    embedder = _embedder
    scraper = WebScraper()

    if session_id:
        log_backend_event(
            "retriever_embedding_started",
            session_id=session_id,
            source_count=len(sources),
        )

    stage_started = time.monotonic()
    processed = 0
    timed_out = False

    for src in sources:
        if processed >= config.RETRIEVER_MAX_EMBED_ITEMS:
            break
        if (time.monotonic() - stage_started) > config.RETRIEVER_EMBEDDING_MAX_SECONDS:
            timed_out = True
            break

        processed += 1
        try:
            abstract = src.get("abstract", "")
            snippet = src.get("full_text_snippet", "")

            text = f"{abstract} {snippet}".strip()

            src["content_hash"] = scraper.compute_content_hash(text)
            src["content_embedding"] = embedder.encode(text[:1000]).tolist()

        except Exception as e:
            logger.warning(f"Embedding failed: {e}")

    if session_id:
        log_backend_event(
            "retriever_embedding_completed",
            session_id=session_id,
            processed_count=processed,
            timed_out=timed_out,
        )
    return sources


# -------------------------------
# Main Node
# -------------------------------
def retriever_node(state: Dict) -> Dict:
    """
    Multi-Corpus Retriever Node
    """

    research_plan = state.get("research_plan", {})
    parsed_query = state.get("parsed_query", {})
    fast_mode = bool(state.get("fast_mode"))

    keywords = parsed_query.get("keywords", [])
    core_question = parsed_query.get("core_question", "")
    exclude_keywords = parsed_query.get("exclude_keywords", [])
    session_id = state.get("session_id", "")

    log_backend_event(
        "retriever_started",
        session_id=session_id,
        fast_mode=fast_mode,
        keyword_count=len(keywords) if isinstance(keywords, list) else 0,
    )

    # -------------------------------
    # STEP 2: Domain-aware routing
    # -------------------------------
    domain = parsed_query.get("domain", "general").lower()
    is_academic = parsed_query.get("is_academic", False)
    # Also check domain name as fallback in case LLM omits is_academic
    if not is_academic:
        is_academic = any(d in domain for d in _ACADEMIC_DOMAINS)

    logger.info(f"Domain: {domain!r} | is_academic: {is_academic}")
    log_backend_event(
        "retriever_domain_routing",
        session_id=session_id,
        domain=domain,
        is_academic=is_academic,
    )

    all_results = []

    # -------------------------------
    # STEP 3: Parallel Retrieval
    # -------------------------------
    executor = ThreadPoolExecutor(max_workers=4)
    try:
        futures = {}

        # Only hit academic APIs for academic queries
        if is_academic:
            futures[executor.submit(semantic_scholar_worker, keywords)] = "semantic"
            futures[executor.submit(pubmed_worker, keywords)] = "pubmed"
            futures[executor.submit(crossref_worker, keywords)] = "crossref"

        # Web search always runs (DuckDuckGo) — now with exclude_keywords
        futures[executor.submit(web_worker, core_question, keywords, exclude_keywords)] = "web"

        done, not_done = wait(
            set(futures.keys()),
            timeout=max(0.1, config.RETRIEVER_PARALLEL_TIMEOUT_SECONDS),
        )

        for future in done:
            source_name = futures[future]
            try:
                results = future.result()
                if isinstance(results, dict):
                    results = [results]
                if not isinstance(results, list):
                    logger.warning(f"{source_name} worker returned non-list payload; ignoring")
                    results = []
                all_results.extend(results)
                log_backend_event(
                    "retriever_corpus_completed",
                    session_id=session_id,
                    corpus=source_name,
                    result_count=len(results),
                )
            except Exception as e:
                logger.warning(f"{source_name} worker failed: {e}")
                log_backend_event(
                    "retriever_corpus_failed",
                    session_id=session_id,
                    corpus=source_name,
                    error=str(e),
                )
        for future in not_done:
            source_name = futures[future]
            future.cancel()
            log_backend_event(
                "retriever_corpus_timeout",
                session_id=session_id,
                corpus=source_name,
                timeout_seconds=config.RETRIEVER_PARALLEL_TIMEOUT_SECONDS,
            )
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    logger.info(f"Total raw sources: {len(all_results)}")
    log_backend_event(
        "retriever_raw_completed",
        session_id=session_id,
        raw_source_count=len(all_results),
    )

    # -------------------------------
    # STEP 4: Relevance filter
    # -------------------------------
    relevant = [s for s in all_results if _is_relevant(s, keywords, exclude_keywords)]
    dropped = len(all_results) - len(relevant)
    if dropped:
        logger.info(f"Relevance filter dropped {dropped} irrelevant sources")
        log_backend_event(
            "retriever_relevance_filter",
            session_id=session_id,
            dropped=dropped,
            kept=len(relevant),
        )
    all_results = relevant

    # -------------------------------
    # STEP 5: Deduplicate
    # -------------------------------
    deduped = deduplicate_sources(all_results)
    logger.info(f"After deduplication: {len(deduped)}")
    log_backend_event(
        "retriever_dedup_completed",
        session_id=session_id,
        deduped_source_count=len(deduped),
    )

    # -------------------------------
    # STEP 6: PDF + S3
    # -------------------------------
    if not fast_mode:
        deduped = enrich_with_pdfs(
            deduped,
            session_id=session_id,
            query_hint=state.get("raw_query") or core_question,
        )

    # -------------------------------
    # STEP 7: Embeddings + Hash
    # -------------------------------
    if not fast_mode:
        deduped = enrich_with_embeddings(deduped, session_id=session_id)

    # -------------------------------
    # STEP 8: Cap results
    # -------------------------------
    deduped.sort(key=lambda x: len(x.get("abstract", "")), reverse=True)

    max_sources = 8 if fast_mode else config.MAX_SOURCES_PER_QUERY
    final_sources = deduped[:max_sources]

    logger.info(f"Final sources returned: {len(final_sources)}")
    log_backend_event(
        "retriever_completed",
        session_id=session_id,
        final_source_count=len(final_sources),
    )

    return {
        "retrieved_sources": final_sources,
        "current_step": "retrieval_complete"
    }
