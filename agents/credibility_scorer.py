import logging
from typing import Dict, List

# Assuming this exists in your tools layer
from tools.retraction_checker import RetractionChecker

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


HIGH_TIER_JOURNALS = [
    "nature", "science", "nejm", "lancet", "cell",
    "jama", "bmj", "plos", "ieee", "acm",
    "arxiv", "biorxiv", "medrxiv", "pubmed", "nih"
]


def _compute_citation_score(citation_count: int) -> float:
    """
    Compute citation score with better scaling.
    - 100+ citations: 0.9
    - 20-99 citations: 0.6-0.8
    - 1-19 citations: 0.3-0.5
    - 0 citations: 0.2 (new papers aren't invalid)
    """
    if not citation_count:
        return 0.2  # Don't penalize uncited papers too harshly
    elif citation_count > 100:
        return 0.9
    elif citation_count > 20:
        return 0.7
    else:
        return 0.4  # Small number of citations still counts


def _compute_journal_score(source: Dict) -> float:
    """
    Compute journal/source quality score.
    Peer-reviewed papers have inherent credibility.
    """
    url = (source.get("url") or "").lower()
    journal = (source.get("journal") or "").lower()
    doi = source.get("doi")

    combined = f"{url} {journal}"

    if any(j in combined for j in HIGH_TIER_JOURNALS):
        return 1.0
    elif doi:
        # Peer-reviewed papers with DOI are credible by default
        return 0.8
    elif "edu" in url or "gov" in url or "org" in url or ".ac." in url:
        # Reputable institutions and organizations
        return 0.7
    elif url:
        return 0.5  # Web sources get neutral score
    else:
        return 0.3


def _compute_recency_score(year: int) -> float:
    """Compute recency score with less penalty for older papers."""
    if not year:
        return 0.4  # Unknown year gets neutral score

    if year >= 2022:
        return 1.0
    elif year >= 2020:
        return 0.85
    elif year >= 2018:
        return 0.75  # More lenient
    elif year >= 2015:
        return 0.60
    elif year >= 2010:
        return 0.45  # Older papers still have value
    else:
        return 0.30


def _compute_retraction_score(status: str) -> float:
    """Compute retraction penalty."""
    if status == "retracted":
        return 0.0
    elif status == "flagged":
        return 0.5
    return 1.0


def _compute_final_score(source: Dict) -> float:
    """
    Compute final credibility score with balanced weighting.
    Minimum floor of 0.3 for any retrieved source.
    """
    citation_score = _compute_citation_score(source.get("citation_count", 0))
    journal_score = _compute_journal_score(source)
    recency_score = _compute_recency_score(source.get("year"))
    retraction_score = _compute_retraction_score(source.get("retraction_status", "unknown"))

    score = (
        0.30 * citation_score +    # Citations are good but not everything
        0.35 * journal_score +     # Journal/source quality is primary
        0.20 * recency_score +     # Recency matters but old papers aren't worthless
        0.15 * retraction_score    # Retraction is a major penalty but weighted lower
    )

    # Floor at 0.3: any source retrieved is at least somewhat credible
    score = max(0.3, min(1.0, score))
    
    return round(score, 3)


def credibility_scorer_node(state: Dict) -> Dict:
    """
    Scores credibility of retrieved sources and sorts them.

    INPUT:
        state["retrieved_sources"]: List[Dict]

    OUTPUT:
        {
            "retrieved_sources": List[Dict],
            "current_step": "credibility_scored"
        }
    """

    sources: List[Dict] = state.get("retrieved_sources", [])

    if not sources:
        logger.warning("No sources received for credibility scoring.")
        return {
            "retrieved_sources": [],
            "current_step": "credibility_scored"
        }

    # STEP 1: Retraction check ONLY for academic sources with DOI (skip web sources)
    # This saves 5-10 seconds by not calling the retraction API for every web result
    try:
        # Filter sources: only check academic papers with DOI
        sources_to_check = [
            s for s in sources
            if s.get("source_type", "web") == "academic" and s.get("doi")
        ]
        
        if sources_to_check:
            raw_retraction_results = RetractionChecker().bulk_check(sources_to_check)
        else:
            raw_retraction_results = []
    except Exception as e:
        logger.error(f"Retraction check failed: {e}")
        raw_retraction_results = []

    retraction_results = {}
    if isinstance(raw_retraction_results, list):
        for src, result in zip(sources_to_check if 'sources_to_check' in locals() else [], raw_retraction_results):
            source_id = (
                src.get("source_id")
                or src.get("id")
                or src.get("url")
                or src.get("doi")
            )
            if not source_id:
                continue

            if result.get("is_retracted"):
                retraction_results[source_id] = "retracted"
            else:
                retraction_results[source_id] = "ok"
    elif isinstance(raw_retraction_results, dict):
        retraction_results = raw_retraction_results

    # STEP 2: Update retraction status
    retracted_count = 0

    for src in sources:
        source_id = (
            src.get("source_id")
            or src.get("id")
            or src.get("url")
            or src.get("doi")
        )

        status = retraction_results.get(source_id, "unknown")

        # Normalize status
        if status not in ["retracted", "flagged"]:
            status = "ok"

        if status == "retracted":
            retracted_count += 1

        src["retraction_status"] = status

    # STEP 3: Compute scores
    for src in sources:
        src["credibility_score"] = _compute_final_score(src)

    # STEP 4: Sort descending
    sources_sorted = sorted(
        sources,
        key=lambda x: x.get("credibility_score", 0),
        reverse=True
    )

    logger.info(f"[CredibilityScorer] Retracted sources found: {retracted_count}")

    return {
        "retrieved_sources": sources_sorted,
        "current_step": "credibility_scored"
    }
