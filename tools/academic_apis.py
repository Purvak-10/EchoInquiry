import requests
import time
import uuid
import hashlib
from typing import List, Dict, Optional
import xml.etree.ElementTree as ET


# =========================
# Helpers
# =========================

def _backoff_request(url: str, params: dict = None) -> Optional[requests.Response]:
    delays = [1, 2]
    for i in range(3):
        try:
            r = requests.get(url, params=params, timeout=6)
            if r.status_code == 200:
                return r

            if r.status_code == 429 or r.status_code >= 500:
                if i < len(delays):
                    time.sleep(delays[i])
                    continue
            return None
        except Exception:
            if i < 1:  # only 1 retry for transient errors, then give up
                time.sleep(delays[i])
            else:
                return None
    return None


def _hash_text(text: str) -> str:
    if not text:
        return ""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _authors_to_str(authors_list) -> str:
    try:
        names = []
        for a in authors_list:
            if isinstance(a, dict):
                if "name" in a:
                    names.append(a["name"])
                else:
                    first = a.get("given", "")
                    last = a.get("family", "")
                    names.append(f"{first} {last}".strip())
        return ", ".join(names)
    except Exception:
        return ""


def _base_source() -> Dict:
    return {
        "source_id": str(uuid.uuid4()),
        "title": "",
        "authors": "",
        "year": 0,
        "doi": None,
        "url": "",
        "abstract": "",
        "full_text_snippet": "",
        "journal": None,
        "citation_count": 0,
        "credibility_score": 0.5,
        "retraction_status": "active",
        "content_hash": "",
        "content_embedding": [],
        "content_xpath": "//body",
        "source_type": "academic",
        "s3_pdf_uri": "",
    }


# =========================
# Semantic Scholar
# =========================

class SemanticScholarAPI:
    BASE = "https://api.semanticscholar.org/graph/v1"

    def search(self, query: str, limit: int = 10) -> List[Dict]:
        url = f"{self.BASE}/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "fields": "title,authors,year,abstract,citationCount,externalIds,openAccessPdf"
        }

        r = _backoff_request(url, params)
        if not r:
            return []

        data = r.json().get("data", [])
        results = []

        for p in data:
            try:
                s = _base_source()
                s["title"] = p.get("title", "")
                s["authors"] = _authors_to_str(p.get("authors", []))
                s["year"] = p.get("year") or 0
                s["abstract"] = p.get("abstract") or ""
                s["citation_count"] = p.get("citationCount") or 0

                doi = (p.get("externalIds") or {}).get("DOI")
                s["doi"] = doi

                pdf = (p.get("openAccessPdf") or {}).get("url")
                s["url"] = pdf or f"https://www.semanticscholar.org/paper/{p.get('paperId','')}"

                s["content_hash"] = _hash_text(s["abstract"])
                results.append(s)
            except Exception:
                continue

        return results

    def get_paper(self, paper_id: str) -> Dict:
        url = f"{self.BASE}/paper/{paper_id}"
        params = {
            "fields": "title,authors,year,abstract,citationCount,externalIds,openAccessPdf,journal"
        }

        r = _backoff_request(url, params)
        if not r:
            return _base_source()

        p = r.json()

        s = _base_source()
        try:
            s["title"] = p.get("title", "")
            s["authors"] = _authors_to_str(p.get("authors", []))
            s["year"] = p.get("year") or 0
            s["abstract"] = p.get("abstract") or ""
            s["citation_count"] = p.get("citationCount") or 0
            s["journal"] = (p.get("journal") or {}).get("name")

            doi = (p.get("externalIds") or {}).get("DOI")
            s["doi"] = doi

            pdf = (p.get("openAccessPdf") or {}).get("url")
            s["url"] = pdf or f"https://www.semanticscholar.org/paper/{paper_id}"

            s["content_hash"] = _hash_text(s["abstract"])
        except Exception:
            pass

        return s

    def get_citations(self, paper_id: str, limit: int = 20) -> List[Dict]:
        url = f"{self.BASE}/paper/{paper_id}/citations"
        params = {
            "limit": limit,
            "fields": "title,authors,year,abstract,citationCount,externalIds"
        }

        r = _backoff_request(url, params)
        if not r:
            return []

        results = []
        for item in r.json().get("data", []):
            try:
                p = item.get("citingPaper", {})
                s = _base_source()

                s["title"] = p.get("title", "")
                s["authors"] = _authors_to_str(p.get("authors", []))
                s["year"] = p.get("year") or 0
                s["abstract"] = p.get("abstract") or ""
                s["citation_count"] = p.get("citationCount") or 0
                s["doi"] = (p.get("externalIds") or {}).get("DOI")

                s["url"] = f"https://www.semanticscholar.org/paper/{p.get('paperId','')}"
                s["content_hash"] = _hash_text(s["abstract"])

                results.append(s)
            except Exception:
                continue

        return results


# =========================
# PubMed
# =========================

class PubMedAPI:
    BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def search(self, query: str, limit: int = 10) -> List[Dict]:

        esearch = f"{self.BASE}/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": limit,
            "retmode": "json"
        }

        r = _backoff_request(esearch, params)
        if not r:
            return []

        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []

        efetch = f"{self.BASE}/efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "xml"
        }

        r = _backoff_request(efetch, params)
        if not r:
            return []

        root = ET.fromstring(r.text)

        results = []

        for article in root.findall(".//PubmedArticle"):
            try:
                s = _base_source()

                title = article.findtext(".//ArticleTitle")
                abstract = article.findtext(".//AbstractText")

                year = article.findtext(".//PubDate/Year")
                journal = article.findtext(".//Journal/Title")

                authors = []
                for a in article.findall(".//Author"):
                    last = a.findtext("LastName") or ""
                    first = a.findtext("ForeName") or ""
                    authors.append(f"{first} {last}".strip())

                pmid = article.findtext(".//PMID")

                s["title"] = title or ""
                s["abstract"] = abstract or ""
                s["year"] = int(year) if year and year.isdigit() else 0
                s["authors"] = ", ".join(authors)
                s["journal"] = journal
                s["url"] = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

                s["content_hash"] = _hash_text(s["abstract"])

                results.append(s)
            except Exception:
                continue

        return results


# =========================
# Crossref
# =========================

class CrossrefAPI:
    BASE = "https://api.crossref.org/works"

    def search(self, query: str, limit: int = 10) -> List[Dict]:
        params = {
            "query": query,
            "rows": limit
        }

        r = _backoff_request(self.BASE, params)
        if not r:
            return []

        items = r.json().get("message", {}).get("items", [])
        results = []

        for it in items:
            try:
                s = _base_source()

                s["title"] = (it.get("title") or [""])[0]
                s["authors"] = _authors_to_str(it.get("author", []))
                s["year"] = (it.get("issued", {}).get("date-parts", [[0]])[0][0])
                s["doi"] = it.get("DOI")
                s["url"] = it.get("URL", "")
                s["journal"] = (it.get("container-title") or [None])[0]
                s["abstract"] = it.get("abstract") or ""
                s["citation_count"] = it.get("is-referenced-by-count") or 0

                s["content_hash"] = _hash_text(s["abstract"])
                results.append(s)
            except Exception:
                continue

        return results

    def get_by_doi(self, doi: str) -> Dict:
        url = f"{self.BASE}/{doi}"
        r = _backoff_request(url)

        if not r:
            return _base_source()

        it = r.json().get("message", {})
        s = _base_source()

        try:
            s["title"] = (it.get("title") or [""])[0]
            s["authors"] = _authors_to_str(it.get("author", []))
            s["year"] = (it.get("issued", {}).get("date-parts", [[0]])[0][0])
            s["doi"] = doi
            s["url"] = it.get("URL", "")
            s["journal"] = (it.get("container-title") or [None])[0]
            s["abstract"] = it.get("abstract") or ""
            s["citation_count"] = it.get("is-referenced-by-count") or 0

            if self.check_retraction(doi):
                s["retraction_status"] = "retracted"

            s["content_hash"] = _hash_text(s["abstract"])
        except Exception:
            pass

        return s

    def check_retraction(self, doi: str) -> bool:
        url = f"{self.BASE}/{doi}"
        r = _backoff_request(url)
        if not r:
            return False

        rel = r.json().get("message", {}).get("relation", {})
        return "is-retracted-by" in rel


# =========================
# Unpaywall
# =========================

class UnpaywallAPI:
    BASE = "https://api.unpaywall.org/v2"

    def get_open_access_url(self, doi: str, email: str = "research@agent.local") -> Optional[str]:
        url = f"{self.BASE}/{doi}"
        params = {"email": email}

        r = _backoff_request(url, params)
        if not r:
            return None

        data = r.json()

        loc = data.get("best_oa_location")
        if loc and loc.get("url_for_pdf"):
            return loc["url_for_pdf"]

        for l in data.get("oa_locations", []):
            if l.get("url_for_pdf"):
                return l["url_for_pdf"]

        return None
