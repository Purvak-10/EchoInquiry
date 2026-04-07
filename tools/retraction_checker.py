# tools/retraction_checker.py
import re
import threading
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class RetractionChecker:
    CROSSREF_URL = "https://api.crossref.org/works/{}"
    RETRACTION_WATCH_RSS = "https://retractionwatch.com/feed/"
    TIMEOUT = 5

    def __init__(self):
        # Setup session with retry logic
        self.session = requests.Session()
        retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _now(self):
        return datetime.now(timezone.utc).isoformat()

    def _safe_result(self, doi):
        return {
            "is_retracted": False,
            "retraction_date": None,
            "reason": None,
            "source": None,
            "checked_at": self._now(),
            "doi": doi,
        }

    # CHECK 1 — Crossref
    # -----------------------------
    def _check_crossref(self, doi, result, lock):
        if not doi:
            return

        try:
            url = self.CROSSREF_URL.format(doi)
            r = self.session.get(url, timeout=self.TIMEOUT)  # ✅ Now uses session with retry
            if r.status_code != 200:
                return

            data = r.json()
            msg = data.get("message", {})
            relation = msg.get("relation", {})

            if "is-retracted-by" in relation:
                rel_items = relation.get("is-retracted-by", [])

                retraction_date = None
                if msg.get("update-to"):
                    try:
                        parts = msg["update-to"][0]["updated"]["date-parts"][0]
                        retraction_date = "-".join(map(str, parts))
                    except Exception:
                        pass

                with lock:
                    result.update(
                        {
                            "is_retracted": True,
                            "retraction_date": retraction_date,
                            "reason": "Crossref relation is-retracted-by",
                            "source": "crossref",
                        }
                    )
        except Exception:
            return

    # -----------------------------
    # CHECK 2 — Retraction Watch RSS
    # -----------------------------

    def _check_rss(self, title, result, lock):
        if not title:
            return

        # require meaningful title
        words = re.findall(r"\w+", title.lower())
        if len(words) < 6:
            return

        query = " ".join(words[:6])

        try:
            r = self.session.get(self.RETRACTION_WATCH_RSS, timeout=self.TIMEOUT)  # ✅ Now uses session with retry
            if r.status_code != 200:
                return

            root = ET.fromstring(r.content)

            for item in root.iter("item"):

                # stop work if already confirmed by other thread
                if result["is_retracted"]:
                    return

                item_title_el = item.find("title")
                if item_title_el is None:
                    continue

                item_words = re.findall(r"\w+", (item_title_el.text or "").lower())

                # require strong overlap (>=4 of first 6 words)
                overlap = len(set(words[:6]) & set(item_words))

                if overlap >= 4:
                    pub_date_el = item.find("pubDate")
                    retraction_date = (
                        pub_date_el.text if pub_date_el is not None else None
                    )

                    with lock:
                        result.update(
                            {
                                "is_retracted": True,
                                "retraction_date": retraction_date,
                                "reason": "Retraction Watch fuzzy title match",
                                "source": "retraction_watch",
                            }
                        )
                    return

        except Exception:
            return

    # -----------------------------
    # PUBLIC API
    # -----------------------------
    def check(self, doi, title):
        result = self._safe_result(doi)
        lock = threading.Lock()

        t1 = threading.Thread(
            target=self._check_crossref, args=(doi, result, lock)
        )
        t2 = threading.Thread(
            target=self._check_rss, args=(title, result, lock)
        )

        t1.start()
        t2.start()
        
        # Set timeout for threads to prevent hanging
        t1.join(timeout=self.TIMEOUT + 1)
        t2.join(timeout=self.TIMEOUT + 1)
        
        # Ensure result has required fields even if threads timed out
        with lock:
            if "checked_at" not in result:
                result["checked_at"] = self._now()

        return result

    def bulk_check(self, sources_list):
        def _run(src):
            doi = src.get("doi")
            title = src.get("title")
            return self.check(doi, title)

        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(_run, sources_list))

        return results


# Alias for backwards compatibility
AsyncRetractionChecker = RetractionChecker
