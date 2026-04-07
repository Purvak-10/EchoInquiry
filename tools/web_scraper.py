import hashlib
from typing import Optional, Dict

import requests
from bs4 import BeautifulSoup
from lxml import html as lxml_html
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class WebScraper:

    def __init__(self):

        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120 Safari/537.36"
            )
        }

        self.session = requests.Session()

        retry = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )

        adapter = HTTPAdapter(max_retries=retry)

        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    # --------------------------------------------------

    def fetch(self, url: str) -> Dict:
        try:
            r = self.session.get(
                url,
                headers=self.headers,
                timeout=15,
                allow_redirects=True,
            )

            html = r.text if "text/html" in r.headers.get("Content-Type", "") else ""
            text = self.extract_main_content(html)

            return {
                "html": html,
                "text": text,
                "status_code": r.status_code,
                "final_url": r.url,
                "error": None,
            }

        except requests.exceptions.SSLError as ssl_error:
            # ⚠️ SSL verification failed - return error instead of disabling verification
            # IMPORTANT: Do NOT disable SSL verification (verify=False) as it creates MITM vulnerability
            import logging
            logging.warning(f"SSL verification failed for {url}: {ssl_error}")
            return {
                "html": "",
                "text": "",
                "status_code": None,
                "final_url": url,
                "error": f"SSL verification failed: {ssl_error}",
            }

        except Exception as e:
            return self._error(url, e)

    # --------------------------------------------------

    def _error(self, url, e):
        return {
            "html": "",
            "text": "",
            "status_code": None,
            "final_url": url,
            "error": str(e),
        }

    # --------------------------------------------------

    def extract_main_content(self, html: str) -> str:
        if not html:
            return ""

        try:
            soup = BeautifulSoup(html, "lxml")

            for tag in soup(["nav", "footer", "header", "script", "style", "aside"]):
                tag.decompose()

            text = soup.get_text("\n")
            text = "\n".join(x.strip() for x in text.splitlines() if x.strip())

            return text[:5000]

        except Exception:
            return ""

    # --------------------------------------------------

    def extract_by_xpath(self, html: str, xpath: str) -> str:
        try:
            tree = lxml_html.fromstring(html)
            nodes = tree.xpath(xpath)

            if not nodes:
                return self.extract_main_content(html)

            text = "\n".join(
                n.text_content() if hasattr(n, "text_content") else str(n)
                for n in nodes
            )

            return text[:5000]

        except Exception:
            return self.extract_main_content(html)

    # --------------------------------------------------

    def compute_content_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    # --------------------------------------------------

    def is_pdf_url(self, url: str) -> bool:
        return url.lower().endswith(".pdf")

    # --------------------------------------------------

    def download_pdf_bytes(self, url: str) -> Optional[bytes]:
        try:
            r = self.session.get(url, headers=self.headers, timeout=20)
            if r.status_code == 200:
                return r.content
            return None
        except Exception:
            return None

    # --------------------------------------------------

    def check_url_accessible(self, url: str) -> bool:
        """Check if a URL is still accessible (for dead link detection)"""
        try:
            r = self.session.head(url, headers=self.headers, timeout=10, allow_redirects=True)
            # Consider 200-399 as accessible, 404+ as dead, timeouts as dead
            return 200 <= r.status_code < 400
        except Exception:
            # Timeouts, connection errors, etc. = dead link
            return False


# Alias for backwards compatibility
AsyncWebScraper = WebScraper
