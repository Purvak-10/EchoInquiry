import re
from typing import Dict, List, Optional

import fitz  # PyMuPDF


class PDFParser:

    # --------------------------------------------------

    def parse(self, file_bytes: bytes) -> Dict:
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")

            pages_text = []
            for page in doc:
                pages_text.append(page.get_text())

            full_text = "\n".join(pages_text)

            abstract = full_text[:1500]
            references = self.extract_references(full_text)

            sections = {
                "introduction": self.extract_section(full_text, "introduction"),
                "methods": self.extract_section(full_text, "methods"),
                "results": self.extract_section(full_text, "results"),
                "discussion": self.extract_section(full_text, "discussion"),
                "conclusion": self.extract_section(full_text, "conclusion"),
            }

            return {
                "full_text": full_text,
                "abstract": abstract,
                "sections": sections,
                "references": references,
                "page_count": doc.page_count,
                "error": None,
            }

        except Exception as e:
            return {
                "full_text": "",
                "abstract": "",
                "sections": {},
                "references": [],
                "page_count": 0,
                "error": str(e),
            }

    # --------------------------------------------------

    def extract_section(self, full_text: str, section_name: str) -> str:
        try:
            pattern = re.compile(
                rf"\n\s*{section_name}\s*\n(.*?)(\n[A-Z][A-Za-z\s]{2,}\n)",
                re.IGNORECASE | re.DOTALL,
            )

            match = pattern.search(full_text)
            if match:
                return match.group(1).strip()

            return ""

        except Exception:
            return ""

    # --------------------------------------------------

    def extract_references(self, full_text: str) -> List[str]:
        refs = []

        try:
            patterns = [
                r"\[\d+\].+",
                r"\(\d{4}\).+",
                r"\d+\.\s.+",
            ]

            lines = full_text.splitlines()

            for line in lines:
                for p in patterns:
                    if re.match(p, line.strip()):
                        refs.append(line.strip())
                        break

            return refs

        except Exception:
            return refs