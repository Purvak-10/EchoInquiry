# aws/s3_client.py

import json
import csv
import io
import zipfile
import logging
import re
from datetime import datetime
from typing import Optional, Dict, Any, List
from botocore.exceptions import ClientError
import boto3

import config


logger = logging.getLogger(__name__)


def _slugify(text: str) -> str:
    if not text:
        return "report"

    compact = " ".join(text.strip().split())
    safe = re.sub(r"[^a-zA-Z0-9]+", "-", compact).strip("-")
    safe = re.sub(r"-{2,}", "-", safe)
    safe = safe.lower()
    safe = safe[:50].strip("-")
    return safe or "report"


class S3Client:
    """
    S3 storage layer for research agent.

    Buckets:
      - reports bucket  → structured reports
      - pdfs bucket     → downloaded research PDFs
      - exports bucket  → knowledge graph + session exports
    """

    def __init__(self) -> None:
        self.client = boto3.client("s3", region_name=config.AWS_REGION)

        self.reports_bucket: str = config.S3_BUCKET_REPORTS
        self.pdfs_bucket: str = config.S3_BUCKET_PDFS
        self.exports_bucket: str = config.S3_BUCKET_EXPORTS

    # ------------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------------

    def _session_chat_prefix(self, session_id: str, chat_hint: str | None, *, max_chars: int = 36) -> str:
        hint_slug = _slugify(chat_hint or "chat")
        if max_chars > 0:
            hint_slug = hint_slug[:max_chars].strip("-")
        hint_slug = hint_slug or "chat"
        return f"{session_id}_{hint_slug}"

    def _handle_error(self, e: ClientError) -> None:
        code = e.response.get("Error", {}).get("Code", "")
        if code == "NoSuchBucket":
            raise ValueError("Run setup_aws.py to create S3 buckets")
        raise e

    def _report_text(self, report_dict: Dict[str, Any]) -> str:
        """
        Convert structured report → readable plain text format.
        """

        title = report_dict.get("title", "")
        summary = report_dict.get("executive_summary") or report_dict.get("summary", "")

        conclusions = report_dict.get("key_conclusions", [])
        sections = report_dict.get("sections", [])
        citations = report_dict.get("citations", [])

        parts: List[str] = []

        parts.append(f"TITLE: {title}\n")
        parts.append(f"\nEXECUTIVE SUMMARY:\n{summary}\n")

        parts.append("\nKEY CONCLUSIONS:")
        for i, c in enumerate(conclusions, 1):
            parts.append(f"{i}. {c}")

        for sec in sections:
            heading = sec.get("heading", "")
            content = sec.get("content", "")
            parts.append(f"\n\n{heading}\n{content}")

        parts.append("\n\nCITATIONS:")
        for i, c in enumerate(citations, 1):
            doi = c.get("doi", "")
            parts.append(f"{i}. {c.get('title','')} — DOI: {doi}")

        return "\n".join(parts)

    def build_report_slug(
        self,
        report_dict: Dict[str, Any],
        query_hint: str | None = None,
        *,
        session_id: str | None = None,
    ) -> str:
        candidate = (
            query_hint
            or report_dict.get("raw_query")
            or report_dict.get("core_question")
            or report_dict.get("title")
            or "report"
        )
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        if session_id:
            return f"{self._session_chat_prefix(session_id, candidate)}-{timestamp}"
        return f"{_slugify(candidate)}-{timestamp}"

    def _report_prefix(self, session_id: str, slug: str) -> str:
        return f"reports/{session_id}/{slug}"

    def _find_latest_report_key(self, session_id: str) -> Optional[str]:
        prefix = f"reports/{session_id}/"
        resp = self.client.list_objects_v2(
            Bucket=self.reports_bucket,
            Prefix=prefix,
        )
        candidates = [
            obj for obj in resp.get("Contents", [])
            if obj.get("Key", "").endswith(".json")
        ]
        if not candidates:
            return None
        latest = max(candidates, key=lambda o: o.get("LastModified", datetime.min))
        return latest["Key"]

    # ------------------------------------------------------------------
    # REPORT METHODS
    # ------------------------------------------------------------------

    def save_report(
        self,
        session_id: str,
        report_dict: Dict[str, Any],
        *,
        query_hint: str | None = None,
        slug: str | None = None,
    ) -> str:
        try:
            slug_value = slug or self.build_report_slug(
                report_dict,
                query_hint=query_hint,
                session_id=session_id,
            )
            key = f"{self._report_prefix(session_id, slug_value)}.json"

            self.client.put_object(
                Bucket=self.reports_bucket,
                Key=key,
                Body=json.dumps(report_dict).encode("utf-8"),
                ContentType="application/json",
                ServerSideEncryption="AES256",
            )

            logger.info(f"S3 PUT report → s3://{self.reports_bucket}/{key}")
            return f"s3://{self.reports_bucket}/{key}"

        except ClientError as e:
            self._handle_error(e)

    def get_report(self, session_id: str) -> Optional[Dict[str, Any]]:
        try:
            key = self._find_latest_report_key(session_id)
            if not key:
                return None

            obj = self.client.get_object(
                Bucket=self.reports_bucket,
                Key=key,
            )

            logger.info(f"S3 GET report ← s3://{self.reports_bucket}/{key}")

            return json.loads(obj["Body"].read())

        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("NoSuchKey", "404"):
                return None
            self._handle_error(e)

    def save_report_as_text(
        self,
        session_id: str,
        report_dict: Dict[str, Any],
        *,
        query_hint: str | None = None,
        slug: str | None = None,
    ) -> str:
        try:
            slug_value = slug or self.build_report_slug(
                report_dict,
                query_hint=query_hint,
                session_id=session_id,
            )
            key = f"{self._report_prefix(session_id, slug_value)}.txt"

            text = self._report_text(report_dict)

            self.client.put_object(
                Bucket=self.reports_bucket,
                Key=key,
                Body=text.encode("utf-8"),
                ContentType="text/plain",
                ServerSideEncryption="AES256",
            )

            logger.info(f"S3 PUT report text → s3://{self.reports_bucket}/{key}")
            return f"s3://{self.reports_bucket}/{key}"

        except ClientError as e:
            self._handle_error(e)

    def generate_presigned_url(self, session_id: str, expires_seconds: int = 3600) -> str:
        try:
            key = self._find_latest_report_key(session_id)
            if not key:
                raise ValueError(f"No reports found for session {session_id}")

            url = self.client.generate_presigned_url(
                ClientMethod="get_object",
                Params={"Bucket": self.reports_bucket, "Key": key},
                ExpiresIn=expires_seconds,
            )

            logger.info(f"S3 PRESIGNED URL generated for {key}")
            return url

        except ClientError as e:
            self._handle_error(e)

    # ------------------------------------------------------------------
    # PDF METHODS
    # ------------------------------------------------------------------

    def _pdf_key(
        self,
        doi: str,
        *,
        session_id: str | None = None,
        query_hint: str | None = None,
        source_hint: str | None = None,
    ) -> str:
        safe_doi = re.sub(r"[^a-zA-Z0-9_.-]+", "_", doi or "unknown-doi").strip("_")
        safe_doi = safe_doi[:120] or "unknown-doi"
        if session_id:
            chat_hint = query_hint or source_hint or "source"
            prefix = self._session_chat_prefix(session_id, chat_hint)
            return f"pdfs/{session_id}/{prefix}_{safe_doi}.pdf"
        return f"pdfs/{safe_doi}.pdf"

    def _find_pdf_key(self, doi: str, *, session_id: str | None = None) -> Optional[str]:
        if not session_id:
            return self._pdf_key(doi)

        safe_doi = re.sub(r"[^a-zA-Z0-9_.-]+", "_", doi or "unknown-doi").strip("_")
        safe_doi = safe_doi[:120] or "unknown-doi"
        suffix = f"_{safe_doi}.pdf"

        resp = self.client.list_objects_v2(
            Bucket=self.pdfs_bucket,
            Prefix=f"pdfs/{session_id}/",
        )

        candidates = [
            obj for obj in resp.get("Contents", [])
            if obj.get("Key", "").endswith(suffix)
        ]
        if candidates:
            latest = max(candidates, key=lambda o: o.get("LastModified", datetime.min))
            return latest["Key"]

        # Backward compatibility with old DOI-only key.
        legacy_key = self._pdf_key(doi)
        try:
            self.client.head_object(Bucket=self.pdfs_bucket, Key=legacy_key)
            return legacy_key
        except ClientError:
            return None

    def pdf_exists(self, doi: str, *, session_id: str | None = None) -> bool:
        key = self._find_pdf_key(doi, session_id=session_id)
        if not key:
            return False
        try:
            self.client.head_object(
                Bucket=self.pdfs_bucket,
                Key=key,
            )
            return True

        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchKey", "NotFound"):
                return False
            self._handle_error(e)

    def save_pdf(
        self,
        pdf_bytes: bytes,
        doi: str,
        *,
        session_id: str | None = None,
        query_hint: str | None = None,
        source_hint: str | None = None,
    ) -> str:
        existing_key = self._find_pdf_key(doi, session_id=session_id)
        if existing_key:
            return f"s3://{self.pdfs_bucket}/{existing_key}"

        key = self._pdf_key(
            doi,
            session_id=session_id,
            query_hint=query_hint,
            source_hint=source_hint,
        )
        try:
            self.client.put_object(
                Bucket=self.pdfs_bucket,
                Key=key,
                Body=pdf_bytes,
                ContentType="application/pdf",
                ServerSideEncryption="AES256",
            )

            logger.info(f"S3 PUT pdf → s3://{self.pdfs_bucket}/{key}")
            return f"s3://{self.pdfs_bucket}/{key}"

        except ClientError as e:
            self._handle_error(e)

    def get_pdf(self, doi: str, *, session_id: str | None = None) -> Optional[bytes]:
        key = self._find_pdf_key(doi, session_id=session_id)
        if not key:
            return None
        try:
            obj = self.client.get_object(
                Bucket=self.pdfs_bucket,
                Key=key,
            )

            logger.info(f"S3 GET pdf ← s3://{self.pdfs_bucket}/{key}")
            return obj["Body"].read()

        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("NoSuchKey", "404"):
                return None
            self._handle_error(e)

    # ------------------------------------------------------------------
    # KNOWLEDGE GRAPH
    # ------------------------------------------------------------------

    def save_knowledge_graph(self, graph_dict: Dict[str, Any]) -> str:
        try:
            key = config.S3_KNOWLEDGE_GRAPH_KEY

            self.client.put_object(
                Bucket=self.exports_bucket,
                Key=key,
                Body=json.dumps(graph_dict).encode("utf-8"),
                ContentType="application/json",
                ServerSideEncryption="AES256",
            )

            logger.info(f"S3 PUT knowledge graph → s3://{self.exports_bucket}/{key}")
            return f"s3://{self.exports_bucket}/{key}"

        except ClientError as e:
            self._handle_error(e)

    def load_knowledge_graph(self) -> Optional[Dict[str, Any]]:
        try:
            key = config.S3_KNOWLEDGE_GRAPH_KEY

            obj = self.client.get_object(
                Bucket=self.exports_bucket,
                Key=key,
            )

            logger.info(f"S3 GET knowledge graph ← s3://{self.exports_bucket}/{key}")
            return json.loads(obj["Body"].read())

        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("NoSuchKey", "404"):
                return None
            self._handle_error(e)

    # ------------------------------------------------------------------
    # EXPORTS
    # ------------------------------------------------------------------

    def export_session_zip(
        self,
        session_id: str,
        report_dict: Dict[str, Any],
        sources_list: List[Dict[str, Any]],
        *,
        query_hint: str | None = None,
    ) -> str:
        try:
            buffer = io.BytesIO()

            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:

                # report.json
                z.writestr(
                    "report.json",
                    json.dumps(report_dict, indent=2),
                )

                # report.txt
                z.writestr(
                    "report.txt",
                    self._report_text(report_dict),
                )

                # sources.json
                z.writestr(
                    "sources.json",
                    json.dumps(sources_list, indent=2),
                )

                # citations.csv
                csv_buf = io.StringIO()
                writer = csv.writer(csv_buf)
                writer.writerow(
                    ["source_id", "title", "authors", "year", "doi", "url"]
                )

                for s in sources_list:
                    writer.writerow(
                        [
                            s.get("source_id"),
                            s.get("title"),
                            ", ".join(s.get("authors", [])),
                            s.get("year"),
                            s.get("doi"),
                            s.get("url"),
                        ]
                    )

                z.writestr("citations.csv", csv_buf.getvalue())

            buffer.seek(0)

            slug = self._session_chat_prefix(
                session_id,
                query_hint
                or report_dict.get("raw_query")
                or report_dict.get("title")
                or "export",
            )
            timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            key = f"exports/{session_id}/{slug}-{timestamp}.zip"

            self.client.put_object(
                Bucket=self.exports_bucket,
                Key=key,
                Body=buffer.read(),
                ContentType="application/zip",
                ServerSideEncryption="AES256",
            )

            logger.info(f"S3 PUT export zip → s3://{self.exports_bucket}/{key}")

            url = self.client.generate_presigned_url(
                ClientMethod="get_object",
                Params={"Bucket": self.exports_bucket, "Key": key},
                ExpiresIn=86400,
            )

            return url

        except ClientError as e:
            self._handle_error(e)

    # ------------------------------------------------------------------
    # LIST REPORTS
    # ------------------------------------------------------------------

    def list_reports(self) -> List[Dict[str, Any]]:
        try:
            resp = self.client.list_objects_v2(
                Bucket=self.reports_bucket,
                Prefix="reports/",
            )

            items: List[Dict[str, Any]] = []

            for obj in resp.get("Contents", []):
                key: str = obj["Key"]

                if not key.endswith(".json"):
                    continue

                parts = key.split("/")
                session_id = parts[1] if len(parts) > 2 else "unknown"
                slug = parts[-1].rsplit(".", 1)[0] if "." in parts[-1] else parts[-1]
                display_name = slug.replace("-", " ").strip()

                items.append(
                    {
                        "session_id": session_id,
                        "last_modified": obj["LastModified"].isoformat(),
                        "size_kb": round(obj["Size"] / 1024, 2),
                        "display_name": display_name or session_id,
                    }
                )

            return items

        except ClientError as e:
            self._handle_error(e)
