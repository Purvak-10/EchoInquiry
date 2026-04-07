import time
import boto3
import config
from uuid import uuid4
from decimal import Decimal
from datetime import datetime
from boto3.dynamodb.conditions import Key, Attr
from botocore.exceptions import ClientError


class DynamoDBClient:
    SESSION_COUNTER_KEY = "__meta_session_counter__"

    # ================= INIT =================

    def __init__(self):
        self.resource = boto3.resource(
            "dynamodb",
            region_name=config.AWS_REGION
        )

        # preload tables with validation
        try:
            self.sessions = self.resource.Table(config.TABLE_SESSIONS)
            self.sessions.load()  # Validate table exists
            self.sources = self.resource.Table(config.TABLE_SOURCES)
            self.sources.load()
            self.hypotheses = self.resource.Table(config.TABLE_HYPOTHESES)
            self.hypotheses.load()
            self.contradictions = self.resource.Table(config.TABLE_CONTRADICTIONS)
            self.contradictions.load()
            self.living_doc_checks = self.resource.Table(config.TABLE_LIVING_DOC_CHECKS)
            self.living_doc_checks.load()
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                raise RuntimeError(
                    f"DynamoDB tables not found. Run setup_aws.py first. Missing tables: {config.TABLE_SESSIONS}"
                )
            raise

    # ================= HELPERS =================

    def generate_id(self) -> str:
        return str(uuid4())

    def now_iso(self) -> str:
        return datetime.utcnow().isoformat() + "Z"

    # ---------- float → Decimal ----------

    def float_to_decimal(self, obj):
        if isinstance(obj, float):
            return Decimal(str(obj))
        elif isinstance(obj, dict):
            return {k: self.float_to_decimal(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.float_to_decimal(v) for v in obj]
        return obj

    # ---------- Decimal → float ----------

    def decimal_to_float(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self.decimal_to_float(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.decimal_to_float(v) for v in obj]
        return obj

    # ---------- retry wrapper ----------

    def _retry(self, fn):
        try:
            return fn()
        except ClientError as e:
            code = e.response["Error"]["Code"]

            if code == "ResourceNotFoundException":
                raise ValueError("Run setup_aws.py first")

            if code == "ProvisionedThroughputExceededException":
                time.sleep(2)
                return fn()

            raise

    # ================= SESSION METHODS =================

    def create_session(self, session_id, raw_query) -> dict:

        item = {
            "session_id": session_id,
            "raw_query": raw_query,
            "created_at": self.now_iso(),
            "status": "in_progress"
        }

        def op():
            self.sessions.put_item(Item=item)

        self._retry(op)
        return item

    def allocate_session_id(self, prefix: str = "session") -> str:
        """
        Allocate monotonic, human-readable session ids: session1, session2, ...
        Uses an atomic counter in the sessions table.
        """
        now = self.now_iso()

        def op():
            return self.sessions.update_item(
                Key={"session_id": self.SESSION_COUNTER_KEY},
                UpdateExpression=(
                    "SET #counter = if_not_exists(#counter, :zero) + :inc, "
                    "#type = :type, updated_at = :updated_at"
                ),
                ExpressionAttributeNames={
                    "#counter": "counter_value",
                    "#type": "item_type",
                },
                ExpressionAttributeValues={
                    ":zero": 0,
                    ":inc": 1,
                    ":type": "meta_counter",
                    ":updated_at": now,
                },
                ReturnValues="UPDATED_NEW",
            )

        res = self._retry(op)
        attrs = res.get("Attributes", {})
        next_value = attrs.get("counter_value", 1)

        try:
            next_number = int(next_value)
        except Exception:
            next_number = int(Decimal(str(next_value)))

        return f"{prefix}{next_number}"

    def get_session(self, session_id):

        def op():
            return self.sessions.get_item(Key={"session_id": session_id})

        res = self._retry(op)
        item = res.get("Item")
        return self.decimal_to_float(item) if item else None

    def update_session_status(self, session_id, status):

        def op():
            self.sessions.update_item(
                Key={"session_id": session_id},
                UpdateExpression="SET #s = :s",
                ExpressionAttributeNames={"#s": "status"},
                ExpressionAttributeValues={":s": status}
            )

        self._retry(op)

    def save_session_report(self, session_id, report_dict, s3_uri):

        report_dict = self.float_to_decimal(report_dict)

        def op():
            self.sessions.update_item(
                Key={"session_id": session_id},
                UpdateExpression="""
                    SET final_report_json = :r,
                        s3_report_uri = :u,
                        #s = :c
                """,
                ExpressionAttributeNames={"#s": "status"},
                ExpressionAttributeValues={
                    ":r": report_dict,
                    ":u": s3_uri,
                    ":c": "complete"
                }
            )

        self._retry(op)

    def list_sessions(self, limit=20):

        def op():
            return self.sessions.scan(
                Limit=limit,
                FilterExpression=Attr("raw_query").exists()
            )

        res = self._retry(op)
        items = res.get("Items", [])

        items = sorted(
            items,
            key=lambda x: x.get("created_at", ""),
            reverse=True
        )

        return self.decimal_to_float(items)

    # ================= SOURCE METHODS =================

    def save_source(self, source_dict) -> str:

        source_dict = self.float_to_decimal(source_dict)
        source_id = source_dict.get("source_id") or self.generate_id()
        source_dict["source_id"] = source_id

        def op():
            self.sources.put_item(Item=source_dict)

        self._retry(op)
        return source_id

    def get_sources_for_session(self, session_id):

        def op():
            return self.sources.query(
                IndexName="session-index",
                KeyConditionExpression=Key("session_id").eq(session_id)
            )

        res = self._retry(op)
        return self.decimal_to_float(res.get("Items", []))

    def get_sources_due_for_recheck(self):

        now = self.now_iso()

        def op():
            return self.sources.scan(
                FilterExpression=Attr("next_check_at").lte(now)
            )

        res = self._retry(op)
        return self.decimal_to_float(res.get("Items", []))

    def update_source_after_recheck(
        self,
        source_id,
        session_id,
        new_hash,
        retraction_status,
        next_check_at
    ):

        def op():
            self.sources.update_item(
                Key={
                    "source_id": source_id,
                    "session_id": session_id
                },
                UpdateExpression="""
                    SET content_hash = :h,
                        retraction_status = :r,
                        next_check_at = :n
                """,
                ExpressionAttributeValues={
                    ":h": new_hash,
                    ":r": retraction_status,
                    ":n": next_check_at
                }
            )

        self._retry(op)

    def update_source(self, source_id, updates):
        """
        Flexible update method for any source fields.
        Updates is a dict like {"citation_count": 42, "access_status": "dead_link", ...}
        """
        if not updates:
            return

        # Convert floats to Decimal
        updates = self.float_to_decimal(updates)

        # Build UpdateExpression dynamically
        update_expr_parts = []
        expr_values = {}

        for key, value in updates.items():
            update_expr_parts.append(f"{key} = :{key}")
            expr_values[f":{key}"] = value

        update_expr = "SET " + ", ".join(update_expr_parts)

        def op():
            self.sources.update_item(
                Key={"source_id": source_id},
                UpdateExpression=update_expr,
                ExpressionAttributeValues=expr_values,
            )

        self._retry(op)

    # ================= HYPOTHESIS METHODS =================

    def save_all_hypotheses(self, hypotheses_list, session_id):

        with self.hypotheses.batch_writer() as batch:
            for h in hypotheses_list:
                h = self.float_to_decimal(h)
                h["session_id"] = session_id
                h["hypothesis_id"] = h.get("hypothesis_id") or self.generate_id()
                batch.put_item(Item=h)

    def get_hypotheses_for_session(self, session_id):

        def op():
            return self.hypotheses.query(
                IndexName="session-index",
                KeyConditionExpression=Key("session_id").eq(session_id)
            )

        res = self._retry(op)
        return self.decimal_to_float(res.get("Items", []))

    # ================= CONTRADICTION METHODS =================

    def save_contradiction(self, contradiction_dict, session_id) -> str:

        contradiction_dict = self.float_to_decimal(contradiction_dict)
        contradiction_id = contradiction_dict.get("contradiction_id") or self.generate_id()

        contradiction_dict["contradiction_id"] = contradiction_id
        contradiction_dict["session_id"] = session_id

        def op():
            self.contradictions.put_item(Item=contradiction_dict)

        self._retry(op)
        return contradiction_id

    def get_contradictions_for_session(self, session_id):

        def op():
            return self.contradictions.query(
                IndexName="session-index",
                KeyConditionExpression=Key("session_id").eq(session_id)
            )

        res = self._retry(op)
        return self.decimal_to_float(res.get("Items", []))

    # ================= LIVING DOC METHODS =================

    def save_living_doc_check(self, check_dict) -> str:

        check_dict = self.float_to_decimal(check_dict)
        check_id = self.generate_id()

        check_dict["check_id"] = check_id
        check_dict["checked_at"] = self.now_iso()

        def op():
            self.living_doc_checks.put_item(Item=check_dict)

        self._retry(op)
        return check_id

    def get_open_alerts(self, limit=50):

        def op():
            return self.living_doc_checks.scan(
                FilterExpression=Attr("alert_sent").eq(0),
                Limit=limit
            )

        res = self._retry(op)
        return self.decimal_to_float(res.get("Items", []))

    def mark_alert_sent(self, check_id, source_id):

        def op():
            self.living_doc_checks.update_item(
                Key={
                    "check_id": check_id,
                    "source_id": source_id
                },
                UpdateExpression="SET alert_sent = :a",
                ExpressionAttributeValues={":a": 1}
            )

        self._retry(op)
