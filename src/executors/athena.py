"""Athena execution.

Structured sources never move their data into the graph — only metadata. Rows are
queried in place, here, at read time. So this module is the only place tenant data
from a warehouse is touched, and the firewall runs *inside* `execute` rather than
being something a caller is trusted to have done first.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import boto3

from src.query.firewall import SQLFirewall

logger = logging.getLogger(__name__)

DEFAULT_WORKGROUP = "primary"
DEFAULT_TIMEOUT_SECONDS = 60.0
_POLL_INITIAL = 0.5
_POLL_MAX = 3.0
_POLL_BACKOFF = 1.5


@dataclass
class QueryResult:
    success: bool
    columns: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    """True when the row cap was hit — the caller is seeing a prefix, not the answer.
    Reported explicitly because a silently truncated aggregate is a wrong number."""

    duration_ms: float = 0.0
    query_execution_id: str = ""
    bytes_scanned: int = 0
    error: str | None = None
    error_code: str | None = None
    """One of: blocked, start_failed, query_error, timeout."""


@dataclass
class AthenaConfig:
    workgroup: str = DEFAULT_WORKGROUP
    output_location: str = ""
    database: str = ""
    region: str = ""
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS


class AthenaExecutor:
    def __init__(
        self,
        config: AthenaConfig,
        firewall: SQLFirewall,
        *,
        client=None,
    ) -> None:
        self._config = config
        self._firewall = firewall
        self._client = client

    @property
    def client(self):
        if self._client is None:
            kwargs = {"region_name": self._config.region} if self._config.region else {}
            self._client = boto3.client("athena", **kwargs)
        return self._client

    def execute(self, sql: str, max_rows: int = 500) -> QueryResult:
        verdict = self._firewall.validate(sql)
        if not verdict.allowed:
            logger.warning("firewall blocked query: %s", verdict.reason)
            return QueryResult(success=False, error=verdict.reason, error_code="blocked")

        start = time.monotonic()
        params: dict = {
            "QueryString": sql,
            "WorkGroup": self._config.workgroup,
        }
        if self._config.output_location:
            params["ResultConfiguration"] = {"OutputLocation": self._config.output_location}
        if self._config.database:
            params["QueryExecutionContext"] = {"Database": self._config.database}

        try:
            query_id = self.client.start_query_execution(**params)["QueryExecutionId"]
        except Exception as e:
            return QueryResult(
                success=False,
                error=str(e),
                error_code="start_failed",
                duration_ms=_ms_since(start),
            )

        state, detail, scanned = self._await_completion(query_id, start)
        if state != "SUCCEEDED":
            return QueryResult(
                success=False,
                error=detail,
                error_code="timeout" if state == "TIMEOUT" else "query_error",
                query_execution_id=query_id,
                bytes_scanned=scanned,
                duration_ms=_ms_since(start),
            )

        columns, rows, truncated = self._fetch(query_id, max_rows)
        duration = _ms_since(start)
        logger.info(
            "athena %s: %d rows in %.0fms%s", query_id, len(rows), duration,
            " (truncated)" if truncated else "",
        )
        return QueryResult(
            success=True,
            columns=columns,
            rows=rows,
            row_count=len(rows),
            truncated=truncated,
            duration_ms=duration,
            query_execution_id=query_id,
            bytes_scanned=scanned,
        )

    def _await_completion(self, query_id: str, start: float) -> tuple[str, str, int]:
        """Poll with backoff. Returns (state, detail, bytes_scanned)."""
        wait = _POLL_INITIAL
        scanned = 0
        while time.monotonic() - start < self._config.timeout_seconds:
            execution = self.client.get_query_execution(QueryExecutionId=query_id)[
                "QueryExecution"
            ]
            scanned = execution.get("Statistics", {}).get("DataScannedInBytes", scanned)
            status = execution["Status"]
            state = status["State"]
            if state == "SUCCEEDED":
                return state, "", scanned
            if state in ("FAILED", "CANCELLED"):
                reason = status.get("StateChangeReason", "unknown error")
                return state, f"query {state}: {reason}", scanned
            time.sleep(wait)
            wait = min(wait * _POLL_BACKOFF, _POLL_MAX)

        # Leaving a query running after we stop caring costs money and holds slots.
        try:
            self.client.stop_query_execution(QueryExecutionId=query_id)
        except Exception as e:
            logger.warning("could not cancel timed-out query %s: %s", query_id, e)
        return "TIMEOUT", f"query timed out after {self._config.timeout_seconds}s", scanned

    def _fetch(self, query_id: str, max_rows: int) -> tuple[list[str], list[list[str]], bool]:
        columns: list[str] = []
        rows: list[list[str]] = []
        # One row beyond the cap distinguishes "exactly max_rows" from "truncated".
        cap = max_rows + 1

        for page_no, page in enumerate(
            self.client.get_paginator("get_query_results").paginate(QueryExecutionId=query_id)
        ):
            result_set = page["ResultSet"]
            if page_no == 0:
                columns = [
                    col.get("Label") or col["Name"]
                    for col in result_set["ResultSetMetadata"]["ColumnInfo"]
                ]
            for row_no, row in enumerate(result_set.get("Rows", [])):
                if page_no == 0 and row_no == 0:
                    continue  # header row
                rows.append([cell.get("VarCharValue", "") for cell in row["Data"]])
                if len(rows) >= cap:
                    break
            if len(rows) >= cap:
                break

        truncated = len(rows) > max_rows
        return columns, rows[:max_rows], truncated


def _ms_since(start: float) -> float:
    return (time.monotonic() - start) * 1000
