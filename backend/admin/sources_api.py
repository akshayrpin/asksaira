"""
backend/admin/sources_api.py — the Sources tab's write path: trigger ingestion and poll a job.

This is a thin CLIENT of the shared Azure ingestion service. That service owns the heavy, slow,
network-sensitive work (crawl -> chunk -> contextualize -> embed -> index --incremental) and runs
from a fixed, WAF-allowlisted egress IP. This backend only hands it a job and reports status, so
nothing long-running ever blocks a Quart request.

Service contract (the fixed seam both sides build to; neither imports the other):
    POST {INGEST_SERVICE_URL}/ingest
        headers: Authorization: Bearer {INGEST_SERVICE_TOKEN}
        body:    { "city": str, "kind": "web"|"pdf", "url": str? }
        -> 202   { "job_id": str }
    GET  {INGEST_SERVICE_URL}/ingest/{job_id}
        headers: Authorization: Bearer {INGEST_SERVICE_TOKEN}
        -> 200   { "job_id", "state", "stage", "log_tail", "source"?, "started_at"?, "finished_at"? }
        state in: queued | crawling | chunking | contextualizing | embedding | indexing | done | failed

Security: `city` is injected here from INGEST_CITY (a deployment fact), never taken from the
client, so staff on one city's console can't drive ingestion into another city's index. Until the
service URL/token are set, every call raises IngestNotConfigured and the route degrades to 503,
exactly like insights falls back when traces aren't wired.
"""

from __future__ import annotations

import os
from urllib.parse import urlsplit

import httpx

SERVICE_URL = os.environ.get("INGEST_SERVICE_URL", "").rstrip("/")
SERVICE_TOKEN = os.environ.get("INGEST_SERVICE_TOKEN", "")
# Which city/index this deployment ingests into. The service maps it to PIPE_OUT + CODE_INDEX.
CITY = os.environ.get("INGEST_CITY", "")

VALID_KINDS = {"web", "pdf"}
_START_TIMEOUT = httpx.Timeout(15.0)    # the service only enqueues and returns a job_id, so quick
_STATUS_TIMEOUT = httpx.Timeout(10.0)


class IngestNotConfigured(Exception):
    """INGEST_SERVICE_URL / token not set yet, so the service can't be reached."""


class IngestError(Exception):
    """The service was reachable but the request failed (network, 4xx/5xx, bad body)."""


def is_configured() -> bool:
    return bool(SERVICE_URL and SERVICE_TOKEN)


def _headers() -> dict:
    return {"Authorization": f"Bearer {SERVICE_TOKEN}", "Accept": "application/json"}


def _validate(kind: str, url: str) -> None:
    """Fail fast on obviously bad input before we bother the service. ValueError -> 400 upstream."""
    if kind not in VALID_KINDS:
        raise ValueError(f"kind must be one of {sorted(VALID_KINDS)}")
    if kind == "web":
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise ValueError("url must be a full http(s) URL")
    # kind == "pdf": file transfer (multipart or blob handoff) is a follow-up; the JSON path
    # here carries web sources. A pdf request without a resolved file is rejected by the service.


async def start_ingest(kind: str = "web", url: str = "") -> dict:
    """Enqueue one ingestion job. Returns {'job_id': ...}. Raises ValueError (bad input),
    IngestNotConfigured, or IngestError."""
    kind = (kind or "web").strip()
    url = (url or "").strip()
    _validate(kind, url)
    if not is_configured():
        raise IngestNotConfigured()

    payload = {"city": CITY, "kind": kind}
    if url:
        payload["url"] = url
    try:
        async with httpx.AsyncClient(timeout=_START_TIMEOUT) as client:
            r = await client.post(f"{SERVICE_URL}/ingest", json=payload, headers=_headers())
    except httpx.HTTPError as e:
        raise IngestError(f"could not reach ingestion service: {e}") from e
    if r.status_code not in (200, 201, 202):
        raise IngestError(f"ingestion service returned {r.status_code}: {r.text[:200]}")
    try:
        data = r.json()
    except ValueError as e:
        raise IngestError("ingestion service returned a non-JSON body") from e
    if not data.get("job_id"):
        raise IngestError("ingestion service did not return a job_id")
    return {"job_id": data["job_id"], "state": data.get("state", "queued")}


async def job_status(job_id: str) -> dict:
    """Fetch one job's status. Raises IngestNotConfigured or IngestError."""
    if not is_configured():
        raise IngestNotConfigured()
    jid = (job_id or "").strip()
    if not jid:
        raise ValueError("job_id is required")
    try:
        async with httpx.AsyncClient(timeout=_STATUS_TIMEOUT) as client:
            r = await client.get(f"{SERVICE_URL}/ingest/{jid}", headers=_headers())
    except httpx.HTTPError as e:
        raise IngestError(f"could not reach ingestion service: {e}") from e
    if r.status_code == 404:
        raise IngestError("job not found")
    if r.status_code != 200:
        raise IngestError(f"ingestion service returned {r.status_code}: {r.text[:200]}")
    try:
        return r.json()
    except ValueError as e:
        raise IngestError("ingestion service returned a non-JSON body") from e
