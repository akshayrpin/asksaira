"""
backend/admin/insights_api.py — assembles the Overview/analytics payload for the staff console.

Reads query_traces (+ backfills thumbs feedback from conversations) via backend.observability,
filters to a time window, and runs the pure aggregation in insights_data.build. Returns the same
dict shape the console renders, so the route handler is a thin wrapper over this.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from backend import observability as obs
from backend.admin import insights_data as I

# window token -> days (None = all time). Matches the console's window selector.
WINDOWS = {"7": 7, "28": 28, "90": 90, "all": None}


def _window_from(days):
    if not days:
        return None
    start = datetime.now(timezone.utc).date() - timedelta(days=days - 1)
    return datetime(start.year, start.month, start.day, tzinfo=timezone.utc).isoformat()


def window_days(token) -> int | None:
    """Resolve a window query param to a day count, defaulting to 28 for anything unexpected."""
    return WINDOWS.get(str(token), 28)


def _shift(iso: str, days: int) -> str:
    """The same-length window immediately before the one starting at `iso`."""
    return (datetime.fromisoformat(iso) - timedelta(days=days)).isoformat()


async def _patch_users(payload: dict, days: int | None, ts_from: str | None) -> None:
    """Overwrite the Users tile with the real session count from the conversations container
    (query_traces can't provide it), plus a delta vs. the prior equal window. Best-effort: on any
    failure the tile keeps its trace-derived fallback value."""
    tile = next((k for k in payload.get("kpis", []) if k.get("metric") == "users"), None)
    if tile is None:
        return
    sessions = await obs.count_sessions(ts_from)
    tile["value"] = f"{sessions:,}"
    tile["delta"], tile["tone"] = "", "neutral"
    if ts_from and days:                  # trend only when the window is bounded
        prior = await obs.count_sessions(_shift(ts_from, days), ts_from)
        if prior:
            d = sessions - prior
            tile["tone"] = "good" if d >= 0 else "bad"
            tile["delta"] = f"{d:+,.0f} vs prior {days}d"


async def build_insights(days: int | None = 28) -> dict:
    """The Overview payload: {kpis, convos_daily, topics, routes, top_questions, gaps}."""
    ts_from = _window_from(days)
    traces = await obs.read_traces(ts_from)
    if traces:
        fb = await obs.read_feedback_map()
        if fb:
            for t in traces:
                if not t.get("feedback"):
                    t["feedback"] = fb.get(t.get("answer_id"), "")
    payload = I.build(traces)
    try:
        await _patch_users(payload, days, ts_from)
    except Exception:                     # never let the Users tile break the whole page
        pass
    return payload
