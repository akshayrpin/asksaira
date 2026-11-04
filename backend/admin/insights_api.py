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


async def build_insights(days: int | None = 28) -> dict:
    """The Overview payload: {kpis, convos_daily, topics, routes, top_questions, gaps}."""
    traces = await obs.read_traces(_window_from(days))
    if traces:
        fb = await obs.read_feedback_map()
        if fb:
            for t in traces:
                if not t.get("feedback"):
                    t["feedback"] = fb.get(t.get("answer_id"), "")
    return I.build(traces)
