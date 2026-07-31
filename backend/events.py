"""
backend/events.py — live City events lookup.

Events are time-sensitive, so they are NOT baked into the RAG index (it would go stale and
can't reason about "today"). Instead we read a JSON of events (produced/refreshed by the ingest
`calendar_fetch.py` Playwright pull), keep only what's upcoming as of today, and let the model
answer the resident's question from that current list. Routed via the classifier's `events`
domain in app.py.

events.json path: EVENTS_JSON env var, else <app>/data/events.json.
"""

import datetime
import json
import os
import re

try:
    from zoneinfo import ZoneInfo
    PT = ZoneInfo("America/Los_Angeles")
except Exception:
    PT = None

EVENTS_JSON = os.environ.get("EVENTS_JSON") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "events.json")

SYSTEM = (
    "You are the City of Burbank's assistant answering about the City calendar, both upcoming "
    "events AND City Council / board / commission MEETINGS, using ONLY the list provided (it is "
    "current as of the given date). Answer with the relevant items, their dates/times, locations, "
    "and links. If an entry's title is marked 'Dark' or 'Canceled', that meeting is NOT being held "
    "(a recess or cancellation), do not present it as happening; give the next actual occurrence "
    "instead. Be specific and helpful; if nothing matches what they asked, say what is coming up "
    "soon. Never invent events, meetings, or details."
)


def _load():
    try:
        d = json.load(open(EVENTS_JSON, encoding="utf-8"))
    except Exception:
        return []
    if isinstance(d, dict):
        d = d.get("events") or next((v for v in d.values() if isinstance(v, list)), [])
    return d if isinstance(d, list) else []


def available():
    return bool(_load())


def _parse(dt):
    try:
        return datetime.datetime.fromisoformat(dt)
    except Exception:
        return None


def _local(st):
    """UTC -> Pacific, so dates/times are right (a Tue 6pm PT meeting is stored as Wed UTC)."""
    if st and st.tzinfo and PT is not None:
        try:
            return st.astimezone(PT)
        except Exception:
            return st
    return st


def _strip_html(s):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s or "").replace("&nbsp;", " ")).strip()


def upcoming(today, days=60, limit=40):
    out = []
    for e in _load():
        st = _local(_parse(e.get("start")))
        if st and today <= st.date() <= today + datetime.timedelta(days=days):
            out.append((st, e))
    out.sort(key=lambda x: x[0])
    return [e for _, e in out[:limit]]


def _fmt(e):
    st = _local(_parse(e.get("start")))
    when = st.strftime("%a %b %d %Y, %I:%M %p") if st else (e.get("start") or "")
    parts = [f"- {e.get('title', '').strip()} — {when}"]
    if e.get("location"):
        parts.append(f"at {e['location']}")
    desc = _strip_html(e.get("description"))
    if desc:
        parts.append(f"({desc[:160]})")
    if e.get("url"):
        parts.append(e["url"])
    return " ".join(parts)


async def answer_events_query(question, client, model, today):
    """Answer an events question from the current upcoming list, or None if nothing is upcoming."""
    evs = upcoming(today)
    if not evs:
        return None
    listing = "\n".join(_fmt(e) for e in evs)
    resp = await client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user",
                   "content": f"Today is {today.isoformat()}.\nQuestion: {question}\n\n"
                              f"Upcoming City events:\n{listing}"}])
    return (resp.choices[0].message.content or "").strip()
