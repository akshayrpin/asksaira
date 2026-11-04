"""
backend/admin/insights_data.py — pure aggregation for the staff console (Overview).

Given a flat list of query_traces (the same documents backend/observability.py writes and
analysis/dashboard.py reads), compute the panels the Insights page draws:
  kpis, convos_daily, topics, routes, top_questions, gaps.

No Streamlit, no Cosmos, no I/O — just dict/list in, dict out — so it unit-tests against a
plain list (e.g. the CSV export) and the UI layer stays a thin renderer. Each trace is one
resident question (what observability.py logs per answered query).

A trace is expected to look like:
  {ts, route, question, answer, feedback, chunks_retrieved | retrieved[],
   top_rerank_score, latency_ms, ...}
Missing/variant fields are tolerated (chunks_retrieved falls back to len(retrieved)).
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from statistics import median

# Reranker (Azure semantic) scores run ~0..4; map to a 0..100 "confidence" for display.
RERANK_MAX = 4.0

# An answer that is really a non-answer. Kept in sync with the trace-review heuristic.
_APOLOGY = re.compile(
    r"\b(i'?m sorry|i am sorry|i (?:do not|don'?t) have|couldn'?t find|could not find|"
    r"no information|unable to|i can'?t find|not able to|please clarify|doesn'?t include a question)\b",
    re.I,
)

# Keyword -> topic buckets for municipal questions. First match wins; order matters
# (more specific before generic). Deliberately simple and inspectable: topics are a
# derived heuristic, there is no topic field in the data.
TOPIC_RULES: list[tuple[str, str]] = [
    (r"permit|inspection|build|construction|contractor|plan check", "Permits & building"),
    (r"setback|zoning|\bzone\b|variance|\br-?1\b|\bc-?3\b|lot line|yard|parcel|"
     r"general plan|subdivi|conditional use|\bcup\b|accessory dwelling|\badu\b", "Zoning & planning"),
    (r"park|meter|citation|ticket|tow|vehicle|street sweep", "Parking & vehicles"),
    (r"trash|garbage|recycl|bulky|refuse|dumpster|waste", "Trash & recycling"),
    (r"water|sewer|utility|utilities|power|electric|bill|outage", "Utilities"),
    (r"business|license|tax|vendor|home occupation", "Business & licensing"),
    (r"event|calendar|meeting|council|commission|agenda|holiday", "Events & meetings"),
    (r"dog|cat|animal|shelter|pet|leash|coyote", "Animals"),
    (r"police|arrest|crime|safety|fire|emergency|911", "Public safety"),
    (r"job|employ|career|hiring|volunteer", "Jobs & volunteering"),
    (r"pay|fee|cost|how much|fine|refund", "Fees & payments"),
    (r"hour|open|closed|location|address|where|phone|contact|email", "Hours & contact"),
]


def _to_float(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _chunks(t: dict) -> int:
    n = t.get("chunks_retrieved")
    if n is None:
        n = len(t.get("retrieved") or [])
    try:
        return int(n)
    except (TypeError, ValueError):
        return 0


def _rerank(t: dict):
    """Top reranker score for a trace, from the flat field or the nested retrieval list."""
    v = _to_float(t.get("top_rerank_score"))
    if v is not None:
        return v
    top = t.get("retrieved") or []
    if top:
        return _to_float((top[0] or {}).get("rerank_score"))
    return None


def _feedback(t: dict):
    """Collapse feedback to 'positive' / 'negative' / None."""
    v = (t.get("feedback") or t.get("_fb") or "").strip().lower()
    if v in ("", "neutral", "none"):
        return None
    return "positive" if v == "positive" else "negative"


def _answered(t: dict) -> bool:
    """Did the assistant actually answer (vs. apologize / ask to rephrase / empty)?"""
    a = (t.get("answer") or "").strip()
    if len(a) < 15:
        return False
    return not _APOLOGY.search(a)


def _day(ts) -> str | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).astimezone(timezone.utc).date().isoformat()
    except Exception:
        return str(ts)[:10] or None


def _norm_q(q: str) -> str:
    return re.sub(r"\s+", " ", (q or "").strip().lower()).rstrip("?.! ")


def topic_of(question: str) -> str:
    q = (question or "").lower()
    for pat, name in TOPIC_RULES:
        if re.search(pat, q):
            return name
    return "Other"


# --------------------------------------------------------------------------- panels

def kpis(traces: list[dict]) -> list[dict]:
    """Five headline tiles: value + delta vs. the previous equal-length window.

    Delta compares the most-recent half of the traces to the prior half when a split is
    possible; otherwise delta is blank (not enough history to trend)."""
    def _metrics(rows):
        n = len(rows)
        answered = sum(1 for t in rows if _answered(t))
        grounded = sum(1 for t in rows if _chunks(t) > 0)
        lat = [v for v in (_to_float(t.get("latency_ms")) for t in rows) if v is not None]
        fbs = [f for f in (_feedback(t) for t in rows) if f]
        pos = sum(1 for f in fbs if f == "positive")
        return {
            "conversations": n,
            "resolution": (answered / n) if n else 0.0,
            "response": (median(lat) / 1000.0) if lat else None,
            "grounded": (grounded / n) if n else 0.0,
            "csat": (pos / len(fbs)) if fbs else None,
            "_rated": len(fbs),
        }

    rows = sorted(traces, key=lambda t: str(t.get("ts") or ""))
    cur = _metrics(rows)
    half = len(rows) // 2
    prev = _metrics(rows[:half]) if half >= 5 else None
    recent = _metrics(rows[half:]) if half >= 5 else None

    def delta(key, fmt, good_up=True):
        if not prev or not recent or prev.get(key) is None or recent.get(key) is None:
            return "", "neutral"
        d = recent[key] - prev[key]
        if abs(d) < 1e-9:
            return "no change", "neutral"
        up = d > 0
        tone = ("good" if up == good_up else "bad")
        return (("▲ " if up else "▼ ") + fmt(abs(d))), tone

    pct = lambda x: f"{x*100:.0f}%"
    out = []

    def tile(metric, value, delta_key=None, delta_fmt=pct, good_up=True):
        dtxt, tone = delta(delta_key, delta_fmt, good_up) if delta_key else ("", "neutral")
        out.append({"metric": metric, "value": value, "delta": dtxt, "tone": tone})

    tile("conversations", f"{cur['conversations']:,}", "conversations", lambda x: f"{x:+,.0f}")
    tile("resolution", pct(cur["resolution"]), "resolution")
    tile("response", f"{cur['response']:.1f}s" if cur["response"] is not None else "—",
         "response", lambda x: f"{x:.1f}s", good_up=False)
    tile("grounded", pct(cur["grounded"]), "grounded")
    tile("csat", (pct(cur["csat"]) + f" ({cur['_rated']})") if cur["csat"] is not None else f"— ({cur['_rated']})",
         "csat")
    return out


def convos_daily(traces: list[dict]) -> list[dict]:
    """Per-day conversation count and answered-without-apology count."""
    conv = Counter()
    res = Counter()
    for t in traces:
        d = _day(t.get("ts"))
        if not d:
            continue
        conv[d] += 1
        if _answered(t):
            res[d] += 1
    return [{"day": d, "conversations": conv[d], "resolved": res[d]} for d in sorted(conv)]


def topics(traces: list[dict]) -> list[dict]:
    c = Counter(topic_of(t.get("question", "")) for t in traces)
    rows = [{"topic": k, "questions": v} for k, v in c.most_common()]
    return rows


def routes(traces: list[dict]) -> list[dict]:
    c = Counter((t.get("route") or "unknown") for t in traces)
    return [{"route": k, "conversations": v} for k, v in c.most_common()]


def top_questions(traces: list[dict], limit: int = 8) -> list[dict]:
    groups: dict[str, dict] = defaultdict(lambda: {"asks": 0, "routes": Counter(), "display": "", "pos": 0, "rated": 0})
    for t in traces:
        key = _norm_q(t.get("question", ""))
        if not key:
            continue
        g = groups[key]
        g["asks"] += 1
        g["routes"][t.get("route") or "unknown"] += 1
        if not g["display"]:
            g["display"] = (t.get("question") or "").strip()
        fb = _feedback(t)
        if fb:
            g["rated"] += 1
            g["pos"] += 1 if fb == "positive" else 0
    rows = []
    for g in sorted(groups.values(), key=lambda x: x["asks"], reverse=True)[:limit]:
        rows.append({
            "question": g["display"],
            "asks": g["asks"],
            "route": g["routes"].most_common(1)[0][0],
            "csat": round(100 * g["pos"] / g["rated"]) if g["rated"] else "",
        })
    return rows


def gaps(traces: list[dict], limit: int = 8, min_asks: int = 1) -> list[dict]:
    """High-volume, low-confidence questions: the pages to add or fix.

    Confidence = median top-reranker score scaled to 0..100. A question is a gap when it is
    answered with an apology, marked unhelpful, or retrieved weakly. Ranked by asks x (how
    far below full confidence)."""
    groups: dict[str, dict] = defaultdict(
        lambda: {"asks": 0, "display": "", "reranks": [], "apology": 0, "neg": 0})
    for t in traces:
        key = _norm_q(t.get("question", ""))
        if not key:
            continue
        g = groups[key]
        g["asks"] += 1
        if not g["display"]:
            g["display"] = (t.get("question") or "").strip()
        rr = _rerank(t)
        if rr is not None:
            g["reranks"].append(rr)
        if not _answered(t):
            g["apology"] += 1
        if _feedback(t) == "negative":
            g["neg"] += 1

    scored = []
    for g in groups.values():
        if g["asks"] < min_asks:
            continue
        conf = round(100 * min(1.0, median(g["reranks"]) / RERANK_MAX)) if g["reranks"] else None
        apology = g["apology"] > 0
        neg = g["neg"] > 0
        weak = conf is not None and conf < 60
        if not (apology or neg or weak):
            continue
        if apology:
            issue, eff_conf = "No answer found", (conf if conf is not None else 15)
        elif neg:
            issue, eff_conf = "Marked unhelpful", (conf if conf is not None else 35)
        else:
            issue, eff_conf = "Weak source match", conf
        scored.append({
            "question": g["display"],
            "asks": g["asks"],
            "issue": issue,
            "confidence": eff_conf,
            "_rank": g["asks"] * (100 - eff_conf),
        })
    scored.sort(key=lambda x: x["_rank"], reverse=True)
    for s in scored:
        s.pop("_rank", None)
    return scored[:limit]


def build(traces: list[dict]) -> dict:
    """All panels at once — the single payload the Insights page renders."""
    return {
        "kpis": kpis(traces),
        "convos_daily": convos_daily(traces),
        "topics": topics(traces),
        "routes": routes(traces),
        "top_questions": top_questions(traces),
        "gaps": gaps(traces),
    }
