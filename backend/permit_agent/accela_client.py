"""
Read-only client for an Accela permit Solr core (Whittier: sairaaccela).

Same tool surface as permit_client (count / search / find_permit_type / find_permit_status /
get_permit / distinct_values) so the read-permit AGENT is unchanged, but the Accela schema differs
from ePALS, so field names are remapped here:

    ePALS (permit_client)      Accela (this core)
    -------------------------  -----------------------------
    type                       typetext   (human-readable; `type` is the coarse base)
    status                     status_text
    applied_date               opened_date
    issued/final/exp_date      (none — Accela has no milestone dates, only status_date)
    module                     department
    act_nbr (permit number)    custom_id  (human, e.g. FF24-0023); id = WHITTIER-ENG24-...
    valuation_calculated       job_value / total_job_cost

Because there are no issued/final/expiry dates, milestone-date queries collapse to the opened date
(the only "when did this happen" the core has). Counts (numFound) + breakdowns (facets) work
identically. Generic helpers are imported from permit_client to avoid divergence.
"""
import json
import os

import aiohttp

from backend.permit_agent.permit_client import (  # field-agnostic helpers, shared verbatim
    _esc, _solr_dt, _match_values, _has_droppable_suffix, _SUFFIX_NOTE, _ALIASES,
)

# Own env var (NOT PERMITS_API_BASE) so the permit route (Accela) and the BL route (ePALS
# permit_client, which uses PERMITS_API_BASE) can point at different cores in the same process.
API_BASE = os.environ.get(
    "ACCELA_API_BASE", "http://whittier.edgesoftinc.com:7337/solr/sairaaccela/query")
TIMEOUT = 25

# --- Accela field map (logical name -> actual field) ---
F_TYPE = "typetext"          # the human type residents search ("Residential Solar")
F_STATUS = "status_text"
F_DEPT = "department"
F_ADDR = "address"
F_OPENED = "opened_date"
F_STATUS_DT = "status_date"
F_NUM = "custom_id"          # human permit number (FF24-0023)

_SUMMARY = [F_NUM, F_TYPE, F_STATUS, F_DEPT, F_ADDR, F_OPENED, "total_fee", "description"]
_DETAIL = _SUMMARY + ["type", "sub_type", F_STATUS_DT, "job_value", "total_job_cost",
                      "total_pay", "parcel_number", "full_name", "id"]

# Only opened + status-change dates exist. Milestone words (issued/final/expires) have no field, so
# they fall back to the opened date — the closest "when" this core supports (answer what it can).
_DATE_FIELDS = {
    "opened": F_OPENED, "applied": F_OPENED, "filed": F_OPENED, "submitted": F_OPENED,
    "status": F_STATUS_DT, "updated": F_STATUS_DT,
}
_FACET_FIELDS = {F_TYPE, F_STATUS, F_DEPT, "sub_type", "type"}


def _date_field(name):
    return _DATE_FIELDS.get((name or "opened").lower().replace("_date", "").strip(), F_OPENED)


def _fqs(type=None, status=None, department=None, address=None,
         date_field="opened", date_from=None, date_to=None, drop_suffix=False, **_ignore):
    """Filter clauses for the Accela schema. Unknown kwargs (module, renewal, exclude_types) are
    accepted and ignored so the shared agent can call this with the same signature."""
    out = []
    if type:
        out.append(("fq", f'{F_TYPE}:"{type}"'))
    if status:
        if isinstance(status, (list, tuple)):
            out.append(("fq", f"{F_STATUS}:(" + " OR ".join(f'"{s}"' for s in status) + ")"))
        else:
            out.append(("fq", f'{F_STATUS}:"{status}"'))
    if department:
        out.append(("fq", f'{F_DEPT}:"{department}"'))
    if address:
        raw = [x for x in str(address).upper().split() if x]
        if drop_suffix and _has_droppable_suffix(address):
            raw = raw[:-1]
        toks = [t for t in (_esc(x) for x in raw) if t]
        for i, t in enumerate(toks):
            if i == 0 and t.isdigit():
                out.append(("fq", f"{F_ADDR}:{t}\\ *"))
            else:
                out.append(("fq", f"{F_ADDR}:(*\\ {t}\\ * OR *\\ {t})"))
    if date_from or date_to:
        f = _date_field(date_field)
        out.append(("fq", f"{f}:[{_solr_dt(date_from)} TO {_solr_dt(date_to, end=True)}]"))
    return out


async def _query(params, facet=None):
    qp = list(params) + [("wt", "json")]
    if facet is not None:
        qp.append(("json.facet", json.dumps(facet)))
    async with aiohttp.ClientSession() as s:
        async with s.get(API_BASE, params=qp, timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as r:
            r.raise_for_status()
            return await r.json(content_type=None)


def _pick(doc, fields):
    return {k: doc[k] for k in fields if k in doc and doc[k] not in (None, "", " ")}


_types_cache = {"list": None}


async def _all_types():
    if _types_cache["list"] is None:
        _types_cache["list"] = [(b["value"], b["count"]) for b in await distinct_values(F_TYPE, 1000)]
    return _types_cache["list"]


async def find_permit_type(keyword):
    return _match_values(keyword, await _all_types(), _ALIASES)


async def _resolve_type(t):
    if not t:
        return t
    if t in {v for v, _ in await _all_types()}:
        return t
    m = await find_permit_type(t)
    return m[0]["value"] if len(m) == 1 else t


async def _statuses_in(type=None, department=None):
    filt = {}
    if type:
        filt["type"] = type
    if department:
        filt["department"] = department
    facet = {"g": {"type": "terms", "field": F_STATUS, "limit": 100, "sort": "count"}}
    data = await _query([("q", "*"), ("rows", "0")] + _fqs(**filt), facet)
    return [(b["val"], b["count"]) for b in data.get("facets", {}).get("g", {}).get("buckets", [])]


async def find_permit_status(keyword, type=None, module=None):
    t = await _resolve_type(type) if type else None
    return _match_values(keyword, await _statuses_in(type=t))


async def _resolve_status(s, type=None):
    if not s:
        return s
    statuses = await _statuses_in(type=type)
    if s in {v for v, _ in statuses}:
        return s
    m = _match_values(s, statuses)
    return m[0]["value"] if len(m) == 1 else s


async def count(group_by=None, **filters):
    if filters.get("type"):
        filters["type"] = await _resolve_type(filters["type"])
    if filters.get("status") and not isinstance(filters["status"], (list, tuple)):
        filters["status"] = await _resolve_status(filters["status"], type=filters.get("type"))
    facet = None
    if group_by:
        gf = group_by if group_by in _FACET_FIELDS else F_TYPE
        facet = {"g": {"type": "terms", "field": gf, "limit": 50, "sort": "count"}}
    data = await _query([("q", "*"), ("rows", "0")] + _fqs(**filters), facet)
    if data["response"]["numFound"] == 0 and _has_droppable_suffix(filters.get("address")):
        data = await _query([("q", "*"), ("rows", "0")] + _fqs(drop_suffix=True, **filters), facet)
    out = {"count": data["response"]["numFound"]}
    buckets = data.get("facets", {}).get("g", {}).get("buckets")
    if buckets is not None:
        out["breakdown"] = [{"value": b["val"], "count": b["count"]} for b in buckets]
    return out


async def search(query=None, **filters):
    if filters.get("type"):
        filters["type"] = await _resolve_type(filters["type"])
    if filters.get("status") and not isinstance(filters["status"], (list, tuple)):
        filters["status"] = await _resolve_status(filters["status"], type=filters.get("type"))
    if query:
        toks = [_esc(t) for t in str(query).split() if _esc(t)]
        q = "_text_:(" + " AND ".join(toks) + ")" if toks else "*"
    else:
        q = "*"
    base = [("q", q), ("rows", "15"), ("sort", f"{F_OPENED} desc")]
    data = await _query(base + _fqs(**filters))
    broadened = False
    if data["response"]["numFound"] == 0 and _has_droppable_suffix(filters.get("address")):
        data = await _query(base + _fqs(drop_suffix=True, **filters))
        broadened = data["response"]["numFound"] > 0
    resp = data["response"]
    total = resp["numFound"]
    docs = resp["docs"] if total < 15 else resp["docs"][:10]
    out = {"total": total, "shown": len(docs), "results": [_pick(d, _SUMMARY) for d in docs]}
    if broadened:
        out["note"] = _SUFFIX_NOTE
    return out


async def get_permit(act_nbr):
    nbr = str(act_nbr).strip()
    data = await _query([("q", f"_text_:{_esc(nbr)}"), ("rows", "25")])
    docs = data["response"]["docs"]
    up = nbr.upper()
    exact = [d for d in docs
             if up in (str(d.get(F_NUM, "")).upper(), str(d.get("id", "")).upper())]
    chosen = exact or docs
    if not chosen:
        return {"found": False}
    return {"found": True, "exact": bool(exact), "permit": _pick(chosen[0], _DETAIL)}


async def distinct_values(field, limit=50):
    gf = field if field in _FACET_FIELDS else F_TYPE
    facet = {"g": {"type": "terms", "field": gf, "limit": int(limit), "sort": "count"}}
    data = await _query([("q", "*"), ("rows", "0")], facet)
    return [{"value": b["val"], "count": b["count"]}
            for b in data.get("facets", {}).get("g", {}).get("buckets", [])]
