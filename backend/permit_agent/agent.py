"""
Read-permits agent.

A tool-calling loop that answers a resident's question about EXISTING permit records
(counts, lists, lookups) by querying the permits index, instead of RAG. It is read-only:
no writes, no auth, no confirmation gate.

Entry point: answer_permit_query(user_query, client, model) -> answer string.
`client` is the app's async AzureOpenAI client; `model` is the deployment name.

The model drives four tools (see TOOLS). The loop runs until the model stops calling
tools and returns prose, or MAX_STEPS is hit.

LYNWOOD MODEL: Lynwood's EdgeSoft core has NO `module` field. The building-vs-planning split
is the `department` field: "CDD - Building and Safety" (most records: building, business
licenses, etc.) vs "CDD - Planning" (zoning/entitlements). There is no business-tax or
code-enforcement module; a business license is a permit TYPE under Building & Safety. So all
scoping here is by `department`, not `module`.
"""

import datetime
import json
import logging

from backend.permit_agent import permit_client as pc

MAX_STEPS = 6

# The two departments the records fall under. A generic permit query defaults to building.
BUILDING_DEPT = "CDD - Building and Safety"
PLANNING_DEPT = "CDD - Planning"

SYSTEM = """You are the City of Lynwood permits assistant. You answer questions about EXISTING permit records: how many, lists, and single-permit lookups. You are read-only and only report what the tools return. Never invent a number, type, status, or permit.

Today is {today}.

Lynwood's records fall under two departments: "CDD - Building and Safety" (building, construction, business licenses, and most activity) and "CDD - Planning" (zoning and planning entitlements). There is no separate business-tax or code-enforcement module; a business license is a permit TYPE under Building & Safety.

How to work:
- A general permit question that names no type, department, or status defaults to BUILDING permits (department "CDD - Building and Safety"). For zoning or planning questions, set department="CDD - Planning".
- When the user names a kind of record in words (e.g. "solar", "ADU", "pool", "electrical", "business license"), call find_permit_type FIRST to get the exact stored type value, then use that exact value in count_permits / search_permits. If find_permit_type returns nothing, tell the user you couldn't find that type and ask them to rephrase; do NOT guess a type.
- When the user describes a status in words (e.g. "pending", "active", "on hold", "void"), call find_permit_status FIRST (pass the type and/or department you're filtering on for context) to get the exact stored status value, then use it. Statuses differ by record kind; do NOT guess a status string.
- MILESTONE WORDS -- "issued"/"approved", "finaled"/"completed", "expired" -- name a step that has BOTH a date and a status. Read intent from phrasing:
  * "<milestone> IN a time period" (e.g. "solar permits issued in December 2025", "permits that expired last year") means the DATE the milestone happened: set date_field ("issued", "final", or "expires") plus the date range, and do NOT add a status filter. e.g. count_permits(type="Solar", date_field="issued", date_from="2025-12-01", date_to="2025-12-31").
  * "permits that ARE issued / currently in the issued status / still issued" (no timing) means the CURRENT state: filter by status only.
  * BOTH together (e.g. "permits in the issued status that were filed in December") means set the status filter AND date_field="applied" over the period.
  * If a bare "<milestone> in <period>" is genuinely ambiguous, DEFAULT to the date reading and NOTE the assumption, e.g. "This counts solar permits by their issued date in December 2025. If you meant permits currently in the 'Permit Issued' status, that is a different number." The status word alone is a weak proxy: most issued permits soon move on to a later status.
- "how many ..." -> count_permits. "show / list / which permits ..." or anything tied to an address -> search_permits. A specific permit number -> get_permit.
- Dates: default date_field is "applied" (filed/submitted). Use "issued" for issued/approved, "final" for completed/finaled. For "this month" use {month_start} to {month_end}; for "this year" use {year}-01-01 to {year}-12-31. Pass dates as YYYY-MM-DD.
- search_permits returns the records to display plus the true total. Show exactly what it returns (address, type, status, date) and state the total, e.g. "Showing 10 of 725 permits."
- Use group_by on count_permits when the user wants a breakdown (by status, type, or department).
- "Recent" means sort by date, newest first, and show the latest records; it does NOT mean filter to the current year. Only filter by a year when the user names a specific year.
- Be concise. Give the number or the list plainly. If a result is 0, say there are none.
- If a tool result includes a "note" field, include it verbatim in your answer.
"""

DEPT_ENUM = [BUILDING_DEPT, PLANNING_DEPT]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "find_permit_type",
            "description": "Resolve a user's word (e.g. 'solar', 'ADU', 'pool', 'business license') to the exact stored permit type value(s) with their counts. Call this before filtering by type.",
            "parameters": {
                "type": "object",
                "properties": {"keyword": {"type": "string"}},
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_permit_status",
            "description": "Resolve a CURRENT-status word (e.g. 'pending', 'active', 'on hold') to the exact stored status value(s), each with its count. Statuses differ by record kind, so pass 'type' and/or 'department' for context. Call this before filtering by status when the user describes a current status. NOTE: 'issued'/'approved'/'finaled'/'completed'/'expired' tied to a time period are DATES (use a date_field on count_permits/search_permits), not statuses.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string"},
                    "type": {"type": "string", "description": "exact permit type for context (from find_permit_type)"},
                    "department": {"type": "string", "enum": DEPT_ENUM},
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_permits",
            "description": "Exact count of permits matching the filters. Optionally group_by status/type/department for a breakdown. Use the exact type value from find_permit_type.",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "description": "exact stored type value"},
                    "status": {"type": "string"},
                    "department": {"type": "string", "enum": DEPT_ENUM,
                                   "description": "'CDD - Building and Safety' for building/construction/business licenses, 'CDD - Planning' for zoning/planning"},
                    "address": {"type": "string", "description": "street address to restrict the count to, e.g. '123 Main St'"},
                    "date_field": {"type": "string", "enum": ["applied", "issued", "final", "updated", "created", "expires"]},
                    "date_from": {"type": "string", "description": "YYYY-MM-DD / YYYY-MM / YYYY"},
                    "date_to": {"type": "string", "description": "YYYY-MM-DD / YYYY-MM / YYYY"},
                    "group_by": {"type": "string", "enum": ["status", "type", "department"]},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_permits",
            "description": "List matching permits (first 12 plus the true total). Use for 'show/list' questions and anything tied to an address.",
            "parameters": {
                "type": "object",
                "properties": {
                    "address": {"type": "string", "description": "street address to match"},
                    "type": {"type": "string", "description": "exact stored type value"},
                    "status": {"type": "string"},
                    "department": {"type": "string", "enum": DEPT_ENUM,
                                   "description": "'CDD - Building and Safety' or 'CDD - Planning'"},
                    "query": {"type": "string", "description": "free-text keywords (applicant name, etc.)"},
                    "date_field": {"type": "string", "enum": ["applied", "issued", "final", "updated", "created", "expires"]},
                    "date_from": {"type": "string"},
                    "date_to": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_permit",
            "description": "Look up one permit by its number.",
            "parameters": {
                "type": "object",
                "properties": {"act_nbr": {"type": "string"}},
                "required": ["act_nbr"],
            },
        },
    },
]


def _building_scope(args):
    """Deterministic default: a general permit query (no permit type, no department, AND no status
    named) is scoped to BUILDING permits = department "CDD - Building and Safety". Returns
    (department, defaulted). If the model set a type (e.g. Solar), a department (Planning), or a
    STATUS, leave the scope to whatever department the model passed (possibly None)."""
    if (not args.get("department") and not args.get("type")
            and not args.get("status")):
        return BUILDING_DEPT, True
    return args.get("department"), False


def _note_building(res, defaulted):
    """When we defaulted to building permits, tell the user (the model includes a result 'note'
    verbatim). Appends to any existing note rather than dropping it."""
    if defaulted and isinstance(res, dict):
        note = ("These figures are for building permits (Building & Safety). For planning/zoning "
                "records or a specific permit type, just ask.")
        res["note"] = f'{res["note"]} {note}' if res.get("note") else note
    return res


async def _dispatch(name, args):
    if name == "find_permit_type":
        return {"matches": await pc.find_permit_type(args.get("keyword", ""))}
    if name == "find_permit_status":
        return {"matches": await pc.find_permit_status(
            args.get("keyword", ""), type=args.get("type"), department=args.get("department"))}
    if name == "count_permits":
        department, defaulted = _building_scope(args)
        res = await pc.count(
            type=args.get("type"), status=args.get("status"), department=department,
            address=args.get("address"),
            date_field=args.get("date_field", "applied"),
            date_from=args.get("date_from"), date_to=args.get("date_to"),
            group_by=args.get("group_by"),
        )
        return _note_building(res, defaulted)
    if name == "search_permits":
        department, defaulted = _building_scope(args)
        res = await pc.search(
            query=args.get("query"), address=args.get("address"),
            type=args.get("type"), status=args.get("status"),
            department=department,
            date_field=args.get("date_field", "applied"),
            date_from=args.get("date_from"), date_to=args.get("date_to"),
        )
        return _note_building(res, defaulted)
    if name == "get_permit":
        return await pc.get_permit(args.get("act_nbr", ""))
    return {"error": f"unknown tool {name}"}


def _system_prompt():
    today = datetime.date.today()
    first = today.replace(day=1)
    import calendar
    last = today.replace(day=calendar.monthrange(today.year, today.month)[1])
    return SYSTEM.format(
        today=today.isoformat(), year=today.year,
        month_start=first.isoformat(), month_end=last.isoformat(),
    )


async def answer_permit_query(user_query, client, model, history=None):
    """Run the tool loop and return the agent's final text answer.

    `history` is the recent user/assistant turns (ending with the current question), so a
    follow-up like "at what locations?" is answered in the context of the prior question.
    """
    if history:
        messages = [{"role": "system", "content": _system_prompt()}] + history
    else:
        messages = [
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": user_query},
        ]
    for _ in range(MAX_STEPS):
        resp = await client.chat.completions.create(
            model=model, messages=messages, tools=TOOLS, temperature=0,
        )
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:
            return msg.content or "I couldn't find that in the permit records."
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
                result = await _dispatch(tc.function.name, args)
            except Exception as e:
                logging.exception("permit tool failed: %s", tc.function.name)
                result = {"error": str(e)}
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
    return "Sorry, I couldn't complete that permit lookup. Please try rephrasing."
