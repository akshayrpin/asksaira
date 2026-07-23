"""
Read-permits agent.

A tool-calling loop that answers a resident's question about EXISTING permit records
(counts, lists, lookups) by querying the permits index, instead of RAG. It is read-only:
no writes, no auth, no confirmation gate.

Entry point: answer_permit_query(user_query, client, model) -> answer string.
`client` is the app's async AzureOpenAI client; `model` is the deployment name.

The model drives four tools (see TOOLS). The loop runs until the model stops calling
tools and returns prose, or MAX_STEPS is hit.
"""

import datetime
import json
import logging

from backend.permit_agent import permit_client as pc

MAX_STEPS = 6

# The "older permits portal" note applies to construction/activity permits only, NOT to business
# tax or code enforcement (those live in other modules). ePALS only holds records from ~2006 on.
NON_PERMIT_MODULES = {"BUSINESS TAX", "BUSINESS LICENSE", "CODE ENFORCEMENT"}
OLDER_PERMITS_NOTE = ("\n\nFor permits older than 2006, view them in the City's permit portal: "
                      "https://www.burbankca.gov/web/city-clerks-office/public-records-portal")

SYSTEM = """You are the City of Burbank permits assistant. You answer questions about EXISTING permit records: how many, lists, and single-permit lookups. You are read-only and only report what the tools return. Never invent a number, type, status, or permit.

Today is {today}.

How to work:
- When the user names a kind of permit in words (e.g. "solar", "ADU", "pool", "electrical"), call find_permit_type FIRST to get the exact stored type value, then use that exact value in count_permits / search_permits. If find_permit_type returns nothing, tell the user you couldn't find that permit type and ask them to rephrase; do NOT guess a type.
- When the user describes a status in words (e.g. "pending", "active", "expired", "issued"), call find_permit_status FIRST (pass the type and/or module you're filtering on for context) to get the exact stored status value, then use it. Statuses differ by permit kind; do NOT guess a status string.
- "how many ..." -> count_permits. "show / list / which permits ..." or anything tied to an address -> search_permits. A specific permit number (like BS2504744) -> get_permit.
- Business questions use the module filter, NOT find_permit_type (a business isn't a single permit type). Use module="BUSINESS TAX" for business tax, module="BUSINESS LICENSE" for business licenses.
- "How many businesses are in the city" (total / current businesses): count_permits(module="BUSINESS TAX", business_active=true). This counts ONLY currently-active accounts. NEVER count all BUSINESS TAX records: each business renews yearly, so the raw total massively over-counts.
- "How many new businesses opened/registered in <year>": count_permits(module="BUSINESS TAX", date_field="created", date_from and date_to set to that year, renewal="NO").
- Dates: default date_field is "applied" (filed/submitted). Use "issued" for issued/approved, "final" for completed/finaled. For "this month" use {month_start} to {month_end}; for "this year" use {year}-01-01 to {year}-12-31. Pass dates as YYYY-MM-DD.
- search_permits returns the records to display plus the true total. Show exactly what it returns (address, type, status, date) and state the total, e.g. "Showing 10 of 725 permits."
- Use group_by on count_permits when the user wants a breakdown (by status, type, or department).
- Code enforcement questions: filter type="Code Enforcement" (call find_permit_type("code enforcement") to get the exact value). For "active" or "open" code enforcement, add status="Admin Pending" ("Admin Completed" means the case is closed).
- "Recent" means sort by date, newest first, and show the latest records; it does NOT mean filter to the current year. Only filter by a year when the user names a specific year.
- Be concise. Give the number or the list plainly. If a result is 0, say there are none.
- If a tool result includes a "note" field, include it verbatim in your answer. It is a city-required line (e.g. the Code Compliance contact for code-enforcement records).
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "find_permit_type",
            "description": "Resolve a user's word (e.g. 'solar', 'ADU', 'pool') to the exact stored permit type value(s) with their counts. Call this before filtering by type.",
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
            "description": "Resolve a status word (e.g. 'pending', 'active', 'expired', 'issued', 'final') to the exact stored status value(s) with counts. Statuses differ by permit kind, so pass 'type' and/or 'module' for context. Call this before filtering by status when the user describes a status in words.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string"},
                    "type": {"type": "string", "description": "exact permit type for context (from find_permit_type)"},
                    "module": {"type": "string", "enum": ["BUSINESS TAX", "BUSINESS LICENSE", "BUILDING", "PUBLIC WORKS", "PLANNING", "PARKING", "HOUSING"]},
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
                    "department": {"type": "string"},
                    "module": {"type": "string", "enum": ["BUSINESS TAX", "BUSINESS LICENSE", "BUILDING", "PUBLIC WORKS", "PLANNING", "PARKING", "HOUSING"],
                               "description": "high-level category; use 'BUSINESS TAX' for new businesses, 'BUSINESS LICENSE' for business licenses"},
                    "address": {"type": "string", "description": "street address to restrict the count to, e.g. '123 Main St'"},
                    "date_field": {"type": "string", "enum": ["applied", "issued", "final", "updated", "created"]},
                    "date_from": {"type": "string", "description": "YYYY-MM-DD / YYYY-MM / YYYY"},
                    "date_to": {"type": "string", "description": "YYYY-MM-DD / YYYY-MM / YYYY"},
                    "group_by": {"type": "string", "enum": ["status", "type", "department"]},
                    "renewal": {"type": "string", "enum": ["NO", "YES"],
                                "description": "BUSINESS TAX only: 'NO' = new/original account, 'YES' = a renewal"},
                    "business_active": {"type": "boolean",
                                        "description": "BUSINESS TAX only: true = count ONLY currently-active businesses (Paid/Current or Pending Renewal). Use for 'how many businesses are in the city'."},
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
                    "module": {"type": "string", "enum": ["BUSINESS TAX", "BUSINESS LICENSE", "BUILDING", "PUBLIC WORKS", "PLANNING", "PARKING", "HOUSING"],
                               "description": "high-level category; 'BUSINESS TAX' for businesses, 'BUSINESS LICENSE' for business licenses"},
                    "query": {"type": "string", "description": "free-text keywords (applicant name, etc.)"},
                    "date_field": {"type": "string", "enum": ["applied", "issued", "final", "updated", "created"]},
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
            "description": "Look up one permit by its number, e.g. BS2504744.",
            "parameters": {
                "type": "object",
                "properties": {"act_nbr": {"type": "string"}},
                "required": ["act_nbr"],
            },
        },
    },
]


async def _dispatch(name, args):
    if name == "find_permit_type":
        return {"matches": await pc.find_permit_type(args.get("keyword", ""))}
    if name == "find_permit_status":
        return {"matches": await pc.find_permit_status(
            args.get("keyword", ""), type=args.get("type"), module=args.get("module"))}
    if name == "count_permits":
        status = pc.ACTIVE_BUSINESS_STATUSES if args.get("business_active") else args.get("status")
        return await pc.count(
            type=args.get("type"), status=status, department=args.get("department"),
            module=args.get("module"), address=args.get("address"),
            date_field=args.get("date_field", "applied"),
            date_from=args.get("date_from"), date_to=args.get("date_to"),
            renewal=args.get("renewal"),
            group_by=args.get("group_by"),
        )
    if name == "search_permits":
        return await pc.search(
            query=args.get("query"), address=args.get("address"),
            type=args.get("type"), status=args.get("status"), module=args.get("module"),
            date_field=args.get("date_field", "applied"),
            date_from=args.get("date_from"), date_to=args.get("date_to"),
        )
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
    modules_used = set()  # module filters the agent used, to scope the older-permits note
    for _ in range(MAX_STEPS):
        resp = await client.chat.completions.create(
            model=model, messages=messages, tools=TOOLS, temperature=0,
        )
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:
            answer = msg.content or "I couldn't find that in the permit records."
            concrete = {m for m in modules_used if m}
            # Append the note for permit-record answers; skip it for business tax / code enforcement.
            if not (concrete and concrete <= NON_PERMIT_MODULES):
                answer += OLDER_PERMITS_NOTE
            return answer
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
                modules_used.add(args.get("module"))
                result = await _dispatch(tc.function.name, args)
            except Exception as e:
                logging.exception("permit tool failed: %s", tc.function.name)
                result = {"error": str(e)}
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
    return "Sorry, I couldn't complete that permit lookup. Please try rephrasing."
