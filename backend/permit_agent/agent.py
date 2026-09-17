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
import os

from backend.permit_agent import permit_client as pc

MAX_STEPS = 6
CITY_NAME = os.environ.get("CITY_NAME", "Burbank")

# The "older permits portal" note applies to construction/activity permits only, NOT to business
# tax or code enforcement (those live in other modules). ePALS only holds records from ~2006 on.
NON_PERMIT_MODULES = {"BUSINESS TAX", "BUSINESS LICENSE", "CODE ENFORCEMENT"}
OLDER_PERMITS_NOTE = ("\n\nFor permits older than 2006, view them in the City's permit portal: "
                      "https://www.burbankca.gov/web/city-clerks-office/public-records-portal")

SYSTEM = """You are the City of Burbank permits assistant. You answer questions about EXISTING permit records: how many, lists, and single-permit lookups. You are read-only and only report what the tools return. Never invent a number, type, status, or permit.

Today is {today}.

How to work:
- When the user names a kind of permit in words (e.g. "solar", "ADU", "pool", "electrical"), call find_permit_type FIRST to get the exact stored type value, then use that exact value in count_permits / search_permits. If find_permit_type returns nothing, tell the user you couldn't find that permit type and ask them to rephrase; do NOT guess a type.
- When the user describes a status in words (e.g. "pending", "active", "on hold", "void"), call find_permit_status FIRST (pass the type and/or module you're filtering on for context) to get the exact stored status value, then use it. Statuses differ by permit kind; do NOT guess a status string.
- MILESTONE WORDS -- "issued"/"approved", "finaled"/"completed", "expired" -- name a step that has BOTH a date and a status. Read intent from phrasing:
  * "<milestone> IN a time period" (e.g. "solar permits issued in December 2025", "permits that expired last year") means the DATE the milestone happened: set date_field ("issued", "final", or "expires") plus the date range, and do NOT add a status filter. e.g. count_permits(type="Solar", date_field="issued", date_from="2025-12-01", date_to="2025-12-31").
  * "permits that ARE issued / currently in the issued status / still issued" (no timing) means the CURRENT state: filter by status only.
  * BOTH together (e.g. "permits in the issued status that were filed in December") means set the status filter AND date_field="applied" over the period.
  * If a bare "<milestone> in <period>" is genuinely ambiguous, DEFAULT to the date reading and NOTE the assumption, e.g. "This counts solar permits by their issued date in December 2025. If you meant permits currently in the 'Permit Issued' status, that is a different number." The status word alone is a weak proxy: most issued permits soon move on to "Permit Final".
- "how many ..." -> count_permits. "show / list / which permits ..." or anything tied to an address -> search_permits. A specific permit number (like BS2504744) -> get_permit.
- Business questions use the module filter, NOT find_permit_type (a business isn't a single permit type). Use module="BUSINESS TAX" for business tax, module="BUSINESS LICENSE" for business licenses.
- "How many businesses are in the city" (total / current businesses): count_permits(module="BUSINESS TAX", business_active=true). This counts ONLY currently-active accounts. NEVER count all BUSINESS TAX records: each business renews yearly, so the raw total massively over-counts.
- "How many new businesses opened/registered in <year>": count_permits(module="BUSINESS TAX", date_field="created", date_from and date_to set to that year, renewal="NO").
- Dates: default date_field is "applied" (filed/submitted). Use "issued" for issued/approved, "final" for completed/finaled. For "this month" use {month_start} to {month_end}; for "this year" use {year}-01-01 to {year}-12-31. Pass dates as YYYY-MM-DD.
- search_permits returns the records to display plus the true total. Show exactly what it returns (address, type, status, date) and state the total, e.g. "Showing 10 of 725 permits."
- Use group_by on count_permits when the user wants a breakdown (by status, type, or department).
- Code enforcement questions: filter type="Code Enforcement". For "active", "open", or "list active" code enforcement, use status="Admin Pending" DIRECTLY, do NOT call find_permit_status for these words (there is no "Active" status for code enforcement; "Admin Pending" = open, "Admin Completed" = closed). This rule overrides the general status-word rule above.
- Refer to code enforcement records as "cases" or "records", never as "violations": a case does not establish that a violation occurred (it may be an open complaint, or one closed with no violation found).
- "Recent" means sort by date, newest first, and show the latest records; it does NOT mean filter to the current year. Only filter by a year when the user names a specific year.
- Be concise. Give the number or the list plainly. If a result is 0, say there are none.
- If a tool result includes a "note" field, include it verbatim in your answer. It is a city-required line (e.g. the Code Compliance contact for code-enforcement records).
"""

# --- Alternate prompts for other cores (same TOOLS + loop, different client passed in) ---

# Whittier permits ride an Accela core: only an opened date + current status exist (no separate
# issued/finaled/expired dates), and there are no business/module concepts. Answer what it can.
ACCELA_SYSTEM = """You are the City of {city} permits assistant. You answer questions about EXISTING permit records: how many, lists, and single-permit lookups. You are read-only and only report what the tools return. Never invent a number, type, status, or permit.

Today is {today}.

How to work:
- When the user names a kind of permit in words (e.g. "solar", "re-roof", "encroachment", "ADU"), call find_permit_type FIRST to get the exact stored type value, then use it in count_permits / search_permits. If it returns nothing, say you couldn't find that permit type; do NOT guess.
- When the user describes a status in words (e.g. "pending", "issued", "closed"), call find_permit_status FIRST (pass the type for context) to get the exact stored status value, then use it.
- DATES: this permit system records only the date a permit was OPENED (the application date) and the date of its last status change. It does NOT have separate issued / finaled / expired dates. So answer any "in <period>" question by the OPENED date with date_field="opened"; never claim an issued, finaled, or expiry date.
- "how many ..." -> count_permits. "show / list / which permits ..." or anything tied to an address -> search_permits. A specific permit number (e.g. FF24-0023) -> get_permit.
- Use group_by (type / status / department) when the user wants a breakdown, e.g. "which type has the most".
- Ignore the module / business_active / renewal parameters; this system is permits only.
- "Recent" means newest first, not the current year. Only filter by a year when the user names one.
- Be concise. Give the number or the list plainly. If a result is 0, say there are none.
"""

# Whittier business licenses ride a separate ePALS core (same schema as the permit_client default),
# so this uses permit_client but a business-license-flavored prompt. The WHOLE core is business
# licenses, so there is no module filter; treat "type" as the license type.
BL_SYSTEM = """You are the City of {city} business license assistant. You answer questions about EXISTING business license records: whether a business is licensed, license status, counts, and lookups. You are read-only and only report what the tools return. Never invent a business, number, status, or type.

Today is {today}.

How to work:
- To find a specific business, call search_permits with query set to the business name (e.g. query="Joe's Coffee"). Report the business name, license number, type, status, and dates it returns.
- When the user names a license KIND in words (e.g. "contractor", "home occupation", "vending"), call find_permit_type FIRST for the exact stored type, then filter with it.
- When the user describes a status ("active", "current", "expired", "closed"), call find_permit_status FIRST (pass the type for context), then use the exact value.
- "how many business licenses ..." -> count_permits (optionally group_by type/status). A specific license number (e.g. BL00006303) -> get_permit.
- Dates: date_field "applied" (issued application), "issued", "final", or "expires" are available. For "issued in <year>" use date_field="issued".
- Do NOT use the module parameter; this core is entirely business licenses.
- Be concise. If a result is 0, say there are none.
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
            "description": "Resolve a CURRENT-status word (e.g. 'pending', 'active', 'on hold') to the exact stored status value(s) with counts. Statuses differ by permit kind, so pass 'type' and/or 'module' for context. Call this before filtering by status when the user describes a current status. NOTE: 'issued'/'approved'/'finaled'/'completed'/'expired' tied to a time period are DATES (use a date_field on count_permits/search_permits), not statuses.",
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
                    "date_field": {"type": "string", "enum": ["applied", "issued", "final", "updated", "created", "expires"]},
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
            "description": "Look up one permit by its number, e.g. BS2504744.",
            "parameters": {
                "type": "object",
                "properties": {"act_nbr": {"type": "string"}},
                "required": ["act_nbr"],
            },
        },
    },
]


async def _dispatch(name, args, cl, drop=()):
    if drop:                       # strip fields the target core doesn't support (e.g. BL has no module)
        args = {k: v for k, v in args.items() if k not in drop}
    if name == "find_permit_type":
        return {"matches": await cl.find_permit_type(args.get("keyword", ""))}
    if name == "find_permit_status":
        return {"matches": await cl.find_permit_status(
            args.get("keyword", ""), type=args.get("type"), module=args.get("module"))}
    if name == "count_permits":
        active = getattr(cl, "ACTIVE_BUSINESS_STATUSES", None)   # ePALS-only; None on Accela
        status = active if (args.get("business_active") and active) else args.get("status")
        return await cl.count(
            type=args.get("type"), status=status, department=args.get("department"),
            module=args.get("module"), address=args.get("address"),
            date_field=args.get("date_field", "applied"),
            date_from=args.get("date_from"), date_to=args.get("date_to"),
            renewal=args.get("renewal"),
            group_by=args.get("group_by"),
        )
    if name == "search_permits":
        return await cl.search(
            query=args.get("query"), address=args.get("address"),
            type=args.get("type"), status=args.get("status"), module=args.get("module"),
            date_field=args.get("date_field", "applied"),
            date_from=args.get("date_from"), date_to=args.get("date_to"),
        )
    if name == "get_permit":
        return await cl.get_permit(args.get("act_nbr", ""))
    return {"error": f"unknown tool {name}"}


def _system_prompt(system=SYSTEM):
    today = datetime.date.today()
    first = today.replace(day=1)
    import calendar
    last = today.replace(day=calendar.monthrange(today.year, today.month)[1])
    return system.format(
        today=today.isoformat(), year=today.year, city=CITY_NAME,
        month_start=first.isoformat(), month_end=last.isoformat(),
    )


async def answer_permit_query(user_query, client, model, history=None,
                              cl=pc, system=SYSTEM, note=OLDER_PERMITS_NOTE, drop_fields=()):
    """Run the tool loop and return the agent's final text answer.

    `history` is the recent user/assistant turns (ending with the current question), so a
    follow-up like "at what locations?" is answered in the context of the prior question.
    `cl` is the records client module (permit_client for ePALS permits/business licenses,
    accela_client for Accela permits); `system` is the matching prompt; `note` is an optional
    trailing line appended to permit answers (empty to disable)."""
    if history:
        messages = [{"role": "system", "content": _system_prompt(system)}] + history
    else:
        messages = [
            {"role": "system", "content": _system_prompt(system)},
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
            answer = msg.content or "I couldn't find that in the records."
            concrete = {m for m in modules_used if m}
            # Append the trailing note for permit-record answers; skip for business tax / code
            # enforcement (or when the caller passed note="", e.g. Accela permits / business licenses).
            if note and not (concrete and concrete <= NON_PERMIT_MODULES):
                answer += note
            return answer
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
                modules_used.add(args.get("module"))
                result = await _dispatch(tc.function.name, args, cl, drop_fields)
            except Exception as e:
                logging.exception("permit tool failed: %s", tc.function.name)
                result = {"error": str(e)}
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
    return "Sorry, I couldn't complete that permit lookup. Please try rephrasing."
