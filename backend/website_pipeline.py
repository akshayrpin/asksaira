"""
backend/website_pipeline.py — answer WEBSITE questions with the code pipeline (burbank-code-v1)
instead of Azure on-your-data. Gated by CODE_PIPELINE_ENABLED in app.py; routing is unchanged,
only the website branch calls this.

Hybrid (BM25 + vector kNN) retrieval + the Azure semantic ranker + an in-depth grounded prompt.
Returns (answer, context) where context = {"citations": [...]} shaped exactly like on-your-data,
so the frontend renders citations identically. The answer cites sources as [doc1], [doc2], ...
matching the frontend's citation parser.

Reuses the app's Azure OpenAI client for both the query embedding and the answer; reads the
search service + code index from env (already present in the app settings).
"""

import os
import re
from datetime import date

import aiohttp

# Generic signals that the resident wants the ordinance TEXT (not a service/how-to answer). When
# absent, municipal-code chunks (source_category=="code") are demoted below the city's own pages so
# an ordinance that merely shares vocabulary can't outrank the page that actually answers.
_CODE_INTENT_RE = re.compile(
    r"\b(codes?|ordinances?|municipal code|statutes?|zoning|setbacks?|"
    r"what (?:does|do) the (?:code|law|ordinance))\b", re.I)


def _is_code_query(query):
    return bool(_CODE_INTENT_RE.search(query or ""))


CODE_SLOTS = 2   # non-code queries: at most this many municipal-code chunks kept in the top-k


def _demote_code(chunks, k):
    """Non-code queries: only reorder when municipal code is FLOODING the natural top-k. If code
    already holds <= CODE_SLOTS of the top-k it earned those slots, so leave the ranking untouched
    (code that wasn't top-ranked is never promoted). If it floods, pull the city's own pages up and
    keep at most CODE_SLOTS code chunks, so an ordinance still supports the answer without dominating
    it. So 'how do I report weeds' leads with the 311 page yet still carries a little of the
    ordinance, while a question where code merely ranked 2nd is left exactly as retrieval had it."""
    top = chunks[:k]
    code_in_top = [c for c in top if c.get("source_category") == "code"]
    if len(code_in_top) <= CODE_SLOTS:
        return top
    site = [c for c in chunks if c.get("source_category") != "code"]
    kept = code_in_top[:CODE_SLOTS]
    out = site[:k - len(kept)] + kept
    if len(out) < k:                                  # few website chunks in the pool -> backfill
        seen = {c["id"] for c in out}
        out += [c for c in chunks if c["id"] not in seen][:k - len(out)]
    return out[:k]

SEARCH_SVC = os.environ.get("AZURE_SEARCH_SERVICE", "")
SEARCH_KEY = os.environ.get("AZURE_SEARCH_KEY", "")
CODE_INDEX = os.environ.get("CODE_INDEX", "burbank-code-v1")
SEM_CONFIG = os.environ.get("CODE_SEMANTIC_CONFIG", "sem")
EMBED_MODEL = os.environ.get("AZURE_OPENAI_EMBEDDING_NAME") or "text-embedding-3-large"
API = "2024-07-01"

SYSTEM = (
    "You are the City of Burbank's assistant. Give the resident a thorough, genuinely helpful "
    "answer using ONLY the numbered sources provided. Rules:\n"
    "- Use only facts present in the sources; never invent names, phone numbers, dates, or URLs.\n"
    "- Be thorough and specific: cover the full picture the sources support, relevant conditions, "
    "eligibility, requirements, steps to take, exceptions, fees, deadlines, and what to do next. "
    "Don't stop at a one-line answer when the sources contain more that would help.\n"
    "- Organize a longer answer with short paragraphs or bullet points so it's easy to read.\n"
    "- Cite the sources you use inline as [doc1], [doc2], etc. (matching the source numbers).\n"
    "- Include specific contacts (email/phone) and links whenever they appear in the sources, so "
    "the resident can actually act on the answer.\n"
    "- If the sources include a link to an online FORM the resident would need (application, "
    "request, registration, permit form, etc.), ALWAYS give that form's ACTUAL URL in the answer, "
    "not just its name, so they can open it directly.\n"
    "- When the sources give separate contacts for different audiences or cases (e.g. residential "
    "vs commercial, or by department or property type), include EACH one that applies; do not "
    "substitute one channel (like a form link) for another audience's contact.\n"
    "- For reporting a violation or nuisance, give the reporting contact the SOURCES provide for "
    "that specific issue, and do NOT default it to the Public Works counter: an ordinance may name "
    "Public Works for the abatement itself, but the resident's reporting channel is on the city's "
    "own service page.\n"
    "- If the sources don't fully cover the question, answer what they do cover, say what's "
    "missing, and point to the closest relevant office or page.\n"
    "- Prefer the most recent/current information when sources differ.\n"
    "- Today is {today}. Use this weekday and date exactly as given; never recompute the day "
    "of the week yourself. Sources may describe past events or expired terms. For any "
    "time-bound fact (terms of office, appointments, whether someone 'currently' holds a role, "
    "deadlines, fees), compare its dates against today: if a stated period has already ended, "
    "describe it in the past tense, not as current. Do not assume the most recently named person "
    "still holds a role."
)


async def _retrieve(vector, query, k, candidates):
    body = {
        "search": query,
        "vectorQueries": [{"kind": "vector", "vector": vector, "fields": "vector", "k": candidates}],
        "queryType": "semantic",
        "semanticConfiguration": SEM_CONFIG,
        "top": k,
        "select": "id,content,title,url,breadcrumb,page_type,content_date,source_category",
    }
    url = f"https://{SEARCH_SVC}.search.windows.net/indexes/{CODE_INDEX}/docs/search?api-version={API}"
    async with aiohttp.ClientSession() as s:
        async with s.post(url, headers={"Content-Type": "application/json", "api-key": SEARCH_KEY},
                          json=body) as r:
            data = await r.json(content_type=None)
    return data.get("value", []) or []


async def _page_chunks(page_url):
    """Every chunk of one page (its parent document), for parent-document assembly. Same Azure AI
    Search index as _retrieve, just FILTERED to the page instead of ranked."""
    body = {
        "search": "*",
        "filter": "url eq '{}'".format((page_url or "").replace("'", "''")),
        "top": 60,
        "select": "content,breadcrumb,url,source_category",
    }
    u = f"https://{SEARCH_SVC}.search.windows.net/indexes/{CODE_INDEX}/docs/search?api-version={API}"
    async with aiohttp.ClientSession() as s:
        async with s.post(u, headers={"Content-Type": "application/json", "api-key": SEARCH_KEY},
                          json=body) as r:
            data = await r.json(content_type=None)
    return data.get("value", []) or []


async def answer_website_query(question, client, model, k=8, candidates=50, pool=30,
                               page_char_cap=8000, total_char_cap=32000):
    """Return (answer, context). Retrieve + demote to k hit chunks, then PARENT-DOCUMENT expand:
    reassemble each hit's whole page (its hit chunks first, then the page's other chunks) so an
    answer split across a page's chunks stays complete (e.g. a FAQ's contact line + its how-to
    step). Sources become one block per page, capped so a long page can't blow up the context."""
    emb = await client.embeddings.create(model=EMBED_MODEL, input=[question])
    chunks = await _retrieve(emb.data[0].embedding, question, pool, candidates)
    hits = chunks[:k] if _is_code_query(question) else _demote_code(chunks, k)

    # group the hit chunks by page (source_url), preserving hit-rank order
    pages, hit_by_page = [], {}
    for h in hits:
        u = h.get("url", "")
        if u not in hit_by_page:
            pages.append(u)
            hit_by_page[u] = {"breadcrumb": h.get("breadcrumb") or h.get("title") or "", "hits": []}
        hit_by_page[u]["hits"].append(h)

    blocks, citations, total = [], [], 0
    for u in pages:
        if total >= total_char_cap:
            break
        try:
            siblings = await _page_chunks(u)
        except Exception:
            siblings = []                                  # degrade to just the hit chunks
        seen, parts, used = set(), [], 0
        for c in hit_by_page[u]["hits"] + siblings:        # hit chunks first, then rest of the page
            text = (c.get("content") or "").strip()
            key = text[:200]
            if not text or key in seen:
                continue
            if parts and used + len(text) > page_char_cap:
                break
            seen.add(key); parts.append(text); used += len(text)
        page_text = "\n\n".join(parts)
        bc = hit_by_page[u]["breadcrumb"]
        blocks.append(f"[doc{len(blocks) + 1}] {bc or u}  ({u})\n{page_text}")
        citations.append({"content": page_text[:2000], "title": bc, "url": u,
                          "filepath": bc or u, "chunk_id": str(len(citations))})
        total += used

    sources = "\n\n".join(blocks)
    resp = await client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "system", "content": SYSTEM.format(today=date.today().strftime("%A, %B %d, %Y (%Y-%m-%d)"))},
                  {"role": "user", "content": f"Question: {question}\n\nSources:\n{sources}"}])
    answer = (resp.choices[0].message.content or "").strip()

    context = {"citations": citations}
    return answer, context
