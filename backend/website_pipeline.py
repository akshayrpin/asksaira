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
    "- For any Public Works topic (streets, sidewalks, sewers/wastewater, trash/recycling, "
    "temporary bins or dumpsters, traffic), ALWAYS include the Public Works counter email "
    "pwonlinecounter@burbankca.gov, in addition to any more specific Public Works contact that "
    "appears in the sources.\n"
    "- For reporting a violation or nuisance, give the reporting contact the SOURCES provide for "
    "that specific issue, and do NOT default it to the Public Works counter: an ordinance may name "
    "Public Works for the abatement itself, but the resident's reporting channel is on the city's "
    "own service page.\n"
    "- If the sources don't fully cover the question, answer what they do cover, say what's "
    "missing, and point to the closest relevant office or page.\n"
    "- Answer the SPECIFIC thing the resident asked. If the sources only support a related or "
    "narrower case than the question (they asked about X, the sources describe Y), do NOT answer "
    "'yes' to X: state plainly what the sources do and don't support and name the distinction, "
    "instead of collapsing X and Y. If the exact thing asked isn't addressed by the sources, say "
    "so rather than inferring a yes.\n"
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


async def answer_website_query(question, client, model, k=8, candidates=50, pool=30):
    """Return (answer, context). context = {"citations":[...]} for the frontend."""
    emb = await client.embeddings.create(model=EMBED_MODEL, input=[question])
    chunks = await _retrieve(emb.data[0].embedding, question, pool, candidates)
    if not _is_code_query(question):
        # Demote municipal-code chunks below the city's own pages, keeping rerank order within each
        # group, so an ordinance that merely shares vocabulary ("weeds") can't outrank the
        # service/how-to page that actually answers. Code still ranks normally on code questions.
        chunks = ([c for c in chunks if c.get("source_category") != "code"]
                  + [c for c in chunks if c.get("source_category") == "code"])
    chunks = chunks[:k]

    sources = "\n\n".join(
        f"[doc{i + 1}] {c.get('breadcrumb') or c.get('title', '')}  ({c.get('url', '')})\n{c.get('content', '')}"
        for i, c in enumerate(chunks)
    )
    resp = await client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "system", "content": SYSTEM.format(today=date.today().strftime("%A, %B %d, %Y (%Y-%m-%d)"))},
                  {"role": "user", "content": f"Question: {question}\n\nSources:\n{sources}"}])
    answer = (resp.choices[0].message.content or "").strip()

    context = {"citations": [{
        "content": c.get("content", ""),
        "title": c.get("title", ""),
        "url": c.get("url", ""),
        "filepath": c.get("title", "") or c.get("url", ""),
        "chunk_id": str(i),
    } for i, c in enumerate(chunks)]}
    return answer, context
