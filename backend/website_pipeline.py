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
from datetime import date

import aiohttp

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
    "- For any Public Works topic (streets, sidewalks, sewers/wastewater, trash/recycling, "
    "temporary bins or dumpsters, traffic), ALWAYS include the Public Works counter email "
    "pwonlinecounter@burbankca.gov, in addition to any more specific Public Works contact that "
    "appears in the sources.\n"
    "- If the sources don't fully cover the question, answer what they do cover, say what's "
    "missing, and point to the closest relevant office or page.\n"
    "- Prefer the most recent/current information when sources differ.\n"
    "- Today's date is {today}. Sources may describe past events or expired terms. For any "
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
        "select": "id,content,title,url,breadcrumb,page_type,content_date",
    }
    url = f"https://{SEARCH_SVC}.search.windows.net/indexes/{CODE_INDEX}/docs/search?api-version={API}"
    async with aiohttp.ClientSession() as s:
        async with s.post(url, headers={"Content-Type": "application/json", "api-key": SEARCH_KEY},
                          json=body) as r:
            data = await r.json(content_type=None)
    return data.get("value", []) or []


async def answer_website_query(question, client, model, k=8, candidates=50):
    """Return (answer, context). context = {"citations":[...]} for the frontend."""
    emb = await client.embeddings.create(model=EMBED_MODEL, input=[question])
    chunks = await _retrieve(emb.data[0].embedding, question, k, candidates)

    sources = "\n\n".join(
        f"[doc{i + 1}] {c.get('title', '')}  ({c.get('url', '')})\n{c.get('content', '')}"
        for i, c in enumerate(chunks)
    )
    resp = await client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "system", "content": SYSTEM.format(today=date.today().isoformat())},
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
