"""backend/planning.py — Planning-department questions, answered ONLY from authoritative sources.

A planning question (development standards -- setbacks, lot coverage, height, FAR; the General Plan;
Housing Element / opportunity sites; specific & master plans; density bonus; inclusionary housing;
historic preservation; CUP / entitlements; hillside & sign rules; ADU standards) is answered from
the municipal code (eCode) PLUS the City's Planning document library and NOTHING ELSE. Retrieval is
hard-restricted to source_category='code' OR the exact set of Planning-library PDFs, so project
decision letters, SB-35 comment matrices, and Economic Development pages can never leak in. If the
restricted sources don't cover it, we say so and point to the Planning Division rather than reach
for another source.

Gated by PLANNING_ROUTE_ENABLED in app.py.
"""
import os

from backend import website_pipeline

CITY_NAME = os.environ.get("CITY_NAME", "Burbank")
PLANNING_CONTACT = os.environ.get(
    "PLANNING_CONTACT", "the Planning Division at (818) 238-5250 or planning@burbankca.gov")

# The Planning document library PDFs, by indexed source_url (from
# https://www.burbankca.gov/web/community-development/document-library). Retrieval for a planning
# question is restricted to the municipal code OR exactly these documents. If these are re-ingested
# with new version params, regenerate this list from pipeline/burbank_doclib/index_manifest.json.
PLANNING_SOURCE_URLS = [
    "https://www.burbankca.gov/documents/173607/0/Adopted+Burbank+Housing+Element.pdf/ceb53074-c32e-c9ba-b7aa-edb473fe6d31?t=1688086065412&download=false",
    "https://www.burbankca.gov/documents/173607/0/Burbank+Center+Plan.pdf/5bb74fe2-250b-7b21-7b8e-d9174668ca53?t=1612453815040",
    "https://www.burbankca.gov/documents/173607/0/HistoricPreservationPlan.pdf/874e1979-a0b6-abac-88d2-c8aeb036ae9d?t=1612304426623",
    "https://www.burbankca.gov/documents/173607/0/InclusionaryHousingRegulation.pdf/459a713c-5c97-8cb5-6c53-2589dbd462ec?t=1616024579955",
    "https://www.burbankca.gov/documents/173607/0/Media+District+Specific+Plan.pdf/458f28af-9c2d-c4dc-1669-df19a6f52856?t=1612453794603",
    "https://www.burbankca.gov/documents/173607/0/North+San+Fernando+Blvd+Master+Plan.pdf/4f76eeb0-670a-9a71-d92e-8ef380e23ad5?t=1612453877511",
    "https://www.burbankca.gov/documents/173607/240353/20200324-citywide-historic-context-report-001.pdf/6878e423-40f3-6b03-45e5-f0b8ce9e4daf?t=1616618394192",
    "https://www.burbankca.gov/documents/173607/240353/20200324-complete-streets-plan.pdf/9d19867a-2d23-aed5-275e-f15fc2304810?t=1616622101495",
    "https://www.burbankca.gov/documents/173607/240353/20200324-density-bonus-regulations-001.pdf/175d1052-4d8e-67da-0029-171c65eeb518?t=1616618254114",
    "https://www.burbankca.gov/documents/d/community-development/20240702-csods-2nd-amendment-adopted02272024-001-pdf?download=false",
    "https://www.burbankca.gov/documents/d/community-development/20240709-transportationstudyguidelinesupdate-001-pdf?download=false",
    "https://www.burbankca.gov/documents/d/community-development/2025-affordable-rents-and-incomes-chart-1-pdf?download=false",
    "https://www.burbankca.gov/documents/d/community-development/adu-handout-updated-4-10-2024-pdf?download=false",
    "https://www.burbankca.gov/documents/d/community-development/burbank2035_updated-002-?download=false",
    "https://www.burbankca.gov/documents/d/community-development/ggrp-update-final_w-app?download=false",
]


def _planning_filter():
    quoted = "|".join(u.replace("'", "''") for u in PLANNING_SOURCE_URLS)
    return "source_category eq 'code' or search.in(url, '{}', '|')".format(quoted)


# Azure $filter: municipal code OR one of the Planning-library docs. Everything else is excluded.
PLANNING_FILTER = _planning_filter()

PLANNING_SYSTEM = (
    f"You are the City of {CITY_NAME}'s Planning assistant answering a land-use, zoning, or "
    "development-standards question. Answer using ONLY the numbered sources below -- the City's "
    "municipal/zoning code and official Planning documents (General Plan, Housing Element, specific "
    "and master plans, development standards, density bonus, inclusionary housing, historic "
    "preservation, ADU rules).\n"
    "- Ground every statement in these sources and cite the code section or document as [doc1], "
    "[doc2], etc. Prefer the municipal code for development standards (setbacks, lot coverage, "
    "height); use the planning documents for plan- and program-level questions (Housing Element "
    "opportunity sites, specific plans, and the like).\n"
    "- Never invent a standard, number, fee, or code section. If the sources do NOT cover the "
    "question, say you don't have that specific detail and refer the resident to " + PLANNING_CONTACT
    + ". Do NOT guess or pull from unrelated material.\n"
    "- For a determination tied to a specific property, note that the property's zoning designation "
    "governs, and give " + PLANNING_CONTACT + " for confirmation.\n"
    "Today is {today}."
)


async def answer_planning_query(question, client, model, system=None):
    """Answer a planning question from the restricted source set (municipal code + the Planning
    document library only). `system` lets the zoning route reuse this retrieval with its own prompt.
    Returns website_pipeline's (answer, context, answer_id), or None when the restricted filter
    matched NOTHING -- a likely misroute, so the caller falls through to the full website pipeline."""
    return await website_pipeline.answer_website_query(
        question, client, model, system=system or PLANNING_SYSTEM,
        retrieval_query=question, retrieval_filter=PLANNING_FILTER, require_hits=True)
