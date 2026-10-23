"""backend/zoning.py — address-specific zoning / land-use answers.

A land-use question ("can I open a medical office at 2019 W Magnolia?") is answered from the
municipal-code index via the website pipeline, but with a zoning-specific prompt:
  - If the conversation hasn't stated the property's zoning designation, ask for it (with the
    Planning contact) instead of answering.
  - If a designation is present, answer strictly from the retrieved zoning-code sources.

No hardcoded zone list or use rules: the model decides whether a designation was given, and the
answer is grounded in the actual indexed code. Gated by ZONING_ROUTE_ENABLED in app.py.
"""
import os

from backend import planning, website_pipeline

CITY_NAME = os.environ.get("CITY_NAME", "Burbank")
PLANNING_CONTACT = os.environ.get(
    "PLANNING_CONTACT", "the Planning Division at (818) 238-5250 or planning@burbankca.gov")
ZONING_MAP_URL = os.environ.get(
    "ZONING_MAP_URL",
    "https://experience.arcgis.com/experience/00653afd849744eab0ca3547d66db78a/page/PIM")


def _ask_for_zone():
    """Deterministic 'I need the zone' reply, shaped like website_pipeline's (answer, context, id)."""
    msg = (
        f"To tell you what's allowed at a specific property, I need its zoning designation (like C-3 "
        f"or R-1). I can't determine that from the address alone. Please look the property up on the "
        f"City's zoning map at {ZONING_MAP_URL}, then reply with its zone and I'll check the rules "
        f"against the municipal code. For help, contact {PLANNING_CONTACT}.")
    return msg, {"citations": []}, None

ZONING_SYSTEM = (
    f"You are the City of {CITY_NAME}'s Planning assistant answering a question about what may be "
    "done at a specific property (a land-use / zoning question). The answer depends on the "
    "property's ZONING DESIGNATION (for example C-3, R-1, MDC-3).\n"
    "- If the conversation does NOT state the property's zoning designation, do NOT try to answer. "
    "Ask the resident to look the property up on the City's zoning map at " + ZONING_MAP_URL + ", "
    "find its zoning designation, and reply with it (for example \"C-3\"). Include that map link "
    "in your reply. Also give this contact: " + PLANNING_CONTACT + ".\n"
    "- If a zoning designation IS given, answer the land-use question using ONLY the numbered sources "
    "below (the zoning code): state whether the use is permitted by-right, allowed with a Conditional "
    "Use Permit (CUP) or an Administrative Use Permit (AUP), or not permitted in that zone; include "
    "the parking requirement if the sources give one; note that additional entitlements may be "
    "required (late-night hours, change of use, alcohol sales, exterior changes); cite sources as "
    "[doc1], [doc2]; and end with: For more information, contact " + PLANNING_CONTACT + ".\n"
    "Never invent a permission status, a parking ratio, or a code section not in the sources. "
    "Today is {today}."
)


async def answer_zoning_query(question, client, model, has_designation=False):
    """Answer a specific-property land-use question. `has_designation` is decided by the LLM router
    (domain 'zoning-answer' vs 'zoning'), not by pattern-matching here.

    GATE: with NO designation we don't retrieve at all -- we return a fixed GIS deflect. This stops
    the route from pulling an address-matching decision letter or Economic Development page and
    answering from it. Only once a designation is present do we retrieve and answer from the code.
    Returns website_pipeline's (answer, context, answer_id)."""
    if not has_designation:
        return _ask_for_zone()
    # A zoning answer is a planning answer: ground it in the SAME restricted source set (municipal
    # code + Planning document library) via the planning route, but keep the zoning prompt.
    return await planning.answer_planning_query(question, client, model, system=ZONING_SYSTEM, route="zoning")
