"""backend/zoning.py — address-specific zoning / land-use answers for Burbank.

Two-turn flow that mirrors how the Planning Division actually answers these:
  1. Resident asks whether a use is allowed at a specific property
     ("can I open a medical office at 2019 W Magnolia?"). If no zoning designation has been
     given yet, we ask them to look it up on the zoning map and reply with it.
  2. Once a designation is present (this turn or an earlier one), we answer in Planning's format:
     permitted by-right / CUP / AUP / not permitted, the parking requirement, a note that other
     entitlements may apply, and the Planning contact.

Curated to match Planning's canonical wording. When the allowance for a use in a zone isn't
certain, it defers to Planning rather than guessing a permission status.
"""

import logging
import re

PLANNING_CONTACT = "the Planning Division at (818) 238-5250 or planning@burbankca.gov"

# Burbank zoning designations. Restricted to the hyphenated / multi-char codes that don't collide
# with ordinary English words (residents type these anyway: "C-3", "R-1", "MDC-3"). Bare two-letter
# zones (GO, OS, AD, RC, NB, AP, RR) are intentionally excluded to avoid matching words like "go".
_ZONE_RE = re.compile(
    r"\b(R-?1-?H?|R-?2|R-?3|R-?4|C-?1|C-?2|C-?3|C-?4|M-?1|M-?2|MDM-?1|MDC-?[1-4]|"
    r"BCC-?[1-3]|BCCM|MPC-?[1-3]|RBP)\b", re.I)

DEFLECT = (
    "It looks like your question is about a specific property, and the answer depends on that "
    "property's zoning designation.\n\n"
    "Please go to the City's zoning map, search for the property's address, find its zoning "
    "designation, and reply with the designation (for example, \"C-3\") so I can give you a "
    "specific answer.\n\n"
    f"For more information, please contact {PLANNING_CONTACT}."
)

SYSTEM = f"""You are the City of Burbank Planning Division assistant. The resident has asked whether a specific land use is allowed at a property and has now given the property's zoning designation. Answer that specific question.

Answer format (concise, authoritative, resident-facing, one short paragraph):
1. State the permission status in that zone: "permitted by-right", "allowed with a Conditional Use Permit (CUP)", "allowed with an Administrative Use Permit (AUP)", or "not permitted".
2. If permitted, give the relevant parking requirement when you know it.
3. Note that additional entitlements may be needed (e.g., for late-night hours, change of use, alcohol sales, or exterior changes).
4. End with exactly: For more information, please contact {PLANNING_CONTACT}

Rules you MUST apply:
- Medical and dental offices are PERMITTED BY-RIGHT in Burbank's commercial zones (C-2, C-3, C-4) and most mixed-use / downtown commercial zones. Parking requirement: 5 spaces per 1,000 sq. ft. of adjusted gross floor area.
- General and professional business offices are permitted by-right in the commercial zones; typical office parking is 4 spaces per 1,000 sq. ft.
- Retail stores and personal-service uses are generally permitted by-right in the commercial zones.

If you are NOT confident whether the specific use is permitted in the given zone, do NOT invent a status, a parking ratio, or a code section. Instead say the allowance depends on the specific zoning and use, and direct them to contact {PLANNING_CONTACT} to confirm the use, parking, and any entitlements. Keep it short."""


def find_zone(text):
    """Return the zoning designation mentioned in `text` (normalized, e.g. 'C-3'), or None."""
    if not text:
        return None
    m = _ZONE_RE.search(text)
    if not m:
        return None
    z = m.group(0).upper().replace(" ", "")
    mm = re.match(r"^([A-Z]+)-?(\d.*)$", z)     # normalize "C3" -> "C-3", "MDC3" -> "MDC-3"
    return f"{mm.group(1)}-{mm.group(2)}" if mm else z


def _zone_from_history(history):
    for m in reversed(history or []):
        if m.get("role") == "user":
            z = find_zone(m.get("content", ""))
            if z:
                return z
    return None


async def answer_zoning_query(user_query, client, model, history=None):
    """Two-turn: deflect for the zoning designation if we don't have one yet, otherwise answer."""
    zone = find_zone(user_query) or _zone_from_history(history)
    if not zone:
        return DEFLECT
    convo = [{"role": "system", "content": SYSTEM + f"\n\nThe resident's property is zoned {zone}."}]
    convo += (history or [])[-6:]      # carries the original use question + the designation
    try:
        resp = await client.chat.completions.create(model=model, temperature=0, messages=convo)
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        logging.exception("zoning answer failed")
        return (f"For a property zoned {zone}, please contact {PLANNING_CONTACT} to confirm whether "
                "your use is permitted, the parking requirement, and any entitlements needed.")
