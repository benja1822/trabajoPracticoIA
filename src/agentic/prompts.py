"""Prompt templates for the agentic pipeline.

Keep prompts versioned here so they can be adjusted without touching logic.
All prompts expect JSON output — Phi-3-mini handles this reliably at temp=0.
"""

# ─── Semantic expansion ───────────────────────────────────────────────────────
EXPANSION_SYSTEM = (
    "You are a search query optimizer. Your job is to expand image search queries "
    "with relevant synonyms and rephrasings to improve retrieval. "
    "Always respond with valid JSON only, no explanation."
)

EXPANSION_USER = """Expand this image search query with synonyms and two alternative phrasings.
Query: "{query}"

Respond with this JSON structure (no markdown, no explanation):
{{"expanded": "<original query plus 2-3 inline synonyms>", "variants": ["<variant 1>", "<variant 2>"]}}"""


# ─── Integrity validation ─────────────────────────────────────────────────────
VALIDATION_SYSTEM = (
    "You are a semantic coherence checker for image search queries. "
    "Detect contradictions, impossible requests, and incoherent expansions. "
    "Always respond with valid JSON only."
)

VALIDATION_USER = """Check if this image search query is semantically coherent and suitable for retrieval.
Positive query: "{positive_query}"
Attributes to exclude: {negative_attributes}

Rules:
1. Is the positive query meaningful for image search?
2. Do the negative attributes directly contradict the core subject of the query?
3. Is the query specific enough to retrieve relevant images?

Respond with this JSON (no markdown, no explanation):
{{"valid": true_or_false, "reason": "<one sentence>", "corrected_query": null_or_"<improved query if fixable>"}}"""


# ─── Phi-3-mini chat template helper ─────────────────────────────────────────
def build_phi3_messages(system: str, user: str) -> list:
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
