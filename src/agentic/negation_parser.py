import re
from dataclasses import dataclass, field
from typing import List

from src.agentic.tracer import TraceStep

# Patrones de negación en inglés (post-traducción).
# Capturamos exactamente 1 token: el atributo negado.
# Dos palabras generaban capturas ruidosas ("red on", "dogs in").
_NEG_PATTERNS = [
    r"\bnot\s+(\w+)",       # not red
    r"\bwithout\s+(\w+)",   # without background
    r"\bno\s+(\w+)",        # no people
    r"\bexcluding\s+(\w+)", # excluding cats
    r"\bavoiding\s+(\w+)",  # avoiding water
]

# Palabras de parada que no son atributos reales aunque sigan a una negación
_STOP_WORDS = {
    "a", "an", "the", "one", "is", "are", "was", "be", "been",
    "longer", "more", "less", "very", "too", "quite",
}


@dataclass
class NegationResult:
    has_negation: bool
    positive_query: str
    negative_attributes: List[str] = field(default_factory=list)


def parse_negations(translated_query: str) -> tuple[NegationResult, TraceStep]:
    query = translated_query.lower().strip()
    negative_attributes: List[str] = []

    for pattern in _NEG_PATTERNS:
        for match in re.finditer(pattern, query):
            attr = match.group(1).strip()
            if attr not in _STOP_WORDS and len(attr) > 1:
                negative_attributes.append(attr)

    # Construir positive_query eliminando los fragmentos negativos
    positive_query = query
    for pattern in _NEG_PATTERNS:
        positive_query = re.sub(pattern, "", positive_query)
    positive_query = re.sub(r"\s+", " ", positive_query).strip().rstrip(",;")

    # Fallback: si la limpieza dejó algo vacío, usar la query original
    if not positive_query:
        positive_query = translated_query

    result = NegationResult(
        has_negation=bool(negative_attributes),
        positive_query=positive_query,
        negative_attributes=list(dict.fromkeys(negative_attributes)),  # dedup preserve order
    )

    step = TraceStep(
        step="negation_detection",
        result={
            "detected":           result.has_negation,
            "positive_query":     result.positive_query,
            "negative_attributes": result.negative_attributes,
        },
        extra={"input": translated_query},
    )
    return result, step
