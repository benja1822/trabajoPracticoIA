"""Generate the Kaggle submission CSV for all 40 queries.

Queries q1–q20 are the 20 VOC classes (English, as registered in config.VOC_CLASSES).
Queries q21–q40 are custom — edit CUSTOM_QUERIES below once the cátedra publishes them.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import VOC_CLASSES, SUBMISSION_FILE
from src.embeddings.clip_extractor import CLIPExtractor, load_embeddings
from src.indexing.faiss_manager import FAISSManager
from src.reranking.negation_reranker import NegationReranker
from src.agentic.pipeline import AgenticPipeline
from src.submission.csv_builder import SubmissionBuilder

# ─── Custom queries q21–q40 (update when published by cátedra) ───────────────
# These are placeholders. Replace with actual query strings.
CUSTOM_QUERIES = {
    "q21": "perro en el jardín",
    "q22": "auto no rojo",
    "q23": "persona con bicicleta",
    "q24": "pájaro volando",
    "q25": "barco en el mar",
    "q26": "gato durmiendo",
    "q27": "niño jugando",
    "q28": "árbol con flores",
    "q29": "comida en la mesa",
    "q30": "transporte público",
    "q31": "animal de granja",
    "q32": "persona no adulta",
    "q33": "vehículo de motor",
    "q34": "animal no perro",
    "q35": "objeto en interior",
    "q36": "escena al aire libre",
    "q37": "grupo de personas",
    "q38": "animal doméstico",
    "q39": "edificio urbano",
    "q40": "escena con agua",
}


def main():
    print("Loading components...")
    embeddings, image_ids = load_embeddings()
    clip     = CLIPExtractor()
    faiss    = FAISSManager.load()
    reranker = NegationReranker(clip_extractor=clip, image_embeddings=embeddings, image_ids=image_ids)
    pipeline = AgenticPipeline(
        clip_extractor=clip,
        faiss_manager=faiss,
        reranker=reranker,
        enable_llm=True,
    )

    builder = SubmissionBuilder()

    # q1–q20: VOC classes in English
    print("\nProcessing q1–q20 (VOC classes)...")
    for i, cls in enumerate(VOC_CLASSES, start=1):
        qid = f"q{i}"
        result = pipeline.run(cls, query_id=qid)
        builder.add_query(qid, result.top_results)
        print(f"  {qid}: {cls} → {result.top_results[:3]}...")

    # q21–q40: custom queries
    print("\nProcessing q21–q40 (custom queries)...")
    for qid, query in CUSTOM_QUERIES.items():
        result = pipeline.run(query, query_id=qid)
        builder.add_query(qid, result.top_results)
        print(f"  {qid}: {query!r} → {result.top_results[:3]}...")

    errors = builder.validate()
    if errors:
        print("\n[ERRORS]")
        for e in errors:
            print(f"  {e}")
    else:
        print("\nValidation passed.")

    builder.save(SUBMISSION_FILE)


if __name__ == "__main__":
    main()
