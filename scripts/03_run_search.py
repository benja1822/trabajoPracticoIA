"""Interactive search test — run a single query through the full pipeline.

Usage:
  python scripts/03_run_search.py --query "perro jugando" [--no-llm]
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import IMAGES_DIR
from src.embeddings.clip_extractor import CLIPExtractor, load_embeddings
from src.indexing.faiss_manager import FAISSManager
from src.reranking.negation_reranker import NegationReranker
from src.agentic.pipeline import AgenticPipeline


def main():
    parser = argparse.ArgumentParser(description="Run a search query through the pipeline")
    parser.add_argument("--query",  type=str, required=True, help="Query in Spanish")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM (translation only)")
    args = parser.parse_args()

    print("Loading components...")
    embeddings, image_ids = load_embeddings()
    clip    = CLIPExtractor()
    faiss   = FAISSManager.load()
    reranker = NegationReranker(clip_extractor=clip, image_embeddings=embeddings, image_ids=image_ids)
    pipeline = AgenticPipeline(
        clip_extractor=clip,
        faiss_manager=faiss,
        reranker=reranker,
        enable_llm=not args.no_llm,
    )

    print(f"\nQuery: {args.query!r}")
    result = pipeline.run(args.query, query_id="interactive")

    print(f"\nTranslated     : {result.translated_query}")
    print(f"Positive query : {result.positive_query}")
    print(f"Negations      : {result.negative_attributes}")
    print(f"Expanded       : {result.expanded_query}")
    print(f"Valid          : {result.is_valid}  ({result.validation_reason})")
    print(f"\nTop-10 results :")
    for i, img_id in enumerate(result.top_results, 1):
        print(f"  {i:2}. {img_id}")

    print(f"\nTrace JSON:\n{result.trace.to_json(indent=2)}")


if __name__ == "__main__":
    main()
