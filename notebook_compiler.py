"""Compile the src/ package into a Kaggle-compatible Jupyter notebook.

Strategy: each module file becomes a code cell. The module's first docstring
(if present) becomes a preceding Markdown cell with a section header.

Usage:
  python notebook_compiler.py [--output TPI_IA_2026P_XX.ipynb]

The generated notebook assumes:
  - Pascal VOC dataset is mounted at /kaggle/input/pascal-voc-2012-dataset/
  - (Optional) src package uploaded as /kaggle/input/tpi-ia-2026-src/
"""
import ast
import argparse
import json
import textwrap
from pathlib import Path

# ─── Module order defines notebook cell order ────────────────────────────────
MANIFEST = [
    # Setup cell (injected, not from a file)
    "__setup__",
    # Config
    "src/config.py",
    # EDA
    "src/eda/dataset_stats.py",
    "src/eda/visualizer.py",
    # Embeddings
    "src/embeddings/clip_extractor.py",
    # Indexing
    "src/indexing/faiss_manager.py",
    # Agentic
    "src/agentic/tracer.py",
    "src/agentic/prompts.py",
    "src/agentic/translator.py",
    "src/agentic/negation_parser.py",
    "src/agentic/llm_client.py",
    "src/agentic/expander.py",
    "src/agentic/validator.py",
    "src/agentic/pipeline.py",
    # Reranking
    "src/reranking/negation_reranker.py",
    # Evaluation
    "src/evaluation/metrics.py",
    "src/evaluation/ground_truth.py",
    "src/evaluation/ablation.py",
    # Submission
    "src/submission/csv_builder.py",
    # Main execution cell
    "__main__",
]

SETUP_CELL = textwrap.dedent("""\
    # ── Instalar dependencias ──────────────────────────────────────────────────
    import subprocess, sys

    deps = [
        "faiss-cpu",
        "transformers>=4.40.0",
        "sentencepiece",
        "sacremoses",
        "langdetect",
        "umap-learn",
        "bitsandbytes",
        "accelerate",
        "lxml",
    ]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + deps, check=False)

    # ── Semilla global ──────────────────────────────────────────────────────────
    import random, numpy as np, torch
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
""")

MAIN_CELL = textwrap.dedent("""\
    # ── Pipeline completo: EDA → Embeddings → Índice → Búsqueda → Evaluación ──

    from src.config import IMAGES_DIR, EMBEDDINGS_FILE, IMAGE_IDS_FILE
    from src.eda.dataset_stats import collect_image_stats, parse_voc_annotations, print_summary
    from src.embeddings.clip_extractor import build_and_save_embeddings, load_embeddings, CLIPExtractor
    from src.indexing.faiss_manager import FAISSManager
    from src.agentic.pipeline import AgenticPipeline
    from src.reranking.negation_reranker import NegationReranker
    from src.evaluation.ground_truth import build_voc_ground_truth, build_query_gt_from_voc
    from src.evaluation.ablation import run_baseline, run_full_pipeline, compare_configurations, print_ablation_table
    from src.submission.csv_builder import SubmissionBuilder
    from src.config import VOC_CLASSES

    # 1. EDA
    stats        = collect_image_stats()
    class_to_ids = parse_voc_annotations()
    print_summary(stats, class_to_ids)

    # 2. Embeddings (skip if already cached)
    if not EMBEDDINGS_FILE.exists():
        embeddings, image_ids = build_and_save_embeddings()
    else:
        embeddings, image_ids = load_embeddings()

    # 3. FAISS index
    mgr = FAISSManager()
    mgr.add(embeddings, image_ids)
    mgr.save()

    # 4. Instanciar componentes — reranker recibe la matriz de embeddings en construcción
    clip     = CLIPExtractor()
    reranker = NegationReranker(clip_extractor=clip, image_embeddings=embeddings, image_ids=image_ids)
    pipeline = AgenticPipeline(clip_extractor=clip, faiss_manager=mgr, reranker=reranker)

    # 5. Evaluación
    voc_gt     = build_voc_ground_truth()
    query_gt   = build_query_gt_from_voc(voc_gt)
    queries_q1_20 = {f"q{i+1}": cls for i, cls in enumerate(VOC_CLASSES)}

    results_a = run_baseline(queries_q1_20, clip, mgr)
    results_c = run_full_pipeline(queries_q1_20, pipeline)
    comparison = compare_configurations(queries_q1_20, query_gt, results_a, results_a, results_c)
    print_ablation_table(comparison)

    # 6. Submission — ver scripts/04_generate_submission.py para las 40 queries completas
""")


def extract_docstring(source: str) -> str:
    try:
        tree = ast.parse(source)
        if (tree.body and isinstance(tree.body[0], ast.Expr)
                and isinstance(tree.body[0].value, ast.Constant)):
            return str(tree.body[0].value.value)
    except SyntaxError:
        pass
    return ""


def make_code_cell(source: str) -> dict:
    lines = source.splitlines(keepends=True)
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


def make_markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def compile_notebook(output_path: str = "TPI_IA_2026P_XX.ipynb") -> None:
    root = Path(__file__).resolve().parent
    cells = []

    for entry in MANIFEST:
        if entry == "__setup__":
            cells.append(make_markdown_cell("## Setup — instalación de dependencias y semilla"))
            cells.append(make_code_cell(SETUP_CELL))
            continue

        if entry == "__main__":
            cells.append(make_markdown_cell("## Ejecución principal del pipeline"))
            cells.append(make_code_cell(MAIN_CELL))
            continue

        file_path = root / entry
        if not file_path.exists():
            print(f"[WARN] Not found, skipping: {entry}")
            continue

        source = file_path.read_text(encoding="utf-8")
        doc    = extract_docstring(source)

        section = entry.replace("src/", "").replace("/", " / ").replace(".py", "")
        md_text = f"## `{section}`\n\n{doc.strip()}" if doc else f"## `{section}`"
        cells.append(make_markdown_cell(md_text))
        cells.append(make_code_cell(source))

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }

    out = Path(output_path)
    out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Notebook compiled → {out}  ({len(cells)} cells)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="TPI_IA_2026P_XX.ipynb")
    args = parser.parse_args()
    compile_notebook(args.output)
