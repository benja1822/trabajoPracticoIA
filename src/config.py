from pathlib import Path
import torch

# ─── Reproducibilidad ────────────────────────────────────────────────────────
SEED = 42

# ─── Paths base ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Kaggle monta los datasets en /kaggle/input; localmente usamos data/
_KAGGLE_VOC = Path("/kaggle/input/pascal-voc-2012-dataset/VOC2012_train_val/VOC2012_train_val")
_LOCAL_VOC  = PROJECT_ROOT / "data" / "pascal-voc-2012-dataset" / "VOC2012_train_val" / "VOC2012_train_val"

VOC_ROOT        = _KAGGLE_VOC if _KAGGLE_VOC.exists() else _LOCAL_VOC
IMAGES_DIR      = VOC_ROOT / "JPEGImages"
ANNOTATIONS_DIR = VOC_ROOT / "Annotations"
SETS_DIR        = VOC_ROOT / "ImageSets" / "Main"

# Artefactos generados durante la ejecución
ARTIFACTS_DIR      = PROJECT_ROOT / "data"
EMBEDDINGS_DIR     = ARTIFACTS_DIR / "embeddings"
INDEX_DIR          = ARTIFACTS_DIR / "index"
GROUND_TRUTH_DIR   = ARTIFACTS_DIR / "ground_truth"
TRACES_DIR         = ARTIFACTS_DIR / "traces"
OUTPUT_DIR         = ARTIFACTS_DIR / "output"

for _d in (EMBEDDINGS_DIR, INDEX_DIR, GROUND_TRUTH_DIR, TRACES_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

EMBEDDINGS_FILE  = EMBEDDINGS_DIR / "image_embeddings.npy"
IMAGE_IDS_FILE   = EMBEDDINGS_DIR / "image_ids.json"
FAISS_INDEX_FILE = INDEX_DIR / "faiss_flat_ip.index"
VOC_GT_FILE      = GROUND_TRUTH_DIR / "voc_gt.json"
COMPLEX_GT_FILE  = GROUND_TRUTH_DIR / "complex_gt.json"
TRACES_FILE      = TRACES_DIR / "pipeline_traces.jsonl"
SUBMISSION_FILE  = OUTPUT_DIR / "submission.csv"

# ─── Hardware ─────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── CLIP ────────────────────────────────────────────────────────────────────
CLIP_MODEL_NAME  = "openai/clip-vit-base-patch32"
CLIP_BATCH_SIZE  = 64     # reducir a 32 si hay OOM en CPU
IMAGE_EMBED_DIM  = 512

# ─── FAISS ───────────────────────────────────────────────────────────────────
FAISS_TOP_K      = 10     # resultados finales a devolver
FAISS_POOL_SIZE  = 50     # candidatos pre-reranking

# ─── Traducción (MarianMT) ───────────────────────────────────────────────────
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-es-en"

# ─── LLM agéntico ────────────────────────────────────────────────────────────
LLM_MODEL_PRIMARY  = "microsoft/Phi-3-mini-4k-instruct"
LLM_MODEL_FALLBACK = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LLM_MAX_NEW_TOKENS = 256
LLM_TEMP_EXPANSION = 0.3   # cierta diversidad para variantes
LLM_TEMP_REASONING = 0.0   # determinista para validación/detección

# ─── Reranking ────────────────────────────────────────────────────────────────
RERANK_LAMBDA = 0.5   # peso de la penalización negativa

# ─── EDA / Clustering ────────────────────────────────────────────────────────
KMEANS_N_CLUSTERS = 20
TSNE_PERPLEXITY   = 30
UMAP_N_NEIGHBORS  = 15
UMAP_MIN_DIST     = 0.1
EDA_SAMPLE_SIZE   = 2000   # imágenes para t-SNE/UMAP (subsample por velocidad)

# ─── 20 clases VOC (en el orden oficial del dataset) ─────────────────────────
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
