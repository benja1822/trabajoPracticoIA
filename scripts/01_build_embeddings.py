"""Generate and cache CLIP embeddings for all Pascal VOC 2012 images.

Run once. Takes ~8 min on Kaggle T4, ~15 min on CPU.
Output:
  data/embeddings/image_embeddings.npy  (float32, L2-normalized)
  data/embeddings/image_ids.json
"""
import random
import numpy as np
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import SEED, IMAGES_DIR, EMBEDDINGS_FILE, IMAGE_IDS_FILE, CLIP_BATCH_SIZE
from src.embeddings.clip_extractor import build_and_save_embeddings

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if __name__ == "__main__":
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"Images directory not found: {IMAGES_DIR}\n"
            "Set VOC_ROOT in src/config.py or mount the dataset."
        )

    embeddings, image_ids = build_and_save_embeddings(
        images_dir=IMAGES_DIR,
        embeddings_file=EMBEDDINGS_FILE,
        ids_file=IMAGE_IDS_FILE,
        batch_size=CLIP_BATCH_SIZE,
    )
    print(f"Done. Shape: {embeddings.shape}  |  {len(image_ids)} image IDs")
