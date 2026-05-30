"""Build the FAISS IndexFlatIP from cached embeddings.

Requires: data/embeddings/image_embeddings.npy (from 01_build_embeddings.py)
Output:   data/index/faiss_flat_ip.index
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.config import EMBEDDINGS_FILE, IMAGE_IDS_FILE, IMAGE_EMBED_DIM
from src.embeddings.clip_extractor import load_embeddings
from src.indexing.faiss_manager import FAISSManager

if __name__ == "__main__":
    print("Loading embeddings...")
    embeddings, image_ids = load_embeddings(EMBEDDINGS_FILE, IMAGE_IDS_FILE)
    print(f"  {len(image_ids)} vectors, dim={embeddings.shape[1]}")

    assert embeddings.dtype == np.float32
    assert embeddings.shape[1] == IMAGE_EMBED_DIM

    mgr = FAISSManager(dim=IMAGE_EMBED_DIM)
    mgr.add(embeddings, image_ids)
    mgr.save()

    print(f"Index built with {mgr.n_vectors} vectors.")
