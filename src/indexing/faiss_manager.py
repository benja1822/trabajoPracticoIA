from enum import Enum
from pathlib import Path
from typing import List, Tuple
import json

import faiss
import numpy as np

from src.config import FAISS_INDEX_FILE, IMAGE_IDS_FILE, FAISS_TOP_K


class IndexType(Enum):
    """Strategy selector for the underlying FAISS index structure.

    FLAT_IP  — Exact inner-product search. O(N·D) per query.
               Best for N ≤ ~500k. No training required. Default.

    IVF_FLAT — Approximate search with inverted-file partitioning.
               Faster at scale but requires calling train() before add().
               Use when N > 500k and ~1% recall loss is acceptable.
    """
    FLAT_IP  = "flat_ip"
    IVF_FLAT = "ivf_flat"


class FAISSManager:
    """Manages a FAISS index with the image ID mapping.

    Vectors must be L2-normalized before adding — inner product on unit vectors
    equals cosine similarity.

    Index type is chosen at construction via the IndexType enum (Strategy pattern).
    The search / save / load interface is identical regardless of index type.
    """

    # IVF default: 100 Voronoi cells, probe 10 at search time
    _IVF_NLIST  = 100
    _IVF_NPROBE = 10

    def __init__(self, dim: int = 512, index_type: IndexType = IndexType.FLAT_IP):
        self.dim        = dim
        self.index_type = index_type
        self.index      = self._build_index()
        self._ids: List[str] = []

    # ── Index factory (Strategy) ───────────────────────────────────────────────

    def _build_index(self) -> faiss.Index:
        if self.index_type == IndexType.FLAT_IP:
            return faiss.IndexFlatIP(self.dim)
        if self.index_type == IndexType.IVF_FLAT:
            quantizer = faiss.IndexFlatIP(self.dim)
            idx = faiss.IndexIVFFlat(quantizer, self.dim, self._IVF_NLIST, faiss.METRIC_INNER_PRODUCT)
            idx.nprobe = self._IVF_NPROBE
            return idx
        raise ValueError(f"Unsupported index type: {self.index_type!r}")

    # ── Build ──────────────────────────────────────────────────────────────────

    def add(self, embeddings: np.ndarray, image_ids: List[str]) -> None:
        assert embeddings.shape[0] == len(image_ids), "embeddings / ids length mismatch"
        assert embeddings.dtype == np.float32, "FAISS expects float32"

        # IVFFlat requires explicit training before the first add
        if self.index_type == IndexType.IVF_FLAT and not self.index.is_trained:
            self.index.train(embeddings)

        self.index.add(embeddings)
        self._ids.extend(image_ids)

    # ── Persist ────────────────────────────────────────────────────────────────

    def save(
        self,
        index_file: Path = FAISS_INDEX_FILE,
        ids_file: Path   = IMAGE_IDS_FILE,
    ) -> None:
        faiss.write_index(self.index, str(index_file))
        ids_file.write_text(json.dumps(self._ids), encoding="utf-8")
        print(f"Index saved → {index_file}  ({self.index.ntotal} vectors, type={self.index_type.value})")

    @classmethod
    def load(
        cls,
        index_file: Path = FAISS_INDEX_FILE,
        ids_file: Path   = IMAGE_IDS_FILE,
    ) -> "FAISSManager":
        mgr        = cls.__new__(cls)
        mgr.index  = faiss.read_index(str(index_file))
        mgr._ids   = json.loads(ids_file.read_text(encoding="utf-8"))
        mgr.dim    = mgr.index.d
        # Reconstruct the enum from the loaded index type
        mgr.index_type = (
            IndexType.IVF_FLAT
            if isinstance(mgr.index, faiss.IndexIVFFlat)
            else IndexType.FLAT_IP
        )
        return mgr

    # ── Search ─────────────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = FAISS_TOP_K,
    ) -> List[Tuple[str, float]]:
        """Returns [(img_id, score), ...] sorted by descending similarity."""
        vec = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(vec, k)
        return [
            (self._ids[idx], float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx != -1
        ]

    def search_batch(
        self,
        query_embeddings: np.ndarray,
        k: int = FAISS_TOP_K,
    ) -> List[List[Tuple[str, float]]]:
        vecs = query_embeddings.astype(np.float32)
        scores_batch, indices_batch = self.index.search(vecs, k)
        return [
            [(self._ids[i], float(s)) for s, i in zip(scores, indices) if i != -1]
            for scores, indices in zip(scores_batch, indices_batch)
        ]

    @property
    def n_vectors(self) -> int:
        return self.index.ntotal
