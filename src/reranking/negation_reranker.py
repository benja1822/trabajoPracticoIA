"""Pool-based reranker that penalizes images containing negated attributes.

Score formula:
    final_score(img) = positive_sim(img) - λ × max(negative_sims(img))

Where:
- positive_sim is the FAISS score from the original retrieval
- negative_sims are CLIP cosine similarities between the image and each negated attribute
- λ (lambda) controls penalization strength — see config.RERANK_LAMBDA
"""
from typing import Dict, List, Tuple

import numpy as np

from src.config import RERANK_LAMBDA


class NegationReranker:
    def __init__(
        self,
        clip_extractor,
        image_embeddings: np.ndarray = None,
        image_ids: List[str] = None,
        lam: float = RERANK_LAMBDA,
    ):
        self._clip             = clip_extractor
        self.lam               = lam
        self._image_embeddings = image_embeddings
        self._id_to_idx: Dict[str, int] = (
            {iid: i for i, iid in enumerate(image_ids)} if image_ids else {}
        )
        self._neg_embed_cache: Dict[str, np.ndarray] = {}

    def load_embeddings(self, image_embeddings: np.ndarray, image_ids: List[str]) -> None:
        """Attach the image embedding matrix after construction."""
        self._image_embeddings = image_embeddings
        self._id_to_idx = {iid: i for i, iid in enumerate(image_ids)}

    def _get_neg_embedding(self, attribute: str) -> np.ndarray:
        if attribute not in self._neg_embed_cache:
            self._neg_embed_cache[attribute] = self._clip.encode_text(
                f"a photo with {attribute}"
            )
        return self._neg_embed_cache[attribute]

    def rerank(
        self,
        candidates: List[Tuple[str, float]],
        negative_attributes: List[str],
    ) -> List[Tuple[str, float]]:
        """Re-score candidates by penalizing negative attributes.

        Requires image_embeddings to be set at construction or via load_embeddings().
        Falls back to original order if embeddings are not available.
        """
        if not negative_attributes or self._image_embeddings is None:
            return candidates

        neg_embeds = np.stack([
            self._get_neg_embedding(attr) for attr in negative_attributes
        ])  # (n_neg, 512)

        reranked = []
        for img_id, pos_score in candidates:
            idx = self._id_to_idx.get(img_id)
            if idx is None:
                reranked.append((img_id, pos_score))
                continue

            img_emb  = self._image_embeddings[idx]   # (512,)
            neg_sims = neg_embeds @ img_emb            # (n_neg,)
            penalty  = float(np.max(neg_sims))
            adjusted = pos_score - self.lam * penalty
            reranked.append((img_id, adjusted))

        reranked.sort(key=lambda x: -x[1])
        return reranked
