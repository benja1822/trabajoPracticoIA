"""Shared fixtures for the test suite.

All fixtures that cross multiple test modules live here.
Heavy fixtures (those loading real models) are marked slow/integration
and excluded from the default run.
"""
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Set
from unittest.mock import MagicMock

import numpy as np
import pytest

# ─── Deterministic RNG ───────────────────────────────────────────────────────
RNG = np.random.default_rng(42)
N_IMAGES = 50
DIM = 512


@pytest.fixture(scope="session")
def mock_embeddings() -> np.ndarray:
    """L2-normalized float32 embeddings — same shape as real CLIP output."""
    raw = RNG.standard_normal((N_IMAGES, DIM)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return (raw / norms).astype(np.float32)


@pytest.fixture(scope="session")
def mock_image_ids() -> List[str]:
    """Fake Pascal VOC-style image IDs."""
    return [f"2007_{i:06d}" for i in range(N_IMAGES)]


@pytest.fixture(scope="session")
def mock_gt(mock_image_ids) -> Dict[str, Set[str]]:
    """Fake ground truth: 5 classes, ~10 images each, some overlap."""
    classes = ["dog", "car", "person", "bird", "cat"]
    rng = np.random.default_rng(99)
    return {
        cls: set(rng.choice(mock_image_ids, size=10, replace=False).tolist())
        for cls in classes
    }


@pytest.fixture
def tmp_artifacts(tmp_path: Path) -> Path:
    """Temporary directory that mimics the project artifacts structure."""
    (tmp_path / "embeddings").mkdir()
    (tmp_path / "index").mkdir()
    (tmp_path / "ground_truth").mkdir()
    (tmp_path / "output").mkdir()
    return tmp_path


@pytest.fixture
def mock_clip_extractor(mock_embeddings, mock_image_ids):
    """CLIPExtractor mock that returns deterministic embeddings without loading CLIP."""
    clip = MagicMock()
    id_to_idx = {iid: i for i, iid in enumerate(mock_image_ids)}

    def encode_text(text: str, normalize: bool = True) -> np.ndarray:
        # Hash the text to pick a deterministic embedding
        idx = hash(text) % len(mock_image_ids)
        return mock_embeddings[idx]

    def encode_images(paths, batch_size=64, normalize=True) -> np.ndarray:
        n = len(paths)
        return mock_embeddings[:n]

    clip.encode_text.side_effect = encode_text
    clip.encode_images.side_effect = encode_images
    return clip
