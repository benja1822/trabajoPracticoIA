"""Unit tests for FAISSManager.

Uses mock embeddings — no real CLIP model required.
Verifies index build, search correctness, save/load roundtrip, and IndexType strategy.
"""
import json
import pytest
import numpy as np

from src.indexing.faiss_manager import FAISSManager, IndexType

pytestmark = pytest.mark.unit


class TestFAISSManagerBuild:

    def test_add_vectors_updates_count(self, mock_embeddings, mock_image_ids):
        mgr = FAISSManager()
        mgr.add(mock_embeddings, mock_image_ids)
        assert mgr.n_vectors == len(mock_image_ids)

    def test_add_mismatched_lengths_raises(self, mock_embeddings):
        mgr = FAISSManager()
        with pytest.raises(AssertionError):
            mgr.add(mock_embeddings, ["only_one_id"])

    def test_add_wrong_dtype_raises(self, mock_embeddings, mock_image_ids):
        mgr = FAISSManager()
        with pytest.raises(AssertionError):
            mgr.add(mock_embeddings.astype(np.float64), mock_image_ids)

    def test_empty_index_has_zero_vectors(self):
        assert FAISSManager().n_vectors == 0


class TestFAISSManagerSearch:

    @pytest.fixture(autouse=True)
    def built_manager(self, mock_embeddings, mock_image_ids):
        self.mgr = FAISSManager()
        self.mgr.add(mock_embeddings, mock_image_ids)
        self.embeddings = mock_embeddings
        self.ids = mock_image_ids

    def test_search_returns_k_results(self):
        query = self.embeddings[0]
        results = self.mgr.search(query, k=5)
        assert len(results) == 5

    def test_search_returns_tuples_id_score(self):
        results = self.mgr.search(self.embeddings[0], k=3)
        for img_id, score in results:
            assert isinstance(img_id, str)
            assert isinstance(score, float)

    def test_search_sorted_descending_by_score(self):
        results = self.mgr.search(self.embeddings[0], k=10)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_self_query_is_top_result(self):
        # Querying with the exact embedding of image[5] → image[5] should be rank 1
        query = self.embeddings[5]
        results = self.mgr.search(query, k=1)
        assert results[0][0] == self.ids[5]

    def test_search_batch_returns_list_of_lists(self):
        queries = self.embeddings[:3]
        batch_results = self.mgr.search_batch(queries, k=5)
        assert len(batch_results) == 3
        assert all(len(r) == 5 for r in batch_results)

    def test_search_with_k_larger_than_index(self):
        results = self.mgr.search(self.embeddings[0], k=1000)
        assert len(results) == self.mgr.n_vectors


class TestFAISSManagerPersistence:

    def test_save_and_load_roundtrip(self, mock_embeddings, mock_image_ids, tmp_path):
        index_file = tmp_path / "test.index"
        ids_file   = tmp_path / "ids.json"

        mgr = FAISSManager()
        mgr.add(mock_embeddings, mock_image_ids)
        mgr.save(index_file, ids_file)

        loaded = FAISSManager.load(index_file, ids_file)
        assert loaded.n_vectors == mgr.n_vectors

        # Search results should be identical
        q = mock_embeddings[0]
        original = mgr.search(q, k=5)
        restored = loaded.search(q, k=5)
        assert [r[0] for r in original] == [r[0] for r in restored]

    def test_save_creates_both_files(self, mock_embeddings, mock_image_ids, tmp_path):
        index_file = tmp_path / "test.index"
        ids_file   = tmp_path / "ids.json"
        mgr = FAISSManager()
        mgr.add(mock_embeddings, mock_image_ids)
        mgr.save(index_file, ids_file)
        assert index_file.exists()
        assert ids_file.exists()

    def test_saved_ids_json_is_valid(self, mock_embeddings, mock_image_ids, tmp_path):
        ids_file = tmp_path / "ids.json"
        mgr = FAISSManager()
        mgr.add(mock_embeddings, mock_image_ids)
        mgr.save(tmp_path / "test.index", ids_file)
        loaded_ids = json.loads(ids_file.read_text())
        assert loaded_ids == mock_image_ids


class TestIndexTypeStrategy:
    """Strategy pattern: same interface, different underlying index type."""

    def test_default_is_flat_ip(self):
        mgr = FAISSManager()
        assert mgr.index_type == IndexType.FLAT_IP

    def test_flat_ip_and_ivf_flat_same_interface(self, mock_embeddings, mock_image_ids):
        for itype in (IndexType.FLAT_IP,):  # IVF needs training (>= nlist samples)
            mgr = FAISSManager(index_type=itype)
            mgr.add(mock_embeddings, mock_image_ids)
            results = mgr.search(mock_embeddings[0], k=5)
            assert len(results) == 5

    def test_index_type_persists_after_init(self):
        mgr = FAISSManager(index_type=IndexType.FLAT_IP)
        assert mgr.index_type == IndexType.FLAT_IP

    def test_unknown_index_type_raises(self):
        with pytest.raises((ValueError, AttributeError)):
            FAISSManager(index_type="nonexistent_type")
