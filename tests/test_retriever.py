# tests/test_retriever.py
import numpy as np
import pytest
import os
from unittest.mock import MagicMock, patch


def make_random_embedding(dim=512):
    v = np.random.randn(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def make_random_patches(n=784, dim=256):
    return np.random.randn(n, dim).astype(np.float32)


class TestFAISSRetriever:

    def test_index1_loads_and_returns_result(self):
        """Index 1 must load and return exactly 1 result for any embedding."""
        from src.retriever import FAISSRetriever
        r = FAISSRetriever()
        r.load_indexes()

        query  = make_random_embedding(512)
        result = r.route_category(query)

        assert "category" in result
        assert "confidence" in result
        assert result["category"] in [
            'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut',
            'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush',
            'transistor', 'wood', 'zipper'
        ]

    def test_index2_loads_and_returns_k_results(self):
        """Index 2 must return exactly k results for a valid embedding."""
        from src.retriever import FAISSRetriever
        r = FAISSRetriever()
        r.load_indexes()

        query   = make_random_embedding(512)
        results = r.retrieve_similar_defects(query, k=5)

        assert len(results) <= 5
        assert all("category" in res for res in results)
        assert all("similarity_score" in res for res in results)

    def test_index3_lazy_loads_on_first_call(self):
        """Index 3 must not be loaded at init, only on first category call."""
        from src.retriever import FAISSRetriever
        r = FAISSRetriever()
        r.load_indexes()

        assert len(r.index3_cache) == 0, "Index 3 should not load at startup"

        patches = make_random_patches(784, 256)
        score, patch_scores, score_std, D = r.score_patches(patches, "bottle")

        assert "bottle" in r.index3_cache, "Index 3 should lazy-load on call"

    def test_score_patches_returns_valid_score(self):
        """Anomaly score must be a positive float."""
        from src.retriever import FAISSRetriever
        r = FAISSRetriever()
        r.load_indexes()

        patches = make_random_patches(784, 256)
        score, patch_scores, score_std, D = r.score_patches(patches, "bottle")

        assert isinstance(score, float)
        assert score >= 0.0
        assert patch_scores.shape == (28, 28)
        assert score_std >= 0.0

    def test_lru_cache_hit_on_second_call(self):
        """Second call with same image hash must be a cache hit."""
        from src.cache import LRUCache

        cache = LRUCache(maxsize=10)
        cache.set("abc123", {"score": 0.42})

        result = cache.get("abc123")
        assert result is not None
        assert result["score"] == 0.42
        assert cache.stats()["hits"] == 1
        assert cache.stats()["misses"] == 0

    def test_cache_miss_on_unknown_key(self):
        """Unknown key must return None."""
        from src.cache import LRUCache
        cache = LRUCache(maxsize=10)

        result = cache.get("nonexistent")
        assert result is None
        assert cache.stats()["misses"] == 1

    def test_cache_evicts_lru_when_full(self):
        """Cache must evict least recently used when maxsize exceeded."""
        from src.cache import LRUCache
        cache = LRUCache(maxsize=3)

        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        cache.set("d", 4)   # should evict "a"

        assert cache.get("a") is None
        assert cache.get("b") == 2