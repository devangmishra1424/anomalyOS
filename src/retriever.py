# src/retriever.py
# Loads and searches all 3 FAISS indexes
#
# Index 1 — Category (15 vectors, IndexFlatIP, CLIP full-image)
# Index 2 — Defect pattern (5354 vectors, IndexFlatIP, CLIP crop)
# Index 3 — PatchCore coreset (per-category, IndexFlatL2, WideResNet patches)
#           LAZY LOADED — only loaded on first request per category
#           Reduces startup time from ~45s to ~15s

import os
import json
import numpy as np
import faiss

# Paths — relative to repo root, mounted in Docker at /app/data/
DATA_DIR = os.environ.get("DATA_DIR", "data")

CATEGORIES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut',
    'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush',
    'transistor', 'wood', 'zipper'
]


class FAISSRetriever:
    """
    Manages all 3 FAISS indexes with lazy loading for Index 3.
    Loaded once at FastAPI startup, kept in memory for server lifetime.
    """

    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.index1 = None          # Category index
        self.index1_metadata = None
        self.index2 = None          # Defect pattern index
        self.index2_metadata = None
        self.index3_cache = {}      # category → loaded FAISS index (lazy)

    def load_indexes(self):
        """
        Load Index 1 and Index 2 at startup.
        Index 3 is lazy-loaded per category on first request.
        """
        # ── Index 1 ──────────────────────────────────────────
        idx1_path = os.path.join(self.data_dir, "index1_category.faiss")
        meta1_path = os.path.join(self.data_dir, "index1_metadata.json")

        if not os.path.exists(idx1_path):
            raise FileNotFoundError(f"Index 1 not found: {idx1_path}")

        self.index1 = faiss.read_index(idx1_path)
        with open(meta1_path) as f:
            self.index1_metadata = json.load(f)
        print(f"Index 1 loaded: {self.index1.ntotal} category vectors")

        # ── Index 2 ──────────────────────────────────────────
        idx2_path = os.path.join(self.data_dir, "index2_defect.faiss")
        meta2_path = os.path.join(self.data_dir, "index2_metadata.json")

        if not os.path.exists(idx2_path):
            raise FileNotFoundError(f"Index 2 not found: {idx2_path}")

        # Memory-mapped — not fully loaded into RAM
        self.index2 = faiss.read_index(idx2_path, faiss.IO_FLAG_MMAP)
        with open(meta2_path) as f:
            self.index2_metadata = json.load(f)
        print(f"Index 2 loaded: {self.index2.ntotal} defect pattern vectors")

    def _load_index3(self, category: str):
        """Lazy load Index 3 for a specific category."""
        if category not in self.index3_cache:
            path = os.path.join(self.data_dir, f"index3_{category}.faiss")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Index 3 not found for {category}: {path}")
            self.index3_cache[category] = faiss.read_index(
                path, faiss.IO_FLAG_MMAP
            )
            print(f"Index 3 lazy-loaded: {category} "
                  f"({self.index3_cache[category].ntotal} coreset vectors)")
        return self.index3_cache[category]

    # ── Index 1: Category routing ─────────────────────────────
    def route_category(self, clip_full_embedding: np.ndarray) -> dict:
        """
        Given a full-image CLIP embedding, return the predicted category.
        Returns: {category, confidence_score}
        """
        query = clip_full_embedding.reshape(1, -1).astype(np.float32)
        # Normalise for cosine similarity
        query = query / (np.linalg.norm(query) + 1e-8)
        D, I = self.index1.search(query, k=1)
        cat_idx = int(I[0][0])
        return {
            "category": CATEGORIES[cat_idx],
            "confidence": float(D[0][0])
        }

    # ── Index 2: Defect pattern retrieval ────────────────────
    def retrieve_similar_defects(self,
                                  clip_crop_embedding: np.ndarray,
                                  k: int = 5,
                                  exclude_hash: str = None,
                                  category_filter: str = None) -> list:
        """
        Given a defect-crop CLIP embedding, return k most similar
        historical defect cases.
        exclude_hash: skip self-match (same image submitted again)
        category_filter: only return cases from specified category
        Returns: list of metadata dicts with similarity scores
        """
        query = clip_crop_embedding.reshape(1, -1).astype(np.float32)
        query = query / (np.linalg.norm(query) + 1e-8)

        # Fetch k+1 to allow filtering self-match
        D, I = self.index2.search(query, k=k + 1)

        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            meta = self.index2_metadata[idx].copy()
            meta["similarity_score"] = float(dist)
            # Filter by category if provided
            if category_filter and meta.get("category") != category_filter:
                continue
            # Skip self-match
            if exclude_hash and meta.get("image_hash") == exclude_hash:
                continue
            results.append(meta)
            if len(results) == k:
                break

        return results

    # ── Index 3: PatchCore k-NN scoring ──────────────────────
    def score_patches(self,
                       patches: np.ndarray,
                       category: str,
                       k: int = 1) -> tuple:
        """
        Given [784, 256] patch features, return anomaly score and
        per-patch distance grid.
        
        Returns:
            image_score: float — max patch distance (anomaly score)
            patch_scores: [28, 28] numpy array of per-patch distances
            nn_distances: [784, k] all k-NN distances (for confidence interval)
        """
        index3 = self._load_index3(category)
        patches_f32 = patches.astype(np.float32)

        # k=5 neighbours: first for scoring, rest for confidence interval
        D, _ = index3.search(patches_f32, k=5)

        # Primary score: nearest neighbour distance per patch
        patch_scores = D[:, 0].reshape(28, 28)
        image_score = float(patch_scores.max())

        # Confidence interval: std of top-5 distances at most anomalous patch
        max_patch_idx = np.argmax(D[:, 0])
        score_std = float(np.std(D[max_patch_idx]))

        return image_score, patch_scores, score_std, D

    def get_status(self) -> dict:
        """Returns index sizes for /health endpoint."""
        return {
            "index1_vectors": self.index1.ntotal if self.index1 else 0,
            "index2_vectors": self.index2.ntotal if self.index2 else 0,
            "index3_loaded_categories": list(self.index3_cache.keys()),
            "index3_total_categories": len(CATEGORIES)
        }


# Global instance — initialised in api/startup.py
retriever = FAISSRetriever()