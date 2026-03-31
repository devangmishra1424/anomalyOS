# tests/test_orchestrator.py
import numpy as np
import pytest
from PIL import Image
from unittest.mock import MagicMock, patch
import io


def make_dummy_image(size=(224, 224)):
    return Image.fromarray(
        np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    )


def make_image_bytes(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


class TestOrchestrator:

    @pytest.fixture(autouse=True)
    def mock_dependencies(self):
        """Mock all heavy components so tests run without model files."""
        with patch("src.orchestrator.patchcore") as mock_pc, \
             patch("src.orchestrator.retriever") as mock_ret, \
             patch("src.orchestrator.knowledge_graph") as mock_graph, \
             patch("src.orchestrator.depth_estimator") as mock_depth, \
             patch("src.orchestrator.shap_explainer") as mock_shap, \
             patch("src.orchestrator._get_clip_embedding") as mock_clip, \
             patch("src.orchestrator.queue_report") as mock_report:

            # Normal image setup
            mock_pc.extract_patches.return_value = np.random.rand(
                784, 256).astype(np.float32)
            mock_pc.build_anomaly_map.return_value = np.random.rand(
                224, 224).astype(np.float32)
            mock_pc.get_anomaly_centroid.return_value = (112, 112)
            mock_pc.calibrate_score.return_value = 0.2

            mock_ret.route_category.return_value = {
                "category": "bottle", "confidence": 0.9
            }
            mock_ret.score_patches.return_value = (
                0.3,                               # score (below threshold)
                np.random.rand(28, 28).astype(np.float32),
                0.02,
                np.random.rand(784, 5).astype(np.float32)
            )
            mock_ret.retrieve_similar_defects.return_value = []
            mock_ret.index3_cache = {}

            mock_graph.get_context.return_value = {
                "root_causes": ["surface_abrasion"],
                "remediations": ["inspect_conveyor"],
                "co_occurs": []
            }

            mock_depth.get_depth_stats.return_value = {
                "mean_depth": 0.5, "depth_variance": 0.1,
                "gradient_magnitude": 0.2, "spatial_entropy": 1.5,
                "depth_range": 0.8
            }
            mock_depth.get_depth_map.return_value = np.zeros(
                (224, 224), dtype=np.float32)

            mock_clip.return_value = np.random.rand(512).astype(np.float32)

            mock_shap.build_feature_vector.return_value = np.array(
                [0.3, 0.5, 0.1, 0.2, 0.6], dtype=np.float32)
            mock_shap.explain.return_value = {
                "feature_names": ["a", "b", "c", "d", "e"],
                "shap_values": [0.1, 0.2, 0.3, 0.1, 0.1],
                "base_value": 0.0,
                "prediction": 0.5
            }

            mock_report.return_value = "test-report-id"

            # Inject thresholds
            import src.orchestrator as orch
            orch._thresholds = {
                "bottle": {"threshold": 0.5, "cal_mean": 0.3, "cal_std": 0.1}
            }

            self.mocks = {
                "patchcore": mock_pc,
                "retriever": mock_ret,
                "graph": mock_graph,
                "depth": mock_depth,
                "shap": mock_shap,
                "clip": mock_clip,
                "report": mock_report
            }
            yield

    def test_normal_image_returns_is_anomalous_false(self):
        """Score below threshold → is_anomalous=False, early exit."""
        from src.orchestrator import run_inspection

        # Score 0.3 < threshold 0.5 → normal
        self.mocks["retriever"].score_patches.return_value = (
            0.3, np.random.rand(28, 28).astype(np.float32), 0.02,
            np.random.rand(784, 5).astype(np.float32)
        )

        img   = make_dummy_image()
        bytes_ = make_image_bytes(img)
        result = run_inspection(img, bytes_)

        assert result.is_anomalous is False
        assert result.score == 0.3
        assert result.category == "bottle"

    def test_defective_image_returns_is_anomalous_true(self):
        """Score above threshold → is_anomalous=True, full pipeline runs."""
        from src.orchestrator import run_inspection

        self.mocks["retriever"].score_patches.return_value = (
            0.85, np.random.rand(28, 28).astype(np.float32), 0.05,
            np.random.rand(784, 5).astype(np.float32)
        )
        self.mocks["patchcore"].calibrate_score.return_value = 0.85

        img    = make_dummy_image()
        bytes_ = make_image_bytes(img)
        result = run_inspection(img, bytes_)

        assert result.is_anomalous is True
        assert result.score == 0.85

    def test_result_has_report_id_when_anomalous(self):
        """Anomalous result must have a report_id for LLM polling."""
        from src.orchestrator import run_inspection

        self.mocks["retriever"].score_patches.return_value = (
            0.85, np.random.rand(28, 28).astype(np.float32), 0.05,
            np.random.rand(784, 5).astype(np.float32)
        )
        self.mocks["patchcore"].calibrate_score.return_value = 0.85

        img    = make_dummy_image()
        bytes_ = make_image_bytes(img)
        result = run_inspection(img, bytes_)

        assert result.report_id == "test-report-id"

    def test_cache_hit_on_repeated_image(self):
        """Same image submitted twice — second call hits cache."""
        from src.orchestrator import run_inspection
        from src.cache import inference_cache
        inference_cache.clear()

        img    = make_dummy_image()
        bytes_ = make_image_bytes(img)

        run_inspection(img, bytes_)
        run_inspection(img, bytes_)

        stats = inference_cache.stats()
        assert stats["hits"] >= 1

    def test_result_contains_graph_context_when_anomalous(self):
        """Graph context must be populated for anomalous images."""
        from src.orchestrator import run_inspection

        self.mocks["retriever"].score_patches.return_value = (
            0.85, np.random.rand(28, 28).astype(np.float32), 0.05,
            np.random.rand(784, 5).astype(np.float32)
        )
        self.mocks["patchcore"].calibrate_score.return_value = 0.85
        self.mocks["retriever"].retrieve_similar_defects.return_value = [
            {"category": "bottle", "defect_type": "broken_large",
             "image_hash": "abc", "similarity_score": 0.9}
        ]

        img    = make_dummy_image()
        bytes_ = make_image_bytes(img)
        result = run_inspection(img, bytes_)

        assert "root_causes" in result.graph_context

    def test_latency_ms_is_positive(self):
        """Latency must always be a positive number."""
        from src.orchestrator import run_inspection

        img    = make_dummy_image()
        bytes_ = make_image_bytes(img)
        result = run_inspection(img, bytes_)

        assert result.latency_ms > 0