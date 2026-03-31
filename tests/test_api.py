# tests/test_api.py
# Tests all API endpoints with correct HTTP status codes
# Uses FastAPI TestClient — no real server needed

import pytest
import io
import json
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


def make_image_bytes(size=(224, 224), fmt="JPEG") -> bytes:
    """Create a minimal valid image as bytes."""
    img = Image.fromarray(
        np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    )
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture(scope="module")
def client():
    """
    Create TestClient with mocked startup.
    We mock load_all() so tests don't need real model files.
    """
    with patch("api.startup.load_all") as mock_load:
        mock_load.return_value = {}

        # Mock all global model instances
        with patch("src.patchcore.patchcore") as mock_pc, \
             patch("src.retriever.retriever") as mock_ret, \
             patch("src.graph.knowledge_graph") as mock_graph, \
             patch("src.depth.depth_estimator") as mock_depth, \
             patch("src.xai.gradcam") as mock_cam, \
             patch("src.xai.shap_explainer") as mock_shap, \
             patch("src.orchestrator.run_inspection") as mock_inspect:

            # Configure mock inspection result
            from src.orchestrator import OrchestratorResult
            mock_inspect.return_value = OrchestratorResult(
                is_anomalous=True,
                score=0.75,
                calibrated_score=0.82,
                score_std=0.05,
                category="bottle",
                heatmap_b64="fake_b64",
                report_id="test-report-id",
                latency_ms=250.0
            )

            mock_ret.get_status.return_value = {
                "index1_vectors": 15,
                "index2_vectors": 5354,
                "index3_loaded_categories": [],
                "index3_total_categories": 15
            }
            mock_ret.index3_cache = {}

            from api.main import app
            from src.cache import inference_cache
            inference_cache.clear()

            yield TestClient(app)


class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_required_fields(self, client):
        resp = client.get("/health")
        body = resp.json()
        assert "status" in body
        assert "model_version" in body
        assert "uptime_seconds" in body
        assert "index_sizes" in body
        assert body["status"] == "ok"


class TestMetricsEndpoint:

    def test_metrics_returns_200(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_has_required_fields(self, client):
        body = client.get("/metrics").json()
        assert "request_count" in body
        assert "latency_p50_ms" in body
        assert "cache_hit_rate" in body


class TestInspectEndpoint:

    def test_valid_jpeg_returns_200(self, client):
        img_bytes = make_image_bytes(fmt="JPEG")
        resp = client.post(
            "/inspect",
            files={"image": ("test.jpg", img_bytes, "image/jpeg")}
        )
        assert resp.status_code == 200

    def test_valid_png_returns_200(self, client):
        img_bytes = make_image_bytes(fmt="PNG")
        resp = client.post(
            "/inspect",
            files={"image": ("test.png", img_bytes, "image/png")}
        )
        assert resp.status_code == 200

    def test_response_has_required_fields(self, client):
        img_bytes = make_image_bytes()
        resp = client.post(
            "/inspect",
            files={"image": ("test.jpg", img_bytes, "image/jpeg")}
        )
        body = resp.json()
        assert "is_anomalous" in body
        assert "anomaly_score" in body
        assert "calibrated_score" in body
        assert "category" in body
        assert "model_version" in body
        assert "image_hash" in body

    def test_non_image_file_returns_422(self, client):
        resp = client.post(
            "/inspect",
            files={"image": ("test.pdf", b"fake pdf content", "application/pdf")}
        )
        assert resp.status_code == 422

    def test_empty_file_returns_422(self, client):
        resp = client.post(
            "/inspect",
            files={"image": ("test.jpg", b"", "image/jpeg")}
        )
        assert resp.status_code == 422

    def test_oversized_file_returns_413(self, client):
        # Create >10MB payload
        big_bytes = b"x" * (11 * 1024 * 1024)
        resp = client.post(
            "/inspect",
            files={"image": ("big.jpg", big_bytes, "image/jpeg")}
        )
        assert resp.status_code == 413

    def test_invalid_category_hint_returns_422(self, client):
        img_bytes = make_image_bytes()
        resp = client.post(
            "/inspect",
            files={"image": ("test.jpg", img_bytes, "image/jpeg")},
            data={"category_hint": "invalid_category_xyz"}
        )
        assert resp.status_code == 422

    def test_valid_category_hint_accepted(self, client):
        img_bytes = make_image_bytes()
        resp = client.post(
            "/inspect",
            files={"image": ("test.jpg", img_bytes, "image/jpeg")},
            data={"category_hint": "bottle"}
        )
        assert resp.status_code == 200


class TestReportEndpoint:

    def test_unknown_report_id_returns_not_found(self, client):
        resp = client.get("/report/nonexistent-id-xyz")
        body = resp.json()
        assert body["status"] == "not_found"

    def test_report_endpoint_returns_200(self, client):
        resp = client.get("/report/any-id")
        assert resp.status_code == 200


class TestArenaEndpoint:

    def test_arena_submit_invalid_rating_returns_422(self, client):
        resp = client.post(
            "/arena/submit/fake-case-id",
            json={"user_rating": 5,    # invalid — must be 0 or 1
                  "user_severity": 3}
        )
        assert resp.status_code == 422

    def test_arena_submit_invalid_severity_returns_422(self, client):
        resp = client.post(
            "/arena/submit/fake-case-id",
            json={"user_rating": 1,
                  "user_severity": 10}  # invalid — must be 1-5
        )
        assert resp.status_code == 422


class TestCorrectionEndpoint:

    def test_valid_correction_returns_200(self, client):
        resp = client.post(
            "/correct/fake-case-id",
            json={"correction_type": "false_positive",
                  "note": "The image is clearly normal."}
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "correction_logged"

    def test_invalid_correction_type_returns_422(self, client):
        resp = client.post(
            "/correct/fake-case-id",
            json={"correction_type": "totally_wrong_type"}
        )
        assert resp.status_code == 422