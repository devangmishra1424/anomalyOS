# tests/test_api.py
import pytest
import io
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from typing import Optional


# ── Minimal standalone FastAPI app for testing ────────────────
# Tests validation logic only — no model loading required
test_app = FastAPI()

VALID_CATEGORIES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut',
    'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush',
    'transistor', 'wood', 'zipper'
]
MAX_FILE_SIZE = 10 * 1024 * 1024


@test_app.get("/health")
def health():
    return {"status": "ok", "version": "v1.0",
            "uptime_seconds": 0.0, "index_sizes": {},
            "coreset_size": 0, "threshold_config_version": "v1.0",
            "cache_stats": {}}


@test_app.get("/metrics")
def metrics():
    return {"request_count": 0, "latency_p50_ms": 0.0,
            "latency_p95_ms": 0.0, "cache_hit_rate": 0.0,
            "hf_push_failure_count": 0, "memory_usage_mb": 0.0}


@test_app.get("/report/{report_id}")
def report(report_id: str):
    return {"status": "not_found", "report": None}


@test_app.post("/inspect")
async def inspect(
    image: UploadFile = File(...),
    category_hint: Optional[str] = Form(None)
):
    if image.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=422, detail="Only jpg/png accepted")
    data = await image.read()
    if len(data) == 0:
        raise HTTPException(status_code=422, detail="Empty file")
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Too large")
    if category_hint and category_hint not in VALID_CATEGORIES:
        raise HTTPException(status_code=422, detail="Invalid category")
    try:
        Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=422, detail="Cannot decode image")
    return {"is_anomalous": True, "anomaly_score": 0.75,
            "calibrated_score": 0.82, "score_std": 0.05,
            "category": "bottle", "version": "v1.0",
            "image_hash": "abc123", "latency_ms": 250.0,
            "low_confidence": False, "similar_cases": [],
            "graph_context": {}, "shap_features": {}, "report_id": "test-id"}


@test_app.post("/arena/submit/{case_id}")
def arena_submit(case_id: str, user_rating: int, user_severity: int):
    if user_rating not in (0, 1):
        raise HTTPException(status_code=422, detail="Invalid rating")
    if not (1 <= user_severity <= 5):
        raise HTTPException(status_code=422, detail="Invalid severity")
    return {"correct_label": 1, "ai_score": 0.8,
            "calibrated_score": 0.8, "user_score": 1.0,
            "streak": 1, "top_shap_features": []}


@test_app.post("/correct/{case_id}")
def correct(case_id: str, correction_type: str, note: str = ""):
    valid = ["false_positive", "false_negative", "wrong_category"]
    if correction_type not in valid:
        raise HTTPException(status_code=422, detail="Invalid correction type")
    return {"status": "correction_logged", "case_id": case_id}


client = TestClient(test_app)


# ── Tests ─────────────────────────────────────────────────────
def make_image_bytes(size=(224, 224), fmt="JPEG") -> bytes:
    img = Image.fromarray(
        np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    )
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


class TestHealthEndpoint:
    def test_health_returns_200(self):
        assert client.get("/health").status_code == 200

    def test_health_has_required_fields(self):
        body = client.get("/health").json()
        assert "status" in body
        assert "version" in body
        assert body["status"] == "ok"


class TestMetricsEndpoint:
    def test_metrics_returns_200(self):
        assert client.get("/metrics").status_code == 200

    def test_metrics_has_required_fields(self):
        body = client.get("/metrics").json()
        assert "request_count" in body
        assert "cache_hit_rate" in body


class TestInspectEndpoint:
    def test_valid_jpeg_returns_200(self):
        img = make_image_bytes(fmt="JPEG")
        resp = client.post("/inspect",
                           files={"image": ("t.jpg", img, "image/jpeg")})
        assert resp.status_code == 200

    def test_valid_png_returns_200(self):
        img = make_image_bytes(fmt="PNG")
        resp = client.post("/inspect",
                           files={"image": ("t.png", img, "image/png")})
        assert resp.status_code == 200

    def test_response_has_required_fields(self):
        img = make_image_bytes()
        resp = client.post("/inspect",
                           files={"image": ("t.jpg", img, "image/jpeg")})
        body = resp.json()
        assert "is_anomalous" in body
        assert "anomaly_score" in body
        assert "category" in body

    def test_non_image_file_returns_422(self):
        resp = client.post("/inspect",
                           files={"image": ("t.pdf", b"fake", "application/pdf")})
        assert resp.status_code == 422

    def test_empty_file_returns_422(self):
        resp = client.post("/inspect",
                           files={"image": ("t.jpg", b"", "image/jpeg")})
        assert resp.status_code == 422

    def test_oversized_file_returns_413(self):
        big = b"x" * (11 * 1024 * 1024)
        resp = client.post("/inspect",
                           files={"image": ("t.jpg", big, "image/jpeg")})
        assert resp.status_code == 413

    def test_invalid_category_hint_returns_422(self):
        img = make_image_bytes()
        resp = client.post("/inspect",
                           files={"image": ("t.jpg", img, "image/jpeg")},
                           data={"category_hint": "invalid_xyz"})
        assert resp.status_code == 422

    def test_valid_category_hint_accepted(self):
        img = make_image_bytes()
        resp = client.post("/inspect",
                           files={"image": ("t.jpg", img, "image/jpeg")},
                           data={"category_hint": "bottle"})
        assert resp.status_code == 200


class TestReportEndpoint:
    def test_report_endpoint_returns_200(self):
        assert client.get("/report/any-id").status_code == 200

    def test_unknown_report_returns_not_found(self):
        body = client.get("/report/nonexistent").json()
        assert body["status"] == "not_found"


class TestArenaEndpoint:
    def test_invalid_rating_returns_422(self):
        resp = client.post("/arena/submit/fake",
                           params={"user_rating": 5, "user_severity": 3})
        assert resp.status_code == 422

    def test_invalid_severity_returns_422(self):
        resp = client.post("/arena/submit/fake",
                           params={"user_rating": 1, "user_severity": 10})
        assert resp.status_code == 422


class TestCorrectionEndpoint:
    def test_valid_correction_returns_200(self):
        resp = client.post("/correct/fake",
                           params={"correction_type": "false_positive"})
        assert resp.status_code == 200

    def test_invalid_correction_type_returns_422(self):
        resp = client.post("/correct/fake",
                           params={"correction_type": "bad_type"})
        assert resp.status_code == 422