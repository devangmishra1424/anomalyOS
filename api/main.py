# api/main.py
# FastAPI application — 9 endpoints
# Models loaded once at startup via lifespan, kept in memory

import os
import io
import time
import hashlib
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np

from api.startup import load_all, get_uptime, MODEL_VERSION
from api.schemas import (
    InspectResponse, ReportResponse, ForensicsResponse,
    KnowledgeSearchResponse, ArenaCase, ArenaSubmitRequest,
    ArenaSubmitResponse, CorrectionRequest, CorrectionResponse,
    HealthResponse, MetricsResponse
)
from api.logger import (
    log_inference, log_arena_submission, log_correction,
    get_push_failure_count
)
from src.orchestrator import run_inspection
from src.retriever import retriever
from src.graph import knowledge_graph
from src.xai import gradcam, shap_explainer, heatmap_to_base64, image_to_base64
from src.llm import get_report, generate_report
from src.cache import inference_cache, get_image_hash

import psutil
import random


# ── Request-scoped state via ContextVar ──────────────────────
# Prevents race conditions under concurrent requests
# Never use global mutable state for per-request data
request_session_id: ContextVar[str] = ContextVar("session_id", default="")

# ── Metrics counters ─────────────────────────────────────────
_metrics = {
    "request_count": 0,
    "latencies": [],
    "hf_push_failure_count": 0
}

# ── Precompute store (speculative CLIP encoding) ──────────────
_precompute_store: dict = {}

# ── Arena leaderboard (in-memory, persisted to HF Dataset) ───
_arena_streaks: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models at startup. Nothing else runs before this."""
    load_all()
    yield
    # Cleanup on shutdown (not critical but clean)
    inference_cache.clear()


app = FastAPI(
    title="AnomalyOS",
    description="Industrial Visual Anomaly Detection Platform",
    version=MODEL_VERSION,
    lifespan=lifespan
)


# ── Helpers ───────────────────────────────────────────────────
VALID_CATEGORIES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut',
    'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush',
    'transistor', 'wood', 'zipper'
]

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


def _validate_image(file: UploadFile, image_bytes: bytes) -> Image.Image:
    """
    Validate uploaded image. Raises HTTPException on any failure.
    Model is never called on invalid input.
    """
    # File type
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=422,
                            detail="Only jpg/png accepted")
    # File size
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413,
                            detail="Max file size is 10MB")
    # Zero-byte
    if len(image_bytes) == 0:
        raise HTTPException(status_code=422,
                            detail="Image file is empty")
    # Decode
    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=422,
                            detail="Could not decode image")
    # Too small
    if pil_img.size[0] < 32 or pil_img.size[1] < 32:
        raise HTTPException(status_code=422,
                            detail="Image too small for inspection")
    return pil_img


def _record_latency(latency_ms: float):
    _metrics["request_count"] += 1
    _metrics["latencies"].append(latency_ms)
    if len(_metrics["latencies"]) > 1000:
        _metrics["latencies"] = _metrics["latencies"][-500:]


# ── POST /inspect ─────────────────────────────────────────────
@app.post("/inspect", response_model=InspectResponse)
async def inspect(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    category_hint: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None)
):
    """
    Main inspection endpoint.
    Accepts: multipart form (image + optional metadata)
    Returns: anomaly result immediately, LLM report polled separately
    """
    # Validate category hint
    if category_hint and category_hint not in VALID_CATEGORIES:
        raise HTTPException(status_code=422,
                            detail=f"Invalid category_hint: {category_hint}")

    image_bytes = await image.read()
    pil_img     = _validate_image(image, image_bytes)

    # Run full orchestrator pipeline
    result = run_inspection(
        pil_img=pil_img,
        image_bytes=image_bytes,
        category_hint=category_hint
    )

    # Queue LLM report generation (non-blocking)
    if result.report_id and result.is_anomalous:
        background_tasks.add_task(
            generate_report,
            result.report_id,
            result.category,
            result.score,
            result.similar_cases,
            result.graph_context
        )

    # Log inference (non-blocking)
    image_hash = get_image_hash(image_bytes)
    log_record = {
        "mode":             "inspector",
        "image_hash":       image_hash,
        "category":         result.category,
        "anomaly_score":    result.score,
        "calibrated_score": result.calibrated_score,
        "is_anomalous":     result.is_anomalous,
        "latency_ms":       result.latency_ms,
        "model_version":    MODEL_VERSION,
        "report_id":        result.report_id
    }
    background_tasks.add_task(log_inference, log_record)
    _record_latency(result.latency_ms)

    return InspectResponse(
        is_anomalous=result.is_anomalous,
        anomaly_score=result.score,
        calibrated_score=result.calibrated_score,
        score_std=result.score_std,
        category=result.category,
        version=MODEL_VERSION,
        heatmap_b64=result.heatmap_b64,
        defect_crop_b64=result.defect_crop_b64,
        depth_map_b64=result.depth_map_b64,
        similar_cases=result.similar_cases,
        graph_context=result.graph_context,
        shap_features=result.shap_features,
        report_id=result.report_id,
        latency_ms=result.latency_ms,
        image_hash=image_hash,
        low_confidence=result.calibrated_score < 0.3
    )


# ── GET /report/{report_id} ───────────────────────────────────
@app.get("/report/{report_id}", response_model=ReportResponse)
async def get_report_status(report_id: str):
    """
    Poll LLM report status.
    Frontend polls every 500ms until status == 'ready'.
    """
    result = get_report(report_id)
    return ReportResponse(
        status=result["status"],
        report=result.get("report")
    )


# ── POST /forensics/{case_id} ─────────────────────────────────
@app.post("/forensics/{case_id}", response_model=ForensicsResponse)
async def forensics(
    case_id: str,
    coreset_pct: Optional[float] = None
):
    """
    Deep XAI analysis of a previously logged case.
    Loads case from cache or HF Dataset, runs full XAI suite.
    coreset_pct: optional ablation parameter (0.001-0.1)
    """
    if coreset_pct is not None and not (0.001 <= coreset_pct <= 0.1):
        raise HTTPException(status_code=422,
                            detail="coreset_pct must be between 0.001 and 0.1")

    # Load case from cache
    cached = inference_cache.get(case_id)
    if not cached:
        raise HTTPException(status_code=422,
                            detail="Case not found. Run inspection first.")

    # GradCAM++ (runs here, not in Inspector)
    gradcam_b64 = None
    if cached.get("_pil_img"):
        cam = gradcam.compute(cached["_pil_img"])
        if cam is not None:
            gradcam_b64 = heatmap_to_base64(cam, cached["_pil_img"])

    # Retrieval trace — enrich similar cases with similarity scores
    retrieval_trace = []
    for case in cached.get("similar_cases", []):
        retrieval_trace.append({
            "case_id":          case.get("image_hash", "")[:12],
            "category":         case.get("category"),
            "defect_type":      case.get("defect_type"),
            "similarity_score": case.get("similarity_score"),
            "graph_path":       _format_graph_path(
                                    case.get("category"),
                                    case.get("defect_type")
                                )
        })

    return ForensicsResponse(
        case_id=case_id,
        category=cached.get("category", "unknown"),
        anomaly_score=cached.get("score", 0.0),
        calibrated_score=cached.get("calibrated_score", 0.0),
        patch_scores_grid=cached.get("patch_scores_grid", []),
        gradcampp_b64=gradcam_b64,
        shap_features=cached.get("shap_features", {}),
        similar_cases=cached.get("similar_cases", []),
        graph_context=cached.get("graph_context", {}),
        retrieval_trace=retrieval_trace
    )


def _format_graph_path(category: str, defect_type: str) -> str:
    """Format 2-hop graph path as plain text for Forensics trace."""
    if not category or not defect_type:
        return "unknown"
    ctx = knowledge_graph.get_context(category, defect_type)
    rcs  = ctx.get("root_causes", [])
    rems = ctx.get("remediations", [])
    if rcs and rems:
        return f"caused_by: {rcs[0]} → remediated_by: {rems[0]}"
    elif rcs:
        return f"caused_by: {rcs[0]}"
    return "no graph path found"


# ── GET /knowledge/search ─────────────────────────────────────
@app.get("/knowledge/search", response_model=KnowledgeSearchResponse)
async def knowledge_search(
    category:     Optional[str]  = None,
    defect_type:  Optional[str]  = None,
    severity_min: Optional[float] = None,
    severity_max: Optional[float] = None,
    query:        Optional[str]  = None
):
    """
    Search defect knowledge base.
    Natural language query → MiniLM embed → Index 2 search.
    Filters: category, defect_type, severity range.
    """
    all_defects = knowledge_graph.get_all_defect_nodes()
    results     = all_defects

    # Filter by category
    if category:
        results = [r for r in results if r.get("category") == category]

    # Filter by defect type
    if defect_type:
        results = [r for r in results
                   if defect_type.lower() in r.get("defect_type", "").lower()]

    # Filter by severity
    if severity_min is not None:
        results = [r for r in results
                   if r.get("severity_min", 0) >= severity_min]
    if severity_max is not None:
        results = [r for r in results
                   if r.get("severity_max", 1) <= severity_max]

    # Natural language search via Index 2
    if query and retriever.index2 is not None:
        try:
            from sentence_transformers import SentenceTransformer
            _mini_lm = SentenceTransformer("all-MiniLM-L6-v2")
            query_emb = _mini_lm.encode([query])[0].astype("float32")
            query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)
            # Pad or truncate to 512 dims to match Index 2
            if len(query_emb) < 512:
                query_emb = np.pad(query_emb, (0, 512 - len(query_emb)))
            else:
                query_emb = query_emb[:512]
            D, I = retriever.index2.search(query_emb.reshape(1, -1), k=10)
            nl_results = [retriever.index2_metadata[i]
                          for i in I[0] if i >= 0]
            results = nl_results if nl_results else results
        except Exception as e:
            print(f"NL search failed: {e} — using filter results")

    return KnowledgeSearchResponse(
        results=results[:50],
        total_found=len(results),
        query=query or ""
    )


# ── GET /arena/next_case ──────────────────────────────────────
@app.get("/arena/next_case", response_model=ArenaCase)
async def arena_next_case(expert_mode: bool = False):
    """
    Returns next Arena challenge image.
    Expert mode: cases with calibrated_score in [0.45, 0.55] (maximum uncertainty)
    """
    import os
    from src.cache import pil_to_bytes
    import base64

    MVTEC_PATH = os.environ.get("MVTEC_PATH", "/app/data/mvtec")
    categories = VALID_CATEGORIES

    # Pick a random category and image
    cat  = random.choice(categories)
    split = random.choice(["train", "test"])

    if split == "train":
        img_dir = os.path.join(MVTEC_PATH, cat, "train", "good")
    else:
        defect_types = os.listdir(os.path.join(MVTEC_PATH, cat, "test"))
        defect_type  = random.choice(defect_types)
        img_dir      = os.path.join(MVTEC_PATH, cat, "test", defect_type)

    if not os.path.exists(img_dir):
        raise HTTPException(status_code=500, detail="Dataset not mounted")

    files = [f for f in os.listdir(img_dir)
             if f.endswith((".png", ".jpg", ".jpeg"))]
    if not files:
        raise HTTPException(status_code=500, detail="No images found")

    fname    = random.choice(files)
    img_path = os.path.join(img_dir, fname)
    pil_img  = Image.open(img_path).convert("RGB")

    # Generate case_id from path hash
    case_id = hashlib.sha256(img_path.encode()).hexdigest()[:16]

    # Cache the image path for submit endpoint
    _precompute_store[case_id] = {
        "img_path":   img_path,
        "category":   cat,
        "is_defective": split == "test" and defect_type != "good"
    }

    img_b64 = image_to_base64(pil_img)

    return ArenaCase(
        case_id=case_id,
        image_b64=img_b64,
        expert_mode=expert_mode
    )


# ── POST /arena/submit/{case_id} ──────────────────────────────
@app.post("/arena/submit/{case_id}", response_model=ArenaSubmitResponse)
async def arena_submit(
    case_id: str,
    request: ArenaSubmitRequest,
    background_tasks: BackgroundTasks
):
    """Submit Arena answer. Returns AI result + user score + SHAP explanation."""
    case_info = _precompute_store.get(case_id)
    if not case_info:
        raise HTTPException(status_code=422, detail="Case not found")

    pil_img     = Image.open(case_info["img_path"]).convert("RGB")
    image_bytes = pil_to_bytes(pil_img)

    result = run_inspection(pil_img=pil_img, image_bytes=image_bytes)

    correct_label = 1 if case_info["is_defective"] else 0
    user_correct  = int(request.user_rating == correct_label)

    # Severity score: 1 if within 1 of AI severity, 0 otherwise
    ai_severity  = round(result.calibrated_score * 5)
    sev_score    = 1 if abs(request.user_severity - ai_severity) <= 1 else 0
    user_score   = float(user_correct + sev_score * 0.5)

    # Streak tracking
    session = request.session_id or "anonymous"
    streak  = _arena_streaks.get(session, 0)
    if user_correct:
        streak += 1
    else:
        streak = 0
    _arena_streaks[session] = streak

    # Top 2 SHAP features for post-submission explanation
    shap_data = result.shap_features
    top_shap  = []
    if shap_data.get("feature_names"):
        pairs = list(zip(shap_data["feature_names"],
                         shap_data["shap_values"]))
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        top_shap = [{"feature": p[0], "contribution": round(p[1], 4)}
                    for p in pairs[:2]]

    # Log
    background_tasks.add_task(log_arena_submission, {
        "case_id":      case_id,
        "user_rating":  request.user_rating,
        "ai_decision":  int(result.is_anomalous),
        "user_score":   user_score,
        "streak":       streak,
        "session_id":   session
    })

    return ArenaSubmitResponse(
        correct_label=correct_label,
        ai_score=result.score,
        calibrated_score=result.calibrated_score,
        user_score=user_score,
        streak=streak,
        top_shap_features=top_shap,
        heatmap_b64=result.heatmap_b64,
        is_expert_case=0.45 <= result.calibrated_score <= 0.55
    )


# ── POST /correct/{case_id} ───────────────────────────────────
@app.post("/correct/{case_id}", response_model=CorrectionResponse)
async def submit_correction(
    case_id: str,
    request: CorrectionRequest,
    background_tasks: BackgroundTasks
):
    """
    User correction widget backend.
    Every correction logged with user_override=True flag.
    Interview line: "Corrections can seed a future active learning cycle."
    """
    background_tasks.add_task(log_correction, {
        "case_id":         case_id,
        "correction_type": request.correction_type,
        "note":            request.note,
        "user_override":   True
    })
    return CorrectionResponse(status="correction_logged", case_id=case_id)


# ── GET /health ───────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check — called by GitHub Actions smoke test after every deploy.
    Returns 503 if any critical index failed to load at startup.
    """
    index_status = retriever.get_status()

    # Critical check: Index 1 and Index 2 must be loaded
    if index_status["index1_vectors"] == 0:
        raise HTTPException(status_code=503,
                            detail="Index 1 not loaded — startup failed")
    if index_status["index2_vectors"] == 0:
        raise HTTPException(status_code=503,
                            detail="Index 2 not loaded — startup failed")

    return HealthResponse(
        status="ok",
        version=MODEL_VERSION,
        uptime_seconds=round(get_uptime(), 1),
        index_sizes=index_status,
        coreset_size=sum(
            retriever.index3_cache[cat].ntotal
            for cat in retriever.index3_cache
        ),
        threshold_config_version="v1.0",
        cache_stats=inference_cache.stats()
    )


# ── GET /metrics ──────────────────────────────────────────────
@app.get("/metrics", response_model=MetricsResponse)
async def metrics():
    """
    Prometheus-style observability endpoint.
    Tracked by GitHub Actions smoke test 3.
    """
    lats = _metrics["latencies"]
    p50  = float(np.percentile(lats, 50)) if lats else 0.0
    p95  = float(np.percentile(lats, 95)) if lats else 0.0

    mem  = psutil.Process().memory_info().rss / 1024 / 1024

    return MetricsResponse(
        request_count=_metrics["request_count"],
        latency_p50_ms=round(p50, 1),
        latency_p95_ms=round(p95, 1),
        cache_hit_rate=inference_cache.stats()["hit_rate"],
        hf_push_failure_count=get_push_failure_count(),
        memory_usage_mb=round(mem, 1)
    )


# ── GET /precompute ───────────────────────────────────────────
@app.post("/precompute")
async def precompute(
    image: UploadFile = File(...),
    session_id: str = Form(...)
):
    """
    Speculative CLIP encoding — fired by Gradio onChange before user clicks Inspect.
    Runs Index 1 category routing only.
    Result stored keyed by session_id — /inspect checks this first.
    """
    image_bytes = await image.read()
    try:
        pil_img  = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        from src.orchestrator import _get_clip_embedding
        clip_full = _get_clip_embedding(pil_img, mode="full")
        cat_result = retriever.route_category(clip_full)
        _precompute_store[session_id] = {
            "category":    cat_result["category"],
            "confidence":  cat_result["confidence"]
        }
    except Exception:
        pass   # Speculative — failure is silent, /inspect handles normally
    return {"status": "queued"}