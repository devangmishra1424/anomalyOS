# api/schemas.py
# Pydantic request and response models for all 7 endpoints
# Validation happens here — model is never called on invalid input

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Any
from enum import Enum


VALID_CATEGORIES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut',
    'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush',
    'transistor', 'wood', 'zipper'
]


# ── /inspect ─────────────────────────────────────────────────
class InspectResponse(BaseModel):
    # Core result
    is_anomalous:       bool
    anomaly_score:      float = Field(..., ge=0.0)
    calibrated_score:   float = Field(..., ge=0.0, le=1.0)
    score_std:          float
    category:           str
    model_version:      str

    # Visuals (base64 PNG strings)
    heatmap_b64:        Optional[str] = None
    defect_crop_b64:    Optional[str] = None
    depth_map_b64:      Optional[str] = None

    # Retrieval
    similar_cases:      List[dict] = []

    # Graph context
    graph_context:      dict = {}

    # XAI
    shap_features:      dict = {}

    # LLM report (polled separately)
    report_id:          Optional[str] = None

    # Meta
    latency_ms:         float
    image_hash:         str
    low_confidence:     bool = False   # calibrated_score < 0.3


# ── /report/{report_id} ──────────────────────────────────────
class ReportResponse(BaseModel):
    status:  str    # "pending" | "ready" | "not_found"
    report:  Optional[str] = None


# ── /forensics/{case_id} ─────────────────────────────────────
class ForensicsResponse(BaseModel):
    case_id:              str
    category:             str
    anomaly_score:        float
    calibrated_score:     float
    patch_scores_grid:    List[List[float]]   # [28][28]
    gradcampp_b64:        Optional[str] = None
    shap_features:        dict = {}
    similar_cases:        List[dict] = []
    graph_context:        dict = {}
    retrieval_trace:      List[dict] = []


# ── /knowledge/search ────────────────────────────────────────
class KnowledgeSearchResponse(BaseModel):
    results:       List[dict]
    total_found:   int
    query:         str


# ── /arena/next_case ─────────────────────────────────────────
class ArenaCase(BaseModel):
    case_id:     str
    image_b64:   str
    expert_mode: bool = False   # True if score in [0.45, 0.55]


# ── /arena/submit/{case_id} ──────────────────────────────────
class ArenaSubmitRequest(BaseModel):
    user_rating:   int   = Field(..., ge=0, le=1)
    user_severity: int   = Field(..., ge=1, le=5)
    session_id:    Optional[str] = None

class ArenaSubmitResponse(BaseModel):
    correct_label:        int
    ai_score:             float
    calibrated_score:     float
    user_score:           float
    streak:               int
    top_shap_features:    List[dict]   # top 2 features for post-submission
    heatmap_b64:          Optional[str] = None
    is_expert_case:       bool = False


# ── /correct/{case_id} ───────────────────────────────────────
class CorrectionType(str, Enum):
    false_positive  = "false_positive"
    false_negative  = "false_negative"
    wrong_category  = "wrong_category"

class CorrectionRequest(BaseModel):
    correction_type: CorrectionType
    note:            Optional[str] = Field(None, max_length=500)

class CorrectionResponse(BaseModel):
    status:  str = "correction_logged"
    case_id: str


# ── /health ──────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status:               str
    model_version:        str
    uptime_seconds:       float
    index_sizes:          dict
    coreset_size:         int
    threshold_config_version: str
    cache_stats:          dict


# ── /metrics ─────────────────────────────────────────────────
class MetricsResponse(BaseModel):
    request_count:        int
    latency_p50_ms:       float
    latency_p95_ms:       float
    cache_hit_rate:       float
    hf_push_failure_count: int
    memory_usage_mb:      float