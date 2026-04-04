# src/orchestrator.py
# Hierarchical Multi-Modal Graph RAG Orchestrator
# Routes through 3 FAISS indexes, knowledge graph, XAI, and LLM
# This is the brain — called by POST /inspect

import gc
import time
import base64
import io
import concurrent.futures
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from PIL import Image

import clip
import torch

from src.patchcore import patchcore
from src.retriever import retriever
from src.graph import knowledge_graph
from src.depth import depth_estimator
from src.xai import gradcam, shap_explainer, heatmap_to_base64, image_to_base64
from src.llm import queue_report
from src.cache import inference_cache, get_image_hash, pil_to_bytes

import os
import json


DATA_DIR  = os.environ.get("DATA_DIR", "data")
DEVICE    = "cpu"
IMG_SIZE  = 224

# Loaded at startup by api/startup.py
_clip_model     = None
_clip_preprocess = None
_thresholds     = {}


def init_orchestrator(clip_model, clip_preprocess, thresholds):
    """Called once at FastAPI startup to inject shared models."""
    global _clip_model, _clip_preprocess, _thresholds
    _clip_model      = clip_model
    _clip_preprocess = clip_preprocess
    _thresholds      = thresholds


@dataclass
class OrchestratorResult:
    is_anomalous:       bool
    score:              float          # raw k-NN distance
    calibrated_score:   float          # sigmoid calibrated [0,1]
    score_std:          float          # uncertainty estimate
    category:           str
    heatmap_b64:        Optional[str]  = None
    defect_crop_b64:    Optional[str]  = None
    depth_map_b64:      Optional[str]  = None
    similar_cases:      list           = field(default_factory=list)
    graph_context:      dict           = field(default_factory=dict)
    shap_features:      dict           = field(default_factory=dict)
    report_id:          Optional[str]  = None
    latency_ms:         float          = 0.0
    patch_scores_grid:  Optional[list] = None  # [28,28] for Forensics


@torch.no_grad()
def _get_clip_embedding(pil_img: Image.Image,
                         mode: str = "full") -> np.ndarray:
    """
    CLIP embedding for full image or centre crop.
    mode: 'full' → Index 1 routing
          'crop' → Index 2 retrieval (defect region)
    """
    if mode == "crop":
        from torchvision import transforms as T
        pil_img = T.CenterCrop(112)(pil_img)

    tensor = _clip_preprocess(pil_img).unsqueeze(0).to(DEVICE)
    feat   = _clip_model.encode_image(tensor)
    feat   = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().squeeze().astype(np.float32)


def _extract_defect_crop(pil_img: Image.Image,
                          heatmap: np.ndarray) -> Image.Image:
    """
    Crop 112x112 region centred on anomaly centroid.
    Used as input for Index 2 CLIP embedding.
    """
    cx, cy = patchcore.get_anomaly_centroid(heatmap)
    half   = 56
    left   = max(0, cx - half)
    top    = max(0, cy - half)
    right  = min(IMG_SIZE, cx + half)
    bottom = min(IMG_SIZE, cy + half)
    return pil_img.resize((IMG_SIZE, IMG_SIZE)).crop((left, top, right, bottom))


def _get_fft_features(pil_img: Image.Image) -> dict:
    """FFT texture features — used for SHAP feature vector."""
    import numpy as np
    gray = np.array(pil_img.convert("L"), dtype=np.float32)
    fft  = np.fft.fftshift(np.fft.fft2(gray))
    mag  = np.abs(fft)
    H, W = mag.shape
    cy, cx = H // 2, W // 2
    radius = min(H, W) // 8
    Y, X   = np.ogrid[:H, :W]
    mask   = (X - cx)**2 + (Y - cy)**2 <= radius**2
    low_e  = mag[mask].sum()
    total  = mag.sum() + 1e-10
    return {"low_freq_ratio": float(low_e / total)}


def _get_edge_features(pil_img: Image.Image) -> dict:
    """Edge density — used for SHAP feature vector."""
    import cv2
    gray  = np.array(pil_img.convert("L").resize((IMG_SIZE, IMG_SIZE)))
    edges = cv2.Canny(gray, 50, 150)
    return {"edge_density": float(edges.sum()) / (IMG_SIZE * IMG_SIZE * 255)}


def run_inspection(pil_img: Image.Image,
                    image_bytes: bytes,
                    category_hint: str = None,
                    run_gradcam: bool = False) -> OrchestratorResult:
    """
    Full inspection pipeline.
    
    STEP 1:  Cache check (skip recomputation for repeated images)
    STEP 2:  CLIP full-image → Index 1 category routing
    STEP 3:  WideResNet patches → Index 3 PatchCore scoring
    STEP 4:  Early exit if normal (skip Index 2 + LLM)
    STEP 5:  Defect crop extraction
    STEP 6:  MiDaS depth + CLIP crop embedding IN PARALLEL
    STEP 7:  Index 2 retrieval (similar historical defects)
    STEP 8:  Knowledge graph 2-hop traversal
    STEP 9:  SHAP feature assembly
    STEP 10: LLM report queued (non-blocking)
    STEP 11: GradCAM++ if requested (Forensics mode)
    STEP 12: Calibrate score, assemble result, gc.collect()
    """
    t_start = time.time()

    # ── STEP 1: Cache check ───────────────────────────────────
    image_hash = get_image_hash(image_bytes)
    cached     = inference_cache.get(image_hash)
    if cached:
        cached["latency_ms"] = (time.time() - t_start) * 1000
        return OrchestratorResult(**cached)

    pil_img = pil_img.resize((IMG_SIZE, IMG_SIZE)).convert("RGB")

    # ── STEP 2: Category routing (Index 1) ───────────────────
    clip_full  = _get_clip_embedding(pil_img, mode="full")
    cat_result = retriever.route_category(clip_full)
    category   = category_hint or cat_result["category"]

    # ── STEP 3: PatchCore scoring (Index 3) ──────────────────
    patches = patchcore.extract_patches(pil_img)       # [784, 256]
    score, patch_scores, score_std, nn_dists = retriever.score_patches(
        patches, category
    )

    # ── STEP 4: Early exit — clearly normal ──────────────────
    threshold = _thresholds.get(category, {}).get("threshold", 0.5)
    if score < threshold:
        calibrated = patchcore.calibrate_score(score, category, _thresholds)
        result_data = dict(
            is_anomalous=False,
            score=score,
            calibrated_score=calibrated,
            score_std=score_std,
            category=category,
            heatmap_b64=None,
            patch_scores_grid=patch_scores.tolist()
        )
        inference_cache.set(image_hash, result_data)
        gc.collect()
        return OrchestratorResult(
            **result_data,
            latency_ms=(time.time() - t_start) * 1000
        )

    # ── STEP 5: Heatmap + defect crop ────────────────────────
    heatmap      = patchcore.build_anomaly_map(patch_scores)
    heatmap_b64  = heatmap_to_base64(heatmap, pil_img)
    defect_crop  = _extract_defect_crop(pil_img, heatmap)
    crop_b64     = image_to_base64(defect_crop, size=(112, 112))

    # ── STEP 6: MiDaS + CLIP crop IN PARALLEL ────────────────
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        depth_future = ex.submit(depth_estimator.get_depth_stats, pil_img)
        depth_map_f  = ex.submit(depth_estimator.get_depth_map, pil_img)
        clip_future  = ex.submit(_get_clip_embedding, defect_crop, "crop")

    depth_stats  = depth_future.result()
    depth_map    = depth_map_f.result()
    clip_crop    = clip_future.result()

    # Encode depth map
    depth_norm   = (depth_map * 255).astype(np.uint8)
    depth_pil    = Image.fromarray(depth_norm)
    depth_b64    = image_to_base64(depth_pil)

    # ── STEP 7: Index 2 retrieval ─────────────────────────────
    similar_cases = retriever.retrieve_similar_defects(
        clip_crop, k=5, exclude_hash=image_hash,
        category_filter=category
    )

    # ── STEP 8: Knowledge graph traversal ────────────────────
    # Use top retrieved defect type for graph lookup
    top_defect_type = (similar_cases[0]["defect_type"]
                       if similar_cases else "unknown")
    graph_context = knowledge_graph.get_context(category, top_defect_type)

    # ── STEP 9: SHAP features ────────────────────────────────
    fft_feats  = _get_fft_features(pil_img)
    edge_feats = _get_edge_features(pil_img)
    feat_vec   = shap_explainer.build_feature_vector(
        patch_scores, depth_stats, fft_feats, edge_feats
    )
    shap_result = shap_explainer.explain(feat_vec)

    # ── STEP 10: LLM report (non-blocking) ───────────────────
    report_id = queue_report(category, score, similar_cases, graph_context)

    # ── STEP 11: GradCAM++ (Forensics only) ──────────────────
    # Not run during normal Inspector Mode — too slow for default path
    # Called explicitly from POST /forensics/{case_id}

    # ── STEP 12: Calibrate + assemble ────────────────────────
    calibrated = patchcore.calibrate_score(score, category, _thresholds)

    result_data = dict(
        is_anomalous=True,
        score=score,
        calibrated_score=calibrated,
        score_std=score_std,
        category=category,
        heatmap_b64=heatmap_b64,
        defect_crop_b64=crop_b64,
        depth_map_b64=depth_b64,
        similar_cases=similar_cases,
        graph_context=graph_context,
        shap_features=shap_result,
        report_id=report_id,
        patch_scores_grid=patch_scores.tolist()
    )

    inference_cache.set(image_hash, result_data)
    gc.collect()

    return OrchestratorResult(
        **result_data,
        latency_ms=(time.time() - t_start) * 1000
    )