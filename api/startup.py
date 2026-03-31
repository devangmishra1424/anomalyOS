# api/startup.py
# All model and index loading happens here — once at FastAPI startup
# Everything stays in memory for the entire server lifetime
# Never load models per-request

import os
import json
import time
import torch
import clip

from src.patchcore import patchcore
from src.retriever import retriever
from src.graph import knowledge_graph
from src.depth import depth_estimator
from src.xai import gradcam, shap_explainer
from src.cache import inference_cache
from src.orchestrator import init_orchestrator
from api.logger import init_logger


# Startup timestamp — used for uptime calculation in /health
STARTUP_TIME = None
MODEL_VERSION = "v1.0"


def load_all():
    """
    Called once from FastAPI lifespan on startup.
    Order matters — patchcore before orchestrator, logger before anything logs.
    """
    global STARTUP_TIME
    STARTUP_TIME = time.time()

    print("=" * 50)
    print("AnomalyOS startup sequence")
    print("=" * 50)

    # ── CPU thread tuning ─────────────────────────────────────
    # HF Spaces CPU Basic = 2 vCPU
    # Limit PyTorch threads to match — prevents over-subscription
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float32)
    print(f"PyTorch threads: {torch.get_num_threads()}")

    # ── Logger ────────────────────────────────────────────────
    hf_token = os.environ.get("HF_TOKEN", "")
    init_logger(hf_token)

    # ── PatchCore extractor ───────────────────────────────────
    patchcore.load()

    # ── FAISS indexes ─────────────────────────────────────────
    # Index 3 is lazy-loaded — not loaded here
    retriever.load_indexes()

    # ── Knowledge graph ───────────────────────────────────────
    knowledge_graph.load()

    # ── MiDaS depth estimator ─────────────────────────────────
    try:
        depth_estimator.load()
    except FileNotFoundError as e:
        print(f"WARNING: {e}")
        print("Depth features will return zeros — inference continues")

    # ── CLIP model ────────────────────────────────────────────
    # Loaded here, injected into orchestrator
    print("Loading CLIP ViT-B/32...")
    clip_model, clip_preprocess = clip.load("ViT-B/32", device="cpu")
    clip_model.eval()
    print("CLIP loaded")

    # ── Thresholds ────────────────────────────────────────────
    thresholds_path = os.path.join(
        os.environ.get("DATA_DIR", "data"), "thresholds.json"
    )
    if os.path.exists(thresholds_path):
        with open(thresholds_path) as f:
            thresholds = json.load(f)
        print(f"Thresholds loaded: {len(thresholds)} categories")
    else:
        thresholds = {}
        print("WARNING: thresholds.json not found — using score > 0.5 fallback")

    # ── GradCAM++ ─────────────────────────────────────────────
    try:
        gradcam.load()
    except Exception as e:
        print(f"WARNING: GradCAM++ load failed: {e}")
        print("Forensics mode will run without GradCAM++")

    # ── SHAP background ───────────────────────────────────────
    bg_path = os.path.join(
        os.environ.get("DATA_DIR", "data"), "shap_background.npy"
    )
    shap_explainer.load_background(bg_path)

    # ── Inject into orchestrator ──────────────────────────────
    init_orchestrator(clip_model, clip_preprocess, thresholds)

    elapsed = time.time() - STARTUP_TIME
    print("=" * 50)
    print(f"Startup complete in {elapsed:.1f}s")
    print(f"Model version: {MODEL_VERSION}")
    print("=" * 50)

    return {
        "clip_model": clip_model,
        "clip_preprocess": clip_preprocess,
        "thresholds": thresholds
    }


def get_uptime() -> float:
    if STARTUP_TIME is None:
        return 0.0
    return time.time() - STARTUP_TIME