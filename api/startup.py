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

def download_artifacts():
    """Download all required artifacts from HF Dataset at startup."""
    from huggingface_hub import hf_hub_download, snapshot_download
    import shutil

    HF_REPO = "CaffeinatedCoding/anomalyos-logs"
    token = os.environ.get("HF_TOKEN")
    
    os.makedirs("data", exist_ok=True)

    files_to_download = [
        ("models/pca_256.pkl",              "data/pca_256.pkl"),
        ("configs/thresholds.json",          "data/thresholds.json"),
        ("graph/knowledge_graph.json",       "data/knowledge_graph.json"),
        ("indexes/index1_category.faiss",    "data/index1_category.faiss"),
        ("indexes/index1_metadata.json",     "data/index1_metadata.json"),
        ("indexes/index2_defect.faiss",      "data/index2_defect.faiss"),
        ("indexes/index2_metadata.json",     "data/index2_metadata.json"),
    ]

    # Index 3 — one per category
    categories = [
        'bottle','cable','capsule','carpet','grid','hazelnut',
        'leather','metal_nut','pill','screw','tile','toothbrush',
        'transistor','wood','zipper'
    ]
    for cat in categories:
        files_to_download.append((
            f"indexes/index3_{cat}.faiss",
            f"data/index3_{cat}.faiss"
        ))

    for repo_path, local_path in files_to_download:
        if os.path.exists(local_path):
            print(f"Already exists: {local_path}")
            continue
        try:
            print(f"Downloading {repo_path}...")
            downloaded = hf_hub_download(
                repo_id=HF_REPO,
                filename=repo_path,
                repo_type="dataset",
                token=token,
                local_dir="/tmp/artifacts"
            )
            shutil.copy(downloaded, local_path)
            print(f"  → {local_path}")
        except Exception as e:
            print(f"  WARNING: Could not download {repo_path}: {e}")

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

    # Download artifacts first
    download_artifacts()
    

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
    print("Loading CLIP ViT-B/32...", flush=True)
    try:
        print("  [Downloading CLIP weights...]", flush=True)
        clip_model, clip_preprocess = clip.load("ViT-B/32", device="cpu")
        print("  [CLIP weights loaded, setting eval mode...]", flush=True)
        clip_model.eval()
        print("CLIP loaded ✓", flush=True)
    except Exception as e:
        print(f"ERROR loading CLIP: {e}", flush=True)
        raise

    print("Loading thresholds...", flush=True)
    thresholds_path = os.path.join(
        os.environ.get("DATA_DIR", "data"), "thresholds.json"
    )
    if os.path.exists(thresholds_path):
        with open(thresholds_path) as f:
            thresholds = json.load(f)
        print(f"Thresholds loaded ✓ {len(thresholds)} categories", flush=True)
    else:
        thresholds = {}
        print("WARNING: thresholds.json not found — using score > 0.5 fallback", flush=True)

    # ── GradCAM++ ─────────────────────────────────────────────
    print("Loading GradCAM++...", flush=True)
    try:
        gradcam.load()
        print("GradCAM++ loaded ✓", flush=True)
    except Exception as e:
        print(f"WARNING: GradCAM++ load failed: {e}", flush=True)
        print("Forensics mode will run without GradCAM++", flush=True)

    # ── SHAP background ───────────────────────────────────────
    print("Loading SHAP background...", flush=True)
    bg_path = os.path.join(
        os.environ.get("DATA_DIR", "data"), "shap_background.npy"
    )
    try:
        if os.path.exists(bg_path):
            shap_explainer.load_background(bg_path)
            print("SHAP background loaded ✓", flush=True)
        else:
            print(f"WARNING: SHAP background not found at {bg_path}", flush=True)
            print("SHAP explanations will use default background", flush=True)
    except Exception as e:
        print(f"WARNING: SHAP background load failed: {e}", flush=True)
        print("SHAP explanations will use default background", flush=True)

    # ── Inject into orchestrator ──────────────────────────────
    print("Initializing orchestrator...", flush=True)
    try:
        init_orchestrator(clip_model, clip_preprocess, thresholds)
        print("Orchestrator initialized ✓", flush=True)
    except Exception as e:
        print(f"ERROR initializing orchestrator: {e}", flush=True)
        raise

    elapsed = time.time() - STARTUP_TIME
    print("=" * 50, flush=True)
    print(f"Startup complete in {elapsed:.1f}s ✓", flush=True)
    print(f"Model version: {MODEL_VERSION}", flush=True)
    print("=" * 50, flush=True)

    return {
        "clip_model": clip_model,
        "clip_preprocess": clip_preprocess,
        "thresholds": thresholds
    }


def get_uptime() -> float:
    if STARTUP_TIME is None:
        return 0.0
    return time.time() - STARTUP_TIME