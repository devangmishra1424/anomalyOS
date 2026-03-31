# api/logger.py
# Two-layer durable logging strategy
#
# Layer 1: Local JSONL (fast write, ephemeral — wiped on HF Space restart)
#           Used by Evidently drift scripts
# Layer 2: HF Dataset API push (durable, survives restarts)
#           Called as FastAPI BackgroundTask — never blocks response
#
# If HF push fails: user unaffected, local log still written

import os
import json
import time
from datetime import datetime, timezone
from huggingface_hub import HfApi


HF_REPO_ID    = os.environ.get("HF_LOG_REPO", "CaffeinatedCoding/anomalyos-logs")
LOCAL_LOG_DIR = "logs"
LOCAL_LOG_PATH = os.path.join(LOCAL_LOG_DIR, "inference.jsonl")

_hf_api: HfApi = None
_hf_push_failure_count: int = 0


def init_logger(hf_token: str):
    """Called once at FastAPI startup."""
    global _hf_api
    os.makedirs(LOCAL_LOG_DIR, exist_ok=True)
    if hf_token:
        _hf_api = HfApi(token=hf_token)
        print(f"Logger initialised | HF repo: {HF_REPO_ID}")
    else:
        print("WARNING: HF_TOKEN not set — only local logging active")


def log_inference(record: dict):
    """
    Layer 1: write to local JSONL synchronously.
    Called as BackgroundTask from FastAPI — does not block response.
    """
    global _hf_push_failure_count

    # Ensure timestamp
    if "timestamp" not in record:
        record["timestamp"] = datetime.now(timezone.utc).isoformat()

    # ── Layer 1: Local JSONL ──────────────────────────────────
    try:
        with open(LOCAL_LOG_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        print(f"Local log write failed: {e}")

    # ── Layer 2: HF Dataset push ─────────────────────────────
    if _hf_api is None:
        return

    try:
        ts = record.get("timestamp", datetime.now(timezone.utc).isoformat())
        # Sanitise timestamp for filename
        ts_safe = ts.replace(":", "-").replace(".", "-")[:26]
        path_in_repo = f"inference_logs/{ts_safe}_{record.get('image_hash', 'unknown')[:8]}.json"

        _hf_api.upload_file(
            path_or_fileobj=json.dumps(record, indent=2).encode("utf-8"),
            path_in_repo=path_in_repo,
            repo_id=HF_REPO_ID,
            repo_type="dataset"
        )
    except Exception as e:
        _hf_push_failure_count += 1
        print(f"HF Dataset push failed (count={_hf_push_failure_count}): {e}")
        # User response is completely unaffected — local log already written


def log_arena_submission(record: dict):
    """Log Arena Mode submissions to shared leaderboard dataset."""
    record["log_type"] = "arena"
    log_inference(record)


def log_correction(record: dict):
    """Log user corrections from /correct/{case_id}."""
    record["log_type"] = "correction"
    log_inference(record)


def get_recent_logs(n: int = 200) -> list:
    """
    Read last n records from local JSONL.
    Used by Evidently drift scripts.
    """
    if not os.path.exists(LOCAL_LOG_PATH):
        return []
    records = []
    try:
        with open(LOCAL_LOG_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except Exception as e:
        print(f"Error reading local log: {e}")
    return records[-n:]


def get_push_failure_count() -> int:
    return _hf_push_failure_count