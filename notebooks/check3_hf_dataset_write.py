# ============================================================
# CHECK 3: HuggingFace Dataset API Write Test
#
# Run this LOCALLY (PowerShell) after installing huggingface_hub
# pip install huggingface_hub
#
# Before running:
# 1. Go to huggingface.co → Settings → Access Tokens
# 2. Create a token with WRITE permissions
# 3. Set it as an environment variable in PowerShell:
#    $env:HF_TOKEN = "hf_xxxxxxxxxxxxx"
# ============================================================

import os
import json
from datetime import datetime
from huggingface_hub import HfApi

# ── Step 1: Load token ──────────────────────────────────────
token = os.environ.get("HF_TOKEN")
if not token:
    raise EnvironmentError(
        "HF_TOKEN not set. Run in PowerShell:\n"
        "$env:HF_TOKEN = 'hf_your_token_here'"
    )
print("Token loaded: YES")

api = HfApi(token=token)

# ── Step 2: Create a private HF Dataset repo ───────────────
# Replace <your-hf-username> with your actual HF username
REPO_ID = "CaffeinatedCoding/anomalyos-logs"

try:
    api.create_repo(
        repo_id=REPO_ID,
        repo_type="dataset",
        private=True,
        exist_ok=True   # no error if repo already exists
    )
    print(f"Dataset repo ready: {REPO_ID}")
except Exception as e:
    print(f"Repo creation failed: {e}")
    raise

# ── Step 3: Push one test record ───────────────────────────
test_record = {
    "timestamp": datetime.utcnow().isoformat(),
    "mode": "check3_test",
    "image_hash": "abc123",
    "anomaly_score": 0.42,
    "is_anomalous": False,
    "message": "Day 1 HF Dataset API write check"
}

# Encode the record as bytes and push as a file in the dataset repo
api.upload_file(
    path_or_fileobj=json.dumps(test_record, indent=2).encode("utf-8"),
    path_in_repo=f"logs/test_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
    repo_id=REPO_ID,
    repo_type="dataset"
)
print("Test record pushed successfully")

# ── Step 4: Verify ─────────────────────────────────────────
print(f"\nVerify manually at:")
print(f"https://huggingface.co/datasets/{REPO_ID}/tree/main/logs")
print("You should see a JSON file there immediately.")
print("\nIf you see it: CHECK PASSED.")
print("If upload errors: check token write permissions. Log in bug_log.md.")

# ── What this proves ───────────────────────────────────────
print("""
WHY THIS MATTERS:
HF Spaces Docker containers are ephemeral.
Every time the Space sleeps or restarts, local files are wiped.
A local inference.jsonl would be destroyed.
This HF Dataset is the durable store that survives restarts.
Every inference AnomalyOS runs will push a row here.
Evidently reads from here for drift monitoring.
If this write fails, the logging strategy is broken before you start.
""")