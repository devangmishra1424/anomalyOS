# app.py
# Gradio frontend — 5 tabs
# Calls FastAPI endpoints running on the same container
# Launched separately from uvicorn — both run in the same HF Space

import gradio as gr
import httpx
import base64
import time
import json
import uuid
from PIL import Image
import io
import numpy as np


API_BASE = "http://localhost:7860"
SESSION_ID = str(uuid.uuid4())

CATEGORIES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut',
    'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush',
    'transistor', 'wood', 'zipper'
]


# ── Helpers ───────────────────────────────────────────────────
def b64_to_pil(b64_str: str) -> Image.Image:
    if not b64_str:
        return None
    return Image.open(io.BytesIO(base64.b64decode(b64_str)))


def call_inspect(image: Image.Image, category_hint: str) -> dict:
    """POST /inspect with image file."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)
    with httpx.Client(timeout=120) as client:
        resp = client.post(
            f"{API_BASE}/inspect",
            files={"image": ("image.jpg", buf, "image/jpeg")},
            data={"category_hint": category_hint or "",
                  "session_id": SESSION_ID}
        )
    if resp.status_code != 200:
        raise ValueError(f"Inspect failed: {resp.status_code} {resp.text[:200]}")
    return resp.json()


def poll_report(report_id: str, max_wait: int = 30) -> str:
    """Poll /report/{report_id} until ready or timeout."""
    with httpx.Client(timeout=10) as client:
        for _ in range(max_wait * 2):   # poll every 500ms
            resp = client.get(f"{API_BASE}/report/{report_id}")
            data = resp.json()
            if data.get("status") == "ready":
                return data.get("report", "No report generated.")
            time.sleep(0.5)
    return "Report generation timed out."


# ── Tab 1: Inspector ──────────────────────────────────────────
def run_inspector(image, category_hint, last_click_state):
    """Main inspection function with 3-second debounce."""
    if image is None:
        return (None, None, None,
                "Upload an image first.", "", "", None)

    # 3-second debounce — prevents Groq rate limit hammering
    now = time.time()
    last_click = last_click_state or 0
    if now - last_click < 3:
        return (None, None, None,
                "⏳ Please wait 3 seconds between requests.", "", "", now)

    try:
        result = call_inspect(image, category_hint)
    except Exception as e:
        return (None, None, None,
                f"❌ Error: {str(e)}", "", "", now)

    # Decode visuals
    heatmap_img  = b64_to_pil(result.get("heatmap_b64"))
    crop_img     = b64_to_pil(result.get("defect_crop_b64"))
    depth_img    = b64_to_pil(result.get("depth_map_b64"))

    # Build score display
    score    = result.get("calibrated_score", 0)
    category = result.get("category", "unknown")
    is_anom  = result.get("is_anomalous", False)

    if is_anom:
        decision = f"⚠️ DEFECT DETECTED — {category.upper()}"
        score_text = f"Anomaly confidence: {score:.1%}"
    else:
        decision = f"✅ NORMAL — {category.upper()}"
        score_text = f"Anomaly confidence: {score:.1%}"

    latency = result.get("latency_ms", 0)
    meta    = (f"Category: {category} | "
               f"Raw score: {result.get('anomaly_score', 0):.4f} | "
               f"Latency: {latency:.0f}ms | "
               f"Model: {result.get('version', 'v1.0')}")

    # Poll LLM report
    report_id = result.get("report_id")
    report    = ""
    if report_id and is_anom:
        report = poll_report(report_id, max_wait=20)

    # Store case_id for Forensics tab
    case_id = result.get("image_hash", "")

    return (heatmap_img, crop_img, depth_img,
            f"{decision}\n{score_text}\n{meta}",
            report, case_id, now)


def build_similar_cases_html(similar_cases: list) -> str:
    if not similar_cases:
        return "<p>No similar cases retrieved.</p>"
    rows = []
    for i, case in enumerate(similar_cases[:5]):
        rows.append(
            f"<div style='margin:8px;padding:8px;border:1px solid #444;border-radius:6px'>"
            f"<b>#{i+1}</b> {case.get('category','?')} / {case.get('defect_type','?')} "
            f"| similarity: {case.get('similarity_score',0):.3f}"
            f"</div>"
        )
    return "".join(rows)


# ── Tab 2: Forensics ──────────────────────────────────────────
def run_forensics(case_id: str):
    if not case_id:
        return None, None, "{}", "Enter a case ID from Inspector."

    with httpx.Client(timeout=60) as client:
        resp = client.post(f"{API_BASE}/forensics/{case_id}")

    if resp.status_code == 422:
        return None, None, "{}", "Case not found. Run an inspection first."
    if resp.status_code != 200:
        return None, None, "{}", f"Error: {resp.status_code}"

    data = resp.json()

    gradcam_img   = b64_to_pil(data.get("gradcampp_b64"))
    shap_json     = json.dumps(data.get("shap_features", {}), indent=2)
    retrieval_txt = "\n".join([
        f"{i+1}. {t.get('category')}/{t.get('defect_type')} "
        f"(sim={t.get('similarity_score',0):.3f}) → {t.get('graph_path','')}"
        for i, t in enumerate(data.get("retrieval_trace", []))
    ])

    summary = (
        f"Category: {data.get('category')} | "
        f"Score: {data.get('anomaly_score', 0):.4f} | "
        f"Calibrated: {data.get('calibrated_score', 0):.3f}"
    )

    return gradcam_img, summary, shap_json, retrieval_txt


# ── Tab 3: Analytics ──────────────────────────────────────────
def load_analytics():
    try:
        with httpx.Client(timeout=10) as client:
            health = client.get(f"{API_BASE}/health").json()
            mets   = client.get(f"{API_BASE}/metrics").json()
        return (
            f"Requests: {mets.get('request_count',0)} | "
            f"P50: {mets.get('latency_p50_ms',0)}ms | "
            f"P95: {mets.get('latency_p95_ms',0)}ms | "
            f"Cache hit rate: {mets.get('cache_hit_rate',0):.1%} | "
            f"Memory: {mets.get('memory_usage_mb',0):.0f}MB\n\n"
            f"Index sizes: {json.dumps(health.get('index_sizes',{}), indent=2)}"
        )
    except Exception as e:
        return f"Could not load analytics: {e}"


# ── Tab 4: Arena ──────────────────────────────────────────────
_arena_state = {"case_id": None, "streak": 0}


def get_arena_case(expert_mode: bool):
    with httpx.Client(timeout=30) as client:
        resp = client.get(f"{API_BASE}/arena/next_case",
                          params={"expert_mode": expert_mode})
    if resp.status_code != 200:
        return None, "Failed to load case.", None

    data    = resp.json()
    case_id = data["case_id"]
    _arena_state["case_id"] = case_id
    img     = b64_to_pil(data["image_b64"])
    label   = "⚡ EXPERT CASE" if data.get("expert_mode") else "Standard case"
    return img, label, case_id


def submit_arena(user_rating: int, user_severity: int, case_id: str):
    if not case_id:
        return "Load a case first.", "", None

    with httpx.Client(timeout=60) as client:
        resp = client.post(
            f"{API_BASE}/arena/submit/{case_id}",
            json={"user_rating": user_rating,
                  "user_severity": user_severity,
                  "session_id": SESSION_ID}
        )

    if resp.status_code != 200:
        return f"Error: {resp.status_code}", "", None

    data   = resp.json()
    streak = data.get("streak", 0)
    score  = data.get("user_score", 0)
    correct_label = data.get("correct_label", 0)
    ai_cal = data.get("calibrated_score", 0)

    result_txt = (
        f"{'✅ CORRECT' if int(user_rating) == correct_label else '❌ WRONG'}\n"
        f"Ground truth: {'DEFECTIVE' if correct_label else 'NORMAL'}\n"
        f"AI confidence: {ai_cal:.1%}\n"
        f"Your score: {score:.1f} | Streak: 🔥 {streak}"
    )

    shap_txt = ""
    for feat in data.get("top_shap_features", []):
        shap_txt += (f"{feat['feature']}: "
                     f"{feat['contribution']:+.4f}\n")

    heatmap_img = b64_to_pil(data.get("heatmap_b64"))
    return result_txt, f"Why the AI scored this:\n{shap_txt}", heatmap_img


# ── Tab 5: Knowledge Base ─────────────────────────────────────
def search_knowledge(query: str, category: str, defect_type: str):
    params = {}
    if query:
        params["query"] = query
    if category and category != "All":
        params["category"] = category
    if defect_type:
        params["defect_type"] = defect_type

    with httpx.Client(timeout=30) as client:
        resp = client.get(f"{API_BASE}/knowledge/search", params=params)

    if resp.status_code != 200:
        return f"Search failed: {resp.status_code}"

    data    = resp.json()
    results = data.get("results", [])
    total   = data.get("total_found", 0)

    if not results:
        return "No results found."

    lines = [f"Found {total} results:\n"]
    for r in results[:20]:
        lines.append(
            f"• {r.get('category','?')} / {r.get('defect_type','?')} "
            f"| severity: {r.get('severity_min',0):.1f}–{r.get('severity_max',1):.1f}"
        )
    return "\n".join(lines)


# ── Build Gradio UI ───────────────────────────────────────────
with gr.Blocks(title="AnomalyOS", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 🔍 AnomalyOS — Industrial Visual Intelligence Platform")
    gr.Markdown("*Zero training on defects. The AI only knows normal.*")

    with gr.Tabs():

        # ── INSPECTOR TAB ─────────────────────────────────────
        with gr.Tab("🔬 Inspector"):
            with gr.Row():
                with gr.Column(scale=1):
                    inp_image    = gr.Image(type="pil", label="Upload Product Image")
                    inp_category = gr.Dropdown(
                        choices=[""] + CATEGORIES,
                        label="Category hint (optional)",
                        value=""
                    )
                    btn_inspect  = gr.Button("🔍 Inspect", variant="primary")
                    gr.Markdown("*3-second cooldown between requests*")

                with gr.Column(scale=2):
                    out_heatmap  = gr.Image(label="Anomaly Heatmap")
                    out_crop     = gr.Image(label="Defect Crop")
                    out_depth    = gr.Image(label="Depth Map")
                    out_decision = gr.Textbox(label="Result", lines=3)
                    out_report   = gr.Textbox(label="AI Defect Report", lines=5)
                    out_case_id  = gr.Textbox(label="Case ID (use in Forensics)",
                                              interactive=False)

            # Correction widget
            with gr.Accordion("⚠️ Is this wrong?", open=False):
                corr_type = gr.Dropdown(
                    choices=["false_positive", "false_negative", "wrong_category"],
                    label="Correction type"
                )
                corr_note = gr.Textbox(label="Optional note", max_lines=2)
                btn_corr  = gr.Button("Submit Correction")
                corr_out  = gr.Textbox(label="Status", interactive=False)

            # State
            last_click = gr.State(value=0)

            btn_inspect.click(
                fn=run_inspector,
                inputs=[inp_image, inp_category, last_click],
                outputs=[out_heatmap, out_crop, out_depth,
                         out_decision, out_report, out_case_id, last_click]
            )

        # ── FORENSICS TAB ─────────────────────────────────────
        with gr.Tab("🧬 Forensics"):
            with gr.Row():
                f_case_input = gr.Textbox(
                    label="Case ID (paste from Inspector)",
                    placeholder="SHA256 hash from Inspector result"
                )
                btn_forensics = gr.Button("🔬 Deep Analyse", variant="primary")

            with gr.Row():
                f_gradcam   = gr.Image(label="GradCAM++ Overlay")
                f_summary   = gr.Textbox(label="Case Summary", lines=2)

            with gr.Row():
                f_shap      = gr.Code(label="SHAP Features (JSON)",
                                       language="json")
                f_retrieval = gr.Textbox(label="Retrieval Trace", lines=8)

            btn_forensics.click(
                fn=run_forensics,
                inputs=[f_case_input],
                outputs=[f_gradcam, f_summary, f_shap, f_retrieval]
            )

        # ── ANALYTICS TAB ─────────────────────────────────────
        with gr.Tab("📊 Analytics"):
            btn_refresh   = gr.Button("🔄 Refresh")
            analytics_out = gr.Textbox(label="System Stats", lines=15)

            btn_refresh.click(
                fn=load_analytics,
                inputs=[],
                outputs=[analytics_out]
            )
            demo.load(fn=load_analytics, inputs=[], outputs=[analytics_out])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)