# src/llm.py
# Groq LLM call with tenacity retry
# Single call per inference — not a multi-step chain
# Non-blocking: queued as FastAPI BackgroundTask, polled via /report/{id}

import os
import json
import time
import uuid
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)


GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.3-70b-versatile"
MAX_TOKENS   = 512

# In-memory report store: report_id → {status, report}
# FastAPI polls this via GET /report/{report_id}
_report_store: dict = {}


class LLMAPIError(Exception):
    pass


def _build_prompt(category: str,
                   anomaly_score: float,
                   similar_cases: list,
                   graph_context: dict) -> list:
    """
    Build LLM messages list.
    Strictly grounded — model must cite case IDs, cannot use outside knowledge.
    One call per inference. Context = retrieved cases + graph context.
    """
    system = (
        "You are an industrial quality control assistant. "
        "Answer ONLY based on the retrieved cases and graph context provided. "
        "Do not use outside knowledge. "
        "Always cite the Case ID when referencing a case. "
        "Be concise — 3 to 5 sentences maximum."
    )

    # Build context block from retrieved similar cases
    context_lines = []
    for i, case in enumerate(similar_cases[:5]):
        context_lines.append(
            f"[Case {i+1}: category={case.get('category')}, "
            f"defect={case.get('defect_type')}, "
            f"similarity={case.get('similarity_score', 0):.3f}]"
        )

    # Add graph context
    root_causes   = graph_context.get("root_causes", [])
    remediations  = graph_context.get("remediations", [])
    if root_causes:
        context_lines.append(f"Root causes: {', '.join(root_causes)}")
    if remediations:
        context_lines.append(f"Remediations: {', '.join(remediations)}")

    context_str = "\n".join(context_lines) if context_lines else "No context available."

    user_msg = (
        f"CONTEXT:\n{context_str}\n\n"
        f"QUERY: Image anomaly score {anomaly_score:.3f}. "
        f"Category: {category}. "
        f"Describe the likely defect, root cause, and recommended action."
        f"\n\nREPORT:"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_msg}
    ]


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    retry=retry_if_exception_type(LLMAPIError),
    reraise=True
)
def _call_groq(messages: list) -> str:
    """
    Single Groq API call with tenacity retry.
    Retries 3 times with 2s/4s/8s backoff on failure.
    Raises LLMAPIError if all 3 attempts fail.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise LLMAPIError("GROQ_API_KEY not set in environment")

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type":  "application/json"
                },
                json={
                    "model":       GROQ_MODEL,
                    "messages":    messages,
                    "max_tokens":  MAX_TOKENS,
                    "temperature": 0.3    # low temp = factual, grounded
                }
            )

        if response.status_code == 429:
            raise LLMAPIError("Groq rate limit hit")
        if response.status_code != 200:
            raise LLMAPIError(f"Groq API error {response.status_code}: "
                               f"{response.text[:200]}")

        data    = response.json()
        content = data["choices"][0]["message"]["content"].strip()

        if not content:
            raise LLMAPIError("Groq returned empty response")

        return content

    except httpx.TimeoutException:
        raise LLMAPIError("Groq API timeout")
    except httpx.RequestError as e:
        raise LLMAPIError(f"Groq request failed: {e}")


def queue_report(category: str,
                  anomaly_score: float,
                  similar_cases: list,
                  graph_context: dict) -> str:
    """
    Queue an LLM report generation.
    Returns report_id immediately — report generated asynchronously.
    Frontend polls GET /report/{report_id} every 500ms.
    """
    report_id = str(uuid.uuid4())
    _report_store[report_id] = {"status": "pending", "report": None}
    return report_id


def generate_report(report_id: str,
                     category: str,
                     anomaly_score: float,
                     similar_cases: list,
                     graph_context: dict):
    """
    Called as FastAPI BackgroundTask.
    Generates report and stores in _report_store under report_id.
    """
    try:
        messages = _build_prompt(category, anomaly_score,
                                  similar_cases, graph_context)
        report = _call_groq(messages)
        _report_store[report_id] = {"status": "ready", "report": report}

    except LLMAPIError as e:
        fallback = (
            "LLM temporarily unavailable. "
            "Retrieved cases and graph context are shown above. "
            f"(Error: {str(e)[:100]})"
        )
        _report_store[report_id] = {"status": "ready", "report": fallback}

    except Exception as e:
        _report_store[report_id] = {
            "status": "ready",
            "report": "Could not generate report. Please retry."
        }


def get_report(report_id: str) -> dict:
    """
    Poll report status.
    Returns: {status: pending} or {status: ready, report: "..."}
    """
    return _report_store.get(
        report_id,
        {"status": "not_found", "report": None}
    )


def cleanup_old_reports(max_age_seconds: int = 3600):
    """Prevent _report_store growing unbounded. Called periodically."""
    # Simple approach: keep only last 500 reports
    if len(_report_store) > 500:
        keys = list(_report_store.keys())
        for key in keys[:250]:
            del _report_store[key]