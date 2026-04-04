#!/bin/bash
# FastAPI on internal port 8000
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Wait for FastAPI to be ready
sleep 15

# Gradio on 7860 (HF Spaces public port)
python app.py