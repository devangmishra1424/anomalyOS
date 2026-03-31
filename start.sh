#!/bin/bash
# Start FastAPI in background
uvicorn api.main:app --host 0.0.0.0 --port 7860 &

# Wait for FastAPI to be ready
sleep 10

# Start Gradio on port 7861
python app.py