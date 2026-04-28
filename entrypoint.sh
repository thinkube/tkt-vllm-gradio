#!/bin/bash
set -e

echo "=== vLLM Inference Server Startup ==="

# Configure HuggingFace to use local models only (no network access)
# Models are stored in MLflow artifacts on JuiceFS at /mlflow-models/artifacts/{run_id}/artifacts/model
export HF_HUB_OFFLINE=1
export HF_HOME="/mlflow-models"

echo "HuggingFace offline mode enabled (HF_HUB_OFFLINE=1)"
echo "Starting in idle mode - waiting for model load via /admin/switch-model"

# Start Gradio frontend (handles admin endpoints and proxies to vllm serve)
exec python3 server.py
