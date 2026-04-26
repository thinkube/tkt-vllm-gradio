#!/bin/bash
set -e

MODEL_ID="${MODEL_ID:?MODEL_ID environment variable is required}"

echo "=== vLLM Inference Server Startup ==="
echo "Model: ${MODEL_ID}"

# Configure HuggingFace to use local models only (no network access)
# Models are stored in MLflow artifacts on JuiceFS at /mlflow-models/artifacts/{run_id}/artifacts/model
export HF_HUB_OFFLINE=1
export HF_HOME="/mlflow-models"

echo "HuggingFace offline mode enabled (HF_HUB_OFFLINE=1)"
echo "Models will be loaded from MLflow artifacts on JuiceFS"

# Query MLflow to get the run_id for the latest version of this model
echo "Querying MLflow for model location..."
MODEL_PATH=$(python3 << 'PYTHON_SCRIPT'
import os
import sys
import requests

# Get auth token
token_url = os.environ['MLFLOW_KEYCLOAK_TOKEN_URL']
token_response = requests.post(
    token_url,
    data={
        'grant_type': 'password',
        'client_id': os.environ['MLFLOW_KEYCLOAK_CLIENT_ID'],
        'client_secret': os.environ['MLFLOW_CLIENT_SECRET'],
        'username': os.environ['MLFLOW_AUTH_USERNAME'],
        'password': os.environ['MLFLOW_AUTH_PASSWORD'],
        'scope': 'openid'
    },
    verify=False,
    timeout=30
)
token_response.raise_for_status()
access_token = token_response.json()['access_token']

# Query MLflow for model
model_id = os.environ['MODEL_ID']
model_name = model_id.replace('/', '-')
mlflow_url = os.environ.get('MLFLOW_TRACKING_URI', 'http://mlflow.mlflow.svc.cluster.local:5000')

response = requests.get(
    f"{mlflow_url}/api/2.0/mlflow/model-versions/search",
    params={'filter': f"name='{model_name}'"},
    headers={'Authorization': f'Bearer {access_token}'},
    verify=False,
    timeout=30
)
response.raise_for_status()

versions = response.json().get('model_versions', [])
if not versions:
    print(f"ERROR: Model {model_name} not found in MLflow registry", file=sys.stderr)
    sys.exit(1)

# Get latest version
latest = max(versions, key=lambda v: int(v['version']))
run_id = latest['run_id']

print(f"Found model version {latest['version']} with run_id: {run_id}", file=sys.stderr)

# Get run details to retrieve experiment_id
run_response = requests.get(
    f"{mlflow_url}/api/2.0/mlflow/runs/get",
    params={'run_id': run_id},
    headers={'Authorization': f'Bearer {access_token}'},
    verify=False,
    timeout=30
)
run_response.raise_for_status()
experiment_id = run_response.json()['run']['info']['experiment_id']

print(f"Experiment ID: {experiment_id}", file=sys.stderr)

# Get artifact path from run_id with experiment_id
# Models are stored at: /mlflow-models/artifacts/{experiment_id}/{run_id}/artifacts/model
model_path = f'/mlflow-models/artifacts/{experiment_id}/{run_id}/artifacts/model'

# Verify the path exists
if not os.path.exists(model_path):
    print(f"ERROR: Model path does not exist: {model_path}", file=sys.stderr)
    sys.exit(1)

print(f"Model found at: {model_path}", file=sys.stderr)

# Output the path to stdout (captured by shell)
print(model_path)
PYTHON_SCRIPT
)

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to locate model in MLflow"
    exit 1
fi

echo "Using model path: $MODEL_PATH"

# Export MODEL_PATH for Python server to use
export MODEL_PATH

# Start vllm serve backend for OpenAI-compatible inference
echo "Starting vllm serve backend on port 8355..."

vllm serve "$MODEL_PATH" \
    --host 127.0.0.1 \
    --port 8355 \
    --dtype auto \
    --served-model-name "$MODEL_ID" &

VLLM_PID=$!
echo "$VLLM_PID" > /tmp/vllm.pid
echo "vllm serve started with PID $VLLM_PID"

# Wait for vllm serve to be ready
echo "Waiting for vllm serve to be ready..."
MAX_RETRIES=120  # 10 minutes max (model loading can take time)
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://127.0.0.1:8355/health > /dev/null 2>&1; then
        echo "vllm serve is ready!"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $((RETRY_COUNT % 10)) -eq 0 ]; then
        echo "Still waiting for vllm serve... ($RETRY_COUNT/$MAX_RETRIES)"
    fi
    sleep 5
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "ERROR: vllm serve failed to start within timeout"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

# Start Gradio frontend (proxies to vllm serve and handles admin endpoints)
echo "Starting Gradio frontend on port 7860..."
exec python3 server.py
