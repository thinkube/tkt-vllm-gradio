#!/usr/bin/env python3
"""
vLLM Inference Server with Gradio UI and Admin Management Endpoints
Uses vllm serve as a subprocess for model switching support
"""

import os
import json
import time
import signal
import logging
import asyncio
import subprocess

import httpx
import requests as req_lib
import gradio as gr
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
from transformers import AutoTokenizer
from thinkube_theme import create_thinkube_theme, THINKUBE_CSS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_ID = None
MODEL_PATH = None

VLLM_BACKEND_URL = "http://127.0.0.1:8355"
VLLM_PID_FILE = "/tmp/vllm.pid"

app = FastAPI(title="vLLM Server")

backend_start_time = None
is_switching = False
tokenizer = None
http_client = None
sync_http_client = None

model_stop_tokens: list = []
model_reasoning_format: str | None = None
model_tool_use: bool = False


def query_mlflow(model_id: str) -> str:
    """Query MLflow to get the JuiceFS artifact path for a model."""
    token_response = req_lib.post(
        os.environ['MLFLOW_KEYCLOAK_TOKEN_URL'],
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

    model_name = model_id.replace('/', '-')
    mlflow_url = os.environ.get('MLFLOW_TRACKING_URI', 'http://mlflow.mlflow.svc.cluster.local:5000')

    response = req_lib.get(
        f"{mlflow_url}/api/2.0/mlflow/model-versions/search",
        params={'filter': f"name='{model_name}'"},
        headers={'Authorization': f'Bearer {access_token}'},
        verify=False,
        timeout=30
    )
    response.raise_for_status()

    versions = response.json().get('model_versions', [])
    if not versions:
        raise ValueError(f"Model {model_name} not found in MLflow registry")

    latest = max(versions, key=lambda v: int(v['version']))
    run_id = latest['run_id']

    run_response = req_lib.get(
        f"{mlflow_url}/api/2.0/mlflow/runs/get",
        params={'run_id': run_id},
        headers={'Authorization': f'Bearer {access_token}'},
        verify=False,
        timeout=30
    )
    run_response.raise_for_status()
    experiment_id = run_response.json()['run']['info']['experiment_id']

    model_path = f'/mlflow-models/artifacts/{experiment_id}/{run_id}/artifacts/model'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    return model_path


def stop_backend():
    """Stop the current vllm subprocess using the PID file."""
    try:
        with open(VLLM_PID_FILE) as f:
            pid = int(f.read().strip())
        logger.info(f"Stopping vllm serve (PID {pid})...")
        os.kill(pid, signal.SIGTERM)
        for _ in range(60):
            try:
                os.kill(pid, 0)
                time.sleep(0.5)
            except ProcessLookupError:
                break
        else:
            logger.warning(f"Force-killing vllm serve (PID {pid})")
            try:
                os.kill(pid, signal.SIGKILL)
                time.sleep(1)
            except ProcessLookupError:
                pass
        logger.info("vllm serve stopped")
    except (FileNotFoundError, ValueError, ProcessLookupError):
        logger.info("No running vllm backend found")


def start_backend(model_path: str, model_id: str, max_context_length: int | None = None) -> subprocess.Popen:
    """Start vllm serve as a background subprocess."""
    gpu_util = os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.75")
    max_model_len = str(max_context_length) if max_context_length else os.environ.get("VLLM_MAX_MODEL_LEN")

    cmd = [
        "vllm", "serve", model_path,
        "--host", "127.0.0.1",
        "--port", "8355",
        "--dtype", "auto",
        "--served-model-name", model_id,
        "--gpu-memory-utilization", gpu_util,
    ]
    if max_model_len:
        cmd.extend(["--max-model-len", max_model_len])
    if model_reasoning_format:
        cmd.extend(["--reasoning-parser", model_reasoning_format])
    # Tensor-parallel across a discrete node's GPUs (set by the gateway when a
    # model doesn't fit one GPU's VRAM). Omitted/1 on UMA and single-GPU loads.
    tensor_parallel = os.environ.get("TENSOR_PARALLEL_SIZE")
    if tensor_parallel and tensor_parallel != "1":
        cmd.extend(["--tensor-parallel-size", tensor_parallel])

    # Speculative decoding (e.g. MTP) — a JSON blob passed straight to vLLM's
    # --speculative-config. The gateway sets this for models that ship a draft /
    # MTP head (e.g. {"method": "mtp", "num_speculative_tokens": 1}); omitted
    # otherwise so non-speculative models are unaffected.
    speculative_config = os.environ.get("SPECULATIVE_CONFIG")
    if speculative_config:
        cmd.extend(["--speculative-config", speculative_config])
        # DFlash reserves num_speculative_tokens extra draft slots per sequence,
        # which underflows vLLM's default token budget (it refuses to start with
        # "max_num_scheduled_tokens is set to <negative>"). Raise the batched-token
        # budget so the draft slots fit — matches the z-lab DFlash recipe
        # (--max-num-batched-tokens 32768). Env-overridable; only for DFlash.
        try:
            import json as _json
            if _json.loads(speculative_config).get("method") == "dflash":
                cmd.extend([
                    "--max-num-batched-tokens",
                    os.environ.get("VLLM_MAX_NUM_BATCHED_TOKENS", "32768"),
                ])
        except (ValueError, TypeError):
            pass

    # Eager mode: skip torch.compile + CUDA-graph capture. On bandwidth-bound
    # large models the decode cost is small (decode waits on memory, not kernel
    # launch), but it removes the compile/graph-capture memory peak and startup
    # time, keeping the init footprint ≈ weights so sizing stays small/predictable.
    if os.environ.get("ENFORCE_EAGER", "").lower() == "true":
        cmd.append("--enforce-eager")

    logger.info(f"Starting vllm serve: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd)
    with open(VLLM_PID_FILE, 'w') as f:
        f.write(str(proc.pid))
    return proc


def wait_for_backend(timeout: int = 1800) -> bool:
    """Wait for vllm serve to become healthy."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = req_lib.get(f"{VLLM_BACKEND_URL}/health", timeout=5)
            if r.status_code == 200:
                return True
        except req_lib.ConnectionError:
            pass
        time.sleep(5)
    return False


def initialize():
    """Initialize HTTP clients."""
    global http_client, sync_http_client
    logger.info(f"Backend URL: {VLLM_BACKEND_URL}")
    logger.info("Starting in idle mode - no model loaded")

    http_client = httpx.AsyncClient(
        base_url=VLLM_BACKEND_URL,
        timeout=httpx.Timeout(300.0, connect=10.0),
    )
    sync_http_client = httpx.Client(
        base_url=VLLM_BACKEND_URL,
        timeout=httpx.Timeout(300.0, connect=10.0),
    )


# ============================================================================
# Auto-load from MODEL_ID env var
# ============================================================================

def _do_auto_load(model_id: str):
    global MODEL_ID, MODEL_PATH, backend_start_time, is_switching, tokenizer
    global model_stop_tokens, model_reasoning_format, model_tool_use

    logger.info(f"Auto-loading model from MODEL_ID env var: {model_id}")
    is_switching = True

    try:
        model_path = query_mlflow(model_id)
        logger.info(f"Resolved model path: {model_path}")

        stop_tokens_str = os.environ.get("STOP_TOKENS")
        if stop_tokens_str:
            model_stop_tokens = json.loads(stop_tokens_str)
        if os.environ.get("REASONING_FORMAT"):
            model_reasoning_format = os.environ["REASONING_FORMAT"]
        model_tool_use = os.environ.get("TOOL_USE", "").lower() == "true"

        start_backend(model_path, model_id,
                      max_context_length=int(os.environ["MAX_CONTEXT_LENGTH"])
                      if os.environ.get("MAX_CONTEXT_LENGTH") else None)

        # Generous health-wait: a DFlash/compiled load (weights + torch.compile +
        # flashinfer autotune + dual graph capture) can take ~700s+ on a cold
        # cache. 600s killed legitimate loads mid-graph-capture → crash loop.
        # Env-overridable. The gateway's LOAD_TIMEOUT_SECONDS must stay above this.
        load_timeout = int(os.environ.get("VLLM_LOAD_TIMEOUT_SECONDS", "1800"))
        if not wait_for_backend(timeout=load_timeout):
            logger.error(
                f"Auto-load failed: {model_id} did not become healthy within "
                f"{load_timeout}s — exiting so the pod goes NotReady (the gateway "
                f"surfaces the error)"
            )
            os._exit(1)

        MODEL_ID = model_id
        MODEL_PATH = model_path
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        backend_start_time = time.time()
        is_switching = False
        logger.info(f"Auto-load complete: {model_id}")
    except Exception as e:
        logger.error(
            f"Auto-load failed: {e} — exiting so the pod goes NotReady",
            exc_info=True,
        )
        os._exit(1)


@app.on_event("startup")
async def startup_auto_load():
    env_model_id = os.environ.get("MODEL_ID")
    if env_model_id:
        asyncio.create_task(asyncio.to_thread(_do_auto_load, env_model_id))


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check():
    if MODEL_ID is None and not is_switching:
        return {"status": "idle", "model": None, "engine": "vllm"}
    if is_switching:
        return {"status": "switching", "model": MODEL_ID, "engine": "vllm"}
    try:
        response = await http_client.get("/health")
        backend_healthy = response.status_code == 200
        if backend_healthy:
            return {"status": "healthy", "model": MODEL_ID, "model_path": MODEL_PATH, "engine": "vllm"}
        return JSONResponse(status_code=503, content={"status": "unhealthy", "model": MODEL_ID, "engine": "vllm"})
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e), "model": MODEL_ID, "engine": "vllm"}
        )


# ============================================================================
# Admin Management Endpoints
# ============================================================================

@app.get("/admin/current-model")
async def admin_current_model():
    if MODEL_ID is None:
        return {"model_id": None, "model_path": None, "status": "idle", "engine": "vllm", "uptime_seconds": 0}
    uptime = time.time() - backend_start_time if backend_start_time else 0
    status = "switching" if is_switching else "serving"
    if not is_switching:
        try:
            r = await http_client.get("/health")
            if r.status_code != 200:
                status = "unhealthy"
        except Exception:
            status = "unhealthy"
    return {
        "model_id": MODEL_ID,
        "model_path": MODEL_PATH,
        "status": status,
        "engine": "vllm",
        "uptime_seconds": round(uptime, 1),
        "stop_tokens": model_stop_tokens,
        "reasoning_format": model_reasoning_format,
        "tool_use": model_tool_use,
    }


@app.post("/admin/switch-model")
async def admin_switch_model(request: Request):
    global MODEL_ID, MODEL_PATH, backend_start_time, is_switching, tokenizer
    global model_stop_tokens, model_reasoning_format, model_tool_use

    if is_switching:
        return JSONResponse(
            status_code=409,
            content={"error": "A model switch is already in progress"}
        )

    body = await request.json()
    new_model_id = body.get("model_id")
    if not new_model_id:
        return JSONResponse(status_code=400, content={"error": "model_id is required"})

    metadata = {}
    if "stop_tokens" in body:
        metadata["stop_tokens"] = body["stop_tokens"]
    if "reasoning_format" in body:
        metadata["reasoning_format"] = body["reasoning_format"]
    if "tool_use" in body:
        metadata["tool_use"] = body["tool_use"]

    max_context_length = body.get("max_context_length")

    previous_model = MODEL_ID
    previous_path = MODEL_PATH
    switch_start = time.time()

    try:
        is_switching = True

        logger.info(f"Switching model: {previous_model} -> {new_model_id}")
        new_model_path = query_mlflow(new_model_id)
        logger.info(f"Found model at: {new_model_path}")

        stop_backend()

        MODEL_ID = new_model_id
        MODEL_PATH = new_model_path
        os.environ["MODEL_ID"] = new_model_id
        os.environ["MODEL_PATH"] = new_model_path

        if metadata.get("stop_tokens") is not None:
            model_stop_tokens = metadata["stop_tokens"]
        if metadata.get("reasoning_format") is not None:
            model_reasoning_format = metadata["reasoning_format"]
        if metadata.get("tool_use") is not None:
            model_tool_use = metadata["tool_use"]

        start_backend(new_model_path, new_model_id, max_context_length=max_context_length)

        if not wait_for_backend(timeout=600):
            logger.error(f"Failed to start vllm serve for {new_model_id}, rolling back...")
            stop_backend()
            MODEL_ID = previous_model
            MODEL_PATH = previous_path
            if previous_model and previous_path:
                try:
                    os.environ["MODEL_ID"] = previous_model
                    os.environ["MODEL_PATH"] = previous_path
                    start_backend(previous_path, previous_model)
                    wait_for_backend(timeout=600)
                    tokenizer = AutoTokenizer.from_pretrained(previous_path)
                    logger.info("Rollback succeeded")
                except Exception as rollback_err:
                    logger.error(f"Rollback also failed: {rollback_err}")

            is_switching = False
            backend_start_time = time.time() if MODEL_ID else None
            return JSONResponse(status_code=500, content={
                "previous_model": previous_model,
                "current_model": MODEL_ID,
                "status": "serving",
                "error": f"Failed to load {new_model_id}: backend did not become healthy within timeout"
            })

        logger.info("Reloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(new_model_path)

        backend_start_time = time.time()
        is_switching = False
        switch_time = time.time() - switch_start

        logger.info(f"Model switch complete: {previous_model} -> {MODEL_ID} in {switch_time:.1f}s")

        return {
            "previous_model": previous_model,
            "current_model": MODEL_ID,
            "status": "serving",
            "switch_time_seconds": round(switch_time, 1),
            "stop_tokens": model_stop_tokens,
            "reasoning_format": model_reasoning_format,
            "tool_use": model_tool_use,
        }

    except Exception as e:
        logger.error(f"Error during model switch: {e}", exc_info=True)
        is_switching = False
        return JSONResponse(status_code=500, content={
            "previous_model": previous_model,
            "current_model": MODEL_ID,
            "status": "error",
            "error": str(e)
        })


@app.get("/admin/status")
async def admin_status():
    if is_switching:
        return {"status": "switching", "ready": False}
    if MODEL_ID is None:
        return {"status": "idle", "ready": True}
    try:
        r = await http_client.get("/health")
        if r.status_code == 200:
            return {"status": "ready", "ready": True}
        return {"status": "unhealthy", "ready": False}
    except Exception:
        return {"status": "unreachable", "ready": False}


# ============================================================================
# OpenAI-Compatible API Endpoints
# ============================================================================

@app.post("/v1/chat/completions")
async def openai_chat_completions(request: Request):
    """Proxy to vllm serve backend with streaming support."""
    if MODEL_ID is None:
        return JSONResponse(status_code=503, content={"error": {"message": "No model loaded", "type": "service_unavailable"}})
    try:
        body = await request.json()
        stream = body.get('stream', False)

        if stream:
            async def stream_proxy():
                async with http_client.stream("POST", "/v1/chat/completions", json=body) as response:
                    async for chunk in response.aiter_bytes():
                        yield chunk
            return StreamingResponse(
                stream_proxy(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )

        response = await http_client.post("/v1/chat/completions", json=body)
        return JSONResponse(content=response.json(), status_code=response.status_code)

    except Exception as e:
        logger.error(f"Error in chat completions: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "internal_error"}}
        )


@app.get("/v1/models")
async def openai_models():
    """OpenAI-compatible models endpoint."""
    if MODEL_ID is None:
        return JSONResponse({"object": "list", "data": []})
    return JSONResponse({
        "object": "list",
        "data": [{
            "id": MODEL_ID,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "thinkube",
            "permission": [],
            "root": MODEL_ID,
            "parent": None
        }]
    })


# ============================================================================
# Gradio Chat UI
# ============================================================================

def generate_response(message: str, history: list, temperature: float = 0.7, max_tokens: int = 512):
    """Generate response via vllm serve backend."""
    messages = []
    if history:
        for item in history:
            if isinstance(item, dict):
                role = item.get("role", "user")
                content = item.get("content", "")
                if isinstance(content, list):
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif isinstance(block, str):
                            text_parts.append(block)
                    content = "".join(text_parts)
                if content:
                    messages.append({"role": role, "content": content})
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                user_msg, assistant_msg = item
                if user_msg:
                    messages.append({"role": "user", "content": str(user_msg)})
                if assistant_msg:
                    messages.append({"role": "assistant", "content": str(assistant_msg)})

    messages.append({"role": "user", "content": str(message)})

    response = sync_http_client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL_ID,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.95,
        }
    )
    response.raise_for_status()
    result = response.json()

    content = result["choices"][0]["message"].get("content", "")
    yield content


thinkube_theme = create_thinkube_theme()

demo = gr.ChatInterface(
    generate_response,
    title="vLLM Chat",
    description="Powered by vLLM",
    examples=[
        ["Hello! How are you?", 0.7, 512],
        ["Can you explain quantum computing in simple terms?", 0.7, 512],
        ["Write a Python function to calculate fibonacci numbers", 0.7, 512],
    ],
    analytics_enabled=False,
    additional_inputs=[
        gr.Slider(0.1, 2.0, value=0.7, label="Temperature"),
        gr.Slider(64, 2048, value=512, label="Max Tokens"),
    ],
)

app = gr.mount_gradio_app(
    app,
    demo,
    path="/",
    theme=thinkube_theme,
    css=THINKUBE_CSS,
    favicon_path="/app/icons/tk_ai.svg"
)

if __name__ == "__main__":
    initialize()
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")
