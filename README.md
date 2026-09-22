# tkt-vllm-gradio

The vLLM inference server behind the LLM Gateway. It is the `vllm` optional
component of Thinkube.

## What it does

`server.py` is a FastAPI app on port 7860. It runs `vllm serve` as a
subprocess on `127.0.0.1:8355` and passes requests to it.

- **Model weights come from MLflow.** The server signs in to Keycloak with
  the MLflow credentials, finds the latest version of the model in the
  MLflow Model Registry, and serves it from
  `/mlflow-models/artifacts/<experiment>/<run>/artifacts/model` on shared
  storage. Hugging Face runs offline (`HF_HUB_OFFLINE=1`); nothing is
  downloaded at run time.
- **It starts idle.** When `MODEL_ID` is set, it loads that model at start.
  If the model is not healthy within `VLLM_LOAD_TIMEOUT_SECONDS` (default
  1800), the process exits so the pod goes NotReady.
- **It switches models in place.** `POST /admin/switch-model` stops the
  running `vllm serve`, starts it with the new model, and rolls back to the
  previous model if the new one is not healthy within 600 seconds.
- **It gives an OpenAI-compatible API.** `/v1/chat/completions` (with
  streaming) is passed to vLLM. `/v1/models` lists the loaded model.
- **It has a chat page.** A Gradio chat UI at `/`, with temperature and
  max-token sliders.

### Endpoints

| Method | Path | What it does |
|---|---|---|
| GET | `/health` | `idle`, `switching`, `healthy`, or HTTP 503 `unhealthy` |
| GET | `/admin/current-model` | model id and path, status, uptime, stop tokens, reasoning format, tool use |
| POST | `/admin/switch-model` | body: `model_id` (required), `stop_tokens`, `reasoning_format`, `tool_use`, `max_context_length` |
| GET | `/admin/status` | `idle`, `switching`, `ready`, `unhealthy` or `unreachable`, and a `ready` flag |
| POST | `/v1/chat/completions` | OpenAI chat completions, passed to `vllm serve`; HTTP 503 when no model is loaded |
| GET | `/v1/models` | the loaded model, or an empty list |
| GET | `/` | Gradio chat page |

### Settings

The LLM Gateway sets these on the pod when it loads a model:

| Variable | Effect on `vllm serve` |
|---|---|
| `MODEL_ID` | the model to load at start, looked up in MLflow |
| `STOP_TOKENS` | JSON list, reported by the admin endpoints |
| `REASONING_FORMAT` | `--reasoning-parser` |
| `TOOL_USE` | `true` adds `--enable-auto-tool-choice` and `--tool-call-parser` |
| `TOOL_CALL_PARSER` | the tool-call parser; without it, Qwen formats use `qwen3_coder` and others `hermes` |
| `MAX_CONTEXT_LENGTH` | `--max-model-len` |
| `VLLM_GPU_MEMORY_UTILIZATION` | `--gpu-memory-utilization` (default `0.75`) |
| `TENSOR_PARALLEL_SIZE` | `--tensor-parallel-size`, when above 1 |
| `SPECULATIVE_CONFIG` | `--speculative-config` (JSON); DFlash also sets `--max-num-batched-tokens` and a `bfloat16` KV cache |
| `ENFORCE_EAGER` | `true` adds `--enforce-eager` |

Other variables the server reads: `VLLM_MAX_MODEL_LEN`, `KV_CACHE_DTYPE`,
`VLLM_STRUCTURED_OUTPUTS_CONFIG` (default: xgrammar with no free
whitespace), `VLLM_CUDAGRAPH_CAPTURE_SIZES` (default `1,2,4,8,16,24,32`),
`VLLM_CUDAGRAPH_MODE` (default `FULL_DECODE_ONLY`),
`VLLM_MAX_NUM_BATCHED_TOKENS`, `VLLM_LOAD_TIMEOUT_SECONDS`, and the MLflow
credentials `MLFLOW_KEYCLOAK_TOKEN_URL`, `MLFLOW_KEYCLOAK_CLIENT_ID`,
`MLFLOW_CLIENT_SECRET`, `MLFLOW_AUTH_USERNAME`, `MLFLOW_AUTH_PASSWORD`,
`MLFLOW_TRACKING_URI`. FlashInfer's caches go to `/root/.cache/vllm/`.

## How it reaches a user

It is the `vllm` optional component of
[Thinkube](https://github.com/thinkube/thinkube). It is installed and
removed from the Optional Components page in thinkube-control. The
Templates page refuses it. The install deploys it with no pods
(`replicas: 0`, `gateway_managed: true` in `thinkube.yaml`); the LLM Gateway
creates a pod on a node when a model is loaded there. It is not installed on
its own.

The image builds on the platform's `vllm-base` image. On arm64 that image
installs vLLM from a
[tk-vllm-wheels](https://github.com/thinkube/tk-vllm-wheels) release; on
amd64 it installs the `vllm` package.

## Working on it

| File | What it is |
|---|---|
| `server.py` | the FastAPI app, the `vllm serve` subprocess, the Gradio page |
| `entrypoint.sh` | sets offline Hugging Face mode and starts `server.py` |
| `thinkube_theme.py` | the Gradio theme |
| `Containerfile` | the image, on `vllm-base` |
| `requirements.txt` | extra Python packages; vLLM comes from the base image |
| `thinkube.yaml` | the component deployment: one GPU, port 7860 |
| `manifest.yaml` | the template metadata |

## License

MIT. Code generated from this template is yours: no attribution required, and you may license the app you build however you choose. See [LICENSE](LICENSE).

Copyright Alejandro Martínez Corriá and the Thinkube contributors
