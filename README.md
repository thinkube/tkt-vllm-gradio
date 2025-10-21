# vLLM Inference Server Template for Thinkube

A GPU-accelerated vLLM inference server template for the [Thinkube](https://github.com/thinkube/thinkube) platform, providing high-performance text generation with a Gradio UI.

## 🚀 Quick Start

[![Deploy to Thinkube](https://img.shields.io/badge/Deploy%20to-Thinkube-blue?style=for-the-badge&logo=kubernetes)](https://thinkube.github.io/tkt-vllm-gradio/deploy.html)

Click the button above to deploy this template to your Thinkube instance!

## ✨ Features

- 🚀 **vLLM Engine**: High-performance inference with PagedAttention
- 🎨 **Gradio UI**: Interactive chat interface
- 🖥️ **GPU Required**: Designed for Ampere+ GPUs (RTX 3090, A100, etc.)
- 📦 **Pre-built Base Image**: Fast deployments with pre-installed dependencies

## ⚠️ Hardware Requirements

**This template requires an Ampere or newer GPU architecture:**
- ✅ NVIDIA RTX 3090 (24GB VRAM)
- ✅ NVIDIA RTX 4090 (24GB VRAM)
- ✅ NVIDIA A100, A40, etc.
- ❌ NVIDIA GTX 1080 (Pascal - NOT supported)
- ❌ NVIDIA RTX 2080 (Turing - NOT supported)

## 📋 Supported Models

Models tested with RTX 3090 (24GB):
- `mistralai/Mistral-7B-Instruct-v0.2`
- `meta-llama/Llama-2-7b-chat-hf` (requires HF token)
- `codellama/CodeLlama-7b-Instruct-hf`
- `NousResearch/Hermes-2-Pro-Mistral-7B`

## 🔧 Configuration

When deploying through Thinkube Control:

| Parameter | Description | Example |
|-----------|-------------|---------|
| `model_id` | Hugging Face model ID | `mistralai/Mistral-7B-Instruct-v0.2` |

## 📁 Template Structure

```
tkt-vllm-gradio/
├── manifest.yaml          # Template configuration
├── thinkube.yaml          # Deployment specification
├── Dockerfile.jinja       # Container definition
├── server.py.jinja        # vLLM server with Gradio
├── requirements.txt       # vLLM dependency
└── docs/
    └── deploy.html        # Deploy button page
```

## 🛠️ Technical Details

- **Engine**: vLLM with PagedAttention
- **Base Image**: Custom AI inference image with CUDA 12.4
- **Runtime**: Python 3.12 on Ubuntu 24.04
- **Port**: 7860 (Gradio default)

## 🚀 After Deployment

Your vLLM server will be available at:
- **Application**: `https://your-app.yourdomain.com`
- **Health Check**: `https://your-app.yourdomain.com/health`

## ⚡ Performance

vLLM provides significant performance improvements over standard transformers:
- Up to 24x higher throughput
- Efficient memory usage with PagedAttention
- Support for continuous batching
- Optimized CUDA kernels

## 📝 Development

To customize this template:

1. Fork this repository
2. Modify `server.py.jinja` for custom inference logic
3. Test on a supported GPU (RTX 3090+)
4. Submit a pull request

## ⚠️ Important Notes

- Models are downloaded on first run (can take time for large models)
- Some models require Hugging Face authentication token
- vLLM only supports specific model architectures (see vLLM docs)
- GPU memory usage depends on model size

## 🤝 Contributing

Contributions welcome! Please test on appropriate hardware (RTX 3090+) before submitting.

## 📄 License

This template is part of the Thinkube project and follows the same license terms.

---

**Note**: This template requires Ampere+ GPU architecture. It will NOT work on older GPUs like GTX 1080 or RTX 2080.