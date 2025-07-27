# AI Model Inference Server Template for Thinkube

A GPU-accelerated inference server template for the [Thinkube](https://github.com/thinkube/thinkube) platform, providing a production-ready Gradio UI for text generation and text-to-image models from Hugging Face.

## 🚀 Quick Start

[![Deploy to Thinkube](https://img.shields.io/badge/Deploy%20to-Thinkube-blue?style=for-the-badge&logo=kubernetes)](https://thinkube.github.io/tkt-vllm-gradio/deploy.html)

Click the button above to deploy this template to your Thinkube instance!

## ✨ Features

- 🤖 **Multiple Model Types**: Support for chat/text generation and text-to-image models
- 🎨 **Gradio UI**: Interactive web interface customized for each model type
- 🚀 **GPU Optimized**: Leverages CUDA for fast inference
- 📦 **Pre-built Base Image**: Fast deployments with pre-installed ML dependencies
- 🔧 **Simple Configuration**: Just specify model ID and type

## 🖥️ Hardware Requirements

- **Minimum**: NVIDIA GTX 1080 (8GB VRAM)
- **Recommended**: NVIDIA RTX 3090 (24GB VRAM) or better
- **CUDA**: 12.4 compatible GPU

## 📋 Supported Models

### For GTX 1080 (8GB):
- **Text**: `microsoft/phi-2`, `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Image**: `runwayml/stable-diffusion-v1-5` (512x512)

### For RTX 3090 (24GB):
- **Text**: `mistralai/Mistral-7B-Instruct-v0.2`, `meta-llama/Llama-2-7b-chat-hf`
- **Image**: `stabilityai/stable-diffusion-xl-base-1.0`

## 🔧 Configuration Options

When deploying through Thinkube Control:

| Parameter | Description | Options |
|-----------|-------------|---------|
| `model_id` | Hugging Face model ID | e.g., `microsoft/phi-2` |
| `model_type` | Type of model | `chat` or `text-to-image` |

## 📁 Template Structure

```
tkt-vllm-gradio/
├── manifest.yaml          # Template configuration
├── thinkube.yaml          # Deployment specification
├── Dockerfile.jinja       # Container definition
├── server.py.jinja        # Main application
├── requirements.txt       # Additional dependencies
└── docs/
    └── deploy.html        # Deploy button page
```

## 🛠️ Technical Details

- **Base Image**: Custom AI inference image with CUDA 12.4 on Ubuntu 24.04
- **Pre-installed**: PyTorch, Transformers, Diffusers, Gradio, FastAPI
- **Runtime**: Python 3.12
- **Port**: 7860 (Gradio default)

## 🚀 After Deployment

Your AI inference server will be available at:
- **Application**: `https://your-app.yourdomain.com`
- **Health Check**: `https://your-app.yourdomain.com/health`

## 📝 Development

To customize this template:

1. Fork this repository
2. Modify `server.py.jinja` for custom inference logic
3. Update `requirements.txt` for additional dependencies
4. Test locally with your GPU
5. Submit a pull request

## ⚠️ Important Notes

- Models are downloaded on first run (can take time for large models)
- GPU memory usage depends on model size and batch size
- Some models require Hugging Face authentication token
- vLLM is only supported on newer GPUs (Ampere architecture and above)

## 🤝 Contributing

Contributions are welcome! Please test your changes on appropriate hardware before submitting.

## 📄 License

This template is part of the Thinkube project and follows the same license terms.

---

**Note**: This template requires GPU resources. Ensure your Thinkube cluster has GPU nodes available before deployment.