ARG CONTAINER_REGISTRY
FROM ${CONTAINER_REGISTRY}/library/vllm-base:0.19-cuda13.0-py3.12

# Copy application code
COPY server.py .
COPY thinkube_theme.py .
COPY entrypoint.sh .

# Copy Thinkube icons
COPY tk_ai.svg /app/icons/

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# Copy and install any template-specific requirements
COPY requirements.txt .
RUN if [ -s requirements.txt ]; then pip install --no-cache-dir --break-system-packages -r requirements.txt; fi

# The base image already has:
# - Working directory set to /app
# - Cache directories created
# - Environment variables set
# - Python packages pre-installed (vllm, gradio, httpx, transformers, etc.)

# Model ID and HF token passed as environment variables at deployment time
# Not baked into the image to enable image reuse across deployments

# Expose ports: 7860 for Gradio, 8355 for vLLM API
EXPOSE 7860 8355

# Run the entrypoint script
CMD ["./entrypoint.sh"]
