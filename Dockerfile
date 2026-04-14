ARG CONTAINER_REGISTRY
FROM ${CONTAINER_REGISTRY}/library/vllm-base:0.11-cuda13.0-py3.12

# Copy application code
COPY server.py .

# Copy and install any template-specific requirements
COPY requirements.txt .
RUN if [ -s requirements.txt ]; then pip install --no-cache-dir --break-system-packages -r requirements.txt; fi

# The base image already has:
# - Working directory set to /app
# - Cache directories created
# - Environment variables set
# - Python packages pre-installed

# Run the application
CMD ["python3", "server.py"]
