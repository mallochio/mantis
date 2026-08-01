FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y git curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir transformers litellm fastapi uvicorn httpx accelerate safetensors huggingface_hub
COPY ./OpenFugu /app
COPY ./openfugu-patch/serve.py /app/openfugu/serve.py
# Router head artifact used to build the TRINITY SVF+head vector.
COPY ./artifacts /app/artifacts
EXPOSE 8088
CMD ["python", "openfugu/serve.py", "--host", "0.0.0.0", "--port", "8088"]
