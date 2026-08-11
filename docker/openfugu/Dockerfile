FROM python:3.13-slim
WORKDIR /app
RUN apt-get update && apt-get install -y git curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir 'transformers>=4.52,<5' accelerate safetensors huggingface_hub fastapi httpx jsonschema uvicorn 'redis>=5,<7'
COPY ./OpenFugu /app
COPY ./openfugu-patch/serve.py /app/openfugu/serve.py
COPY ./openfugu-patch/api.py /app/openfugu/api.py
COPY ./openfugu-patch/provider_protocols.py /app/openfugu/provider_protocols.py
COPY ./openfugu-patch/anthropic_protocols.py /app/openfugu/anthropic_protocols.py
# The catalog facade plus its three schema/ABI/runtime modules are copied
# together so the container can re-validate the catalog before serving.
COPY ./scripts/model_catalog*.py /app/openfugu/
# Router head artifact used to build the TRINITY SVF+head vector; the trained
# ABI manifest alongside it anchors the catalog slot contract to this artifact.
COPY ./artifacts /app/artifacts
COPY ./scripts/make_vec.py /app/scripts/make_vec.py
COPY ./scripts/learn_router.py /app/scripts/learn_router.py
# Build the TRINITY base vector from the committed small safetensors head.
RUN python /app/scripts/make_vec.py
# Fail-closed catalog gate: validates the catalog before the server starts.
COPY ./docker/openfugu/entrypoint.sh /app/openfugu/entrypoint.sh
RUN chmod +x /app/openfugu/entrypoint.sh
EXPOSE 8088
ENTRYPOINT ["/app/openfugu/entrypoint.sh"]
CMD ["sh", "-c", "if [ -f /app/learning/promoted/model_iter_60.npy ]; then export MANTIS_VECTOR=/app/learning/promoted/model_iter_60.npy; fi; if [ \"${MANTIS_LEARNING:-0}\" = 1 ]; then python /app/scripts/learn_router.py --watch --promote & fi; exec python -m uvicorn api:app --app-dir /app/openfugu --host 0.0.0.0 --port 8088"]
