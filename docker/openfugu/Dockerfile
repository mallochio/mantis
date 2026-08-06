FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y git curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir transformers accelerate safetensors huggingface_hub fastapi httpx uvicorn
COPY ./OpenFugu /app
COPY ./openfugu-patch/serve.py /app/openfugu/serve.py
COPY ./openfugu-patch/api.py /app/openfugu/api.py
# Router head artifact used to build the TRINITY SVF+head vector.
COPY ./artifacts /app/artifacts
COPY ./scripts/make_vec.py /app/scripts/make_vec.py
COPY ./scripts/learn_router.py /app/scripts/learn_router.py
# Build the TRINITY base vector from the committed small safetensors head.
RUN python /app/scripts/make_vec.py
EXPOSE 8088
CMD ["sh", "-c", "if [ -f /app/learning/promoted/model_iter_60.npy ]; then export MANTIS_VECTOR=/app/learning/promoted/model_iter_60.npy; fi; if [ \"${MANTIS_LEARNING:-0}\" = 1 ]; then python /app/scripts/learn_router.py --watch --promote & fi; exec python -m uvicorn api:app --app-dir /app/openfugu --host 0.0.0.0 --port 8088"]
