FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y git curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir transformers litellm fastapi uvicorn httpx accelerate
COPY ./OpenFugu /app
COPY ./openfugu-patch/serve.py /app/openfugu/serve.py
EXPOSE 8088
CMD ["python", "openfugu/serve.py", "--host", "0.0.0.0", "--port", "8088"]
