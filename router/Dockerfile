FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y git curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir \
    fastapi uvicorn httpx routellm transformers torch --extra-index-url https://download.pytorch.org/whl/cpu
COPY ./llm-router /app
EXPOSE 5500
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "5500"]
