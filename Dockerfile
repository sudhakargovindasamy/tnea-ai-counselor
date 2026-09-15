FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

# Copy requirements and install CPU-only PyTorch
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 🔽 PRE-DOWNLOAD ML MODELS INTO DOCKER IMAGE (Prevents cold-start delays!)
RUN python -c "import os; os.environ['HF_HUB_DISABLE_TELEMETRY']='1'; \
    from sentence_transformers import SentenceTransformer, CrossEncoder; \
    print('Downloading BGE-Large Embedding Model...'); \
    SentenceTransformer('BAAI/bge-large-en-v1.5'); \
    print('Downloading MS-Marco Reranker...'); \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
    print('Models cached successfully!')"

# Copy application code
COPY . .

# Hugging Face Spaces strictly requires port 7860
EXPOSE 7860

# Start FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]