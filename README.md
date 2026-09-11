# 🎓 TNEA Counselor AI

> An **AI-Powered Engineering College Guidance System** for Tamil Nadu students, built using **RAG (Retrieval-Augmented Generation)** technology.

## 🚀 Features

- 🧠 **Agentic RAG Pipeline** - Smart routing between Vector Search, SQL Filtering, and Numerical Queries
- 🔄 **Conversational Memory** - Multi-turn chat with history-aware query rewriting
- 🎯 **Fuzzy Entity Matching** - Handles typos, abbreviations (SSN, PSG, CIT), and missing spaces
- ⚡ **Semantic Caching** - Instant responses for repeated questions (0 LLM calls)
- 🛡️ **Anti-Hallucination Guardrails** - Confidence scoring and strict context-only answers
- 🤖 **Multi-Model Fallback** - Auto-switches between Gemini models on rate limits

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Backend Framework** | FastAPI |
| **Vector Database** | Supabase + pgvector |
| **LLM** | Google Gemini (Multi-model fallback) |
| **Embeddings** | BAAI/bge-large-en-v1.5 |
| **Reranker** | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| **Package Manager** | uv |

## 📁 Project Structure

```
RAG-final/
├── app/
│   ├── main.py                    # FastAPI application & routes
│   ├── models.py                  # Pydantic request/response models
│   ├── config.py                  # Configuration & env variables
│   └── services/
│       ├── retrieval.py           # Agentic RAG retrieval engine
│       ├── llm.py                 # LLM generation with fallback
│       ├── query_understanding.py # Intent extraction & query rewriting
│       ├── memory.py              # Conversational memory
│       ├── semantic_cache.py      # Semantic caching layer
│       └── database.py            # Supabase client
├── data/
│   └── raw/                       # Source CSV files
├── scripts/
│   ├── 01_setup_supabase.sql      # Database setup script
│   ├── 02_ingest_colleges.py      # College data ingestion
│   ├── 03_ingest_branches.py      # Branch data ingestion
│   └── 04_ingest_performance.py   # Performance data ingestion
├── .env.example                   # Environment variables template
├── .gitignore                     # Git ignore rules
├── README.md                      # This file
└── pyproject.toml                 # Project dependencies
```

## 🏃 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/tnea-counselor-ai.git
cd tnea-counselor-ai
```

### 2. Install Dependencies (using uv)
```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv sync
```

### 3. Set Up Environment Variables
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your actual keys
nano .env
```

### 4. Run the Application
```bash
uv run uvicorn app.main:app --reload --port 8000
```

### 5. Access the API
- **Interactive Docs (Swagger UI):** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Check if the API is running |
| `POST` | `/query` | Ask a question to the AI Counselor |
| `POST` | `/clear_chat/{session_id}` | Clear chat history for a session |

## 🤝 Team

- **Sudhakar** - Backend Developer & AI Engineer

## 📄 License

This project is created for educational purposes as part of an internship.

---

*Built with ❤️ for Tamil Nadu Engineering Students*