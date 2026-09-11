---
title: TNEA Counselor AI API
emoji: 🎓
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

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

This project is developed as part of our internship by a collaborative team, with each member contributing to specific areas of the system.

| Member | Role | Responsibilities |
|--------|------|------------------|
| **Sudhakar** | Backend Developer & AI Engineer | Designed and developed the complete RAG pipeline, FastAPI backend, Supabase/pgvector integration, Agentic Query Routing, Semantic Caching, Conversational Memory, and Gemini LLM integration with fallback chains. |
| **Kavivarshini** | AI/ML Engineer | Co-developed the RAG model alongside the backend team, including retrieval optimization, embedding strategies, query understanding, and anti-hallucination guardrails. |
| **Poojitha** | Data Engineer | Responsible for extracting raw data from source documents, designing the document structure, data cleaning, and converting unstructured data into clean, structured CSV formats for the knowledge base. |
| **Lekhana** | Frontend Developer | Designed and developed the user-facing chat interface, integrating with the FastAPI backend to deliver a seamless conversational experience. |

---

### 👨‍💻 Individual Contributions

#### Sudhakar — Backend Developer & AI Engineer
- Built the **FastAPI backend** with production-grade error handling
- Implemented **Agentic RAG pipeline** (Vector Search, SQL Filtering, Numerical Queries)
- Developed **Conversational Memory** with History-Aware Query Translation
- Integrated **Semantic Caching** for zero-cost repeated queries
- Set up **Supabase + pgvector** vector database
- Configured **Gemini LLM** with multi-model fallback chains
- Implemented **Anti-Hallucination Guardrails** and confidence scoring

#### Kavivarshini — AI/ML Engineer
- Co-developed the **RAG retrieval pipeline**
- Optimized **embedding strategies** using Sentence Transformers
- Fine-tuned **query understanding** and intent classification
- Implemented **Cross-Encoder reranking** for improved retrieval quality
- Collaborated on **Fuzzy Entity Matching** and alias resolution

#### Poojitha — Data Engineer
- **Extracted** raw data from official TNEA source documents
- **Designed** structured document schemas for colleges, branches, and performance data
- **Cleaned and normalized** 418 colleges, 3518 branches, and performance records
- **Converted** unstructured data into structured CSV formats (`colleges_db_df.csv`, `branches_db_df.csv`, `performance_db_df.csv`)
- Ensured data quality and consistency for accurate retrieval

#### Lekhana — Frontend Developer
- Built the **Chat UI** for the AI Counselor
- Integrated with **FastAPI backend** (`/query`, `/clear_chat`, `/health` endpoints)
- Implemented **Markdown rendering** for AI responses
- Designed **source citation cards** to display retrieved college information
- Handled **session management** and loading states

## 📄 License

This project is created for educational purposes as part of an internship.

---

*Built with ❤️ for Tamil Nadu Engineering Students*