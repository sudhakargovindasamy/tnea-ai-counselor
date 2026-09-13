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

> A **production-grade AI Engineering College Counselor** for Tamil Nadu students, powered by advanced **RAG (Retrieval-Augmented Generation)** with Hybrid Search, Conversational Memory, and Enterprise Guardrails.

Built to handle **418 Colleges**, **3,500+ Branches**, and **Official TNEA Admission Rules** with zero hallucinations.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green?logo=fastapi)
![Supabase](https://img.shields.io/badge/Supabase-pgvector-3ecf8e?logo=supabase)
![Gemini](https://img.shields.io/badge/LLM-Gemini_Flash-4285F4?logo=google)

---

## 🚀 Key Features

- 🎯 **Smart Intent Routing** — Automatically routes queries to College, Branch, or Admission data silos
- 🔗 **Hybrid Search with SQL Joins** — Combines Vector Search + Relational SQL for structured data queries
- 🔄 **Conversational Memory** — Multi-turn chat with history-aware query rewriting (resolves "there", "it", etc.)
- 🔍 **Fuzzy Entity Matching** — Handles typos, abbreviations (SSN, PSG, CIT, CEG), and missing spaces
- ⚡ **Semantic Caching with TTL** — Instant responses for repeated questions (0 LLM calls), auto-expires in 7 days
- 🛡️ **Anti-Hallucination Guardrails** — Confidence gates + cache poisoning prevention
- 🤖 **Multi-Model Fallback** — Auto-switches between Gemini models on rate limits (429 errors)
- 👍 **User Feedback Loop** — "Thumbs Down" endpoint to self-heal poisoned cache entries
- 🔧 **Admin Cache Management** — Secure purge endpoint for yearly data updates

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Backend Framework** | FastAPI | Production REST API |
| **Vector Database** | Supabase + pgvector | Storage + similarity search |
| **LLM** | Google Gemini Flash | Generation + Query Rewriting |
| **Embeddings** | BAAI/bge-large-en-v1.5 | 1024-dim semantic vectors |
| **Reranker** | cross-encoder/ms-marco-MiniLM-L-6-v2 | Context relevance scoring |
| **Package Manager** | uv | Fast, reproducible installs |
| **Deployment** | Render / Hugging Face Spaces | Cloud hosting |

---

## 🏗️ System Architecture

```mermaid
graph TD
    User[User Query] --> API[FastAPI /query Endpoint]
    API --> Cache{Semantic Cache<br/>pgvector + TTL}
    Cache -- HIT --> Response[Return Cached Answer]
    Cache -- MISS --> Rewriter[Query Rewriter<br/>Gemini Flash]
    Rewriter --> Router[Smart Router<br/>Intent Classification]
    
    Router -- College Intent --> SQL1[Metadata Filter<br/>college_info]
    Router -- Branch Intent --> SQL2[Direct SQL Join<br/>branch + college]
    Router -- Rules Intent --> SQL3[Admission Docs<br/>Search]
    
    SQL1 --> Vector[Vector Search<br/>BGE-Large]
    SQL2 --> Vector
    SQL3 --> Vector
    
    Vector --> Reranker[Cross-Encoder<br/>MS-Marco Reranking]
    Reranker --> Guardrail{Confidence<br/>Gate}
    
    Guardrail -- Pass --> LLM[LLM Generation<br/>Gemini Flash]
    Guardrail -- Fail --> Fallback[Safe Fallback<br/>No Hallucination]
    
    LLM --> SaveCache[Save to Cache<br/>7-day TTL]
    SaveCache --> Response

---

## 📁 Project Structure

```text
RAG-final/
├── app/
│   ├── main.py                    # FastAPI app, CORS, Admin & Feedback endpoints
│   ├── models.py                  # Pydantic request/response schemas
│   ├── config.py                  # Environment configuration
│   └── services/
│       ├── retrieval.py           # Hybrid Search, Fuzzy Matching, SQL Joins
│       ├── llm.py                 # Gemini generation with fallback chains
│       ├── query_understanding.py # Intent extraction + Query Rewriting
│       ├── memory.py              # Conversational history management
│       ├── semantic_cache.py      # pgvector caching with TTL + guardrails
│       └── database.py            # Supabase client initialization
├── data/raw/                      # Source CSV and JSON datasets
├── scripts/
│   ├── 01_setup_supabase.sql      # Database schema + pgvector functions
│   ├── 08_master_ingest.py        # Production ingestion pipeline
│   └── 06_verify_supabase.py      # Data health & integrity checker
├── .env.example                   # Required environment variables
├── requirements.txt               # CPU-optimized dependencies
├── Dockerfile                     # Container configuration
└── README.md                      # This file
```

---

## 🏃 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/tnea-counselor-ai.git
cd tnea-counselor-ai
```

### 2. Install Dependencies
```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

### 3. Set Up Environment Variables
```bash
cp .env.example .env
nano .env
```

Required variables:
```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-key
GEMINI_API_KEY=your-gemini-api-key
ADMIN_SECRET_KEY=your-custom-secret
```

### 4. Ingest Data to Supabase
```bash
# Upload 418 colleges, 3500+ branches, and TNEA admission rules
python scripts/08_master_ingest.py

# Verify data integrity
python scripts/06_verify_supabase.py
```

### 5. Run the API
```bash
uvicorn app.main:app --reload --port 8000
```

### 6. Access the API
- **Interactive Docs:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | System health check |
| `POST` | `/query` | Main RAG endpoint (accepts `session_id`, `question`, `top_k`) |
| `POST` | `/clear_chat/{session_id}` | Clear conversation history |
| `POST` | `/admin/purge_cache` | Wipe semantic cache (requires `admin_secret`) |
| `POST` | `/feedback/downvote` | Delete poisoned cache entries |

### Example Query
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "student_001",
    "question": "Which colleges in Coimbatore offer Computer Science?",
    "top_k": 5
  }'
```

---

## 🧪 Test Cases

### Test 1: Hybrid Search (Branch + College Join)
```json
{
  "session_id": "test_01",
  "question": "Which colleges in Coimbatore offer CS and what is their intake?",
  "top_k": 5
}
```
✅ Returns enriched branch data with actual college names via SQL JOIN.

### Test 2: Conversational Memory
**Turn 1:**
```json
{
  "session_id": "test_02",
  "question": "Tell me about Thiagarajar College of Engineering"
}
```
**Turn 2:**
```json
{
  "session_id": "test_02",
  "question": "What is the hostel fee there?"
}
```
✅ Query Rewriter resolves "there" → "Thiagarajar College of Engineering".

### Test 3: Hallucination Guardrail
```json
{
  "session_id": "test_03",
  "question": "What is the WiFi password at PSG College?"
}
```
✅ Returns safe fallback (no hallucination).

---

## ☁️ Deployment

### Render (Recommended)
| Setting | Value |
|---------|-------|
| **Runtime** | Python 3 |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |
| **Instance Type** | Free (512MB RAM) |

### Hugging Face Spaces
The included `Dockerfile` and frontmatter enable one-click deployment to HF Spaces.

---

## 🤝 Team

This project was developed collaboratively during our internship, with each member owning specific components of the system.

| Member | Role | Key Contributions |
|--------|------|-------------------|
| **Sudhakar** | Backend & AI Engineer | Complete RAG pipeline, FastAPI backend, Supabase integration, Smart Routing, Hybrid SQL Joins, Semantic Caching with TTL, Conversational Memory, Hallucination Guardrails, Multi-model Fallback |
| **Kavivarshini** | AI/ML Engineer | RAG retrieval optimization, embedding strategy design, query understanding & intent classification, Cross-Encoder reranking implementation, confidence scoring |
| **Poojitha** | Data Engineer | Data extraction from TNEA sources, document schema design, data cleaning & normalization (418 colleges, 3518 branches, 10 admission rule sets), CSV formatting |
| **Lekhana** | Frontend Developer | Chat UI implementation, FastAPI integration, markdown rendering, source citation cards, session management, Google Auth integration |

---

### 👨‍💻 Detailed Individual Contributions

#### Sudhakar — Backend & AI Engineer
- Architected the **complete FastAPI backend** with production-grade error handling, CORS, and global exception handlers
- Designed and implemented **Hybrid Search** combining Vector Search + Direct SQL Relational Joins
- Built **Smart Intent Routing** that classifies queries and directs them to correct data silos (college/branch/admission)
- Implemented **Conversational Memory** with Gemini-powered Query Rewriting for pronoun resolution
- Developed **Semantic Caching** with pgvector similarity matching and 7-day TTL auto-expiration
- Added **Admin & Feedback endpoints** for cache management and self-healing
- Configured **Multi-model Fallback** chains for Gemini rate limit resilience
- Implemented **Confidence Gates** to block hallucinations before generation

#### Kavivarshini — AI/ML Engineer
- Co-designed the **RAG retrieval strategy** with hybrid search approach
- Optimized **embedding pipeline** using BAAI/bge-large-en-v1.5 (1024 dimensions)
- Developed **query understanding** module with Gemini for intent extraction
- Implemented **Cross-Encoder reranking** (MS-Marco) for improved retrieval precision
- Collaborated on **Fuzzy Entity Matching** with alias dictionaries for college abbreviations
- Tuned **confidence thresholds** for the hallucination guardrail system

#### Poojitha — Data Engineer
- **Extracted** raw data from official TNEA brochures, PDFs, and source documents
- **Designed** document schemas for three data types: college profiles, branch details, admission rules
- **Cleaned and normalized** 418 colleges, 3,518 branches, and 10 admission rule sets
- **Converted** unstructured data into structured CSV and JSON formats
- Implemented **data quality checks** and validation scripts
- Ensured consistent metadata tagging (`doc_type`, `tnea_code`, `district`) across all records

#### Lekhana — Frontend Developer
- Built the **modern Chat UI** with real-time streaming support
- Integrated with **FastAPI backend** (`/query`, `/clear_chat`, `/health` endpoints)
- Implemented **Markdown rendering** for rich AI responses with proper formatting
- Designed **source citation cards** displaying college names, TNEA codes, and districts
- Added **session management** with conversation history persistence
- Integrated **Google OAuth** authentication via Supabase Auth
- Implemented **Thumbs Down feedback** button connecting to `/feedback/downvote` endpoint

---

## 📊 Dataset Statistics

| Dataset | Records | Source |
|---------|---------|--------|
| College Profiles | 418 | TNEA Official List |
| Branch Details | 3,518 | AICTE + TNEA |
| Admission Rules | 10 | TNEA Information Brochure 2026 |
| **Total Documents** | **3,946** | Embedded in pgvector |

---

## 📄 License

This project is created for educational purposes as part of our internship program.

---

*Built with ❤️ for Tamil Nadu Engineering Students*
```
