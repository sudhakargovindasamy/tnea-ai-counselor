import logging
import os
import math
import time
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from dotenv import load_dotenv

load_dotenv()

from app.models import QueryRequest, QueryResponse, Source
from app.services.retrieval import retrieve
from app.services.llm import generate_answer
from app.services.query_understanding import understand_query, rewrite_query
from app.services.memory import memory
from app.services.semantic_cache import check_cache, save_to_cache
from app.services.database import supabase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 🛠️ FIX 5: Universal retry wrapper for ALL Gemini calls
RETRY_TOKENS = ("429", "exhausted", "quota", "resource has been", "unavailable", "rate", "deadline")

def _llm_retry(fn, *args, **kwargs):
    last = None
    for attempt in range(3):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last = e
            msg = str(e).lower()
            if attempt < 2 and any(t in msg for t in RETRY_TOKENS):
                wait = 2 * (attempt + 1)  # Reduced to prevent Render 504 Gateway Timeouts
                logger.warning(f"⏳ Rate-limited in {getattr(fn, '__name__', 'llm')}. Backoff {wait}s (attempt {attempt+1}/3)")
                time.sleep(wait)
            else:
                raise
    raise last

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting TNEA Counselor AI API...")
    logger.info("ℹ️ Models will be loaded lazily on first request (memory optimization).")
    
    # Quick DB check to ensure credentials are valid
    try:
        supabase.table("documents").select("id").limit(1).execute()
        logger.info("✅ Supabase database connection verified.")
    except Exception as e:
        logger.error(f"❌ Supabase connection failed: {e}")

    yield 
    
    logger.info("👋 Shutting down TNEA Counselor AI API...")

app = FastAPI(
    title="TN-Engineering Q/A System API",
    version="1.0-final",
    description="AI Counselor for Tamil Nadu Engineering Colleges. Built for Frontend Integration.",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🛠️ Request timeout middleware to prevent 502 errors on Render
@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    try:
        # Set a 60-second timeout for all requests
        response = await asyncio.wait_for(call_next(request), timeout=60.0)
        return response
    except asyncio.TimeoutError:
        logger.error("⏱️ Request timed out after 60 seconds")
        return JSONResponse(
            status_code=504,
            content={
                "error": "timeout",
                "message": "Request took too long. Please try a simpler query or try again later."
            }
        )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled Exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "The AI counselor is currently experiencing technical difficulties. Please try again in a few seconds."
        },
    )

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>TNEA Counselor AI API</title>
        <style>
            body { font-family: system-ui, -apple-system, sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; background: #f4f4f9; color: #333; }
            .container { text-align: center; background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); max-width: 500px; }
            h1 { color: #2563eb; }
            .badge { display: inline-block; background: #10b981; color: white; padding: 5px 10px; border-radius: 20px; font-size: 14px; margin-bottom: 20px; }
            a { color: #2563eb; text-decoration: none; font-weight: bold; }
            a:hover { text-decoration: underline; }
            .endpoints { text-align: left; background: #f8fafc; padding: 15px; border-radius: 8px; margin-top: 20px; font-family: monospace; font-size: 14px; }
        </style>
    </head>
    <body>
        <div class="container">
            <span class="badge">🟢 API ONLINE</span>
            <h1>🎓 TNEA Counselor AI</h1>
            <p>The backend API is running successfully.</p>
            <p>This is a headless API. Please use the interactive documentation below to test endpoints or connect your React frontend.</p>
            <div class="endpoints">
                <strong>Available Routes:</strong><br>
                📚 <a href="/docs">/docs</a> (Swagger UI)<br>
                🩺 <a href="/health">/health</a> (Status Check)<br>
                🔥 <a href="/warmup">/warmup</a> (Pre-load Models)<br>
                💬 <a href="/redoc">/redoc</a> (ReDoc)
            </div>
        </div>
    </body>
    </html>
    """

@app.get("/health")
def health():
    """Quick health check - responds immediately without loading models"""
    return {
        "status": "healthy",
        "version": "1.0-final",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/warmup")
def warmup():
    """Pre-load models to avoid cold start on first query. Call this from frontend on page load."""
    try:
        from app.services.retrieval import _load_models
        _load_models()
        return {
            "status": "warmed",
            "message": "Models loaded successfully. Ready for queries.",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Warmup failed: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/clear_chat/{session_id}")
def clear_chat(session_id: str):
    memory.clear_history(session_id)
    return {"status": "cleared", "session_id": session_id}

def extract_source_info(doc: dict) -> Source:
    meta = doc.get("metadata", {})
    doc_type = meta.get("doc_type", "general_info")
    
    if doc_type == "admission_info":
        college_name = f"TNEA Rules: {meta.get('section', 'General Information')}"
        tnea_code = "N/A"
        district = "Tamil Nadu"
        
    elif doc_type == "branch_info":
        college_name = meta.get("college_name", f"Branch: {meta.get('branch_code', 'Unknown')} (Code: {meta.get('tnea_code', 'N/A')})")
        tnea_code = str(meta.get("tnea_code", "N/A"))
        district = meta.get("district", "N/A")
        
    else:  
        college_name = meta.get("college_name", "Unknown College")
        tnea_code = str(meta.get("tnea_code", "Unknown"))
        district = meta.get("district", "Unknown")
    
    raw_score = doc.get("similarity", doc.get("rerank_score", 0.0))
    
    if raw_score < 0 or raw_score > 1:
        normalized_score = 1 / (1 + math.exp(-raw_score / 5))
    else:
        normalized_score = raw_score
    
    return Source(
        college_name=college_name,
        tnea_code=tnea_code,
        district=district,
        score=round(normalized_score, 4)
    )

def deduplicate_sources(sources: list[Source]) -> list[Source]:
    """Remove duplicate sources based on tnea_code"""
    seen = set()
    unique = []
    for source in sources:
        key = source.tnea_code
        if key not in seen:
            seen.add(key)
            unique.append(source)
    return unique

def format_cache_response(cached_response):
    if isinstance(cached_response, dict):
        cached_sources = [
            Source(**s) if isinstance(s, dict) else s 
            for s in cached_response.get("sources", [])
        ]
        return QueryResponse(answer=cached_response.get("answer", ""), sources=cached_sources)
    elif isinstance(cached_response, tuple) and len(cached_response) == 2:
        return QueryResponse(answer=cached_response[0], sources=cached_response[1])
    else:
        return QueryResponse(answer=str(cached_response), sources=[])

@app.post("/admin/purge_cache")
def purge_cache(admin_secret: str):
    expected_secret = os.getenv("ADMIN_SECRET_KEY", "TNEA_SUPER_SECRET_ADMIN_KEY_2026")
    if admin_secret != expected_secret:
        raise HTTPException(status_code=403, detail="Unauthorized")
        
    try:
        supabase.table("query_cache").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
        logger.info("🚨 ADMIN ACTION: Semantic cache completely purged due to data update.")
        return {"status": "success", "message": "Cache purged. AI will now fetch fresh data."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback/downvote")
def report_bad_answer(question: str):
    try:
        supabase.table("query_cache").delete().eq("question", question).execute()
        logger.info(f"👎 USER FEEDBACK: Deleted poisoned cache for question: {question[:50]}...")
        return {"status": "success", "message": "Feedback recorded. Cache cleared for this question."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    start_time = time.time()
    try:
        logger.info(f"[{req.session_id}] Query: {req.question}")

        # 1) Fast-path Semantic Cache
        cached_response = check_cache(req.question)
        if cached_response:
            logger.info("⚡ Cache HIT! Serving from Semantic Cache (0 LLM calls)")
            cached_answer = cached_response.get("answer", "") if isinstance(cached_response, dict) else str(cached_response)
            memory.add_message(req.session_id, "user", req.question)
            memory.add_message(req.session_id, "model", cached_answer)
            return format_cache_response(cached_response)

        # 2) Get Chat History
        history = memory.get_history(req.session_id)

        # 3) HISTORY-AWARE QUERY TRANSLATION (Wrapped in universal retry)
        try:
            search_query = _llm_retry(rewrite_query, req.question, history) if history else req.question
        except Exception as e:
            logger.warning(f"⚠️ Query rewriting completely failed: {e}")
            search_query = req.question

        # 4) Secondary Cache Check
        if search_query != req.question:
            cached_response = check_cache(search_query)
            if cached_response:
                logger.info("⚡ Cache HIT on Rewritten Query!")
                cached_answer = cached_response.get("answer", "") if isinstance(cached_response, dict) else str(cached_response)
                memory.add_message(req.session_id, "user", req.question)
                memory.add_message(req.session_id, "model", cached_answer)
                return format_cache_response(cached_response)

        # 5) Smart routing (Wrapped in universal retry)
        try:
            understood = _llm_retry(understand_query, search_query)
        except Exception as e:
            logger.warning(f"⚠️ Query understanding completely failed: {e}. Defaulting to search intent.")
            understood = {"intent": "search"}

        intent = understood.pop("intent", "search")
        
        # 🛠️ Map new Pydantic intents ("list", "filter") to existing retrieval routes
        if intent in ["list", "filter"]:
            intent = "search"
            
        compare_colleges = understood.pop("compare_colleges", None)
        
        # 🛠️ The updated understand_query returns a flat dict of filters (district, branch_code, etc.)
        # We merge them directly into active_filters, overriding empty frontend filters
        active_filters = req.filters.copy() if req.filters else {}
        active_filters.update(understood)

        # 6) Retrieve
        docs, context_data = retrieve(
            search_query, top_k=req.top_k, filters=active_filters,
            compare_colleges=compare_colleges, intent=intent, chat_history=history
        )

        # 🚨 CRITICAL GUARDRAIL: Block LLM if retrieval fails
        if not docs:
            logger.warning("⚠️ No documents retrieved. Blocking LLM to prevent hallucination.")
            safe_answer = context_data if isinstance(context_data, str) else "I don't have enough specific information in my database to answer this accurately."
            memory.add_message(req.session_id, "user", req.question)
            memory.add_message(req.session_id, "model", safe_answer)
            return QueryResponse(answer=safe_answer, sources=[])

        # 7) Generate Answer (with Graceful Degradation)
        try:
            answer = _llm_retry(generate_answer, req.question, context_data, req.session_id, 
                               reference_answer=None, intent=intent)
        except Exception as e:
            logger.error(f"🚨 LLM Generation completely failed (Quota/Deprecation): {e}")
            answer = "I'm sorry, I'm currently experiencing high demand or technical difficulties. Please try again in a few minutes."

        # 8) Smart Source Extraction + Deduplication
        sources = [extract_source_info(d) for d in docs]
        sources = deduplicate_sources(sources)

        # 9) Save to cache 
        sources_dict = [s.model_dump() if hasattr(s, 'model_dump') else s.dict() for s in sources]
        save_to_cache(req.question, answer, sources_dict)
        if search_query != req.question:
            save_to_cache(search_query, answer, sources_dict)

        # 10) Save to memory
        memory.add_message(req.session_id, "user", req.question)
        memory.add_message(req.session_id, "model", answer)

        duration = time.time() - start_time
        logger.info(f"✅ Query completed in {duration:.2f} seconds")

        return QueryResponse(answer=answer, sources=sources)

    except Exception as e:
        logger.error(f"Error in /query endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))