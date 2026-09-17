import asyncio
import json
import logging
import math
import os
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

# Rate Limiter
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

load_dotenv()

from app.models import QueryRequest, QueryResponse, Source
from app.services.database import supabase
from app.services.llm import generate_answer, generate_answer_stream
from app.services.memory import memory
from app.services.observability import RAGTracer
from app.services.query_understanding import rewrite_query, understand_query
from app.services.retrieval import retrieve
from app.services.semantic_cache import check_cache, purge_all_cache, save_to_cache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
# 🛡️ RATE LIMITER (10 requests per minute per IP)
# ═══════════════════════════════════════════════════════════
limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])

# 🛠️ Universal retry wrapper for Gemini calls
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
                wait = 2 * (attempt + 1)
                logger.warning(f"⏳ Rate-limited in {getattr(fn, '__name__', 'llm')}. Backoff {wait}s (attempt {attempt+1}/3)")
                time.sleep(wait)
            else:
                raise
    raise last

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting TNEA Counselor AI API...")
    logger.info("ℹ️ Models will be loaded lazily on first request (memory optimization).")
    
    try:
        supabase.table("documents").select("id").limit(1).execute()
        logger.info("✅ Supabase database connection verified.")
    except Exception as e:
        logger.error(f"❌ Supabase connection failed: {e}")

    yield 
    
    logger.info("👋 Shutting down TNEA Counselor AI API...")

app = FastAPI(
    title="TN-Engineering Q/A System API",
    version="2.0-production",
    description="AI Counselor for Tamil Nadu Engineering Colleges. Built for Frontend Integration with SSE Streaming & Grounded Citations.",
    lifespan=lifespan
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    try:
        # 🚨 CRITICAL: Increased to 180s for Render free-tier cold starts
        # Loading PyTorch models from disk into RAM takes 40-60s on 0.5 vCPU
        response = await asyncio.wait_for(call_next(request), timeout=180.0)
        return response
    except asyncio.TimeoutError:
        logger.error("⏱️ Request timed out after 180 seconds")
        return JSONResponse(
            status_code=504,
            content={
                "error": "timeout",
                "message": "The AI is waking up and loading models. Please try again in a moment."
            }
        )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled Exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": f"Server encountered an error: {type(exc).__name__}: {exc!s}"
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
            .container { text-align: center; background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); max-width: 520px; }
            h1 { color: #2563eb; }
            .badge { display: inline-block; background: #10b981; color: white; padding: 5px 12px; border-radius: 20px; font-size: 14px; margin-bottom: 20px; }
            a { color: #2563eb; text-decoration: none; font-weight: bold; }
            a:hover { text-decoration: underline; }
            .endpoints { text-align: left; background: #f8fafc; padding: 15px; border-radius: 8px; margin-top: 20px; font-family: monospace; font-size: 14px; }
        </style>
    </head>
    <body>
        <div class="container">
            <span class="badge">🟢 API ONLINE (v2.0 Production)</span>
            <h1>🎓 TNEA Counselor AI</h1>
            <p>Production RAG API with Hybrid Search, L1/L2 Caching, and Token-by-Token SSE Streaming.</p>
            <div class="endpoints">
                <strong>Available Routes:</strong><br>
                🌊 <a href="/docs#/default/chat_chat_post">POST /chat</a> (SSE Token Streaming)<br>
                💬 <a href="/docs#/default/query_query_post">POST /query</a> (Standard JSON)<br>
                📚 <a href="/docs">/docs</a> (Swagger UI)<br>
                🩺 <a href="/health">/health</a> (Status Check)<br>
                🔥 <a href="/warmup">/warmup</a> (Pre-load Models)
            </div>
        </div>
    </body>
    </html>
    """

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "version": "2.0-production",
        "features": ["l1_cache", "l2_semantic_cache", "sse_streaming", "slowapi_rate_limit", "observability_tracing"],
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/warmup")
def warmup():
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
        return {"status": "error", "message": str(e)}

@app.post("/clear_chat/{session_id}")
def clear_chat(session_id: str):
    memory.clear_history(session_id)
    return {"status": "cleared", "session_id": session_id}

@app.get("/search_colleges")
def search_colleges(
    district: str | None = None,
    branch_code: str | None = None,
    has_hostel: bool = False,
    autonomous: bool | None = None,
    limit: int = 10
):
    """Direct college catalog search by district, branch, and facilities."""
    from app.services.retrieval import get_colleges_by_filters
    docs = get_colleges_by_filters(district=district, branch_code=branch_code, has_hostel=has_hostel, limit=limit)
    if autonomous is not None:
        docs = [d for d in docs if d.get("metadata", {}).get("autonomous") == autonomous]
    return {
        "count": len(docs),
        "results": [d.get("metadata") for d in docs]
    }

def extract_source_info(doc: dict) -> Source:
    meta = doc.get("metadata", {})
    doc_type = meta.get("doc_type", "college_info")
    
    if doc_type == "admission_info":
        college_name = f"TNEA Rules: {meta.get('section', 'General Information')}"
        tnea_code = "N/A"
        district = "Tamil Nadu"
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
        purge_all_cache()
        logger.info("🚨 ADMIN ACTION: Semantic & In-memory caches purged.")
        return {"status": "success", "message": "All caches purged successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback/downvote")
def report_bad_answer(question: str):
    try:
        supabase.table("query_cache").delete().eq("question", question).execute()
        logger.info(f"👎 USER FEEDBACK: Cleared cache for question: {question[:50]}...")
        return {"status": "success", "message": "Feedback recorded. Cache cleared for this question."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════
# 🌊 SERVER-SENT EVENTS (SSE) STREAMING /chat ENDPOINT
# ═══════════════════════════════════════════════════════════
@app.post("/chat")
@limiter.limit("10/minute")
async def chat(request: Request, req: QueryRequest):
    """
    Production Chat Endpoint with Server-Sent Events (SSE) Streaming.
    Yields tokens incrementally: `data: {"token": "..."}\n\n`
    Concludes with citation cards: `data: {"sources": [...]}\n\n` followed by `data: [DONE]\n\n`
    """
    tracer = RAGTracer(session_id=req.session_id, query=req.question)

    async def sse_stream_generator() -> AsyncGenerator[str, None]:
        nonlocal tracer
        try:
            logger.info(f"🌊 [{req.session_id}] SSE Chat Query: '{req.question}'")

            # 1. Check L1/L2 Caching (Latency < 1ms on repeated questions)
            cached_response = check_cache(req.question)
            if cached_response:
                cached_answer = cached_response.get("answer", "")
                cached_sources = cached_response.get("sources", [])
                tracer.log_cache_hit(cached_answer)
                tracer.finish()
                
                # Save to memory
                memory.add_message(req.session_id, "user", req.question)
                memory.add_message(req.session_id, "assistant", cached_answer)

                # Stream cached answer in chunks for natural UI rendering
                words = cached_answer.split(" ")
                for i in range(0, len(words), 4):
                    chunk = " ".join(words[i:i+4]) + " "
                    yield f"data: {json.dumps({'token': chunk})}\n\n"
                    await asyncio.sleep(0.01)

                yield f"data: {json.dumps({'sources': cached_sources})}\n\n"
                yield "data: [DONE]\n\n"
                return

            # 2. Query History & Translation
            history = memory.get_history(req.session_id)
            try:
                search_query = rewrite_query(req.question, history) if history else req.question
            except Exception as e:
                logger.warning(f"Query rewriting warning: {e}")
                search_query = req.question

            # 3. Intent & Filter Understanding
            try:
                understood = understand_query(search_query)
            except Exception:
                understood = {"intent": "search"}

            intent = understood.pop("intent", "search")
            if intent in ["list", "filter", "cutoff"] and intent != "cutoff":
                intent = "search"

            compare_colleges = understood.pop("compare_colleges", None)
            active_filters = req.filters.copy() if req.filters else {}
            active_filters.update(understood)

            # 4. Retrieval & Pre-filtering
            docs, context_data = retrieve(
                search_query, top_k=req.top_k, filters=active_filters,
                compare_colleges=compare_colleges, intent=intent, chat_history=history
            )
            tracer.log_retrieval(docs, filters=active_filters)

            # Guardrail: No documents retrieved
            if not docs:
                safe_msg = context_data if (isinstance(context_data, str) and context_data.strip()) else "The provided TNEA database does not contain information to answer this."
                tracer.log_llm_call(prompt="None", answer=safe_msg)
                tracer.finish()
                
                # Save to memory
                memory.add_message(req.session_id, "user", req.question)
                memory.add_message(req.session_id, "assistant", safe_msg)
                
                yield f"data: {json.dumps({'token': safe_msg})}\n\n"
                yield f"data: {json.dumps({'sources': []})}\n\n"
                yield "data: [DONE]\n\n"
                return

            # 5. Extract Sources
            sources = [extract_source_info(d) for d in docs]
            sources = deduplicate_sources(sources)
            sources_dict = [s.model_dump() if hasattr(s, 'model_dump') else s.dict() for s in sources]

            # 6. Stream Generation Token-by-Token
            accumulated_tokens = []
            prompt_preview = f"Context Length: {len(context_data)} chars | Question: {req.question}"

            for token in generate_answer_stream(req.question, context_data, req.session_id, docs=docs):
                accumulated_tokens.append(token)
                yield f"data: {json.dumps({'token': token})}\n\n"
                await asyncio.sleep(0.005) # Cooperative yield

            full_answer = "".join(accumulated_tokens).strip()
            tracer.log_llm_call(prompt=prompt_preview, answer=full_answer)
            tracer.finish()

            # 🚨 Prevent caching error messages (Poisoned Cache Fix)
            is_error = full_answer.startswith("Error:") or full_answer.startswith("[Error:") or "LLM Generation failed" in full_answer

            # 7. Cache Response & Save Memory
            if not is_error:
                save_to_cache(req.question, full_answer, sources_dict)
            
            # 💾 Save to conversational memory
            memory.add_message(req.session_id, "user", req.question)
            memory.add_message(req.session_id, "assistant", full_answer)

            # 8. Send Sources Metadata & Done Signal
            yield f"data: {json.dumps({'sources': sources_dict})}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Error in SSE stream: {e}", exc_info=True)
            tracer.log_error(str(e))
            tracer.finish()
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        sse_stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# ═══════════════════════════════════════════════════════════
# 💬 STANDARD JSON /query ENDPOINT
# ═══════════════════════════════════════════════════════════
@app.post("/query", response_model=QueryResponse)
@limiter.limit("10/minute")
def query(request: Request, req: QueryRequest):
    tracer = RAGTracer(session_id=req.session_id, query=req.question)
    
    try:
        logger.info(f"[{req.session_id}] Standard Query: {req.question}")

        # 1) Check L1 / L2 Cache
        cached_response = check_cache(req.question)
        if cached_response:
            tracer.log_cache_hit(cached_response.get("answer", ""))
            tracer.finish()
            memory.add_message(req.session_id, "user", req.question)
            memory.add_message(req.session_id, "assistant", cached_response.get("answer", ""))
            return format_cache_response(cached_response)

        # 2) Chat History & Query Translation
        history = memory.get_history(req.session_id)
        try:
            search_query = _llm_retry(rewrite_query, req.question, history) if history else req.question
        except Exception as e:
            logger.warning(f"Query rewrite warning: {e}")
            search_query = req.question

        # 3) Intent & Filters
        try:
            understood = _llm_retry(understand_query, search_query)
        except Exception:
            understood = {"intent": "search"}

        intent = understood.pop("intent", "search")
        if intent in ["list", "filter", "cutoff"] and intent != "cutoff":
            intent = "search"

        compare_colleges = understood.pop("compare_colleges", None)
        active_filters = req.filters.copy() if req.filters else {}
        active_filters.update(understood)

        # 4) Retrieval
        docs, context_data = retrieve(
            search_query, top_k=req.top_k, filters=active_filters,
            compare_colleges=compare_colleges, intent=intent, chat_history=history
        )
        tracer.log_retrieval(docs, filters=active_filters)

        # Guardrail: No data
        if not docs:
            safe_answer = context_data if isinstance(context_data, str) else "The provided TNEA database does not contain information to answer this."
            tracer.log_llm_call(prompt="None", answer=safe_answer)
            tracer.finish()
            memory.add_message(req.session_id, "user", req.question)
            memory.add_message(req.session_id, "assistant", safe_answer)
            return QueryResponse(answer=safe_answer, sources=[])

        # 5) Generate Answer with Grounded Citations
        prompt_preview = f"Context Length: {len(context_data)} chars | Question: {req.question}"
        is_error = False
        try:
            answer = _llm_retry(generate_answer, req.question, context_data, req.session_id, 
                               reference_answer=None, intent=intent, docs=docs)
            if answer.startswith("Error:"):
                is_error = True
        except Exception as e:
            logger.error(f"🚨 LLM Generation failed: {e}", exc_info=True)
            answer = f"Error: LLM Generation failed with error: {type(e).__name__}: {e!s}"
            is_error = True

        tracer.log_llm_call(prompt=prompt_preview, answer=answer)
        tracer.finish()

        # 6) Deduplicate Sources
        sources = [extract_source_info(d) for d in docs]
        sources = deduplicate_sources(sources)

        # 7) Save to Cache & Memory (🚨 ONLY IF NOT AN ERROR)
        sources_dict = [s.model_dump() if hasattr(s, 'model_dump') else s.dict() for s in sources]
        
        if not is_error:
            save_to_cache(req.question, answer, sources_dict)

        memory.add_message(req.session_id, "user", req.question)
        memory.add_message(req.session_id, "assistant", answer)

        return QueryResponse(answer=answer, sources=sources)

    except Exception as e:
        tracer.log_error(str(e))
        tracer.finish()
        logger.error(f"Error in /query endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))