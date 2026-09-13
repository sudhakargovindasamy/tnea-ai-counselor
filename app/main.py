import logging
import os
import math
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.models import QueryRequest, QueryResponse, Source
from app.services.retrieval import retrieve
from app.services.llm import generate_answer
from app.services.query_understanding import understand_query, rewrite_query
from app.services.memory import memory
from app.services.semantic_cache import check_cache, save_to_cache
from app.services.database import supabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="TN-Engineering Q/A System API",
    version="1.0-final",
    description="AI Counselor for Tamil Nadu Engineering Colleges. Built for Frontend Integration."
)

# 🚀 1. CORS MIDDLEWARE
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🚀 2. GLOBAL EXCEPTION HANDLER
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

@app.get("/")
def root():
    return {"message": "TNEA Counselor AI API is running", "docs": "/docs", "health": "/health"}

@app.get("/health")
def health():
    return {"status": "healthy", "version": "1.0-final"}

@app.post("/clear_chat/{session_id}")
def clear_chat(session_id: str):
    memory.clear_history(session_id)
    return {"status": "cleared", "session_id": session_id}

# 🚀 HELPER: Smart Source Extraction (UPDATED FOR BRANCH ENRICHMENT)
def extract_source_info(doc: dict) -> Source:
    meta = doc.get("metadata", {})
    doc_type = meta.get("doc_type", "general_info")
    
    if doc_type == "admission_info":
        college_name = f"TNEA Rules: {meta.get('section', 'General Information')}"
        tnea_code = "N/A"
        district = "Tamil Nadu"
        
    elif doc_type == "branch_info":
        # 🚀 CRITICAL FIX: Uses the enriched college_name from the Relational SQL JOIN!
        # If enrichment worked, it shows the College Name. If not, it falls back to the Branch Code.
        college_name = meta.get("college_name", f"Branch: {meta.get('branch_code', 'Unknown')} (Code: {meta.get('tnea_code', 'N/A')})")
        tnea_code = str(meta.get("tnea_code", "N/A"))
        district = meta.get("district", "N/A")
        
    else:  
        college_name = meta.get("college_name", "Unknown College")
        tnea_code = str(meta.get("tnea_code", "Unknown"))
        district = meta.get("district", "Unknown")
    
    raw_score = doc.get("similarity", doc.get("rerank_score", 0.0))
    
    # Normalize reranker logits to 0-1 range
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

# 🚀 HELPER: Format Cache Response
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

# 🚀 3. ADMIN PURGE CACHE ENDPOINT
@app.post("/admin/purge_cache")
def purge_cache(admin_secret: str):
    expected_secret = os.getenv("ADMIN_SECRET_KEY", "TNEA_SUPER_SECRET_ADMIN_KEY_2026")
    if admin_secret != expected_secret:
        raise HTTPException(status_code=403, detail="Unauthorized")
        
    try:
        supabase.table("query_cache").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
        logger.info("🚨 ADMIN ACTION: Semantic Cache completely purged due to data update.")
        return {"status": "success", "message": "Cache purged. AI will now fetch fresh data."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 🚀 4. USER FEEDBACK ENDPOINT (The "Thumbs Down" Button)
@app.post("/feedback/downvote")
def report_bad_answer(question: str):
    try:
        supabase.table("query_cache").delete().eq("question", question).execute()
        logger.info(f"👎 USER FEEDBACK: Deleted poisoned cache for question: {question[:50]}...")
        return {"status": "success", "message": "Feedback recorded. Cache cleared for this question."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 🚀 5. MAIN QUERY ENDPOINT
@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
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

        # 3) HISTORY-AWARE QUERY TRANSLATION
        search_query = rewrite_query(req.question, history) if history else req.question

        # 4) Secondary Cache Check
        if search_query != req.question:
            cached_response = check_cache(search_query)
            if cached_response:
                logger.info("⚡ Cache HIT on Rewritten Query!")
                cached_answer = cached_response.get("answer", "") if isinstance(cached_response, dict) else str(cached_response)
                memory.add_message(req.session_id, "user", req.question)
                memory.add_message(req.session_id, "model", cached_answer)
                return format_cache_response(cached_response)

        # 5) Agentic routing
        understood = understand_query(search_query)
        intent = understood.pop("intent", "search")
        compare_colleges = understood.pop("compare_colleges", None)
        
        nested_filters = understood.pop("filters", {})
        if isinstance(nested_filters, dict):
            understood.update(nested_filters)
        
        active_filters = req.filters if req.filters else {}
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

        # 7) Generate
        answer = generate_answer(req.question, context_data, req.session_id, reference_answer=None, intent=intent)

        # 8) Smart Source Extraction
        sources = [extract_source_info(d) for d in docs]

        # 9) Save to cache
        save_to_cache(req.question, answer, [s.dict() for s in sources])
        if search_query != req.question:
            save_to_cache(search_query, answer, [s.dict() for s in sources])

        # 10) Save to memory
        memory.add_message(req.session_id, "user", req.question)
        memory.add_message(req.session_id, "model", answer)

        return QueryResponse(answer=answer, sources=sources)

    except Exception as e:
        logger.error(f"Error in /query endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))