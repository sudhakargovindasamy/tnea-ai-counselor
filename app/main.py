import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.models import QueryRequest, QueryResponse, Source
from app.services.retrieval import retrieve
from app.services.llm import generate_answer
from app.services.query_understanding import understand_query, rewrite_query
from app.services.memory import memory
from app.services.semantic_cache import check_cache, save_to_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="TN-Engineering Q/A System API",
    version="1.0-final",
    description="AI Counselor for Tamil Nadu Engineering Colleges. Built for Frontend Integration."
)

# 🚀 1. CORS MIDDLEWARE (Crucial for Frontend)
# This prevents the browser from blocking the frontend's requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, change "*" to your actual frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allows POST, GET, OPTIONS, etc.
    allow_headers=["*"],  # Allows Content-Type, Authorization, etc.
)

# 🚀 2. GLOBAL EXCEPTION HANDLER (Frontend devs love this)
# Catches any unhandled crash and returns clean JSON instead of an ugly HTML 500 error.
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

@app.get("/health")
def health():
    return {"status": "healthy", "version": "1.0-final"}

@app.post("/clear_chat/{session_id}")
def clear_chat(session_id: str):
    memory.clear_history(session_id)
    return {"status": "cleared", "session_id": session_id}

# 🚀 Helper to format cache responses cleanly (DRY Principle)
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

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    try:
        logger.info(f"[{req.session_id}] Query: {req.question}")

        # 🚀 1) Fast-path Semantic Cache (Check original question first)
        cached_response = check_cache(req.question)
        if cached_response:
            logger.info("⚡ Cache HIT! Serving from Semantic Cache (0 LLM calls)")
            
            # 🚀 CRITICAL FIX: Save to memory even on Cache HIT!
            cached_answer = ""
            if isinstance(cached_response, dict):
                cached_answer = cached_response.get("answer", "")
            elif isinstance(cached_response, tuple):
                cached_answer = cached_response[0]
            else:
                cached_answer = str(cached_response)
                
            memory.add_message(req.session_id, "user", req.question)
            memory.add_message(req.session_id, "model", cached_answer)
            
            return format_cache_response(cached_response)

        # 2) Get Chat History
        history = memory.get_history(req.session_id)

        # 🚀 3) HISTORY-AWARE QUERY TRANSLATION (Fixes the "there" / "it" problem)
        if history:
            search_query = rewrite_query(req.question, history)
        else:
            search_query = req.question

        # 🚀 4) Secondary Cache Check (Check the rewritten question if it changed)
        if search_query != req.question:
            cached_response = check_cache(search_query)
            if cached_response:
                logger.info("⚡ Cache HIT on Rewritten Query!")
                
                # Save to memory for rewritten query cache hit too!
                cached_answer = ""
                if isinstance(cached_response, dict):
                    cached_answer = cached_response.get("answer", "")
                elif isinstance(cached_response, tuple):
                    cached_answer = cached_response[0]
                else:
                    cached_answer = str(cached_response)
                    
                memory.add_message(req.session_id, "user", req.question)
                memory.add_message(req.session_id, "model", cached_answer)
                
                return format_cache_response(cached_response)

        # 5) Agentic routing (Use the rewritten search_query!)
        understood = understand_query(search_query)
        intent = understood.pop("intent", "search")
        compare_colleges = understood.pop("compare_colleges", None)
        
        # Handle both flat and nested filter formats
        nested_filters = understood.pop("filters", {})
        if isinstance(nested_filters, dict):
            understood.update(nested_filters)
        
        active_filters = req.filters if req.filters else {}
        active_filters.update(understood)

        # 6) Retrieve (Use the rewritten search_query for Vector DB!)
        docs, context_data = retrieve(
            search_query,  
            top_k=req.top_k,
            filters=active_filters,
            compare_colleges=compare_colleges,
            intent=intent
        )

        if docs is None:
            return QueryResponse(answer=context_data, sources=[])

        # 7) Generate (Pass the ORIGINAL question to the LLM, it has the conversational history)
        answer = generate_answer(
            req.question, context_data, req.session_id, 
            reference_answer=None, intent=intent 
        )

        sources = [
            Source(
                college_name=d.get("metadata", {}).get("college_name", "Unknown"),
                tnea_code=str(d.get("metadata", {}).get("tnea_code", "Unknown")),
                district=d.get("metadata", {}).get("district", "Unknown"),
                score=d.get("rerank_score", d.get("similarity", 0.0))
            ) for d in docs
        ]

        # 8) Save to cache for next time
        save_to_cache(req.question, answer, [s.dict() for s in sources])
        
        # Also save the rewritten query to cache so future users asking the explicit question get a fast hit
        if search_query != req.question:
            save_to_cache(search_query, answer, [s.dict() for s in sources])

        return QueryResponse(answer=answer, sources=sources)

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))