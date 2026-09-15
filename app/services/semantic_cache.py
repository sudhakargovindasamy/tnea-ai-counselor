import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from app.services.database import supabase

logger = logging.getLogger(__name__)

# 🧠 LAZY LOADING: Model loads on first cache call, not at import time
# Shares the SAME model instance as retrieval.py to avoid double memory usage
embedding_model = None
CACHE_THRESHOLD = 0.92

def _get_embedding_model():
    """Lazily load the embedding model, reusing retrieval.py's instance to save RAM."""
    global embedding_model
    if embedding_model is None:
        # 🚀 Reuse the model already loaded by retrieval.py (zero extra memory!)
        from app.services.retrieval import embedding_model as retrieval_model, _load_models
        _load_models()  # Ensure retrieval's model is loaded first
        embedding_model = retrieval_model
        logger.info("✅ Cache embedding model ready (shared with retrieval).")
    return embedding_model

def check_cache(question: str) -> Optional[Dict[str, Any]]:
    try:
        model = _get_embedding_model()
        emb = model.encode(question, normalize_embeddings=True).tolist()
        resp = supabase.rpc("match_cache", {
            "query_embedding": emb,
            "match_threshold": CACHE_THRESHOLD,
            "match_count": 1
        }).execute()
        
        if resp.data and len(resp.data) > 0:
            hit = resp.data[0]
            logger.info(f"✅ Cache HIT! Similarity: {hit.get('similarity', 0):.4f}")
            return {
                "answer": hit.get("answer", ""),
                "sources": hit.get("sources", [])
            }
            
        logger.info(f"❌ Cache MISS for: {question[:60]}...")
        return None
        
    except Exception as e:
        logger.warning(f"⚠️ Cache check failed: {e}")
        return None

def save_to_cache(question: str, answer: str, sources: list):
    # 🚨 PRODUCTION GUARDRAIL: NEVER cache hallucinations or empty answers
    if not sources or len(sources) == 0:
        logger.info("⚠️ Skipped caching: No verified sources found (Prevents poisoning).")
        return
        
    if not answer or "I don't have enough specific information" in answer:
        logger.info("⚠️ Skipped caching: Fallback/Error message detected.")
        return

    try:
        model = _get_embedding_model()
        emb = model.encode(question, normalize_embeddings=True).tolist()
        
        clean_sources = []
        for s in sources:
            if hasattr(s, "dict"): clean_sources.append(s.dict())
            elif hasattr(s, "model_dump"): clean_sources.append(s.model_dump())
            elif isinstance(s, dict): clean_sources.append(s)
                
        # 🚀 Calculate Expiration (7 Days from now)
        expires_at = (datetime.utcnow() + timedelta(days=7)).isoformat()
        
        supabase.table("query_cache").insert({
            "question": question,
            "embedding": emb,
            "answer": answer,
            "sources": clean_sources,
            "has_sources": True,
            "expires_at": expires_at
        }).execute()
        
        logger.info(f"💾 Saved to cache (Expires in 7 days): {question[:60]}...")
        
    except Exception as e:
        err_str = str(e).lower()
        if "duplicate key" in err_str or "unique constraint" in err_str:
            pass  # Ignore duplicates silently
        else:
            logger.warning(f"⚠️ Cache save failed: {e}")