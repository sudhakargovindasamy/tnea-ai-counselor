import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from app.services.database import supabase

logger = logging.getLogger(__name__)
embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
CACHE_THRESHOLD = 0.92

def check_cache(question: str) -> Optional[Dict[str, Any]]:
    try:
        emb = embedding_model.encode(question, normalize_embeddings=True).tolist()
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
        emb = embedding_model.encode(question, normalize_embeddings=True).tolist()
        
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
            pass # Ignore duplicates silently
        else:
            logger.warning(f"⚠️ Cache save failed: {e}")