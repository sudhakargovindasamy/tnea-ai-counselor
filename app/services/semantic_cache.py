import logging
from typing import Optional
from sentence_transformers import SentenceTransformer
from app.services.database import supabase

logger = logging.getLogger(__name__)
embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
CACHE_THRESHOLD = 0.92

def check_cache(question: str) -> Optional[str]:
    try:
        emb = embedding_model.encode(question, normalize_embeddings=True).tolist()
        resp = supabase.rpc("match_cache", {
            "query_embedding": emb,
            "match_threshold": CACHE_THRESHOLD,
            "match_count": 1
        }).execute()
        if resp.data and len(resp.data) > 0:
            logger.info(f"✅ Cache HIT! Similarity: {resp.data[0]['similarity']:.4f}")
            return resp.data[0]["answer"]
        logger.info("❌ Cache MISS")
        return None
    except Exception as e:
        logger.warning(f"Cache check failed: {e}")
        return None

def save_to_cache(question: str, answer: str, sources: list):
    try:
        emb = embedding_model.encode(question, normalize_embeddings=True).tolist()
        supabase.table("query_cache").insert({
            "question": question,
            "embedding": emb,
            "answer": answer,
            "sources": sources
        }).execute()
        logger.info("💾 Saved to cache")
    except Exception as e:
        logger.warning(f"Cache save failed: {e}")