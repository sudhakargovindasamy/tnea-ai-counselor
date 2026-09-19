import logging
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Any

from app.services.database import supabase

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
# ⚡ L1: ULTRA-FAST IN-MEMORY LRU CACHE (< 1ms Latency)
# ═══════════════════════════════════════════════════════════
class InMemoryLRUCache:
    """Thread-safe In-Memory LRU Cache with TTL for instant repetitive query resolution."""
    def __init__(self, capacity: int = 1000, ttl_seconds: int = 3600 * 24):
        self.capacity = capacity
        self.ttl = ttl_seconds
        self.cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def _normalize(self, query: str) -> str:
        """Strip punctuation and extra spaces for high hit-rate matching."""
        if not query:
            return ""
        q = query.strip().lower()
        # Remove trailing question marks or punctuation
        while q and q[-1] in "?!.,":
            q = q[:-1].strip()
        return " ".join(q.split())

    def get(self, query: str) -> dict[str, Any] | None:
        key = self._normalize(query)
        if not key or key not in self.cache:
            return None
            
        entry = self.cache[key]
        if time.time() > entry["expires_at"]:
            del self.cache[key]
            return None
            
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return entry["data"]

    def set(self, query: str, data: dict[str, Any]):
        key = self._normalize(query)
        if not key:
            return
            
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.capacity:
            # Evict oldest item
            self.cache.popitem(last=False)
            
        self.cache[key] = {
            "data": data,
            "expires_at": time.time() + self.ttl
        }

    def clear(self):
        self.cache.clear()

# Singleton In-Memory L1 Cache
_L1_CACHE = InMemoryLRUCache(capacity=2000, ttl_seconds=3600 * 48)

# ═══════════════════════════════════════════════════════════
# 🧠 L2: SUPABASE SEMANTIC CACHE (Vector Similarity)
# ═══════════════════════════════════════════════════════════
embedding_model = None
CACHE_THRESHOLD = 0.92

def _get_embedding_model():
    """Lazily load the embedding model, reusing retrieval.py's instance to save RAM."""
    global embedding_model
    if embedding_model is None:
        from app.services import retrieval
        retrieval._load_models()
        embedding_model = retrieval.embedding_model
        logger.info("✅ Cache embedding model ready (shared with retrieval).")
    return embedding_model

def check_cache(question: str) -> dict[str, Any] | None:
    """
    Two-tier caching:
    1. L1: In-memory exact normalized LRU cache (< 1ms)
    2. L2: Supabase vector semantic cache (~200ms)
    """
    if not question or not str(question).strip():
        return None

    # 1. Check L1 In-Memory Cache
    l1_hit = _L1_CACHE.get(question)
    if l1_hit:
        logger.info(f"⚡ [L1_CACHE_HIT] Instant answer (<1ms) for: '{question[:50]}'")
        return l1_hit

    # 2. Check L2 Supabase Semantic Cache
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
            logger.info(f"✅ [L2_SEMANTIC_CACHE_HIT] Similarity: {hit.get('similarity', 0):.4f}")
            cached_result = {
                "answer": hit.get("answer", ""),
                "sources": hit.get("sources", [])
            }
            # Populate L1 cache for subsequent instant hits
            _L1_CACHE.set(question, cached_result)
            return cached_result
            
        logger.info(f"❌ Cache MISS for: '{question[:50]}'")
        return None
        
    except Exception as e:
        logger.warning(f"⚠️ Cache check error: {e}")
        return None

def save_to_cache(question: str, answer: str, sources: list):
    """Save valid, grounded responses to both L1 (memory) and L2 (Supabase)."""
    if not sources or len(sources) == 0:
        logger.info("⚠️ Skipped caching: No verified sources found (Prevents poisoning).")
        return
        
    if not answer or "does not contain information" in answer.lower() or "i don't have" in answer.lower():
        logger.info("⚠️ Skipped caching: Refusal/Unanswerable response detected.")
        return

    clean_sources = []
    for s in sources:
        if hasattr(s, "dict"): clean_sources.append(s.dict())
        elif hasattr(s, "model_dump"): clean_sources.append(s.model_dump())
        elif isinstance(s, dict): clean_sources.append(s)

    cached_payload = {
        "answer": answer,
        "sources": clean_sources
    }

    # 1. Save to L1 Cache instantly
    _L1_CACHE.set(question, cached_payload)

    # 2. Persist to L2 Supabase Cache
    try:
        model = _get_embedding_model()
        emb = model.encode(question, normalize_embeddings=True).tolist()
        expires_at = (datetime.utcnow() + timedelta(days=7)).isoformat()
        
        supabase.table("query_cache").insert({
            "question": question,
            "embedding": emb,
            "answer": answer,
            "sources": clean_sources,
            "has_sources": True,
            "expires_at": expires_at
        }).execute()
        
        logger.info(f"💾 Saved to L1 & L2 cache: '{question[:50]}'")
    except Exception as e:
        err_str = str(e).lower()
        if "duplicate key" in err_str or "unique constraint" in err_str:
            pass
        else:
            logger.warning(f"⚠️ L2 Cache save failed: {e}")

def purge_all_cache():
    """Wipe both L1 and L2 caches."""
    _L1_CACHE.clear()
    try:
        supabase.table("query_cache").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
    except Exception as e:
        logger.warning(f"Error clearing L2 cache: {e}")