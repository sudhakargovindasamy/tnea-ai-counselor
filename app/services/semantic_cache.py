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
# 🧠 L2: SUPABASE CACHE (Fast DB Lookup)
# ═══════════════════════════════════════════════════════════
CACHE_VERSION = "v2.3"
CACHE_VERSION = "v2.4"

def check_cache(question: str) -> dict[str, Any] | None:
    """
    Two-tier caching with Cache Versioning:
    1. L1: In-memory exact normalized LRU cache (< 1ms)
    2. L2: Supabase database cache (~50ms)
    """
    if not question or not str(question).strip():
        return None

    cache_key = f"[{CACHE_VERSION}] {_L1_CACHE._normalize(question)}"

    # 1. Check L1 In-Memory Cache
    l1_hit = _L1_CACHE.get(cache_key)
    if l1_hit:
        logger.info(f"⚡ [L1_CACHE_HIT] Instant answer (<1ms) for: '{question[:50]}'")
        return l1_hit

    # 2. Check L2 Supabase Cache
    try:
        resp = supabase.table("query_cache").select("answer, sources").eq("question", cache_key).limit(1).execute()
        if resp.data and len(resp.data) > 0:
            hit = resp.data[0]
            logger.info(f"✅ [L2_CACHE_HIT] Found in DB for: '{question[:50]}'")
            cached_result = {
                "answer": hit.get("answer", ""),
                "sources": hit.get("sources", [])
            }
            # Populate L1 cache for subsequent instant hits
            _L1_CACHE.set(cache_key, cached_result)
            return cached_result
            
        logger.info(f"❌ Cache MISS for: '{question[:50]}'")
        return None
        
    except Exception as e:
        logger.warning(f"⚠️ Cache check error: {e}")
        return None

def purge_question_cache(question: str):
    """Purge a specific question from both L1 and L2 caches."""
    if not question or not str(question).strip():
        return
    norm_q = _L1_CACHE._normalize(question)
    cache_key = f"[{CACHE_VERSION}] {norm_q}"
    try:
        if cache_key in _L1_CACHE.cache:
            del _L1_CACHE.cache[cache_key]
        if norm_q in _L1_CACHE.cache:
            del _L1_CACHE.cache[norm_q]
    except Exception as e:
        logger.debug(f"L1 question purge error: {e}")

    try:
        supabase.table("query_cache").delete().ilike("question", f"%{norm_q}%").execute()
        logger.info(f"🗑️ Purged cache for question: '{question[:50]}'")
    except Exception as e:
        logger.warning(f"L2 question purge error: {e}")

def save_to_cache(question: str, answer: str, sources: list, overwrite: bool = False):
    """Save valid, grounded responses to both L1 (memory) and L2 (Supabase)."""
    if not sources or len(sources) == 0:
        logger.info("⚠️ Skipped caching: No verified sources found (Prevents poisoning).")
        return
        
    if not answer or len(answer.strip()) < 15:
        return

    lower_ans = answer.lower()
    poison_tokens = (
        "error:", "[error:", "llm generation failed", "resource_exhausted",
        "service unavailable", "rate limit", "503", "429", "timeout",
        "does not contain information", "i don't have"
    )
    if any(tok in lower_ans for tok in poison_tokens):
        logger.info(f"🛡️ Cache Poison Guard: Skipped caching refusal/error response for: '{question[:40]}'")
        return

    # Truncation Guard: Do not cache responses cut off mid-sentence or mid-token
    stripped_ans = answer.strip()
    if stripped_ans.endswith("TNEA Code:") or stripped_ans.endswith("Code:") or stripped_ans.endswith("..."):
        logger.info(f"🛡️ Cache Poison Guard: Skipped caching truncated response for: '{question[:40]}'")
        return

    # Entity Consistency Guard: Prevent caching misrouted responses
    # If a query specifically targets College A, but sources belong to different colleges, skip caching.
    try:
        from app.services.retrieval import resolve_college_entity
        resolved_entity = resolve_college_entity(question)
        if resolved_entity:
            expected_code = str(resolved_entity.get("metadata", {}).get("tnea_code", ""))
            source_codes = [str(s.get("tnea_code", "") if isinstance(s, dict) else getattr(s, "tnea_code", "")) for s in sources]
            if expected_code and expected_code not in source_codes:
                logger.info(f"🛡️ Cache Poison Guard: Entity mismatch (Query targeted {expected_code}, sources had {source_codes[:3]}). Skipped caching.")
                return
    except Exception as e:
        logger.debug(f"Entity consistency guard check: {e}")

    clean_sources = []
    for s in sources:
        if hasattr(s, "dict"): clean_sources.append(s.dict())
        elif hasattr(s, "model_dump"): clean_sources.append(s.model_dump())
        elif isinstance(s, dict): clean_sources.append(s)

    cached_payload = {
        "answer": answer,
        "sources": clean_sources
    }

    cache_key = f"[{CACHE_VERSION}] {_L1_CACHE._normalize(question)}"

    # 1. Save to L1 Cache instantly
    _L1_CACHE.set(cache_key, cached_payload)

    # 2. Persist to L2 Supabase Cache (overwrite if requested)
    try:
        if overwrite:
            try:
                supabase.table("query_cache").delete().eq("question", cache_key).execute()
            except Exception:
                pass

        expires_at = (datetime.utcnow() + timedelta(days=1)).isoformat()
        
        supabase.table("query_cache").insert({
            "question": cache_key,
            "answer": answer,
            "sources": clean_sources,
            "has_sources": True,
            "expires_at": expires_at
        }).execute()
        
        logger.info(f"💾 Saved to L1 & L2 cache: '{cache_key[:50]}'")
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
        # id is a bigint primary key
        supabase.table("query_cache").delete().gt("id", 0).execute()
        logger.info("🗑️ Supabase query_cache successfully cleared.")
    except Exception as e:
        logger.warning(f"Error clearing L2 cache: {e}")