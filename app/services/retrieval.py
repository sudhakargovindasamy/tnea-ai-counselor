import logging
import difflib
import os
import re
import gc
import torch
from typing import List, Dict, Any, Optional, Tuple
from app.services.database import supabase

logger = logging.getLogger(__name__)

# ═══════════════ 🧠 STRICT LAZY LOADING & MEMORY GUARDS ═══════════════
embedding_model = None
reranker = None
GEMINI_AVAILABLE = False

def _load_models():
    global embedding_model, reranker
    if embedding_model is None:
        logger.info("🧠 Loading embedding model (first request)...")
        from sentence_transformers import SentenceTransformer
        # Using device='cpu' explicitly prevents accidental GPU memory allocation attempts
        embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        logger.info("✅ Embedding model loaded.")
        
    if reranker is None:
        logger.info("🧠 Loading reranker (first request)...")
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
        logger.info("✅ Reranker loaded.")
        
        # 🛠️ CRITICAL: Aggressive memory cleanup after loading ~160MB of models
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("🧹 Garbage collection complete. Memory stabilized.")

QUERY_PREFIX = ""

# ═══════════════ 🧠 ALIAS DICTIONARY (EXACT CSV MATCHES) ═══════════════
# Fixed typos in original aliases (e.g., "Enginering" -> "Engineering")
COLLEGE_ALIASES = {
    "ceg": "University Departments of Anna University , Chennai - CEG Campus",
    "mit": "University Departments of Anna University , Chennai - MIT Campus",
    "act": "University Departments of Anna University , Chennai - ACT Campus",
    "psg": "PSG College of Technology",
    "ssn": "SSN College of Engineering",
    "svce": "Sri Venkateswara College of Engineering",
    "sairam": "Sri Sai Ram Engineering College",  
    "thiagarajar": "Thiagarajar College of Engineering",
    "kct": "Kumaraguru College of Technology",
    "skcet": "Sri Krishna College of Engineering and Technology", 
    "bitsathy": "Bannari Amman Institute of Technology",
    "cit": "Coimbatore Institute of Technology",
    "easwari": "Easwari Engineering College",
    "panimalar": "Panimalar Engineering College",
    "rajalakshmi": "Rajalakshmi Engineering College",
    "velammal": "Velammal Engineering College",
    "sona": "Sona College of Technology",
    "kongu": "Kongu Engineering College",
    "mepco": "Mepco Schlenk Engineering College",
    "saveetha": "Saveetha Engineering College",
    "rmk": "R M K Engineering College",
    "srm": "SRM Institute of Science and Technology"
}

# ═══════════════ 🚀 GLOBAL CACHES (Crucial for 512MB Limit) ═══════════════
# These load ONCE per worker lifecycle, eliminating repeated DB fetches.
_COLLEGE_MAP_CACHE: Dict[str, Dict] = {}       # tnea_code -> metadata
_COLLEGE_DIR_CACHE: Dict[str, str] = {}        # lower_name -> exact_name
_COLLEGE_DIR_CLEAN_CACHE: Dict[str, str] = {}  # lower_name -> cleaned_name

def _load_college_caches():
    global _COLLEGE_MAP_CACHE, _COLLEGE_DIR_CACHE, _COLLEGE_DIR_CLEAN_CACHE
    if _COLLEGE_MAP_CACHE:
        return  # Already loaded
        
    try:
        logger.info("🔄 Warming up college caches...")
        # 🛠️ FIX: Only fetch college_info, not all documents!
        res = supabase.table("documents").select("metadata").eq("metadata->>doc_type", "college_info").execute()
        
        stopwords = {"engineering", "college", "technology", "institute", "of", "and", 
                     "autonomous", "the", "for", "engg", "eng", "tech", "enginering", "technolgoy"}
                     
        for row in res.data:
            meta = row.get("metadata", {})
            tnea_code = str(meta.get("tnea_code", ""))
            name = meta.get("college_name", "")
            
            if tnea_code and name:
                _COLLEGE_MAP_CACHE[tnea_code] = meta
                name_lower = name.lower()
                _COLLEGE_DIR_CACHE[name_lower] = name
                
                # Pre-compute cleaned names for ultra-fast fuzzy matching
                clean_name = name_lower
                for word in stopwords:
                    clean_name = clean_name.replace(word, "")
                _COLLEGE_DIR_CLEAN_CACHE[name_lower] = " ".join(clean_name.split())
                
        logger.info(f"✅ Cached {len(_COLLEGE_MAP_CACHE)} colleges into memory.")
    except Exception as e:
        logger.error(f"Failed to load college caches: {e}")

# ═══════════════ 🧠 IN-MEMORY FUZZY RESOLVER ═══════════════
def fuzzy_resolve_college(user_input: str) -> Optional[str]:
    if not user_input:
        return None
    _load_college_caches()
    if not _COLLEGE_DIR_CACHE:
        return None

    expanded_input = user_input.lower()
    for alias, full_name in COLLEGE_ALIASES.items():
        if f" {alias} " in f" {expanded_input} " or expanded_input == alias:
            expanded_input = expanded_input.replace(alias, full_name)
            logger.info(f"🧠 Alias expanded '{alias}' -> '{full_name}'")
            break

    stopwords = {"engineering", "college", "technology", "institute", "of", "and", 
                 "autonomous", "the", "for", "engg", "eng", "tech", "enginering", "technolgoy"}
    
    clean_input = expanded_input
    for word in stopwords:
        clean_input = clean_input.replace(word, "")
    clean_input = " ".join(clean_input.split())
    input_nospace = clean_input.replace(" ", "")
    
    best_match = None
    highest_score = 0.0
    
    # 🛠️ OPTIMIZATION: Iterate over pre-computed clean cache
    for db_name_lower, clean_db in _COLLEGE_DIR_CLEAN_CACHE.items():
        db_nospace = clean_db.replace(" ", "")
        
        # Fast-path substring check
        if len(input_nospace) >= 4 and (input_nospace in db_nospace or db_nospace in input_nospace):
            return _COLLEGE_DIR_CACHE[db_name_lower]
            
        score1 = difflib.SequenceMatcher(None, clean_input, clean_db).ratio()
        score2 = difflib.SequenceMatcher(None, input_nospace, db_nospace).ratio()
        
        score = max(score1, score2)
        if score > highest_score:
            highest_score = score
            best_match = db_name_lower
            
    if highest_score > 0.75:
        exact_name = _COLLEGE_DIR_CACHE[best_match]
        logger.info(f"🧠 Fuzzy matched '{user_input}' to '{exact_name}' (Score: {highest_score:.2f})")
        return exact_name
    return None

# ─────────────── 🛠️ FILTER NORMALIZER ───────────────
def _normalize_filters(filters: dict) -> dict:
    if not filters:
        return {}
    clean = {}
    for k, v in filters.items():
        if k == "district" and isinstance(v, str):
            v = v.strip().title()
        elif k == "nba_accredited":
            if isinstance(v, str):
                v = v.strip().lower() in ("yes", "true", "1", "accredited")
            clean[k] = "true" if bool(v) else "false"
            continue
        elif k in ("tnea_code", "branch_code"):
            v = str(v).strip()
        clean[k] = v
    return clean

# ─────────────── XML FORMATTING ───────────────
def format_context_xml(docs: List[Dict]) -> str:
    ctx = "<knowledge_base>\n"
    for i, doc in enumerate(docs):
        meta = doc.get("metadata", {})
        doc_type = meta.get("doc_type", "general_info")
        source = meta.get("source", "unknown").replace("_db_df.csv", "").replace(".json", "")
        ctx += (f'<document id="{i+1}" type="{doc_type}" source="{source}" '
                f'tnea_code="{meta.get("tnea_code", "N/A")}" '
                f'district="{meta.get("district", "N/A")}">\n')
        ctx += doc.get("content", "") + "\n</document>\n"
    ctx += "</knowledge_base>"
    return ctx

def reorder_for_llm(docs: List[Dict]) -> List[Dict]:
    """Mitigates 'Lost in the Middle' phenomenon for LLMs."""
    if len(docs) < 3:
        return docs
    return [docs[0]] + docs[2:] + [docs[1]]

# ─────────────── QUERY REWRITER ───────────────
def rewrite_query_with_history(current_question: str, chat_history: list) -> str:
    global GEMINI_AVAILABLE
    if not chat_history:
        return current_question

    if not GEMINI_AVAILABLE:
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            GEMINI_AVAILABLE = True
        except Exception:
            return current_question

    pronouns = ["there", "it", "that college", "this college", "they", "its", "those"]
    if not any(p in current_question.lower() for p in pronouns):
        return current_question

    hist_text = "\n".join([f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in chat_history[-4:]])
    
    prompt = f"""You are a search query rewriter for a college counseling AI.
Look at the Chat History and the Current Question.
If the Current Question uses pronouns like "there", "it", or "that college", rewrite it to include the specific college name from the history.
If it is already specific, return it exactly as is.
Output ONLY the rewritten question. Nothing else.

Chat History:
{hist_text}

Current Question: {current_question}

Rewritten Question:"""

    try:
        import google.generativeai as genai
        # 🛠️ FIX: Corrected hallucinated model name to standard 1.5 Flash
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        rewritten = response.text.strip().strip('"')
        logger.info(f"🧠 Query Rewritten: '{current_question}' -> '{rewritten}'")
        return rewritten
    except Exception as e:
        logger.warning(f"⚠️ Query rewriting failed: {e}")
        return current_question

# ─────────────── SMART ENTITY LOOKUP ───────────────
def entity_lookup(college_name: str, limit: int = 15) -> List[Dict]:
    if not college_name or not str(college_name).strip():
        return []
        
    college_lower = college_name.lower()
    for alias, full_name in COLLEGE_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", college_lower):
            logger.info(f"🧠 Alias match: '{alias}' found in '{college_name}' -> '{full_name}'")
            college_name = full_name
            break

    search_terms = college_name.lower()
    stopwords = {"engineering", "college", "technology", "institute", "of", "and", "autonomous", "the", "for"}
    for word in stopwords:
        search_terms = search_terms.replace(word, "")
    search_terms = " ".join(search_terms.split())
    
    if len(search_terms) < 3:
        search_terms = college_name
        
    db_term = f"%{search_terms}%"
    
    try:
        res = (supabase.table("documents")
               .select("id, content, metadata")
               .filter("metadata->>college_name", "ilike", db_term)
               .limit(limit)
               .execute())
               
        if res.data:
            logger.info(f"✅ Metadata match found {len(res.data)} documents for '{college_name}'")
            return res.data
            
        logger.info(f"⚠️ Exact metadata match failed. Attempting Fuzzy Resolution...")
        resolved_name = fuzzy_resolve_college(college_name)
        
        if resolved_name:
            res_fuzzy = (supabase.table("documents")
                         .select("id, content, metadata")
                         .filter("metadata->>college_name", "ilike", f"%{resolved_name}%")
                         .limit(limit)
                         .execute())
            if res_fuzzy.data:
                logger.info(f"✅ Fuzzy fallback succeeded for '{resolved_name}'")
                return res_fuzzy.data

        res_fallback = (supabase.table("documents")
                        .select("id, content, metadata")
                        .ilike("content", f"%{college_name}%")
                        .limit(limit)
                        .execute())
                        
        if res_fallback.data:
            return res_fallback.data
        return []
        
    except Exception as e:
        logger.warning(f"Entity lookup primary failed: {e}.")
        return []

# ─────────────── 🚀 OPTIMIZED BRANCH FETCHER ───────────────
def get_branches_by_filters(district: str = None, branch_code: str = None, nba_required: str = None, limit: int = 10) -> List[Dict]:
    _load_college_caches()  # Ensure cache is warm
    try:
        q = supabase.table("documents").select("id, content, metadata").eq("metadata->>doc_type", "branch_info")
        if branch_code:
            q = q.ilike("metadata->>branch_code", f"%{branch_code}%")
        if nba_required is not None:
            if isinstance(nba_required, str):
                nba_required = nba_required.strip().lower() in ("yes", "true", "1", "accredited")
            nba_val = "true" if nba_required else "false"
            q = q.eq("metadata->>nba_accredited", nba_val)
            
        res = q.execute()
        if not res.data:
            return []

        # 🛠️ MASSIVE OPTIMIZATION: Filter district using IN-MEMORY cache instead of DB query
        if district:
            district_codes = {
                str(meta.get("tnea_code")) 
                for meta in _COLLEGE_MAP_CACHE.values() 
                if district.lower() in str(meta.get("district", "")).lower()
            }
            filtered_docs = [doc for doc in res.data if str(doc["metadata"].get("tnea_code")) in district_codes]
        else:
            filtered_docs = res.data

        if not filtered_docs:
            return []

        # 🛠️ MASSIVE OPTIMIZATION: Map college names using IN-MEMORY cache
        for doc in filtered_docs:
            tc = str(doc["metadata"].get("tnea_code"))
            if tc in _COLLEGE_MAP_CACHE:
                info = _COLLEGE_MAP_CACHE[tc]
                doc["metadata"]["college_name"] = info.get("college_name", "Unknown College")
                doc["metadata"]["district"] = info.get("district", "Unknown District")
                doc["content"] = f"college_name: {info.get('college_name')}\n" + doc.get("content", "")

        return filtered_docs[:limit]

    except Exception as e:
        logger.error(f"Direct SQL branch fetch failed: {e}", exc_info=True)
        return []

def get_top_by_pass_percentage(top_k: int, district: str = None) -> List[Dict]:
    try:
        q = (supabase.table("performance")
             .select("tnea_code, college_name, district, pass_percentage, total_appeared, total_passed")
             .order("pass_percentage", desc=True))
        if district:
            q = q.ilike("district", f"%{district}%")
        res = q.limit(top_k).execute()
        return res.data if res.data else []
    except Exception:
        return [] 

def retrieve_for_comparison(college_names: List[str], top_k: int = 5) -> Tuple[List[Dict], str]:
    if not college_names:
        return None, "Please tell me which colleges you would like to compare."
        
    all_docs = []
    for name in college_names:
        docs = entity_lookup(name, limit=5)
        if docs:
            all_docs.extend(docs)
    if not all_docs:
        return None, "I could not find the colleges you want to compare in the database."
    for c in all_docs:
        c["rerank_score"] = 10.0
    
    all_docs = all_docs[:top_k * 2] 
    return all_docs, format_context_xml(all_docs)

# ─────────────── MAIN RETRIEVE ORCHESTRATOR ───────────────
def retrieve(query: str, top_k: int = 5, filters: dict = None,
             compare_colleges: list = None, intent: str = "search", chat_history: list = None):

    _load_models()
    filters = _normalize_filters(filters)
    query = rewrite_query_with_history(query, chat_history)

    active_filters = dict(filters) if filters else {}

    if not active_filters.get("college_name") and not compare_colleges:
        resolved_fallback = fuzzy_resolve_college(query)
        if resolved_fallback:
            logger.info(f"🧠 Fallback fuzzy match extracted college: '{resolved_fallback}'")
            active_filters["college_name"] = resolved_fallback

    if not compare_colleges:
        compare_colleges = active_filters.pop("compare_colleges", None)

    if intent == "compare" and compare_colleges:
        logger.info(f"🆚 Comparison mode: {compare_colleges}")
        return retrieve_for_comparison(compare_colleges, top_k)

    numerical_metric = active_filters.pop("numerical_metric", None)
    if intent == "numerical" and numerical_metric == "pass_percentage":
        district = active_filters.get("district")
        rows = get_top_by_pass_percentage(top_k, district)
        if not rows:
            return None, "No performance data found matching your criteria."
        lines, docs = [], []
        for i, r in enumerate(rows):
            line = (f"{i+1}. {r['college_name']} (TNEA: {r['tnea_code']}, "
                    f"District: {r['district']}) - Pass Rate: {r['pass_percentage']}%, "
                    f"Passed: {r['total_passed']}/{r['total_appeared']}")
            lines.append(line)
            docs.append({"metadata": {"college_name": r["college_name"],
                                       "tnea_code": r["tnea_code"],
                                       "district": r["district"]},
                         "content": line, "rerank_score": 10.0})
        ctx = "<knowledge_base>\n" + "\n".join(lines) + "\n</knowledge_base>"
        return docs, ctx

    college_name = active_filters.pop("college_name", None)
    if college_name:
        logger.info(f"🔍 Entity lookup: {college_name}")
        exact = entity_lookup(college_name, limit=15)
        if exact:
            for c in exact:
                c["rerank_score"] = 10.0
            return exact, format_context_xml(exact)
        else:
            logger.info(f"⚠️ Exact entity lookup found nothing, falling back to vector search.")

    query_lower = query.lower()
    table_name = "documents"
    
    district_filter = active_filters.get("district")
    branch_code_filter = active_filters.get("branch_code")
    nba_filter = active_filters.get("nba_accredited")

    branch_keywords = r"\b(branch|intake|cse|cs|mechanical|me|ece|ec|nba|seat|course|computer science)\b"
    is_branch_query = bool(re.search(branch_keywords, query_lower)) or branch_code_filter or nba_filter
    
    if is_branch_query:
        logger.info("🎯 Routing to: branch_info (Direct SQL)")
        sql_docs = get_branches_by_filters(
            district=district_filter, 
            branch_code=branch_code_filter, 
            nba_required=nba_filter, 
            limit=top_k * 2
        )
        if sql_docs:
            for c in sql_docs:
                c["rerank_score"] = 10.0
            sql_docs = sql_docs[:top_k]
            return sql_docs, format_context_xml(sql_docs)
        
    elif re.search(r"\b(hostel|fee|placement|transport|mess|rent|principal|address|autonomous)\b", query_lower):
        active_filters["doc_type"] = "college_info"
        logger.info("🎯 Routing to: college_info")
        
    elif re.search(r"\b(reservation|counselling|eligibility|native|certificate|tnea rule|first graduate|community|oc|bc|mbc)\b", query_lower):
        table_name = "admission_documents"
        logger.info("🎯 Routing to: admission_documents")

    active_filters.pop("branch_code", None)
    active_filters.pop("nba_accredited", None)
    active_filters.pop("district", None) 
    
    q_emb = embedding_model.encode(QUERY_PREFIX + query, normalize_embeddings=True).tolist()
    fetch_count = top_k * 4
    
    rpc_name = "match_documents" if table_name == "documents" else "match_admission_documents"
    
    params = {
        "query_embedding": q_emb,
        "filter": active_filters if active_filters else {},
        "match_count": fetch_count
    }
        
    try:
        resp = supabase.rpc(rpc_name, params).execute()
        candidates = resp.data
    except Exception as e:
        logger.error(f"Vector search failed on {rpc_name}: {e}")
        candidates = []

    if not candidates:
        return None, "No documents found matching your criteria."

    # 🛠️ MEMORY GUARD: Explicitly manage memory during heavy reranking
    pairs = [[query, c["content"]] for c in candidates]
    scores = reranker.predict(pairs)
    for i, sc in enumerate(scores):
        candidates[i]["rerank_score"] = float(sc)
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    
    # 🛠️ MEMORY GUARD: Dereference heavy lists immediately
    del pairs
    del scores
    gc.collect()

    best_rerank = candidates[0]["rerank_score"]
    best_vector = candidates[0].get("similarity", 0.0) 
    
    if best_rerank < -2.0 and best_vector < 0.5:
        logger.warning(f"Low confidence (Rerank: {best_rerank:.2f}, Vector: {best_vector:.2f}). Aborting.")
        return None, "I don't have enough specific information in my database to answer this accurately."

    top_docs = candidates[:top_k]
    top_docs = reorder_for_llm(top_docs)
    
    # 🛠️ MEMORY GUARD: Clean up remaining candidates list
    del candidates
    
    return top_docs, format_context_xml(top_docs)