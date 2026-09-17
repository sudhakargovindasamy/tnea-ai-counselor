import logging
import difflib
import os
import re
import gc
import json
import torch
from typing import List, Dict, Any, Optional, Tuple
from app.services.database import supabase

# ═══════════════════════════════════════════════════════════
# 🚨 512MB RAM OPTIMIZATION: Force single-threaded inference
# ═══════════════════════════════════════════════════════════
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

logger = logging.getLogger(__name__)

# ═══════════════ 🧠 STRICT LAZY LOADING & MEMORY GUARDS ═══════════════
embedding_model = None
reranker = None
GEMINI_AVAILABLE = False

def _load_models():
    global embedding_model, reranker
    
    if embedding_model is None:
        logger.info("🧠 Loading embedding model (MiniLM-L6-v2)...")
        torch.set_num_threads(1)  # 🚨 Critical for memory
        try:
            from sentence_transformers import SentenceTransformer
            embedding_model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                device="cpu"
            )
            logger.info("✅ Embedding model loaded.")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            
    if reranker is None:
        logger.info("🧠 Loading Cross-Encoder reranker (ms-marco-MiniLM-L-6-v2)...")
        try:
            from sentence_transformers import CrossEncoder
            reranker = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                device="cpu"
            )
            logger.info("✅ Reranker loaded.")
        except Exception as e:
            logger.error(f"❌ Failed to load reranker: {e}")
            
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("🧹 Memory cleanup complete.")

QUERY_PREFIX = ""

# ═══════════════ 🧠 ALIAS DICTIONARY (EXACT CSV MATCHES) ═══════════════
COLLEGE_ALIASES = {
    "ceg": "University Departments of Anna University , Chennai - CEG Campus",
    "mit": "University Departments of Anna University , Chennai - MIT Campus",
    "act": "University Departments of Anna University , Chennai - ACT Campus",
    "psg": "PSG College of Technology",
    "psg tech": "PSG College of Technology",
    "psg college of technology": "PSG College of Technology",
    "ssn": "SSN College of Engineering",
    "ssn college of engineering": "SSN College of Engineering",
    "svce": "Sri Venkateswara College of Engineering",
    "sairam": "Sri Sai Ram Engineering College",  
    "thiagarajar": "Thiagarajar College of Engineering",
    "tce": "Thiagarajar College of Engineering",
    "thiagarajar college of engineering": "Thiagarajar College of Engineering",
    "kct": "Kumaraguru College of Technology",
    "kumaraguru": "Kumaraguru College of Technology",
    "kumaraguru college of technology": "Kumaraguru College of Technology",
    "skcet": "Sri Krishna College of Engineering and Technology", 
    "sri krishna": "Sri Krishna College of Engineering and Technology",
    "bitsathy": "Bannari Amman Institute of Technology",
    "bannari amman": "Bannari Amman Institute of Technology",
    "bannari amman institute of technology": "Bannari Amman Institute of Technology",
    "bannari amman institute": "Bannari Amman Institute of Technology",
    "bannari": "Bannari Amman Institute of Technology",
    "cit": "Coimbatore Institute of Technology",
    "coimbatore institute of technology": "Coimbatore Institute of Technology",
    "easwari": "Easwari Engineering College",
    "panimalar": "Panimalar Engineering College",
    "rajalakshmi": "Rajalakshmi Engineering College",
    "rec": "Rajalakshmi Engineering College",
    "velammal": "Velammal Engineering College",
    "sona": "Sona College of Technology",
    "kongu": "Kongu Engineering College",
    "mepco": "Mepco Schlenk Engineering College",
    "saveetha": "Saveetha Engineering College",
    "rmk": "R M K Engineering College",
    "srm": "SRM Institute of Science and Technology",
    "gct": "Government College of Technology",
    "government college of technology": "Government College of Technology"
}

# ═══════════════ 🎓 BRANCH SYNONYMS & CANONICAL MAPPINGS ═══════════════
BRANCH_SYNONYMS = {
    "computer science": "CS", "computer science courses": "CS", "computer science and engineering": "CS",
    "computer science engineering": "CS", "cse": "CS", "cs": "CS",
    "artificial intelligence and data science": "AD", "ai & ds": "AD", "ai and ds": "AD", "ai&ds": "AD",
    "ai & data science": "AD", "ai and data science": "AD", "ad": "AD",
    "ai & ml": "AL", "ai and ml": "AL", "ai&ml": "AL", "artificial intelligence and machine learning": "AL", "al": "AL",
    "cyber security": "CY", "cybersecurity": "CY", "cyber security specialization": "CY", "cy": "CY",
    
    # ❌ REMOVED "me": "ME" to prevent false positive on "tell me about..."
    "mechanical": "ME", "mechanical engineering": "ME", "mechanical engineering courses": "ME", "mech": "ME",
    
    "electronics and communication": "EC", "electronics and communication engineering": "EC", "ece": "EC", "ec": "EC",
    "electrical and electronics": "EE", "electrical and electronics engineering": "EE", "eee": "EE", "ee": "EE",
    "civil": "CE", "civil engineering": "CE", "ce": "CE",
    
    # ❌ REMOVED "it": "IT" to prevent false positive on "tell me about it..."
    "information technology": "IT",
    
    "computer science and business systems": "CB", "csbs": "CB", "cb": "CB",
    "mechatronics": "MZ", "mechatronics engineering": "MZ", "mz": "MZ",
    "biotechnology": "BT", "bio technology": "BT", "biotech": "BT", "bt": "BT",
    "biomedical": "BM", "bio medical": "BM", "biomedical engineering": "BM", "bm": "BM",
    "agricultural": "AG", "agriculture": "AG", "agri": "AG", "ag": "AG",
    "chemical": "CH", "chemical engineering": "CH", "ch": "CH",
    "automobile": "AU", "automobile engineering": "AU", "au": "AU",
    "aeronautical": "AE", "aeronautical engineering": "AE", "ae": "AE",
    "textile": "TX", "textile technology": "TX", "fashion": "FT", "fashion technology": "FT"
}

DISTRICT_SYNONYMS = {
    "kancheepuram": "Kanchipuram", "kanchipuram": "Kanchipuram", "chengalpet": "Chengalpattu", "chengalpattu": "Chengalpattu",
    "trichirappalli": "Tiruchirappalli", "trichy": "Tiruchirappalli", "kanniyakumari": "Kanyakumari", "kanyakumari": "Kanyakumari",
    "villupuram": "Viluppuram", "viluppuram": "Viluppuram", "the nilgiris": "Nilgiris", "nilgiris": "Nilgiris",
    "salem": "Salem", "coimbatore": "Coimbatore", "chennai": "Chennai", "madurai": "Madurai", "erode": "Erode",
    "thiruvallur": "Tiruvallur", "tiruvallur": "Tiruvallur", "thiruvannamalai": "Tiruvannamalai", "tiruvannamalai": "Tiruvannamalai",
    "thiruvarur": "Tiruvarur", "tiruvarur": "Tiruvarur", "thirunelveli": "Tirunelveli", "tirunelveli": "Tirunelveli",
    "dharmapuri": "Dharmapuri", "krishnagiri": "Krishnagiri", "namakkal": "Namakkal", "dindigul": "Dindigul",
    "thanjavur": "Thanjavur", "karur": "Karur", "cuddalore": "Cuddalore", "vellore": "Vellore",
    "virudhunagar": "Virudhunagar", "sivagangai": "Sivagangai", "ranipet": "Ranipet", "tirupattur": "Tirupattur"
}

TAMIL_NADU_DISTRICTS = set(DISTRICT_SYNONYMS.keys()) | {v.lower() for v in DISTRICT_SYNONYMS.values()}

# ═══════════════ 🚀 GLOBAL CACHES & LOCAL FALLBACK ═══════════════
_COLLEGE_MAP_CACHE: Dict[str, Dict] = {}
_COLLEGE_DIR_CACHE: Dict[str, str] = {}
_COLLEGE_DIR_CLEAN_CACHE: Dict[str, str] = {}
_LOCAL_DOCUMENTS: Optional[List[Dict[str, Any]]] = None
_LOCAL_ADMISSION_DOCUMENTS: Optional[List[Dict[str, Any]]] = None

def _get_local_documents() -> List[Dict[str, Any]]:
    """
    Loads full local documents for fallback. 
    ~450 docs * 2KB = ~900KB RAM, perfectly safe for 512MB limits.
    Truncating previously destroyed RAG context (intakes/fees are at the end of docs).
    """
    global _LOCAL_DOCUMENTS
    if _LOCAL_DOCUMENTS is not None:
        return _LOCAL_DOCUMENTS
        
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    doc_path = os.path.join(base_dir, "data", "processed", "college_documents.json")
    if os.path.exists(doc_path):
        try:
            with open(doc_path, "r", encoding="utf-8") as f:
                _LOCAL_DOCUMENTS = json.load(f)
            logger.info(f"📂 Loaded {len(_LOCAL_DOCUMENTS)} full local college documents for fallback.")
            return _LOCAL_DOCUMENTS
        except Exception as e:
            logger.warning(f"Could not load local documents: {e}")
    _LOCAL_DOCUMENTS = []
    return _LOCAL_DOCUMENTS

def _get_local_admission_documents() -> List[Dict[str, Any]]:
    """Loads admission rules for zero-dependency fallback."""
    global _LOCAL_ADMISSION_DOCUMENTS
    if _LOCAL_ADMISSION_DOCUMENTS is not None:
        return _LOCAL_ADMISSION_DOCUMENTS
        
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    doc_path = os.path.join(base_dir, "data", "processed", "admission_documents.json")
    if os.path.exists(doc_path):
        try:
            with open(doc_path, "r", encoding="utf-8") as f:
                _LOCAL_ADMISSION_DOCUMENTS = json.load(f)
            logger.info(f"📂 Loaded {len(_LOCAL_ADMISSION_DOCUMENTS)} local admission documents for fallback.")
            return _LOCAL_ADMISSION_DOCUMENTS
        except Exception as e:
            logger.warning(f"Could not load local admission documents: {e}")
    _LOCAL_ADMISSION_DOCUMENTS = []
    return _LOCAL_ADMISSION_DOCUMENTS

def _load_college_caches():
    global _COLLEGE_MAP_CACHE, _COLLEGE_DIR_CACHE, _COLLEGE_DIR_CLEAN_CACHE
    if _COLLEGE_MAP_CACHE:
        return

    stopwords = {"engineering", "college", "technology", "institute", "of", "and", 
                 "autonomous", "the", "for", "engg", "eng", "tech", "enginering", "technolgoy"}

    try:
        res = supabase.table("documents").select("metadata").eq("metadata->>doc_type", "college_info").execute()
        if res.data:
            for row in res.data:
                meta = row.get("metadata", {})
                tnea_code = str(meta.get("tnea_code", ""))
                name = meta.get("college_name", "")
                if tnea_code and name:
                    _COLLEGE_MAP_CACHE[tnea_code] = meta
                    name_lower = name.lower()
                    _COLLEGE_DIR_CACHE[name_lower] = name
                    core_name = name_lower.split(",")[0].split("(")[0].strip()
                    clean_name = core_name
                    for word in stopwords:
                        clean_name = clean_name.replace(word, "")
                    clean_name = " ".join(clean_name.split())
                    if clean_name and clean_name not in TAMIL_NADU_DISTRICTS:
                        _COLLEGE_DIR_CLEAN_CACHE[name_lower] = clean_name
            logger.info(f"✅ Cached {len(_COLLEGE_MAP_CACHE)} colleges into memory from Supabase.")
            return
    except Exception as e:
        logger.debug(f"Supabase cache warmup skipped (using local fallback): {e}")

    local_docs = _get_local_documents()
    for doc in local_docs:
        meta = doc.get("metadata", {})
        tnea_code = str(meta.get("tnea_code", ""))
        name = meta.get("college_name", "")
        if tnea_code and name:
            _COLLEGE_MAP_CACHE[tnea_code] = meta
            name_lower = name.lower()
            _COLLEGE_DIR_CACHE[name_lower] = name
            core_name = name_lower.split(",")[0].split("(")[0].strip()
            clean_name = core_name
            for word in stopwords:
                clean_name = clean_name.replace(word, "")
            clean_name = " ".join(clean_name.split())
            if clean_name and clean_name not in TAMIL_NADU_DISTRICTS:
                _COLLEGE_DIR_CLEAN_CACHE[name_lower] = clean_name

    logger.info(f"✅ Cached {len(_COLLEGE_MAP_CACHE)} colleges into memory from local dataset.")

# ═══════════════ 🧠 IN-MEMORY FUZZY RESOLVER ═══════════════
def fuzzy_resolve_college(user_input: str) -> Optional[str]:
    if not user_input:
        return None
    _load_college_caches()
    if not _COLLEGE_DIR_CACHE:
        return None

    expanded_input = user_input.lower().strip()
    for alias, full_name in COLLEGE_ALIASES.items():
        pattern = rf"\b{re.escape(alias)}\b"
        if re.search(pattern, expanded_input):
            logger.info(f"🧠 Alias match: '{alias}' -> '{full_name}'")
            return full_name

    stopwords = {"engineering", "college", "colleges", "technology", "institute", "of", "and", 
                 "autonomous", "the", "for", "engg", "eng", "tech", "offer", "does", "is", 
                 "available", "courses", "which", "what", "are", "in"}
    
    clean_input = expanded_input
    for word in stopwords:
        clean_input = re.sub(rf"\b{re.escape(word)}\b", "", clean_input)
    clean_input = " ".join(clean_input.split())
    input_nospace = clean_input.replace(" ", "")

    if not input_nospace or clean_input in TAMIL_NADU_DISTRICTS or input_nospace in TAMIL_NADU_DISTRICTS:
        return None
    
    best_match = None
    highest_score = 0.0
    
    for db_name_lower, clean_db in _COLLEGE_DIR_CLEAN_CACHE.items():
        db_nospace = clean_db.replace(" ", "")
        
        if db_nospace in TAMIL_NADU_DISTRICTS or clean_db in TAMIL_NADU_DISTRICTS:
            continue
            
        if len(db_nospace) >= 5 and db_nospace in input_nospace:
            return _COLLEGE_DIR_CACHE[db_name_lower]
        if len(input_nospace) >= 5 and input_nospace in db_nospace:
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

# ─────────────── 🛠️ QUERY PARSING HELPERS ───────────────
def extract_district(query: str, filters: dict = None) -> Optional[str]:
    if filters and filters.get("district"):
        raw_d = str(filters["district"]).strip().lower()
        if raw_d in DISTRICT_SYNONYMS:
            return DISTRICT_SYNONYMS[raw_d]
        return filters["district"].strip().title()

    query_lower = query.lower()
    for d_key, d_norm in DISTRICT_SYNONYMS.items():
        if re.search(rf"\b{re.escape(d_key)}\b", query_lower):
            return d_norm
    return None

def extract_branch_code(query: str, filters: dict = None) -> Optional[str]:
    if filters and filters.get("branch_code"):
        code = str(filters["branch_code"]).strip().upper()
        if code:
            return code

    query_lower = query.lower()
    sorted_synonyms = sorted(BRANCH_SYNONYMS.items(), key=lambda x: len(x[0]), reverse=True)
    for synonym, code in sorted_synonyms:
        pattern = rf"\b{re.escape(synonym)}\b"
        if re.search(pattern, query_lower):
            logger.info(f"🎯 Branch detected from query: '{synonym}' -> '{code}'")
            return code
    return None

def _normalize_filters(filters: dict) -> dict:
    if not filters:
        return {}
    clean = {}
    for k, v in filters.items():
        if k == "district" and isinstance(v, str):
            raw_v = v.strip().lower()
            clean[k] = DISTRICT_SYNONYMS.get(raw_v, v.strip().title())
        elif k == "nba_accredited":
            if isinstance(v, str):
                v = v.strip().lower() in ("yes", "true", "1", "accredited")
            clean[k] = bool(v)
        elif k == "autonomous":
            if isinstance(v, str):
                v = v.strip().lower() in ("yes", "true", "1")
            clean[k] = bool(v)
        elif k in ("tnea_code", "branch_code"):
            clean[k] = str(v).strip().upper()
        else:
            clean[k] = v
    return clean

# ─────────────── 🆕 STRICT DEDUPLICATION FUNCTION ───────────────
def deduplicate_docs(docs: List[Dict]) -> List[Dict]:
    if not docs:
        return []
    
    seen_codes = set()
    seen_hashes = set()
    unique_docs = []
    
    for doc in docs:
        meta = doc.get("metadata", {})
        tnea_code = str(meta.get("tnea_code", "")).strip()
        content = doc.get("content", "")
        content_hash = hash(content[:500])
        
        if tnea_code and tnea_code not in ("N/A", "None", ""):
            if tnea_code in seen_codes:
                continue
            seen_codes.add(tnea_code)
            unique_docs.append(doc)
        else:
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_docs.append(doc)
    
    return unique_docs

# ─────────────── 🆕 CUTOFF QUERY HANDLER (ST-007) ───────────────
def handle_cutoff_query(query: str, filters: dict = None, top_k: int = 5) -> Tuple[Optional[List[Dict]], Optional[str]]:
    q_lower = query.lower()
    if "cutoff" in q_lower or "closing rank" in q_lower:
        msg = ("I don't have cutoff/closing rank data in my database. I can help with college facilities, "
               "branches, and admission rules. For cutoff predictions, check tneaonline.org.")
        logger.info("🎯 Cutoff query detected: Gracefully declined without vector search.")
        return None, msg
    return None, None

# ─────────────── XML FORMATTING ───────────────
def format_context_xml(docs: List[Dict]) -> str:
    ctx = "<knowledge_base>\n"
    for i, doc in enumerate(docs):
        meta = doc.get("metadata", {})
        doc_type = meta.get("doc_type", "college_info")
        source = meta.get("source", "unknown").replace("_db_df.csv", "").replace(".json", "")
        tnea_code = meta.get("tnea_code", "N/A")
        district = meta.get("district", "N/A")
        # Handle admission docs which use 'section' instead of 'college_name'
        college_name = meta.get("college_name", meta.get("section", "N/A")) 
        
        ctx += (f'<document id="{i+1}" type="{doc_type}" source="{source}" '
                f'tnea_code="{tnea_code}" district="{district}" college="{college_name}">\n')
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
Look at Chat History and Current Question.
Rewrite pronoun references ("there", "it", "that college") into the specific college name.
Output ONLY the rewritten question.

Chat History:
{hist_text}

Current Question: {current_question}
Rewritten Question:"""

    try:
        import google.generativeai as genai
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        rewritten = response.text.strip().strip('"')
        logger.info(f"🧠 Query Rewritten: '{current_question}' -> '{rewritten}'")
        return rewritten
    except Exception as e:
        logger.warning(f"⚠️ Query rewriting failed: {e}")
        return current_question

# ─────────────── SMART ENTITY LOOKUP ───────────────
def entity_lookup(college_name: str, limit: int = 10) -> List[Dict]:
    if not college_name or not str(college_name).strip():
        return []
        
    resolved_alias = COLLEGE_ALIASES.get(college_name.lower().strip())
    if resolved_alias:
        college_name = resolved_alias

    local_docs = _get_local_documents()
    name_clean = college_name.lower().split(",")[0].split("(")[0].strip()

    matched = []
    for d in local_docs:
        c_name = d.get("metadata", {}).get("college_name", "").lower()
        if name_clean in c_name:
            matched.append(d)

    if matched:
        logger.info(f"✅ Found {len(matched)} matching documents locally for '{college_name}'")
        return matched[:limit]

    try:
        search_terms = name_clean
        stopwords = {"engineering", "college", "technology", "institute", "of", "and", "autonomous", "the", "for"}
        words = [w for w in search_terms.split() if w not in stopwords]
        db_term = f"%{' '.join(words)}%" if words else f"%{search_terms}%"
        
        res = (supabase.table("documents")
               .select("id, content, metadata")
               .ilike("metadata->>college_name", db_term)
               .limit(limit)
               .execute())
        if res.data:
            return res.data
    except Exception as e:
        logger.debug(f"Supabase entity lookup exception: {e}")

    fuzzy_name = fuzzy_resolve_college(college_name)
    if fuzzy_name:
        for d in local_docs:
            if fuzzy_name.lower().split(",")[0].strip() in d.get("metadata", {}).get("college_name", "").lower():
                return [d]

    return []

# ─────────────── 🚀 DIRECT CATALOG FILTERING ───────────────
def get_colleges_by_filters(district: str = None, branch_code: str = None, 
                            has_hostel: bool = False, limit: int = 10) -> List[Dict]:
    local_docs = _get_local_documents()
    candidates = []

    norm_district = DISTRICT_SYNONYMS.get(district.lower().strip(), district.title()) if district else None

    for d in local_docs:
        meta = d.get("metadata", {})
        c_district = meta.get("district", "")
        c_branches = meta.get("branch_codes", [])
        
        if norm_district and norm_district.lower() != c_district.lower():
            continue
        if branch_code and branch_code not in c_branches:
            continue
        if has_hostel:
            has_boys = bool(meta.get("hostel_facilities_boys"))
            has_girls = bool(meta.get("hostel_facilities_girls"))
            has_mess = meta.get("mess_bill_boys") is not None or meta.get("mess_bill_girls") is not None
            has_room = meta.get("room_rent_boys") is not None or meta.get("room_rent_girls") is not None
            if not (has_boys or has_girls or has_mess or has_room):
                continue

        candidates.append(d)

    if candidates:
        logger.info(f"🎯 Filter match: Found {len(candidates)} colleges (District: {norm_district}, Branch: {branch_code}, Hostel: {has_hostel})")
        return deduplicate_docs(candidates)[:limit]

    try:
        q = supabase.table("documents").select("id, content, metadata").eq("metadata->>doc_type", "college_info")
        if norm_district:
            q = q.ilike("metadata->>district", norm_district)
        res = q.limit(limit * 2).execute()
        if res.data:
            filtered = res.data
            if branch_code:
                filtered = [doc for doc in filtered if branch_code in doc.get("metadata", {}).get("branch_codes", [])]
            return deduplicate_docs(filtered)[:limit]
    except Exception as e:
        logger.debug(f"Supabase filter query exception: {e}")

    return []

# ─────────────── MAIN RETRIEVE ORCHESTRATOR ───────────────
def retrieve(query: str, top_k: int = 5, filters: dict = None,
             compare_colleges: list = None, intent: str = "search", chat_history: list = None):
    """
    Production hybrid retrieval orchestrator with Cross-Encoder reranking.
    """
    _load_models()
    filters = _normalize_filters(filters)

    # Resolve coreferences from chat history (e.g. "what is the fee there?")
    query = rewrite_query_with_history(query, chat_history)

    active_filters = dict(filters) if filters else {}
    query_lower = query.lower()

    # 1. CUTOFF / CLOSING RANK QUERIES
    if "cutoff" in query_lower or "closing rank" in query_lower:
        cutoff_docs, cutoff_msg = handle_cutoff_query(query, active_filters, top_k)
        return cutoff_docs, cutoff_msg

    # Extract district and branch code
    district = extract_district(query, active_filters)
    branch_code = extract_branch_code(query, active_filters)

    if district and "district" not in active_filters:
        active_filters["district"] = district
    if branch_code and "branch_code" not in active_filters:
        active_filters["branch_code"] = branch_code

    # 2. SPECIFIC COLLEGE ENTITY LOOKUP
    detected_college = None
    if not compare_colleges:
        college_from_filter = active_filters.pop("college_name", None)
        if college_from_filter:
            detected_college = college_from_filter
        else:
            for alias, full_name in COLLEGE_ALIASES.items():
                if re.search(rf"\b{re.escape(alias)}\b", query_lower):
                    detected_college = full_name
                    break

    if detected_college:
        logger.info(f"🔍 Entity lookup: '{detected_college}'")
        exact = entity_lookup(detected_college, limit=5)
        if exact:
            exact = deduplicate_docs(exact)
            for c in exact:
                c["rerank_score"] = 10.0
            logger.info(f"✅ Found {len(exact)} documents for college entity '{detected_college}'")
            return exact[:top_k], format_context_xml(exact[:top_k])

    # 3. DIRECT CATALOG FILTERING
    is_hostel_query = "hostel" in query_lower or "mess" in query_lower or "room rent" in query_lower
    is_college_list_query = bool(district) and (
        "colleges" in query_lower or 
        "engineering colleges" in query_lower or 
        "what engineering" in query_lower or 
        "which college" in query_lower or 
        "what college" in query_lower or 
        "offer" in query_lower or 
        "list" in query_lower or 
        is_hostel_query
    )

    if is_college_list_query:
        catalog_docs = get_colleges_by_filters(
            district=district,
            branch_code=branch_code,
            has_hostel=is_hostel_query,
            limit=top_k
        )
        if catalog_docs:
            catalog_docs = deduplicate_docs(catalog_docs)
            for c in catalog_docs:
                c["rerank_score"] = 9.0
            return catalog_docs, format_context_xml(catalog_docs)

    # 4. FALLBACK FUZZY ENTITY RESOLUTION
    if not compare_colleges:
        fuzzy_match = fuzzy_resolve_college(query)
        if fuzzy_match:
            exact = entity_lookup(fuzzy_match, limit=5)
            if exact:
                exact = deduplicate_docs(exact)
                for c in exact:
                    c["rerank_score"] = 10.0
                return exact[:top_k], format_context_xml(exact[:top_k])

    # 5. ROUTING: ADMISSION RULES VS COLLEGE INFO
    table_name = "documents"
    rpc_name = "match_documents"
    is_admission_query = bool(re.search(
        r"\b(reservation|counselling|eligibility|native|certificate|tnea rule|first graduate|community|oc|bc|mbc|sc|st|7\.5%|quota)\b",
        query_lower
    ))
    if is_admission_query:
        table_name = "admission_documents"
        rpc_name = "match_admission_documents"

    # 6. HYBRID VECTOR SEARCH WITH SUPABASE RPC PRE-FILTER
    supabase_filter = {}
    if table_name == "documents":
        supabase_filter["doc_type"] = "college_info"
        if district:
            supabase_filter["district"] = district
        if branch_code:
            supabase_filter["branch_codes"] = [branch_code]

    candidates = []
    if embedding_model:
        q_emb = embedding_model.encode(QUERY_PREFIX + query, normalize_embeddings=True).tolist()
        fetch_count = max(top_k * 3, 15)
        
        try:
            resp = supabase.rpc(rpc_name, {
                "query_embedding": q_emb,
                "filter": supabase_filter,
                "match_count": fetch_count
            }).execute()
            candidates = resp.data or []
        except Exception as e:
            logger.debug(f"Supabase RPC search skipped: {e}")

    # Fallback to local documents if Supabase returned 0 or errored
    if not candidates:
        if table_name == "admission_documents":
            candidates = _get_local_admission_documents()
        else:
            local_docs = _get_local_documents()
            filtered_local = []
            for d in local_docs:
                meta = d.get("metadata", {})
                if district and district.lower() != meta.get("district", "").lower():
                    continue
                if branch_code and branch_code not in meta.get("branch_codes", []):
                    continue
                filtered_local.append(d)
            candidates = filtered_local if filtered_local else local_docs[:15]

    # 7. STRICT DEDUPLICATION
    candidates = deduplicate_docs(candidates)

    if not candidates:
        return None, "The provided TNEA database does not contain information to answer this."

    # 8. CROSS-ENCODER RERANKING (Memory Optimized)
    if reranker:
        MAX_RERANK_CANDIDATES = 15
        rerank_pool = candidates[:MAX_RERANK_CANDIDATES]
        
        pairs = [[query, c.get("content", "")] for c in rerank_pool]
        
        with torch.inference_mode():
            scores = reranker.predict(pairs)
            
        for i, sc in enumerate(scores):
            rerank_pool[i]["rerank_score"] = float(sc)
            
        rerank_pool.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        del pairs
        del scores
        gc.collect()
        
        top_docs = deduplicate_docs(rerank_pool)[:top_k]
    else:
        top_docs = candidates[:top_k]
        
    top_docs = reorder_for_llm(top_docs)

    return top_docs, format_context_xml(top_docs)