import logging
import difflib
from sentence_transformers import SentenceTransformer, CrossEncoder
from app.services.database import supabase

logger = logging.getLogger(__name__)

logger.info("Loading embedding model...")
embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
logger.info("Loading reranker...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ═══════════════ 🧠 ALIAS DICTIONARY (For famous abbreviations) ═══════════════
# Maps common short forms and colloquial names to their exact database substrings
COLLEGE_ALIASES = {
    "ssn": "sri sivasubramaniya nadar",
    "psg": "psg college of technology",
    "psg tech": "psg college of technology",
    "cit": "coimbatore institute of technology",
    "ceg": "college of engineering, guindy",
    "mit": "madras institute of technology",
    "svce": "sri venkateswara college of engineering",
    "srm": "srm institute of science and technology",
    "sathyabama": "sathyabama institute of science and technology",
    "vit": "vellore institute of technology",
    "skcet": "sri krishna college of engineering and technology",
    "kct": "kumaraguru college of technology",
    "rmk": "rmk engineering college",
    "sairam": "sri sai ram",
    "saveetha": "saveetha engineering college",
    "citchennai": "chennai institute of technology",
    "kongu": "kongu engineering college",
    "bitsathy": "bannari amman institute of technology",
    "bannari": "bannari amman institute of technology",
    "mepco": "mepco schlenk engineering college",
    "thiagarajar": "thiagarajar college of engineering",
    "easwari": "easwari engineering college",
    "panimalar": "panimalar engineering college",
    "rajalakshmi": "rajalakshmi engineering college",
    "velammal": "velammal engineering college",
    "sona": "sona college of technology",
    "kpr": "k p r institute of engineering and technology",
    "sns": "sns college of technology",
    "hindustan": "hindusthan college of engineering and technology"
}

# ═══════════════ 🧠 IN-MEMORY FUZZY RESOLVER ═══════════════
COLLEGE_DIRECTORY = {}

def load_college_directory():
    """Loads all unique college names into memory at startup for instant fuzzy matching."""
    global COLLEGE_DIRECTORY
    if COLLEGE_DIRECTORY:
        return # Already loaded
        
    try:
        # Fetch all unique college names from your documents table
        res = supabase.table("documents").select("metadata->>college_name").execute()
        
        seen = set()
        for row in res.data:
            name = row.get("college_name")
            if name and name not in seen:
                seen.add(name)
                COLLEGE_DIRECTORY[name.lower()] = name
                
        logger.info(f"✅ Loaded {len(COLLEGE_DIRECTORY)} colleges into memory for fuzzy matching.")
    except Exception as e:
        logger.error(f"Failed to load college directory: {e}")

def fuzzy_resolve_college(user_input: str) -> str:
    """
    Finds the closest matching college name, handling missing spaces, typos, shortened names, and aliases.
    """
    if not COLLEGE_DIRECTORY:
        load_college_directory()
        
    if not COLLEGE_DIRECTORY:
        return None

    # 🚀 STEP 1: Expand Aliases BEFORE doing any math (Handles "SSN" -> "Sri Sivasubramaniya Nadar")
    expanded_input = user_input.lower()
    for alias, full_name in COLLEGE_ALIASES.items():
        # Check if the alias exists as a distinct word in the user's prompt
        if f" {alias} " in f" {expanded_input} " or expanded_input == alias:
            expanded_input = expanded_input.replace(alias, full_name)
            logger.info(f"🧠 Alias expanded '{alias}' -> '{full_name}'")
            break

    # 🛠️ STEP 2: Stopwords list including CSV typos
    stopwords = [
        "engineering", "college", "technology", "institute", "of", "and", 
        "autonomous", "the", "for", "engg", "eng", "tech", "enginering", "technolgoy"
    ]
    
    clean_input = expanded_input
    for word in stopwords:
        clean_input = clean_input.replace(word, "")
    clean_input = " ".join(clean_input.split())
    
    best_match = None
    highest_score = 0.0
    
    for db_name_lower, db_name_exact in COLLEGE_DIRECTORY.items():
        clean_db = db_name_lower
        for word in stopwords:
            clean_db = clean_db.replace(word, "")
        clean_db = " ".join(clean_db.split())
        
        # 1. Standard Sequence Matching
        score1 = difflib.SequenceMatcher(None, clean_input, clean_db).ratio()
        
        # 2. Space-Removed Matching
        input_nospace = clean_input.replace(" ", "")
        db_nospace = clean_db.replace(" ", "")
        score2 = difflib.SequenceMatcher(None, input_nospace, db_nospace).ratio()
        
        # 🚀 STEP 3: SUBSTRING BOOST (The Silver Bullet)
        # If the user's cleaned input is fully contained inside the DB name (e.g., "sairam" inside "srisairam"),
        # it's almost certainly a match. We force a perfect score.
        # (We require len >= 4 to prevent short words like "sri" from triggering false positives)
        if len(input_nospace) >= 4 and (input_nospace in db_nospace or db_nospace in input_nospace):
            score2 = 1.0 
            
        score = max(score1, score2)
        
        if score > highest_score:
            highest_score = score
            best_match = db_name_exact
            
    # Threshold: 75% similarity required to prevent false positives
    if highest_score > 0.75:
        logger.info(f"🧠 Fuzzy matched '{user_input}' to '{best_match}' (Score: {highest_score:.2f})")
        return best_match
        
    return None

# ─────────────── XML FORMATTING ───────────────
def format_context_xml(docs):
    """Formats retrieved documents into clean XML for the LLM context window."""
    ctx = "<knowledge_base>\n"
    for i, doc in enumerate(docs):
        meta = doc.get("metadata", {})
        # Extract source type (colleges, branches, performance) to help LLM distinguish data types
        source = meta.get("source", "unknown").replace("_db_df.csv", "")
        ctx += (f'<document id="{i+1}" source="{source}" tnea_code="{meta.get("tnea_code", "N/A")}" '
                f'district="{meta.get("district", "N/A")}">\n')
        ctx += doc.get("content", "") + "\n</document>\n"
    ctx += "</knowledge_base>"
    return ctx

def reorder_for_llm(docs):
    """Puts the most relevant doc first, third second, etc. to avoid middle-loss in LLMs."""
    if len(docs) < 3:
        return docs
    return [docs[0]] + docs[2:] + [docs[1]]

# ─────────────── SMART ENTITY LOOKUP (specific college) ───────────────
def entity_lookup(college_name: str, limit: int = 15):
    """
    Fetches ALL related documents (Profile, Branches, Performance) 
    for a specific college to provide comprehensive context to the LLM.
    """
    search_terms = college_name.lower()
    # Remove common stopwords to improve match rate
    stopwords = ["engineering", "college", "technology", "institute", "of", "and", "autonomous", "the", "for"]
    for word in stopwords:
        search_terms = search_terms.replace(word, "")
    search_terms = " ".join(search_terms.split())
    
    if len(search_terms) < 3:
        search_terms = college_name
        
    db_term = f"%{search_terms}%"
    
    try:
        # 1. Primary search: Match against the college_name in metadata (Fast & Accurate)
        res = (supabase.table("documents")
               .select("id, content, metadata")
               .filter("metadata->>college_name", "ilike", db_term)
               .limit(limit)
               .execute())
               
        if res.data:
            logger.info(f"✅ Metadata match found {len(res.data)} documents for '{college_name}'")
            return res.data
            
        # 🚀 2. Fuzzy Match Fallback (Handles 'sairam' -> 'Sri Sai Ram' & 'SSN' -> 'Sri Sivasubramaniya Nadar')
        logger.info(f"⚠️ Exact metadata match failed. Attempting Fuzzy Resolution for '{college_name}'...")
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

        # 3. Fallback: Strict Content Search (Only if fuzzy fails)
        logger.info(f"⚠️ Fuzzy failed, falling back to strict content search.")
        res_fallback = (supabase.table("documents")
                        .select("id, content, metadata")
                        .ilike("content", f"%{college_name}%") # Use original name
                        .limit(limit)
                        .execute())
                        
        if res_fallback.data:
            logger.info(f"✅ Content fallback found {len(res_fallback.data)} documents")
            return res_fallback.data
            
        return []
        
    except Exception as e:
        logger.warning(f"Entity lookup primary failed: {e}. Trying content fallback...")
        try:
            res_fallback = (supabase.table("documents")
                            .select("id, content, metadata")
                            .ilike("content", f"%{college_name}%")
                            .limit(limit)
                            .execute())
            return res_fallback.data if res_fallback.data else []
        except Exception as e2:
            logger.error(f"Fallback entity lookup also failed: {e2}")
            return []

# ─────────────── BRANCH SQL FILTER ───────────────
def get_branch_codes(branch_code, nba_required):
    q = supabase.table("branches").select("tnea_code")
    if branch_code:
        q = q.ilike("branch_code", f"%{branch_code}%")
    if nba_required is not None:
        # Handle boolean/string conversion for NBA accreditation
        if isinstance(nba_required, bool):
            nba_val = "Yes" if nba_required else "No"
            q = q.ilike("nba_accredited", nba_val)
        else:
            q = q.ilike("nba_accredited", f"%{nba_required}%")
    res = q.execute()
    return list(set(r["tnea_code"] for r in res.data)) if res.data else []

# ─────────────── NUMERICAL (pass percentage) ───────────────
def get_top_by_pass_percentage(top_k, district=None):
    q = (supabase.table("performance")
         .select("tnea_code, college_name, district, pass_percentage, total_appeared, total_passed")
         .order("pass_percentage", desc=True))
    if district:
        q = q.ilike("district", f"%{district}%")
    res = q.limit(top_k).execute()
    return res.data if res.data else []

# ─────────────── COMPARISON MODE ───────────────
def retrieve_for_comparison(college_names, top_k=5):
    all_docs = []
    for name in college_names:
        # Fetch up to 5 docs per college for comparison
        docs = entity_lookup(name, limit=5)
        if docs:
            all_docs.extend(docs)
    if not all_docs:
        return None, "I could not find the colleges you want to compare in the database."
    for c in all_docs:
        c["rerank_score"] = 10.0
    
    # Limit total comparison docs to avoid context overflow
    all_docs = all_docs[:top_k * 2] 
    return all_docs, format_context_xml(all_docs)

# ─────────────── MAIN RETRIEVE ───────────────
def retrieve(query: str, top_k: int = 5, filters: dict = None,
             compare_colleges: list = None, intent: str = "search"):

    active_filters = dict(filters) if filters else {}

    # 🛠️ FIX: Extract compare_colleges from filters if the LLM nested it there
    if not compare_colleges:
        compare_colleges = active_filters.pop("compare_colleges", None)

    # 1) COMPARISON MODE
    if intent == "compare" and compare_colleges:
        logger.info(f"🆚 Comparison mode: {compare_colleges}")
        return retrieve_for_comparison(compare_colleges, top_k)

    # 2) NUMERICAL MODE (pass percentage ranking)
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

    # 3) SMART ENTITY LOOKUP (specific college)
    college_name = active_filters.pop("college_name", None)
    if college_name:
        logger.info(f"🔍 Entity lookup: {college_name}")
        # Fetch up to 15 docs to ensure we get Profile + Branches + Performance
        exact = entity_lookup(college_name, limit=15)
        if exact:
            for c in exact:
                c["rerank_score"] = 10.0
            # Return ALL fetched docs for this college to give full context
            return exact, format_context_xml(exact)
        else:
            logger.info(f"⚠️ Exact entity lookup found nothing, falling back to vector search.")

    # 4) BRANCH SQL FILTER
    branch_code = active_filters.pop("branch_code", None)
    nba_required = active_filters.pop("nba_accredited", None)
    used_sql_filter = (branch_code is not None) or (nba_required is not None)
    allowed_tnea_codes = None
    if used_sql_filter:
        allowed_tnea_codes = get_branch_codes(branch_code, nba_required)
        if not allowed_tnea_codes:
            return None, "No colleges found matching the branch/accreditation criteria."
        logger.info(f"✅ Branch SQL found {len(allowed_tnea_codes)} colleges")

    # 5) VECTOR SEARCH
    q_emb = embedding_model.encode(QUERY_PREFIX + query, normalize_embeddings=True).tolist()
    fetch_count = top_k * 4
    params = {
        "query_embedding": q_emb,
        "filter": active_filters if active_filters else {},
        "match_count": fetch_count
    }
    if allowed_tnea_codes:
        params["allowed_tnea_codes"] = allowed_tnea_codes
        
    try:
        resp = supabase.rpc("match_documents", params).execute()
        candidates = resp.data
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        candidates = []

    if not candidates:
        return None, "No documents found matching your criteria."

    # 6) RERANK
    pairs = [[query, c["content"]] for c in candidates]
    scores = reranker.predict(pairs)
    for i, sc in enumerate(scores):
        candidates[i]["rerank_score"] = float(sc)
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    # 7) CONFIDENCE GATE
    best_score = candidates[0].get("rerank_score", candidates[0].get("similarity", 0))
    if best_score < -5.0:
        if used_sql_filter:
            logger.info(f"⚠️ Low rerank score ({best_score:.2f}) but bypassing (SQL filter used)")
        else:
            logger.warning(f"Low confidence ({best_score:.2f}). Aborting.")
            return None, "I don't have enough specific information in my database to answer this accurately."

    top_docs = candidates[:top_k]
    top_docs = reorder_for_llm(top_docs)
    return top_docs, format_context_xml(top_docs)