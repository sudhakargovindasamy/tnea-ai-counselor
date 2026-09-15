import logging
import difflib
import os
import re
from sentence_transformers import SentenceTransformer, CrossEncoder
from app.services.database import supabase

# Optional Gemini integration for Query Rewriting (Fixes the "there" memory leak)
try:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

logger = logging.getLogger(__name__)

logger.info("Loading embedding model...")
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
logger.info("Loading reranker...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

QUERY_PREFIX = ""

# ═══════════════ 🧠 ALIAS DICTIONARY (EXACT CSV MATCHES) ═══════════════
COLLEGE_ALIASES = {
    "ceg": "University Departments of Anna University , Chennai - CEG Campus",
    "mit": "University Departments of Anna University , Chennai - MIT Campus",
    "act": "University Departments of Anna University , Chennai - ACT Campus",
    "psg": "PSG College of Technology",
    "ssn": "SSN College of Engineering",
    "svce": "Sri Venkateswara College of Engineering",
    "sairam": "Sri Sai Ram Enginering College",  
    "thiagarajar": "Thiagarajar College of Engineering",
    "kct": "Kumaraguru College of Technology",
    "skcet": "Sri Krishna College of Enginering and Technology", 
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

# ═══════════════ 🧠 IN-MEMORY FUZZY RESOLVER ═══════════════
COLLEGE_DIRECTORY = {}

def load_college_directory():
    global COLLEGE_DIRECTORY
    if COLLEGE_DIRECTORY:
        return 
        
    try:
        res = supabase.table("documents").select("metadata->>college_name").execute()
        seen = set()
        for row in res.data:
            name = row.get("college_name")
            if name and name not in seen:
                seen.add(name)
                COLLEGE_DIRECTORY[name.lower()] = name
        logger.info(f"✅ Loaded {len(COLLEGE_DIRECTORY)} colleges into memory.")
    except Exception as e:
        logger.error(f"Failed to load college directory: {e}")

def fuzzy_resolve_college(user_input: str) -> str:
    if not user_input:
        return None
    if not COLLEGE_DIRECTORY:
        load_college_directory()
    if not COLLEGE_DIRECTORY:
        return None

    expanded_input = user_input.lower()
    for alias, full_name in COLLEGE_ALIASES.items():
        if f" {alias} " in f" {expanded_input} " or expanded_input == alias:
            expanded_input = expanded_input.replace(alias, full_name)
            logger.info(f"🧠 Alias expanded '{alias}' -> '{full_name}'")
            break

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
        
        score1 = difflib.SequenceMatcher(None, clean_input, clean_db).ratio()
        input_nospace = clean_input.replace(" ", "")
        db_nospace = clean_db.replace(" ", "")
        score2 = difflib.SequenceMatcher(None, input_nospace, db_nospace).ratio()
        
        if len(input_nospace) >= 4 and (input_nospace in db_nospace or db_nospace in input_nospace):
            score2 = 1.0 
            
        score = max(score1, score2)
        if score > highest_score:
            highest_score = score
            best_match = db_name_exact
            
    if highest_score > 0.75:
        logger.info(f"🧠 Fuzzy matched '{user_input}' to '{best_match}' (Score: {highest_score:.2f})")
        return best_match
    return None

# ─────────────── 🛠️ FIX 2: FILTER NORMALIZER ───────────────
def _normalize_filters(filters: dict) -> dict:
    """Normalize filter keys/values to match what is actually stored in Supabase metadata."""
    if not filters:
        return {}
    clean = {}
    for k, v in filters.items():
        if k == "district" and isinstance(v, str):
            v = v.strip().title()  # "COIMBATORE" -> "Coimbatore"
        elif k == "nba_accredited":
            if isinstance(v, bool):
                v = "Yes" if v else "No"
            elif isinstance(v, str):
                v = v.strip().title()
        elif k in ("tnea_code", "branch_code"):
            v = str(v).strip()
        clean[k] = v
    return clean

# ─────────────── XML FORMATTING ───────────────
def format_context_xml(docs):
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

def reorder_for_llm(docs):
    if len(docs) < 3:
        return docs
    return [docs[0]] + docs[2:] + [docs[1]]

# ─────────────── QUERY REWRITER ───────────────
def rewrite_query_with_history(current_question: str, chat_history: list) -> str:
    if not chat_history or not GEMINI_AVAILABLE:
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
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        rewritten = response.text.strip().strip('"')
        logger.info(f"🧠 Query Rewritten: '{current_question}' -> '{rewritten}'")
        return rewritten
    except Exception as e:
        logger.warning(f"⚠️ Query rewriting failed: {e}")
        return current_question

# ─────────────── SMART ENTITY LOOKUP ───────────────
def entity_lookup(college_name: str, limit: int = 15):
    if not college_name or not str(college_name).strip():
        return []
        
    search_terms = college_name.lower()
    stopwords = ["engineering", "college", "technology", "institute", "of", "and", "autonomous", "the", "for"]
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

def get_branches_by_filters(district: str = None, branch_code: str = None, nba_required: str = None, limit: int = 10):
    try:
        q = supabase.table("documents").select("id, content, metadata").eq("metadata->>doc_type", "branch_info")
        if branch_code:
            q = q.ilike("metadata->>branch_code", f"%{branch_code}%")
        if nba_required is not None:
            nba_val = "Yes" if isinstance(nba_required, bool) and nba_required else "No"
            q = q.eq("metadata->>nba_accredited", nba_val)
            
        res = q.execute()
        if not res.data:
            return []

        if district:
            dist_res = supabase.table("documents").select("metadata").eq("metadata->>doc_type", "college_info").ilike("metadata->>district", f"%{district}%").execute()
            district_codes = set(str(r["metadata"].get("tnea_code")) for r in dist_res.data if r["metadata"].get("tnea_code"))
            filtered_docs = [doc for doc in res.data if str(doc["metadata"].get("tnea_code")) in district_codes]
        else:
            filtered_docs = res.data

        if not filtered_docs:
            return []

        all_colleges_res = supabase.table("documents").select("metadata").eq("metadata->>doc_type", "college_info").execute()
        college_map = {str(r["metadata"].get("tnea_code")): r["metadata"] for r in all_colleges_res.data if r["metadata"].get("tnea_code")}
        
        for doc in filtered_docs:
            tc = str(doc["metadata"].get("tnea_code"))
            if tc in college_map:
                info = college_map[tc]
                doc["metadata"]["college_name"] = info.get("college_name", "Unknown College")
                doc["metadata"]["district"] = info.get("district", "Unknown District")
                doc["content"] = f"college_name: {info.get('college_name')}\n" + doc.get("content", "")

        return filtered_docs[:limit]

    except Exception as e:
        logger.error(f"Direct SQL branch fetch failed: {e}", exc_info=True)
        return []

def get_top_by_pass_percentage(top_k, district=None):
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

def retrieve_for_comparison(college_names, top_k=5):
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

# ─────────────── MAIN RETRIEVE ───────────────
def retrieve(query: str, top_k: int = 5, filters: dict = None,
             compare_colleges: list = None, intent: str = "search", chat_history: list = None):

    # 🛠️ FIX 2: Normalize filters immediately upon entry
    filters = _normalize_filters(filters)

    # 0) QUERY REWRITING 
    query = rewrite_query_with_history(query, chat_history)

    active_filters = dict(filters) if filters else {}

    # 🛠️ FIX 2: FALLBACK ENTITY EXTRACTION (Catches typos if LLM query understanding fails)
    if not active_filters.get("college_name") and not compare_colleges:
        resolved_fallback = fuzzy_resolve_college(query)
        if resolved_fallback:
            logger.info(f"🧠 Fallback fuzzy match extracted college: '{resolved_fallback}'")
            active_filters["college_name"] = resolved_fallback

    if not compare_colleges:
        compare_colleges = active_filters.pop("compare_colleges", None)

    # 1) COMPARISON MODE
    if intent == "compare" and compare_colleges:
        logger.info(f"🆚 Comparison mode: {compare_colleges}")
        return retrieve_for_comparison(compare_colleges, top_k)

    # 2) NUMERICAL MODE
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

    # 3) SMART ENTITY LOOKUP
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

    # 4) 🚀 SMART DOC_TYPE ROUTING & DIRECT SQL FETCH
    query_lower = query.lower()
    table_name = "documents"
    
    district_filter = active_filters.get("district")
    branch_code_filter = active_filters.get("branch_code")
    nba_filter = active_filters.get("nba_accredited")

    # 🛠️ FIX 1: Regex word boundaries (\b) prevent "ec" from matching "technology"
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
            # 🛠️ FIX: Slice down to exact top_k requested by user
            sql_docs = sql_docs[:top_k]
            return sql_docs, format_context_xml(sql_docs)
        # If SQL returns nothing, fall through to vector search
        
    elif re.search(r"\b(hostel|fee|placement|transport|mess|rent|principal|address|autonomous)\b", query_lower):
        active_filters["doc_type"] = "college_info"
        logger.info("🎯 Routing to: college_info")
        
    elif re.search(r"\b(reservation|counselling|eligibility|native|certificate|tnea rule|first graduate|community|oc|bc|mbc)\b", query_lower):
        table_name = "admission_documents"
        logger.info("🎯 Routing to: admission_documents")

    # 5) VECTOR SEARCH 
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

    # 6) RERANK
    pairs = [[query, c["content"]] for c in candidates]
    scores = reranker.predict(pairs)
    for i, sc in enumerate(scores):
        candidates[i]["rerank_score"] = float(sc)
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    # 7) CONFIDENCE GATE 
    best_rerank = candidates[0]["rerank_score"]
    best_vector = candidates[0].get("similarity", 0.0) 
    
    if best_rerank < -2.0 and best_vector < 0.5:
        logger.warning(f"Low confidence (Rerank: {best_rerank:.2f}, Vector: {best_vector:.2f}). Aborting.")
        return None, "I don't have enough specific information in my database to answer this accurately."

    top_docs = candidates[:top_k]
    top_docs = reorder_for_llm(top_docs)
    return top_docs, format_context_xml(top_docs)