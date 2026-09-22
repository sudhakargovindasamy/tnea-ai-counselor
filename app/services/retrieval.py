import logging
import difflib
import os
import re
import gc
import json
from typing import List, Dict, Any, Optional, Tuple
from app.services.database import supabase

logger = logging.getLogger(__name__)

# ═══════════════ 🧠 STRICT LAZY LOADING & MEMORY GUARDS ═══════════════
embedding_model = None
reranker = None

def _load_models():
    """No-op: In-memory lightweight retrieval requires zero PyTorch models (512MB RAM safe)."""
    pass

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
    "textile": "TX", "textile technology": "TX", "fashion": "FT", "fashion technology": "FT",
    "marine": "MR", "marine engineering": "MR", "marine engg": "MR", "marine engineering courses": "MR", "marine course": "MR", "mr": "MR",
    "aerospace": "AO", "aerospace engineering": "AO", "ao": "AO",
    "robotics": "RM", "robotics and automation": "RM", "robotics engineering": "RM", "rm": "RM",
    "petroleum": "PE", "petroleum engineering": "PE", "pe": "PE",
    "pharmaceutical": "PH", "pharmaceutical technology": "PH", "ph": "PH",
    "food technology": "FD", "food tech": "FD", "fd": "FD",
    "mining": "MI", "mining engineering": "MI", "mi": "MI",
    "industrial": "IE", "industrial engineering": "IE", "ie": "IE",
    "manufacturing": "MN", "manufacturing engineering": "MN", "mn": "MN",
    "environmental": "EN", "environmental engineering": "EN", "en": "EN",
    "metallurgical": "MT", "metallurgical engineering": "MT", "mt": "MT",
    "safety and fire": "SF", "safety engineering": "SF", "safety and fire engineering": "SF", "sf": "SF",
    "ceramic": "CR", "ceramic technology": "CR", "cr": "CR",
    "leather": "LE", "leather technology": "LE", "le": "LE",
    "printing": "PT", "printing technology": "PT", "pt": "PT",
    "architecture": "AR", "ar": "AR",
    "apparel": "AP", "apparel technology": "AP", "ap": "AP",
    "medical electronics": "MD", "medical electronics engineering": "MD", "md": "MD",
    "instrumentation and control": "IC", "ice": "IC", "ic": "IC",
    "electronics and instrumentation": "EI", "eie": "EI", "ei": "EI"
}

CANONICAL_BRANCH_NAMES = {
    "CS": "Computer Science and Engineering (CSE)",
    "EC": "Electronics and Communication Engineering (ECE)",
    "ME": "Mechanical Engineering",
    "EE": "Electrical and Electronics Engineering (EEE)",
    "IT": "Information Technology (IT)",
    "CE": "Civil Engineering",
    "AD": "Artificial Intelligence and Data Science (AI & DS)",
    "AL": "Artificial Intelligence and Machine Learning (AI & ML)",
    "CB": "Computer Science and Business Systems (CSBS)",
    "CY": "Cyber Security",
    "BT": "Biotechnology",
    "BM": "Biomedical Engineering",
    "AG": "Agricultural Engineering",
    "AU": "Automobile Engineering",
    "CH": "Chemical Engineering",
    "AE": "Aeronautical Engineering",
    "MR": "Marine Engineering",
    "AO": "Aerospace Engineering",
    "RM": "Robotics and Automation",
    "MZ": "Mechatronics Engineering",
    "PE": "Petroleum Engineering",
    "PH": "Pharmaceutical Technology",
    "FD": "Food Technology",
    "TX": "Textile Technology",
    "FT": "Fashion Technology",
    "MI": "Mining Engineering",
    "IE": "Industrial Engineering",
    "MN": "Manufacturing Engineering",
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
_COLLEGE_INDEX: List[Dict[str, Any]] = []
_LOCAL_DOCUMENTS: Optional[List[Dict[str, Any]]] = None
_LOCAL_ADMISSION_DOCUMENTS: Optional[List[Dict[str, Any]]] = None

def _get_local_documents() -> List[Dict[str, Any]]:
    """
    Loads full local documents for fallback.
    If local JSON is missing (e.g. cloud container without assets), fetches from Supabase.
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
            logger.warning(f"Could not load local documents from disk: {e}")

    # Cloud container fallback: fetch all 418 colleges from Supabase documents table
    try:
        logger.info("🌐 Fetching all college documents from Supabase...")
        res = supabase.table("documents").select("id, content, metadata").eq("metadata->>doc_type", "college_info").execute()
        if res.data:
            _LOCAL_DOCUMENTS = res.data
            logger.info(f"✅ Loaded {len(_LOCAL_DOCUMENTS)} colleges from Supabase.")
            return _LOCAL_DOCUMENTS
    except Exception as e:
        logger.warning(f"Could not load colleges from Supabase: {e}")

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
            logger.warning(f"Could not load local admission documents from disk: {e}")

    # Cloud container fallback: fetch admission documents from Supabase
    try:
        logger.info("🌐 Fetching admission documents from Supabase...")
        res = supabase.table("documents").select("id, content, metadata").eq("metadata->>doc_type", "admission_info").execute()
        if res.data:
            _LOCAL_ADMISSION_DOCUMENTS = res.data
            logger.info(f"✅ Loaded {len(_LOCAL_ADMISSION_DOCUMENTS)} admission documents from Supabase.")
            return _LOCAL_ADMISSION_DOCUMENTS
    except Exception as e:
        logger.warning(f"Could not load admission documents from Supabase: {e}")

    _LOCAL_ADMISSION_DOCUMENTS = []
    return _LOCAL_ADMISSION_DOCUMENTS

COLLEGE_PREFIXES = {"sri", "shri", "dr", "smt", "prof", "st", "saint", "the"}
COLLEGE_STOPWORDS = {
    "engineering", "enginering", "college", "colleges", "clg", "clgs",
    "technology", "technolgoy", "tech", "engg", "eng", "institute", "institutes",
    "institution", "institutions", "autonomous", "deemed", "university", "campus",
    "of", "and", "for", "in", "at", "naac", "grade"
}

def _extract_college_clean_words(text: str) -> List[str]:
    clean = re.sub(r"[^\w\s]", " ", text.lower())
    return [w for w in clean.split() if w not in COLLEGE_PREFIXES and w not in COLLEGE_STOPWORDS and len(w) > 2]

def _load_college_caches():
    global _COLLEGE_MAP_CACHE, _COLLEGE_DIR_CACHE, _COLLEGE_DIR_CLEAN_CACHE, _COLLEGE_INDEX
    if _COLLEGE_INDEX:
        return

    local_docs = _get_local_documents()
    for doc in local_docs:
        meta = doc.get("metadata", {})
        tnea_code = str(meta.get("tnea_code", "")).strip()
        name = meta.get("college_name", "")
        if tnea_code and name:
            _COLLEGE_MAP_CACHE[tnea_code] = meta
            name_lower = name.lower()
            _COLLEGE_DIR_CACHE[name_lower] = name

            core = name.split("(")[0].split(",")[0].strip()
            words = _extract_college_clean_words(core)
            fused = "".join(words)

            _COLLEGE_INDEX.append({
                "code": tnea_code,
                "name": name,
                "core": core,
                "core_lower": core.lower(),
                "words": set(words),
                "words_list": words,
                "fused": fused,
                "doc": doc
            })

            clean_name = " ".join(words)
            if clean_name and clean_name not in TAMIL_NADU_DISTRICTS:
                _COLLEGE_DIR_CLEAN_CACHE[name_lower] = clean_name

    logger.info(f"✅ Cached {len(_COLLEGE_INDEX)} colleges into universal memory index.")

# ═══════════════ 🧠 UNIVERSAL MULTI-STAGE RESOLVER ═══════════════
def resolve_college_entity(query_or_name: str) -> Optional[Dict[str, Any]]:
    """
    Universal multi-stage college entity resolver.
    Handles typos, spacing variations, compound words ('sairam' vs 'sai ram'),
    and missing prefixes across all 418+ TNEA institutions without manual aliases.
    """
    if not query_or_name or not str(query_or_name).strip():
        return None
    _load_college_caches()
    if not _COLLEGE_INDEX:
        return None

    raw_input = query_or_name.strip()
    lower_input = raw_input.lower()

    # 1. Check direct 4-digit TNEA Code
    code_match = re.search(r"\b(\d{4})\b", raw_input)
    if code_match:
        found_code = code_match.group(1)
        for item in _COLLEGE_INDEX:
            if item["code"] == found_code:
                logger.info(f"🧠 Matched TNEA code: {found_code} -> {item['name']}")
                return item["doc"]

    # 2. Check Alias dictionary
    for alias, full_name in COLLEGE_ALIASES.items():
        pattern = rf"\b{re.escape(alias)}\b"
        if re.search(pattern, lower_input):
            # If query specifies a district, disambiguate across multi-campus institutions
            user_district = None
            for d_name in TAMIL_NADU_DISTRICTS:
                if re.search(rf"\b{re.escape(d_name)}\b", lower_input):
                    user_district = d_name
                    break

            if user_district:
                for item in _COLLEGE_INDEX:
                    if (alias in item["fused"] or full_name.lower() in item["name"].lower()) and user_district in item["name"].lower():
                        logger.info(f"🧠 Alias + District match: '{alias}' in '{user_district}' -> '{item['name']}'")
                        return item["doc"]

            logger.info(f"🧠 Alias match: '{alias}' -> '{full_name}'")
            for item in _COLLEGE_INDEX:
                if full_name.lower() in item["name"].lower() or item["name"].lower() in full_name.lower():
                    return item["doc"]

    # If the input is clearly a search / listing intent sentence, skip entity resolution
    is_search_intent = any(
        re.search(rf"\b{re.escape(w)}\b", lower_input)
        for w in [
            "what", "which", "how", "list", "top", "best", "colleges", "clgs",
            "have", "has", "offer", "offers", "offering", "available",
            "courses", "course", "branch", "branches", "department", "departments",
            "any", "show", "find", "where"
        ]
    )
    if is_search_intent:
        return None

    q_words = _extract_college_clean_words(lower_input)
    q_fused = "".join(q_words)

    candidates = []
    for item in _COLLEGE_INDEX:
        dist_bonus = 10.0 if any(w in item["name"].lower() for w in q_words if w in TAMIL_NADU_DISTRICTS) else 0.0

        # A. Exact core title in query (e.g. 'sri sai ram engineering college' in query)
        if len(item["core_lower"]) >= 6 and item["core_lower"] in lower_input:
            candidates.append((len(item["core_lower"]), 25.0 + dist_bonus, item))
            continue

        # B. Exact fused brand match (e.g. 'sairam' matching 'sairam' in fused query or word)
        if item["fused"] and len(item["fused"]) >= 4:
            pattern = rf"\b{re.escape(item['fused'])}\b"
            if re.search(pattern, lower_input) or any(item["fused"] == w for w in q_words) or item["fused"] == q_fused:
                candidates.append((len(item["fused"]), 20.0 + dist_bonus, item))
                continue

        # C. All distinctive words of college present in query (e.g. 'sai' and 'ram' both in query)
        if len(item["words"]) >= 2 and item["words"].issubset(set(q_words)):
            candidates.append((sum(len(w) for w in item["words"]), 15.0 + dist_bonus, item))
            continue

        # D. Single distinctive word (>= 5 characters, e.g. 'saranathan', 'kumaraguru', 'mepco', 'velammal')
        distinctive = [w for w in item["words"] if len(w) >= 5]
        if distinctive and all(w in q_words for w in distinctive):
            candidates.append((sum(len(w) for w in distinctive), 10.0 + dist_bonus, item))
            continue

    if candidates:
        candidates.sort(key=lambda x: (x[1], x[0]), reverse=True)
        best_doc = candidates[0][2]["doc"]
        logger.info(f"🧠 Universal entity match: '{query_or_name}' -> '{candidates[0][2]['name']}' (Score: {candidates[0][1]})")
        return best_doc

    # E. Fuzzy fallback for minor typos on fused brand names (threshold 0.80)
    best_fuzzy = None
    best_fuzzy_score = 0.0
    for item in _COLLEGE_INDEX:
        if len(item["fused"]) >= 5:
            for qw in q_words:
                if len(qw) >= 4:
                    s = difflib.SequenceMatcher(None, qw, item["fused"]).ratio()
                    if s > best_fuzzy_score:
                        best_fuzzy_score = s
                        best_fuzzy = item

    if best_fuzzy_score >= 0.80 and best_fuzzy:
        logger.info(f"🧠 Fuzzy entity match: '{query_or_name}' -> '{best_fuzzy['name']}' (Score: {best_fuzzy_score:.2f})")
        return best_fuzzy["doc"]

    return None

def fuzzy_resolve_college(user_input: str) -> Optional[str]:
    doc = resolve_college_entity(user_input)
    if doc:
        return doc.get("metadata", {}).get("college_name")
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

_BRANCH_SYNONYMS_ENRICHED = False

def _enrich_branch_synonyms():
    global _BRANCH_SYNONYMS_ENRICHED, BRANCH_SYNONYMS
    if _BRANCH_SYNONYMS_ENRICHED:
        return
    local_docs = _get_local_documents()
    for d in local_docs:
        meta = d.get("metadata", {})
        codes = meta.get("department_codes", meta.get("branch_codes", []))
        names = meta.get("department_names", [])
        for c, n in zip(codes, names):
            if c and n:
                clean_n = n.lower().split("(ss)")[0].split("(tamil")[0].strip()
                if clean_n and clean_n not in BRANCH_SYNONYMS and len(clean_n) > 3:
                    BRANCH_SYNONYMS[clean_n] = c
    _BRANCH_SYNONYMS_ENRICHED = True

def extract_branch_code(query: str, filters: dict = None) -> Optional[str]:
    if filters and filters.get("branch_code"):
        code = str(filters["branch_code"]).strip().upper()
        if code:
            return code
    if filters and filters.get("department_code"):
        code = str(filters["department_code"]).strip().upper()
        if code:
            return code

    _enrich_branch_synonyms()
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
        elif k in ("tnea_code", "branch_code", "department_code"):
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

def compute_course_aggregation(branch_code: str, district: Optional[str] = None) -> Dict[str, Any]:
    """
    Computes exact database-wide statistics (total offering colleges and sum of approved seats)
    across the entire dataset, preventing partial 6-college hallucinations.
    """
    local_docs = _get_local_documents()
    b_code = branch_code.strip().upper()
    matching_colleges = []
    total_seats = 0
    district_counts = {}

    for d in local_docs:
        meta = d.get("metadata", {})
        c_district = meta.get("district", "")
        if district and district.lower() != c_district.lower():
            continue
        c_branches = meta.get("department_codes", meta.get("branch_codes", []))
        if b_code in c_branches:
            intakes = meta.get("department_intakes", {})
            intake = 0
            if isinstance(intakes, dict):
                intake = int(intakes.get(b_code, 0) or 0)
            elif isinstance(intakes, list) and b_code in c_branches:
                idx = c_branches.index(b_code)
                if idx < len(intakes):
                    intake = int(intakes[idx] or 0)

            total_seats += intake
            district_counts[c_district] = district_counts.get(c_district, 0) + 1
            matching_colleges.append({
                "tnea_code": meta.get("tnea_code"),
                "college_name": meta.get("college_name"),
                "district": c_district,
                "intake": intake,
                "autonomous": meta.get("autonomous", False),
                "placement_rate": meta.get("placement_rate", 0.0),
                "pass_percentage": meta.get("pass_percentage", 0.0),
                "doc": d
            })

    TIER_1_CODES = ["1", "4", "2", "2006", "1315", "5008", "2005", "2007", "1219", "2712", "2718", "2702", "1113", "1211", "2711"]
    matching_colleges.sort(
        key=lambda x: (
            (1000.0 - TIER_1_CODES.index(str(x.get("tnea_code", ""))) * 40.0) if str(x.get("tnea_code", "")) in TIER_1_CODES else 0.0,
            50.0 if x.get("autonomous") else 0.0,
            float(x.get("pass_percentage") or 0.0) + float(x.get("placement_rate") or 0.0),
            x.get("intake", 0)
        ),
        reverse=True
    )

    canonical_name = CANONICAL_BRANCH_NAMES.get(b_code, f"{b_code} Engineering")

    return {
        "branch_code": b_code,
        "branch_name": canonical_name,
        "district": district,
        "total_colleges": len(matching_colleges),
        "total_seats": total_seats,
        "district_breakdown": district_counts,
        "colleges": matching_colleges
    }

def format_aggregation_response(agg: Dict[str, Any]) -> str:
    b_name = agg["branch_name"]
    code = agg["branch_code"]
    total_colleges = agg["total_colleges"]
    total_seats = agg["total_seats"]
    district = agg.get("district")
    
    dist_part = f" in **{district.title()} District**" if district else " across Tamil Nadu"
    
    lines = [
        f"According to the official TNEA database, there are a total of **{total_seats:,} approved seats** across **{total_colleges} colleges** offering **{b_name} ({code})**{dist_part}.\n",
        "**Key Summary:**",
        f"- **Program / Course:** {b_name} ({code})",
        f"- **Total Participating Institutions:** {total_colleges} Colleges",
        f"- **Total Approved Intake:** {total_seats:,} Seats\n"
    ]
    
    if agg.get("district_breakdown") and not district:
        top_districts = sorted(agg["district_breakdown"].items(), key=lambda x: x[1], reverse=True)[:5]
        if top_districts:
            lines.append("**Top Districts by Number of Institutions:**")
            for d_name, d_count in top_districts:
                lines.append(f"- **{d_name}:** {d_count} colleges")
            lines.append("")

    if agg.get("colleges"):
        lines.append("**Top Premier Institutions with Approved Intake:**")
        for i, c in enumerate(agg["colleges"][:5]):
            raw_c_name = c["college_name"]
            parts = [p.strip() for p in raw_c_name.split(",") if p.strip()]
            if "University Departments of Anna University" in parts[0] and len(parts) > 1:
                disp_name = f"{parts[0]} - {parts[1]}"
            else:
                disp_name = parts[0]
            clean_name = re.sub(r"\s*\(Autonomous\)", "", disp_name, flags=re.IGNORECASE).strip()
            auto_str = "Autonomous" if c.get("autonomous") else "Affiliated"
            lines.append(f"{i+1}. **{clean_name}** (TNEA Code: {c['tnea_code']}) — {c['district']} • {auto_str} • **{c['intake']} Seats**")
            
    return "\n".join(lines)

def format_complete_college_list_response(docs: List[Dict], branch_code: str = None, district: str = None) -> str:
    b_name = CANONICAL_BRANCH_NAMES.get(branch_code, branch_code) if branch_code else "Engineering"
    total_colleges = len(docs)
    
    # Calculate total approved seats
    total_seats = 0
    for d in docs:
        meta = d.get("metadata", d)
        intakes = meta.get("department_intakes", {})
        if branch_code and isinstance(intakes, dict) and branch_code in intakes:
            try:
                total_seats += int(intakes[branch_code])
            except (ValueError, TypeError):
                pass
        elif "intake" in meta and meta["intake"] is not None:
            try:
                total_seats += int(meta["intake"])
            except (ValueError, TypeError):
                pass

    seats_text = f", with a total approved intake of **{total_seats:,} seats**" if total_seats > 0 else ""
    dist_text = f" in **{district.title()} District**" if district else ""

    lines = [
        f"According to the official TNEA database, there are **{total_colleges} colleges** offering {b_name}{dist_text}{seats_text}. Here is the complete list of colleges:\n"
    ]

    for i, d in enumerate(docs):
        meta = d.get("metadata", d)
        code = meta.get("tnea_code", "N/A")
        raw_name = meta.get("college_name", "Unknown College")
        parts = [p.strip() for p in raw_name.split(",") if p.strip()]
        if "University Departments of Anna University" in parts[0] and len(parts) > 1:
            name = f"{parts[0]} - {parts[1]}"
        else:
            name = parts[0]
            
        clean_name = re.sub(r"\s*\(Autonomous\)", "", name, flags=re.IGNORECASE).strip()
        c_dist = meta.get("district", "Tamil Nadu")
        is_auto = meta.get("autonomous", False)
        status_label = "Autonomous" if is_auto else "Affiliated"
        
        intakes = meta.get("department_intakes", {})
        seats_val = None
        if branch_code and isinstance(intakes, dict) and branch_code in intakes:
            seats_val = intakes[branch_code]
        elif "intake" in meta:
            seats_val = meta["intake"]
            
        seats_part = f", Seats: {seats_val}" if seats_val is not None else ""
        lines.append(f"{i+1}. TNEA Code: {code} - {clean_name} ({status_label}), {c_dist}{seats_part}")

    return "\n".join(lines)

def reorder_for_llm(docs: List[Dict]) -> List[Dict]:
    """Mitigates 'Lost in the Middle' phenomenon for LLMs."""
    if len(docs) < 3:
        return docs
    return [docs[0]] + docs[2:] + [docs[1]]

# ─────────────── QUERY REWRITER ───────────────
def rewrite_query_with_history(current_question: str, chat_history: list) -> str:
    """Pass-through: Query rewriting is handled by query_understanding.rewrite_query in main.py."""
    return current_question

# ─────────────── SMART ENTITY LOOKUP ───────────────
def entity_lookup(college_name: str, limit: int = 10) -> List[Dict]:
    if not college_name or not str(college_name).strip():
        return []

    # 1. Primary: Universal multi-stage resolver
    resolved_doc = resolve_college_entity(college_name)
    if resolved_doc:
        return [resolved_doc]
        
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

    return []

# ─────────────── 🚀 DIRECT CATALOG FILTERING ───────────────
def get_colleges_by_filters(district: str = None, branch_code: str = None, 
                            has_hostel: bool = False, autonomous: bool = None, limit: Optional[int] = 10,
                            department_code: str = None) -> List[Dict]:
    local_docs = _get_local_documents()
    candidates = []

    norm_district = DISTRICT_SYNONYMS.get(district.lower().strip(), district.title()) if district else None
    target_code = (department_code or branch_code or "").strip().upper() or None

    for d in local_docs:
        meta = d.get("metadata", {})
        c_district = meta.get("district", "")
        c_branches = meta.get("department_codes", meta.get("branch_codes", []))
        
        if norm_district and norm_district.lower() != c_district.lower():
            continue
        if target_code and target_code not in c_branches:
            continue
        if autonomous is not None:
            c_auto = meta.get("autonomous")
            if c_auto is not None and bool(c_auto) != bool(autonomous):
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
        # Sort by institutional tier, autonomy, and placement/pass percentage
        TIER_1_CODES = ["1", "4", "2", "2006", "1315", "5008", "2005", "2007", "1219", "2712", "2718", "2702", "1113", "1211", "2711"]
        def rank_sort_key(doc: Dict) -> float:
            m = doc.get("metadata", {})
            c_code = str(m.get("tnea_code", ""))
            tier_bonus = (1000.0 - TIER_1_CODES.index(c_code) * 40.0) if c_code in TIER_1_CODES else 0.0
            auto_bonus = 50.0 if m.get("autonomous") else 0.0
            perf = float(m.get("pass_percentage") or 0.0) + float(m.get("placement_rate") or 0.0)
            return tier_bonus + auto_bonus + perf

        candidates.sort(key=rank_sort_key, reverse=True)
        logger.info(f"🎯 Filter match: Found {len(candidates)} colleges (District: {norm_district}, Branch/Dept: {target_code}, Autonomous: {autonomous}, Hostel: {has_hostel})")
        deduped = deduplicate_docs(candidates)
        if limit is not None and limit > 0:
            return deduped[:limit]
        return deduped

    try:
        q = supabase.table("documents").select("id, content, metadata").eq("metadata->>doc_type", "college_info")
        if norm_district:
            q = q.ilike("metadata->>district", norm_district)
        if autonomous is not None:
            q = q.eq("metadata->>autonomous", str(autonomous).lower())
        fetch_limit = (limit * 2) if (limit is not None and limit > 0) else 500
        res = q.limit(fetch_limit).execute()
        if res.data:
            filtered = res.data
            if target_code:
                filtered = [
                    doc for doc in filtered
                    if target_code in doc.get("metadata", {}).get("department_codes", doc.get("metadata", {}).get("branch_codes", []))
                ]
            deduped = deduplicate_docs(filtered)
            if limit is not None and limit > 0:
                return deduped[:limit]
            return deduped
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

    active_filters = dict(filters) if filters else {}
    query_lower = query.lower()

    # 0. COMPARISON QUERIES
    resolved_compare_names = list(compare_colleges) if compare_colleges else []
    if not resolved_compare_names and ("compare " in query_lower or " vs " in query_lower or " versus " in query_lower):
        parts = re.split(r"\b(?:compare|and|vs|versus|with)\b", query_lower)
        candidates = [p.strip() for p in parts if len(p.strip()) >= 3]
        for cand in candidates:
            m = resolve_college_entity(cand)
            if m and cand not in resolved_compare_names:
                resolved_compare_names.append(cand)

    if resolved_compare_names:
        compare_docs = []
        for c_name in resolved_compare_names:
            matched_doc = resolve_college_entity(c_name)
            if matched_doc and matched_doc not in compare_docs:
                compare_docs.append(matched_doc)
        if len(compare_docs) >= 2:
            compare_docs = deduplicate_docs(compare_docs)
            for c in compare_docs:
                c["rerank_score"] = 10.0
            logger.info(f"✅ Found {len(compare_docs)} colleges for comparison: {resolved_compare_names}")
            return compare_docs[:top_k], format_context_xml(compare_docs[:top_k])

    # 1. CUTOFF / CLOSING RANK QUERIES
    if "cutoff" in query_lower or "closing rank" in query_lower:
        cutoff_docs, cutoff_msg = handle_cutoff_query(query, active_filters, top_k)
        return cutoff_docs, cutoff_msg

    # Extract district and branch code
    district = extract_district(query, active_filters)
    branch_code = extract_branch_code(query, active_filters)

    if district and "district" not in active_filters:
        active_filters["district"] = district
    if branch_code:
        if "branch_code" not in active_filters:
            active_filters["branch_code"] = branch_code
        if "department_code" not in active_filters:
            active_filters["department_code"] = branch_code

    # Extract autonomous intent
    is_autonomous = "autonomous" in query_lower or active_filters.get("autonomous") is True or str(active_filters.get("autonomous", "")).lower() in ("yes", "true", "1")
    autonomous_filter = True if is_autonomous else (False if active_filters.get("autonomous") in [False, "No", "no"] else None)

    # 2. SPECIFIC COLLEGE ENTITY LOOKUP (Universal Multi-Stage Resolver)
    detected_college = active_filters.pop("college_name", None)
    entity_doc = None
    if detected_college:
        entity_doc = resolve_college_entity(detected_college)
    if not entity_doc and not compare_colleges:
        entity_doc = resolve_college_entity(query)

    if entity_doc:
        doc_copy = dict(entity_doc)
        doc_copy["rerank_score"] = 10.0
        return [doc_copy], format_context_xml([doc_copy])

    # 3. AGGREGATIONS & DIRECT CATALOG FILTERING
    is_aggregation_query = (
        active_filters.get("is_aggregation") is True or
        (intent == "numerical" and active_filters.get("numerical_metric") in ["total_intake", "fees", "hostel_rent"]) or
        bool(re.search(r'\b(sum|total\s+(?:number\s+of\s+)?seats|total\s+intake|how\s+many\s+seats|sum\s+of\s+seats|count\s+(?:of\s+)?colleges|how\s+many\s+colleges)\b', query_lower))
    )

    if is_aggregation_query and branch_code:
        agg = compute_course_aggregation(branch_code=branch_code, district=district)
        agg_text = format_aggregation_response(agg)
        preview_docs = [c["doc"] for c in agg["colleges"][:10]]
        for d in preview_docs:
            d["rerank_score"] = 10.0
        logger.info(f"📊 Aggregation handled: {agg['branch_name']} -> {agg['total_colleges']} colleges, {agg['total_seats']} seats")
        return preview_docs, agg_text

    # Check for Explicit Top-K request (e.g. "top 3 colleges", "top 5", "best 3", "give me 3 colleges")
    explicit_k = active_filters.get("explicit_top_k")
    if explicit_k is None:
        top_match = (
            re.search(r'\b(?:top|best|first)\s+(\d+)\b', query_lower) or
            re.search(r'\b(\d+)\s+(?:colleges?|clgs?)\b', query_lower) or
            re.search(r'\bgive\s+me\s+(\d+)\b', query_lower)
        )
        if top_match:
            explicit_k = int(top_match.group(1))

    is_hostel_query = "hostel" in query_lower or "mess" in query_lower or "room rent" in query_lower
    is_college_list_query = (bool(district) or bool(branch_code) or is_autonomous or is_hostel_query) and (
        "colleges" in query_lower or 
        "college" in query_lower or 
        "clgs" in query_lower or 
        "clg" in query_lower or 
        "engineering colleges" in query_lower or 
        "what engineering" in query_lower or 
        "which college" in query_lower or 
        "what college" in query_lower or 
        "offer" in query_lower or 
        "list" in query_lower or 
        "have" in query_lower or
        "has" in query_lower or
        bool(branch_code) or
        is_autonomous or
        is_hostel_query
    )

    if is_college_list_query or bool(branch_code):
        if explicit_k is not None:
            # User specifically asked for "top N" or "best N" (e.g. top 3, top 5)
            target_limit = max(1, explicit_k)
            catalog_docs = get_colleges_by_filters(
                district=district,
                branch_code=branch_code,
                has_hostel=is_hostel_query,
                autonomous=autonomous_filter,
                limit=target_limit
            )
            if catalog_docs:
                catalog_docs = deduplicate_docs(catalog_docs)
                for c in catalog_docs:
                    c["rerank_score"] = 9.5
                return catalog_docs, format_context_xml(catalog_docs)
        else:
            # Broad listing query (e.g. "List the colleges that offer CSE course", "Which colleges have Marine Engineering")
            # Fetch ALL matching colleges without artificial top_k truncation!
            catalog_docs = get_colleges_by_filters(
                district=district,
                branch_code=branch_code,
                has_hostel=is_hostel_query,
                autonomous=autonomous_filter,
                limit=None
            )
            if catalog_docs:
                catalog_docs = deduplicate_docs(catalog_docs)
                for c in catalog_docs:
                    c["rerank_score"] = 9.0
                
                # If small list (<= 15 colleges, e.g. Marine, Mining, Petroleum), format full cards
                if len(catalog_docs) <= 15:
                    return catalog_docs, format_context_xml(catalog_docs)
                else:
                    # For complete in-chat listings (e.g. 408 CSE colleges), format full college list
                    return catalog_docs, format_complete_college_list_response(catalog_docs, branch_code=branch_code, district=district)

    # 4. Check if the user specifically asked for an explicit NON-EXISTENT college entity
    is_search_or_list_intent = any(
        re.search(rf"\b{re.escape(w)}\b", query_lower)
        for w in [
            "what", "which", "how", "list", "top", "best", "colleges", "clgs",
            "have", "has", "offer", "offers", "offering", "available",
            "courses", "course", "branch", "branches", "department", "departments",
            "any", "show", "find", "where", "lowest", "highest", "minimum", "maximum"
        ]
    )

    explicit_target = detected_college
    if not explicit_target and not is_search_or_list_intent and not branch_code and not district:
        if any(w in query_lower for w in ["college", "institute", "campus"]):
            tokens = [w for w in re.sub(r"[^\w\s]", " ", query_lower).split() if w not in {"tell", "me", "about", "details", "of", "the", "in", "is", "for", "engineering", "college", "colleges", "technology", "institute"} and len(w) > 2]
            if tokens:
                explicit_target = query.strip()

    if explicit_target and not is_search_or_list_intent:
        return None, f"The college '{explicit_target}' does not exist in the official TNEA database. Please verify the college name or check if it participates in TNEA counselling."

    # 5. ROUTING: ADMISSION RULES
    is_admission_query = bool(re.search(
        r"\b(reservation|counselling|eligibility|native|certificate|tnea rule|first graduate|community|oc|bc|mbc|sc|st|7\.5%|quota)\b",
        query_lower
    ))
    if is_admission_query:
        adm_docs = _get_local_admission_documents()
        words = [w for w in re.sub(r"[^\w\s]", " ", query_lower).split() if len(w) > 2]
        scored_adm = []
        for d in adm_docs:
            content_low = d.get("content", "").lower()
            score = sum(content_low.count(w) for w in words)
            if score > 0:
                scored_adm.append((score, d))
        if scored_adm:
            scored_adm.sort(key=lambda x: x[0], reverse=True)
            top_adm = [doc for _, doc in scored_adm[:top_k]]
            return top_adm, format_context_xml(top_adm)

    # 6. BROAD IN-MEMORY TEXT RANKING (0 MB PyTorch overhead, runs in 1ms)
    local_docs = _get_local_documents()
    filtered_local = []
    for d in local_docs:
        meta = d.get("metadata", {})
        if district and district.lower() != meta.get("district", "").lower():
            continue
        c_branches = meta.get("department_codes", meta.get("branch_codes", []))
        if branch_code and branch_code not in c_branches:
            continue
        if autonomous_filter is not None:
            c_auto = meta.get("autonomous")
            if c_auto is not None and bool(c_auto) != bool(autonomous_filter):
                continue
        filtered_local.append(d)

    candidates = filtered_local if filtered_local else local_docs

    # Rank by search query keyword overlap in content & title
    search_words = [w for w in re.sub(r"[^\w\s]", " ", query_lower).split() if len(w) > 2]
    if search_words:
        scored_candidates = []
        for c in candidates:
            text = (c.get("content", "") + " " + c.get("metadata", {}).get("college_name", "")).lower()
            score = sum(text.count(w) for w in search_words)
            perf_bonus = (float(c.get("metadata", {}).get("pass_percentage") or 0.0) + float(c.get("metadata", {}).get("placement_rate") or 0.0)) / 100.0
            scored_candidates.append((score + perf_bonus, c))
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        candidates = [doc for _, doc in scored_candidates]

    top_docs = deduplicate_docs(candidates)[:top_k]
    if not top_docs:
        return None, "The provided TNEA database does not contain information to answer this."

    top_docs = reorder_for_llm(top_docs)
    return top_docs, format_context_xml(top_docs)