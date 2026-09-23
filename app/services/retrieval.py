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

# ═══════════════ 🧠 ALIAS DICTIONARY (EXACT TNEA CODES) ═══════════════
COLLEGE_ALIASES = {
    # Premier Anna University Departments
    "ceg": "1",
    "anna university ceg": "1",
    "act": "2",
    "anna university act": "2",
    "sap": "3",
    "mit": "4",
    "anna university mit": "4",
    "mit campus": "4",
    
    # Top Tier-1 Colleges
    "gct": "2005",
    "government college of technology": "2005",
    "psg": "2006",
    "psg tech": "2006",
    "psg college of technology": "2006",
    "psg i tech": "2377",
    "psg itech": "2377",
    "psg institute of technology": "2377",
    "cit": "2007",
    "coimbatore institute of technology": "2007",
    "tce": "5008",
    "thiagarajar": "5008",
    "thiagarajar college of engineering": "5008",
    "svce": "1219",
    "sri venkateswara": "1219",
    "sri venkateswara college of engineering": "1219",
    "kct": "2712",
    "kumaraguru": "2712",
    "kumaraguru college of technology": "2712",
    "skcet": "2718",
    "sri krishna": "2718",
    "sri krishna college of engineering and technology": "2718",
    "sri krishna college of enginering and technology": "2718",
    "bitsathy": "2702",
    "bannari amman": "2702",
    "bannari": "2702",
    "bannari amman institute of technology": "2702",
    "bannari amman institute": "2702",
    "rmk": "1113",
    "rmk engineering college": "1113",
    "r m k": "1113",
    "rec": "1211",
    "rajalakshmi": "1211",
    "rajalakshmi engineering college": "1211",
    "kongu": "2711",
    "kongu engineering college": "2711",
    "sairam": "1419",
    "sai ram": "1419",
    "sri sairam": "1419",
    "sri sai ram": "1419",
    "sairam engineering college": "1419",
    "sairam enginering college": "1419",
    "sri sairam engineering college": "1419",
    "sri sai ram engineering college": "1419",
    "sri sai ram enginering college": "1419",
    "mepco": "4960",
    "mepco schlenk": "4960",
    "saveetha": "1216",
    "saveetha engineering college": "1216",
    "easwari": "1304",
    "easwari engineering college": "1304",
    "panimalar": "1210",
    "panimalar engineering college": "1210",
    "sona": "2618",
    "sona college of technology": "2618",
    "velammal": "1115",
    "velammal engineering college": "1115",
    "cit chennai": "1399",
    "chennai institute of technology": "1399",
    "loyola icam": "1149",
    "licet": "1149",
    "srm trp": "3795",
    "srm valliammai": "1422",
    "valliammai": "1422",
    "st joseph": "1317",
    "st josephs": "1317",
    "st. joseph's": "1317",
    "st joseph's": "1317",
    "meenakshi sundararajan": "1309",
    "ramco": "4678",
    "ramco institute of technology": "4678",
    "saranathan": "3819",
    "saranathan college of engineering": "3819"
}

# ═══════════════ 🎓 BRANCH SYNONYMS & CANONICAL MAPPINGS ═══════════════
BRANCH_SYNONYMS = {
    "computer science": "CS", "computer science courses": "CS", "computer science and engineering": "CS",
    "computer science engineering": "CS", "cse": "CS", "cs": "CS",
    "computer science engineering": "CS", "computer science eng": "CS", "computer science engg": "CS",
    "comp science": "CS", "comp sci": "CS", "cse": "CS", "cs": "CS", "cs eng": "CS", "cs engg": "CS", "cse eng": "CS", "cse engg": "CS",
    "artificial intelligence and data science": "AD", "ai & ds": "AD", "ai and ds": "AD", "ai&ds": "AD", "ai & data science": "AD", "ai and data science": "AD", "ad": "AD",
    "ai & ml": "AL", "ai and ml": "AL", "ai&ml": "AL", "artificial intelligence and machine learning": "AL", "al": "AL",
    "cse ai & ml": "AM", "cse aiml": "AM", "cse ai ml": "AM", "am": "AM",
    "cyber security": "CY", "cybersecurity": "CY", "cyber security specialization": "CY", "cy": "CY",
    
    # ❌ REMOVED "me": "ME" to prevent false positive on "tell me about..."
    "mechanical": "ME", "mechanical engineering": "ME", "mechanical engineering courses": "ME", "mech": "ME",
    "mechanical eng": "ME", "mechanical engg": "ME", "mech eng": "ME", "mech engg": "ME",
    
    "electronics and communication": "EC", "electronics and communication engineering": "EC", "ece": "EC", "ec": "EC",
    "electronics and communication eng": "EC", "electronics and communication engg": "EC", "ece eng": "EC", "ece engg": "EC",
    "electronics": "EC", "electronics engineering": "EC", "electronics eng": "EC", "electronics engg": "EC",
    "electrical and electronics": "EE", "electrical and electronics engineering": "EE", "eee": "EE", "ee": "EE",
    "civil": "CE", "civil engineering": "CE", "ce": "CE",
    "electrical and electronics eng": "EE", "electrical and electronics engg": "EE", "eee eng": "EE", "eee engg": "EE",
    "electrical": "EE", "electrical engineering": "EE", "electrical eng": "EE", "electrical engg": "EE",
    "civil": "CE", "civil engineering": "CE", "ce": "CE", "civil eng": "CE", "civil engg": "CE",
    
    # ❌ REMOVED "it": "IT" to prevent false positive on "tell me about it..."
    "information technology": "IT",
    
    "computer science and business systems": "CB", "csbs": "CB", "cb": "CB",
    "mechatronics": "MZ", "mechatronics engineering": "MZ", "mz": "MZ",
    "biotechnology": "BT", "bio technology": "BT", "biotech": "BT", "bt": "BT",
    "biomedical": "BM", "bio medical": "BM", "biomedical engineering": "BM", "bm": "BM",
    "mechatronics": "MZ", "mechatronics engineering": "MZ", "mechatronics eng": "MZ", "mechatronics engg": "MZ", "mz": "MZ",
    "biotechnology": "BT", "bio technology": "BT", "biotech": "BT", "biotechnology eng": "BT", "biotechnology engg": "BT", "bt": "BT",
    "biomedical": "BM", "bio medical": "BM", "biomedical engineering": "BM", "biomedical eng": "BM", "biomedical engg": "BM", "bm": "BM",
    "agricultural": "AG", "agriculture": "AG", "agri": "AG", "ag": "AG",
    "chemical": "CH", "chemical engineering": "CH", "ch": "CH",
    "automobile": "AU", "automobile engineering": "AU", "au": "AU",
    "aeronautical": "AE", "aeronautical engineering": "AE", "ae": "AE",
    "chemical": "CH", "chemical engineering": "CH", "chemical eng": "CH", "chemical engg": "CH", "ch": "CH",
    "automobile": "AU", "automobile engineering": "AU", "automobile eng": "AU", "automobile engg": "AU", "auto eng": "AU", "auto engg": "AU", "au": "AU",
    "aeronautical": "AE", "aeronautical engineering": "AE", "aeronautical eng": "AE", "aeronautical engg": "AE", "aero eng": "AE", "aero engg": "AE", "ae": "AE",
    "textile": "TX", "textile technology": "TX", "fashion": "FT", "fashion technology": "FT",
    "marine": "MR", "marine engineering": "MR", "marine engg": "MR", "marine engineering courses": "MR", "marine course": "MR", "mr": "MR",
    "aerospace": "AO", "aerospace engineering": "AO", "ao": "AO",
    "robotics": "RM", "robotics and automation": "RM", "robotics engineering": "RM", "rm": "RM",
    "petroleum": "PE", "petroleum engineering": "PE", "pe": "PE",
    "marine": "MR", "marine engineering": "MR", "marine engg": "MR", "marine eng": "MR", "marine engineering courses": "MR", "marine course": "MR", "mr": "MR",
    "aerospace": "AO", "aerospace engineering": "AO", "aerospace eng": "AO", "aerospace engg": "AO", "ao": "AO",
    "robotics": "RM", "robotics and automation": "RM", "robotics engineering": "RM", "robotics eng": "RM", "robotics engg": "RM", "rm": "RM",
    "petroleum": "PE", "petroleum engineering": "PE", "petroleum eng": "PE", "petroleum engg": "PE", "pe": "PE",
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
    "AM": "Computer Science and Engineering (Artificial Intelligence and Machine Learning)",
}

DISTRICT_SYNONYMS = {
    # Ariyalur
    "ariyalur": "Ariyalur",
    # Chengalpattu
    "chengalpattu": "Chengalpattu", "chengalpet": "Chengalpattu", "chengai": "Chengalpattu",
    # Chennai
    "chennai": "Chennai", "madras": "Chennai",
    # Coimbatore
    "coimbatore": "Coimbatore", "kovai": "Coimbatore", "cbe": "Coimbatore",
    # Cuddalore
    "cuddalore": "Cuddalore",
    # Dharmapuri
    "dharmapuri": "Dharmapuri",
    # Dindigul
    "dindigul": "Dindigul", "dindukkal": "Dindigul",
    # Erode
    "erode": "Erode",
    # Kallakurichi
    "kallakurichi": "Kallakurichi", "kallakuruchi": "Kallakurichi",
    # Kanchipuram
    "kanchipuram": "Kanchipuram", "kancheepuram": "Kanchipuram", "kanchi": "Kanchipuram",
    # Kanyakumari
    "kanyakumari": "Kanyakumari", "kanniyakumari": "Kanyakumari", "nagercoil": "Kanyakumari",
    # Karur
    "karur": "Karur",
    # Krishnagiri
    "krishnagiri": "Krishnagiri", "hosur": "Krishnagiri",
    # Madurai
    "madurai": "Madurai",
    # Mayiladuthurai
    "mayiladuthurai": "Mayiladuthurai", "mayavaram": "Mayiladuthurai", "mayiladuthorai": "Mayiladuthurai",
    # Nagapattinam
    "nagapattinam": "Nagapattinam", "nagappattinam": "Nagapattinam", "nagai": "Nagapattinam",
    # Namakkal
    "namakkal": "Namakkal", "rasipuram": "Namakkal", "tiruchengode": "Namakkal",
    # Perambalur
    "perambalur": "Perambalur",
    # Pudukkottai
    "pudukkottai": "Pudukkottai", "pudukottai": "Pudukkottai",
    # Ramanathapuram
    "ramanathapuram": "Ramanathapuram", "ramnad": "Ramanathapuram",
    # Ranipet
    "ranipet": "Ranipet", "melvisharam": "Ranipet",
    # Salem
    "salem": "Salem",
    # Sivagangai
    "sivagangai": "Sivagangai", "sivaganga": "Sivagangai", "karaikudi": "Sivagangai",
    # Tenkasi
    "tenkasi": "Tenkasi",
    # Thanjavur
    "thanjavur": "Thanjavur", "tanjore": "Thanjavur", "thanjai": "Thanjavur", "kumbakonam": "Thanjavur",
    # The Nilgiris
    "the nilgiris": "The Nilgiris", "nilgiris": "The Nilgiris", "ooty": "The Nilgiris", "nilgiri": "The Nilgiris", "uadhagamandalam": "The Nilgiris",
    # Theni
    "theni": "Theni",
    # Thoothukudi
    "thoothukudi": "Thoothukudi", "thoothukkudi": "Thoothukudi", "tuticorin": "Thoothukudi",
    # Tiruchirappalli
    "tiruchirappalli": "Tiruchirappalli", "trichirappalli": "Tiruchirappalli", "trichy": "Tiruchirappalli", "tiruchi": "Tiruchirappalli", "tiruchy": "Tiruchirappalli",
    # Tirunelveli
    "tirunelveli": "Tirunelveli", "thirunelveli": "Tirunelveli", "nellai": "Tirunelveli",
    # Tirupattur
    "tirupattur": "Tirupattur", "thirupattur": "Tirupattur",
    # Tiruppur
    "tiruppur": "Tiruppur", "thiruppur": "Tiruppur", "tirupur": "Tiruppur",
    # Tiruvallur
    "tiruvallur": "Tiruvallur", "thiruvallur": "Tiruvallur",
    # Tiruvannamalai
    "tiruvannamalai": "Tiruvannamalai", "thiruvannamalai": "Tiruvannamalai",
    # Tiruvarur
    "tiruvarur": "Tiruvarur", "thiruvarur": "Tiruvarur",
    # Vellore
    "vellore": "Vellore",
    # Viluppuram
    "viluppuram": "Viluppuram", "villupuram": "Viluppuram",
    # Virudhunagar
    "virudhunagar": "Virudhunagar", "sivakasi": "Virudhunagar", "rajapalayam": "Virudhunagar"
}

TAMIL_NADU_DISTRICTS = set(DISTRICT_SYNONYMS.keys()) | {v.lower() for v in DISTRICT_SYNONYMS.values()}

# ═══════════════ 🚀 GLOBAL CACHES & LOCAL FALLBACK ═══════════════
_COLLEGE_MAP_CACHE: Dict[str, Dict] = {}
_COLLEGE_DIR_CACHE: Dict[str, str] = {}
_COLLEGE_DIR_CLEAN_CACHE: Dict[str, str] = {}
_COLLEGE_INDEX: List[Dict[str, Any]] = []
_COLLEGE_BY_CODE: Dict[str, Dict] = {}
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
    "of", "and", "for", "in", "at", "naac", "grade",
    # Academic & generic terms that should never identify a college on their own
    "science", "sciences", "arts", "applied", "research", "studies", "higher",
    "management", "education", "educational", "academy", "polytechnic", "trust",
    "memorial", "center", "centre", "school", "schools", "general"
}

def _extract_college_clean_words(text: str) -> List[str]:
    clean = re.sub(r"[^\w\s]", " ", text.lower())
    return [w for w in clean.split() if w not in COLLEGE_PREFIXES and w not in COLLEGE_STOPWORDS and len(w) > 2]

def _load_college_caches():
    global _COLLEGE_MAP_CACHE, _COLLEGE_DIR_CACHE, _COLLEGE_DIR_CLEAN_CACHE, _COLLEGE_INDEX, _COLLEGE_BY_CODE
    if _COLLEGE_INDEX:
        return

    local_docs = _get_local_documents()
    for doc in local_docs:
        meta = doc.get("metadata", {})
        tnea_code = str(meta.get("tnea_code", "")).strip()
        name = meta.get("college_name", "")
        if tnea_code and name:
            _COLLEGE_MAP_CACHE[tnea_code] = meta
            _COLLEGE_BY_CODE[tnea_code] = doc
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
        if found_code in _COLLEGE_BY_CODE:
            doc = _COLLEGE_BY_CODE[found_code]
            logger.info(f"🧠 Matched TNEA code: {found_code} -> {doc.get('metadata', {}).get('college_name')}")
            return doc

    # 2. Check Alias dictionary (sorted longest first so full phrases match before short acronyms)
    for alias in sorted(COLLEGE_ALIASES.keys(), key=len, reverse=True):
        pattern = rf"\b{re.escape(alias)}\b"
        if re.search(pattern, lower_input):
            target = COLLEGE_ALIASES[alias]
            
            # If query specifies a district, disambiguate across multi-campus institutions
            user_district = None
            for d_name in TAMIL_NADU_DISTRICTS:
                if re.search(rf"\b{re.escape(d_name)}\b", lower_input):
                    user_district = d_name
                    break

            if target.isdigit():
                if user_district:
                    for item in _COLLEGE_INDEX:
                        if (alias in item["fused"] or item["code"] == target) and user_district in item["name"].lower():
                            logger.info(f"🧠 Alias + District match: '{alias}' in '{user_district}' -> '{item['name']}'")
                            return item["doc"]
                doc = _COLLEGE_BY_CODE.get(target)
                if doc:
                    logger.info(f"🧠 Alias match: '{alias}' -> Code {target} ('{doc.get('metadata', {}).get('college_name')}')")
                    return doc
            else:
                for item in _COLLEGE_INDEX:
                    if target.lower() in item["name"].lower() or item["name"].lower() in target.lower():
                        logger.info(f"🧠 Alias match: '{alias}' -> '{target}' ('{item['name']}')")
                        return item["doc"]

    q_words = _extract_college_clean_words(lower_input)
    q_fused = "".join(q_words)

    candidates = []
    for item in _COLLEGE_INDEX:
        dist_bonus = 10.0 if any(w in item["name"].lower() for w in q_words if w in TAMIL_NADU_DISTRICTS) else 0.0

        # A. Exact core title in query (e.g. 'sri sai ram engineering college' in query)
        if len(item["core_lower"]) >= 6 and item["core_lower"] in lower_input:
            candidates.append((len(item["core_lower"]), 25.0 + dist_bonus, item))
            continue

        # B. Exact fused brand match (excluding pure district names)
        fused = item["fused"]
        if fused and len(fused) >= 4 and fused not in TAMIL_NADU_DISTRICTS:
            pattern = rf"\b{re.escape(fused)}\b"
            if re.search(pattern, lower_input) or any(fused == w for w in q_words) or fused == q_fused:
                candidates.append((len(fused), 20.0 + dist_bonus, item))
                continue

        # C. All distinctive words of college present in query (>= 2 words, excluding districts)
        non_dist_words = {w for w in item["words"] if w not in TAMIL_NADU_DISTRICTS}
        if len(non_dist_words) >= 2 and non_dist_words.issubset(set(q_words)):
            candidates.append((sum(len(w) for w in non_dist_words), 15.0 + dist_bonus, item))
            continue

        # D. Single distinctive word (>= 5 characters, excluding districts)
        distinctive = [w for w in item["words"] if len(w) >= 5 and w not in TAMIL_NADU_DISTRICTS]
        if distinctive and all(w in q_words for w in distinctive):
            candidates.append((sum(len(w) for w in distinctive), 10.0 + dist_bonus, item))
            continue
        # D. Single distinctive word college (where college's core brand is exactly 1 word >= 4 chars, excluding districts & course names)
        if len(non_dist_words) == 1:
            dist_word = list(non_dist_words)[0]
            if len(dist_word) >= 4 and dist_word in q_words and dist_word not in BRANCH_SYNONYMS:
                candidates.append((len(dist_word), 10.0 + dist_bonus, item))
                continue

    if candidates:
        candidates.sort(key=lambda x: (x[1], x[0]), reverse=True)
        best_doc = candidates[0][2]["doc"]
        logger.info(f"🧠 Universal entity match: '{query_or_name}' -> '{candidates[0][2]['name']}' (Score: {candidates[0][1]})")
        return best_doc

    # E. Fuzzy fallback for minor typos on fused brand names (threshold 0.85, excluding districts)
    best_fuzzy = None
    best_fuzzy_score = 0.0
    for item in _COLLEGE_INDEX:
        if len(item["fused"]) >= 5 and item["fused"] not in TAMIL_NADU_DISTRICTS:
            for qw in q_words:
                if len(qw) >= 4 and qw not in TAMIL_NADU_DISTRICTS:
                    s = difflib.SequenceMatcher(None, qw, item["fused"]).ratio()
                    if s > best_fuzzy_score:
                        best_fuzzy_score = s
                        best_fuzzy = item

    if best_fuzzy_score >= 0.85 and best_fuzzy:
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
    for d_key, d_norm in sorted(DISTRICT_SYNONYMS.items(), key=lambda x: len(x[0]), reverse=True):
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
        # If student asks about cutoff calculation, formula, or merit marks out of 200, let it pass to admission rules
        if re.search(r'\b(how\s+is\s+(?:the\s+)?cutoff\s+calculated|cutoff\s+formula|calculate\s+cutoff|cutoff\s+calculation|merit\s+mark|marks?\s+out\s+of\s+200|how\s+is\s+merit\s+calculated)\b', q_lower):
            return None, None
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

# ═══════════════ 📜 TNEA ADMISSION RULES RETRIEVER ═══════════════
ADMISSION_TRIGGER_PATTERNS = [
    r"\b(rule\s+of\s+reservation|reservation\s+(?:percentage|percentages|rule|rules|policy|system|ratio)|seat\s+allocation|quota\s+percentage|categories?\s+of\s+reservation|percentage\s+(?:of|for)\s+(?:oc|bc|bcm|mbc|sc|sca|st)|st\s+quota\s+percentage)\b",
    r"\b(percentage\s+of\s+seats|seats?\s+(?:are\s+)?reserved\s+for|how\s+many\s+(?:seats|percentage)\s+(?:are\s+)?reserved)\b",
    r"\b(7\.5%|7\.5\s*percent|government\s+school\s+(?:reservation|quota|fee|waiver|students?))\b",
    r"\b(first\s+graduate|first\s+graduation|fg\s+concession|fg\s+fee|first\s+generation\s+graduate)\b",
    r"\b(aicte\s+(?:tuition\s+)?fee\s+waiver|aicte\s+waiver|aicte\s+scheme)\b",
    r"\b(post[\s\-]matric|sc\s+scholarship|st\s+scholarship|sca\s+scholarship)\b",
    r"\b(tuition\s+fee\s+concession|fee\s+concession|fee\s+waiver)\b",
    r"\b(registration\s+fee|application\s+fee|cost\s+of\s+application|how\s+to\s+register|registration\s+procedure|registration\s+requirement|application\s+portal|tneaonline\.org|dte\.tn\.gov\.in|how\s+to\s+apply\s+for\s+tnea|steps\s+to\s+apply|websites?\s+to\s+register|official\s+websites?)\b",
    r"\b(upload(?:ing)?\s+(?:copy\s+of\s+)?(?:original\s+)?certificates?|certificates?\s+(?:that\s+are\s+|are\s+|is\s+)?(?:required|needed|to\s+upload)|documents?\s+(?:that\s+are\s+|are\s+|is\s+)?(?:required|needed|to\s+upload)|what\s+(?:documents?|certificates?)\s+(?:are\s+|is\s+)?needed|which\s+certificates?\s+(?:are\s+|is\s+)?needed)\b",
    r"\b(merit\s+list|merit\s+mark|rank\s+calculation|cutoff\s+formula|mark\s+distribution|marks?\s+out\s+of\s+200|how\s+is\s+merit\s+calculated|tnea\s+merit|tie[\s\-]break|tie[\s\-]breaker|how\s+is\s+(?:the\s+)?cutoff\s+calculated|mathematics\s+marks\s+calculated|marks\s+are\s+allocated\s+to\s+physics)\b",
    r"\b(tentative\s+allotment|choice\s+filling|accept\s+and\s+join|accept\s+and\s+upward|decline\s+and\s+move|decline\s+and\s+quit|confirmation\s+options|counselling\s+stages|stages\s+of\s+(?:online\s+)?counselling|allotment\s+confirmation|counselling\s+rounds?|counselling\s+procedure|counselling\s+process|steps\s+in\s+counselling|fails?\s+to\s+report|procedure\s+for\s+allotment)\b",
    r"\b(special\s+reservation|ex[\s\-]servicemen|differently\s+abled|benchmark\s+disabilit|eminent\s+sports|sports\s+quota|pwd\s+quota|quota\s+for\s+differently\s+abled|sports\s+persons?)\b",
    r"\b(marine\b.*\b(eligibilit\w*|require\w*|physical\w*|medic\w*|imu|cet|height\w*|weight\w*|age|vision|eyesight|fitness|criteria|rule\w*|pcm|qualification\w*)|(eligibilit\w*|require\w*|physical\w*|medic\w*|imu|cet|height\w*|weight\w*|age|vision|eyesight|fitness|criteria|rule\w*|pcm|qualification\w*)\b.*\bmarine)\b",
    r"\b(mining\b.*\b(female|women|woman|girl|gender|mines\s+act|rule|rules|restriction|underground|shaft|permitted|allowed|prohibited|criteria|eligibility)|(female|women|woman|girl|gender|mines\s+act|rule|rules|restriction|underground|shaft|permitted|allowed|prohibited)\b.*\bmining)\b",
    r"\b(nativity\s+certificates?|nativity\s+rules?|other\s+state\s+candidates|central\s+government\s+employees|general\s+eligibility|who\s+needs\s+nativity|sri\s*lankan\s+tamil\s+refugees?)\b"
]

def is_admission_rule_query(q_low: str) -> bool:
    for pat in ADMISSION_TRIGGER_PATTERNS:
        if re.search(pat, q_low):
            return True
    return False

COMMON_STOPWORDS = {
    "the", "is", "at", "which", "on", "what", "how", "for", "and", "in", "to",
    "of", "a", "an", "does", "are", "can", "who", "where", "why", "from", "with",
    "by", "about", "any", "some", "be", "been", "being", "have", "has", "had",
    "do", "did", "doing", "would", "should", "could", "there", "their", "they",
    "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "tnea",
    "rule", "rules", "according"
}

def rank_admission_docs(query: str, top_k: int = 2) -> List[Dict]:
    q_low = query.lower()
    adm_docs = _get_local_admission_documents()
    words = [w for w in re.sub(r"[^\w\s\.\%]", " ", q_low).split() if w not in COMMON_STOPWORDS and len(w) > 1]
    
    scored = []
    for d in adm_docs:
        meta = d.get("metadata", {})
        sec = meta.get("section", "").lower()
        content = d.get("content", "").lower()
        score = 0.0

        # Exact domain boosters
        if "marine" in q_low:
            if "course: marine engineering" in content: score += 150.0
            else: score -= 50.0
        if "mining" in q_low:
            if "course: mining engineering" in content: score += 150.0
            else: score -= 50.0
            
        if any(w in q_low for w in ["differently abled", "disabilit", "ex-servicemen", "sports person", "sports quota", "special reservation"]):
            if "special reservation categories" in sec: score += 120.0
        elif "reservation" in q_low or "rule of reservation" in q_low or "allocation of seats" in q_low or "quota percentage" in q_low:
            if "allocation of seats" in sec: score += 100.0
            
        if any(w in q_low for w in ["first graduate", "aicte", "fee concession", "post matric", "post-matric", "scholarship", "7.5", "fee waiver"]):
            if any(w in q_low for w in ["upload", "documents", "certificate"]):
                if "uploading copy of original certificates" in sec: score += 110.0
                elif "tuition fee concession" in sec: score += 90.0
            else:
                if "tuition fee concession" in sec: score += 120.0
                
        if "registration fee" in q_low or "application fee" in q_low or "how to register" in q_low or "registration portal" in q_low or "application portal" in q_low or "websites to register" in q_low or "cost of application" in q_low or "registration procedure" in q_low:
            if "procedure for registration" in sec: score += 120.0
            
        if "upload" in q_low or "certificates required" in q_low or "documents required" in q_low or "original certificates" in q_low or "documents to upload" in q_low or "certificates must be uploaded" in q_low or "certificates are required" in q_low:
            if "uploading copy of original certificates" in sec: score += 120.0
            
        if "merit mark" in q_low or "merit list" in q_low or "formula" in q_low or "200" in q_low or "cutoff calculated" in q_low or "mathematics marks" in q_low or "allocated to physics" in q_low or "stages of online counselling" in q_low:
            if "online choice filling and confirmation" in sec: score += 120.0
            
        if "tentative allotment" in q_low or "confirmation options" in q_low or "accept and join" in q_low or "accept and upward" in q_low or "decline and" in q_low or "fails to report" in q_low:
            if "counselling allotment and confirmation" in sec: score += 120.0
            
        if "nativity" in q_low or "other state" in q_low or "central government" in q_low or "general eligibility" in q_low or "refugee" in q_low:
            if "general eligibility rules" in sec: score += 120.0

        # Section title match
        for w in words:
            if w in sec: score += 15.0
            score += content.count(w) * (3.0 if len(w) > 4 else 1.0)
            
        scored.append((score, d))
        
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:top_k]]

AGGREGATION_PATTERN = (
    r'\b('
    r'sum(\s+of)?|'
    r'total\s+(?:number\s+of\s+|no\s+of\s+|num\s+of\s+)?(?:seats|intake|colleges?|collages?|clgs?)|'
    r'total\s+(?:seats|intake|colleges?|collages?|clgs?)|'
    r'how\s+many\s+(?:seats|intake|colleges?|collages?|clgs?)|'
    r'(?:number|no|num|count)\s+of\s+(?:seats|intake|colleges?|collages?|clgs?)'
    r')\b'
)

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
        if cutoff_msg:
            return cutoff_docs, cutoff_msg

    # 1.5 TNEA ADMISSION RULES & COUNSELLING BROCHURE ROUTING
    # Evaluated before college lookup so policy queries mentioning "government", "act", "mines",
    # "hostel fee", etc. are not hijacked by colleges like GCE Bargur or ACT Campus.
    if is_admission_rule_query(query_lower):
        adm_docs = rank_admission_docs(query, top_k=2)
        if adm_docs:
            for d in adm_docs:
                d["rerank_score"] = 10.0
            sec_name = adm_docs[0].get("metadata", {}).get("section", "Admission Information")
            logger.info(f"📜 Admission rule query routed to '{sec_name}'")
            return adm_docs, format_context_xml(adm_docs)

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
    # 2. AGGREGATIONS & DIRECT CATALOG FILTERING CHECK
    is_aggregation_query = (
        active_filters.get("is_aggregation") is True or
        (intent == "numerical" and active_filters.get("numerical_metric") in ["total_intake", "fees", "hostel_rent"]) or
        bool(re.search(AGGREGATION_PATTERN, query_lower))
    )

    # 2.5 SPECIFIC COLLEGE ENTITY LOOKUP (Universal Multi-Stage Resolver)
    detected_college = active_filters.pop("college_name", None)
    entity_doc = None
    if detected_college:
        entity_doc = resolve_college_entity(detected_college)
    if not entity_doc and not compare_colleges and not is_aggregation_query and not (branch_code and not detected_college):
        entity_doc = resolve_college_entity(query)

    if entity_doc:
        doc_copy = dict(entity_doc)
        doc_copy["rerank_score"] = 10.0
        return [doc_copy], format_context_xml([doc_copy])

    if is_aggregation_query and branch_code:
        agg = compute_course_aggregation(branch_code=branch_code, district=district)
        agg_text = format_aggregation_response(agg)
        preview_docs = [c["doc"] for c in agg["colleges"][:10]]
        for d in preview_docs:
            d["rerank_score"] = 10.0
        logger.info(f"📊 Aggregation handled: {agg['branch_name']} -> {agg['total_colleges']} colleges, {agg['total_seats']} seats")
        return preview_docs, agg_text

    # Non-TN Out-of-Scope Detection
    NON_TN_LOCATIONS = [
        "delhi", "bangalore", "bengaluru", "hyderabad", "mumbai", "pune", "kolkata",
        "karnataka", "kerala", "andhra", "andhra pradesh", "telangana", "maharashtra",
        "noida", "gurgaon", "gurugram", "chandigarh", "jaipur", "ahmedabad", "patna",
        "bhopal", "indore", "lucknow", "kochi", "cochin", "trivandrum", "thiruvananthapuram",
        "calicut", "kozhikode", "mangalore", "mangaluru", "mysore", "mysuru"
    ]
    if any(re.search(rf"\b{re.escape(loc)}\b", query_lower) for loc in NON_TN_LOCATIONS) and not district and not detected_college:
        return None, "The current database only contains TNEA-approved engineering colleges within Tamil Nadu. I cannot provide information for colleges outside Tamil Nadu."

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
    is_placement_query = bool(re.search(r'\b(placement|placements|package|packages|salary|salaries|job|jobs|hiring|recruitment|placed)\b', query_lower))
    is_performance_query = bool(re.search(r'\b(performance|pass\s+percentage|pass\s+rate|results?|academics?)\b', query_lower))
    is_rank_query = bool(re.search(r'\b(top|best|rank|ranking|rankings|leading|premier|good)\b', query_lower))

    is_college_list_query = (
        bool(district) or 
        bool(branch_code) or 
        is_autonomous or 
        is_hostel_query or
        is_placement_query or
        is_performance_query or
        is_rank_query or
        any(w in query_lower for w in ["colleges", "college", "clgs", "clg", "engineering", "institutes", "options", "list", "which", "what", "where"])
    )

    is_listing_indicator = (
        bool(re.search(
            r'\b(list|colleges|college|clgs|clg|which|what|show|find|options|where|top|best|offer|offers|offering|have|has|placement|placements|package|packages|salary|salaries|job|jobs|hiring|performance|pass\s+rate|pass\s+percentage|results?|rank|ranking|rankings|across|region|district)\b',
            query_lower
        )) or 
        bool(district)
    )

    if is_listing_indicator and (is_college_list_query or bool(branch_code) or bool(district)):
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
            # Broad listing query or regional overview (e.g. "How is placement across erode region", "List colleges with CSE")
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
                
                # If district query, or small/medium list (<= 25 colleges, e.g. Erode 14, Salem 15, Madurai 15, Marine, Mining)
                # or analytical query (placement/performance/ranking):
                # Pass the rich XML cards to LLM so it has all placement rates, pass rates, and branch details!
                if len(catalog_docs) <= 25 or is_placement_query or is_performance_query or is_rank_query or bool(district):
                    # For very large districts like Coimbatore (60 colleges), take top 25 so LLM receives deep analysis without token limit issues
                    return catalog_docs[:25], format_context_xml(catalog_docs[:25])
                else:
                    # For pure statewide directories (e.g. 408 CSE colleges across Tamil Nadu), format full college list
                    return catalog_docs, format_complete_college_list_response(catalog_docs, branch_code=branch_code, district=district)
            else:
                # Filters were applied (e.g. district and/or branch) but 0 colleges matched!
                # Do NOT fall through to generic statewide colleges. Give a clear, factual answer.
                if district and branch_code:
                    b_name = CANONICAL_BRANCH_NAMES.get(branch_code, f"{branch_code} Engineering")
                    return None, f"According to the official TNEA database, there are currently no engineering colleges in {district} offering {b_name} ({branch_code})."
                elif district and is_autonomous:
                    return None, f"According to the official TNEA database, there are currently no autonomous engineering colleges in {district}."
                elif district:
                    return None, f"According to the official TNEA database, there are no engineering colleges in {district} matching your criteria."
                elif branch_code:
                    b_name = CANONICAL_BRANCH_NAMES.get(branch_code, f"{branch_code} Engineering")
                    return None, f"According to the official TNEA database, there are currently no engineering colleges offering {b_name} ({branch_code})."

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

    # 5. ROUTING: ADMISSION RULES (Fallback if not caught at Step 3.5)
    is_admission_query = is_admission_rule_query(query_lower) or bool(re.search(
        r"\b(reservation|counselling|eligibility|native|certificate|tnea rule|first graduate|community|oc|bc|mbc|sc|st|7\.5%|quota)\b",
        query_lower
    ))
    if is_admission_query:
        adm_docs = rank_admission_docs(query, top_k=top_k)
        if adm_docs:
            for d in adm_docs:
                d["rerank_score"] = 10.0
            return adm_docs, format_context_xml(adm_docs)

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