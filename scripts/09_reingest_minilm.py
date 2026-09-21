"""
Re-ingest with all-MiniLM-L6-v2 (384 dims) - FIXED VERSION
Fix 1: global supabase declaration (kills UnboundLocalError)
Fix 2: recursive flattening of nested admission JSON dicts into text
"""
import json
import os
import time

import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

BATCH_SIZE = 20
MAX_RETRIES = 5

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

print("Loading all-MiniLM-L6-v2 model...")
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print(f"Embedding dimensions: {embedding_model.encode('test').shape}")

batch_counter = {"n": 0}

def refresh_client():
    global supabase
    batch_counter["n"] += 1
    if batch_counter["n"] % 50 == 0:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("🔄 Recreated Supabase client (fresh connection)")

def insert_with_retry(table, batch):
    global supabase                      # <-- FIX 1
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            supabase.table(table).insert(batch).execute()
            refresh_client()
            return
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            wait = 2 ** attempt
            print(f"⚠️  Insert failed (attempt {attempt}/{MAX_RETRIES}): {type(e).__name__}: {e}. Retrying in {wait}s...")
            time.sleep(wait)
            supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def flatten(obj):                        # <-- FIX 2
    """Recursively convert ANY JSON structure into plain text."""
    if isinstance(obj, str):
        return obj
    if obj is None or isinstance(obj, (int, float, bool)):
        return str(obj)
    if isinstance(obj, dict):
        return "\n".join(f"{k}: {flatten(v)}" for k, v in obj.items())
    if isinstance(obj, list):
        return "\n".join(flatten(v) for v in obj)
    return str(obj)

def count_by_doc_type(doc_type):
    res = supabase.table("documents").select("id", count="exact").eq("metadata->>doc_type", doc_type).execute()
    return res.count

def row_to_document(row, columns):
    return "\n".join(f"{c}: {row[c]}" for c in columns).strip()

def ingest_dataframe(df, source_name, doc_type, extra_meta_fn):
    total = len(df)
    print(f"\nEncoding {total} {doc_type} documents...")
    docs = [row_to_document(row, df.columns) for _, row in df.iterrows()]
    embeddings = embedding_model.encode(docs, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    records = []
    for i, (doc, emb) in enumerate(zip(docs, embeddings)):
        meta = {"source": source_name, "row_number": int(i), "doc_type": doc_type}
        meta.update(extra_meta_fn(df.iloc[i]))
        records.append({"content": doc, "metadata": meta, "embedding": emb.tolist()})
    for i in range(0, total, BATCH_SIZE):
        insert_with_retry("documents", records[i:i + BATCH_SIZE])
        print(f"Uploaded {doc_type}: {min(i + BATCH_SIZE, total)}/{total}")
        time.sleep(0.2)

# ============================================
# COLLEGES (skip if complete)
# ============================================
df_colleges = pd.read_csv("data/raw/colleges_db_df.csv").fillna("")
n_colleges = count_by_doc_type("college_info")
if n_colleges >= len(df_colleges):
    print(f"\n✅ Colleges already ingested ({n_colleges}). Skipping.")
else:
    if n_colleges > 0:
        supabase.table("documents").delete().eq("metadata->>doc_type", "college_info").execute()
    ingest_dataframe(df_colleges, "colleges_db_df.csv", "college_info",
                     lambda r: {"tnea_code": str(r["tnea_code"]),
                                "college_name": r["college_name"],
                                "district": r["district"]})

# ============================================
# BRANCHES (skip if complete)
# ============================================
branches_file = "data/raw/college_branches_rows.csv" if os.path.exists("data/raw/college_branches_rows.csv") else "data/raw/branches_db_df.csv"
df_branches = pd.read_csv(branches_file).fillna("")
n_branches = count_by_doc_type("branch_info")
if n_branches >= len(df_branches):
    print(f"\n✅ Branches already ingested ({n_branches}). Skipping.")
else:
    if n_branches > 0:
        supabase.table("documents").delete().eq("metadata->>doc_type", "branch_info").execute()
    ingest_dataframe(df_branches, os.path.basename(branches_file), "branch_info",
                     lambda r: {
                         "id": int(float(r["id"])) if "id" in r and str(r["id"]).strip() not in ("", "nan", "NaN") else None,
                         "tnea_code": str(r["tnea_code"]).strip(),
                         "sl_no": int(float(r["sl_no"])) if "sl_no" in r and str(r["sl_no"]).strip() not in ("", "nan", "NaN") else None,
                         "department_code": str(r.get("department_code", r.get("branch_code", ""))).strip().upper(),
                         "department_name": str(r.get("department_name", "")).strip(),
                         "approved_intake": int(float(r["approved_intake"])) if str(r.get("approved_intake", "")).strip() not in ("", "nan", "NaN") else 0,
                         "year_of_starting": int(float(r["year_of_starting"])) if str(r.get("year_of_starting", "")).strip() not in ("", "nan", "NaN") else None,
                         "nba_accredited": str(r.get("nba_accredited", "")).strip() if str(r.get("nba_accredited", "")).strip() not in ("", "nan", "NaN") else None,
                         "accreditation_valid_upto": str(r.get("accreditation_valid_upto", "")).strip() if str(r.get("accreditation_valid_upto", "")).strip() not in ("", "nan", "NaN") else None,
                         "approval_marker": str(r.get("approval_marker", r.get("approval_note", ""))).strip() or None,
                         # Backward compatibility
                         "branch_code": str(r.get("department_code", r.get("branch_code", ""))).strip().upper(),
                         "approval_note": str(r.get("approval_marker", r.get("approval_note", ""))).strip() or None
                     })

# ============================================
# ADMISSION RULES (flattened text)
# ============================================
admission_file = "data/raw/tnea_admission_info.json"
if os.path.exists(admission_file):
    try:
        n_adm = supabase.table("admission_documents").select("id", count="exact").execute().count
    except Exception:
        n_adm = 0
    if n_adm > 0:
        print(f"\n✅ Admission docs already ingested ({n_adm}). Skipping.")
    else:
        with open(admission_file, "r") as f:
            admission_data = json.load(f)
        # Unwrap root dict if the list is nested inside it
        if isinstance(admission_data, dict):
            admission_data = next((v for v in admission_data.values() if isinstance(v, list)), [admission_data])
        print(f"\nEncoding {len(admission_data)} admission documents...")
        docs = [flatten(item) for item in admission_data]
        embeddings = embedding_model.encode(docs, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
        records = []
        for i, (doc, emb) in enumerate(zip(docs, embeddings)):
            section = admission_data[i].get("section", f"Rule {i+1}") if isinstance(admission_data[i], dict) else f"Rule {i+1}"
            records.append({
                "content": doc,
                "metadata": {"source": "tnea_admission_info.json", "row_number": i,
                             "doc_type": "admission_info", "section": section},
                "embedding": emb.tolist()
            })
        for i in range(0, len(records), BATCH_SIZE):
            insert_with_retry("admission_documents", records[i:i + BATCH_SIZE])
            print(f"Uploaded admission: {min(i + BATCH_SIZE, len(records))}/{len(records)}")
            time.sleep(0.2)

# ============================================
# FINAL VERIFICATION
# ============================================
print("\n" + "=" * 50)
print("VERIFICATION")
print("=" * 50)
print(f"Colleges  : {count_by_doc_type('college_info')} / {len(df_colleges)}")
print(f"Branches  : {count_by_doc_type('branch_info')} / {len(df_branches)}")
try:
    print(f"Admission : {supabase.table('admission_documents').select('id', count='exact').execute().count}")
except Exception as e:
    print(f"Admission : check failed ({e})")

test_query = "Which colleges in Coimbatore offer Computer Science?"
test_emb = embedding_model.encode(test_query, normalize_embeddings=True).tolist()
results = supabase.rpc("match_documents", {"query_embedding": test_emb, "match_count": 3}).execute()
print(f"\nTest query: '{test_query}' -> {len(results.data)} results")
for r in results.data:
    print(f"  - Similarity: {r['similarity']:.4f} | {r['content'][:80]}...")

print("\n✅ RE-INGESTION COMPLETE!")