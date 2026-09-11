import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import pandas as pd
from sentence_transformers import SentenceTransformer
from app.services.database import supabase

# ─────────── Retry Helper ───────────
def insert_with_retry(table, records, max_retries=4):
    for attempt in range(max_retries):
        try:
            supabase.table(table).insert(records).execute()
            return True
        except Exception as e:
            wait = 2 ** attempt  # 1s, 2s, 4s, 8s backoff
            print(f"    ⚠️  Attempt {attempt+1} failed: {type(e).__name__}. Retrying in {wait}s...")
            time.sleep(wait)
    return False

# ─────────── Load Model ───────────
print("Loading model...")
embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# ─────────── Load Data ───────────
print("Loading performance...")
df = pd.read_csv("data/raw/performance_db_df.csv")
df["tnea_code"] = df["tnea_code"].astype(str)
df = df.drop_duplicates(subset=["tnea_code"], keep="first")
df = df.fillna("")
print(f"📊 {len(df)} performance records")

# ─────────── IDEMPOTENT CLEANUP (safe re-run) ───────────
print("🧹 Clearing existing performance data (idempotent)...")
try:
    supabase.table("performance").delete().neq("id", 0).execute()
    print("   ✅ performance table cleared")
except Exception as e:
    print(f"   ⚠️  Could not clear performance: {e}")

try:
    supabase.table("documents").delete().eq("metadata->>doc_type", "performance").execute()
    print("   ✅ old performance documents cleared")
except Exception as e:
    print(f"   ⚠️  Could not clear perf docs: {e}")

# ─────────── Part A: SQL table for numerical sorting ───────────
sql_records = []
for _, row in df.iterrows():
    sql_records.append({
        "tnea_code": str(row["tnea_code"]),
        "college_name": str(row["college_name"]),
        "district": str(row["district"]),
        "total_appeared": int(row["total_appeared"]) if row["total_appeared"] != "" else 0,
        "total_passed": int(row["total_passed"]) if row["total_passed"] != "" else 0,
        "pass_percentage": float(row["pass_percentage"]) if row["pass_percentage"] != "" else 0.0
    })

print("Uploading performance SQL table...")
BATCH = 100
for i in range(0, len(sql_records), BATCH):
    ok = insert_with_retry("performance", sql_records[i:i+BATCH])
    if ok:
        print(f"  ✅ {min(i+BATCH, len(sql_records))}/{len(sql_records)}")
    else:
        print(f"  ❌ FAILED batch at {i} after retries")
    time.sleep(0.3)

# ─────────── Part B: Embed summaries into documents ───────────
def build_perf_text(row):
    text = f"Academic Performance Report\n"
    text += f"College Name: {row['college_name']}\n"
    text += f"TNEA Code: {row['tnea_code']}\n"
    text += f"District: {row['district']}\n"
    text += f"Overall Pass Percentage: {row['pass_percentage']}%\n"
    for sem in [1, 3, 5, 7, 9]:
        app, pas = f"sem{sem}_appeared", f"sem{sem}_passed"
        if app in row and pas in row and row[app] != "" and float(row[app]) > 0:
            text += f"Semester {sem}: {int(float(row[pas]))} passed out of {int(float(row[app]))} appeared\n"
    return text.strip()

df["perf_text"] = df.apply(build_perf_text, axis=1)

print("Generating performance embeddings...")
embeddings = embedding_model.encode(df["perf_text"].tolist(), show_progress_bar=True, normalize_embeddings=True)

doc_records = []
for i, row in df.iterrows():
    doc_records.append({
        "content": row["perf_text"],
        "metadata": {
            "source": "performance_db_df.csv",
            "tnea_code": str(row["tnea_code"]),
            "college_name": str(row["college_name"]),
            "district": str(row["district"]),
            "doc_type": "performance"
        },
        "embedding": embeddings[i].tolist()
    })

print("Uploading performance documents (batch=5, with retry)...")
BATCH = 5  # smaller batch = more reliable over flaky connections
failed_batches = []
for i in range(0, len(doc_records), BATCH):
    ok = insert_with_retry("documents", doc_records[i:i+BATCH])
    if ok:
        print(f"  ✅ {min(i+BATCH, len(doc_records))}/{len(doc_records)}")
    else:
        print(f"  ❌ FAILED batch at index {i}")
        failed_batches.append(i)
    time.sleep(0.5)  # gentle pacing to avoid connection drops

# ─────────── Final Summary ───────────
print("\n" + "="*50)
if failed_batches:
    print(f"⚠️  {len(failed_batches)} batch(es) failed. Re-run the script to retry them (it's idempotent).")
else:
    print("🎉 Performance ingested successfully!")

# Verify counts
try:
    perf_count = supabase.table("performance").select("id", count="exact").execute().count
    print(f"📊 performance table rows: {perf_count}")
except Exception as e:
    print(f"Could not verify: {e}")