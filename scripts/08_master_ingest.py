"""
08_master_ingest.py
Master Ingestion Script for TNEA Counselor RAG System.
Uses sentence-transformers/all-MiniLM-L6-v2 (384-dim) for fast embeddings and low memory footprint.
Ingests consolidated, denormalized college documents and admission documents into Supabase.
"""

import json
import logging
import os
import sys
import gc

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

if not supabase_url or not supabase_key:
    logger.error("❌ SUPABASE_URL and SUPABASE_KEY must be set in environment.")
    sys.exit(1)

supabase = create_client(supabase_url, supabase_key)
logger.info("✅ Connected to Supabase")

# ═══════════════════════════════════════════════════════════
# 1. LOAD EMBEDDING MODEL
# ═══════════════════════════════════════════════════════════
EXPECTED_DIMENSIONS = 384
logger.info(f"🧠 Loading sentence-transformers/all-MiniLM-L6-v2 ({EXPECTED_DIMENSIONS}-dim)...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_and_upload(table_name: str, texts: list, metadatas: list, batch_size: int = 50):
    logger.info(f"🧠 Generating embeddings for {len(texts)} records...")
    # normalize_embeddings=True is required for cosine similarity via dot product
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    
    # 🚨 Pre-flight dimension check
    if len(embeddings[0]) != EXPECTED_DIMENSIONS:
        logger.error(f"❌ Dimension mismatch! Model output {len(embeddings[0])} but expected {EXPECTED_DIMENSIONS}.")
        logger.error("   Ensure your Supabase table uses VECTOR(384).")
        sys.exit(1)

    records = [
        {"content": t, "metadata": m, "embedding": e.tolist()}
        for t, m, e in zip(texts, metadatas, embeddings)
    ]
    
    # 🧹 Free up memory after encoding
    del embeddings
    gc.collect()

    logger.info(f"📤 Uploading {len(records)} records to '{table_name}' in batches of {batch_size}...")
    success_count = 0
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        try:
            supabase.table(table_name).insert(batch).execute()
            success_count += len(batch)
            logger.info(f"   Uploaded {success_count}/{len(records)}")
        except Exception as e:
            logger.error(f"❌ Failed to upload batch {i}-{i+batch_size}: {e}")
            logger.error(f"   First record metadata: {batch[0]['metadata']}")

def wipe_table(table_name: str):
    """Safely wipes a table, handling both BIGINT and UUID primary keys."""
    logger.info(f"🗑️  Wiping '{table_name}' table...")
    try:
        # Try BIGINT wipe first (most common for documents)
        supabase.table(table_name).delete().neq("id", 0).execute()
    except Exception:
        try:
            # Fallback to UUID wipe (common for query_cache)
            supabase.table(table_name).delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
        except Exception as e:
            logger.warning(f"⚠️ Could not wipe {table_name} (it might be empty or have strict RLS): {e}")

# ═══════════════════════════════════════════════════════════
# 2. SAFETY WIPE CONFIRMATION
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("⚠️  WARNING: This will wipe and re-ingest the following tables:")
print("   - documents")
print("   - admission_documents")
print("   - query_cache (Clears old cached answers)")
print("=" * 60)
confirm = input("Type 'YES' to wipe tables and re-upload: ")

if confirm.strip().upper() != "YES":
    logger.info("Aborted by user.")
    sys.exit(0)

wipe_table("documents")
wipe_table("admission_documents")
wipe_table("query_cache") # 🚨 CRITICAL: Clear semantic cache so old answers aren't served

# ═══════════════════════════════════════════════════════════
# 3. LOAD PROCESSED DENORMALIZED DATA
# ═══════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
college_json = os.path.join(BASE_DIR, "data", "processed", "college_documents.json")
adm_json = os.path.join(BASE_DIR, "data", "processed", "admission_documents.json")

# Ensure processed files exist; if not, run preprocessing
if not os.path.exists(college_json):
    logger.info("⚙️ Processed data not found. Running scripts/preprocess_data.py...")
    import subprocess
    # 🚨 Use sys.executable to ensure it runs in the active virtual environment
    subprocess.run([sys.executable, os.path.join(BASE_DIR, "scripts", "preprocess_data.py")], check=True)

with open(college_json, "r", encoding="utf-8") as f:
    colleges_data = json.load(f)

c_texts = [d["content"] for d in colleges_data]
c_metas = [d["metadata"] for d in colleges_data]

logger.info(f"\n📊 Ingesting {len(c_texts)} Denormalized College Documents...")
embed_and_upload("documents", c_texts, c_metas, batch_size=40)

# Ingest Admission Rules
if os.path.exists(adm_json):
    with open(adm_json, "r", encoding="utf-8") as f:
        adm_data = json.load(f)

    a_texts = [d["content"] for d in adm_data]
    a_metas = [d["metadata"] for d in adm_data]

    logger.info(f"\n📊 Ingesting {len(a_texts)} Admission Documents...")
    embed_and_upload("admission_documents", a_texts, a_metas, batch_size=20)
else:
    logger.warning("⚠️ admission_documents.json not found. Skipping admission rules ingestion.")

logger.info("\n🎉 MASTER INGESTION COMPLETE!")
logger.info(f"✅ Successfully uploaded {len(c_texts)} college profiles and admission documents.")