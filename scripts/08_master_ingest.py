"""
08_master_ingest.py
The Master Ingestion Script.
Fixes the "Collapsed Content" issue and properly ingests Colleges, Branches, and Admission Info.
"""
import os
import json
import pandas as pd
import logging
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
logger.info("✅ Connected to Supabase")

# ==========================================
# 1. SAFETY WIPE (Clear old bad data)
# ==========================================
print("\n" + "="*60)
print("⚠️  WARNING: We must wipe the tables to fix the collapsed 128-char data.")
print("="*60)
confirm = input("Type 'YES' to wipe 'documents' and 'admission_documents' and re-upload: ")

if confirm.strip().upper() != "YES":
    logger.info("Aborted by user.")
    exit(0)

logger.info("🗑️  Wiping 'documents' table...")
supabase.table("documents").delete().neq("id", 0).execute()
logger.info("🗑️  Wiping 'admission_documents' table...")
try:
    supabase.table("admission_documents").delete().neq("id", 0).execute()
except Exception:
    pass

# ==========================================
# 2. LOAD EMBEDDING MODEL
# ==========================================
logger.info("🧠 Loading BAAI/bge-large-en-v1.5...")
model = SentenceTransformer("BAAI/bge-large-en-v1.5")

def embed_and_upload(table_name, texts, metadatas, batch_size=50):
    logger.info(f"🧠 Generating embeddings for {len(texts)} records...")
    # Show progress bar so you know it's working
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    
    records = [{"content": t, "metadata": m, "embedding": e.tolist()} 
               for t, m, e in zip(texts, metadatas, embeddings)]
        
    logger.info(f"📤 Uploading {len(records)} records to '{table_name}'...")
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        supabase.table(table_name).insert(batch).execute()
        logger.info(f"   Uploaded {min(i+batch_size, len(records))}/{len(records)}")

# Helper to find files whether they are in root or data/raw/
def get_path(filename):
    return f"data/raw/{filename}" if os.path.exists(f"data/raw/{filename}") else filename

# ==========================================
# 3. INGEST COLLEGES (Fixes the 128-char issue!)
# ==========================================
logger.info("\n📊 Processing Colleges...")
df_c = pd.read_csv(get_path("colleges_db_df.csv")).fillna("")

c_texts, c_metas = [], []
for _, r in df_c.iterrows():
    # Create RICH text so the LLM can actually see the fees and hostels!
    text = f"College Name: {r['college_name']}\nTNEA Code: {r['tnea_code']}\nDistrict: {r['district']}\n"
    text += f"Address: {r['address']}, {r['taluk']}, {r['district']} - {r['pincode']}\n"
    text += f"Autonomous: {r['autonomous_status']} | Placement: {r['placement']}%\n"
    text += f"\nHostel Facilities:\n"
    text += f"  Boys: {r['hostel_facilities_boys']} (Accommodation: {r['accommodation_ug_boys']}, Type: {r['permanent_or_rental_boys']})\n"
    text += f"  Girls: {r['hostel_facilities_girls']} (Accommodation: {r['accommodation_ug_girls']}, Type: {r['permanent_or_rental_girls']})\n"
    text += f"  Mess Bill (Boys): {r['mess_bill_boys']}, (Girls): {r['mess_bill_girls']}\n"
    text += f"  Room Rent (Boys): {r['room_rent_boys']}, (Girls): {r['room_rent_girls']}\n"
    text += f"  Transport: {r['transport_facilities']} (Min: {r['min_transport_charges']}, Max: {r['max_transport_charges']})\n"
    
    c_texts.append(text)
    c_metas.append({"doc_type": "college_info", "source": "colleges_db_df.csv", "tnea_code": str(r["tnea_code"]), "college_name": str(r["college_name"]), "district": str(r["district"])})

embed_and_upload("documents", c_texts, c_metas)

# ==========================================
# 4. INGEST BRANCHES
# ==========================================
logger.info("\n📊 Processing Branches...")
df_b = pd.read_csv(get_path("branches_db_df.csv")).fillna("")

b_texts, b_metas = [], []
for _, r in df_b.iterrows():
    text = f"TNEA Code: {r['tnea_code']}\nBranch Code: {r['branch_code']}\n"
    text += f"Approved Intake: {r['approved_intake']} seats\nYear of Starting: {r['year_of_starting']}\n"
    text += f"NBA Accredited: {r['nba_accredited']}\n"
    if r.get('accreditation_valid_upto'): text += f"Valid Upto: {r['accreditation_valid_upto']}\n"
    if r.get('approval_note'): text += f"Note: {r['approval_note']}\n"
        
    b_texts.append(text)
    b_metas.append({"doc_type": "branch_info", "source": "branches_db_df.csv", "tnea_code": str(r["tnea_code"]), "branch_code": str(r["branch_code"])})

embed_and_upload("documents", b_texts, b_metas)

# ==========================================
# 5. INGEST ADMISSION INFO
# ==========================================
logger.info("\n📊 Processing Admission Info...")
with open(get_path("tnea_admission_info.json"), "r", encoding="utf-8") as f:
    adm_data = json.load(f)

def flatten(k, v, indent=0):
    lines, prefix = [], "  " * indent
    if isinstance(v, dict):
        lines.append(f"{prefix}{k}:")
        for sk, sv in v.items(): lines.extend(flatten(sk, sv, indent + 1))
    elif isinstance(v, list):
        lines.append(f"{prefix}{k}: {', '.join(str(i) for i in v)}")
    else:
        lines.append(f"{prefix}{k}: {v}")
    return lines

a_texts, a_metas = [], []
for entry in adm_data:
    lines = [f"Topic: {entry.get('section', 'General')}"]
    for k, v in entry.get("content", {}).items(): lines.extend(flatten(k, v))
    
    a_texts.append("\n".join(lines))
    a_metas.append({"doc_type": "admission_info", "source": "tnea_admission_info.json", "section": entry.get("section", "General")})

embed_and_upload("admission_documents", a_texts, a_metas)

logger.info("\n🎉 MASTER INGESTION COMPLETE!")