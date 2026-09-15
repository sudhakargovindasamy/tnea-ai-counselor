"""
08_master_ingest.py
The Master Ingestion Script (Production Ready).
Uses all-MiniLM-L6-v2 (384-dim) for Render Free-Tier compatibility.
Enriches branch metadata with college names and districts for accurate filtering.
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
print("⚠️  WARNING: This will wipe 'documents' and 'admission_documents'.")
print("="*60)
confirm = input("Type 'YES' to wipe tables and re-upload: ")

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
# 2. LOAD EMBEDDING MODEL (384-dim for 512MB RAM limit)
# ==========================================
logger.info("🧠 Loading all-MiniLM-L6-v2 (384-dim)...")
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_and_upload(table_name, texts, metadatas, batch_size=50):
    logger.info(f"🧠 Generating embeddings for {len(texts)} records...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    
    records = [{"content": t, "metadata": m, "embedding": e.tolist()} 
               for t, m, e in zip(texts, metadatas, embeddings)]
        
    logger.info(f"📤 Uploading {len(records)} records to '{table_name}'...")
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        try:
            supabase.table(table_name).insert(batch).execute()
            logger.info(f"   Uploaded {min(i+batch_size, len(records))}/{len(records)}")
        except Exception as e:
            logger.error(f"Failed to upload batch {i}: {e}")

# Helper to find files whether they are in root or data/raw/
def get_path(filename):
    return f"data/raw/{filename}" if os.path.exists(f"data/raw/{filename}") else filename

# ==========================================
# 3. INGEST COLLEGES
# ==========================================
logger.info("\n📊 Processing Colleges...")
df_c = pd.read_csv(get_path("colleges_db_df.csv")).fillna("")

c_texts, c_metas = [], []
for _, r in df_c.iterrows():
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
    c_metas.append({
        "doc_type": "college_info", 
        "source": "colleges_db_df.csv", 
        "tnea_code": str(r["tnea_code"]), 
        "college_name": str(r["college_name"]), 
        "district": str(r["district"]).strip().title() # Normalize district casing
    })

embed_and_upload("documents", c_texts, c_metas)

# ==========================================
# 4. INGEST BRANCHES (Enriched with College Metadata)
# ==========================================
logger.info("\n📊 Processing Branches...")
df_b = pd.read_csv(get_path("branches_db_df.csv")).fillna("")

# Create a lookup dictionary from colleges to enrich branch metadata
college_lookup = {
    str(row["tnea_code"]): {
        "college_name": str(row["college_name"]),
        "district": str(row["district"]).strip().title()
    }
    for _, row in df_c.iterrows()
}

b_texts, b_metas = [], []
for _, r in df_b.iterrows():
    tnea_code = str(r['tnea_code'])
    college_info = college_lookup.get(tnea_code, {"college_name": "Unknown", "district": "Unknown"})
    
    text = f"College: {college_info['college_name']} (TNEA: {tnea_code})\n"
    text += f"Branch Code: {r['branch_code']}\n"
    text += f"Approved Intake: {r['approved_intake']} seats\nYear of Starting: {r['year_of_starting']}\n"
    text += f"NBA Accredited: {r['nba_accredited']}\n"
    if r.get('accreditation_valid_upto'): text += f"Valid Upto: {r['accreditation_valid_upto']}\n"
    if r.get('approval_note'): text += f"Note: {r['approval_note']}\n"
        
    b_texts.append(text)
    b_metas.append({
        "doc_type": "branch_info", 
        "source": "branches_db_df.csv", 
        "tnea_code": tnea_code, 
        "branch_code": str(r["branch_code"]),
        "college_name": college_info["college_name"],
        "district": college_info["district"],
        "nba_accredited": str(r["nba_accredited"]).strip().lower() == "yes" # Boolean for strict filtering
    })

embed_and_upload("documents", b_texts, b_metas)

# ==========================================
# 5. INGEST ADMISSION INFO
# ==========================================
logger.info("\n📊 Processing Admission Info...")
try:
    with open(get_path("tnea_admission_info.json"), "r", encoding="utf-8") as f:
        adm_data = json.load(f)
except FileNotFoundError:
    logger.warning("⚠️ tnea_admission_info.json not found. Skipping admission rules ingestion.")
    adm_data = []

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
    for k, v in entry.get("content", {}).items(): 
        lines.extend(flatten(k, v))
    
    a_texts.append("\n".join(lines))
    a_metas.append({
        "doc_type": "admission_info", 
        "source": "tnea_admission_info.json", 
        "section": entry.get("section", "General")
    })

if a_texts:
    embed_and_upload("admission_documents", a_texts, a_metas)

logger.info("\n🎉 MASTER INGESTION COMPLETE!")
logger.info("✅ 384-dim vectors uploaded. System is ready for Render deployment.")