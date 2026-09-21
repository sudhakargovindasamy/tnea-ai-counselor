import os
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client
from tqdm import tqdm

load_dotenv()

# 1. Initialize Supabase and Embedding Model
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 2. Dynamically find the CSVs in data/raw/ or root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

colleges_path = os.path.join(RAW_DIR, "colleges_db_df.csv") if os.path.exists(os.path.join(RAW_DIR, "colleges_db_df.csv")) else "colleges_db_df.csv"
branches_path = os.path.join(RAW_DIR, "college_branches_rows.csv") if os.path.exists(os.path.join(RAW_DIR, "college_branches_rows.csv")) else os.path.join(RAW_DIR, "branches_db_df.csv")

print("📂 Loading CSVs...")
colleges = pd.read_csv(colleges_path).fillna("")
branches = pd.read_csv(branches_path).fillna("")

# 3. Standardize column names for new/legacy formats
if "department_code" not in branches.columns and "branch_code" in branches.columns:
    branches["department_code"] = branches["branch_code"]
if "department_name" not in branches.columns:
    BRANCH_MAP = {
        'CS': 'Computer Science and Engineering', 'EC': 'Electronics and Communication Engineering',
        'ME': 'Mechanical Engineering', 'EE': 'Electrical and Electronics Engineering',
        'CE': 'Civil Engineering', 'IT': 'Information Technology',
        'AD': 'Artificial Intelligence and Data Science', 'AI': 'Artificial Intelligence and Machine Learning',
        'CB': 'Computer Science and Business Systems', 'CY': 'Cyber Security'
    }
    branches['department_name'] = branches['department_code'].map(BRANCH_MAP).fillna(branches['department_code'])
if "approval_marker" not in branches.columns and "approval_note" in branches.columns:
    branches["approval_marker"] = branches["approval_note"]

# 4. MERGE: Attach College District and Info to every Branch row
print("🔗 Merging relational data...")
merged_branches = pd.merge(
    branches, 
    colleges[['tnea_code', 'district', 'college_name']], 
    on='tnea_code', 
    how='left'
)

# 5. Create Rich Text Chunks for Embedding (Phase 4 standard format)
def create_branch_document(row):
    dept_code = str(row.get("department_code", "")).strip()
    dept_name = str(row.get("department_name", "")).strip()
    t_code = str(row.get("tnea_code", "")).strip()
    college = str(row.get("college_name", "")).strip()
    district = str(row.get("district", "")).strip()
    intake = str(row.get("approved_intake", "")).strip()
    yos = str(row.get("year_of_starting", "")).strip()
    if yos.endswith(".0"):
        yos = yos[:-2]
    nba = str(row.get("nba_accredited", "")).strip()
    valid_upto = str(row.get("accreditation_valid_upto", "")).strip()
    if valid_upto.endswith(".0"):
        valid_upto = valid_upto[:-2]
    marker = str(row.get("approval_marker", "")).strip()

    parts = []
    if college:
        parts.append(f"College: {college} (District: {district}).")
    parts.append(f"TNEA Code: {t_code} offers {dept_name} ({dept_code}).")
    if intake and intake not in ("nan", "NaN"):
        parts.append(f"Approved Intake: {intake}.")
    if yos and yos not in ("nan", "NaN"):
        parts.append(f"Year of Starting: {yos}.")
    if nba and nba not in ("nan", "NaN"):
        nba_str = nba
        if valid_upto and valid_upto not in ("nan", "NaN"):
            nba_str += f" (Valid upto {valid_upto})"
        parts.append(f"NBA Accredited: {nba_str}.")
    if marker and marker not in ("nan", "NaN"):
        parts.append(f"Approval Marker: {marker}.")
    return " ".join(parts)

merged_branches['content'] = merged_branches.apply(create_branch_document, axis=1)

# 6. Wipe old branch data and Upload new enriched data
print("🗑️ Wiping old branch_info from Supabase...")
try:
    supabase.table("documents").delete().eq("metadata->>doc_type", "branch_info").execute()
except Exception as e:
    print(f"Notice during wipe: {e}")

print("🚀 Uploading enriched relational documents...")
records = []
for _, row in tqdm(merged_branches.iterrows(), total=len(merged_branches)):
    content = row['content']
    embedding = model.encode(content, normalize_embeddings=True).tolist()
    
    dept_code = str(row.get("department_code", "")).strip()
    dept_name = str(row.get("department_name", "")).strip()
    marker = str(row.get("approval_marker", "")).strip()
    yos_val = row.get("year_of_starting")
    yos_clean = int(float(yos_val)) if pd.notna(yos_val) and str(yos_val).strip() not in ("", "nan", "NaN") else None

    metadata = {
        "doc_type": "branch_info",
        "id": int(float(row["id"])) if "id" in row and pd.notna(row["id"]) and str(row["id"]).strip() not in ("", "nan", "NaN") else None,
        "college_name": str(row.get('college_name', 'Unknown')).strip(),
        "tnea_code": str(row['tnea_code']).strip(),
        "district": str(row.get('district', '')).strip(),
        "department_code": dept_code,
        "department_name": dept_name,
        "branch_code": dept_code,  # Backward compatibility
        "branch_full_name": dept_name,  # Backward compatibility
        "approved_intake": int(float(row["approved_intake"])) if pd.notna(row.get("approved_intake")) and str(row.get("approved_intake")).strip() not in ("", "nan", "NaN") else 0,
        "year_of_starting": yos_clean,
        "nba_accredited": str(row.get('nba_accredited', '')).strip() if pd.notna(row.get('nba_accredited')) and str(row.get('nba_accredited')).strip() not in ("", "nan", "NaN") else None,
        "accreditation_valid_upto": str(row.get('accreditation_valid_upto', '')).strip() if pd.notna(row.get('accreditation_valid_upto')) and str(row.get('accreditation_valid_upto')).strip() not in ("", "nan", "NaN") else None,
        "approval_marker": marker if marker and marker not in ("nan", "NaN") else None,
        "approval_note": marker if marker and marker not in ("nan", "NaN") else None,  # Backward compatibility
        "source": os.path.basename(branches_path)
    }
    
    records.append({
        "content": content,
        "embedding": embedding,
        "metadata": metadata
    })

# Batch upload to Supabase (chunks of 100)
for i in range(0, len(records), 100):
    batch = records[i:i+100]
    supabase.table("documents").insert(batch).execute()

print(f"🎉 Relational Data Enrichment Complete! Uploaded {len(records)} branch documents.")