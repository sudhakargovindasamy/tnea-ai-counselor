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
branches_path = os.path.join(RAW_DIR, "branches_db_df.csv") if os.path.exists(os.path.join(RAW_DIR, "branches_db_df.csv")) else "branches_db_df.csv"

print("📂 Loading CSVs...")
colleges = pd.read_csv(colleges_path).fillna("")
branches = pd.read_csv(branches_path).fillna("")

# 3. Branch Code Mapping Dictionary
BRANCH_MAP = {
    'CS': 'Computer Science and Engineering', 'EC': 'Electronics and Communication Engineering',
    'ME': 'Mechanical Engineering', 'EE': 'Electrical and Electronics Engineering',
    'CE': 'Civil Engineering', 'IT': 'Information Technology',
    'AD': 'Artificial Intelligence and Data Science', 'AI': 'Artificial Intelligence and Machine Learning',
    'CB': 'Computer Science and Business Systems', 'CY': 'Cyber Security'
}

branches['Branch_Full_Name'] = branches['branch_code'].map(BRANCH_MAP).fillna(branches['branch_code'])

# 4. MERGE: Attach College District and Info to every Branch row
print("🔗 Merging relational data...")
merged_branches = pd.merge(
    branches, 
    colleges[['tnea_code', 'district', 'college_name']], 
    on='tnea_code', 
    how='left'
)

# 5. Create Rich Text Chunks for Embedding
def create_branch_document(row):
    return (
        f"College Name: {row.get('college_name', 'Unknown')}\n"
        f"TNEA Code: {row['tnea_code']}\n"
        f"District: {row['district']}\n"
        f"Branch Offered: {row['Branch_Full_Name']} (Code: {row['branch_code']})\n"
        f"Approved Intake: {row.get('approved_intake', 'N/A')} seats\n"
        f"NBA Accredited: {row.get('nba_accredited', 'N/A')}\n"
    )

merged_branches['content'] = merged_branches.apply(create_branch_document, axis=1)

# 6. Wipe old branch data and Upload new enriched data
print("🗑️ Wiping old branch_info from Supabase...")
supabase.table("documents").delete().eq("metadata->>doc_type", "branch_info").execute()

print("🚀 Uploading enriched relational documents...")
records = []
for _, row in tqdm(merged_branches.iterrows(), total=len(merged_branches)):
    content = row['content']
    embedding = model.encode(content, normalize_embeddings=True).tolist()
    
    metadata = {
        "doc_type": "branch_info",
        "college_name": row.get('college_name', 'Unknown'),
        "tnea_code": str(row['tnea_code']),
        "district": str(row['district']),
        "branch_code": str(row['branch_code']),
        "branch_full_name": row['Branch_Full_Name'],
        "nba_accredited": str(row.get('nba_accredited', 'N/A')).lower() in ['yes', 'true', '1']
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