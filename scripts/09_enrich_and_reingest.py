import pandas as pd
import os
from supabase import create_client
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

load_dotenv()

# 1. Initialize Supabase and Embedding Model
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 2. Load your CSVs
print("📂 Loading CSVs...")
colleges = pd.read_csv("colleges_db_df.csv")
branches = pd.read_csv("branches_db_df.csv")

# 3. Branch Code Mapping Dictionary
BRANCH_MAP = {
    'CS': 'Computer Science and Engineering', 'EC': 'Electronics and Communication Engineering',
    'ME': 'Mechanical Engineering', 'EE': 'Electrical and Electronics Engineering',
    'CE': 'Civil Engineering', 'IT': 'Information Technology',
    'AD': 'Artificial Intelligence and Data Science', 'AI': 'Artificial Intelligence and Machine Learning',
    'CB': 'Computer Science and Business Systems', 'CY': 'Cyber Security'
}

# Map codes to full names
branches['Branch_Full_Name'] = branches['Branch Code'].map(BRANCH_MAP).fillna(branches['Branch Code'])

# 4. MERGE: Attach College District and Info to every Branch row
print("🔗 Merging relational data...")
# Assuming both CSVs share a 'TNEA Code' or 'College Name' column. Adjust 'on=' if needed.
merged_branches = pd.merge(
    branches, 
    colleges[['TNEA Code', 'District', 'College Name']], 
    on='TNEA Code', 
    how='left', 
    suffixes=('', '_college')
)

# 5. Create Rich Text Chunks for Embedding
def create_branch_document(row):
    text = (
        f"College Name: {row.get('College Name_college', row.get('College Name', 'Unknown'))}\n"
        f"TNEA Code: {row['TNEA Code']}\n"
        f"District: {row['District']}\n"
        f"Branch Offered: {row['Branch_Full_Name']} (Code: {row['Branch Code']})\n"
        f"Approved Intake: {row.get('Approved Intake', 'N/A')} seats\n"
        f"NBA Accredited: {row.get('NBA Accredited', 'N/A')}\n"
    )
    return text

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
        "college_name": row.get('College Name_college', row.get('College Name', 'Unknown')),
        "tnea_code": str(row['TNEA Code']),
        "district": row['District'],
        "branch_code": row['Branch Code'],
        "branch_full_name": row['Branch_Full_Name'],
        "nba_accredited": str(row.get('NBA Accredited', 'N/A')).lower() in ['yes', 'true', '1']
    }
    
    records.append({
        "content": content,
        "embedding": embedding,
        "metadata": metadata
    })

# Batch upload to Supabase (chunks of 500)
for i in range(0, len(records)import pandas as pd
import os
from supabase import create_client
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

load_dotenv()

# 1. Initialize Supabase and Embedding Model
print("🔌 Connecting to Supabase and loading Embedding Model...")
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 2. Dynamically find the CSVs in the data/raw/ folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Goes up to RAG-final/
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

colleges_path = os.path.join(RAW_DIR, "colleges_db_df.csv")
branches_path = os.path.join(RAW_DIR, "branches_db_df.csv")

print(f"📂 Loading CSVs from {RAW_DIR}...")
colleges = pd.read_csv(colleges_path)
branches = pd.read_csv(branches_path)

# Fill NaNs to prevent string/int conversion errors
colleges = colleges.fillna("")
branches = branches.fillna("")

# 3. Branch Code Mapping Dictionary
BRANCH_MAP = {
    'CS': 'Computer Science and Engineering', 
    'EC': 'Electronics and Communication Engineering',
    'ME': 'Mechanical Engineering', 
    'EE': 'Electrical and Electronics Engineering',
    'CE': 'Civil Engineering', 
    'IT': 'Information Technology',
    'AD': 'Artificial Intelligence and Data Science', 
    'AI': 'Artificial Intelligence and Machine Learning',
    'CB': 'Computer Science and Business Systems', 
    'CY': 'Cyber Security',
    'AU': 'Automobile Engineering',
    'CH': 'Chemical Engineering',
    'AG': 'Agricultural Engineering',
    'BM': 'Bio Medical Engineering'
}

# Map codes to full names (using correct lowercase column name)
branches['branch_full_name'] = branches['branch_code'].map(BRANCH_MAP).fillna(branches['branch_code'])

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
    nba_status = "Yes" if str(row.get('nba_accredited', '')).strip().lower() in ['yes', 'true', '1'] else "No"
    valid_upto = f" (Valid upto {row['accreditation_valid_upto']})" if nba_status == "Yes" and row.get('accreditation_valid_upto') else ""
    
    text = (
        f"College Name: {row.get('college_name', 'Unknown')}\n"
        f"TNEA Code: {row['tnea_code']}\n"
        f"District: {row['district']}\n"
        f"Branch Offered: {row['branch_full_name']} (Code: {row['branch_code']})\n"
        f"Approved Intake: {int(row['approved_intake']) if row.get('approved_intake') else 'N/A'} seats\n"
        f"NBA Accredited: {nba_status}{valid_upto}\n"
    )
    return text

merged_branches['content'] = merged_branches.apply(create_branch_document, axis=1)

# 6. Wipe old branch data and Upload new enriched data
print("🗑️ Wiping old branch_info from Supabase...")
supabase.table("documents").delete().eq("metadata->>doc_type", "branch_info").execute()

print("🚀 Uploading enriched relational documents...")
records = []
for _, row in tqdm(merged_branches.iterrows(), total=len(merged_branches)):
    content = row['content']
    embedding = model.encode(content, normalize_embeddings=True).tolist()
    
    nba_bool = str(row.get('nba_accredited', '')).strip().lower() in ['yes', 'true', '1']
    
    metadata = {
        "doc_type": "branch_info",
        "college_name": row.get('college_name', 'Unknown'),
        "tnea_code": str(row['tnea_code']),
        "district": str(row['district']).strip().title(),
        "branch_code": row['branch_code'],
        "branch_full_name": row['branch_full_name'],
        "nba_accredited": nba_bool,
        "approved_intake": int(row['approved_intake']) if row.get('approved_intake') else 0
    }
    
    records.append({
        "content": content,
        "embedding": embedding,
        "metadata": metadata
    })

# Batch upload to Supabase (chunks of 500)
for i in range(0, len(records), 500):
    batch = records[i:i+500]
    supabase.table("documents").insert(batch).execute()

print(f"🎉 Relational Data Enrichment Complete! Uploaded {len(records)} branch documents."), 500):
    batch = records[i:i+500]
    supabase.table("documents").insert(batch).execute()

print("🎉 Relational Data Enrichment Complete! The AI will never mix up districts and branches again.")