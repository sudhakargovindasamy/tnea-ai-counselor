"""
07_fix_doc_type.py
Adds doc_type to existing records based on their metadata structure.
- Records with 'branch_code' → branch_info
- Records without 'branch_code' but with 'college_name' → college_info
"""
import os

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

print("🔍 Fetching all 713 records to analyze metadata structure...")

# Fetch all records with their metadata
response = supabase.table("documents").select("id, metadata").execute()
records = response.data

print(f"✅ Fetched {len(records)} records")

college_ids = []
branch_ids = []
unknown_ids = []

for record in records:
    meta = record.get("metadata", {})
    record_id = record["id"]
    
    # Determine doc_type based on metadata structure
    if "branch_code" in meta or "approved_intake" in meta:
        branch_ids.append(record_id)
    elif "college_name" in meta or "tnea_code" in meta:
        college_ids.append(record_id)
    else:
        unknown_ids.append(record_id)

print("\n📊 Analysis Results:")
print(f"   🏫 College Info: {len(college_ids)} records")
print(f"   🎓 Branch Info: {len(branch_ids)} records")
print(f"   ❓ Unknown: {len(unknown_ids)} records")

# ==========================================
# UPDATE COLLEGE RECORDS
# ==========================================
print(f"\n🔄 Updating {len(college_ids)} college records with doc_type='college_info'...")

BATCH_SIZE = 50
for i in range(0, len(college_ids), BATCH_SIZE):
    batch_ids = college_ids[i:i + BATCH_SIZE]
    
    for record_id in batch_ids:
        # Fetch current metadata
        current = supabase.table("documents").select("metadata").eq("id", record_id).single().execute()
        if current.data:
            meta = current.data["metadata"]
            meta["doc_type"] = "college_info"
            
            # Update the record
            supabase.table("documents").update({"metadata": meta}).eq("id", record_id).execute()
    
    print(f"   Updated {min(i + BATCH_SIZE, len(college_ids))}/{len(college_ids)}")

# ==========================================
# UPDATE BRANCH RECORDS
# ==========================================
print(f"\n🔄 Updating {len(branch_ids)} branch records with doc_type='branch_info'...")

for i in range(0, len(branch_ids), BATCH_SIZE):
    batch_ids = branch_ids[i:i + BATCH_SIZE]
    
    for record_id in batch_ids:
        current = supabase.table("documents").select("metadata").eq("id", record_id).single().execute()
        if current.data:
            meta = current.data["metadata"]
            meta["doc_type"] = "branch_info"
            
            supabase.table("documents").update({"metadata": meta}).eq("id", record_id).execute()
    
    print(f"   Updated {min(i + BATCH_SIZE, len(branch_ids))}/{len(branch_ids)}")

# ==========================================
# HANDLE UNKNOWN RECORDS
# ==========================================
if unknown_ids:
    print(f"\n⚠️  Found {len(unknown_ids)} records with unknown structure:")
    for uid in unknown_ids[:5]:
        current = supabase.table("documents").select("metadata, content").eq("id", uid).single().execute()
        if current.data:
            print(f"   ID {uid}: metadata keys = {list(current.data['metadata'].keys())}")
            print(f"   Content preview: {current.data['content'][:100]}...")

print("\n🎉 doc_type tagging complete!")
print("Run 06_verify_supabase_data.py again to confirm.")