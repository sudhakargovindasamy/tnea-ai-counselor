"""
06_verify_supabase_data.py
Analyzes and verifies the integrity of data in Supabase.
"""
import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ Error: Missing SUPABASE_URL or SUPABASE_KEY in .env")
    exit(1)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

print("="*60)
print("🔍 SUPABASE DATA HEALTH CHECK & ANALYSIS")
print("="*60)

# ==========================================
# 1. ANALYZE 'documents' TABLE (Colleges & Branches)
# ==========================================
print("\n📊 TABLE: documents (Colleges & Branches)")
print("-" * 40)

try:
    # Total count
    total_docs = supabase.table("documents").select("id", count="exact").execute()
    print(f"✅ Total Records: {total_docs.count}")

    # Count by doc_type
    college_count = supabase.table("documents").select("id").filter("metadata->>doc_type", "eq", "college_info").execute()
    branch_count = supabase.table("documents").select("id").filter("metadata->>doc_type", "eq", "branch_info").execute()
    
    print(f"   🏫 College Info Records: {len(college_count.data)}")
    print(f"   🎓 Branch Info Records: {len(branch_count.data)}")

    # Check for missing embeddings (Critical for RAG)
    missing_embeddings = supabase.table("documents").select("id").is_("embedding", "null").execute()
    if len(missing_embeddings.data) > 0:
        print(f"❌ WARNING: {len(missing_embeddings.data)} records are missing embeddings!")
    else:
        print("✅ All records have valid vector embeddings.")

    # Print College Sample
    print("\n📄 SAMPLE: College Info (Checking for data collapse)")
    sample_college = supabase.table("documents").select("content, metadata").filter("metadata->>doc_type", "eq", "college_info").limit(1).execute()
    if sample_college.data:
        meta = sample_college.data[0]['metadata']
        content = sample_college.data[0]['content']
        print(f"   College Name: {meta.get('college_name', 'N/A')}")
        print(f"   TNEA Code: {meta.get('tnea_code', 'N/A')}")
        print(f"   District: {meta.get('district', 'N/A')}")
        print(f"   Content Length: {len(content)} characters")
        print(f"   Preview: {content[:150].replace(chr(10), ' ')}...")

    # Print Branch Sample
    print("\n📄 SAMPLE: Branch Info")
    sample_branch = supabase.table("documents").select("content, metadata").filter("metadata->>doc_type", "eq", "branch_info").limit(1).execute()
    if sample_branch.data:
        meta = sample_branch.data[0]['metadata']
        content = sample_branch.data[0]['content']
        print(f"   TNEA Code: {meta.get('tnea_code', 'N/A')}")
        print(f"   Branch Code: {meta.get('branch_code', 'N/A')}")
        print(f"   Approved Intake: {meta.get('approved_intake', 'N/A')}")
        print(f"   Content Length: {len(content)} characters")
        print(f"   Preview: {content[:150].replace(chr(10), ' ')}...")

except Exception as e:
    print(f"❌ Error querying 'documents' table: {e}")

# ==========================================
# 2. ANALYZE 'admission_documents' TABLE
# ==========================================
print("\n\n📊 TABLE: admission_documents (TNEA Rules)")
print("-" * 40)

try:
    total_admissions = supabase.table("admission_documents").select("id", count="exact").execute()
    print(f"✅ Total Records: {total_admissions.count}")

    missing_adm_embeddings = supabase.table("admission_documents").select("id").is_("embedding", "null").execute()
    if len(missing_adm_embeddings.data) > 0:
        print(f"❌ WARNING: {len(missing_adm_embeddings.data)} records are missing embeddings!")
    else:
        print("✅ All records have valid vector embeddings.")

    print("\n📄 SAMPLE: Admission Info")
    sample_adm = supabase.table("admission_documents").select("content, metadata").limit(1).execute()
    if sample_adm.data:
        meta = sample_adm.data[0]['metadata']
        content = sample_adm.data[0]['content']
        print(f"   Section: {meta.get('section', 'N/A')}")
        print(f"   Source: {meta.get('source_document', 'N/A')}")
        print(f"   Content Length: {len(content)} characters")
        print(f"   Preview: {content[:150].replace(chr(10), ' ')}...")

except Exception as e:
    print(f"⚠️  Could not query 'admission_documents' table (Might not exist yet): {e}")

print("\n" + "="*60)
print("✅ ANALYSIS COMPLETE")
print("="*60)