"""
05_ingest_admission_info.py
Ingests TNEA Admission Information (JSON) into Supabase vector DB.
Handles nested JSON by recursively flattening into readable text.
"""

import os
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client

# ==========================================
# 1. LOAD CONFIG
# ==========================================
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ==========================================
# 2. LOAD JSON DATA
# ==========================================
JSON_FILE = "data/raw/tnea_admission_info.json"

with open(JSON_FILE, "r", encoding="utf-8") as f:
    admission_data = json.load(f)

print(f"✅ Loaded {len(admission_data)} admission info sections")


# ==========================================
# 3. FLATTEN NESTED JSON INTO TEXT
# ==========================================
def flatten_value(key, value, indent=0):
    """
    Recursively flattens nested dicts/lists into readable 'key: value' lines.
    """
    lines = []
    prefix = "  " * indent

    if isinstance(value, dict):
        lines.append(f"{prefix}{key}:")
        for sub_key, sub_value in value.items():
            lines.extend(flatten_value(sub_key, sub_value, indent + 1))
    elif isinstance(value, list):
        # Join list items into a comma-separated string
        joined = ", ".join(str(item) for item in value)
        lines.append(f"{prefix}{key}: {joined}")
    else:
        lines.append(f"{prefix}{key}: {value}")

    return lines


def entry_to_document(entry):
    """
    Converts one admission info entry into a single text document.
    """
    lines = []

    # Add section as the heading (most important for retrieval)
    lines.append(f"Topic: {entry.get('section', 'General Information')}")

    # Flatten the nested content
    content = entry.get("content", {})
    for key, value in content.items():
        lines.extend(flatten_value(key, value))

    # Add source info
    lines.append(f"Source Document: {entry.get('source_document', 'TNEA Brochure')}")
    lines.append(f"Source Page: {entry.get('source_page', 'N/A')}")

    return "\n".join(lines)


# ==========================================
# 4. BUILD DOCUMENTS + METADATA
# ==========================================
documents = []
metadatas = []

for entry in admission_data:
    doc_text = entry_to_document(entry)

    metadata = {
        "source": "tnea_admission_info.json",
        "doc_type": "admission_info",          # 🚀 Key marker to distinguish from college data
        "section": entry.get("section", "General"),
        "source_document": entry.get("source_document", "TNEA Brochure"),
        "source_page": entry.get("source_page", "N/A"),
    }

    documents.append(doc_text)
    metadatas.append(metadata)

print(f"✅ Prepared {len(documents)} documents")
print("\n--- Sample Document ---")
print(documents[0])
print("-----------------------\n")


# ==========================================
# 5. GENERATE EMBEDDINGS
# ==========================================
print("🧠 Loading embedding model...")
embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")

print("🧠 Generating embeddings...")
embeddings = embedding_model.encode(
    documents,
    show_progress_bar=True,
    normalize_embeddings=True
)

print(f"✅ Embeddings shape: {embeddings.shape}")


# ==========================================
# 6. UPLOAD TO SUPABASE (in batches)
# ==========================================
embeddings_list = embeddings.tolist()
records = []

for i in range(len(documents)):
    records.append({
        "content": documents[i],
        "metadata": metadatas[i],
        "embedding": embeddings_list[i],
    })

BATCH_SIZE = 50
total = len(records)

for i in range(0, total, BATCH_SIZE):
    batch = records[i:i + BATCH_SIZE]
    supabase.table("documents").insert(batch).execute()
    print(f"📤 Uploaded {min(i + BATCH_SIZE, total)}/{total}")

print("\n🎉 Admission info ingestion complete!")