import sys
import os

# Fix: Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.database import supabase
from sentence_transformers import SentenceTransformer
import pandas as pd

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Load embedding model
print("Loading embedding model...")
embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# Load data
print("Loading colleges data...")
df = pd.read_csv("data/raw/colleges_db_df.csv")
df = df.fillna("")
print(f"Loaded {len(df)} colleges")

# Generate embeddings
print("Generating embeddings...")
embeddings = embedding_model.encode(
    df["college_name"].tolist(),  # Embed the college name or full content
    show_progress_bar=True,
    normalize_embeddings=True
)

print(f"Generated {len(embeddings)} embeddings")

# Prepare records
records = []
for i, row in df.iterrows():
    metadata = {
        "source": "colleges_db_df.csv",
        "tnea_code": str(row["tnea_code"]),
        "college_name": str(row["college_name"]),
        "district": str(row["district"]),
        "autonomous": str(row["autonomous_status"])
    }
    
    records.append({
        "content": str(row["college_name"]),  # or full content
        "metadata": metadata,
        "embedding": embeddings[i].tolist()
    })

print(f"Prepared {len(records)} records")

# Upload to Supabase
print("Uploading to Supabase...")
BATCH_SIZE = 10

for i in range(0, len(records), BATCH_SIZE):
    batch = records[i:i + BATCH_SIZE]
    supabase.table("documents").insert(batch).execute()
    print(f"  Uploaded {min(i + BATCH_SIZE, len(records))}/{len(records)}")

print("✅ Done!")