import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from app.services.database import supabase

print("Loading branches...")
df = pd.read_csv("data/raw/branches_db_df.csv")
df["tnea_code"] = df["tnea_code"].astype(str)
print(f"🌿 {len(df)} branch records")

records = []
for _, row in df.iterrows():
    records.append({
        "tnea_code": str(row["tnea_code"]),
        "sl_no": int(row["sl_no"]) if pd.notna(row["sl_no"]) else None,
        "branch_code": str(row["branch_code"]).strip(),
        "approved_intake": int(row["approved_intake"]) if pd.notna(row["approved_intake"]) else 0,
        "year_of_starting": float(row["year_of_starting"]) if pd.notna(row["year_of_starting"]) else None,
        "nba_accredited": str(row["nba_accredited"]).strip() if pd.notna(row["nba_accredited"]) else "",
        "accreditation_valid_upto": str(int(row["accreditation_valid_upto"])) if pd.notna(row["accreditation_valid_upto"]) else "",
        "approval_note": str(row["approval_note"]).strip() if pd.notna(row["approval_note"]) else ""
    })

print("Uploading...")
BATCH = 100
for i in range(0, len(records), BATCH):
    supabase.table("branches").insert(records[i:i+BATCH]).execute()
    print(f"  ✅ {min(i+BATCH, len(records))}/{len(records)}")

print("🎉 Branches ingested!")