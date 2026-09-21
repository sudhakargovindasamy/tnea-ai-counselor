import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from app.services.database import supabase

print("Loading branches...")
raw_file = "data/raw/college_branches_rows.csv" if os.path.exists("data/raw/college_branches_rows.csv") else "data/raw/branches_db_df.csv"
df = pd.read_csv(raw_file)
df["tnea_code"] = df["tnea_code"].astype(str)
print(f"🌿 {len(df)} branch records from {raw_file}")

records = []
for _, row in df.iterrows():
    dept_code = str(row.get("department_code", row.get("branch_code", ""))).strip().upper()
    dept_name = str(row.get("department_name", "")).strip()
    appr_marker = str(row.get("approval_marker", row.get("approval_note", ""))).strip()
    if appr_marker.lower() in ("nan", "none"):
        appr_marker = ""

    valid_upto = ""
    if pd.notna(row.get("accreditation_valid_upto")):
        v = str(row.get("accreditation_valid_upto")).strip()
        if v.lower() not in ("nan", "none"):
            valid_upto = str(int(float(v))) if v.replace(".", "").isdigit() else v

    records.append({
        "tnea_code": str(row["tnea_code"]).strip(),
        "sl_no": int(float(row["sl_no"])) if pd.notna(row.get("sl_no")) and str(row.get("sl_no")).strip() not in ("nan", "none", "") else None,
        "department_code": dept_code,
        "department_name": dept_name,
        "approved_intake": int(float(row["approved_intake"])) if pd.notna(row.get("approved_intake")) and str(row.get("approved_intake")).strip() not in ("nan", "none", "") else 0,
        "year_of_starting": float(row["year_of_starting"]) if pd.notna(row.get("year_of_starting")) and str(row.get("year_of_starting")).strip() not in ("nan", "none", "") else None,
        "nba_accredited": str(row["nba_accredited"]).strip() if pd.notna(row.get("nba_accredited")) and str(row.get("nba_accredited")).strip() not in ("nan", "none") else "",
        "accreditation_valid_upto": valid_upto,
        "approval_marker": appr_marker,
        # Backward-compatibility aliases
        "branch_code": dept_code,
        "approval_note": appr_marker
    })

print("Wiping old branches from Supabase...")
try:
    supabase.table("branches").delete().neq("id", 0).execute()
except Exception as e:
    print(f"Notice during wipe: {e}")

print("Uploading branches...")
BATCH = 100
for i in range(0, len(records), BATCH):
    supabase.table("branches").insert(records[i:i+BATCH]).execute()
    print(f"  ✅ {min(i+BATCH, len(records))}/{len(records)}")

print("🎉 Branches ingested!")