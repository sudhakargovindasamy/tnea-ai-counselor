import pandas as pd
import json
import os
import re

# ═══════════════════════════════════════════════════════════
# 1. SETUP PATHS
# ═══════════════════════════════════════════════════════════
BASE_DIR = "/home/sudhakar/Intern/RAG-final"
RAW_DIR = os.path.join(BASE_DIR, "data/raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data/processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

print("📥 Loading raw CSV files...")
colleges_df = pd.read_csv(os.path.join(RAW_DIR, "colleges_db_df.csv"))
branches_df = pd.read_csv(os.path.join(RAW_DIR, "branches_db_df.csv"))

colleges_df = colleges_df.fillna("")
branches_df = branches_df.fillna("")

print(f"   → Colleges: {len(colleges_df)} rows")
print(f"   → Branches: {len(branches_df)} rows")

# ═══════════════════════════════════════════════════════════
# 2. DISTRICT NORMALIZATION (Fix spelling variants)
# ═══════════════════════════════════════════════════════════
DISTRICT_NORMALIZATION = {
    "kancheepuram": "Kanchipuram",
    "kanchipuram": "Kanchipuram",
    "chengalpet": "Chengalpattu",
    "chengalpattu": "Chengalpattu",
    "trichirappalli": "Tiruchirappalli",
    "tiruchirappalli": "Tiruchirappalli",
    "trichy": "Tiruchirappalli",
    "kanniyakumari": "Kanyakumari",
    "kanyakumari": "Kanyakumari",
    "villupuram": "Viluppuram",
    "viluppuram": "Viluppuram",
    "tirupathur": "Tirupattur",
    "tirupattur": "Tirupattur",
    "the nilgiris": "The Nilgiris",
    "nilgiris": "The Nilgiris",
}

def normalize_district(raw_district: str) -> str:
    """Normalize district name to fix spelling variants."""
    if not raw_district or not str(raw_district).strip():
        return "Unknown"
    cleaned = str(raw_district).strip().title()
    lookup = cleaned.lower()
    return DISTRICT_NORMALIZATION.get(lookup, cleaned)

# ═══════════════════════════════════════════════════════════
# 3. BRANCH CODE → FULL NAME MAPPING
# ═══════════════════════════════════════════════════════════
BRANCH_MAP = {
    "CS": "Computer Science and Engineering",
    "EC": "Electronics and Communication Engineering",
    "ME": "Mechanical Engineering",
    "EE": "Electrical and Electronics Engineering",
    "CE": "Civil Engineering",
    "IT": "Information Technology",
    "AD": "Artificial Intelligence and Data Science",
    "AI": "Artificial Intelligence and Machine Learning",
    "AL": "Artificial Intelligence and Machine Learning",
    "AG": "Agricultural Engineering",
    "AU": "Automobile Engineering",
    "AM": "Aeronautical Engineering",
    "BM": "Bio Medical Engineering",
    "BT": "Bio Technology",
    "CB": "Computer Science and Business Systems",
    "CH": "Chemical Engineering",
    "CY": "Cyber Security",
    "EI": "Electronics and Instrumentation Engineering",
    "IC": "Information and Communication Engineering",
    "MZ": "Mechatronics Engineering",
    "FD": "Food Technology",
    "FT": "Fashion Technology",
    "TX": "Textile Technology",
    "SC": "Computer Science and Engineering",
    "CO": "Computer Science and Engineering",
    "CN": "Construction Engineering",
    "EM": "Embedded Systems",
    "EY": "Energy Engineering",
    "EL": "Electrical Engineering",
    "GI": "Geo Informatics",
    "MD": "Mechanical Engineering (Sandwich)",
    "MC": "Mechatronics",
    "MT": "Manufacturing Technology",
    "PC": "Petrochemical Engineering",
    "PH": "Pharmaceutical Technology",
    "PE": "Production Engineering",
    "RM": "Robotics and Mechatronics",
    "SF": "Safety and Fire Engineering",
    "VL": "VLSI Design",
    "CD": "Computer Science and Design",
    "CJ": "Computer Science and IoT",
    "DA": "Data Science",
    "DS": "Data Science and Engineering",
    "EA": "Electronics and Automation",
    "EF": "Electronics and VLSI",
    "EN": "Environmental Engineering",
    "EV": "Electric Vehicle Engineering",
    "EX": "Electronics and Communication (Extended)",
    "HT": "Horticulture Technology",
    "IB": "Industrial Biotechnology",
    "IN": "Instrumentation Engineering",
    "MM": "Marine Engineering",
    "MR": "Marine Engineering (Offshore)",
    "MU": "Mechanical Engineering (Automation)",
    "TC": "Textile Chemistry",
    "TT": "Textile Technology",
    "AR": "Architecture",
    "AS": "Aerospace Engineering",
    "BS": "Bio Science",
    "BY": "Biomedical Engineering",
    "CC": "Ceramic Technology",
    "CF": "Computer Science and Cyber Security",
    "CG": "Computer Science and Gaming",
    "CI": "Civil and Infrastructure Engineering",
    "CK": "Chemical Engineering (Petroleum)",
    "CL": "Chemical Engineering (Polymer)",
    "CM": "Computer Science and Engineering (AI & ML)",
    "FY": "Food and Dairy Technology",
    "IY": "Information Technology (Cyber Security)",
    "MF": "Mechanical Engineering (Design)",
    "MS": "Mechanical Engineering (Smart Manufacturing)",
    "MY": "Mechanical Engineering (Robotics)",
    "PN": "Petroleum Engineering",
    "PR": "Production Engineering",
    "RA": "Robotics and Automation",
    "RI": "Robotics and Industrial Automation",
    "SB": "Computer Science and Business Analytics",
    "TS": "Computer Science and Technology",
    "X": "Computer Science and Engineering",
    "B*": "Computer Science and Engineering",
    "AT": "Automobile Technology",
    "AE": "Aeronautical Engineering",
    "AO": "Aerospace and Ocean Engineering",
    "BC": "Bio Chemical Engineering",
    "BP": "Bio Process Engineering",
    "BA": "Bio Agriculture",
    "LE": "Leather Technology",
    "PP": "Pulp and Paper Technology",
    "PM": "Polymer Technology",
    "IS": "Industrial Safety",
    "AP": "Applied Psychology",
    "PT": "Printing Technology",
    "MB": "Mechanical Engineering (Business)",
    "MN": "Mining Engineering",
    "MA": "Mechanical Engineering (AI)",
}

def get_branch_full_name(code: str) -> str:
    """Map branch code to full name."""
    code = str(code).strip().upper()
    return BRANCH_MAP.get(code, f"Engineering ({code})")

# ═══════════════════════════════════════════════════════════
# 4. CREATE COLLEGE_INFO DOCUMENTS
# ═══════════════════════════════════════════════════════════
print("\n📝 Creating college_info documents...")

college_documents = []

for _, row in colleges_df.iterrows():
    district = normalize_district(row['district'])
    
    # Build rich text content
    text = f"College Name: {row['college_name']}\n"
    text += f"TNEA Code: {row['tnea_code']}\n"
    text += f"District: {district}\n"
    text += f"Taluk: {row['taluk']}\n"
    text += f"Address: {row['address']}\n"
    text += f"Pincode: {row['pincode']}\n"
    text += f"Principal: {row['dean_principal']}\n"
    text += f"Phone: {row['phone_fax']}\n"
    text += f"Email: {row['email_id']}\n"
    text += f"Website: {row['website']}\n"
    text += f"Anti-Ragging Helpline: {row['anti_ragging_phone_no']}\n"
    text += f"Autonomous Status: {row['autonomous_status']}\n"
    text += f"Minority Status: {row['minority_status']}\n"
    
    if row['placement']:
        text += f"Placement Rate: {row['placement']}%\n"
    
    text += f"Hostel Facilities: Boys ({row['hostel_facilities_boys']}), Girls ({row['hostel_facilities_girls']})\n"
    text += f"Accommodation UG: Boys ({row['accommodation_ug_boys']}), Girls ({row['accommodation_ug_girls']})\n"
    text += f"Hostel Type: Boys ({row['permanent_or_rental_boys']}), Girls ({row['permanent_or_rental_girls']})\n"
    text += f"Mess Type: Boys ({row['mess_type_boys']}), Girls ({row['mess_type_girls']})\n"
    
    if row['mess_bill_boys']:
        text += f"Mess Bill: Boys (₹{row['mess_bill_boys']}), Girls (₹{row['mess_bill_girls']})\n"
    if row['room_rent_boys']:
        text += f"Room Rent: Boys (₹{row['room_rent_boys']}), Girls (₹{row['room_rent_girls']})\n"
    if row['electricity_charges_boys']:
        text += f"Electricity Charges: Boys (₹{row['electricity_charges_boys']}), Girls (₹{row['electricity_charges_girls']})\n"
    if row['caution_deposit']:
        text += f"Caution Deposit: ₹{row['caution_deposit']}\n"
    if row['establishment_charges']:
        text += f"Establishment Charges: ₹{row['establishment_charges']}\n"
    if row['admission_fees']:
        text += f"Admission Fees: ₹{row['admission_fees']}\n"
    
    text += f"Transport Facilities: {row['transport_facilities']}\n"
    if row['min_transport_charges']:
        text += f"Transport Charges: ₹{row['min_transport_charges']} to ₹{row['max_transport_charges']}\n"
    
    text += f"Distance from District HQ: {row['distance_from_district_hq']} km\n"
    text += f"Nearest Railway Station: {row['nearest_railway_station']} ({row['distance_from_railway_station']} km)\n"
    
    # Build metadata
    metadata = {
        "doc_type": "college_info",
        "tnea_code": str(row['tnea_code']),
        "college_name": str(row['college_name']).strip(),
        "district": district,
        "autonomous_status": str(row['autonomous_status']).strip(),
        "minority_status": str(row['minority_status']).strip(),
        "placement_rate": float(row['placement']) if row['placement'] else 0.0,
        "source": "colleges_db_df.csv",
    }
    
    college_documents.append({
        "content": text.strip(),
        "metadata": metadata,
    })

print(f"   → Created {len(college_documents)} college_info documents")

# ═══════════════════════════════════════════════════════════
# 5. CREATE BRANCH_INFO DOCUMENTS
# ═══════════════════════════════════════════════════════════
print("📝 Creating branch_info documents...")

# Build college lookup for district and name
college_lookup = {}
for _, row in colleges_df.iterrows():
    college_lookup[str(row['tnea_code'])] = {
        "college_name": str(row['college_name']).strip(),
        "district": normalize_district(row['district']),
    }

branch_documents = []

for _, row in branches_df.iterrows():
    tnea_code = str(row['tnea_code'])
    branch_code = str(row['branch_code']).strip().upper()
    branch_full_name = get_branch_full_name(branch_code)
    
    college_info = college_lookup.get(tnea_code, {})
    college_name = college_info.get("college_name", f"College {tnea_code}")
    district = college_info.get("district", "Unknown")
    
    approved_intake = int(row['approved_intake']) if row['approved_intake'] else 0
    year_of_starting = int(row['year_of_starting']) if row['year_of_starting'] else 0
    nba_accredited = str(row['nba_accredited']).strip().lower() in ['yes', 'true', '1']
    accreditation_valid_upto = str(row['accreditation_valid_upto']).strip() if row['accreditation_valid_upto'] else ""
    approval_note = str(row['approval_note']).strip() if row['approval_note'] else ""
    
    # Build rich text content
    text = f"College Name: {college_name}\n"
    text += f"TNEA Code: {tnea_code}\n"
    text += f"District: {district}\n"
    text += f"Branch: {branch_full_name} (Code: {branch_code})\n"
    text += f"Approved Intake: {approved_intake} seats\n"
    
    if year_of_starting:
        text += f"Year of Starting: {year_of_starting}\n"
    
    text += f"NBA Accredited: {'Yes' if nba_accredited else 'No'}"
    if nba_accredited and accreditation_valid_upto:
        text += f" (Valid till {accreditation_valid_upto})"
    text += "\n"
    
    if approval_note:
        text += f"Approval Note: {approval_note}\n"
    
    # Build metadata
    metadata = {
        "doc_type": "branch_info",
        "tnea_code": tnea_code,
        "college_name": college_name,
        "district": district,
        "branch_code": branch_code,
        "branch_full_name": branch_full_name,
        "approved_intake": approved_intake,
        "year_of_starting": year_of_starting,
        "nba_accredited": nba_accredited,
        "accreditation_valid_upto": accreditation_valid_upto,
        "source": "branches_db_df.csv",
    }
    
    branch_documents.append({
        "content": text.strip(),
        "metadata": metadata,
    })

print(f"   → Created {len(branch_documents)} branch_info documents")

# ═══════════════════════════════════════════════════════════
# 6. SAVE PROCESSED DOCUMENTS
# ═══════════════════════════════════════════════════════════
print("\n💾 Saving processed documents...")

# Save college documents
college_records = []
for doc in college_documents:
    college_records.append({
        "content": doc["content"],
        "metadata": json.dumps(doc["metadata"]),
    })

college_output = pd.DataFrame(college_records)
college_file = os.path.join(PROCESSED_DIR, "college_documents.csv")
college_output.to_csv(college_file, index=False)
print(f"   → Saved {len(college_output)} college documents to {college_file}")

# Save branch documents
branch_records = []
for doc in branch_documents:
    branch_records.append({
        "content": doc["content"],
        "metadata": json.dumps(doc["metadata"]),
    })

branch_output = pd.DataFrame(branch_records)
branch_file = os.path.join(PROCESSED_DIR, "branch_documents.csv")
branch_output.to_csv(branch_file, index=False)
print(f"   → Saved {len(branch_output)} branch documents to {branch_file}")

# ═══════════════════════════════════════════════════════════
# 7. SUMMARY
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("📊 PREPROCESSING SUMMARY")
print("=" * 60)
print(f"   College documents: {len(college_documents)}")
print(f"   Branch documents:  {len(branch_documents)}")
print(f"   Total documents:   {len(college_documents) + len(branch_documents)}")
print(f"   Districts found:   {len(set(normalize_district(d) for d in colleges_df['district'] if d))}")
print(f"   Branch codes found: {len(set(str(b).strip().upper() for b in branches_df['branch_code'] if b))}")
print("=" * 60)
print("✅ Preprocessing complete!")
print("\n📌 Next step: Run ingestion script to upload to Supabase:")
print("   python scripts/08_master_ingest.py")