"""
scripts/preprocess_data.py
Data Denormalization & Preprocessing for TNEA Counselor RAG System.

Merges colleges_db_df.csv, branches_db_df.csv, and performance_db_df.csv
into ONE unified, rich document per college (tnea_code).
Also converts tnea_admission_info.json into structured admission_info documents.
"""

import json
import os
from typing import Any

import pandas as pd

# ═══════════════════════════════════════════════════════════
# 1. PATH RESOLUTION
# ═══════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

def find_raw_file(filename: str) -> str:
    """Find file in data/raw/ or root directory."""
    p1 = os.path.join(RAW_DIR, filename)
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(BASE_DIR, filename)
    if os.path.exists(p2):
        return p2
    return filename

# ═══════════════════════════════════════════════════════════
# 2. NORMALIZATION & MAPPINGS
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
    "thiruvallur": "Tiruvallur",
    "tiruvallur": "Tiruvallur",
    "thiruvannamalai": "Tiruvannamalai",
    "tiruvannamalai": "Tiruvannamalai",
    "thiruvarur": "Tiruvarur",
    "tiruvarur": "Tiruvarur",
    "thirunelveli": "Tirunelveli",
    "tirunelveli": "Tirunelveli",
}

def normalize_district(raw_district: Any) -> str:
    """Standardize district name to title case and resolve spelling variants."""
    if pd.isna(raw_district) or not str(raw_district).strip():
        return "Unknown"
    cleaned = str(raw_district).strip().title()
    lookup = cleaned.lower()
    return DISTRICT_NORMALIZATION.get(lookup, cleaned)

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
    """Return full branch name for given code."""
    c = str(code).strip().upper()
    return BRANCH_MAP.get(c, f"Engineering ({c})")

# ═══════════════════════════════════════════════════════════
# 3. LOAD RAW DATA
# ═══════════════════════════════════════════════════════════
print("📥 Loading raw datasets...")
colleges_path = find_raw_file("colleges_db_df.csv")
branches_path = find_raw_file("branches_db_df.csv")
perf_path = find_raw_file("performance_db_df.csv")
adm_path = find_raw_file("tnea_admission_info.json")

colleges_df = pd.read_csv(colleges_path).fillna("")
branches_df = pd.read_csv(branches_path).fillna("")
perf_df = pd.read_csv(perf_path).fillna("")

print(f"   ✓ Colleges loaded: {len(colleges_df)} rows from {colleges_path}")
print(f"   ✓ Branches loaded: {len(branches_df)} rows from {branches_path}")
print(f"   ✓ Performance loaded: {len(perf_df)} rows from {perf_path}")

# Pre-index branches by tnea_code
branches_by_tnea: dict[str, list[dict[str, Any]]] = {}
for _, row in branches_df.iterrows():
    t_code = str(row["tnea_code"]).strip()
    if not t_code:
        continue
    branches_by_tnea.setdefault(t_code, []).append({
        "sl_no": row["sl_no"],
        "branch_code": str(row["branch_code"]).strip().upper(),
        "approved_intake": int(float(row["approved_intake"])) if str(row["approved_intake"]).strip() else 0,
        "year_of_starting": int(float(row["year_of_starting"])) if str(row["year_of_starting"]).strip() else None,
        "nba_accredited": str(row["nba_accredited"]).strip().lower() in ["yes", "true", "1"],
        "accreditation_valid_upto": str(row["accreditation_valid_upto"]).strip(),
        "approval_note": str(row["approval_note"]).strip()
    })

# Pre-index performance by tnea_code
perf_by_tnea: dict[str, dict[str, Any]] = {}
for _, row in perf_df.iterrows():
    t_code = str(row["tnea_code"]).strip()
    if not t_code:
        continue
    # Keep highest appeared if multiple records exist
    appeared = int(float(row["total_appeared"])) if str(row["total_appeared"]).strip() else 0
    passed = int(float(row["total_passed"])) if str(row["total_passed"]).strip() else 0
    pass_pct = float(row["pass_percentage"]) if str(row["pass_percentage"]).strip() else 0.0
    
    if t_code not in perf_by_tnea or appeared > perf_by_tnea[t_code]["total_appeared"]:
        perf_by_tnea[t_code] = {
            "total_appeared": appeared,
            "total_passed": passed,
            "pass_percentage": pass_pct,
            "college_name_perf": str(row["college_name"]).strip(),
            "district_perf": normalize_district(row["district"])
        }

# ═══════════════════════════════════════════════════════════
# 4. CREATE DENORMALIZED COLLEGE DOCUMENTS
# ═══════════════════════════════════════════════════════════
print("\n🏗️ Denormalizing data into rich college documents...")

college_documents = []

for _, r in colleges_df.iterrows():
    tnea_code = str(r["tnea_code"]).strip()
    if not tnea_code:
        continue

    college_name = str(r["college_name"]).strip()
    district = normalize_district(r["district"])
    taluk = str(r["taluk"]).strip()
    address = str(r["address"]).strip()
    pincode = str(r["pincode"]).strip()
    dean_principal = str(r["dean_principal"]).strip()
    phone_fax = str(r["phone_fax"]).strip()
    email_id = str(r["email_id"]).strip()
    website = str(r["website"]).strip()
    anti_ragging = str(r["anti_ragging_phone_no"]).strip()
    
    # Autonomous & Minority
    autonomous_bool = str(r["autonomous_status"]).strip().lower() in ["yes", "true", "1"]
    minority_status = str(r["minority_status"]).strip() or "No"
    
    # Placement
    placement_val = float(r["placement"]) if str(r["placement"]).strip() else 0.0
    
    # Hostels & Fees
    boys_hostel = str(r["hostel_facilities_boys"]).strip() or "N/A"
    girls_hostel = str(r["hostel_facilities_girls"]).strip() or "N/A"
    mess_boys = str(r["mess_bill_boys"]).strip()
    mess_girls = str(r["mess_bill_girls"]).strip()
    room_boys = str(r["room_rent_boys"]).strip()
    room_girls = str(r["room_rent_girls"]).strip()
    electricity_boys = str(r["electricity_charges_boys"]).strip()
    electricity_girls = str(r["electricity_charges_girls"]).strip()
    caution_deposit = str(r["caution_deposit"]).strip()
    establishment = str(r["establishment_charges"]).strip()
    admission_fee = str(r["admission_fees"]).strip()
    transport = str(r["transport_facilities"]).strip() or "N/A"
    min_transport = str(r["min_transport_charges"]).strip()
    max_transport = str(r["max_transport_charges"]).strip()
    
    # Distances
    dist_hq = str(r["distance_from_district_hq"]).strip()
    railway_stn = str(r["nearest_railway_station"]).strip()
    railway_dist = str(r["distance_from_railway_station"]).strip()

    # ── Branches ──
    college_branches = branches_by_tnea.get(tnea_code, [])
    branch_codes = []
    branches_summary_parts = []
    branches_detailed_lines = []
    total_approved_intake = 0

    for b in college_branches:
        b_code = b["branch_code"]
        if b_code and b_code not in branch_codes:
            branch_codes.append(b_code)
        
        intake = b["approved_intake"]
        total_approved_intake += intake
        branches_summary_parts.append(f"{b_code} (Intake {intake})")
        
        nba_str = "NBA: Accredited" if b["nba_accredited"] else "NBA: No"
        if b["accreditation_valid_upto"]:
            nba_str += f" (Valid till {b['accreditation_valid_upto']})"
        
        b_full = get_branch_full_name(b_code)
        branches_detailed_lines.append(
            f"  - {b_code}: {b_full} | Intake: {intake} seats | {nba_str}"
        )

    branches_summary_str = ", ".join(branches_summary_parts) if branches_summary_parts else "None listed"

    # ── Performance ──
    perf_data = perf_by_tnea.get(tnea_code)
    if perf_data:
        pass_pct = perf_data["pass_percentage"]
        perf_summary_str = f"{pass_pct}% pass ({perf_data['total_passed']}/{perf_data['total_appeared']} students passed)"
    else:
        pass_pct = 0.0
        perf_summary_str = "N/A"

    # ── Build Rich Document Text ──
    # Exact required format: "College: X. District: Y. Branches: CS (Intake 60), AD (Intake 30). Performance: 85% pass."
    doc_lines = [
        f"College: {college_name}. District: {district}. Branches: {branches_summary_str}. Performance: {perf_summary_str}.",
        f"TNEA Code: {tnea_code}",
        f"Autonomous Status: {'Yes' if autonomous_bool else 'No'} | Minority Status: {minority_status}",
        f"Address: {address}, Taluk: {taluk}, District: {district} - {pincode}",
        f"Dean/Principal: {dean_principal} | Phone: {phone_fax} | Email: {email_id} | Website: {website}",
        f"Anti-Ragging Helpline: {anti_ragging}"
    ]

    if placement_val > 0:
        doc_lines.append(f"Placement Rate: {placement_val}%")

    if perf_data:
        doc_lines.append(
            f"Academic Performance: Pass Rate {perf_data['pass_percentage']}% "
            f"(Passed: {perf_data['total_passed']} out of {perf_data['total_appeared']} appeared in Anna University exams)"
        )

    # Detailed Branch Section
    if branches_detailed_lines:
        doc_lines.append(f"Branches Offered ({len(branch_codes)} branches, Total Intake: {total_approved_intake} seats):")
        doc_lines.extend(branches_detailed_lines)

    # Hostel and Amenities Section
    hostel_info = []
    if boys_hostel and boys_hostel != "N/A":
        hostel_info.append(f"Boys Hostel: {boys_hostel}")
    if girls_hostel and girls_hostel != "N/A":
        hostel_info.append(f"Girls Hostel: {girls_hostel}")
    if mess_boys:
        hostel_info.append(f"Mess Bill Boys: ₹{mess_boys}")
    if mess_girls:
        hostel_info.append(f"Mess Bill Girls: ₹{mess_girls}")
    if room_boys:
        hostel_info.append(f"Room Rent Boys: ₹{room_boys}")
    if room_girls:
        hostel_info.append(f"Room Rent Girls: ₹{room_girls}")
    if caution_deposit:
        hostel_info.append(f"Caution Deposit: ₹{caution_deposit}")
    if establishment:
        hostel_info.append(f"Establishment Charges: ₹{establishment}")
    if admission_fee:
        hostel_info.append(f"Admission Fees: ₹{admission_fee}")

    if hostel_info:
        doc_lines.append("Hostel & Fee Details: " + " | ".join(hostel_info))

    # Transport & Location
    loc_info = []
    if transport and transport != "N/A":
        trans_str = f"Transport: {transport}"
        if min_transport and max_transport:
            trans_str += f" (Charges: ₹{min_transport} to ₹{max_transport})"
        loc_info.append(trans_str)
    if dist_hq:
        loc_info.append(f"Distance from District HQ: {dist_hq} km")
    if railway_stn:
        loc_info.append(f"Nearest Railway Station: {railway_stn} ({railway_dist} km)")

    if loc_info:
        doc_lines.append("Location & Transport: " + " | ".join(loc_info))

    content_text = "\n".join(doc_lines)

    # Clean Supabase metadata
    metadata = {
        "doc_type": "college_info",
        "tnea_code": str(tnea_code),
        "college_name": college_name,
        "district": district,
        "branch_codes": branch_codes,
        "branch_intakes": {b["branch_code"]: b["approved_intake"] for b in college_branches},
        "autonomous": autonomous_bool,
        "placement_rate": placement_val,
        "pass_percentage": pass_pct,
        "total_intake": total_approved_intake,
        "hostel_facilities_boys": boys_hostel if boys_hostel not in ("N/A", "nan", "") else None,
        "hostel_facilities_girls": girls_hostel if girls_hostel not in ("N/A", "nan", "") else None,
        "mess_bill_boys": float(mess_boys) if mess_boys and mess_boys not in ("nan", "N/A", "") else None,
        "mess_bill_girls": float(mess_girls) if mess_girls and mess_girls not in ("nan", "N/A", "") else None,
        "room_rent_boys": float(room_boys) if room_boys and room_boys not in ("nan", "N/A", "") else None,
        "room_rent_girls": float(room_girls) if room_girls and room_girls not in ("nan", "N/A", "") else None,
        "source": "colleges_db_df.csv"
    }

    college_documents.append({
        "content": content_text,
        "metadata": metadata
    })

print(f"   ✓ Successfully generated {len(college_documents)} denormalized college documents.")

# ═══════════════════════════════════════════════════════════
# 5. PROCESS TNEA ADMISSION RULES JSON
# ═══════════════════════════════════════════════════════════
print("\n📜 Processing TNEA admission rules JSON...")

def flatten_json(obj: Any, indent: int = 0) -> list[str]:
    """Recursively flatten dictionary or list into clean human-readable text lines."""
    lines = []
    prefix = "  " * indent
    if isinstance(obj, dict):
        for k, v in obj.items():
            clean_key = str(k).replace("_", " ").title()
            if isinstance(v, (dict, list)):
                lines.append(f"{prefix}{clean_key}:")
                lines.extend(flatten_json(v, indent + 1))
            else:
                lines.append(f"{prefix}{clean_key}: {v}")
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                lines.extend(flatten_json(item, indent + 1))
            else:
                lines.append(f"{prefix}- {item}")
    else:
        lines.append(f"{prefix}{obj}")
    return lines

admission_documents = []

if os.path.exists(adm_path):
    with open(adm_path, "r", encoding="utf-8") as f:
        raw_adm = json.load(f)

    for entry in raw_adm:
        section = entry.get("section", "General")
        source_doc = entry.get("source_document", "TNEA Information Brochure 2026")
        source_page = entry.get("source_page", 1)
        content_obj = entry.get("content", {})

        content_lines = [
            f"Topic / Section: {section}",
            f"Source: {source_doc} (Page {source_page})"
        ]
        content_lines.extend(flatten_json(content_obj, indent=0))
        text = "\n".join(content_lines)

        meta = {
            "doc_type": "admission_info",
            "section": section,
            "source_document": source_doc,
            "source_page": source_page,
            "source": "tnea_admission_info.json"
        }

        admission_documents.append({
            "content": text,
            "metadata": meta
        })
    print(f"   ✓ Successfully generated {len(admission_documents)} admission rules documents.")
else:
    print(f"   ⚠️ Admission rules file not found at: {adm_path}")

# ═══════════════════════════════════════════════════════════
# 6. EXPORT PROCESSED DATA
# ═══════════════════════════════════════════════════════════
print("\n💾 Saving processed files to data/processed/ ...")

# 1. College documents CSV
college_records = [
    {"content": d["content"], "metadata": json.dumps(d["metadata"])}
    for d in college_documents
]
college_df = pd.DataFrame(college_records)
college_csv_path = os.path.join(PROCESSED_DIR, "college_documents.csv")
college_df.to_csv(college_csv_path, index=False)
print(f"   ✓ Saved {len(college_df)} rows to {college_csv_path}")

# 2. College documents JSON
college_json_path = os.path.join(PROCESSED_DIR, "college_documents.json")
with open(college_json_path, "w", encoding="utf-8") as f:
    json.dump(college_documents, f, indent=2, ensure_ascii=False)
print(f"   ✓ Saved {len(college_documents)} items to {college_json_path}")

# 3. Admission documents CSV & JSON
if admission_documents:
    adm_records = [
        {"content": d["content"], "metadata": json.dumps(d["metadata"])}
        for d in admission_documents
    ]
    adm_df = pd.DataFrame(adm_records)
    adm_csv_path = os.path.join(PROCESSED_DIR, "admission_documents.csv")
    adm_df.to_csv(adm_csv_path, index=False)
    print(f"   ✓ Saved {len(adm_df)} rows to {adm_csv_path}")

    adm_json_path = os.path.join(PROCESSED_DIR, "admission_documents.json")
    with open(adm_json_path, "w", encoding="utf-8") as f:
        json.dump(admission_documents, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved {len(admission_documents)} items to {adm_json_path}")

print("\n" + "=" * 65)
print("🎯 PREPROCESSING SUMMARY")
print("=" * 65)
print(f"• Total Rich College Documents:   {len(college_documents)}")
print(f"• Total Admission Rule Documents: {len(admission_documents)}")
print("• Sample Document Preview:")
print("-" * 65)
if college_documents:
    print(college_documents[0]["content"][:320] + "\n...")
    print("Sample Metadata:", json.dumps(college_documents[0]["metadata"], indent=2))
print("=" * 65)
print("✅ Done! Data is ready for Supabase ingestion via scripts/08_master_ingest.py")