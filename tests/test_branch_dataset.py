"""
tests/test_branch_dataset.py
Tests for the new branch dataset (college_branches_rows.csv)
Verifies:
1. Dataset loading, row counts, and column schemas.
2. Null handling and clean chunk formatting (no 'nan', 'None', or 'NaN').
3. Metadata structure and backward compatibility (department_code & branch_code).
4. Retrieval with department_code and branch_code filters.
"""

import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import pandas as pd
import pytest

from app.services.retrieval import (
    retrieve,
    get_colleges_by_filters,
    extract_branch_code,
    _normalize_filters,
)
RAW_CSV_PATH = os.path.join(BASE_DIR, "data", "raw", "college_branches_rows.csv")
PROCESSED_BRANCH_JSON = os.path.join(BASE_DIR, "data", "processed", "branch_documents.json")
PROCESSED_COLLEGE_JSON = os.path.join(BASE_DIR, "data", "processed", "college_documents.json")


# ═══════════════════════════════════════════════════════════
# 1. RAW DATASET VALIDATION
# ═══════════════════════════════════════════════════════════

def test_raw_dataset_exists_and_row_count():
    """Verify college_branches_rows.csv exists and has exactly 3,516 rows and 10 columns."""
    assert os.path.exists(RAW_CSV_PATH), f"Raw dataset missing at {RAW_CSV_PATH}"
    df = pd.read_csv(RAW_CSV_PATH)
    assert len(df) == 3516, f"Expected 3516 rows, got {len(df)}"
    assert len(df.columns) == 10, f"Expected 10 columns, got {len(df.columns)}"


def test_raw_dataset_schema_columns():
    """Verify new column names are present in college_branches_rows.csv."""
    df = pd.read_csv(RAW_CSV_PATH)
    expected_columns = {
        "id",
        "tnea_code",
        "sl_no",
        "department_code",
        "department_name",
        "approved_intake",
        "year_of_starting",
        "nba_accredited",
        "accreditation_valid_upto",
        "approval_marker"
    }
    assert expected_columns.issubset(set(df.columns)), f"Missing columns: {expected_columns - set(df.columns)}"


def test_raw_dataset_critical_fields_no_nulls():
    """Verify critical fields have zero nulls or empties."""
    df = pd.read_csv(RAW_CSV_PATH)
    assert df["id"].isnull().sum() == 0, "Null IDs found"
    assert df["tnea_code"].isnull().sum() == 0, "Null tnea_codes found"
    assert df["department_code"].isnull().sum() == 0, "Null department_codes found"
    assert df["department_name"].isnull().sum() == 0, "Null department_names found"
    assert df["approved_intake"].isnull().sum() == 0, "Null approved_intake found"


# ═══════════════════════════════════════════════════════════
# 2. PROCESSED CHUNK & METADATA FORMATTING TESTS
# ═══════════════════════════════════════════════════════════

def test_processed_branch_documents_exist():
    """Verify branch_documents.json exists and contains 3516 documents."""
    assert os.path.exists(PROCESSED_BRANCH_JSON), f"Missing {PROCESSED_BRANCH_JSON}"
    with open(PROCESSED_BRANCH_JSON, "r", encoding="utf-8") as f:
        branch_docs = json.load(f)
    assert len(branch_docs) == 3516, f"Expected 3516 branch documents, got {len(branch_docs)}"


def test_chunk_text_cleanliness_no_nan_or_none():
    """Verify no 'nan', 'None', or 'NaN' strings leaked into branch chunk content."""
    with open(PROCESSED_BRANCH_JSON, "r", encoding="utf-8") as f:
        branch_docs = json.load(f)

    for doc in branch_docs:
        content = doc["content"]
        # Words must not contain literal NaN or None
        words = content.split()
        for bad_word in ["None", "nan", "NaN", "null"]:
            assert bad_word not in words, f"Leaked '{bad_word}' in content: {content}"


def test_chunk_preferred_format_structure():
    """Verify chunks start with 'TNEA Code: {tnea_code} offers {department_name} ({department_code}).'"""
    with open(PROCESSED_BRANCH_JSON, "r", encoding="utf-8") as f:
        branch_docs = json.load(f)

    first_doc = branch_docs[0]
    meta = first_doc["metadata"]
    expected_prefix = f"TNEA Code: {meta['tnea_code']} offers {meta['department_name']} ({meta['department_code']})."
    assert first_doc["content"].startswith(expected_prefix)
    assert f"Approved Intake: {meta['approved_intake']}." in first_doc["content"]


def test_branch_metadata_backward_compatibility():
    """Verify branch metadata contains both new department fields and backward-compatible branch fields."""
    with open(PROCESSED_BRANCH_JSON, "r", encoding="utf-8") as f:
        branch_docs = json.load(f)

    sample = branch_docs[0]["metadata"]
    assert "department_code" in sample
    assert "department_name" in sample
    assert "approval_marker" in sample
    assert "branch_code" in sample  # backward compatibility
    assert "approval_note" in sample  # backward compatibility
    assert sample["branch_code"] == sample["department_code"]
    assert sample["doc_type"] == "branch_info"


def test_denormalized_college_metadata_contains_both_schemas():
    """Verify denormalized college documents contain both department_codes and branch_codes."""
    with open(PROCESSED_COLLEGE_JSON, "r", encoding="utf-8") as f:
        college_docs = json.load(f)

    assert len(college_docs) >= 400
    for doc in college_docs[:20]:
        meta = doc["metadata"]
        assert "department_codes" in meta
        assert "branch_codes" in meta
        assert "department_names" in meta
        assert "department_intakes" in meta
        assert "branch_intakes" in meta
        assert meta["department_codes"] == meta["branch_codes"]
        assert meta["department_intakes"] == meta["branch_intakes"]


# ═══════════════════════════════════════════════════════════
# 3. RETRIEVAL & FILTER INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════

def test_filter_by_department_code():
    """Test get_colleges_by_filters with department_code parameter."""
    colleges_by_dept = get_colleges_by_filters(district="Coimbatore", department_code="CS", limit=5)
    assert len(colleges_by_dept) > 0, "No colleges returned for Coimbatore CS using department_code"
    for c in colleges_by_dept:
        meta = c["metadata"]
        assert meta["district"] == "Coimbatore"
        assert "CS" in meta.get("department_codes", [])


def test_filter_backward_compatibility_branch_code():
    """Test get_colleges_by_filters gives identical results using branch_code vs department_code."""
    by_dept = get_colleges_by_filters(district="Chennai", department_code="EC", limit=5)
    by_branch = get_colleges_by_filters(district="Chennai", branch_code="EC", limit=5)
    codes_dept = [d["metadata"]["tnea_code"] for d in by_dept]
    codes_branch = [d["metadata"]["tnea_code"] for d in by_branch]
    assert codes_dept == codes_branch, f"Results mismatch: {codes_dept} vs {codes_branch}"


def test_extract_branch_code_from_department_filter():
    """Test extract_branch_code reads department_code from filters dict."""
    extracted = extract_branch_code("tell me about colleges", filters={"department_code": "ad"})
    assert extracted == "AD"


def test_normalize_filters_includes_department_code():
    """Test _normalize_filters upper-cases department_code."""
    normalized = _normalize_filters({"department_code": "me", "district": "salem"})
    assert normalized["department_code"] == "ME"
    assert normalized["district"] == "Salem"


def test_retrieve_query_with_department_code_filter():
    """Test full retrieve function with department_code in filters dict."""
    docs, ctx = retrieve("engineering colleges in Salem", filters={"department_code": "CS"}, top_k=3)
    assert docs is not None
    assert len(docs) > 0
    for d in docs:
        meta = d["metadata"]
        assert meta["district"].lower() == "salem"
        assert "CS" in meta.get("department_codes", meta.get("branch_codes", []))
