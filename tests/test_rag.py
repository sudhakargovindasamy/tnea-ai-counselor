"""
tests/test_rag.py
Unit and Regression Tests for TNEA RAG Retrieval Engine and Student Test Cases.
"""

import os
import sys
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from app.services.retrieval import (
    retrieve,
    deduplicate_docs,
    get_colleges_by_filters,
    extract_district,
    extract_branch_code,
    fuzzy_resolve_college
)


# ═══════════════════════════════════════════════════════════
# STUDENT TEST CASES (ST-001 through ST-008)
# ═══════════════════════════════════════════════════════════

def test_st_001_coimbatore_cs():
    """ST-001: Returns 3-5 distinct colleges in Coimbatore with branch_code='CS'."""
    docs, ctx = retrieve("Which colleges in Coimbatore offer computer science courses?", top_k=5)
    assert docs is not None, "No documents returned for Coimbatore CS query"
    assert len(docs) >= 3, f"Expected at least 3 colleges, got {len(docs)}"
    tnea_codes = [d["metadata"]["tnea_code"] for d in docs]
    assert len(tnea_codes) == len(set(tnea_codes)), f"Duplicate tnea_code found: {tnea_codes}"
    for d in docs:
        assert d["metadata"]["district"] == "Coimbatore", \
            f"Expected district 'Coimbatore', got '{d['metadata']['district']}' for {d['metadata'].get('college_name')}"
        assert "CS" in d["metadata"]["branch_codes"], \
            f"'CS' not in branch_codes for {d['metadata'].get('college_name')}"


def test_st_002_coimbatore_hostel():
    """ST-002: Returns colleges in Coimbatore where hostel data is not empty."""
    docs, ctx = retrieve("Which colleges in Coimbatore have hostel facilities?", top_k=5)
    assert docs is not None, "No documents returned for Coimbatore hostel query"
    assert len(docs) >= 3, f"Expected at least 3 colleges with hostel, got {len(docs)}"
    for d in docs:
        m = d["metadata"]
        assert m["district"] == "Coimbatore", \
            f"Expected district 'Coimbatore', got '{m['district']}'"
        has_hostel_data = bool(
            m.get("hostel_facilities_boys") or
            m.get("hostel_facilities_girls") or
            m.get("mess_bill_boys") or
            m.get("mess_bill_girls")
        )
        assert has_hostel_data, \
            f"Missing hostel data in metadata for {m.get('college_name')} (TNEA: {m.get('tnea_code')})"


def test_st_003_salem_case_insensitivity():
    """ST-003: Returns all colleges with district='Salem' across case variations."""
    queries = [
        "What engineering colleges are in Salem?",
        "What engineering colleges are in SALEM?",
        "what engineering colleges are in salem?"
    ]
    for q in queries:
        docs, ctx = retrieve(q, top_k=5)
        assert docs is not None, f"No documents returned for query: '{q}'"
        assert len(docs) >= 3, f"Expected at least 3 Salem colleges for '{q}', got {len(docs)}"
        for d in docs:
            assert d["metadata"]["district"].lower() == "salem", \
                f"Expected district 'Salem', got '{d['metadata']['district']}' for query '{q}'"


def test_st_004_bannari_amman_ad():
    """ST-004: Resolves Bannari Amman (2702) and maps AI & DS to AD with intake 360."""
    docs, ctx = retrieve(
        "Does Bannari Amman Institute of Technology offer Artificial Intelligence and Data Science?",
        top_k=3
    )
    assert docs is not None, "No documents returned for Bannari Amman query"
    assert len(docs) >= 1, "Expected at least 1 document for Bannari Amman"
    top_meta = docs[0]["metadata"]
    assert str(top_meta["tnea_code"]) == "2702", \
        f"Expected tnea_code '2702', got '{top_meta['tnea_code']}'"
    assert "AD" in top_meta["branch_codes"], \
        f"'AD' not in branch_codes: {top_meta['branch_codes']}"
    assert top_meta.get("branch_intakes", {}).get("AD") == 360, \
        f"Expected AD intake 360, got {top_meta.get('branch_intakes', {}).get('AD')}"


def test_st_005_cit_mechanical():
    """ST-005: Resolves CIT (2007) and finds Mechanical Engineering (ME)."""
    docs, ctx = retrieve(
        "What mechanical engineering courses are available at Coimbatore Institute of Technology?",
        top_k=3
    )
    assert docs is not None, "No documents returned for CIT query"
    assert len(docs) >= 1, "Expected at least 1 document for CIT"
    top_meta = docs[0]["metadata"]
    assert str(top_meta["tnea_code"]) == "2007", \
        f"Expected tnea_code '2007', got '{top_meta['tnea_code']}'"
    assert "ME" in top_meta["branch_codes"], \
        f"'ME' not in branch_codes for CIT: {top_meta['branch_codes']}"


def test_st_006_kumaraguru_no_cyber():
    """ST-006: Resolves Kumaraguru (2712) and verifies Cyber Security is absent."""
    docs, ctx = retrieve(
        "Does Kumaraguru College of Technology offer a Cyber Security specialization?",
        top_k=3
    )
    assert docs is not None, "No documents returned for Kumaraguru query"
    assert len(docs) >= 1, "Expected at least 1 document for Kumaraguru"
    top_meta = docs[0]["metadata"]
    assert str(top_meta["tnea_code"]) == "2712", \
        f"Expected tnea_code '2712', got '{top_meta['tnea_code']}'"
    b_codes = top_meta["branch_codes"]
    assert "CY" not in b_codes, \
        f"Kumaraguru should NOT have 'CY' (Cyber Security) in branches: {b_codes}"
    assert "SC" not in b_codes, \
        f"Kumaraguru should NOT have 'SC' in branches: {b_codes}"


def test_st_007_cutoff_graceful_refusal():
    """ST-007: Cutoff query gracefully declines without attempting vector search."""
    docs, msg = retrieve("Which college can I get with a cutoff of 185 in Coimbatore?", top_k=5)
    assert docs is None, "Cutoff query should return None docs"
    assert msg is not None, "Cutoff query should return a refusal message"
    assert "cutoff" in msg.lower(), "Refusal message should mention 'cutoff'"
    assert "tneaonline.org" in msg.lower(), "Refusal message should redirect to tneaonline.org"


def test_st_008_tce_mess_bill():
    """ST-008: Returns exact mess bill ₹3,200 for boys at TCE (5008)."""
    docs, ctx = retrieve(
        "What is the mess bill for boys at Thiagarajar College of Engineering?",
        top_k=3
    )
    assert docs is not None, "No documents returned for TCE query"
    assert len(docs) >= 1, "Expected at least 1 document for TCE"
    top_meta = docs[0]["metadata"]
    assert str(top_meta["tnea_code"]) == "5008", \
        f"Expected tnea_code '5008', got '{top_meta['tnea_code']}'"
    assert top_meta.get("mess_bill_boys") == 3200.0, \
        f"Expected mess_bill_boys=3200.0, got {top_meta.get('mess_bill_boys')}"


# ═══════════════════════════════════════════════════════════
# UNIT TESTS: Deduplication, Branch Synonyms, Districts
# ═══════════════════════════════════════════════════════════

def test_deduplicate_docs():
    """Verifies deduplication by tnea_code."""
    docs = [
        {"metadata": {"tnea_code": "1001", "name": "A"}, "content": "Sample content 1"},
        {"metadata": {"tnea_code": "1001", "name": "A duplicate"}, "content": "Sample content 1"},
        {"metadata": {"tnea_code": "1002", "name": "B"}, "content": "Sample content 2"},
    ]
    deduped = deduplicate_docs(docs)
    assert len(deduped) == 2, f"Expected 2 unique docs, got {len(deduped)}"
    assert [d["metadata"]["tnea_code"] for d in deduped] == ["1001", "1002"]


def test_deduplicate_docs_empty():
    """Verifies deduplication handles empty list."""
    assert deduplicate_docs([]) == []
    assert deduplicate_docs(None) == []


def test_branch_synonym_extraction():
    """Verifies that extract_branch_code maps synonyms to canonical TNEA codes."""
    # Computer Science variants
    assert extract_branch_code("computer science and engineering") == "CS"
    assert extract_branch_code("cse") == "CS"
    assert extract_branch_code("cs") == "CS"

    # AI & Data Science variants
    assert extract_branch_code("ai & ds") == "AD"
    assert extract_branch_code("ai and ds") == "AD"
    assert extract_branch_code("artificial intelligence and data science") == "AD"

    # ✅ Cyber Security maps to 'CY' (correct TNEA code, NOT 'SC')
    assert extract_branch_code("cyber security") == "CY"
    assert extract_branch_code("cybersecurity") == "CY"

    # Mechanical
    assert extract_branch_code("mechanical engineering") == "ME"
    assert extract_branch_code("mech") == "ME"

    # Electronics and Communication
    assert extract_branch_code("ece") == "EC"
    assert extract_branch_code("electronics and communication") == "EC"

    # Electrical and Electronics
    assert extract_branch_code("eee") == "EE"

    # Civil
    assert extract_branch_code("civil") == "CE"
    assert extract_branch_code("civil engineering") == "CE"

    # Information Technology
    assert extract_branch_code("information technology") == "IT"

    # AI & ML
    assert extract_branch_code("ai & ml") == "AL"
    assert extract_branch_code("artificial intelligence and machine learning") == "AL"

    # No branch detected
    assert extract_branch_code("tell me about PSG college") is None


def test_district_extraction():
    """Verifies that extract_district normalizes district names correctly."""
    assert extract_district("colleges in coimbatore") == "Coimbatore"
    assert extract_district("colleges in COIMBATORE") == "Coimbatore"
    assert extract_district("engineering colleges in trichy") == "Tiruchirappalli"
    assert extract_district("colleges in chennai") == "Chennai"
    assert extract_district("colleges in salem") == "Salem"
    assert extract_district("colleges in kanchipuram") == "Kanchipuram"

    # No district detected
    assert extract_district("tell me about PSG") is None


def test_fuzzy_resolve_college_aliases():
    """Verifies alias resolution for common college shorthand."""
    assert fuzzy_resolve_college("psg") == "PSG College of Technology"
    assert fuzzy_resolve_college("ssn") == "SSN College of Engineering"
    assert fuzzy_resolve_college("tce") == "Thiagarajar College of Engineering"
    assert fuzzy_resolve_college("kct") == "Kumaraguru College of Technology"
    assert fuzzy_resolve_college("bitsathy") == "Bannari Amman Institute of Technology"
    assert fuzzy_resolve_college("cit") == "Coimbatore Institute of Technology"
    assert fuzzy_resolve_college("gct") == "Government College of Technology"


def test_catalog_filter_coimbatore_cs():
    """Verifies get_colleges_by_filters returns correct Coimbatore CS colleges."""
    docs = get_colleges_by_filters(district="Coimbatore", branch_code="CS", limit=5)
    assert len(docs) >= 3, f"Expected at least 3 Coimbatore CS colleges, got {len(docs)}"
    for d in docs:
        m = d["metadata"]
        assert m["district"] == "Coimbatore"
        assert "CS" in m["branch_codes"]