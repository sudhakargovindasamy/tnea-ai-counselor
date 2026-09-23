import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.retrieval import retrieve

REGIONAL_TESTS = [
    # ── North Tamil Nadu ──
    {"query": "Colleges in Chennai", "min_docs": 10, "check": "Chennai"},
    {"query": "Top colleges in Kanchipuram", "min_docs": 20, "check": "Kanchipuram"},
    {"query": "Engineering colleges in Tiruvallur", "min_docs": 20, "check": "Tiruvallur"},
    {"query": "Colleges in Chengalpattu", "min_docs": 15, "check": "Chengalpattu"},
    {"query": "Colleges in Ranipet", "min_docs": 5, "check": "Ranipet"},
    {"query": "Colleges in Tirupattur", "min_docs": 3, "check": "Tirupattur"},
    {"query": "Colleges in Vellore", "min_docs": 4, "check": "Vellore"},
    {"query": "Colleges in Viluppuram", "min_docs": 8, "check": "Viluppuram"},
    {"query": "Colleges in Cuddalore", "min_docs": 6, "check": "Cuddalore"},
    {"query": "Colleges in Kallakurichi", "min_docs": 4, "check": "Kallakurichi"},
    {"query": "Colleges in Tiruvannamalai", "min_docs": 5, "check": "Tiruvannamalai"},

    # ── Kongu Belt / West TN ──
    {"query": "Colleges in Coimbatore", "min_docs": 20, "check": "Coimbatore"},
    {"query": "How is placement across erode region", "min_docs": 14, "check": "Erode"},
    {"query": "Engineering colleges in Tiruppur", "min_docs": 8, "check": "Tiruppur"},
    {"query": "Colleges in Salem", "min_docs": 15, "check": "Salem"},
    {"query": "Colleges in Namakkal", "min_docs": 15, "check": "Namakkal"},
    {"query": "Colleges in Dharmapuri", "min_docs": 4, "check": "Dharmapuri"},
    {"query": "Colleges in Krishnagiri", "min_docs": 4, "check": "Krishnagiri"},

    # ── Central TN & Delta Region ──
    {"query": "Colleges in Trichy", "min_docs": 15, "check": "Tiruchirappalli"},
    {"query": "Colleges in Tiruchirappalli", "min_docs": 15, "check": "Tiruchirappalli"},
    {"query": "Thanjavur engineering colleges", "min_docs": 8, "check": "Thanjavur"},
    {"query": "Colleges in Tiruvarur", "min_docs": 2, "check": "Tiruvarur"},
    {"query": "Colleges in Mayiladuthurai", "min_docs": 1, "check": "Mayiladuthurai"},
    {"query": "Colleges in Nagapattinam", "min_docs": 4, "check": "Nagapattinam"},
    {"query": "Colleges in Pudukkottai", "min_docs": 10, "check": "Pudukkottai"},
    {"query": "Colleges in Karur", "min_docs": 4, "check": "Karur"},
    {"query": "Colleges in Perambalur", "min_docs": 3, "check": "Perambalur"},
    {"query": "Colleges in Ariyalur", "min_docs": 3, "check": "Ariyalur"},

    # ── South Tamil Nadu ──
    {"query": "Colleges in Madurai", "min_docs": 10, "check": "Madurai"},
    {"query": "Colleges in Tirunelveli", "min_docs": 10, "check": "Tirunelveli"},
    {"query": "Thoothukudi engineering colleges", "min_docs": 10, "check": "Thoothukudi"},
    {"query": "Colleges in Kanyakumari", "min_docs": 20, "check": "Kanyakumari"},
    {"query": "Colleges in Dindigul", "min_docs": 8, "check": "Dindigul"},
    {"query": "Colleges in Tenkasi", "min_docs": 5, "check": "Tenkasi"},
    {"query": "Colleges in Virudhunagar", "min_docs": 7, "check": "Virudhunagar"},
    {"query": "Colleges in Ramanathapuram", "min_docs": 4, "check": "Ramanathapuram"},
    {"query": "Colleges in Theni", "min_docs": 4, "check": "Theni"},
    {"query": "Colleges in Sivagangai", "min_docs": 6, "check": "Sivagangai"},

    # ── Nilgiris ──
    {"query": "Colleges in Nilgiris", "min_docs": 1, "check": "The Nilgiris"},
    {"query": "Engineering colleges in Ooty", "min_docs": 1, "check": "The Nilgiris"},

    # ── Colloquial & Informal Names ──
    {"query": "Colleges in Kovai", "min_docs": 20, "check": "Coimbatore"},
    {"query": "Colleges in Nellai", "min_docs": 10, "check": "Tirunelveli"},
    {"query": "Colleges in Tuticorin", "min_docs": 10, "check": "Thoothukudi"},
    {"query": "Engineering in Tanjore", "min_docs": 8, "check": "Thanjavur"},
    {"query": "Colleges in Hosur", "min_docs": 4, "check": "Krishnagiri"},

    # ── Graceful Zero-Match Handling ──
    {"query": "Marine engineering colleges in Theni", "min_docs": 0, "expected_refusal": "no engineering colleges in Theni offering Marine Engineering"},
    {"query": "Mining engineering colleges in Tiruppur", "min_docs": 0, "expected_refusal": "no engineering colleges in Tiruppur offering Mining Engineering"},

    # ── Out of Scope Non-TN Boundaries ──
    {"query": "What are the engineering colleges in Bangalore?", "min_docs": 0, "expected_refusal": "only contains TNEA-approved engineering colleges within Tamil Nadu"},
    {"query": "Best colleges in Delhi", "min_docs": 0, "expected_refusal": "only contains TNEA-approved engineering colleges within Tamil Nadu"}
]

def run_all_regional_tests():
    print("=" * 80)
    print("🌍 TAMIL NADU ALL-REGION COMPREHENSIVE VALIDATION SUITE")
    print("=" * 80)
    passed = 0
    failed = 0

    for i, t in enumerate(REGIONAL_TESTS, 1):
        q = t["query"]
        docs, ctx = retrieve(q)

        if "expected_refusal" in t:
            if docs is None and t["expected_refusal"].lower() in str(ctx).lower():
                passed += 1
                print(f"✅ [{i:02d}/49] PASS (Refusal): '{q}' -> Refused accurately without error")
            else:
                failed += 1
                print(f"❌ [{i:02d}/49] FAIL: '{q}' -> Expected refusal containing '{t['expected_refusal']}', got docs={len(docs) if docs else 0}, msg='{ctx}'")
        else:
            if not docs:
                failed += 1
                print(f"❌ [{i:02d}/49] FAIL: '{q}' -> No docs returned! msg='{ctx}'")
                continue
            
            # Check district metadata of returned colleges
            check_norm = t["check"].lower().replace("the ", "").strip()
            m_count = sum(1 for d in docs if check_norm in d.get("metadata", {}).get("district", "").lower().replace("the ", ""))
            if len(docs) >= t["min_docs"] and m_count > 0:
                passed += 1
                print(f"✅ [{i:02d}/49] PASS ({len(docs):2d} docs): '{q}' -> Cleanly mapped to {t['check']}")
            else:
                failed += 1
                print(f"❌ [{i:02d}/49] FAIL: '{q}' -> Expected >={t['min_docs']} docs for {t['check']}, got {len(docs)} (matching district={m_count})")

    print("\n" + "=" * 80)
    print(f"FINAL RESULT: {passed}/{len(REGIONAL_TESTS)} Passed ({(passed/len(REGIONAL_TESTS))*100:.1f}%) | Failed: {failed}")
    print("=" * 80 + "\n")
    return failed == 0

if __name__ == "__main__":
    success = run_all_regional_tests()
    sys.exit(0 if success else 1)
