import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.retrieval import retrieve

def test_regional_placement_queries():
    print("=" * 70)
    print("🧪 RUNNING REGIONAL PLACEMENT & DISTRICT SUITE")
    print("=" * 70)

    # 1. User's exact query
    q1 = "How is placement across erode region"
    docs, ctx = retrieve(q1)
    assert docs is not None, "Expected docs for Erode placement query"
    codes = [d["metadata"]["tnea_code"] for d in docs]
    assert "2711" in codes, f"Kongu Engineering College (2711) must be in {codes}"
    assert "2702" in codes, f"Bannari Amman (2702) must be in {codes}"
    print(f"✅ Test 1 Passed: '{q1}' returned {len(docs)} colleges including Kongu (2711) and Bannari Amman (2702)!")

    # 2. Variation: "How are placements in erode"
    q2 = "How are placements in erode"
    docs, ctx = retrieve(q2)
    assert docs is not None
    codes = [d["metadata"]["tnea_code"] for d in docs]
    assert "2711" in codes and "2702" in codes
    print(f"✅ Test 2 Passed: '{q2}' returned {len(docs)} colleges including Kongu and Bannari Amman!")

    # 3. Complete district catalog: "Colleges in erode"
    q3 = "Colleges in erode"
    docs, ctx = retrieve(q3)
    assert docs is not None
    assert len(docs) == 14, f"Expected all 14 Erode colleges, got {len(docs)}"
    print(f"✅ Test 3 Passed: '{q3}' returned all 14 colleges in Erode district!")

    # 4. Top-K ranking: "Top 5 colleges in erode"
    q4 = "Top 5 colleges in erode"
    docs, ctx = retrieve(q4)
    assert docs is not None
    assert len(docs) == 5, f"Expected 5 colleges, got {len(docs)}"
    codes = [d["metadata"]["tnea_code"] for d in docs]
    assert "2702" in codes and "2711" in codes, "Top 5 in Erode must include Bannari Amman and Kongu"
    print(f"✅ Test 4 Passed: '{q4}' returned exactly top 5 colleges with premier institutions!")

    # 5. Out of scope non-TN query
    q5 = "What are the engineering colleges in Bangalore?"
    docs, ctx = retrieve(q5)
    assert docs is None
    assert "only contains TNEA-approved engineering colleges within Tamil Nadu" in ctx
    print(f"✅ Test 5 Passed: '{q5}' correctly refused with official non-TN boundary statement!")

    # 6. Salem placements
    q6 = "Placement in salem colleges"
    docs, ctx = retrieve(q6)
    assert docs is not None
    assert len(docs) >= 15
    print(f"✅ Test 6 Passed: '{q6}' returned {len(docs)} colleges in Salem district!")

    print("\n🎉 ALL 6 REGIONAL & OUT-OF-SCOPE TESTS PASSED PERFECTLY!\n")

if __name__ == "__main__":
    test_regional_placement_queries()

