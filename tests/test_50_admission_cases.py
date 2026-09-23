import sys
import os
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.retrieval import retrieve

TEST_CASES = [
    # ── Section 0: Allocation of Seats (Reservation) ──
    {
        "id": 1,
        "query": "What is the reservation percentage for Scheduled Castes (SC) in TNEA?",
        "expected_sec": "Allocation of Seats",
        "keywords": ["15.00%", "sc"]
    },
    {
        "id": 2,
        "query": "What percentage of seats are allocated under Open Competition (OC)?",
        "expected_sec": "Allocation of Seats",
        "keywords": ["31.00%", "oc"]
    },
    {
        "id": 3,
        "query": "What is the quota percentage for MBC & DNC in TNEA counselling?",
        "expected_sec": "Allocation of Seats",
        "keywords": ["20.00%", "mbc"]
    },
    {
        "id": 4,
        "query": "How many percentage of seats are reserved for Backward Class Muslim (BCM)?",
        "expected_sec": "Allocation of Seats",
        "keywords": ["3.50%", "bcm"]
    },
    {
        "id": 5,
        "query": "What is the reservation percentage for Scheduled Caste Arunthathiyars (SCA)?",
        "expected_sec": "Allocation of Seats",
        "keywords": ["3.00%", "sca"]
    },
    {
        "id": 6,
        "query": "Explain the rule of reservation in TNEA seat allocation.",
        "expected_sec": "Allocation of Seats",
        "keywords": ["reservation", "31.00%"]
    },
    {
        "id": 7,
        "query": "What is the ST quota percentage in TNEA?",
        "expected_sec": "Allocation of Seats",
        "keywords": ["1.00%", "st"]
    },

    # ── Section 1: Tuition Fee Concession ──
    {
        "id": 8,
        "query": "What are the benefits under the 7.5% government school quota?",
        "expected_sec": "Tuition Fee Concession",
        "keywords": ["7 5 percent", "tuition fee"]
    },
    {
        "id": 9,
        "query": "Who is eligible for First Graduate tuition fee concession?",
        "expected_sec": "Tuition Fee Concession",
        "keywords": ["first graduate", "concession"]
    },
    {
        "id": 10,
        "query": "What is the annual income limit for AICTE Tuition Fee Waiver scheme?",
        "expected_sec": "Tuition Fee Concession",
        "keywords": ["aicte", "8,00,000"]
    },
    {
        "id": 11,
        "query": "How many seats are available under AICTE fee waiver scheme in each college?",
        "expected_sec": "Tuition Fee Concession",
        "keywords": ["5%", "sanctioned"]
    },
    {
        "id": 12,
        "query": "What is the income ceiling for Post-Matric Scholarship for SC/ST students?",
        "expected_sec": "Tuition Fee Concession",
        "keywords": ["post matric", "2,50,000"]
    },
    {
        "id": 13,
        "query": "Does the government pay hostel fee for 7.5% quota government school students?",
        "expected_sec": "Tuition Fee Concession",
        "keywords": ["hostel fee", "7 5 percent"]
    },
    {
        "id": 14,
        "query": "What is the fee waiver for first generation graduate?",
        "expected_sec": "Tuition Fee Concession",
        "keywords": ["first graduate", "concession"]
    },

    # ── Section 2: Procedure for Registration of Application ──
    {
        "id": 15,
        "query": "What is the registration fee for general category candidates in TNEA?",
        "expected_sec": "Procedure for Registration of Application",
        "keywords": ["500", "registration"]
    },
    {
        "id": 16,
        "query": "How much is the application fee for SC/SCA/ST candidates?",
        "expected_sec": "Procedure for Registration of Application",
        "keywords": ["250", "sc"]
    },
    {
        "id": 17,
        "query": "What are the official websites to register for TNEA application?",
        "expected_sec": "Procedure for Registration of Application",
        "keywords": ["tneaonline.org", "dte.tn.gov.in"]
    },
    {
        "id": 18,
        "query": "What is the registration procedure for TNEA?",
        "expected_sec": "Procedure for Registration of Application",
        "keywords": ["registration", "tneaonline.org"]
    },
    {
        "id": 19,
        "query": "How to register application for TNEA counselling online?",
        "expected_sec": "Procedure for Registration of Application",
        "keywords": ["tneaonline.org", "registration"]
    },
    {
        "id": 20,
        "query": "What is the cost of application for TNEA registration?",
        "expected_sec": "Procedure for Registration of Application",
        "keywords": ["500", "250"]
    },

    # ── Section 3: Uploading Copy of Original Certificates ──
    {
        "id": 21,
        "query": "What certificates are required to be uploaded during TNEA registration?",
        "expected_sec": "Uploading Copy of Original Certificates",
        "keywords": ["certificates", "marksheet"]
    },
    {
        "id": 22,
        "query": "What documents should be uploaded for first graduate fee concession?",
        "expected_sec": "Uploading Copy of Original Certificates",
        "keywords": ["first graduate", "joint declaration"]
    },
    {
        "id": 23,
        "query": "What are the documents to upload for TNEA application?",
        "expected_sec": "Uploading Copy of Original Certificates",
        "keywords": ["certificates", "hsc"]
    },
    {
        "id": 24,
        "query": "Which certificates are needed for ex-servicemen special reservation?",
        "expected_sec": ["Uploading Copy of Original Certificates", "Special Reservation Categories"],
        "keywords": ["ex-servicemen", "certificate"]
    },
    {
        "id": 25,
        "query": "What certificates must be uploaded for Differently Abled candidates?",
        "expected_sec": "Uploading Copy of Original Certificates",
        "keywords": ["differently abled", "medical"]
    },

    # ── Section 4: Online Choice Filling and Confirmation (Merit/Formula) ──
    {
        "id": 26,
        "query": "How is merit calculated out of 200 in TNEA counselling?",
        "expected_sec": "Online Choice Filling and Confirmation",
        "keywords": ["200", "mathematics"]
    },
    {
        "id": 27,
        "query": "What is the cutoff formula for engineering admission in Tamil Nadu?",
        "expected_sec": "Online Choice Filling and Confirmation",
        "keywords": ["physics", "chemistry", "mathematics"]
    },
    {
        "id": 28,
        "query": "How are Mathematics marks calculated in TNEA merit list?",
        "expected_sec": "Online Choice Filling and Confirmation",
        "keywords": ["100", "mathematics"]
    },
    {
        "id": 29,
        "query": "How many marks are allocated to Physics and Chemistry in TNEA cutoff calculation?",
        "expected_sec": "Online Choice Filling and Confirmation",
        "keywords": ["50", "physics", "chemistry"]
    },
    {
        "id": 30,
        "query": "What are the stages of online counselling in TNEA?",
        "expected_sec": ["Online Choice Filling and Confirmation", "Counselling Allotment and Confirmation"],
        "keywords": ["choice filling", "tentative allotment"]
    },
    {
        "id": 31,
        "query": "How is TNEA merit mark calculated out of 200?",
        "expected_sec": "Online Choice Filling and Confirmation",
        "keywords": ["200", "merit"]
    },

    # ── Section 5: Counselling Allotment and Confirmation (4 Options) ──
    {
        "id": 32,
        "query": "What does Accept and Join mean in TNEA allotment confirmation?",
        "expected_sec": "Counselling Allotment and Confirmation",
        "keywords": ["accept and join", "report"]
    },
    {
        "id": 33,
        "query": "Explain the Accept and Upward option during tentative allotment.",
        "expected_sec": "Counselling Allotment and Confirmation",
        "keywords": ["accept and upward", "tfc"]
    },
    {
        "id": 34,
        "query": "What happens if a candidate chooses Decline and Move to Next Round?",
        "expected_sec": "Counselling Allotment and Confirmation",
        "keywords": ["decline and move", "next round"]
    },
    {
        "id": 35,
        "query": "What is the Decline and Quit option in TNEA counselling?",
        "expected_sec": "Counselling Allotment and Confirmation",
        "keywords": ["decline and quit"]
    },
    {
        "id": 36,
        "query": "What are the 4 options given to a candidate during tentative allotment confirmation?",
        "expected_sec": "Counselling Allotment and Confirmation",
        "keywords": ["accept and join", "accept and upward"]
    },
    {
        "id": 37,
        "query": "What is the procedure for allotment confirmation in TNEA counselling?",
        "expected_sec": "Counselling Allotment and Confirmation",
        "keywords": ["confirmation", "allotment"]
    },
    {
        "id": 38,
        "query": "What happens if a candidate fails to report to the college after choosing Accept and Join?",
        "expected_sec": "Counselling Allotment and Confirmation",
        "keywords": ["cancellation", "accept and join"]
    },

    # ── Section 6: Special Reservation Categories ──
    {
        "id": 39,
        "query": "How many seats are reserved for children of ex-servicemen in self-financing colleges?",
        "expected_sec": "Special Reservation Categories",
        "keywords": ["ex servicemen", "1 seat"]
    },
    {
        "id": 40,
        "query": "What is the reservation percentage for Differently Abled persons in TNEA?",
        "expected_sec": "Special Reservation Categories",
        "keywords": ["5%", "disabilities"]
    },
    {
        "id": 41,
        "query": "How many seats are reserved for Eminent Sports persons in TNEA?",
        "expected_sec": "Special Reservation Categories",
        "keywords": ["sports", "6 seats"]
    },
    {
        "id": 42,
        "query": "Can candidates from other states apply under special reservation categories?",
        "expected_sec": "Special Reservation Categories",
        "keywords": ["tamil nadu", "native"]
    },
    {
        "id": 43,
        "query": "What are the rules for sports quota in TNEA?",
        "expected_sec": "Special Reservation Categories",
        "keywords": ["sports", "reservation"]
    },

    # ── Section 7: Course Specific Requirements (Marine Engineering) ──
    {
        "id": 44,
        "query": "What is the minimum 10+2 PCM average required for Marine Engineering?",
        "expected_sec": "Course Specific Requirements",
        "keywords": ["60%", "marine engineering"]
    },
    {
        "id": 45,
        "query": "What is the minimum English mark requirement for Marine Engineering?",
        "expected_sec": "Course Specific Requirements",
        "keywords": ["50%", "marine engineering"]
    },
    {
        "id": 46,
        "query": "Is IMU CET mandatory for Marine Engineering admission?",
        "expected_sec": "Course Specific Requirements",
        "keywords": ["imu cet", "marine engineering"]
    },
    {
        "id": 47,
        "query": "What is the maximum age limit for admission to Marine Engineering?",
        "expected_sec": "Course Specific Requirements",
        "keywords": ["25", "marine engineering"]
    },
    {
        "id": 48,
        "query": "What are the physical requirements for Marine Engineering regarding height and weight?",
        "expected_sec": "Course Specific Requirements",
        "keywords": ["157", "48", "marine engineering"]
    },

    # ── Section 8: Course Specific Requirements (Mining Engineering) ──
    {
        "id": 49,
        "query": "Can female candidates join Mining Engineering according to the Mines Act?",
        "expected_sec": "Course Specific Requirements",
        "keywords": ["mines act 1952", "below ground", "mining engineering"]
    },
    {
        "id": 50,
        "query": "What are the restrictions for women in Mining Engineering under Mines Act 1952?",
        "expected_sec": "Course Specific Requirements",
        "keywords": ["below ground", "women", "6 a.m.", "mining engineering"]
    },

    # ── Section 9: General Eligibility Rules & Nativity ──
    {
        "id": 51,
        "query": "Who needs to submit a Nativity Certificate in TNEA counselling?",
        "expected_sec": "General Eligibility Rules",
        "keywords": ["nativity certificate", "tamil nadu"]
    },
    {
        "id": 52,
        "query": "Do candidates who studied 8th to 12th in Tamil Nadu need a nativity certificate?",
        "expected_sec": "General Eligibility Rules",
        "keywords": ["nativity certificate", "viii"]
    },
    {
        "id": 53,
        "query": "Are children of Central Government employees eligible for TNEA without a nativity certificate?",
        "expected_sec": "General Eligibility Rules",
        "keywords": ["central government", "5 years"]
    },
    {
        "id": 54,
        "query": "What certificate is required for Sri Lankan Tamil refugees in TNEA?",
        "expected_sec": ["Uploading Copy of Original Certificates", "General Eligibility Rules"],
        "keywords": ["srilankan tamil refugee"]
    }
]

def run_tests():
    print(f"\n================================================================================")
    print(f"      TNEA ADMISSION BROCHURE EVALUATION SUITE: 54 TEST CASES                  ")
    print(f"================================================================================\n")
    
    passed = 0
    failed = 0
    total = len(TEST_CASES)
    start_total_time = time.time()
    
    for case in TEST_CASES:
        cid = case["id"]
        query = case["query"]
        expected_sec = case["expected_sec"]
        
        t0 = time.time()
        docs, context_text = retrieve(query, top_k=2)
        elapsed_ms = (time.time() - t0) * 1000
        
        if not docs:
            print(f"❌ [FAIL] Test {cid:02d}: No documents returned! Query: '{query}'")
            failed += 1
            continue
            
        top_doc = docs[0]
        actual_sec = top_doc.get("metadata", {}).get("section", "Unknown")
        content = (top_doc.get("content", "") + " " + (context_text or "")).lower()
        
        # Verify section match
        if isinstance(expected_sec, list):
            sec_match = any(
                (es.lower() in actual_sec.lower()) or (actual_sec.lower() in es.lower())
                for es in expected_sec
            )
        else:
            sec_match = (expected_sec.lower() in actual_sec.lower()) or (actual_sec.lower() in expected_sec.lower())
            
        keywords_match = all(k.lower() in content for k in case["keywords"])
        
        if sec_match and keywords_match:
            passed += 1
            print(f"✅ [PASS] Test {cid:02d} ({elapsed_ms:5.1f}ms): '{actual_sec}' -> {query[:55]}...")
        else:
            failed += 1
            print(f"❌ [FAIL] Test {cid:02d} ({elapsed_ms:5.1f}ms):")
            print(f"   Query:        '{query}'")
            print(f"   Expected Sec: '{expected_sec}'")
            print(f"   Actual Sec:   '{actual_sec}'")
            print(f"   Sec Match:    {sec_match}, Keywords Match: {keywords_match}")
            if not keywords_match:
                missing = [k for k in case["keywords"] if k.lower() not in content]
                print(f"   Missing Keys: {missing}")
            
    total_time = time.time() - start_total_time
    print(f"\n================================================================================")
    print(f"RESULTS: {passed}/{total} Passed ({(passed/total)*100:.1f}%) in {total_time:.2f}s | Failed: {failed}")
    print(f"================================================================================\n")
    return failed == 0

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
