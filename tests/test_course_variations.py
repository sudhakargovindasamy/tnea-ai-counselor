import unittest
import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.retrieval import retrieve, extract_branch_code
from app.services.query_understanding import understand_query

class TestCourseVariationsAndAggregations(unittest.TestCase):
    
    def test_cse_spelling_and_abbreviations(self):
        variations = [
            "what are the total number of collages offer cse ?",
            "what are the total number of collages offer computer science eng?",
            "what are the total number of collages offer computer science engg?",
            "what are the total number of colleges offer computer science engineering?",
            "how many colleges offer computer science eng",
            "total seats in computer science eng",
            "sum total seats in cse"
        ]
        for q in variations:
            with self.subTest(query=q):
                b_code = extract_branch_code(q, {})
                self.assertEqual(b_code, "CS", f"Failed to extract CS for query: {q}")
                
                u = understand_query(q)
                self.assertTrue(u.get("is_aggregation"), f"Expected is_aggregation=True for: {q}")
                self.assertEqual(u.get("branch_code"), "CS", f"Expected branch_code=CS in query understanding for: {q}")
                
                docs, text = retrieve(q, filters=u)
                self.assertIsNotNone(text, f"Expected non-null response text for: {q}")
                self.assertIn("55,255 approved seats", text, f"Expected 55,255 seats in response for: {q}")
                self.assertIn("408 colleges", text, f"Expected 408 colleges in response for: {q}")

    def test_other_branches_with_eng_abbreviations(self):
        branch_tests = [
            ("total seats in mechanical eng", "ME", 21519),
            ("how many colleges offer civil eng", "CE", 13745),
            ("total number of colleges offering electrical eng", "EE", 22895),
        ]
        for q, expected_branch, min_seats in branch_tests:
            with self.subTest(query=q):
                b_code = extract_branch_code(q, {})
                self.assertEqual(b_code, expected_branch, f"Failed branch extraction for {q}")
                
                u = understand_query(q)
                self.assertTrue(u.get("is_aggregation"), f"Expected is_aggregation for {q}")
                
                docs, text = retrieve(q, filters=u)
                self.assertIsNotNone(text)
                self.assertIn(f"approved seats", text)

    def test_college_entities_not_accidentally_matched_by_course_words(self):
        # Queries with course words must NOT trigger PSN Institute of Technology and Science
        queries = [
            "what are the total number of collages offer computer science eng?",
            "colleges offering computer science",
            "how many seats in data science"
        ]
        for q in queries:
            with self.subTest(query=q):
                u = understand_query(q)
                # college_name should NOT be set to PSN
                self.assertNotEqual(u.get("college_name"), "PSN Institute of Technology and Science (Autonomous) ,Melathediyoor, Tirunelveli District 627 152")

    def test_legitimate_college_lookup_still_works(self):
        # Legitimate college queries must still resolve correctly
        colleges = [
            ("tell me about sairam engineering college", "1419"),
            ("tell me about saveetha engineering college", "1216"),
            ("tell me about kongu engineering college", "2711"),
            ("tell me about bannari amman", "2702")
        ]
        for q, expected_code in colleges:
            with self.subTest(query=q):
                docs, text = retrieve(q)
                self.assertIsNotNone(docs)
                self.assertTrue(len(docs) >= 1)
                meta = docs[0].get("metadata", {})
                self.assertEqual(str(meta.get("tnea_code")), expected_code)

if __name__ == "__main__":
    unittest.main()

