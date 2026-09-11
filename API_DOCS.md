# 🎓 TN Engineering AI Counselor — API Documentation

**Base URL:** `http://127.0.0.1:8000` (local) or `[DEPLOYED_URL]` (production)
**Swagger UI:** `{BASE_URL}/docs` — Open this to test all endpoints live!

---

## 1. Ask a Question (Main Chat Endpoint)

**POST** `/query`

### Request Body:
{
  "session_id": "user_12345",       // REQUIRED: Unique ID per user (use UUID)
  "question": "Tell me about PSG College of Technology",  // REQUIRED
  "top_k": 5,                       // OPTIONAL: Number of sources (default: 5)
  "filters": {}                     // OPTIONAL: e.g. {"district": "CHENNAI"}
}

### Success Response (200 OK):
{
  "answer": "### 🏛️ College Overview\n**PSG College of Technology**...",
  "sources": [
    {
      "college_name": "PSG College of Technology",
      "tnea_code": "2006",
      "district": "COIMBATORE",
      "score": 8.45
    }
  ]
}

### ⚠️ Important for Frontend:
- `answer` is **Markdown**. Render it using `react-markdown` or `marked`.
- If `answer` contains "don't have enough specific information", 
  hide the sources panel and show a friendly fallback message.
- If `sources` array is empty or all scores < 0, hide source cards.

---

## 2. Clear Chat History

**POST** `/clear_chat/{session_id}`

### Success Response (200 OK):
{
  "status": "cleared",
  "session_id": "user_12345"
}

---

## 3. Health Check

**GET** `/health`

### Success Response (200 OK):
{
  "status": "healthy",
  "version": "1.0-final"
}

---

## 4. Error Response Format

If the API fails, you will ALWAYS get clean JSON (never HTML):

{
  "error": "internal_server_error",
  "message": "The AI counselor is currently experiencing technical difficulties. Please try again in a few seconds."
}

---

## 📊 Source Score Interpretation

| Score Range | Meaning | UI Action |
|-------------|---------|-----------|
| > 5.0 | Highly Relevant | Show with green ✅ badge |
| 2.0 - 5.0 | Relevant | Show normally |
| 0.0 - 2.0 | Weak | Show collapsed |
| < 0.0 | Irrelevant | Hide sources entirely |

---

## 💡 Frontend Tips

1. **Session Management:** Generate a UUID per user. Keep it in localStorage.
2. **Loading State:** `/query` takes 3-8 seconds. Show a typing indicator.
3. **Clear Button:** Call `/clear_chat/{session_id}` when user clicks "New Chat".
4. **Health Check:** Call `/health` on page load to verify API is online.