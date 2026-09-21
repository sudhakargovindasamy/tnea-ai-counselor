# 🎓 TNEA Counselor AI — Frontend Integration Guide

This is the working reference for wiring the frontend to every backend endpoint. It covers what to call, when to call it, exact request/response shapes, and how to handle errors and edge cases.

**Base URL:** `http://127.0.0.1:8000` (local) or your deployed Render/HF Spaces URL.
**Live API docs (Swagger):** `{BASE_URL}/docs`

---

## 0. Session ID — do this first

Every conversational call (`/query`, `/chat`, `/clear_chat`) needs a `session_id`. Generate one UUID per user/browser tab and keep it in `localStorage` so the backend's conversational memory works across page reloads.

```js
function getSessionId() {
  let id = localStorage.getItem("tnea_session_id");
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem("tnea_session_id", id);
  }
  return id;
}
```

---

## 1. `GET /health` — call on app load

Use this to show an "API online / offline" indicator before the user starts chatting.

**Response (200):**
```json
{
  "status": "healthy",
  "version": "2.0-production",
  "features": ["l1_cache", "l2_semantic_cache", "sse_streaming", "slowapi_rate_limit", "observability_tracing"],
  "timestamp": "2026-09-17T10:00:00.000000"
}
```
No auth, no rate limit. If this fails (network error / non-200), show a banner: *"Counselor is waking up, please wait a moment"* — see the `/warmup` section below, since this usually means the free-tier instance is asleep.

---

## 2. `GET /warmup` — call right after `/health` on cold start

The embedding + reranker models are loaded **lazily** on the first real query, which makes that first user question slow (30–45s on Render free tier). Call `/warmup` proactively (e.g. right when the chat UI mounts) so the models are already loaded before the user finishes typing.

**Response (200):**
```json
{
  "status": "warmed",
  "message": "Models loaded successfully. Ready for queries.",
  "timestamp": "2026-09-17T10:00:03.000000"
}
```
**Response (error, still 200 status):**
```json
{ "status": "error", "message": "<exception text>" }
```
Fire-and-forget this call — don't block the UI on it, and don't show its result to the user. It's purely a latency optimization.

---

## 3. `POST /query` — non-streaming chat (simplest option)

Use this if you don't need token-by-token streaming — e.g. a simple chat widget, or while you build the UI before adding SSE.

**Rate limit:** 10 requests/minute per IP (`429` if exceeded — see Error Handling).

**Request body:**
```json
{
  "session_id": "abc-123-uuid",
  "question": "Which colleges in Coimbatore offer Computer Science?",
  "top_k": 5,
  "filters": {}
}
```
| Field | Required | Notes |
|---|---|---|
| `session_id` | ✅ | string, from step 0 |
| `question` | ✅ | string |
| `top_k` | ❌ | int, 1–20, default 5. **Sending 0, negative, or >20 returns HTTP 422** |
| `filters` | ❌ | object, e.g. `{"district": "COIMBATORE", "department_code": "CS"}` (both `department_code` and `branch_code` are supported; backend normalizes casing) |

**Success response (200):**
```json
{
  "answer": "### 🏛️ College Overview\n**PSG College of Technology**...",
  "sources": [
    { "college_name": "PSG College of Technology", "tnea_code": "2744", "district": "Coimbatore", "score": 0.9123 }
  ]
}
```

**Frontend handling rules:**
- `answer` is **Markdown** — render with `react-markdown` / `marked`, don't display raw.
- If `answer` contains the phrase "don't have enough specific information" (or similar refusal), hide the sources panel and show a friendly fallback instead of an empty/awkward source list.
- If `sources` is empty, hide the source-cards panel entirely.
- Typical latency is 3–8 seconds on a cache miss — show a typing/loading indicator.
- On a cache hit, response is near-instant.

```js
async function askQuery(question, topK = 5, filters = {}) {
  const res = await fetch(`${BASE_URL}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: getSessionId(), question, top_k: topK, filters }),
  });
  if (!res.ok) throw await handleApiError(res);
  return res.json(); // { answer, sources }
}
```

---

## 4. `POST /chat` — streaming chat (recommended for the real chat UI)

Same inputs as `/query`, but streams the answer via **Server-Sent Events** for a "typing" effect, and is rate-limited (10/min/IP).

**Request body:** identical to `/query`, plus an optional `stream` flag (defaults to `true` already, you don't need to set it):
```json
{ "session_id": "abc-123-uuid", "question": "Tell me about PSG College", "top_k": 5 }
```

**Response:** `Content-Type: text/event-stream`. Each line is `data: {...}\n\n`. Three event shapes appear, in order:
1. Zero or more: `{"token": "some text "}` — append to the answer as it streams in.
2. Exactly one: `{"sources": [ {...}, ... ]}` — the final citation list, sent after the answer is complete.
3. Terminator: `[DONE]` (literal string, not JSON) — stop listening.

An error mid-stream looks like `{"error": "..."}` followed by `[DONE]`.

**Browser fetch + ReadableStream implementation** (native `EventSource` doesn't support POST bodies, so use `fetch`):
```js
async function streamChat(question, { onToken, onSources, onError, onDone }) {
  const res = await fetch(`${BASE_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: getSessionId(), question, top_k: 5 }),
  });

  if (!res.ok) { onError(await handleApiError(res)); return; }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split("\n\n");
    buffer = lines.pop(); // keep incomplete chunk for next read

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const payload = line.slice(6).trim();

      if (payload === "[DONE]") { onDone(); return; }

      try {
        const data = JSON.parse(payload);
        if (data.token) onToken(data.token);
        if (data.sources) onSources(data.sources);
        if (data.error) onError(new Error(data.error));
      } catch (e) {
        console.error("Bad SSE payload:", payload);
      }
    }
  }
}
```

**Frontend handling rules:**
- Append each `token` chunk directly to the message bubble as it arrives — don't wait for `[DONE]`.
- Only render source cards once the `sources` event fires (it's always last, right before `[DONE]`).
- Because of the 10/min rate limit, disable the send button while a stream is in flight and show a cooldown message if the user hits `429`.
- If the network drops mid-stream, treat it the same as `onError` and offer a retry.

---

## 5. `GET /search_colleges` — direct catalog browsing (no AI/LLM call)

Use this for a filter sidebar / browse-by-criteria UI (district dropdown, branch picker, hostel checkbox) — it's a plain DB lookup, so it's fast and doesn't count against the chat rate limit.

**Query params (all optional except none are required):**
| Param | Type | Example |
|---|---|---|
| `district` | string | `Coimbatore` |
| `branch_code` | string | `CSE` |
| `has_hostel` | bool | `true` |
| `autonomous` | bool | `true` |
| `limit` | int | `10` (default) |

```js
const res = await fetch(`${BASE_URL}/search_colleges?district=Coimbatore&branch_code=CSE&limit=20`);
const { count, results } = await res.json();
```

**Response (200):**
```json
{ "count": 3, "results": [ { "college_name": "...", "tnea_code": "...", "district": "...", ... } ] }
```
Render `results` as a plain list/table — this is metadata only, no `answer` or `sources` fields.

---

## 6. `POST /clear_chat/{session_id}` — "New Chat" button

Call when the user clicks "New conversation" / "Clear chat" so the backend's memory of prior turns is wiped (otherwise pronoun resolution like "tell me more about **it**" will reference the old topic).

```js
async function clearChat() {
  const sid = getSessionId();
  await fetch(`${BASE_URL}/clear_chat/${sid}`, { method: "POST" });
  // Also clear the local message list in your UI state
}
```
**Response (200):** `{ "status": "cleared", "session_id": "abc-123-uuid" }`

---

## 7. `POST /feedback/downvote` — "thumbs down" button

Attach to a thumbs-down icon under each AI answer. It deletes that exact question's cached answer so the next person gets a freshly generated (hopefully better) response instead of the same bad cached one.

**Request:** `question` is sent as a query param (not JSON body):
```js
await fetch(`${BASE_URL}/feedback/downvote?question=${encodeURIComponent(question)}`, { method: "POST" });
```
**Response (200):** `{ "status": "success", "message": "Feedback recorded. Cache cleared for this question." }`
This doesn't need `session_id` — it targets the cache entry for that literal question text, not a specific user's turn. Show a quick "Thanks, noted" toast; don't block on the response.

---

## 8. `POST /admin/purge_cache` — admin/internal only, do NOT expose in the public UI

Wipes the entire semantic cache. Only call this from an internal admin panel, never from the public-facing chat UI, since it requires the `admin_secret`.

```js
await fetch(`${BASE_URL}/admin/purge_cache?admin_secret=YOUR_SECRET`, { method: "POST" });
```
- Correct secret → `200 { "status": "success", ... }`
- Wrong/missing secret → `403 { "detail": "Unauthorized" }`

**⚠️ Never hardcode `admin_secret` in frontend JS bundles** — it would be visible to anyone who opens devtools. Only call this endpoint from a server-side admin tool.

---

## 9. Error handling — apply this everywhere

| Status | Meaning | UI action |
|---|---|---|
| `422` | Validation error (e.g. bad `top_k`) | Show the field-level message from `detail` |
| `403` | Wrong admin secret | Admin panel only — show "unauthorized" |
| `429` | Rate limit hit (10/min on `/query` and `/chat`) | Show "Too many requests, please wait a minute" and disable input briefly |
| `500` | Server/LLM error | Show a generic friendly error, offer retry. Body is always clean JSON, never HTML |
| `504` | Request exceeded the server's 60s timeout | Suggest a simpler/shorter question and retry |
| network error / no response | API asleep or down | Show "Counselor is waking up" and retry `/health` every few seconds |

The `500` error body always looks like:
```json
{ "error": "internal_server_error", "message": "Server encountered an error: ..." }
```

```js
async function handleApiError(res) {
  let body;
  try { body = await res.json(); } catch { body = {}; }
  if (res.status === 429) return new Error("You're sending questions too fast — please wait a moment.");
  if (res.status === 422) return new Error(body.detail?.[0]?.msg || "Invalid request.");
  return new Error(body.message || body.detail || `Request failed (${res.status})`);
}
```

---

## 10. Score interpretation (for source cards)

| Score Range | Meaning | UI |
|---|---|---|
| > 0.8 | Highly relevant | Green ✅ badge |
| 0.5 – 0.8 | Relevant | Show normally |
| 0.2 – 0.5 | Weak | Collapsed by default |
| < 0.2 | Irrelevant | Hide the card |

*(Scores are normalized to 0–1 by the backend before being returned.)*

---

## Quick checklist for the frontend build

- [ ] Generate & persist `session_id` in `localStorage`
- [ ] Call `/health` on load; call `/warmup` right after
- [ ] Wire the chat box to `/chat` (streaming) — fall back to `/query` if you skip SSE for v1
- [ ] Render `answer` as Markdown
- [ ] Show source cards using the score table above, hide when empty
- [ ] "New chat" button → `/clear_chat/{session_id}`
- [ ] Thumbs-down icon → `/feedback/downvote`
- [ ] Disable send button + show cooldown on `429`
- [ ] Never call `/admin/purge_cache` from public frontend code
- [ ] Optional: district/branch filter sidebar → `/search_colleges`