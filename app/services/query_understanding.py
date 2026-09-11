import json
import re
import logging
from google import genai
from google.genai import errors
from app.config import GEMINI_API_KEY

logger = logging.getLogger(__name__)
client = genai.Client(api_key=GEMINI_API_KEY)

# 🚀 Production Fallback Chain (Prioritize working models)
FALLBACK_MODELS = [
    "gemini-3.5-flash",   # Primary: Confirmed working in your logs
    "gemini-3.6-flash",   # Secondary: Alternative working model
    "gemini-2.5-pro",     # Tertiary: Pro fallback
    "gemini-2.5-flash"    # Last resort
]

def understand_query(question: str) -> dict:
    prompt = f"""You are an expert query router for Tamil Nadu Engineering colleges.
Analyze the student's question and extract intent and filters.

Available intents:
- "compare" : user wants to compare two or more specific colleges side-by-side
- "numerical" : user asks for ranking/sorting by a number (highest/lowest/top placement, pass percentage, intake)
- "search" : default informational query

Filters to extract (only if mentioned):
- compare_colleges : list of college names when intent is "compare"
- college_name : single college name
- district : UPPERCASE district (e.g., CHENNAI, COIMBATORE)
- autonomous : "Yes" or "No"
- branch_code : CS, EC, ME, IT, AD, CE, EE (map "Computer Science"/"CSE"->CS, "Mechanical"->ME)
- nba_accredited : true or false
- numerical_metric : "placement" or "pass_percentage" or "total_intake" when intent is "numerical"

RULES:
1. If comparing colleges, set intent "compare" and list names in compare_colleges.
2. If ranking by a number, set intent "numerical" and set numerical_metric.
3. Return ONLY valid JSON. No markdown, no explanation.

Question: {question}
JSON:"""

    last_error = None
    
    for model in FALLBACK_MODELS:
        try:
            logger.info(f"🧠 Query Understanding: Trying {model}")
            response = client.models.generate_content(model=model, contents=prompt)
            text = response.text.strip()
            text = re.sub(r"^```json\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
            extracted = json.loads(text)
            logger.info(f"🧠 Extracted ({model}): {extracted}")
            return extracted
            
        except errors.ClientError as e:
            # 🚀 FIX: Handle BOTH 429 (rate limit) AND 404 (model not found) errors
            if e.code in [429, 404]:
                logger.warning(f"⚠️ {e.code} Error on {model}: {e.message}. Auto-switching to fallback...")
                last_error = e
                continue  # Try the next model in the chain
            else:
                logger.warning(f"Query understanding failed on {model}: {e}")
                last_error = e
                continue

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                logger.warning(f"⚠️ 429 Rate Limit hit on {model}. Auto-switching to fallback...")
                last_error = e
                continue
            else:
                logger.warning(f"Query understanding failed on {model}: {e}")
                last_error = e
                continue

    # If all models fail
    logger.error(f"🚨 All fallback models exhausted for Query Understanding. Last error: {last_error}")
    return {"intent": "search"}


# 🚀 UPDATED: HISTORY-AWARE QUERY TRANSLATION (Fixes the memory format mismatch)
def rewrite_query(current_question: str, history: list) -> str:
    if not history:
        return current_question
        
    recent_history = history[-4:] if len(history) > 4 else history
    
    history_lines = []
    for msg in recent_history:
        role = msg.get('role', 'user').upper()
        
        # 🚀 FIX: Handle Gemini's specific memory format {"parts": [{"text": "..."}]}
        content = msg.get('content', '')
        if not content and 'parts' in msg and msg['parts']:
            content = msg['parts'][0].get('text', '')
            
        history_lines.append(f"{role}: {content}")
        
    history_text = "\n".join(history_lines)
    
    prompt = f"""You are an expert at resolving coreferences in chat conversations.
Given the chat history and a follow-up question, rewrite the follow-up question to be a standalone question that can be understood without the history.

Chat History:
{history_text}

Follow-up Question: {current_question}

Rules:
1. Replace pronouns like "there", "it", "they", "this college", "that one" with the actual entity from the history.
2. If the question is already standalone or completely unrelated to the history, return it exactly as is.
3. Return ONLY the rewritten question. No explanations, no quotes, no markdown.

Standalone Question:"""

    last_error = None
    for model in FALLBACK_MODELS:
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            rewritten = response.text.strip().strip('"').strip("'")
            logger.info(f"🔄 Query Rewritten: '{current_question}' -> '{rewritten}'")
            return rewritten
        except Exception as e:
            last_error = e
            continue
            
    logger.warning(f"Query rewriting failed, using original question. Error: {last_error}")
    return current_question