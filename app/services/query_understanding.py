import json
import re
import os
import logging
from typing import List, Optional, Literal
from functools import lru_cache

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# 🚀 Production Fallback Chain
FALLBACK_MODELS = [
    "gemini-3.6-flash",         # Primary
    "gemini-3.5-flash",         # Secondary
    "gemini-2.0-flash",         # Tertiary
    "gemini-1.5-flash-latest"   # Last resort
]

# ==========================================
# 🛠️ PYDANTIC SCHEMAS (Guarantees Valid JSON)
# ==========================================
class QueryUnderstandingSchema(BaseModel):
    intent: Literal["search", "compare", "numerical", "filter", "list"] = Field(
        description="Primary intent. 'filter' for constraints, 'list' for broad lists, 'search' for general facts."
    )
    compare_colleges: Optional[List[str]] = Field(default=None, description="List of college names if intent is compare")
    college_name: Optional[str] = Field(default=None, description="Specific college name")
    district: Optional[str] = Field(default=None, description="Uppercase district (e.g., CHENNAI, COIMBATORE)")
    autonomous: Optional[Literal["Yes", "No"]] = Field(default=None)
    branch_code: Optional[Literal["CS", "EC", "ME", "IT", "AD", "CE", "EE", "CH", "AU", "CB"]] = Field(
        default=None, 
        description="Strict branch code. Map: Computer Science/CSE->CS, Mechanical->ME, AI&DS/AI->AD, Civil->CE, ECE->EC, EEE->EE, IT->IT, Chemical->CH, Automobile->AU, Cyber Security->CB"
    )
    nba_accredited: Optional[bool] = Field(default=None)
    numerical_metric: Optional[Literal["placement", "pass_percentage", "total_intake", "fees", "hostel_rent", "transport"]] = Field(default=None)
    numerical_operator: Optional[Literal["highest", "lowest", "greater_than", "less_than"]] = Field(default=None)
    numerical_value: Optional[float] = Field(default=None, description="Target number if operator is greater_than/less_than")

class QueryRewriteSchema(BaseModel):
    standalone_question: str = Field(description="The fully resolved standalone question")
    was_rewritten: bool = Field(description="True if coreferences were resolved, False if original was already standalone")

# ==========================================
# 🚀 SINGLETON CLIENT (Prevents TCP/Auth overhead)
# ==========================================
@lru_cache(maxsize=1)
def get_genai_client():
    from google import genai
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")
    return genai.Client(api_key=api_key)

# ==========================================
# 🔄 FALLBACK HELPER
# ==========================================
def _generate_with_fallback(client, model_list: list, contents: str, config, schema_class):
    """Helper to handle the fallback chain and native structured output."""
    from google.genai import errors
    
    last_error = None
    for model in model_list:
        try:
            logger.info(f"🧠 Trying model: {model}")
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )
            
            # Try native Pydantic parsing first (Safest)
            if hasattr(response, 'parsed') and response.parsed:
                return response.parsed
                
            # Fallback manual parsing if native SDK parsing fails but returns text
            elif hasattr(response, 'text') and response.text:
                clean_text = re.sub(r"^```json\s*|\s*```$", "", response.text.strip(), flags=re.MULTILINE)
                parsed_dict = json.loads(clean_text)
                return schema_class(**parsed_dict)
                
        except errors.ClientError as e:
            if e.code in [429, 404, 400]: # 400 added for schema mismatches
                logger.warning(f"⚠️ {e.code} Error on {model}: {e.message}. Falling back...")
                last_error = e
                continue
            else:
                logger.error(f"❌ Unhandled ClientError on {model}: {e}")
                last_error = e
                continue
        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "quota" in error_str or "resource_exhausted" in error_str:
                logger.warning(f"⚠️ Rate Limit/Quota hit on {model}. Falling back...")
                last_error = e
                continue
            else:
                logger.error(f"❌ General Exception on {model}: {e}")
                last_error = e
                continue
                
    logger.error(f"🚨 All fallback models exhausted. Last error: {last_error}")
    return None

# ==========================================
# 🧠 MAIN FUNCTIONS
# ==========================================
def understand_query(question: str) -> dict:
    from google.genai import types
    
    client = get_genai_client()
    
    prompt = f"""You are an expert query router for Tamil Nadu Engineering colleges.
Analyze the student's question and extract intent, filters, and numerical constraints.

RULES:
1. Intent "compare": Side-by-side comparison (e.g., "Compare CEG and MIT").
2. Intent "numerical": Ranking or threshold (e.g., "Top 5 colleges", "Placement > 90%").
3. Intent "filter": Specific constraints without ranking (e.g., "Autonomous colleges in Chennai with CS").
4. Intent "list": Broad requests (e.g., "List all colleges in Coimbatore").
5. Intent "search": General informational queries (e.g., "What is the hostel fee at SSN?").
6. Map branches strictly to the provided Enum codes.
7. Extract numerical_operator and numerical_value if applicable.

Question: {question}
"""

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=QueryUnderstandingSchema,
        temperature=0.1 # Low temperature for strict extraction
    )

    result = _generate_with_fallback(client, FALLBACK_MODELS, prompt, config, QueryUnderstandingSchema)
    
    if result:
        # Convert Pydantic model to dict, excluding None values to keep payload clean for your router
        extracted = result.model_dump(exclude_none=True)
        logger.info(f"🧠 Extracted: {extracted}")
        return extracted
        
    # Fallback if all models fail
    return {"intent": "search", "raw_query": question}


def rewrite_query(current_question: str, history: list) -> str:
    if not history:
        return current_question

    from google.genai import types
    client = get_genai_client()
        
    # Keep last 4 turns (2 pairs of user/assistant) to save tokens and context window
    recent_history = history[-4:] if len(history) > 4 else history
    
    history_lines = []
    for msg in recent_history:
        role = msg.get('role', 'user').upper()
        content = msg.get('content', '')
        if not content and 'parts' in msg and msg['parts']:
            content = msg['parts'][0].get('text', '')
        if content:
            history_lines.append(f"{role}: {content}")
        
    history_text = "\n".join(history_lines)
    
    prompt = f"""You are an expert at resolving coreferences in chat conversations.
Given the chat history and a follow-up question, rewrite the follow-up question to be a standalone question.

Chat History:
{history_text}

Follow-up Question: {current_question}

Rules:
1. Replace pronouns ("there", "it", "they", "this college", "that one", "its") with the actual entity from the history.
2. If the question is ALREADY standalone or completely unrelated to the history, return it EXACTLY as is and set was_rewritten to false.
3. Preserve the original intent and technical terms.
"""

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=QueryRewriteSchema,
        temperature=0.0
    )

    result = _generate_with_fallback(client, FALLBACK_MODELS, prompt, config, QueryRewriteSchema)
    
    if result and result.standalone_question:
        if result.was_rewritten:
            logger.info(f"🔄 Query Rewritten: '{current_question}' -> '{result.standalone_question}'")
        else:
            logger.info(f"✅ Query kept as standalone: '{result.standalone_question}'")
        return result.standalone_question
        
    logger.warning("Query rewriting failed, using original question.")
    return current_question