import json
import logging
import os
import re
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# 🚀 Production Fallback Chain (Updated for current API availability)
FALLBACK_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3-flash-preview"
]

# ==========================================
# 🛠️ PYDANTIC SCHEMAS (Guarantees Valid JSON)
# ==========================================
class QueryUnderstandingSchema(BaseModel):
    intent: Literal["search", "compare", "numerical", "filter", "list", "cutoff"] = Field(
        description="Primary intent. 'filter' for constraints, 'list' for broad lists, 'search' for general facts, 'cutoff' for marks/cutoff queries."
    )
    compare_colleges: list[str] | None = Field(default=None, description="List of college names if intent is compare")
    college_name: str | None = Field(default=None, description="Specific college name")
    district: str | None = Field(default=None, description="Title Case district (e.g., Chennai, Coimbatore, Salem)")
    autonomous: Literal["Yes", "No"] | None = Field(default=None)
    branch_code: Literal["CS", "EC", "ME", "IT", "AD", "CE", "EE", "CH", "AU", "CB", "AL", "SC", "AE", "BM", "BT", "AG", "EV", "IC", "EI", "CY", "FT", "TX", "AM", "CN", "EM", "EY", "EL", "GI", "MD", "MC", "MT", "PC", "PH", "PE", "RM", "SF", "VL", "CD", "CJ", "DA", "DS", "EA", "EF", "EN", "EX", "HT", "IB", "IN", "MM", "MR", "MU", "TC", "TT", "AR", "AS", "BS", "BY", "CC", "CF", "CG", "CI", "CK", "CL", "CM", "FY", "IY", "MF", "MS", "MY", "PN", "PR", "RA", "RI", "SB", "TS", "X", "B*", "AT", "AO", "BC", "BP", "BA", "LE", "PP", "PM", "IS", "AP", "PT", "MB", "MN", "MA"] | None = Field(
        default=None, 
        description="Strict branch code mapping. Examples: Computer Science/CSE->CS, Mechanical->ME, AI&DS->AD, ECE->EC, EEE->EE, IT->IT, Civil->CE, Cyber Security->CY, AI&ML->AL, Bio Medical->BM, Bio Tech->BT, Automobile->AU, Chemical->CH, Aeronautical->AE, Agricultural->AG"
    )
    department_code: str | None = Field(default=None, description="Department code alias")
    nba_accredited: bool | None = Field(default=None)
    numerical_metric: Literal["placement", "pass_percentage", "total_intake", "fees", "hostel_rent", "transport", "cutoff", "marks"] | None = Field(default=None)
    numerical_operator: Literal["highest", "lowest", "greater_than", "less_than"] | None = Field(default=None)
    numerical_value: float | None = Field(default=None, description="Target number if operator is greater_than/less_than")

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
# 🔄 FALLBACK HELPER WITH ENHANCED ERROR HANDLING
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
                logger.info(f"✅ Native Pydantic parsing succeeded with {model}")
                return response.parsed
                
            # Fallback manual parsing if native SDK parsing fails but returns text
            elif hasattr(response, 'text') and response.text:
                clean_text = re.sub(r"^```json\s*|\s*```$", "", response.text.strip(), flags=re.MULTILINE)
                try:
                    parsed_dict = json.loads(clean_text)
                    logger.info(f"✅ Manual JSON parsing succeeded with {model}")
                    return schema_class(**parsed_dict)
                except json.JSONDecodeError as json_err:
                    logger.warning(f"⚠️ JSON decode failed on {model}: {json_err}")
                    continue
                except Exception as validation_err:
                    logger.warning(f"⚠️ Pydantic validation failed on {model}: {validation_err}")
                    continue
                
        except errors.ClientError as e:
            if e.code in [429, 404, 400]:
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
    
    prompt = f"""You are an expert query router for Tamil Nadu Engineering colleges (TNEA).
Analyze the student's question and extract intent, filters, and numerical constraints.

RULES:
1. Intent "compare": Side-by-side comparison (e.g., "Compare CEG and MIT").
2. Intent "numerical": Ranking or threshold (e.g., "Top 5 colleges", "Placement > 90%").
3. Intent "filter": Specific constraints without ranking (e.g., "Autonomous colleges in Chennai with CS").
4. Intent "list": Broad requests (e.g., "List all colleges in Coimbatore").
5. Intent "cutoff": Queries about cutoff marks or minimum marks required (e.g., "cutoff of 185", "180 marks cutoff").
6. Intent "search": General informational queries (e.g., "What is the hostel fee at SSN?").
7. Map branches STRICTLY to the provided Enum codes. Common mappings:
   - Computer Science/CSE/CS -> CS
   - Mechanical/MECH -> ME
   - Electronics/ECE -> EC
   - Electrical/EEE -> EE
   - Information Technology/IT -> IT
   - Civil -> CE
   - AI & Data Science/AI&DS/AD -> AD
   - AI & Machine Learning/AI&ML/AL -> AL
   - Cyber Security/CY -> CY
   - Bio Medical/Biomedical/BM -> BM
   - Bio Technology/Biotech/BT -> BT
   - Automobile/AUTO -> AU
   - Chemical/CHEM -> CH
   - Aeronautical/AERO -> AE
   - Agricultural/AGRI -> AG
8. For district names, use Title Case (e.g., "Chennai", "Coimbatore", "Salem").
9. Extract numerical_operator and numerical_value if applicable (e.g., "Top 5" -> operator: "highest", value: 5).
10. If asking about cutoff or marks, set intent to "cutoff" and extract the cutoff value.

Question: {question}
"""

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=QueryUnderstandingSchema,
        temperature=0.1  # Low temperature for strict extraction
    )

    result = _generate_with_fallback(client, FALLBACK_MODELS, prompt, config, QueryUnderstandingSchema)
    
    if result:
        # Convert Pydantic model to dict, excluding None values to keep payload clean
        extracted = result.model_dump(exclude_none=True)
        if "branch_code" in extracted and "department_code" not in extracted:
            extracted["department_code"] = extracted["branch_code"]
        elif "department_code" in extracted and "branch_code" not in extracted:
            extracted["branch_code"] = extracted["department_code"]
        logger.info(f"🧠 Extracted: {extracted}")
        return extracted
        
    # Fallback if all models fail
    logger.warning("⚠️ All models failed, defaulting to search intent")
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
4. Keep the rewritten question concise and clear.
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