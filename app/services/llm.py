import logging
import os
from typing import Optional, List, Dict, Any
from app.services.memory import memory

logger = logging.getLogger(__name__)

# 🚀 Production Fallback Chain 
# Note: Ensure these model names match the exact strings supported by your google-genai SDK version.
FALLBACK_MODELS = [
    "gemini-2.0-flash",         # Primary: Latest stable fast model
    "gemini-1.5-flash-latest",  # Secondary: Highly reliable fallback
    "gemini-1.5-pro-latest",    # Tertiary: Higher reasoning if flash fails
    "gemini-1.0-pro"            # Last resort legacy
]

SYSTEM_PROMPT = """You are an expert AI counselor for Tamil Nadu Engineering Colleges (TNEA).
You must answer the student's question using ONLY the provided <knowledge_base> XML context.

🚨 STRICT RULES:
1. NEVER use your internal internet training data, guess, or estimate facts. 
2. If the exact answer is NOT in the context, you MUST reply exactly with: "I don't have the exact information for this in my database."
3. When calculating fees or intake, use the EXACT numbers from the context. DO NOT invent ranges.
4. Do not make up amenities (like gym, ambulance, or specific clubs) unless they are explicitly written in the context.

🧠 BRANCH CODE TRANSLATOR (Crucial for matching user queries to database codes):
- CS = Computer Science and Engineering (CSE)
- EC = Electronics and Communication Engineering (ECE)
- ME = Mechanical Engineering
- EE = Electrical and Electronics Engineering (EEE)
- CE = Civil Engineering
- IT = Information Technology
- AD = Artificial Intelligence and Data Science (AI & DS)
- AI = Artificial Intelligence and Machine Learning (AI & ML)
- CB = Computer Science and Business Systems (CSBS)
- CY = Cyber Security
- AU = Automobile Engineering
- CH = Chemical Engineering

INSTRUCTION: If the context shows a college has branch code "AD", and the student asks if it offers "Artificial Intelligence and Data Science", you MUST answer YES. Treat the codes and full names as identical.
"""

def format_history_for_gemini(history: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Converts standard OpenAI-style memory history [{"role": "user", "content": "..."}]
    to Google GenAI SDK format [{"role": "user", "parts": [{"text": "..."}]}].
    """
    if not history:
        return []
        
    formatted = []
    for msg in history:
        role = msg.get("role")
        content = msg.get("content")
        
        # Google SDK uses 'model' instead of 'assistant'
        if role == "assistant":
            role = "model"
            
        if role in ["user", "model"] and content:
            formatted.append({
                "role": role,
                "parts": [{"text": content}]
            })
    return formatted

def generate_answer(
    question: str, 
    xml_context: str, 
    session_id: str,
    reference_answer: Optional[str] = None, 
    intent: str = "search"
) -> str:

    # 🛠️ FIX 1: Strict Lazy Imports (Prevents 512MB Render OOM crash on startup)
    from google import genai
    from google.genai import types
    from google.genai import errors
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in environment variables.")
        
    client = genai.Client(api_key=api_key)

    # Fetch and format history
    raw_history = memory.get_history(session_id) or []
    history = format_history_for_gemini(raw_history)

    # Build Context & System Instruction
    if reference_answer:
        context_block = (
            f"REFERENCE ANSWER (from a previous similar question):\n{reference_answer}\n\n"
            f"CURRENT KNOWLEDGE BASE:\n{xml_context}"
        )
        system_instruction = SYSTEM_PROMPT + (
            "\n\nADDITIONAL: A similar question was asked before and a reference answer is provided. "
            "Analyze it against the current knowledge base. Generate a FRESH, natural response. "
            "Do not copy it verbatim; rephrase and improve it using the current XML context."
        )
    else:
        context_block = f"CONTEXT:\n{xml_context}"
        system_instruction = SYSTEM_PROMPT

    user_message = f"{context_block}\n\nSTUDENT QUESTION: {question}"
    
    # Generation Config (Low temperature is CRITICAL for RAG to prevent hallucinations)
    gen_config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.1, 
        max_output_tokens=1024,
    )

    last_error = None

    for model in FALLBACK_MODELS:
        try:
            logger.info(f"💬 Generation: Trying {model}")
            
            chat = client.chats.create(
                model=model,
                history=history,
                config=gen_config
            )
            response = chat.send_message(user_message)
            
            # Handle potential empty responses (happens if Gemini blocks prompt due to safety filters)
            if not response.text:
                logger.warning(f"⚠️ Empty response from {model}. Prompt might be blocked by safety filters.")
                raise ValueError("Empty response generated by model.")
                
            answer = response.text
            logger.info(f"✅ Success with {model}")

            # Save only the question and answer to memory (not the huge XML context)
            memory.add_message(session_id, "user", question)
            memory.add_message(session_id, "assistant", answer) # Use 'assistant' for standard memory schemas
            
            return answer
            
        except errors.ClientError as e:
            # Handle 429 (rate limit), 404 (model not found), 400 (bad request)
            error_code = getattr(e, 'code', getattr(e, 'status_code', 500))
            if error_code in [429, 404, 400]:
                logger.warning(f"⚠️ {error_code} Error on {model}: {e}. Auto-switching to fallback...")
                last_error = e
                continue
            else:
                logger.error(f"❌ Client Error on {model}: {e}")
                last_error = e
                break # Break on non-retryable client errors like 403 Forbidden
                
        except Exception as e:
            logger.error(f"❌ Unexpected Error on {model}: {e}")
            last_error = e
            continue

    # If all models fail, return a graceful message instead of crashing the FastAPI server
    logger.error(f"🚨 All fallback models exhausted for Generation. Last error: {last_error}")
    return "I am currently experiencing technical difficulties connecting to the AI counselor. Please try again in a few moments."