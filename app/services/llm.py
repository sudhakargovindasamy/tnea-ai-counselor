import logging
import os
from app.services.memory import memory

logger = logging.getLogger(__name__)

# 🚀 Production Fallback Chain (Based on what actually works in your logs)
FALLBACK_MODELS = [
    "gemini-3.6-flash",         # Primary: Confirmed working
    "gemini-3.5-flash",         # Secondary: Works but hits quota
    "gemini-2.0-flash",         # Tertiary: Try anyway
    "gemini-1.5-flash-latest"   # Last resort
]

SYSTEM_PROMPT = """You are an expert AI counselor for Tamil Nadu Engineering Colleges (TNEA).
You must answer the student's question using ONLY the provided <knowledge_base> XML context.

🚨 STRICT RULES:
1. NEVER use your internal training data, guess, or estimate facts. 
2. If the exact answer is NOT in the context, you MUST reply exactly with: "I don't have the exact information for this in my database."
3. When calculating hostel fees, use the EXACT numbers from the context (e.g., add 'mess_bill' and 'room_rent'). DO NOT invent ranges like "₹90,000 to ₹1,10,000".
4. Do not make up amenities (like "gymnasium" or "ambulance") unless they are explicitly written in the context.
"""

def generate_answer(question: str, xml_context: str, session_id: str,
                    reference_answer: str = None, intent: str = "search") -> str:

    # 🛠️ FIX 1: Strict Lazy Imports (Prevents 512MB Render OOM crash on startup)
    # PyTorch and Google libs load ONLY when a query actually arrives.
    from google import genai
    from google.genai import types
    from google.genai import errors
    
    # Initialize client lazily inside the function
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    history = memory.get_history(session_id)

    if reference_answer:
        context_block = (f"REFERENCE ANSWER (from a previous similar question):\n{reference_answer}\n\n"
                         f"CURRENT KNOWLEDGE BASE:\n{xml_context}")
        system_instruction = SYSTEM_PROMPT + (
            "\n\nADDITIONAL: A similar question was asked before and a reference answer is provided. "
            "Analyze it against the current knowledge base. Generate a FRESH, natural response. "
            "Do not copy it verbatim; rephrase and improve it using the current XML context.")
    else:
        context_block = f"CONTEXT:\n{xml_context}"
        system_instruction = SYSTEM_PROMPT

    user_message = f"{context_block}\n\nSTUDENT QUESTION: {question}"
    
    last_error = None

    for model in FALLBACK_MODELS:
        try:
            logger.info(f"💬 Generation: Trying {model}")
            
            chat = client.chats.create(
                model=model,
                history=history,
                config=types.GenerateContentConfig(system_instruction=system_instruction)
            )
            response = chat.send_message(user_message)
            answer = response.text
            
            logger.info(f"✅ Success with {model}")

            # Save only the question to memory (not the huge XML context)
            memory.add_message(session_id, "user", question)
            memory.add_message(session_id, "model", answer)
            return answer
            
        except errors.ClientError as e:
            # 🚀 FIX: Handle BOTH 429 (rate limit) AND 404 (model not found) errors
            if e.code in [429, 404]:
                logger.warning(f"⚠️ {e.code} Error on {model}: {e.message}. Auto-switching to fallback...")
                last_error = e
                continue  # Automatically try the next model in the chain
            else:
                logger.error(f"❌ Client Error on {model}: {e}")
                raise e
                
        except Exception as e:
            logger.error(f"❌ Unexpected Error on {model}: {e}")
            last_error = e
            continue

    # If all models fail
    logger.error(f"🚨 All fallback models exhausted for Generation. Last error: {last_error}")
    raise last_error