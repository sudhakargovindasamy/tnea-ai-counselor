import logging
from google import genai
from google.genai import types
from google.genai import errors
from app.config import GEMINI_API_KEY
from app.services.memory import memory

logger = logging.getLogger(__name__)
client = genai.Client(api_key=GEMINI_API_KEY)

# 🚀 Production Fallback Chain (Prioritize working models)
FALLBACK_MODELS = [
    "gemini-3.5-flash",   # Primary: Confirmed working in your logs
    "gemini-3.6-flash",   # Secondary: Alternative working model
    "gemini-2.5-pro",     # Tertiary: Pro fallback
    "gemini-2.5-flash"    # Last resort (may not work for new users)
]

SYSTEM_PROMPT = """You are an expert Academic Counselor specializing in Tamil Nadu Engineering Colleges (TNEA). 
Your goal is to provide students and parents with comprehensive, clear, and highly structured information.

When answering based on the provided context:
1. Use a warm, professional, and encouraging tone.
2. Structure your response using Markdown headers and bullet points. Use sections like:
   - 🏛️ **College Overview** (Location, Autonomy, TNEA Code)
   - 📚 **Academics & Branches** (List available courses and intakes if available)
   - 📊 **Performance & Placements** (Pass percentages, placement records)
   - 🏢 **Infrastructure & Hostel** (Fees, facilities, transport)
3. Synthesize ALL provided context documents. Combine the main profile, branch details, and performance stats into one unified report.
4. Never say "information is not present" if it exists anywhere in the context.
5. End with a brief 1-sentence summary or helpful tip for the student.

🚨 6. CRITICAL ANTI-HALLUCINATION RULE: You must ONLY use the information provided in the <knowledge_base> XML. 
If the user asks a question that cannot be answered using the provided XML context (e.g., questions about sports, cooking, non-TNEA colleges, or general knowledge), you MUST NOT use your own internal knowledge. 
Instead, you must reply EXACTLY with this phrase: "I only have information about Tamil Nadu Engineering Colleges (TNEA). I don't have enough specific information in my database to answer this accurately."

🕵️ 7. ENTITY MISMATCH RULE (The "Frankenstein" Check): 
If the user asks about a specific college name (e.g., "Sudhakar Engineering College"), but the provided context belongs to a COMPLETELY DIFFERENT college (e.g., "Sri Krishna Engineering College"), DO NOT pretend the fake college exists. 
Instead, politely correct the user: State that the exact college name they mentioned is NOT in the TNEA database, but point out that the details they provided (like the principal, address, or courses) actually belong to the real college found in the context.
"""

def generate_answer(question: str, xml_context: str, session_id: str,
                    reference_answer: str = None, intent: str = "search") -> str:

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