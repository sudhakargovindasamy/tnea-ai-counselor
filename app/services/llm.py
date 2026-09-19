import concurrent.futures
import logging
import os
from collections.abc import Generator
from typing import Any

from app.services.memory import memory

logger = logging.getLogger(__name__)


# 🚀 Production Fallback Chain (Updated for current API availability)
FALLBACK_MODELS = [
    "gemini-3.8-flash",   # Latest and fastest (Released Sept 2026)
    "gemini-3.5-flash",   # Highly stable fallback
    "gemini-3.6-flash"    # Stable mid-tier fallback
]

SYSTEM_PROMPT = """You are an expert AI counselor for Tamil Nadu Engineering Colleges (TNEA).
Answer ONLY using the provided <knowledge_base> context. If the context does not contain the answer, explicitly state: 'The provided TNEA database does not contain information to answer this.' Do not make up college names, cutoffs, or branch codes.

🚨 STRICT GROUNDING RULES:
1. NEVER use your internal internet training data, guess, or estimate facts. 
2. If the exact answer is NOT in the context, you MUST reply with: "The provided TNEA database does not contain information to answer this."
3. When calculating fees or intake, use the EXACT numbers from the context. DO NOT invent numbers or ranges.
4. Do not make up amenities (like gym, ambulance, or specific clubs) unless explicitly written in the context.
5. Cutoff data does not exist in the database. If a student asks about cutoff marks or closing ranks, state: "I don't have cutoff/closing rank data in my database. I can help with college facilities, branches, and admission rules. For cutoff predictions, check tneaonline.org."
6. If the student asks about a specific branch at a specific college, and that branch code is NOT listed in the retrieved context for that college, you MUST answer 'No' and list the branches that ARE available. NEVER confirm a branch exists unless it appears in the context.

🧠 BRANCH CODE TRANSLATOR (Crucial for matching user queries to database codes):
- CS = Computer Science and Engineering (CSE)
- EC = Electronics and Communication Engineering (ECE)
- ME = Mechanical Engineering
- EE = Electrical and Electronics Engineering (EEE)
- CE = Civil Engineering
- IT = Information Technology
- AD = Artificial Intelligence and Data Science (AI & DS / AI and DS)
- AI / AL = Artificial Intelligence and Machine Learning (AI & ML)
- CB = Computer Science and Business Systems (CSBS)
- CY = Cyber Security
- AU = Automobile Engineering
- CH = Chemical Engineering
- BM = Biomedical Engineering
- BT = Biotechnology
- AG = Agricultural Engineering

INSTRUCTION: If the context shows a college has branch code "AD" or lists Artificial Intelligence and Data Science, and the student asks if it offers "AI & DS" or "Artificial Intelligence", you MUST answer YES and specify the approved intake. Treat branch codes and full names as identical.
"""

def format_history_for_gemini(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Converts standard memory history to Google GenAI SDK format.
    Handles both {"content": "..."} and {"parts": [{"text": "..."}]}.
    """
    if not history:
        return []
        
    formatted = []
    for msg in history:
        role = msg.get("role")
        content = msg.get("content")
        if not content and "parts" in msg and msg["parts"]:
            content = msg["parts"][0].get("text", "")
            
        if role == "assistant":
            role = "model"
            
        if role in ["user", "model"] and content:
            formatted.append({
                "role": role,
                "parts": [{"text": str(content)}]
            })
    return formatted

def append_citations(answer: str, docs: list[dict[str, Any]] | None) -> str:
    """
    Appends clean, structured source citations at the end of the LLM response
    so the frontend and user can verify information against official TNEA data.
    """
    if not docs:
        return answer
        
    # Don't append citations to unanswerable refusal messages
    if "The provided TNEA database does not contain information to answer this." in answer:
        return answer

    citations = []
    seen = set()
    for d in docs:
        meta = d.get("metadata", {})
        doc_type = meta.get("doc_type", "college_info")
        
        if doc_type == "admission_info":
            sec = meta.get("section", "General Information")
            src_doc = meta.get("source_document", "TNEA Information Brochure 2026")
            cite_str = f"[Source: TNEA Rules - {sec} ({src_doc})]"
        else:
            college = meta.get("college_name", "Unknown College")
            # Truncate long college address for concise citation
            short_college = college.split(",")[0].strip()
            tnea = meta.get("tnea_code", "N/A")
            district = meta.get("district", "Tamil Nadu")
            cite_str = f"[Source: {short_college}, TNEA Code: {tnea}, District: {district}]"
            
        if cite_str not in seen:
            seen.add(cite_str)
            citations.append(cite_str)

    if not citations:
        return answer

    citation_block = "\n\n**Sources & Citations:**\n" + "\n".join(f"- {c}" for c in citations)
    return answer + citation_block

def _call_gemini_with_timeout(chat, user_message: str, timeout_seconds: float = 30.0):
    """
    Execute Gemini send_message in a worker thread with a strict timeout.
    Prevents API hangs from blocking the FastAPI event loop or client.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(chat.send_message, user_message)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Gemini API generation timed out after {timeout_seconds} seconds")

def generate_answer(
    question: str, 
    xml_context: str, 
    session_id: str,
    reference_answer: str | None = None, 
    intent: str = "search",
    docs: list[dict[str, Any]] | None = None
) -> str:
    """
    Synchronous answer generation with 30s timeout, fallback models,
    and automatic grounding citations.
    """
    from google import genai
    from google.genai import errors, types
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        error_msg = "GEMINI_API_KEY is not set in environment variables."
        logger.error(f"❌ {error_msg}")
        return f"Configuration Error: {error_msg}"
        
    client = genai.Client(api_key=api_key)

    raw_history = memory.get_history(session_id) or []
    history = format_history_for_gemini(raw_history)

    if reference_answer:
        context_block = (
            f"REFERENCE ANSWER:\n{reference_answer}\n\n"
            f"CURRENT KNOWLEDGE BASE:\n{xml_context}"
        )
        system_instruction = SYSTEM_PROMPT + (
            "\n\nADDITIONAL: Rephrase and improve the answer using the current XML context."
        )
    else:
        context_block = f"CONTEXT:\n{xml_context}"
        system_instruction = SYSTEM_PROMPT

    user_message = f"{context_block}\n\nSTUDENT QUESTION: {question}"
    
    gen_config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.1, 
        max_output_tokens=1024,
    )

    last_error = None

    for model in FALLBACK_MODELS:
        try:
            logger.info(f"💬 Generation: Sending request to {model} (30s timeout)...")
            
            chat = client.chats.create(
                model=model,
                history=history,
                config=gen_config
            )
            
            response = _call_gemini_with_timeout(chat, user_message, timeout_seconds=30.0)
            
            if not response.text:
                raise ValueError("Empty response generated by model.")
                
            raw_answer = response.text.strip()
            final_answer = append_citations(raw_answer, docs)
            logger.info(f"✅ Generation successful with model '{model}'.")
            return final_answer

        except TimeoutError as te:
            logger.error(f"⏱️ TimeoutError on model '{model}': {te}", exc_info=True)
            last_error = te
            continue
            
        except errors.ClientError as ce:
            error_code = getattr(ce, "code", getattr(ce, "status_code", 500))
            logger.error(f"❌ ClientError on model '{model}' (HTTP {error_code}): {ce}", exc_info=True)
            last_error = ce
            if error_code in [429, 404, 400]:
                continue
            else:
                break
                
        except Exception as e:
            last_error = e
            err_name = type(e).__name__
            if "Connect" in err_name or "Resolution" in err_name or "Network" in err_name:
                logger.warning(f"🌐 Generation network unreachable on model '{model}': {e}")
                break
            logger.error(f"❌ Generation Exception on model '{model}': {err_name}: {e}", exc_info=True)
            continue

    error_summary = f"{type(last_error).__name__}: {last_error!s}" if last_error else "Unknown generation error"
    logger.error(f"🚨 All fallback models exhausted for LLM Generation: {error_summary}")
    return f"Error: LLM Generation failed ({error_summary})"

def generate_answer_stream(
    question: str, 
    xml_context: str, 
    session_id: str,
    docs: list[dict[str, Any]] | None = None
) -> Generator[str, None, None]:
    """
    Streaming token generator for Server-Sent Events (SSE).
    Streams chunks token-by-token and concludes with citations.
    """
    from google import genai
    from google.genai import types
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        yield "Error: GEMINI_API_KEY is not set."
        return
        
    client = genai.Client(api_key=api_key)

    raw_history = memory.get_history(session_id) or []
    history = format_history_for_gemini(raw_history)

    context_block = f"CONTEXT:\n{xml_context}"
    user_message = f"{context_block}\n\nSTUDENT QUESTION: {question}"

    gen_config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.1, 
        max_output_tokens=1024,
    )

    full_response_text = []

    for model in FALLBACK_MODELS:
        try:
            logger.info(f"🌊 SSE Streaming with model '{model}'...")
            chat = client.chats.create(
                model=model,
                history=history,
                config=gen_config
            )
            
            stream = chat.send_message_stream(user_message)
            for chunk in stream:
                if chunk.text:
                    full_response_text.append(chunk.text)
                    yield chunk.text

            # Stream finished successfully
            complete_answer = "".join(full_response_text).strip()
            
            # Append citations at end of stream if applicable
            if docs and "The provided TNEA database does not contain information to answer this." not in complete_answer:
                citations_text = append_citations("", docs)
                yield citations_text
                complete_answer += citations_text

            memory.add_message(session_id, "user", question)
            memory.add_message(session_id, "assistant", complete_answer)
            return

        except Exception as e:
            err_name = type(e).__name__
            if "Connect" in err_name or "Resolution" in err_name or "Network" in err_name:
                logger.warning(f"🌐 Streaming network unreachable on model '{model}': {e}")
                break
            logger.warning(f"⚠️ Streaming failed on model '{model}': {e}. Trying fallback...")
            continue

    yield "\n[Error: LLM streaming generation failed across all models.]"