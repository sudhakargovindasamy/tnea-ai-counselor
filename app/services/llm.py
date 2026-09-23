import concurrent.futures
import logging
import os
from collections.abc import Generator
from typing import Any

from app.services.memory import memory

logger = logging.getLogger(__name__)


# 🚀 Production Fallback Chain (Updated for current API availability)
FALLBACK_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-flash-latest"
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
7. NAAC & NIRF ACCREDITATION RULE:
The official TNEA database tracks Autonomous status and NBA branch-level accreditation, but does NOT record institutional NAAC grades (e.g. NAAC A, A+, A++) or NIRF rankings.
- If a student asks for colleges with NAAC grades or NIRF rankings alongside other criteria (e.g. "Autonomous colleges in Coimbatore with NAAC A grade"):
  * Do NOT refuse with "The provided TNEA database does not contain information to answer this."
  * Explicitly explain: "Note: Institutional NAAC grades and NIRF rankings are not recorded in the official TNEA database (which officially tracks Autonomous status and NBA branch accreditation)."
  * Answer the student's request by presenting the relevant colleges from the context (e.g., the Autonomous colleges in Coimbatore from the retrieved context), highlighting their Autonomous status, NBA accredited branches, intake, and academic performance.
  * Advise the student to verify official NAAC grade ratings on naac.gov.in or the college's official website.
- If a student asks purely for the NAAC grade or NIRF rank of a college (e.g. "What is the NAAC grade of PSG Tech?"):
  * State clearly: "The official TNEA counselling database does not track institutional NAAC accreditation grades or NIRF rankings (it only records Autonomous status and NBA program accreditation). For official NAAC grade ratings, please check naac.gov.in or the college's official website."
  * Provide the available accreditation details from the context (such as whether the college is Autonomous and which branches are NBA accredited).

🧠 BRANCH CODE TRANSLATOR (Crucial for matching user queries to database codes):
- CS = Computer Science and Engineering (CSE)
- EC = Electronics and Communication Engineering (ECE)
- ME = Mechanical Engineering
- EE = Electrical and Electronics Engineering (EEE)
- CE = Civil Engineering
- IT = Information Technology
- AD / AT = Artificial Intelligence and Data Science (AI & DS / AI and DS)
- AL / AM / CG = Artificial Intelligence and Machine Learning (AI & ML / CSE with AI & ML)
- CB = Computer Science and Business Systems (CSBS)
- CY / SC = Cyber Security / CSE (Cyber Security)
- AU = Automobile Engineering
- CH = Chemical Engineering
- BM = Biomedical Engineering
- BT = Biotechnology
- AG = Agricultural Engineering

8. AGGREGATIONS & TOTAL SEATS / STATISTICS RULE:
- If the context contains an <aggregation_data> block, state the EXACT total numbers provided in that block (e.g. total_colleges_offering_course, total_approved_seats).
- For example, if asked "Sum the total number of seats in CSE course", explicitly state the exact total approved seats and total colleges from the aggregation block. DO NOT recalculate, guess, or sum partial lists.

9. COMPREHENSIVE COLLEGE LISTINGS RULE:
- When a student asks to "list the colleges", "which colleges offer...", or for colleges offering a course/branch:
  * If the student explicitly specifies a number (e.g. "top 3", "top 5", "give me 5 colleges"): Provide ONLY that requested number of colleges.
  * If the student asks generally (e.g. "List the colleges that offer Marine Engineering course", "List colleges with CSE"):
    1. State the authoritative total count of colleges offering that course.
    2. Provide the complete list of all matching colleges from the context in a clean, compact numbered format (with TNEA Code, Name, District, and Intake Seats).
    3. Do NOT provide PDF download links. Render the entire response directly inside the chat.

10. COLLEGE COURSE INQUIRY RULE:
- If the student asks whether a specific college offers a course (e.g. "Does Sairam offer AI & ML?"):
  * Check the retrieved college's branch list carefully.
  * For Artificial Intelligence and Machine Learning (AI & ML), check for AL, AM (Computer Science and Engineering - Artificial Intelligence and Machine Learning), or CG. If present, answer YES, state the exact degree name (e.g. B.E. Computer Science and Engineering - AI & ML, branch code AM), and state the approved intake (e.g. 180 seats). Also mention any closely related AI branches like Artificial Intelligence and Data Science (AD) and their intake.
  * If the branch is genuinely not offered by that college, state clearly that it is not offered, and list the relevant computer/engineering branches that are available.

11. TNEA ADMISSION RULES & COUNSELLING BROCHURE RULE:
- When answering questions about TNEA admission procedures, seat allocations, reservations, concessions, documents, counselling stages, or course-specific restrictions (Marine/Mining):
  * Strictly use the retrieved TNEA Information Brochure documents in the context.
  * Allocation of Seats / Reservation: OC (31.00%), BC (26.50%), BCM (3.50%), MBC & DNC (20.00%), SC (15.00%), SCA (3.00%), ST (1.00%).
  * Tuition Fee Concessions:
    - 7.5% Government School Quota: Full tuition fee, hostel fee, and development fee waived by TN Government for students who studied 6th to 12th in TN government schools.
    - First Graduate (FG) Concession: Tuition fee concession for eligible first-generation graduates.
    - AICTE Tuition Fee Waiver (TFW): Up to 5% supernumerary seats in approved institutions for parents with annual family income below Rs. 8.0 Lakhs.
    - Post-Matric Scholarship: SC/SCA/ST and converted Christian SC with annual income < Rs. 2.5 Lakhs.
  * Procedure for Registration: Online registration at www.tneaonline.org or www.dte.tn.gov.in. Application fee is Rs. 500 for General categories and Rs. 250 for SC/SCA/ST candidates.
  * Merit Calculation / Cutoff Formula: Total 200 marks (Mathematics reduced to 100, Physics reduced to 50, Chemistry reduced to 50).
  * Tentative Allotment Confirmation: 4 options during counselling:
    1. Accept and Join (confirm allotment and report to college)
    2. Accept and Upward (accept current seat, look for higher preferences in upward movement)
    3. Decline and Move to Next Round (reject current allotment and participate in next round)
    4. Decline and Quit (reject allotment and exit counselling)
  * Marine Engineering Specifics: 10+2 PCM minimum average 60%, English minimum 50%, IMU CET qualification, maximum age 25 years, minimum height 157 cm, minimum weight 48 kg, and normal colour vision.
  * Mining Engineering Specifics: Female candidates are prohibited from working in underground mines (Mines Act 1952), allowed on surface between 6:00 AM and 7:00 PM only, and shaft cage accompanied by adult males.
  * Nativity Rules: Candidates who studied VIII to XII in Tamil Nadu do not need to produce a Nativity Certificate. TN natives who studied outside TN must produce a Nativity Certificate. Children of Central Govt employees with 5+ continuous years of service in TN are eligible without nativity certificate.

12. REGIONAL / DISTRICT-LEVEL PLACEMENT & PERFORMANCE QUERIES:
- When a user asks about placements or performance across a district or region (e.g., "How is placement across Erode region?", "Placement in Coimbatore colleges", "Salem colleges placement"):
  * Always provide a comprehensive and structured overview of the institutions in that region from the retrieved context.
  * DO NOT miss marquee or premier colleges of the district (e.g., in Erode, prominently include Kongu Engineering College [TNEA 2711] with 94.0% placement and Bannari Amman Institute of Technology [TNEA 2702] with 95.0% placement).
  * Structure the response with:
    1. Overall Regional Placement Summary (tier-1 institutions, autonomous hubs, reported placement trends).
    2. Premier / High-Placement Institutions (TNEA Code, Autonomous status, Placement Rate %, and key highlights).
    3. Other Participating Colleges in the district with their reported placement rates and Anna University pass percentages.
  * Always cite exact quantitative metrics (TNEA Code, District, Placement %, Pass %).

13. SOURCE BOUNDARIES & OUT-OF-SCOPE QUERIES:
- The database consists strictly of TNEA-approved engineering colleges within Tamil Nadu.
- If the user asks for colleges in non-TN locations (e.g., Delhi, Bangalore, international), explicitly state:
  "The current database only contains TNEA-approved engineering colleges within Tamil Nadu. I cannot provide information for colleges outside Tamil Nadu."

14. RANKING & TOP-N QUERIES:
- If a query asks for "top 5", "best", or ranked lists for a course or district where an explicit numerical rank is missing in the database, DO NOT fail.
- Instead, evaluate and present the top institutions based on available quantitative metrics in the context: Placement Rate (%), Anna University Exam Pass Rate (%), and NBA Accreditations.
- Explicitly mention the criteria used for sorting (e.g., "Sorted by highest reported placement percentage and NBA accreditation status").

15. CITATION & TRANSPARENCY:
- Always cite the TNEA Code, District, and specific data points (Intake, Placement %, Pass %) when retrieved from context.
- If a specific metric or college is absent from the provided retrieved context, inform the user clearly while providing whatever related info is available in the retrieved chunks.
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

    if len(citations) > 15:
        citation_block = f"\n\n**Sources & Citations (Showing Top 15 of {len(citations)}):**\n" + "\n".join(f"- {c}" for c in citations[:15])
    else:
        citation_block = "\n\n**Sources & Citations:**\n" + "\n".join(f"- {c}" for c in citations)
    return answer + citation_block

def _call_gemini_with_timeout(chat, user_message: str, timeout_seconds: float = 60.0):
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
    Synchronous answer generation with 60s timeout, fallback models,
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
        max_output_tokens=8192,
    )

    last_error = None

    for model in FALLBACK_MODELS:
        try:
            logger.info(f"💬 Generation: Sending request to {model} (60s timeout)...")
            
            chat = client.chats.create(
                model=model,
                history=history,
                config=gen_config
            )
            
            response = _call_gemini_with_timeout(chat, user_message, timeout_seconds=60.0)
            
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
        max_output_tokens=8192,
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