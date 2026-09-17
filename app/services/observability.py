"""
app/services/observability.py
Observability & Tracing Layer for TNEA Counselor RAG System.
Provides structured JSON logging and trace capturing for:
  Query → Retrieved Chunks → LLM Prompt → Final Answer → Latency
Compatible with stdout collectors (Datadog, CloudWatch, Render, ELK) and Langfuse.
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Any

logger = logging.getLogger("tnea.observability")

# Check if Langfuse credentials are configured
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
_langfuse_client = None

def _get_langfuse():
    global _langfuse_client
    if _langfuse_client is None and LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
        try:
            from langfuse import Langfuse
            _langfuse_client = Langfuse(
                public_key=LANGFUSE_PUBLIC_KEY,
                secret_key=LANGFUSE_SECRET_KEY,
                host=LANGFUSE_HOST
            )
            logger.info("🔭 Langfuse observability client initialized.")
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize Langfuse: {e}")
    return _langfuse_client

class RAGTracer:
    """Context manager for tracing end-to-end RAG pipeline execution."""
    def __init__(self, session_id: str, query: str):
        self.session_id = session_id
        self.query = query
        self.start_time = time.time()
        self.retrieved_chunks: list[dict[str, Any]] = []
        self.llm_prompt: str = ""
        self.final_answer: str = ""
        self.metadata: dict[str, Any] = {}
        self.cache_hit: bool = False
        self.error: str | None = None

    def log_retrieval(self, docs: list[dict[str, Any]], filters: dict = None):
        self.retrieved_chunks = [
            {
                "tnea_code": str(d.get("metadata", {}).get("tnea_code", "N/A")),
                "college_name": d.get("metadata", {}).get("college_name", "N/A"),
                "district": d.get("metadata", {}).get("district", "N/A"),
                "score": round(float(d.get("rerank_score", d.get("similarity", 0.0))), 4),
                "content_preview": d.get("content", "")[:120].replace("\n", " ") + "..."
            }
            for d in docs
        ] if docs else []
        if filters:
            self.metadata["filters"] = filters

    def log_llm_call(self, prompt: str, answer: str):
        self.llm_prompt = prompt
        self.final_answer = answer

    def log_cache_hit(self, answer: str):
        self.cache_hit = True
        self.final_answer = answer

    def log_error(self, err: str):
        self.error = err

    def finish(self) -> dict[str, Any]:
        latency_s = round(time.time() - self.start_time, 4)
        trace_record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event": "rag_pipeline_trace",
            "session_id": self.session_id,
            "query": self.query,
            "cache_hit": self.cache_hit,
            "latency_seconds": latency_s,
            "retrieved_count": len(self.retrieved_chunks),
            "retrieved_chunks": self.retrieved_chunks,
            "llm_prompt_preview": (self.llm_prompt[:250] + "...") if self.llm_prompt else "N/A",
            "final_answer_preview": (self.final_answer[:250] + "...") if self.final_answer else "N/A",
            "metadata": self.metadata,
            "error": self.error
        }

        # Structured JSON Output to Logger
        logger.info(f"📊 [RAG_TRACE] {json.dumps(trace_record)}")

        # Optional: Send trace to Langfuse if configured
        langfuse = _get_langfuse()
        if langfuse:
            try:
                trace = langfuse.trace(
                    name="tnea_counselor_rag",
                    session_id=self.session_id,
                    input={"query": self.query, "metadata": self.metadata},
                    output={"answer": self.final_answer},
                    metadata={"latency": latency_s, "cache_hit": self.cache_hit}
                )
                if self.retrieved_chunks:
                    trace.span(
                        name="retrieval",
                        input={"query": self.query},
                        output={"chunks": self.retrieved_chunks}
                    )
                if not self.cache_hit and self.llm_prompt:
                    trace.generation(
                        name="gemini_generation",
                        input=self.llm_prompt,
                        output=self.final_answer
                    )
            except Exception as e:
                logger.debug(f"Langfuse dispatch error: {e}")

        return trace_record

