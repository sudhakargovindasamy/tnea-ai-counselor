import os
from dotenv import load_dotenv


from pathlib import Path


# 1. Hardcode your exact .env path
ENV_PATH = Path("/home/sudhakar/Intern/RAG-final/.env")

# 2. Force load it
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH, override=True)
else:
    print(f"❌ CRITICAL: .env file NOT FOUND at {ENV_PATH}")

# 3. Quick check to confirm the key is loaded
_gemini_key = os.getenv("GEMINI_API_KEY", "")
if _gemini_key:
    print(f"✅ GEMINI_API_KEY loaded successfully (starts with {_gemini_key[:8]}...)")
else:
    print("❌ WARNING: GEMINI_API_KEY is missing in the .env file!")

"""
app/config.py
Configuration management using pydantic-settings.
Securely loads and validates environment variables from .env.
"""

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    SUPABASE_URL: str
    SUPABASE_KEY: str
    GEMINI_API_KEY: str
    ADMIN_SECRET_KEY: str = "TNEA_SUPER_SECRET_ADMIN_KEY_2026"
    IS_PRODUCTION: bool = False
    LANGFUSE_PUBLIC_KEY: str | None = None
    LANGFUSE_SECRET_KEY: str | None = None
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"
    ALLOWED_ORIGINS: list[str] = ["*"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()

# Backwards-compatibility exports
SUPABASE_URL = settings.SUPABASE_URL
SUPABASE_KEY = settings.SUPABASE_KEY
GEMINI_API_KEY = settings.GEMINI_API_KEY
ADMIN_SECRET_KEY = settings.ADMIN_SECRET_KEY