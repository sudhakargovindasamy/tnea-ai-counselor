"""
app/config.py
Configuration management using pydantic-settings.
Automatically loads environment variables from .env (local) or Render dashboard (production).
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENVIRONMENT DETECTION & LOADING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Detect if running on Render (production) or locally
IS_RENDER = os.getenv("RENDER", "").lower() == "true"
IS_PRODUCTION = os.getenv("ENVIRONMENT", "").lower() in ("production", "prod")

# On local machine, try to load .env file
if not IS_RENDER:
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
        print(f"📄 Loaded environment from {env_path.absolute()}")
    else:
        print("⚠️ No .env file found - using system environment variables")
else:
    print("🚀 Running on Render - using dashboard environment variables")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PYDANTIC SETTINGS CLASS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""
    
    # Required (must be set in .env or Render dashboard)
    SUPABASE_URL: str
    SUPABASE_KEY: str
    GEMINI_API_KEY: str
    
    # Optional with defaults
    ADMIN_SECRET_KEY: str = "TNEA_SUPER_SECRET_ADMIN_KEY_2026"
    LANGFUSE_PUBLIC_KEY: str | None = None
    LANGFUSE_SECRET_KEY: str | None = None
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"
    
    # CORS settings
    ALLOWED_ORIGINS: list[str] = ["*"]
    
    # Environment flags
    IS_PRODUCTION: bool = IS_PRODUCTION or IS_RENDER
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INITIALIZE SETTINGS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

settings = Settings()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VALIDATION & FEEDBACK
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n" + "="*60)
print("🔧 TNEA COUNSELOR CONFIGURATION")
print("="*60)

# Check critical keys
if settings.SUPABASE_URL and settings.SUPABASE_KEY:
    print(f"✅ Supabase: Connected to {settings.SUPABASE_URL[:30]}...")
else:
    print("⚠️ Supabase: Using local document fallback")

if settings.GEMINI_API_KEY:
    print(f"✅ Gemini API: Key loaded (starts with {settings.GEMINI_API_KEY[:8]}...)")
else:
    print("❌ Gemini API: MISSING - LLM generation will fail!")

if settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY:
    print(f"✅ Langfuse: Observability enabled")
else:
    print("ℹ️ Langfuse: Disabled (optional)")

print(f"🌍 Environment: {'Production' if settings.IS_PRODUCTION else 'Development'}")
print("="*60 + "\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BACKWARDS-COMPATIBILITY EXPORTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SUPABASE_URL = settings.SUPABASE_URL
SUPABASE_KEY = settings.SUPABASE_KEY
GEMINI_API_KEY = settings.GEMINI_API_KEY
ADMIN_SECRET_KEY = settings.ADMIN_SECRET_KEY