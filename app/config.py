import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    SEARCH_KEY: str = os.getenv("SEARCH_KEY")

    # Supabase
    SUPABASE_URL: str = os.getenv("SUPABASE_URL")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY")

    # Gemini
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL_ID: str = os.getenv("GEMINI_MODEL_ID")
    GEMINI_EMBEDDING_ID: str = os.getenv("GEMINI_EMBEDDING_ID")

    # RAG
    RAG_ENABLED: bool = True
    RAG_EMBEDDING_SIZE: int = 768
    RAG_MATCH_THRESHOLD: float = 0.36
    RAG_MATCH_COUNT: int = 4

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
