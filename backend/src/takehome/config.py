from __future__ import annotations

import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://orbital:orbital@db:5432/orbital_takehome"
    anthropic_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_embedding_deployment: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072
    azure_document_intelligence_endpoint: str = ""
    azure_document_intelligence_api_key: str = ""
    upload_dir: str = "uploads"
    max_upload_size: int = 25 * 1024 * 1024  # 25MB
    rag_token_threshold: int = 50_000
    rag_top_k: int = 50

    model_config = {"env_file": ".env"}

    @property
    def embeddings_enabled(self) -> bool:
        return bool(self.azure_openai_endpoint and self.azure_openai_api_key)

    @property
    def ocr_provider(self) -> str | None:
        """Return the best available OCR provider, or None."""
        if self.azure_document_intelligence_endpoint and self.azure_document_intelligence_api_key:
            return "azure_di"
        if self.anthropic_api_key:
            return "anthropic"
        return None


settings = Settings()

if settings.anthropic_api_key:
    os.environ.setdefault("ANTHROPIC_API_KEY", settings.anthropic_api_key)
