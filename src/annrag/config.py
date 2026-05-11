"""Project-wide configuration, loaded from environment / .env via Pydantic."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LLMProvider = Literal["anthropic", "ollama"]

# Project root resolves relative to this file so the defaults remain stable
# regardless of where the process is launched from.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration sourced from environment variables and `.env`.

    All fields have sensible defaults so the project can be imported without
    a populated `.env`. Secrets default to `None`; the components that need
    them raise loudly if invoked without one.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="ANNRAG_",
        extra="ignore",
        frozen=True,
        populate_by_name=True,
    )

    # ---- paths ------------------------------------------------------------
    data_dir: Path = Field(
        default=_PROJECT_ROOT / "data",
        description="Root for raw + processed Wikivoyage data (XML dump, chunks, etc).",
    )
    artifacts_dir: Path = Field(
        default=_PROJECT_ROOT / "artifacts",
        description="Root for derived artifacts: indexes, embeddings, eval CSVs.",
    )

    # ---- runtime ----------------------------------------------------------
    log_level: LogLevel = Field(
        default="INFO",
        description="Root logger level; structured handler is installed by logging_setup.",
    )
    random_seed: int = Field(
        default=42,
        description="Seed used wherever sampling is involved, for reproducibility.",
        ge=0,
    )

    # ---- LLM provider selection -------------------------------------------
    llm_provider: LLMProvider = Field(
        default="ollama",
        description="Which provider to use for Q&A generation and RAG answer synthesis.",
    )

    # ---- LLM: Anthropic ---------------------------------------------------
    # Reads ANTHROPIC_API_KEY (SDK convention) first, ANNRAG_ANTHROPIC_API_KEY second.
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        description="Anthropic API key. Read first from ANTHROPIC_API_KEY (SDK convention).",
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "ANNRAG_ANTHROPIC_API_KEY"),
    )
    anthropic_model: str = Field(
        default="claude-sonnet-4-6",
        description="Default Anthropic model for Q&A generation and RAG answer synthesis.",
    )

    # ---- LLM: Ollama (local) ----------------------------------------------
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for the local Ollama daemon.",
    )
    ollama_model: str = Field(
        default="llama3:8b",
        description="Default Ollama model tag (e.g. 'llama3:8b', 'mistral:7b').",
    )
    ollama_request_timeout_s: float = Field(
        default=300.0,
        description="HTTP timeout for Ollama calls (local generation can be slow).",
        ge=1.0,
    )

    # ---- Wikivoyage -------------------------------------------------------
    wikivoyage_lang: str = Field(
        default="en",
        description="Wikivoyage language edition ('en', 'fr', 'de', ...).",
        min_length=2,
        max_length=8,
    )
    wikivoyage_user_agent: str = Field(
        default=(
            "annrag-coursework/0.1 (+https://meta.wikimedia.org/wiki/User-Agent_policy; coursework)"
        ),
        description=(
            "HTTP User-Agent. MediaWiki policy requires a URL or email contact. "
            "Override via .env for production / extended use."
        ),
    )
    wikivoyage_request_delay_s: float = Field(
        default=0.5,
        description="Minimum seconds between consecutive API calls, to be a polite citizen.",
        ge=0.0,
    )

    # ---- Ground-truth defaults --------------------------------------------
    gt_questions_per_article: int = Field(
        default=5,
        description="How many Q&A candidates to generate per seed article.",
        ge=1,
        le=20,
    )

    @field_validator("data_dir", "artifacts_dir")
    @classmethod
    def _expand_and_resolve(cls, value: Path) -> Path:
        return value.expanduser().resolve()
