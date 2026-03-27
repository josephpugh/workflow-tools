from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Settings:
    app_name: str = "Workflow Automation API"
    api_prefix: str = "/api/v1"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_reasoning_model: str = os.getenv("OPENAI_REASONING_MODEL", "gpt-5.2")
    openai_extraction_model: str = os.getenv("OPENAI_EXTRACTION_MODEL", "gpt-5.2")
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./workflow_automation.db")
    workflows_dir: Path = Path(os.getenv("WORKFLOWS_DIR", "./workflows"))
    auto_select_confidence: float = float(os.getenv("AUTO_SELECT_CONFIDENCE", "0.72"))
    auto_select_margin: float = float(os.getenv("AUTO_SELECT_MARGIN", "0.15"))
    min_grounding_overlap: int = int(os.getenv("MIN_GROUNDING_OVERLAP", "2"))
    rrf_k: int = int(os.getenv("RRF_K", "60"))


def get_settings() -> Settings:
    return Settings()
