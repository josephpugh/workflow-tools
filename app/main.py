from __future__ import annotations

from datetime import date
import logging
from pathlib import Path
from typing import Callable

from fastapi import FastAPI

from app.api.routes import build_router
from app.core.config import Settings, get_settings
from app.core.logging import setup_logging
from app.db.repository import ConversationRepository
from app.services.intelligence import HashingIntelligenceService, IntelligenceService, OpenAIIntelligenceService
from app.services.orchestrator import ConversationOrchestrator
from app.services.providers.registry import ProviderRegistry
from app.services.retrieval import WorkflowMatcher
from app.services.workflow_registry import WorkflowRegistry
from app.services.capability_runner import CapabilityRunner

logger = logging.getLogger(__name__)


def build_intelligence_service(settings: Settings) -> IntelligenceService:
    if settings.openai_api_key:
        logger.info(
            "Using OpenAI intelligence service (reasoning_model=%s, extraction_model=%s, embedding_model=%s)",
            settings.openai_reasoning_model,
            settings.openai_extraction_model,
            settings.openai_embedding_model,
        )
        return OpenAIIntelligenceService(
            api_key=settings.openai_api_key,
            reasoning_model=settings.openai_reasoning_model,
            extraction_model=settings.openai_extraction_model,
            embedding_model=settings.openai_embedding_model,
        )
    logger.info("Using deterministic local intelligence service")
    return HashingIntelligenceService()


def create_app(
    settings: Settings | None = None,
    intelligence_service: IntelligenceService | None = None,
    workflows_dir: Path | None = None,
    database_url: str | None = None,
    current_date_provider: Callable[[], date] | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()
    setup_logging(resolved_settings.log_level)
    registry = WorkflowRegistry(workflows_dir or resolved_settings.workflows_dir)
    provider_registry = ProviderRegistry()
    provider_registry.validate_workflows(registry.list())
    intelligence = intelligence_service or build_intelligence_service(resolved_settings)
    repository = ConversationRepository(database_url or resolved_settings.database_url)
    matcher = WorkflowMatcher(registry=registry, intelligence=intelligence, settings=resolved_settings)
    orchestrator = ConversationOrchestrator(
        repository=repository,
        registry=registry,
        intelligence=intelligence,
        matcher=matcher,
        capability_runner=CapabilityRunner(provider_registry=provider_registry),
        current_date_provider=current_date_provider,
    )

    app = FastAPI(title=resolved_settings.app_name)
    app.include_router(build_router(orchestrator, registry), prefix=resolved_settings.api_prefix)
    logger.info(
        "Application initialized (workflows=%s, database=%s, log_level=%s)",
        len(registry.list()),
        resolved_settings.database_url,
        resolved_settings.log_level.upper(),
    )
    return app


app = create_app()
