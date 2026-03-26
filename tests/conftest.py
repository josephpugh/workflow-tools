from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.services.intelligence import HashingIntelligenceService


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        workflows_dir=Path("/Users/joepugh/workspace/workflow-tools/workflows"),
        auto_select_confidence=0.72,
        auto_select_margin=0.15,
    )


@pytest.fixture
def client(settings: Settings) -> TestClient:
    app = create_app(
        settings=settings,
        intelligence_service=HashingIntelligenceService(),
        workflows_dir=settings.workflows_dir,
        database_url=settings.database_url,
        current_date_provider=lambda: date(2026, 3, 26),
    )
    with TestClient(app) as test_client:
        yield test_client
