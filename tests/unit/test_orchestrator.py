from datetime import date
from pathlib import Path

from app.core.config import Settings
from app.db.repository import ConversationRepository
from app.models.domain import TurnRequest
from app.services.intelligence import HashingIntelligenceService
from app.services.orchestrator import ConversationOrchestrator
from app.services.retrieval import WorkflowMatcher
from app.services.workflow_registry import WorkflowRegistry


def build_orchestrator(tmp_path: Path) -> ConversationOrchestrator:
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'orchestrator.db'}",
        workflows_dir=Path("/Users/joepugh/workspace/workflow-tools/workflows"),
    )
    registry = WorkflowRegistry(settings.workflows_dir)
    intelligence = HashingIntelligenceService()
    repository = ConversationRepository(settings.database_url)
    matcher = WorkflowMatcher(registry=registry, intelligence=intelligence, settings=settings)
    return ConversationOrchestrator(
        repository=repository,
        registry=registry,
        intelligence=intelligence,
        matcher=matcher,
        current_date_provider=lambda: date(2026, 3, 26),
    )


def test_orchestrator_prefills_context_and_returns_batched_missing_fields(tmp_path: Path) -> None:
    orchestrator = build_orchestrator(tmp_path)
    response = orchestrator.handle_turn(
        TurnRequest(
            message="Generate a monthly portfolio report",
            context={"client_name": "Alice Johnson"},
        )
    )
    assert response.status == "needs_inputs"
    assert response.selected_workflow is not None
    assert response.selected_workflow.workflow_id == "generate_monthly_portfolio_report"
    assert response.collected_inputs["client_name"] == "Alice Johnson"
    assert set(response.missing_fields) == {"report_month", "delivery_channel"}
    assert "Alice Johnson" in response.assistant_message
    assert "month" in response.assistant_message.lower()
    assert "deliver" in response.assistant_message.lower()
    assert {field.name for field in response.requested_fields} == {"report_month", "delivery_channel"}


def test_orchestrator_resolves_relative_dates_from_reference_day(tmp_path: Path) -> None:
    orchestrator = build_orchestrator(tmp_path)
    response = orchestrator.handle_turn(
        TurnRequest(
            message="Book a client meeting with Alice Johnson next Wednesday at 11:30 for 45 minutes",
        )
    )
    assert response.status == "needs_inputs"
    assert response.selected_workflow is not None
    assert response.selected_workflow.workflow_id == "book_client_meeting"
    assert response.collected_inputs["meeting_date"] == "2026-04-01"
    assert response.collected_inputs["meeting_start_time"] == "11:30"
    assert response.collected_inputs["duration_minutes"] == 45
