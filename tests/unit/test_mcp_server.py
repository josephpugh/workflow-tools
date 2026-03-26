import asyncio
from pathlib import Path

from app.core.config import Settings
from app.db.repository import ConversationRepository
from app.mcp_server import build_mcp_server, execute_conversation_turn
from app.services.intelligence import HashingIntelligenceService
from app.services.orchestrator import ConversationOrchestrator
from app.services.retrieval import WorkflowMatcher
from app.services.workflow_registry import WorkflowRegistry


def build_orchestrator(tmp_path: Path) -> ConversationOrchestrator:
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'mcp.db'}",
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
    )


def test_mcp_server_exposes_only_conversation_turn_tool(tmp_path: Path) -> None:
    mcp_server = build_mcp_server(build_orchestrator(tmp_path))
    tools = asyncio.run(mcp_server.list_tools())
    assert [tool.name for tool in tools] == ["conversation_turn"]
    assert "full structured conversation state" in (tools[0].description or "").lower()


def test_execute_conversation_turn_reuses_rest_orchestration(tmp_path: Path) -> None:
    orchestrator = build_orchestrator(tmp_path)
    response = asyncio.run(
        execute_conversation_turn(
            orchestrator,
            message="Generate a monthly portfolio report",
            context={"client_name": "Alice Johnson"},
        )
    )
    assert response.status == "needs_inputs"
    assert response.selected_workflow is not None
    assert response.selected_workflow.workflow_id == "generate_monthly_portfolio_report"
    assert response.collected_inputs["client_name"] == "Alice Johnson"
