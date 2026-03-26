from pathlib import Path

from app.services.workflow_registry import WorkflowRegistry


def test_registry_loads_ten_workflows() -> None:
    registry = WorkflowRegistry(Path("/Users/joepugh/workspace/workflow-tools/workflows"))
    workflows = registry.list()
    assert len(workflows) == 10
    assert registry.get("update_client_mailing_address").name == "Update Client Mailing Address"

