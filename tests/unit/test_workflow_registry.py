from pathlib import Path

import pytest

from app.models.domain import WorkflowCapability, WorkflowDefinition
from app.services.providers.registry import ProviderRegistry
from app.services.workflow_registry import WorkflowRegistry


def test_registry_loads_ten_workflows() -> None:
    registry = WorkflowRegistry(Path("/Users/joepugh/workspace/workflow-tools/workflows"))
    workflows = registry.list()
    assert len(workflows) == 11
    assert registry.get("update_client_mailing_address").name == "Update Client Mailing Address"
    assert registry.get("setup_client_subscription").name == "Set Up Client Subscription"


def test_provider_registry_validates_declared_workflow_capabilities() -> None:
    registry = WorkflowRegistry(Path("/Users/joepugh/workspace/workflow-tools/workflows"))
    ProviderRegistry().validate_workflows(registry.list())


def test_provider_registry_rejects_unknown_provider() -> None:
    workflow = WorkflowDefinition(
        workflow_id="example",
        name="Example",
        description="Example workflow",
        domain="operations",
        capabilities=[
            WorkflowCapability(
                id="missing_provider",
                type="suggestion",
                provider="missing.provider",
                field_names=["meeting_date"],
            )
        ],
    )
    with pytest.raises(ValueError, match="missing.provider"):
        ProviderRegistry().validate_workflows([workflow])
