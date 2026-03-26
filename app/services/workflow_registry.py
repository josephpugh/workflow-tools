from __future__ import annotations

from pathlib import Path

import yaml

from app.models.domain import WorkflowDefinition


class WorkflowRegistry:
    def __init__(self, workflows_dir: Path) -> None:
        self._workflows_dir = workflows_dir
        self._workflows = self._load()

    def _load(self) -> dict[str, WorkflowDefinition]:
        workflows: dict[str, WorkflowDefinition] = {}
        for path in sorted(self._workflows_dir.glob("*.yaml")):
            workflow = WorkflowDefinition.model_validate(yaml.safe_load(path.read_text()))
            if workflow.workflow_id in workflows:
                raise ValueError(f"Duplicate workflow id found: {workflow.workflow_id}")
            workflows[workflow.workflow_id] = workflow
        if not workflows:
            raise ValueError(f"No workflow definitions found in {self._workflows_dir}")
        return workflows

    def list(self) -> list[WorkflowDefinition]:
        return list(self._workflows.values())

    def get(self, workflow_id: str) -> WorkflowDefinition:
        try:
            return self._workflows[workflow_id]
        except KeyError as exc:
            raise KeyError(f"Unknown workflow: {workflow_id}") from exc

