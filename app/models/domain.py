from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


FieldType = Literal["string", "number", "date"]
ConversationStatus = Literal["needs_disambiguation", "needs_inputs", "needs_choice", "ready"]


class WorkflowField(BaseModel):
    name: str
    type: FieldType
    required: bool = True
    description: str
    aliases: list[str] = Field(default_factory=list)
    context_keys: list[str] = Field(default_factory=list)


class WorkflowCapability(BaseModel):
    id: str
    type: Literal["suggestion", "validation"]
    provider: str
    field_names: list[str] = Field(default_factory=list)
    description: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class WorkflowDefinition(BaseModel):
    workflow_id: str
    name: str
    description: str
    domain: str
    subdomain: str | None = None
    entities: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    qualifiers: list[str] = Field(default_factory=list)
    canonical_intents: list[str] = Field(default_factory=list)
    trigger_utterances: list[str] = Field(default_factory=list)
    input_fields: list[WorkflowField] = Field(default_factory=list)
    capabilities: list[WorkflowCapability] = Field(default_factory=list)

    def searchable_text(self) -> str:
        parts = [
            self.name,
            self.description,
            self.domain,
            self.subdomain or "",
            " ".join(self.entities),
            " ".join(self.actions),
            " ".join(self.qualifiers),
            " ".join(self.canonical_intents),
            " ".join(self.trigger_utterances),
            " ".join(f"{field.name} {field.description} {' '.join(field.aliases)}" for field in self.input_fields),
        ]
        return " ".join(part for part in parts if part).strip()


class WorkflowSummary(BaseModel):
    workflow_id: str
    name: str
    description: str
    domain: str
    subdomain: str | None = None
    entities: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    qualifiers: list[str] = Field(default_factory=list)

    @classmethod
    def from_definition(cls, workflow: WorkflowDefinition) -> "WorkflowSummary":
        return cls(
            workflow_id=workflow.workflow_id,
            name=workflow.name,
            description=workflow.description,
            domain=workflow.domain,
            subdomain=workflow.subdomain,
            entities=workflow.entities,
            actions=workflow.actions,
            qualifiers=workflow.qualifiers,
        )


class IntermediateRequestRepresentation(BaseModel):
    action: str | None = None
    entities: list[str] = Field(default_factory=list)
    domain: str | None = None
    subdomain: str | None = None
    qualifiers: list[str] = Field(default_factory=list)
    context: str | None = None
    raw_text: str


class RankedWorkflow(BaseModel):
    workflow: WorkflowSummary
    rrf_score: float
    confidence: float
    semantic_score: float
    fuzzy_score: float
    structured_score: float
    support_count: int
    reasons: list[str] = Field(default_factory=list)


class ConversationEvent(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ExecutableContract(BaseModel):
    workflow: WorkflowSummary
    gathered_inputs: dict[str, Any]
    explanation: str


class AssistantTurnPlan(BaseModel):
    assistant_message: str
    requested_fields: list[str] = Field(default_factory=list)


class WorkflowSelectionPlan(BaseModel):
    selected_workflow_id: str | None = None
    assistant_message: str | None = None


class ChoiceOption(BaseModel):
    choice_id: str
    label: str
    value: dict[str, Any]
    source: str


class ValidationResult(BaseModel):
    provider: str
    result: Literal["passed", "failed"]
    reason: str | None = None
    field_names: list[str] = Field(default_factory=list)


class ConversationState(BaseModel):
    session_id: str
    status: ConversationStatus
    history: list[ConversationEvent] = Field(default_factory=list)
    intent: IntermediateRequestRepresentation | None = None
    candidate_workflow_ids: list[str] = Field(default_factory=list)
    selected_workflow_id: str | None = None
    collected_inputs: dict[str, Any] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)
    choices: list[ChoiceOption] = Field(default_factory=list)
    validation_result: ValidationResult | None = None
    assistant_message: str | None = None
    executable_contract: ExecutableContract | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class TurnRequest(BaseModel):
    session_id: str | None = None
    message: str
    context: dict[str, Any] = Field(default_factory=dict)


class TurnResponse(BaseModel):
    session_id: str
    status: ConversationStatus
    assistant_message: str
    intent: IntermediateRequestRepresentation | None = None
    candidate_workflows: list[RankedWorkflow] = Field(default_factory=list)
    selected_workflow: WorkflowSummary | None = None
    requested_fields: list[WorkflowField] = Field(default_factory=list)
    collected_inputs: dict[str, Any] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)
    choices: list[ChoiceOption] = Field(default_factory=list)
    validation_result: ValidationResult | None = None
    executable_contract: ExecutableContract | None = None
