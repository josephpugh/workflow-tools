from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.models.domain import ConversationState, TurnRequest, TurnResponse, WorkflowSummary
from app.services.orchestrator import ConversationOrchestrator
from app.services.workflow_registry import WorkflowRegistry

logger = logging.getLogger(__name__)


def build_router(orchestrator: ConversationOrchestrator, registry: WorkflowRegistry) -> APIRouter:
    router = APIRouter()

    @router.get("/healthz")
    def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    @router.post("/conversations/turn", response_model=TurnResponse)
    def handle_turn(request: TurnRequest) -> TurnResponse:
        logger.info("POST /conversations/turn session_id=%s", request.session_id or "<new>")
        response = orchestrator.handle_turn(request)
        logger.info(
            "POST /conversations/turn completed session_id=%s status=%s selected_workflow=%s",
            response.session_id,
            response.status,
            response.selected_workflow.workflow_id if response.selected_workflow else None,
        )
        return response

    @router.get("/conversations/{session_id}", response_model=ConversationState)
    def get_conversation(session_id: str) -> ConversationState:
        state = orchestrator.get_state(session_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return state

    @router.get("/workflows", response_model=list[WorkflowSummary])
    def list_workflows() -> list[WorkflowSummary]:
        return [WorkflowSummary.from_definition(workflow) for workflow in registry.list()]

    @router.get("/workflows/{workflow_id}", response_model=WorkflowSummary)
    def get_workflow(workflow_id: str) -> WorkflowSummary:
        try:
            workflow = registry.get(workflow_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Workflow not found") from exc
        return WorkflowSummary.from_definition(workflow)

    return router
