from __future__ import annotations

import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from app.models.domain import TurnRequest, TurnResponse
from app.services.orchestrator import ConversationOrchestrator

logger = logging.getLogger(__name__)


async def execute_conversation_turn(
    orchestrator: ConversationOrchestrator,
    *,
    message: str,
    session_id: str | None = None,
    context: dict[str, Any] | None = None,
) -> TurnResponse:
    logger.info("MCP conversation_turn called session_id=%s", session_id or "<new>")
    response = orchestrator.handle_turn(
        TurnRequest(
            session_id=session_id,
            message=message,
            context=context or {},
        )
    )
    logger.info(
        "MCP conversation_turn completed session_id=%s status=%s selected_workflow=%s",
        response.session_id,
        response.status,
        response.selected_workflow.workflow_id if response.selected_workflow else None,
    )
    return response


def build_mcp_server(
    orchestrator: ConversationOrchestrator,
    *,
    log_level: str = "INFO",
) -> FastMCP:
    mcp = FastMCP(
        name="Workflow Automation MCP",
        instructions=(
            "Use the conversation_turn tool to submit one conversational turn to the workflow "
            "automation backend. Reuse the returned session_id on follow-up turns so the "
            "conversation can continue across disambiguation, input collection, validation, "
            "and ready states."
        ),
        json_response=True,
        log_level=log_level.upper(),
        streamable_http_path="/",
    )

    @mcp.tool(
        name="conversation_turn",
        title="Conversation Turn",
        description=(
            "Submit one user message to the workflow automation conversation engine and return "
            "the full structured conversation state, including session_id, status, matched "
            "workflow candidates, gathered inputs, requested fields, optional choices, and the "
            "executable contract when the workflow is ready."
        ),
        structured_output=True,
    )
    async def conversation_turn(
        message: str,
        session_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> TurnResponse:
        """Handle one workflow-automation conversation turn.

        Args:
            message: The latest user message to process.
            session_id: Existing session identifier for follow-up turns. Omit for a new conversation.
            context: Optional structured context to prefill workflow inputs.
        """

        return await execute_conversation_turn(
            orchestrator,
            message=message,
            session_id=session_id,
            context=context,
        )

    return mcp
