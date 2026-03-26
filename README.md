# Workflow Automation Backend

FastAPI backend for an AI-powered workflow automation service. The system is built around a simple product idea: users should be able to say what they want in natural language, and the backend should figure out which workflow fits, gather the remaining information in a polished way, and stop only when it has a complete, executable contract.

In practice, that means this service does not expect users to know workflow IDs, field names, or rigid forms. It takes conversational requests, maps them onto a workflow catalog, asks clarifying questions when multiple workflows are plausible, pulls in any information already present in context, and continues the interaction until it has everything required to run the chosen workflow.

## What this system does

- Accepts natural-language user requests
- Converts them into an Intermediate Request Representation (IRR)
- Matches that IRR against a metadata-rich YAML workflow catalog
- Selects a workflow or asks a disambiguating question when needed
- Extracts inputs from the conversation and from explicit context
- Requests missing information in a calm, assistant-like tone
- Keeps `ready` sessions editable so users can revise inputs later
- Returns an executable contract once all required inputs are gathered
- Exposes the same functionality through both REST and MCP

## Current implementation

- FastAPI API layer for conversation turns, workflow discovery, health checks, and persisted session lookup
- Official MCP server using the Python MCP SDK, with one exposed tool: `conversation_turn`
- Workflow catalog stored as YAML definitions
- Hybrid workflow retrieval using:
  - semantic matching with embeddings
  - BM25-style fuzzy retrieval
  - deterministic metadata overlap on action, domain, entity, and qualifier
  - Reciprocal Rank Fusion (RRF)
- AI-driven orchestration for:
  - intent classification
  - workflow selection vs disambiguation
  - input extraction
  - conversational missing-input planning
  - natural-language response generation
- Optional capability / validator extensions for workflows
- Demonstration capability providers for meeting scheduling:
  - suggest open slots
  - validate requested slots
  - offer alternatives when a slot is unavailable
- SQLite-backed conversation persistence
- Terminal logging for request flow, matching, selection, capability execution, and state transitions
- Deterministic fallback intelligence service for offline development and tests

## Workflow model

The backend is centered on workflow definitions, not hardcoded UI forms. Each workflow defines:

- business metadata such as domain, entities, actions, and qualifiers
- required input fields with type, description, aliases, and optional context keys
- optional runtime capabilities such as suggestions or validations

Supported field types:

- `string`
- `number`
- `date`

Optional capability extensions are declarative in YAML and resolved in backend code through a provider registry. Workflows that do not declare capabilities still work normally.

## Included workflows

The repository currently includes 11 representative workflows:

- `book_client_meeting`
- `close_client_account`
- `create_wire_transfer`
- `escalate_compliance_case`
- `generate_monthly_portfolio_report`
- `lookup_client_profile`
- `open_client_account`
- `schedule_vendor_payment`
- `setup_client_subscription`
- `update_client_billing_address`
- `update_client_mailing_address`

## Conversation lifecycle

Primary conversation states:

- `needs_disambiguation`
- `needs_inputs`
- `needs_choice`
- `ready`

Typical flow:

1. User describes an outcome in natural language.
2. The backend classifies intent and retrieves candidate workflows.
3. If one workflow is clearly best, it is selected. Otherwise the backend asks a clarifying question.
4. The backend extracts known inputs from the request and any provided context.
5. Missing information is requested in grouped, natural prompts.
6. Optional validators or suggestion capabilities may run.
7. Once everything required is available, the backend returns an executable contract.
8. If the user later says something like `change effective date to 2026-05-15`, the same session can remain `ready` while updating the gathered inputs and contract.

## Architecture

Runtime responsibilities are split cleanly:

- `WorkflowRegistry`
  Loads and validates workflow YAML files.

- `WorkflowMatcher`
  Handles retrieval and ranking only.

- `ConversationOrchestrator`
  Manages session state, persistence, coercion, capability execution, and response assembly.

- `OpenAIIntelligenceService`
  Handles LLM-driven reasoning:
  - IRR generation
  - workflow selection planning
  - disambiguation wording
  - input extraction
  - missing-input planning
  - choice phrasing
  - ready/update response wording

- `CapabilityRunner`
  Executes optional workflow capabilities through providers.

- Provider registry
  Maps declarative capability names in YAML onto backend implementations.

This split keeps retrieval explainable, orchestration deterministic, and conversational reasoning prompt-driven rather than buried in per-workflow application code.

## REST API

### `POST /api/v1/conversations/turn`

Primary interaction endpoint.

Request:

```json
{
  "session_id": null,
  "message": "Update David's address",
  "context": {
    "client_name": "David"
  }
}
```

Response includes:

- `session_id`
- `status`
- `assistant_message`
- `intent`
- candidate workflows when relevant
- selected workflow when relevant
- collected inputs
- missing fields
- optional choices
- optional validation result
- executable contract when `status` is `ready`

### `GET /api/v1/conversations/{session_id}`

Returns the persisted conversation state.

### `GET /api/v1/workflows`

Lists the workflow catalog summary.

### `GET /api/v1/workflows/{workflow_id}`

Returns one workflow summary.

### `GET /api/v1/healthz`

Health check.

## MCP server

The application also mounts an MCP server at:

`/mcp`

It uses the official Python MCP SDK package:

- `mcp`

Only one tool is exposed:

- `conversation_turn`

Tool behavior:

- accepts `message`
- accepts optional `session_id`
- accepts optional `context`
- delegates to the same orchestration path as `POST /api/v1/conversations/turn`
- returns the same structured turn result

This makes the MCP surface intentionally narrow: a client only needs one tool to drive the full conversation lifecycle.

Example client usage:

```python
import asyncio

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


async def main():
    async with streamable_http_client("http://127.0.0.1:8000/mcp") as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(
                "conversation_turn",
                {
                    "message": "Update Dave Smith's mailing address to 117 Hayworth Drive"
                },
            )
            print(result)


asyncio.run(main())
```

## OpenAI usage

When `OPENAI_API_KEY` is present, the production path uses OpenAI for:

- intent classification into IRR
- semantic embeddings for workflow matching
- workflow selection vs clarification planning
- workflow input extraction
- missing-input turn planning
- conversational assistant responses

Environment variables:

```bash
export OPENAI_API_KEY=...
export OPENAI_REASONING_MODEL=gpt-5.2
export OPENAI_EXTRACTION_MODEL=gpt-5.2
export OPENAI_EMBEDDING_MODEL=text-embedding-3-small
export LOG_LEVEL=INFO
```

If `OPENAI_API_KEY` is not set, the app falls back to a deterministic local intelligence service. That fallback exists mainly for tests and offline development; the intended production experience is the OpenAI-backed path.

## Logging

The application now logs key runtime events to the terminal, including:

- incoming conversation turn requests
- session creation / continuation
- intent classification
- workflow matching and top candidates
- auto-selection vs disambiguation
- input updates and missing fields
- capability execution
- final conversation status

For more detail:

```bash
LOG_LEVEL=DEBUG uvicorn app.main:app --reload
```

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

The server starts on `http://127.0.0.1:8000`.

## Run tests

```bash
source .venv/bin/activate
pytest
```

## Example interactions

Address update:

1. User: `Update David's address`
2. API: `needs_disambiguation` with mailing and billing address candidates
3. User: `Use the mailing address workflow`
4. API: `needs_inputs` and acknowledges any details it already inferred
5. User: `Use 22 Broad St, Boston, MA 02110 effective April 1, 2026`
6. API: `ready` with the executable contract
7. User: `Change effective date to 2026-05-15`
8. API: `ready` again with an updated contract

Meeting scheduling:

1. User: `Help me book a meeting with Dave Smith for next Wednesday`
2. API: `needs_choice` with suggested meeting times for that date
3. User: `The first option works for me`
4. API: continues gathering or returns `ready`, depending on what is still missing

