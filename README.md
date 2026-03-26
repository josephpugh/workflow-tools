# Workflow Automation Backend

FastAPI backend for an AI-powered workflow automation service that implements the architecture described in [`docs/article.md`](/Users/joepugh/workspace/workflow-tools/docs/article.md).

The service turns natural-language requests into a structured Intermediate Request Representation (IRR), matches that IRR against a metadata-rich workflow catalog, disambiguates when multiple workflows are plausible, gathers required inputs in a white-glove conversational flow, and returns an executable contract once all required inputs are available.

## What is implemented

- FastAPI API layer for workflow interaction, session retrieval, workflow discovery, and health checks
- MCP server exposing a single `conversation_turn` tool over Streamable HTTP
- Workflow Metadata Representation (WMR) catalog stored as YAML
- Hybrid retrieval engine with:
  - semantic search using embeddings
  - BM25-style fuzzy retrieval
  - deterministic entity/domain/action/qualifier matching
  - Reciprocal Rank Fusion (RRF) to combine the rankings
- Conversation orchestration with:
  - intent classification into IRR
  - candidate retrieval via RRF
  - AI-driven workflow selection or disambiguation
  - AI-driven conversational planning for missing input turns
  - context-based prefilling
  - post-`ready` input updates such as `change effective date to ...`
  - executable contract generation
- SQLite-backed conversation persistence
- OpenAI integration for:
  - IRR generation
  - semantic embeddings
  - workflow selection vs clarification planning
  - conversational input extraction
  - natural-language assistant responses for disambiguation, input collection, and ready/update states
- Deterministic local fallback intelligence service for offline development and tests
- Full unit and integration test coverage with 11 representative workflows
- Terminal logging for request flow, workflow matching, selection, capabilities, and state transitions

## Project layout

```text
app/
  api/             FastAPI routes
  core/            settings
  db/              SQLite repository
  models/          Pydantic models
  services/        workflow registry, retrieval, orchestration, intelligence
workflows/         11 representative YAML workflow definitions
tests/             unit and integration tests
docs/article.md    original product article
```

## API surface

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

Possible response states:

- `needs_disambiguation`
- `needs_inputs`
- `ready`

When the response reaches `ready`, the payload includes an `executable_contract` with the selected workflow and gathered inputs.
If the user sends a later update such as `change effective date to 2026-05-15`, the same session can stay in `ready` while revising the gathered inputs and contract.

### `GET /api/v1/conversations/{session_id}`

Returns the persisted conversation state for auditing and debugging.

### `GET /api/v1/workflows`

Lists the workflow catalog summary.

### `GET /api/v1/workflows/{workflow_id}`

Returns one workflow summary.

### `GET /api/v1/healthz`

Health check.

## OpenAI configuration

The production service uses OpenAI for:

- IRR generation
- semantic embeddings
- workflow selection vs disambiguation
- workflow input extraction from conversation history
- conversational planning for what to ask next
- assistant response generation

Environment variables:

```bash
export OPENAI_API_KEY=...
export OPENAI_REASONING_MODEL=gpt-4.1-mini
export OPENAI_EXTRACTION_MODEL=gpt-4.1-mini
export OPENAI_EMBEDDING_MODEL=text-embedding-3-small
export LOG_LEVEL=INFO
```

If `OPENAI_API_KEY` is not set, the app falls back to a deterministic local intelligence service. That fallback is generic and schema-driven, but it exists mainly for tests and local development rather than production-quality conversation handling.

## Architecture notes

The current runtime split is:

- `WorkflowMatcher` handles retrieval and ranking only
- `ConversationOrchestrator` handles session state, persistence, coercion, and API response assembly
- `OpenAIIntelligenceService` handles:
  - intent classification
  - workflow selection planning
  - disambiguation wording
  - input extraction
  - missing-input turn planning
  - ready/update response wording

This keeps retrieval explainable while moving conversational reasoning into prompts instead of hardcoded per-workflow branches.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

The server starts on `http://127.0.0.1:8000`.
You will also see application logs in the terminal. Set `LOG_LEVEL=DEBUG` for more detail.

## Run tests

```bash
source .venv/bin/activate
pytest
```

## Example interaction

1. User: `Update David's address`
2. API: `needs_disambiguation` with mailing and billing address candidates
3. User: `Use the mailing address workflow`
4. API: `needs_inputs` and acknowledges any inputs it already inferred before asking for the remaining details
5. User: `Use 22 Broad St, Boston, MA 02110 effective April 1, 2026`
6. API: `ready` with the workflow and gathered inputs
7. User: `Change effective date to 2026-05-15`
8. API: `ready` again with an updated executable contract

## MCP tool

The application also mounts an MCP server at `/mcp` using the official Python MCP SDK (`mcp`).

Exposed tool:

- `conversation_turn`

Tool behavior:

- accepts `message`, optional `session_id`, and optional `context`
- delegates to the same backend conversation orchestration used by `POST /api/v1/conversations/turn`
- returns the full structured turn response, including disambiguation candidates, missing inputs, choices, validation state, and executable contract

Example client connection:

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
                    "message": "Update Dave Smith's mailing address to 117 Hayworth Drive",
                },
            )
            print(result)


asyncio.run(main())
```

## Notes

- This implementation intentionally stops at API-enabled backend services. It does not create MCP wrappers.
- The workflow execution layer is represented as an executable contract, not downstream system side effects.
- Date, string, and number field types are supported as requested.
- The current test suite passes with `19` tests.
