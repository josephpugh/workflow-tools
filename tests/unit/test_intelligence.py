from types import SimpleNamespace

from app.models.domain import RankedWorkflow, WorkflowDefinition, WorkflowSummary
from app.services.intelligence import OpenAIIntelligenceService


class FakeChatCompletions:
    def __init__(self, contents: list[str]) -> None:
        self._contents = list(contents)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self._contents.pop(0)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content),
                )
            ]
        )


class FakeEmbeddings:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3]) for _ in kwargs["input"]],
        )


class FakeOpenAIClient:
    def __init__(self, chat_contents: list[str]) -> None:
        self.chat = SimpleNamespace(completions=FakeChatCompletions(chat_contents))
        self.embeddings = FakeEmbeddings()


def build_workflow() -> WorkflowDefinition:
    return WorkflowDefinition(
        workflow_id="open_client_account",
        name="Open Client Account",
        description="Open a new client account",
        domain="client_servicing",
        entities=["client", "account"],
        actions=["open"],
        qualifiers=["new"],
    )


def build_candidate() -> RankedWorkflow:
    workflow = build_workflow()
    return RankedWorkflow(
        workflow=WorkflowSummary.from_definition(workflow),
        rrf_score=1.0,
        confidence=0.9,
        semantic_score=0.9,
        fuzzy_score=0.8,
        structured_score=0.9,
        support_count=3,
    )


def build_service(fake_client: FakeOpenAIClient) -> OpenAIIntelligenceService:
    service = OpenAIIntelligenceService(
        api_key="test",
        reasoning_model="gpt-5.4-mini",
        extraction_model="gpt-5.4-mini",
        embedding_model="text-embedding-3-small",
    )
    service._client = fake_client
    return service


def test_openai_service_uses_chat_completions_for_json_and_text_generation() -> None:
    fake_client = FakeOpenAIClient(
        chat_contents=[
            '{"action":"open","entities":["client","account"],"domain":"client_servicing","subdomain":null,"qualifiers":["new"],"context":null,"raw_text":"ignored"}',
            "Please confirm whether you want to open a new client account, and I’ll continue from there.",
        ]
    )
    service = build_service(fake_client)

    intent = service.classify_intent("Open a new client account", {}, [build_workflow()])
    message = service.build_disambiguation_message([build_candidate()], [{"role": "user", "content": "Help"}])

    assert intent.action == "open"
    assert intent.entities == ["client", "account"]
    assert intent.domain == "client_servicing"
    assert intent.raw_text == "Open a new client account"
    assert "confirm" in message.lower()

    json_call = fake_client.chat.completions.calls[0]
    assert json_call["model"] == "gpt-5.4-mini"
    assert json_call["response_format"] == {"type": "json_object"}
    assert json_call["messages"][0]["role"] == "user"

    text_call = fake_client.chat.completions.calls[1]
    assert text_call["model"] == "gpt-5.4-mini"
    assert "response_format" not in text_call
    assert text_call["messages"][0]["role"] == "user"


def test_openai_service_keeps_embeddings_on_embeddings_api() -> None:
    fake_client = FakeOpenAIClient(chat_contents=[])
    service = build_service(fake_client)

    embeddings = service.embed_texts(["alpha", "beta"])

    assert embeddings == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
    assert fake_client.embeddings.calls == [
        {
            "model": "text-embedding-3-small",
            "input": ["alpha", "beta"],
        }
    ]
