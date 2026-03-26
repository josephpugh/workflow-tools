from __future__ import annotations

import json
import math
import re
from abc import ABC, abstractmethod
from datetime import date
from typing import Any

from openai import OpenAI

from app.core.date_utils import extract_date_expression
from app.models.domain import (
    AssistantTurnPlan,
    ChoiceOption,
    IntermediateRequestRepresentation,
    RankedWorkflow,
    ValidationResult,
    WorkflowDefinition,
    WorkflowSelectionPlan,
)


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", cleaned)
        cleaned = re.sub(r"\n```$", "", cleaned)
    return cleaned.strip()


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = _strip_code_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(cleaned[start : end + 1])


class IntelligenceService(ABC):
    @abstractmethod
    def classify_intent(
        self,
        text: str,
        context: dict[str, Any],
        workflows: list[WorkflowDefinition],
    ) -> IntermediateRequestRepresentation:
        raise NotImplementedError

    @abstractmethod
    def extract_inputs(
        self,
        workflow: WorkflowDefinition,
        history: list[dict[str, str]],
        latest_user_message: str,
        use_full_history: bool,
        current_date: date,
        context: dict[str, Any],
        existing_inputs: dict[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    @abstractmethod
    def build_disambiguation_message(
        self,
        candidates: list[RankedWorkflow],
        history: list[dict[str, str]],
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def plan_workflow_selection(
        self,
        candidates: list[RankedWorkflow],
        history: list[dict[str, str]],
    ) -> WorkflowSelectionPlan:
        raise NotImplementedError

    @abstractmethod
    def plan_missing_input_turn(
        self,
        workflow: WorkflowDefinition,
        history: list[dict[str, str]],
        collected_inputs: dict[str, Any],
        missing_fields: list[str],
        first_prompt_after_selection: bool,
    ) -> AssistantTurnPlan:
        raise NotImplementedError

    @abstractmethod
    def build_ready_message(
        self,
        workflow: WorkflowDefinition,
        collected_inputs: dict[str, Any],
        changed_fields: list[str],
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def build_choice_message(
        self,
        workflow: WorkflowDefinition,
        choices: list[ChoiceOption],
        collected_inputs: dict[str, Any],
        validation_result: ValidationResult | None = None,
    ) -> str:
        raise NotImplementedError


class OpenAIIntelligenceService(IntelligenceService):
    def __init__(
        self,
        api_key: str,
        reasoning_model: str,
        extraction_model: str,
        embedding_model: str,
    ) -> None:
        self._client = OpenAI(api_key=api_key)
        self._reasoning_model = reasoning_model
        self._extraction_model = extraction_model
        self._embedding_model = embedding_model

    def classify_intent(
        self,
        text: str,
        context: dict[str, Any],
        workflows: list[WorkflowDefinition],
    ) -> IntermediateRequestRepresentation:
        catalog = {
            "domains": sorted({workflow.domain for workflow in workflows}),
            "subdomains": sorted({workflow.subdomain for workflow in workflows if workflow.subdomain}),
            "actions": sorted({action for workflow in workflows for action in workflow.actions}),
            "entities": sorted({entity for workflow in workflows for entity in workflow.entities}),
            "qualifiers": sorted({qualifier for workflow in workflows for qualifier in workflow.qualifiers}),
        }
        prompt = f"""
You are classifying a workflow automation request into an Intermediate Request Representation.
Return JSON with keys:
- action: string or null
- entities: array of strings
- domain: string or null
- subdomain: string or null
- qualifiers: array of strings
- context: string or null
- raw_text: string

Use the workflow catalog vocabulary where possible.
Catalog vocabulary: {json.dumps(catalog)}
Conversation context: {json.dumps(context)}
User request: {text}
""".strip()
        response = self._client.responses.create(model=self._reasoning_model, input=prompt)
        payload = _extract_json(response.output_text)
        payload["raw_text"] = text
        return IntermediateRequestRepresentation.model_validate(payload)

    def extract_inputs(
        self,
        workflow: WorkflowDefinition,
        history: list[dict[str, str]],
        latest_user_message: str,
        use_full_history: bool,
        current_date: date,
        context: dict[str, Any],
        existing_inputs: dict[str, Any],
    ) -> dict[str, Any]:
        field_schema = [
            {
                "name": field.name,
                "type": field.type,
                "description": field.description,
                "aliases": field.aliases,
            }
            for field in workflow.input_fields
        ]
        prompt = f"""
Extract workflow inputs from the conversation.
Return a JSON object containing only recognized workflow field values.
Do not invent values. Prefer exact dates in YYYY-MM-DD and plain numbers for numeric fields.

Workflow:
{json.dumps({"workflow_id": workflow.workflow_id, "name": workflow.name, "description": workflow.description, "fields": field_schema})}

Existing inputs:
{json.dumps(existing_inputs)}

Latest user message:
{latest_user_message}

Use full history for extraction:
{json.dumps(use_full_history)}

Current date:
{current_date.isoformat()}

Context:
{json.dumps(context)}

Conversation:
{json.dumps(history)}
""".strip()
        response = self._client.responses.create(model=self._extraction_model, input=prompt)
        return _extract_json(response.output_text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self._embedding_model, input=texts)
        return [record.embedding for record in response.data]

    def build_disambiguation_message(
        self,
        candidates: list[RankedWorkflow],
        history: list[dict[str, str]],
    ) -> str:
        prompt = f"""
You are a calm, friendly personal assistant helping a user choose the correct workflow.
Write one short professional message asking the user to choose between the candidate workflows.
Do not use bullet points.
Make it clear that once the user confirms, you will continue.

Candidate workflows:
{json.dumps([candidate.model_dump(mode="json") for candidate in candidates])}

Conversation:
{json.dumps(history)}
""".strip()
        response = self._client.responses.create(model=self._reasoning_model, input=prompt)
        return response.output_text.strip()

    def plan_workflow_selection(
        self,
        candidates: list[RankedWorkflow],
        history: list[dict[str, str]],
    ) -> WorkflowSelectionPlan:
        prompt = f"""
You are deciding whether a workflow can be selected confidently from retrieved candidates.
Return JSON with:
- selected_workflow_id: string or null
- assistant_message: string or null

Rules:
- Select a workflow only if the user intent is clear from the conversation and one candidate is the best fit.
- If clarification is needed, selected_workflow_id must be null and assistant_message must ask a calm, professional clarifying question.
- Base your decision only on the conversation and the candidate workflows provided.
- Do not invent workflow IDs.

Candidates:
{json.dumps([candidate.model_dump(mode="json") for candidate in candidates])}

Conversation:
{json.dumps(history)}
""".strip()
        response = self._client.responses.create(model=self._reasoning_model, input=prompt)
        payload = _extract_json(response.output_text)
        selected_workflow_id = payload.get("selected_workflow_id")
        valid_ids = {candidate.workflow.workflow_id for candidate in candidates}
        if selected_workflow_id not in valid_ids:
            selected_workflow_id = None
        assistant_message = payload.get("assistant_message")
        if assistant_message is not None:
            assistant_message = str(assistant_message).strip() or None
        if selected_workflow_id is None and assistant_message is None:
            assistant_message = self.build_disambiguation_message(candidates[:3], history)
        return WorkflowSelectionPlan(
            selected_workflow_id=selected_workflow_id,
            assistant_message=assistant_message,
        )

    def plan_missing_input_turn(
        self,
        workflow: WorkflowDefinition,
        history: list[dict[str, str]],
        collected_inputs: dict[str, Any],
        missing_fields: list[str],
        first_prompt_after_selection: bool,
    ) -> AssistantTurnPlan:
        prompt = f"""
You are a calm, friendly personal assistant collecting the remaining information needed to run a workflow.
Return JSON with:
- assistant_message: string
- requested_fields: array of field names

Rules:
- requested_fields must be a non-empty subset of missing_fields.
- Choose a natural subset to ask for now; do not ask for everything if that would feel clumsy.
- On the first prompt after the workflow is selected, naturally acknowledge any useful collected inputs.
- On later prompts, do not repeat all known details unless directly helpful.
- The message should be concise, professional, and conversational.
- Do not mention internal terms like "field", "workflow metadata", or "schema".

Workflow:
{json.dumps(workflow.model_dump(mode="json"))}

Collected inputs:
{json.dumps(collected_inputs)}

Missing fields:
{json.dumps(missing_fields)}

First prompt after selection:
{json.dumps(first_prompt_after_selection)}

Conversation:
{json.dumps(history)}
""".strip()
        response = self._client.responses.create(model=self._reasoning_model, input=prompt)
        payload = _extract_json(response.output_text)
        requested_fields = [
            field_name for field_name in payload.get("requested_fields", []) if field_name in set(missing_fields)
        ]
        assistant_message = str(payload.get("assistant_message", "")).strip()
        if not requested_fields:
            requested_fields = missing_fields[: min(3, len(missing_fields))]
        if not assistant_message:
            assistant_message = "Of course. Please share the remaining details when you're ready."
        return AssistantTurnPlan(
            assistant_message=assistant_message,
            requested_fields=requested_fields,
        )

    def build_ready_message(
        self,
        workflow: WorkflowDefinition,
        collected_inputs: dict[str, Any],
        changed_fields: list[str],
    ) -> str:
        prompt = f"""
You are a calm, friendly personal assistant.
Write one short professional message for a workflow that is ready to submit.
If changed_fields is non-empty, acknowledge that those details were updated and say the executable contract reflects the latest information.
If changed_fields is empty, say everything is ready and invite the user to review the executable contract.

Workflow:
{json.dumps({"workflow_id": workflow.workflow_id, "name": workflow.name, "description": workflow.description})}

Collected inputs:
{json.dumps(collected_inputs)}

Changed fields:
{json.dumps(changed_fields)}
        """.strip()
        response = self._client.responses.create(model=self._reasoning_model, input=prompt)
        return response.output_text.strip()

    def build_choice_message(
        self,
        workflow: WorkflowDefinition,
        choices: list[ChoiceOption],
        collected_inputs: dict[str, Any],
        validation_result: ValidationResult | None = None,
    ) -> str:
        prompt = f"""
You are a calm, friendly personal assistant.
Write one short message presenting the available choices to the user.
If validation_result indicates failure, acknowledge the issue and then offer the alternatives.
The message should be concise, professional, and conversational.
Do not use bullet points.

Workflow:
{json.dumps(workflow.model_dump(mode="json"))}

Collected inputs:
{json.dumps(collected_inputs)}

Validation result:
{json.dumps(validation_result.model_dump(mode="json") if validation_result else None)}

Choices:
{json.dumps([choice.model_dump(mode="json") for choice in choices])}
""".strip()
        response = self._client.responses.create(model=self._reasoning_model, input=prompt)
        return response.output_text.strip()


class HashingIntelligenceService(IntelligenceService):
    """Deterministic local implementation used for tests and offline development."""

    _ACTION_KEYWORDS = {
        "update": ["update", "change", "modify"],
        "lookup": ["lookup", "find", "show", "view", "check", "status"],
        "open": ["open", "create"],
        "close": ["close", "terminate"],
        "create": ["create", "send", "transfer"],
        "schedule": ["schedule", "pay"],
        "book": ["book", "schedule", "arrange"],
        "generate": ["generate", "produce"],
        "escalate": ["escalate", "raise"],
    }

    def classify_intent(
        self,
        text: str,
        context: dict[str, Any],
        workflows: list[WorkflowDefinition],
    ) -> IntermediateRequestRepresentation:
        normalized = text.lower()
        action = None
        for candidate, keywords in self._ACTION_KEYWORDS.items():
            if any(keyword in normalized for keyword in keywords):
                action = candidate
                break
        entities = []
        vocabulary = sorted({entity for workflow in workflows for entity in workflow.entities}, key=len, reverse=True)
        for entity in vocabulary:
            if entity.replace("_", " ") in normalized:
                entities.append(entity)
        qualifiers = []
        qualifier_vocab = sorted({qualifier for workflow in workflows for qualifier in workflow.qualifiers})
        for qualifier in qualifier_vocab:
            if qualifier.replace("_", " ") in normalized:
                qualifiers.append(qualifier)
        if "address" in normalized and "client" not in entities:
            entities.append("client")
            entities.append("address")
        if "payment" in normalized:
            entities.extend(entity for entity in ["payment", "invoice"] if entity not in entities and entity in {"payment", "invoice"})
        if "report" in normalized and "report" not in entities:
            entities.append("report")
        if "meeting" in normalized and "meeting" not in entities:
            entities.append("meeting")

        domain = None
        if any(token in normalized for token in ["client", "address", "account", "meeting", "portfolio"]):
            domain = "client_servicing"
        if any(token in normalized for token in ["payment", "wire", "invoice"]):
            domain = "operations"
        if "compliance" in normalized or "case" in normalized:
            domain = "compliance"
        if "employee" in normalized or "leave" in normalized:
            domain = "hr"

        return IntermediateRequestRepresentation(
            action=action,
            entities=list(dict.fromkeys(entities)),
            domain=domain,
            qualifiers=qualifiers,
            context=context.get("context_name"),
            raw_text=text,
        )

    def extract_inputs(
        self,
        workflow: WorkflowDefinition,
        history: list[dict[str, str]],
        latest_user_message: str,
        use_full_history: bool,
        current_date: date,
        context: dict[str, Any],
        existing_inputs: dict[str, Any],
    ) -> dict[str, Any]:
        user_messages = [item["content"] for item in history if item["role"] == "user"]
        full_text = " ".join(user_messages)
        text = full_text if use_full_history else latest_user_message
        search_messages = user_messages if use_full_history else [latest_user_message]
        extracted: dict[str, Any] = {}
        address_components = self._extract_address_components(search_messages)

        for field in workflow.input_fields:
            value = self._extract_field_value(
                field=field,
                text=text,
                full_text=full_text,
                search_messages=search_messages,
                existing_inputs=existing_inputs,
                address_components=address_components,
            )
            if value not in (None, ""):
                extracted[field.name] = value

        for field in workflow.input_fields:
            for key in [field.name, *field.aliases, *field.context_keys]:
                if key in context and field.name not in extracted:
                    extracted[field.name] = context[key]

        return {key: value for key, value in extracted.items() if value not in ("", None)}

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * 64
            for token in re.findall(r"[a-z0-9_]+", text.lower()):
                slot = hash(token) % len(vector)
                vector[slot] += 1.0
            magnitude = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / magnitude for value in vector])
        return vectors

    def build_disambiguation_message(
        self,
        candidates: list[RankedWorkflow],
        history: list[dict[str, str]],
    ) -> str:
        options = "; ".join(
            f"{candidate.workflow.name} ({candidate.workflow.description})" for candidate in candidates
        )
        return (
            "I found a few likely options and wanted to confirm the right one with you: "
            f"{options}. Once you confirm, I'll take it from there."
        )

    def plan_workflow_selection(
        self,
        candidates: list[RankedWorkflow],
        history: list[dict[str, str]],
    ) -> WorkflowSelectionPlan:
        if not candidates:
            return WorkflowSelectionPlan(
                selected_workflow_id=None,
                assistant_message="I couldn't find a suitable workflow yet. Please rephrase the request.",
            )
        if len(candidates) == 1:
            return WorkflowSelectionPlan(selected_workflow_id=candidates[0].workflow.workflow_id)
        top = candidates[0]
        second = candidates[1]
        if (
            top.confidence >= 0.95
            and top.semantic_score >= 0.95
            and top.fuzzy_score >= 0.95
            and top.structured_score >= 0.85
            and (top.confidence - second.confidence) >= 0.10
            and top.support_count >= 2
        ):
            return WorkflowSelectionPlan(selected_workflow_id=top.workflow.workflow_id)
        if (
            top.confidence >= 0.72
            and (top.confidence - second.confidence) >= 0.15
            and top.support_count >= 2
        ):
            return WorkflowSelectionPlan(selected_workflow_id=top.workflow.workflow_id)
        return WorkflowSelectionPlan(
            selected_workflow_id=None,
            assistant_message=self.build_disambiguation_message(candidates[:3], history),
        )

    def plan_missing_input_turn(
        self,
        workflow: WorkflowDefinition,
        history: list[dict[str, str]],
        collected_inputs: dict[str, Any],
        missing_fields: list[str],
        first_prompt_after_selection: bool,
    ) -> AssistantTurnPlan:
        requested_fields = self._choose_requested_fields(workflow, missing_fields)
        preface = self._build_collected_input_preface(
            workflow=workflow,
            collected_inputs=collected_inputs,
            first_prompt_after_selection=first_prompt_after_selection,
        )
        request_text = self._build_request_text(workflow, requested_fields)
        assistant_message = f"{preface} {request_text}".strip() if preface else f"Of course. {request_text}"
        return AssistantTurnPlan(
            assistant_message=assistant_message,
            requested_fields=requested_fields,
        )

    def build_ready_message(
        self,
        workflow: WorkflowDefinition,
        collected_inputs: dict[str, Any],
        changed_fields: list[str],
    ) -> str:
        if changed_fields:
            return (
                f"Absolutely. I've updated {', '.join(changed_fields)} for {workflow.name.lower()}, "
                "and the executable contract below reflects the latest details."
            )
        return (
            f"Everything is in place for {workflow.name}. Please review the executable contract below, "
            "and when it looks right, you can submit it."
        )

    def build_choice_message(
        self,
        workflow: WorkflowDefinition,
        choices: list[ChoiceOption],
        collected_inputs: dict[str, Any],
        validation_result: ValidationResult | None = None,
    ) -> str:
        labels = [choice.label for choice in choices]
        if not labels:
            return "I couldn't find any suitable options just yet."
        if len(labels) == 1:
            options_text = labels[0]
        elif len(labels) == 2:
            options_text = f"{labels[0]} or {labels[1]}"
        else:
            options_text = f"{', '.join(labels[:-1])}, or {labels[-1]}"
        if validation_result and validation_result.result == "failed":
            return f"That time is already booked. The closest available options are {options_text}."
        return f"I found a few available options for you: {options_text}. Let me know which one you prefer."

    def _choose_requested_fields(self, workflow: WorkflowDefinition, missing_fields: list[str]) -> list[str]:
        address_fields = [field_name for field_name in ["street_address", "city", "state", "postal_code"] if field_name in missing_fields]
        if address_fields:
            if len(address_fields) == 1:
                date_fields = [field.name for field in workflow.input_fields if field.name in missing_fields and field.type == "date"]
                if date_fields:
                    return [address_fields[0], date_fields[0]]
            return address_fields
        if len(missing_fields) <= 3:
            return missing_fields
        prioritized: list[str] = []
        for field in workflow.input_fields:
            if field.name not in missing_fields:
                continue
            if field.type in {"date", "number"} or "name" in field.name or "time" in field.name:
                prioritized.append(field.name)
            if len(prioritized) >= 3:
                return prioritized
        return missing_fields[: min(3, len(missing_fields))]

    def _build_collected_input_preface(
        self,
        workflow: WorkflowDefinition,
        collected_inputs: dict[str, Any],
        first_prompt_after_selection: bool,
    ) -> str | None:
        if not first_prompt_after_selection:
            return None

        subject = self._workflow_subject(workflow)
        summary_bits = self._summarize_collected_inputs(workflow, collected_inputs)
        if not summary_bits:
            return f"Of course. I can help with the {subject}."
        if len(summary_bits) == 1:
            return f"Of course. I can help with the {subject}, and I already have {summary_bits[0]}."
        return f"Of course. I can help with the {subject}, and I already have {summary_bits[0]} and {summary_bits[1]}."

    def _build_request_text(self, workflow: WorkflowDefinition, requested_fields: list[str]) -> str:
        address_fields = {"street_address", "city", "state", "postal_code"}
        requested = set(requested_fields)
        if requested.intersection(address_fields):
            missing_address = [field for field in ["street_address", "city", "state", "postal_code"] if field in requested]
            address_parts = {
                "street_address": "street address",
                "city": "city",
                "state": "state",
                "postal_code": "postal code",
            }
            date_fields = [field.name for field in workflow.input_fields if field.name in requested and field.type == "date"]
            address_request = ""
            if set(missing_address) == address_fields:
                address_request = "Please let me know the full address, including street, city, state, and postal code."
            elif len(missing_address) == 1:
                address_request = f"Please let me know the {address_parts[missing_address[0]]}."
            elif len(missing_address) == 2:
                address_request = (
                    f"Please let me know the {address_parts[missing_address[0]]} "
                    f"and {address_parts[missing_address[1]]}."
                )
            else:
                pieces = [address_parts[field_name] for field_name in missing_address]
                address_request = f"Please let me know the {', '.join(pieces[:-1])}, and {pieces[-1]}."
            if date_fields:
                return f"{address_request} Please also let me know {self._humanize_field_name(date_fields[0])}."
            return address_request

        field_lookup = {field.name: field for field in workflow.input_fields}
        labels = [self._humanize_field_name(name, field_lookup.get(name)) for name in requested_fields if name in field_lookup]
        if not labels:
            return "Please share the remaining details when you're ready."
        if len(labels) == 1:
            return f"Please let me know {labels[0]}."
        if len(labels) == 2:
            return f"Please let me know {labels[0]} and {labels[1]}."
        return f"Please let me know {', '.join(labels[:-1])}, and {labels[-1]}."

    def _extract_field_value(
        self,
        field: Any,
        text: str,
        full_text: str,
        search_messages: list[str],
        existing_inputs: dict[str, Any],
        address_components: dict[str, str],
    ) -> str | None:
        field_name = field.name.lower()
        descriptor = " ".join([field.name, field.description, *field.aliases]).lower()

        if field_name in address_components:
            return address_components[field_name]

        if field.type == "date":
            if "month" in descriptor:
                report_month_match = re.search(r"\b(20\d{2}-\d{2})\b", text)
                if report_month_match:
                    return f"{report_month_match.group(1)}-01"
            return extract_date_expression(text)

        if field.type == "number":
            duration_match = re.search(r"\b(\d+)\s*(minutes|minute|mins|min)\b", text, re.IGNORECASE)
            if duration_match and ("duration" in field_name or "duration" in descriptor):
                return duration_match.group(1)
            amount_match = re.search(r"\$?(\d[\d,]*(?:\.\d+)?)", text)
            return amount_match.group(1).replace(",", "") if amount_match else None

        if "time" in field_name or ("start" in descriptor and "time" in descriptor):
            time_match = re.search(r"\b(\d{1,2}:\d{2})\b", text)
            if time_match:
                hours, minutes = time_match.group(1).split(":")
                return f"{int(hours):02d}:{minutes}"

        if "delivery" in descriptor or "channel" in descriptor:
            delivery_channel_match = re.search(r"\b(email|portal|print|mail)\b", text, re.IGNORECASE)
            return delivery_channel_match.group(1).lower() if delivery_channel_match else None

        if field_name == "meeting_format" or ("meeting format" in descriptor):
            format_match = re.search(r"\b(virtual|in-person|in person|phone)\b", text, re.IGNORECASE)
            if format_match:
                value = format_match.group(1).lower()
                return "in-person" if value == "in person" else value

        if "agenda" in descriptor:
            agenda_match = re.search(r"(?:agenda|about|to discuss)\s+(.+)$", text, re.IGNORECASE)
            return agenda_match.group(1).strip() if agenda_match else None

        if "invoice" in descriptor:
            invoice_match = re.search(r"\b([A-Z]{2,}-\d+)\b", text)
            return invoice_match.group(1) if invoice_match else None

        if "case" in descriptor and "id" in descriptor:
            case_match = re.search(r"\bcase\s+([A-Z0-9-]+)\b", text, re.IGNORECASE)
            if case_match:
                return case_match.group(1)
            generic_id_match = re.search(r"\b([A-Z]{2,}-\d+)\b", text)
            return generic_id_match.group(1) if generic_id_match else None

        if "source_account" in field_name or ("source" in descriptor and "account" in descriptor):
            account_match = re.search(r"from\s+([A-Za-z0-9 ]+?account)\b", text, re.IGNORECASE)
            return account_match.group(1).strip() if account_match else None

        if "account_number" in field_name:
            account_number_match = re.search(r"\b(\d{6,})\b", text)
            return account_number_match.group(1) if account_number_match else None

        if "account_type" in field_name:
            account_type_match = re.search(r"\b(brokerage|ira|checking|savings|investment)\b", text, re.IGNORECASE)
            return account_type_match.group(1).lower() if account_type_match else None

        if "beneficiary" in descriptor:
            beneficiary_match = re.search(r"to\s+([A-Z][A-Za-z0-9&. ]+?)(?:\s+on\b|\s+from\b|\s+for\b|$)", text)
            return beneficiary_match.group(1).strip() if beneficiary_match else None

        if "payee" in descriptor or "vendor" in descriptor or "supplier" in descriptor:
            payee_match = re.search(r"to\s+([A-Z][A-Za-z0-9&. ]+?)(?:\s+on\b|\s+from\b|\s+for\b|$)", text)
            return payee_match.group(1).strip() if payee_match else None

        if "name" in descriptor:
            if "client" in descriptor or "customer" in descriptor:
                return self._extract_person_name(full_text, existing_inputs)
            person_match = re.search(r"\b([A-Z][A-Za-z0-9&.]+(?:\s+[A-Z][A-Za-z0-9&.]+)*)\b", text)
            if person_match:
                return person_match.group(1).strip()

        if "reason" in descriptor:
            reason_match = re.search(r"(?:because|reason is|reason:)\s+(.+)$", text, re.IGNORECASE)
            return reason_match.group(1).strip() if reason_match else None

        return None

    def _extract_address_components(self, search_messages: list[str]) -> dict[str, str]:
        components: dict[str, str] = {}
        for message in search_messages:
            address_match = re.search(
                r"(?P<street>\d+\s+[^,]+),\s*(?P<city>[A-Za-z .]+),\s*(?P<state>[A-Z]{2})\s+(?P<postal>\d{5})",
                message,
            )
            location_only_match = re.search(
                r"(?:to|in)\s+(?P<city>[A-Z][A-Za-z .]+),\s*(?P<state>[A-Z]{2}),?\s*(?P<postal>\d{5})",
                message,
                re.IGNORECASE,
            )
            street_only_match = re.search(
                r"(?:to|at|is)\s+(?P<street>\d+\s+[A-Za-z0-9 .]+?)(?=,|\s+effective\b|\s+and\b|$)",
                message,
                re.IGNORECASE,
            )
            if address_match:
                components["street_address"] = address_match.group("street").strip()
                components["city"] = address_match.group("city").strip()
                components["state"] = address_match.group("state").strip()
                components["postal_code"] = address_match.group("postal").strip()
                break
            if location_only_match:
                components["city"] = location_only_match.group("city").strip()
                components["state"] = location_only_match.group("state").strip()
                components["postal_code"] = location_only_match.group("postal").strip()
            if street_only_match and "street_address" not in components:
                components["street_address"] = street_only_match.group("street").strip()
        return components

    def _extract_person_name(self, full_text: str, existing_inputs: dict[str, Any]) -> str | None:
        if "client_name" in existing_inputs:
            return None
        patterns = [
            r"(?i:\b(?:update|open|close|lookup|find|show|generate|book|schedule|create)\b)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"(?i:\bfor\b)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"(?i:\bwith\b)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        ]
        for pattern in patterns:
            match = re.search(pattern, full_text)
            if match:
                return match.group(1).removesuffix("'s").strip()
        return None

    def _summarize_collected_inputs(
        self,
        workflow: WorkflowDefinition,
        collected_inputs: dict[str, Any],
    ) -> list[str]:
        summary_bits: list[str] = []
        client_name = collected_inputs.get("client_name")
        if client_name:
            summary_bits.append(str(client_name))

        address_bits = [
            str(collected_inputs[field_name]).strip()
            for field_name in ["street_address", "city", "state", "postal_code"]
            if field_name in collected_inputs
        ]
        if address_bits:
            summary_bits.append(", ".join(address_bits))

        if len(summary_bits) < 2:
            for field in workflow.input_fields:
                if field.name in {"client_name", "street_address", "city", "state", "postal_code"}:
                    continue
                if field.name in collected_inputs:
                    summary_bits.append(f"{self._humanize_field_name(field.name, field)} as {collected_inputs[field.name]}")
                if len(summary_bits) >= 2:
                    break
        return summary_bits[:2]

    def _humanize_field_name(self, field_name: str, field: Any | None = None) -> str:
        overrides = {
            "effective_date": "the date the change should take effect",
            "report_month": "which month the report should cover",
            "delivery_channel": "how you would like it delivered",
            "payment_date": "the requested payment date",
            "transfer_date": "the requested transfer date",
            "meeting_date": "the preferred meeting date",
            "meeting_start_time": "the meeting start time in HH:MM format",
            "duration_minutes": "the meeting duration in minutes",
            "meeting_format": "whether you would like it to be virtual or in person",
            "source_account": "which account should fund it",
        }
        if field_name in overrides:
            return overrides[field_name]
        if field is not None and getattr(field, "description", None):
            return str(field.description).rstrip(".").lower()
        return field_name.replace("_", " ")

    def _workflow_subject(self, workflow: WorkflowDefinition) -> str:
        subject = workflow.name.lower()
        for prefix in ["update ", "generate ", "create ", "open ", "close ", "lookup ", "book ", "schedule ", "escalate "]:
            if subject.startswith(prefix):
                return subject.removeprefix(prefix)
        return subject
