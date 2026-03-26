from __future__ import annotations

from datetime import UTC, date, datetime
import logging
from typing import Any, Callable
from uuid import uuid4

from app.core.date_utils import resolve_date_expression
from app.db.repository import ConversationRepository
from app.models.domain import (
    ChoiceOption,
    ConversationEvent,
    ConversationState,
    ExecutableContract,
    IntermediateRequestRepresentation,
    RankedWorkflow,
    TurnRequest,
    TurnResponse,
    ValidationResult,
    WorkflowDefinition,
    WorkflowField,
    WorkflowSummary,
)
from app.services.capability_runner import CapabilityRunner
from app.services.intelligence import IntelligenceService
from app.services.retrieval import WorkflowMatcher
from app.services.workflow_registry import WorkflowRegistry

logger = logging.getLogger(__name__)


class ConversationOrchestrator:
    def __init__(
        self,
        repository: ConversationRepository,
        registry: WorkflowRegistry,
        intelligence: IntelligenceService,
        matcher: WorkflowMatcher,
        capability_runner: CapabilityRunner | None = None,
        current_date_provider: Callable[[], date] | None = None,
    ) -> None:
        self._repository = repository
        self._registry = registry
        self._intelligence = intelligence
        self._matcher = matcher
        self._capability_runner = capability_runner or CapabilityRunner()
        self._current_date_provider = current_date_provider or (lambda: datetime.now().date())

    def handle_turn(self, request: TurnRequest) -> TurnResponse:
        state = self._repository.load(request.session_id) if request.session_id else None
        if state is None:
            state = ConversationState(session_id=str(uuid4()), status="needs_inputs")
            logger.info("Starting new conversation session_id=%s", state.session_id)
        else:
            logger.info("Continuing conversation session_id=%s status=%s", state.session_id, state.status)
        state.history.append(ConversationEvent(role="user", content=request.message))
        state.updated_at = datetime.now(UTC)

        if state.selected_workflow_id:
            response = self._continue_input_collection(state, request.message, request.context)
        elif state.candidate_workflow_ids:
            response = self._resolve_disambiguation(state, request)
        else:
            response = self._start_matching(state, request)

        self._repository.save(state)
        return response

    def get_state(self, session_id: str) -> ConversationState | None:
        return self._repository.load(session_id)

    def _start_matching(self, state: ConversationState, request: TurnRequest) -> TurnResponse:
        intent = self._intelligence.classify_intent(request.message, request.context, self._registry.list())
        state.intent = intent
        logger.info(
            "Intent classified session_id=%s action=%s domain=%s entities=%s qualifiers=%s",
            state.session_id,
            intent.action,
            intent.domain,
            intent.entities,
            intent.qualifiers,
        )
        candidates = self._matcher.match(intent)
        state.candidate_workflow_ids = [candidate.workflow.workflow_id for candidate in candidates]
        selection = self._intelligence.plan_workflow_selection(
            candidates=candidates,
            history=[{"role": item.role, "content": item.content} for item in state.history],
        )
        if selection.selected_workflow_id:
            logger.info("Workflow auto-selected session_id=%s workflow=%s", state.session_id, selection.selected_workflow_id)
            state.selected_workflow_id = selection.selected_workflow_id
            state.candidate_workflow_ids = []
            return self._continue_input_collection(state, request.message, request.context, candidates=candidates)

        top_candidates = candidates[:3]
        logger.info(
            "Workflow disambiguation required session_id=%s candidates=%s",
            state.session_id,
            [candidate.workflow.workflow_id for candidate in top_candidates],
        )
        assistant_message = selection.assistant_message or self._intelligence.build_disambiguation_message(
            top_candidates,
            [{"role": item.role, "content": item.content} for item in state.history],
        )
        state.status = "needs_disambiguation"
        state.assistant_message = assistant_message
        state.history.append(ConversationEvent(role="assistant", content=assistant_message))
        return TurnResponse(
            session_id=state.session_id,
            status=state.status,
            assistant_message=assistant_message,
            intent=intent,
            candidate_workflows=top_candidates,
            collected_inputs=state.collected_inputs,
            missing_fields=state.missing_fields,
        )

    def _resolve_disambiguation(self, state: ConversationState, request: TurnRequest) -> TurnResponse:
        intent = self._intelligence.classify_intent(request.message, request.context, self._registry.list())
        if state.intent is not None:
            merged_intent = IntermediateRequestRepresentation(
                action=intent.action or state.intent.action,
                entities=list(dict.fromkeys([*state.intent.entities, *intent.entities])),
                domain=intent.domain or state.intent.domain,
                subdomain=intent.subdomain or state.intent.subdomain,
                qualifiers=list(dict.fromkeys([*state.intent.qualifiers, *intent.qualifiers])),
                context=intent.context or state.intent.context,
                raw_text=f"{state.intent.raw_text} {request.message}".strip(),
            )
        else:
            merged_intent = intent
        state.intent = merged_intent

        candidates = self._matcher.match(
            merged_intent,
            restrict_to=set(state.candidate_workflow_ids),
        )
        selection = self._intelligence.plan_workflow_selection(
            candidates=candidates,
            history=[{"role": item.role, "content": item.content} for item in state.history],
        )
        if selection.selected_workflow_id:
            logger.info("Workflow selected after disambiguation session_id=%s workflow=%s", state.session_id, selection.selected_workflow_id)
            state.selected_workflow_id = selection.selected_workflow_id
            state.candidate_workflow_ids = []
            return self._continue_input_collection(state, request.message, request.context, candidates=candidates)

        logger.info(
            "Disambiguation still unresolved session_id=%s candidates=%s",
            state.session_id,
            [candidate.workflow.workflow_id for candidate in candidates[:3]],
        )
        assistant_message = selection.assistant_message or self._intelligence.build_disambiguation_message(
            candidates[:3],
            [{"role": item.role, "content": item.content} for item in state.history],
        )
        state.status = "needs_disambiguation"
        state.assistant_message = assistant_message
        state.history.append(ConversationEvent(role="assistant", content=assistant_message))
        return TurnResponse(
            session_id=state.session_id,
            status=state.status,
            assistant_message=assistant_message,
            intent=merged_intent,
            candidate_workflows=candidates[:3],
            collected_inputs=state.collected_inputs,
            missing_fields=state.missing_fields,
        )

    def _continue_input_collection(
        self,
        state: ConversationState,
        latest_user_message: str,
        context: dict[str, Any],
        candidates: list[RankedWorkflow] | None = None,
    ) -> TurnResponse:
        workflow = self._registry.get(state.selected_workflow_id or candidates[0].workflow.workflow_id)
        state.selected_workflow_id = workflow.workflow_id
        previous_inputs = dict(state.collected_inputs)
        is_first_prompt_after_selection = not previous_inputs
        logger.info("Collecting inputs session_id=%s workflow=%s", state.session_id, workflow.workflow_id)

        selected_choice = self._capability_runner.resolve_choice(state.choices, latest_user_message) if state.choices else None
        if selected_choice is not None:
            logger.info("Resolved user choice session_id=%s workflow=%s choice_id=%s", state.session_id, workflow.workflow_id, selected_choice.choice_id)
            state.collected_inputs = {**state.collected_inputs, **selected_choice.value}
            previous_inputs = dict(previous_inputs)
            changed_fields = [
                key for key, value in selected_choice.value.items() if previous_inputs.get(key) != value
            ]
            state.choices = []
            state.validation_result = None
        else:
            changed_fields = []

        gathered = self._prefill_from_context(workflow, context)
        extracted = self._intelligence.extract_inputs(
            workflow=workflow,
            history=[{"role": item.role, "content": item.content} for item in state.history],
            latest_user_message=latest_user_message,
            use_full_history=is_first_prompt_after_selection,
            current_date=self._current_date_provider(),
            context=context,
            existing_inputs={**state.collected_inputs, **gathered},
        )
        normalized = self._coerce_inputs(workflow, {**extracted, **gathered})
        state.collected_inputs = {**state.collected_inputs, **normalized}
        changed_fields.extend([key for key, value in normalized.items() if previous_inputs.get(key) != value and key not in changed_fields])
        logger.info(
            "Updated collected inputs session_id=%s workflow=%s changed_fields=%s missing_fields=%s",
            state.session_id,
            workflow.workflow_id,
            changed_fields,
            [
                field.name
                for field in workflow.input_fields
                if field.required and field.name not in state.collected_inputs
            ],
        )

        missing_fields = [
            field.name
            for field in workflow.input_fields
            if field.required and field.name not in state.collected_inputs
        ]
        state.missing_fields = missing_fields
        reference_date = self._current_date_provider()

        suggestion_result = self._capability_runner.run_suggestions(
            workflow=workflow,
            collected_inputs=state.collected_inputs,
            missing_fields=missing_fields,
            reference_date=reference_date,
        )
        if suggestion_result is not None and suggestion_result.choices:
            logger.info("Returning needs_choice from suggestion session_id=%s workflow=%s", state.session_id, workflow.workflow_id)
            assistant_message = self._intelligence.build_choice_message(
                workflow=workflow,
                choices=suggestion_result.choices,
                collected_inputs=state.collected_inputs,
            )
            state.status = "needs_choice"
            state.choices = suggestion_result.choices
            state.validation_result = None
            state.assistant_message = assistant_message
            state.history.append(ConversationEvent(role="assistant", content=assistant_message))
            return TurnResponse(
                session_id=state.session_id,
                status=state.status,
                assistant_message=assistant_message,
                intent=state.intent,
                candidate_workflows=candidates or [],
                selected_workflow=WorkflowSummary.from_definition(workflow),
                collected_inputs=state.collected_inputs,
                missing_fields=missing_fields,
                choices=state.choices,
                validation_result=state.validation_result,
            )

        validation_execution = self._capability_runner.validate(
            workflow=workflow,
            collected_inputs=state.collected_inputs,
            changed_fields=changed_fields,
            reference_date=reference_date,
        )
        if validation_execution is not None:
            state.validation_result = validation_execution.validation_result
            if validation_execution.validation_result and validation_execution.validation_result.result == "failed":
                logger.info("Returning needs_choice from validation failure session_id=%s workflow=%s", state.session_id, workflow.workflow_id)
                state.status = "needs_choice"
                state.choices = validation_execution.choices
                assistant_message = self._intelligence.build_choice_message(
                    workflow=workflow,
                    choices=validation_execution.choices,
                    collected_inputs=state.collected_inputs,
                    validation_result=validation_execution.validation_result,
                )
                state.assistant_message = assistant_message
                state.history.append(ConversationEvent(role="assistant", content=assistant_message))
                return TurnResponse(
                    session_id=state.session_id,
                    status=state.status,
                    assistant_message=assistant_message,
                    intent=state.intent,
                    candidate_workflows=candidates or [],
                    selected_workflow=WorkflowSummary.from_definition(workflow),
                    collected_inputs=state.collected_inputs,
                    missing_fields=missing_fields,
                    choices=state.choices,
                    validation_result=state.validation_result,
                )
            state.validation_result = validation_execution.validation_result
        else:
            state.validation_result = None
        state.choices = []

        if missing_fields:
            logger.info("Returning needs_inputs session_id=%s workflow=%s missing_fields=%s", state.session_id, workflow.workflow_id, missing_fields)
            plan = self._intelligence.plan_missing_input_turn(
                workflow,
                history=[{"role": item.role, "content": item.content} for item in state.history],
                collected_inputs=state.collected_inputs,
                missing_fields=missing_fields,
                first_prompt_after_selection=is_first_prompt_after_selection,
            )
            requested_field_names = [
                field_name for field_name in plan.requested_fields if field_name in set(missing_fields)
            ]
            if not requested_field_names:
                requested_field_names = missing_fields[: min(3, len(missing_fields))]
            assistant_message = plan.assistant_message
            state.status = "needs_inputs"
            state.assistant_message = assistant_message
            state.history.append(ConversationEvent(role="assistant", content=assistant_message))
            return TurnResponse(
                session_id=state.session_id,
                status=state.status,
                assistant_message=assistant_message,
                intent=state.intent,
                candidate_workflows=candidates or [],
                selected_workflow=WorkflowSummary.from_definition(workflow),
                requested_fields=[field for field in workflow.input_fields if field.name in requested_field_names],
                collected_inputs=state.collected_inputs,
                missing_fields=missing_fields,
                choices=state.choices,
                validation_result=state.validation_result,
            )

        contract = ExecutableContract(
            workflow=WorkflowSummary.from_definition(workflow),
            gathered_inputs=state.collected_inputs,
            explanation=f"Run {workflow.name} with the gathered inputs below.",
        )
        logger.info("Workflow ready session_id=%s workflow=%s", state.session_id, workflow.workflow_id)
        state.status = "ready"
        state.assistant_message = self._intelligence.build_ready_message(
            workflow=workflow,
            collected_inputs=state.collected_inputs,
            changed_fields=changed_fields,
        )
        state.executable_contract = contract
        state.history.append(ConversationEvent(role="assistant", content=state.assistant_message))
        return TurnResponse(
            session_id=state.session_id,
            status=state.status,
            assistant_message=state.assistant_message,
            intent=state.intent,
            candidate_workflows=candidates or [],
            selected_workflow=WorkflowSummary.from_definition(workflow),
            collected_inputs=state.collected_inputs,
            missing_fields=[],
            choices=state.choices,
            validation_result=state.validation_result,
            executable_contract=contract,
        )

    def _prefill_from_context(self, workflow: WorkflowDefinition, context: dict[str, Any]) -> dict[str, Any]:
        prefilled: dict[str, Any] = {}
        normalized_context = {key.lower(): value for key, value in context.items()}
        for field in workflow.input_fields:
            candidate_keys = {field.name.lower(), *(alias.lower() for alias in field.aliases), *(key.lower() for key in field.context_keys)}
            for key in candidate_keys:
                if key in normalized_context:
                    prefilled[field.name] = normalized_context[key]
                    break
        return self._coerce_inputs(workflow, prefilled)

    def _coerce_inputs(self, workflow: WorkflowDefinition, payload: dict[str, Any]) -> dict[str, Any]:
        coerced: dict[str, Any] = {}
        field_lookup = {field.name: field for field in workflow.input_fields}
        for field_name, value in payload.items():
            field = field_lookup.get(field_name)
            if field is None or value in (None, ""):
                continue
            normalized = self._coerce_value(field, value)
            if normalized is not None:
                coerced[field_name] = normalized
        return coerced

    def _coerce_value(self, field: WorkflowField, value: Any) -> Any:
        if field.type == "string":
            return self._normalize_string_value(field, str(value).strip())
        if field.type == "number":
            raw = str(value).replace(",", "").replace("$", "").strip()
            try:
                number = float(raw)
            except ValueError:
                return None
            return int(number) if number.is_integer() else number
        if field.type == "date":
            text = str(value).strip()
            for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y", "%b %d, %Y", "%Y/%m/%d"):
                try:
                    return datetime.strptime(text, fmt).date().isoformat()
                except ValueError:
                    continue
            relative = resolve_date_expression(text, self._current_date_provider())
            if relative is not None:
                return relative
            return None
        return None

    def _normalize_string_value(self, field: WorkflowField, value: str) -> str:
        if not value:
            return value
        if not self._should_sentence_case(field, value):
            return value
        for index, char in enumerate(value):
            if char.isalpha():
                return f"{value[:index]}{char.upper()}{value[index + 1:]}"
        return value

    def _should_sentence_case(self, field: WorkflowField, value: str) -> bool:
        descriptor = " ".join([field.name, field.description, *field.aliases]).lower()
        if not any(keyword in descriptor for keyword in {"agenda", "subject", "summary", "reason", "description", "notes"}):
            return False
        letters = [char for char in value if char.isalpha()]
        if not letters:
            return False
        return value == value.lower()
