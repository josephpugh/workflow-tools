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

_WORKFLOW_SWITCH_PHRASES = (
    "actually",
    "instead",
    "never mind",
    "different workflow",
    "not this one",
    "not this workflow",
    "i want to",
    "i need to",
    "switch to",
)
_WORKFLOW_SWITCH_GENERIC_TOKENS = {
    "a",
    "an",
    "and",
    "client",
    "create",
    "do",
    "for",
    "help",
    "i",
    "me",
    "my",
    "need",
    "new",
    "please",
    "set",
    "setup",
    "something",
    "switch",
    "the",
    "to",
    "up",
    "want",
    "workflow",
}


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
        if not candidates:
            return self._return_unsupported(state, intent)
        if self._matcher.should_auto_select(candidates):
            selected_workflow_id = candidates[0].workflow.workflow_id
            logger.info("Workflow auto-selected session_id=%s workflow=%s", state.session_id, selected_workflow_id)
            state.selected_workflow_id = selected_workflow_id
            state.candidate_workflow_ids = []
            state.candidate_input_overrides = {}
            return self._continue_input_collection(state, request.message, request.context, candidates=candidates)

        top_candidates = candidates[:3]
        state.candidate_workflow_ids = [candidate.workflow.workflow_id for candidate in top_candidates]
        state.candidate_input_overrides = {}
        logger.info(
            "Workflow disambiguation required session_id=%s candidates=%s",
            state.session_id,
            [candidate.workflow.workflow_id for candidate in top_candidates],
        )
        assistant_message = self._intelligence.build_disambiguation_message(
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
        current_candidates = self._current_disambiguation_candidates(state)
        resolution = self._intelligence.resolve_disambiguation_turn(
            candidates=current_candidates,
            history=[{"role": item.role, "content": item.content} for item in state.history],
            latest_user_message=request.message,
        )
        if resolution.decision == "select" and resolution.selected_workflow_id in set(state.candidate_workflow_ids):
            logger.info(
                "Resolved disambiguation via intelligence session_id=%s workflow=%s",
                state.session_id,
                resolution.selected_workflow_id,
            )
            self._apply_selected_workflow(state, resolution.selected_workflow_id, state.intent)
            return self._continue_input_collection(state, request.message, request.context)

        latest_intent = self._intelligence.classify_intent(request.message, request.context, self._registry.list())
        if resolution.decision == "restart":
            logger.info("Disambiguation follow-up introduced a materially new request session_id=%s", state.session_id)
            return self._rematch_disambiguation_intent(
                state=state,
                request=request,
                intent=latest_intent,
                log_message="Retrying workflow match across full catalog after disambiguation restart session_id=%s",
            )

        merged_intent = self._merge_intents(state.intent, latest_intent, request.message)
        candidates = self._matcher.match(
            merged_intent,
            restrict_to=set(state.candidate_workflow_ids),
        )
        if candidates:
            return self._return_disambiguation_candidates(
                state=state,
                request=request,
                intent=merged_intent,
                candidates=candidates,
            )

        logger.info("Restricted disambiguation match failed session_id=%s; retrying full catalog", state.session_id)
        return self._rematch_disambiguation_intent(
            state=state,
            request=request,
            intent=latest_intent,
            log_message="Retrying workflow match across full catalog after restricted miss session_id=%s",
        )

    def _rematch_disambiguation_intent(
        self,
        state: ConversationState,
        request: TurnRequest,
        intent: IntermediateRequestRepresentation,
        log_message: str,
    ) -> TurnResponse:
        logger.info(log_message, state.session_id)
        candidates = self._matcher.match(intent)
        if not candidates:
            return self._return_unsupported(state, intent)
        return self._return_disambiguation_candidates(
            state=state,
            request=request,
            intent=intent,
            candidates=candidates,
        )

    def _return_disambiguation_candidates(
        self,
        state: ConversationState,
        request: TurnRequest,
        intent: IntermediateRequestRepresentation,
        candidates: list[RankedWorkflow],
    ) -> TurnResponse:
        state.intent = intent
        if self._matcher.should_auto_select(candidates):
            selected_workflow_id = candidates[0].workflow.workflow_id
            logger.info("Workflow selected after disambiguation session_id=%s workflow=%s", state.session_id, selected_workflow_id)
            self._apply_selected_workflow(state, selected_workflow_id, intent)
            return self._continue_input_collection(state, request.message, request.context, candidates=candidates)

        top_candidates = candidates[:3]
        state.candidate_workflow_ids = [candidate.workflow.workflow_id for candidate in top_candidates]
        state.candidate_input_overrides = {}
        logger.info(
            "Disambiguation still unresolved session_id=%s candidates=%s",
            state.session_id,
            [candidate.workflow.workflow_id for candidate in top_candidates],
        )
        assistant_message = self._intelligence.build_disambiguation_message(
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

    def _merge_intents(
        self,
        previous_intent: IntermediateRequestRepresentation | None,
        latest_intent: IntermediateRequestRepresentation,
        latest_message: str,
    ) -> IntermediateRequestRepresentation:
        if previous_intent is None:
            return latest_intent
        return IntermediateRequestRepresentation(
            action=latest_intent.action or previous_intent.action,
            entities=list(dict.fromkeys([*previous_intent.entities, *latest_intent.entities])),
            domain=latest_intent.domain or previous_intent.domain,
            subdomain=latest_intent.subdomain or previous_intent.subdomain,
            qualifiers=list(dict.fromkeys([*previous_intent.qualifiers, *latest_intent.qualifiers])),
            context=latest_intent.context or previous_intent.context,
            raw_text=f"{previous_intent.raw_text} {latest_message}".strip(),
        )

    def _current_disambiguation_candidates(self, state: ConversationState) -> list[RankedWorkflow]:
        if state.intent is None or not state.candidate_workflow_ids:
            return []
        return self._matcher.match(
            state.intent,
            top_k=len(state.candidate_workflow_ids),
            restrict_to=set(state.candidate_workflow_ids),
        )

    def _return_unsupported(
        self,
        state: ConversationState,
        intent: IntermediateRequestRepresentation,
    ) -> TurnResponse:
        assistant_message = (
            "I’m sorry, I couldn’t identify a workflow that matches your request. "
            "If you believe a supported workflow exists, please try rewording your request and I’ll check again."
        )
        state.status = "unsupported"
        state.candidate_workflow_ids = []
        state.candidate_input_overrides = {}
        state.selected_workflow_id = None
        state.assistant_message = assistant_message
        state.history.append(ConversationEvent(role="assistant", content=assistant_message))
        return TurnResponse(
            session_id=state.session_id,
            status=state.status,
            assistant_message=assistant_message,
            intent=intent,
            candidate_workflows=[],
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

        missing_fields_before_collection = [
            field.name
            for field in workflow.input_fields
            if field.required and field.name not in state.collected_inputs
        ]
        switch_response = self._maybe_switch_workflow(
            state=state,
            workflow=workflow,
            latest_user_message=latest_user_message,
            context=context,
            missing_fields=missing_fields_before_collection,
        )
        if switch_response is not None:
            return switch_response

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

    def _maybe_switch_workflow(
        self,
        state: ConversationState,
        workflow: WorkflowDefinition,
        latest_user_message: str,
        context: dict[str, Any],
        missing_fields: list[str],
    ) -> TurnResponse | None:
        if not self._should_probe_workflow_switch(workflow, latest_user_message, missing_fields):
            return None

        logger.info("Possible workflow switch detected session_id=%s current_workflow=%s", state.session_id, workflow.workflow_id)
        intent = self._intelligence.classify_intent(latest_user_message, context, self._registry.list())
        candidates = self._matcher.match(intent)
        if not candidates:
            logger.info("Workflow switch probe returned no candidates session_id=%s", state.session_id)
            return self._return_unsupported(state, intent)

        top_workflow_id = candidates[0].workflow.workflow_id
        if self._matcher.should_auto_select(candidates):
            if top_workflow_id == workflow.workflow_id:
                logger.info("Workflow switch probe resolved back to current workflow session_id=%s workflow=%s", state.session_id, workflow.workflow_id)
                state.intent = intent
                return None

            logger.info(
                "Workflow switch probe selected new workflow session_id=%s from=%s to=%s",
                state.session_id,
                workflow.workflow_id,
                top_workflow_id,
            )
            self._apply_selected_workflow(state, top_workflow_id, intent)
            return self._continue_input_collection(state, latest_user_message, context, candidates=candidates)

        logger.info(
            "Workflow switch probe returned multiple candidates session_id=%s candidates=%s",
            state.session_id,
            [candidate.workflow.workflow_id for candidate in candidates[:3]],
        )
        return self._start_disambiguation(
            state=state,
            intent=intent,
            candidates=candidates,
            previous_inputs=state.collected_inputs,
        )

    def _should_probe_workflow_switch(
        self,
        workflow: WorkflowDefinition,
        latest_user_message: str,
        missing_fields: list[str],
    ) -> bool:
        normalized = latest_user_message.lower()
        explicit_switch = any(phrase in normalized for phrase in _WORKFLOW_SWITCH_PHRASES)
        current_field_signal = self._looks_like_current_workflow_input(workflow, latest_user_message, missing_fields)
        other_workflow_signal = self._other_workflow_signal_score(workflow.workflow_id, latest_user_message) >= 2
        return (explicit_switch and other_workflow_signal) or (other_workflow_signal and not current_field_signal)

    def _looks_like_current_workflow_input(
        self,
        workflow: WorkflowDefinition,
        latest_user_message: str,
        missing_fields: list[str],
    ) -> bool:
        normalized = latest_user_message.lower()
        missing_field_lookup = {field.name: field for field in workflow.input_fields if field.name in set(missing_fields)}
        if not missing_field_lookup:
            return False

        for field in missing_field_lookup.values():
            labels = {field.name.lower(), *(alias.lower() for alias in field.aliases), *(key.lower() for key in field.context_keys)}
            if any(label.replace("_", " ") in normalized for label in labels):
                return True

        if any(field.type == "date" for field in missing_field_lookup.values()):
            if resolve_date_expression(latest_user_message, self._current_date_provider()) is not None:
                return True
            if any(token in normalized for token in ("today", "tomorrow", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")):
                return True

        if any(field.type == "number" for field in missing_field_lookup.values()) and any(char.isdigit() for char in latest_user_message):
            return True

        if any("email" in field.name.lower() or "email" in field.description.lower() for field in missing_field_lookup.values()) and "@" in latest_user_message:
            return True

        address_fields = {"street_address", "city", "state", "postal_code"}
        if address_fields.intersection(missing_field_lookup) and any(
            token in normalized for token in ("street", " st", "road", " rd", "avenue", " ave", "drive", " dr", "lane", " ln", "court", " ct", ",")
        ):
            return True

        return False

    def _other_workflow_signal_score(self, current_workflow_id: str, latest_user_message: str) -> int:
        normalized = latest_user_message.lower().replace("-", " ").replace("_", " ")
        message_tokens = {
            token
            for token in normalized.split()
            if len(token) > 2 and token not in _WORKFLOW_SWITCH_GENERIC_TOKENS
        }
        current_vocabulary = self._workflow_switch_vocabulary(self._registry.get(current_workflow_id))
        best_score = 0
        for candidate in self._registry.list():
            if candidate.workflow_id == current_workflow_id:
                continue
            candidate_vocabulary = self._workflow_switch_vocabulary(candidate) - current_vocabulary
            token_score = sum(1 for token in candidate_vocabulary if token in message_tokens)
            phrase_score = sum(
                2
                for phrase in self._workflow_switch_phrases(candidate)
                if phrase and phrase not in current_vocabulary and phrase in normalized
            )
            best_score = max(best_score, token_score + phrase_score)
        return best_score

    def _workflow_switch_vocabulary(self, workflow: WorkflowDefinition) -> set[str]:
        vocabulary = {
            token
            for token in [*workflow.entities, *workflow.qualifiers]
            if len(token) > 2 and token not in _WORKFLOW_SWITCH_GENERIC_TOKENS
        }
        vocabulary.update(
            token
            for token in workflow.workflow_id.lower().split("_")
            if len(token) > 2 and token not in _WORKFLOW_SWITCH_GENERIC_TOKENS
        )
        vocabulary.update(
            token
            for token in workflow.name.lower().replace("-", " ").split()
            if len(token) > 2 and token not in _WORKFLOW_SWITCH_GENERIC_TOKENS
        )
        return vocabulary

    def _workflow_switch_phrases(self, workflow: WorkflowDefinition) -> set[str]:
        return {
            workflow.workflow_id.replace("_", " ").lower(),
            workflow.name.lower(),
            *(entity.replace("_", " ").lower() for entity in workflow.entities),
            *(qualifier.replace("_", " ").lower() for qualifier in workflow.qualifiers),
        }

    def _start_disambiguation(
        self,
        state: ConversationState,
        intent: IntermediateRequestRepresentation,
        candidates: list[RankedWorkflow],
        previous_inputs: dict[str, Any] | None = None,
    ) -> TurnResponse:
        top_candidates = candidates[:3]
        state.intent = intent
        state.selected_workflow_id = None
        state.candidate_workflow_ids = [candidate.workflow.workflow_id for candidate in top_candidates]
        preserved_inputs = previous_inputs or {}
        candidate_workflows = [self._registry.get(candidate.workflow.workflow_id) for candidate in top_candidates]
        state.candidate_input_overrides = {
            workflow.workflow_id: self._preserve_matching_inputs(preserved_inputs, workflow)
            for workflow in candidate_workflows
        }
        state.collected_inputs = self._common_preserved_inputs(preserved_inputs, candidate_workflows)
        state.missing_fields = []
        state.choices = []
        state.validation_result = None
        state.executable_contract = None
        assistant_message = self._intelligence.build_disambiguation_message(
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

    def _apply_selected_workflow(
        self,
        state: ConversationState,
        workflow_id: str,
        intent: IntermediateRequestRepresentation | None,
    ) -> None:
        workflow = self._registry.get(workflow_id)
        preserved_inputs = state.candidate_input_overrides.get(workflow_id)
        if preserved_inputs is None:
            preserved_inputs = self._preserve_matching_inputs(state.collected_inputs, workflow)
        state.intent = intent
        state.selected_workflow_id = workflow_id
        state.candidate_workflow_ids = []
        state.candidate_input_overrides = {}
        state.collected_inputs = preserved_inputs
        state.missing_fields = []
        state.choices = []
        state.validation_result = None
        state.executable_contract = None

    def _preserve_matching_inputs(
        self,
        existing_inputs: dict[str, Any],
        workflow: WorkflowDefinition,
    ) -> dict[str, Any]:
        allowed_fields = {field.name for field in workflow.input_fields}
        return self._coerce_inputs(
            workflow,
            {key: value for key, value in existing_inputs.items() if key in allowed_fields},
        )

    def _common_preserved_inputs(
        self,
        existing_inputs: dict[str, Any],
        workflows: list[WorkflowDefinition],
    ) -> dict[str, Any]:
        if not workflows:
            return {}
        common_fields = set.intersection(*(set(field.name for field in workflow.input_fields) for workflow in workflows))
        common_payload = {key: value for key, value in existing_inputs.items() if key in common_fields}
        return self._coerce_inputs(workflows[0], common_payload)

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
