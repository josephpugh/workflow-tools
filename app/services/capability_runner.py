from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import logging
from typing import Any

from app.models.domain import ChoiceOption, ValidationResult, WorkflowCapability, WorkflowDefinition
from app.services.providers.registry import ProviderRegistry

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CapabilityExecutionResult:
    choices: list[ChoiceOption]
    validation_result: ValidationResult | None = None


class CapabilityRunner:
    def __init__(self, provider_registry: ProviderRegistry | None = None) -> None:
        self._provider_registry = provider_registry or ProviderRegistry()

    def run_suggestions(
        self,
        workflow: WorkflowDefinition,
        collected_inputs: dict[str, Any],
        missing_fields: list[str],
        reference_date: date,
    ) -> CapabilityExecutionResult | None:
        capability = self._find_suggestion_capability(workflow, missing_fields)
        if capability is None:
            return None
        provider = self._provider_registry.get_suggestion_provider(capability.provider)
        if provider is None:
            return None
        logger.info(
            "Running suggestion capability workflow=%s capability=%s provider=%s",
            workflow.workflow_id,
            capability.id,
            capability.provider,
        )
        choices = provider.suggest(
            workflow=workflow,
            capability=capability,
            collected_inputs=collected_inputs,
            reference_date=reference_date,
        )
        if not choices:
            logger.info("Suggestion capability returned no choices workflow=%s capability=%s", workflow.workflow_id, capability.id)
            return None
        logger.info(
            "Suggestion capability returned %s choices workflow=%s capability=%s",
            len(choices),
            workflow.workflow_id,
            capability.id,
        )
        return CapabilityExecutionResult(choices=choices)

    def validate(
        self,
        workflow: WorkflowDefinition,
        collected_inputs: dict[str, Any],
        changed_fields: list[str],
        reference_date: date,
    ) -> CapabilityExecutionResult | None:
        capability = self._find_validation_capability(workflow, changed_fields)
        if capability is None:
            return None
        provider = self._provider_registry.get_validation_provider(capability.provider)
        if provider is None:
            return None

        target_fields = set(capability.field_names)
        if not target_fields.issubset(collected_inputs):
            return None

        logger.info(
            "Running validation capability workflow=%s capability=%s provider=%s",
            workflow.workflow_id,
            capability.id,
            capability.provider,
        )
        validation_result, alternatives = provider.validate(
            workflow=workflow,
            capability=capability,
            collected_inputs=collected_inputs,
            reference_date=reference_date,
        )
        logger.info(
            "Validation capability finished workflow=%s capability=%s result=%s alternatives=%s",
            workflow.workflow_id,
            capability.id,
            validation_result.result,
            len(alternatives),
        )
        return CapabilityExecutionResult(
            choices=alternatives,
            validation_result=validation_result,
        )

    def resolve_choice(self, choices: list[ChoiceOption], user_message: str) -> ChoiceOption | None:
        normalized = user_message.lower()
        ordinal_map = {
            0: ["first", "1st", "option 1", "choice 1"],
            1: ["second", "2nd", "option 2", "choice 2"],
            2: ["third", "3rd", "option 3", "choice 3"],
        }
        for index, patterns in ordinal_map.items():
            if index < len(choices) and any(pattern in normalized for pattern in patterns):
                return choices[index]

        for choice in choices:
            if choice.choice_id.lower() in normalized:
                return choice
            start_time = str(choice.value.get("meeting_start_time", "")).lower()
            meeting_date = str(choice.value.get("meeting_date", "")).lower()
            if start_time and start_time in normalized:
                return choice
            if meeting_date and meeting_date in normalized:
                return choice
        return None

    def _find_suggestion_capability(
        self,
        workflow: WorkflowDefinition,
        missing_fields: list[str],
    ) -> WorkflowCapability | None:
        missing = set(missing_fields)
        for capability in workflow.capabilities:
            if capability.type != "suggestion":
                continue
            if not set(capability.field_names).intersection(missing):
                continue
            trigger_when_missing_any = set(capability.config.get("trigger_when_missing_any", []))
            if trigger_when_missing_any and trigger_when_missing_any.intersection(missing):
                return capability
            if {"meeting_date", "meeting_start_time"}.issubset(missing):
                return capability
        return None

    def _find_validation_capability(
        self,
        workflow: WorkflowDefinition,
        changed_fields: list[str],
    ) -> WorkflowCapability | None:
        changed = set(changed_fields)
        for capability in workflow.capabilities:
            if capability.type != "validation":
                continue
            if changed.intersection(capability.field_names):
                return capability
        return None
