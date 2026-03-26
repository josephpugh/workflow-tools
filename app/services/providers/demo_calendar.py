from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from app.models.domain import ChoiceOption, ValidationResult, WorkflowCapability, WorkflowDefinition
from app.services.providers.base import SuggestionProvider, ValidationProvider


class DemoCalendarSuggestionProvider(SuggestionProvider):
    def suggest(
        self,
        workflow: WorkflowDefinition,
        capability: WorkflowCapability,
        collected_inputs: dict[str, Any],
        reference_date: date,
        exclude: set[tuple[str, str]] | None = None,
    ) -> list[ChoiceOption]:
        exclude = exclude or set()
        duration = int(collected_inputs.get("duration_minutes", capability.config.get("default_duration_minutes", 30)))
        if "meeting_date" in collected_inputs:
            try:
                base_dates = [date.fromisoformat(str(collected_inputs["meeting_date"]))]
            except ValueError:
                base_dates = self._suggestion_dates(reference_date)
        else:
            base_dates = self._suggestion_dates(reference_date)
        start_times = capability.config.get("start_times", ["10:00", "11:30", "15:00"])
        blocked = {
            (item["date"], item["start_time"])
            for item in capability.config.get("unavailable_slots", [])
        }
        blocked |= exclude

        choices: list[ChoiceOption] = []
        for meeting_day in base_dates:
            for start in start_times:
                key = (meeting_day.isoformat(), start)
                if key in blocked:
                    continue
                label = f"{meeting_day.strftime('%A, %B %d')} at {start}"
                choices.append(
                    ChoiceOption(
                        choice_id=f"{capability.id}_{len(choices) + 1}",
                        label=label,
                        value={
                            "meeting_date": meeting_day.isoformat(),
                            "meeting_start_time": start,
                            "duration_minutes": duration,
                        },
                        source=capability.id,
                    )
                )
                if len(choices) >= int(capability.config.get("max_options", 3)):
                    return choices
        return choices

    def _suggestion_dates(self, reference_date: date) -> list[date]:
        current = reference_date + timedelta(days=1)
        results: list[date] = []
        while len(results) < 5:
            if current.weekday() < 5:
                results.append(current)
            current += timedelta(days=1)
        return results


class DemoCalendarValidationProvider(ValidationProvider):
    def __init__(self, suggestion_provider: DemoCalendarSuggestionProvider) -> None:
        self._suggestion_provider = suggestion_provider

    def validate(
        self,
        workflow: WorkflowDefinition,
        capability: WorkflowCapability,
        collected_inputs: dict[str, Any],
        reference_date: date,
    ) -> tuple[ValidationResult, list[ChoiceOption]]:
        meeting_date = str(collected_inputs["meeting_date"])
        meeting_start_time = str(collected_inputs["meeting_start_time"])
        blocked = {
            (item["date"], item["start_time"])
            for item in capability.config.get("unavailable_slots", [])
        }
        if (meeting_date, meeting_start_time) not in blocked:
            return (
                ValidationResult(
                    provider=capability.provider,
                    result="passed",
                    field_names=capability.field_names,
                ),
                [],
            )

        suggestion_capability = self._find_capability(workflow, capability.config.get("suggestion_capability_id"))
        alternatives = (
            self._suggestion_provider.suggest(
                workflow=workflow,
                capability=suggestion_capability,
                collected_inputs=collected_inputs,
                reference_date=reference_date,
                exclude={(meeting_date, meeting_start_time)},
            )
            if suggestion_capability is not None
            else []
        )
        return (
            ValidationResult(
                provider=capability.provider,
                result="failed",
                reason="The requested meeting time is already booked.",
                field_names=capability.field_names,
            ),
            alternatives,
        )

    def _find_capability(self, workflow: WorkflowDefinition, capability_id: str | None) -> WorkflowCapability | None:
        if capability_id is None:
            return None
        for capability in workflow.capabilities:
            if capability.id == capability_id:
                return capability
        return None

