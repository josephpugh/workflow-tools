from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Any

from app.models.domain import ChoiceOption, ValidationResult, WorkflowCapability, WorkflowDefinition


class SuggestionProvider(ABC):
    @abstractmethod
    def suggest(
        self,
        workflow: WorkflowDefinition,
        capability: WorkflowCapability,
        collected_inputs: dict[str, Any],
        reference_date: date,
        exclude: set[tuple[str, str]] | None = None,
    ) -> list[ChoiceOption]:
        raise NotImplementedError


class ValidationProvider(ABC):
    @abstractmethod
    def validate(
        self,
        workflow: WorkflowDefinition,
        capability: WorkflowCapability,
        collected_inputs: dict[str, Any],
        reference_date: date,
    ) -> tuple[ValidationResult, list[ChoiceOption]]:
        raise NotImplementedError

