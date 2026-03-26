from __future__ import annotations

from app.models.domain import WorkflowDefinition
from app.services.providers.base import SuggestionProvider, ValidationProvider
from app.services.providers.demo_calendar import DemoCalendarSuggestionProvider, DemoCalendarValidationProvider


class ProviderRegistry:
    def __init__(
        self,
        suggestion_providers: dict[str, SuggestionProvider] | None = None,
        validation_providers: dict[str, ValidationProvider] | None = None,
    ) -> None:
        demo_suggestion = DemoCalendarSuggestionProvider()
        self._suggestion_providers = suggestion_providers or {
            "demo_calendar.open_slots": demo_suggestion,
        }
        self._validation_providers = validation_providers or {
            "demo_calendar.validate_slot": DemoCalendarValidationProvider(demo_suggestion),
        }

    def register_suggestion_provider(self, provider_name: str, provider: SuggestionProvider) -> None:
        self._suggestion_providers[provider_name] = provider

    def register_validation_provider(self, provider_name: str, provider: ValidationProvider) -> None:
        self._validation_providers[provider_name] = provider

    def get_suggestion_provider(self, provider_name: str) -> SuggestionProvider | None:
        return self._suggestion_providers.get(provider_name)

    def get_validation_provider(self, provider_name: str) -> ValidationProvider | None:
        return self._validation_providers.get(provider_name)

    def validate_workflows(self, workflows: list[WorkflowDefinition]) -> None:
        missing: list[str] = []
        for workflow in workflows:
            for capability in workflow.capabilities:
                if capability.type == "suggestion":
                    provider = self.get_suggestion_provider(capability.provider)
                else:
                    provider = self.get_validation_provider(capability.provider)
                if provider is None:
                    missing.append(f"{workflow.workflow_id}:{capability.id}:{capability.provider}")
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"Unknown capability providers declared in workflow definitions: {joined}")
