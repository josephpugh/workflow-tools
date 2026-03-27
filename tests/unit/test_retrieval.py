from pathlib import Path

from app.core.config import Settings
from app.models.domain import IntermediateRequestRepresentation
from app.services.intelligence import HashingIntelligenceService
from app.services.retrieval import WorkflowMatcher
from app.services.workflow_registry import WorkflowRegistry


def build_matcher() -> WorkflowMatcher:
    registry = WorkflowRegistry(Path("/Users/joepugh/workspace/workflow-tools/workflows"))
    intelligence = HashingIntelligenceService()
    settings = Settings(workflows_dir=Path("/Users/joepugh/workspace/workflow-tools/workflows"))
    return WorkflowMatcher(registry=registry, intelligence=intelligence, settings=settings)


def test_rrf_keeps_ambiguous_address_updates_for_disambiguation() -> None:
    matcher = build_matcher()
    intent = IntermediateRequestRepresentation(
        action="update",
        entities=["client", "address"],
        domain="client_servicing",
        raw_text="Update David's address",
    )
    candidates = matcher.match(intent)
    top_ids = [candidate.workflow.workflow_id for candidate in candidates[:2]]
    assert set(top_ids) == {"update_client_mailing_address", "update_client_billing_address"}
    assert matcher.should_auto_select(candidates) is False


def test_rrf_auto_selects_explicit_vendor_payment() -> None:
    matcher = build_matcher()
    intent = IntermediateRequestRepresentation(
        action="schedule",
        entities=["payment", "invoice", "vendor"],
        domain="operations",
        qualifiers=["payable"],
        raw_text="Schedule a vendor payment for invoice INV-1001",
    )
    candidates = matcher.match(intent)
    assert candidates[0].workflow.workflow_id == "schedule_vendor_payment"
    assert matcher.should_auto_select(candidates) is True


def test_match_filters_out_ungrounded_candidates_for_unsupported_requests() -> None:
    matcher = build_matcher()
    intent = IntermediateRequestRepresentation(
        action="create",
        entities=["portfolio", "client"],
        domain="operations",
        qualifiers=["investment"],
        context="Client name from context: David Smith. Request implies initiating an investment trade for $500 in .SPX for the client.",
        raw_text="Place a trade for 500 dollars for .SPX for this guy",
    )
    candidates = matcher.match(intent)
    assert candidates == []
