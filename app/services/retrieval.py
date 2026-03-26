from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict

from app.core.config import Settings
from app.models.domain import IntermediateRequestRepresentation, RankedWorkflow, WorkflowDefinition, WorkflowSummary
from app.services.intelligence import IntelligenceService
from app.services.workflow_registry import WorkflowRegistry

logger = logging.getLogger(__name__)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    minimum = min(scores.values())
    maximum = max(scores.values())
    if math.isclose(minimum, maximum):
        return {key: 0.0 if math.isclose(maximum, 0.0) else 1.0 for key in scores}
    return {key: (value - minimum) / (maximum - minimum) for key, value in scores.items()}


class BM25Index:
    def __init__(self, workflows: list[WorkflowDefinition]) -> None:
        self._docs = {workflow.workflow_id: self._tokenize(workflow.searchable_text()) for workflow in workflows}
        self._doc_lengths = {workflow_id: len(tokens) for workflow_id, tokens in self._docs.items()}
        self._avg_doc_length = sum(self._doc_lengths.values()) / max(len(self._doc_lengths), 1)
        self._term_frequencies = {workflow_id: Counter(tokens) for workflow_id, tokens in self._docs.items()}
        self._document_frequencies: Counter[str] = Counter()
        for tokens in self._docs.values():
            self._document_frequencies.update(set(tokens))
        self._doc_count = len(self._docs)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9_]+", text.lower())

    def score(self, query: str) -> dict[str, float]:
        query_terms = self._tokenize(query)
        scores: dict[str, float] = {}
        for workflow_id, frequencies in self._term_frequencies.items():
            total = 0.0
            doc_length = self._doc_lengths[workflow_id]
            for term in query_terms:
                doc_frequency = self._document_frequencies.get(term, 0)
                if doc_frequency == 0:
                    continue
                idf = math.log(((self._doc_count - doc_frequency + 0.5) / (doc_frequency + 0.5)) + 1.0)
                tf = frequencies.get(term, 0)
                if tf == 0:
                    continue
                k1 = 1.5
                b = 0.75
                denominator = tf + k1 * (1 - b + b * (doc_length / max(self._avg_doc_length, 1)))
                total += idf * ((tf * (k1 + 1)) / denominator)
            scores[workflow_id] = total
        return scores


class WorkflowMatcher:
    def __init__(
        self,
        registry: WorkflowRegistry,
        intelligence: IntelligenceService,
        settings: Settings,
    ) -> None:
        self._registry = registry
        self._intelligence = intelligence
        self._settings = settings
        self._workflows = registry.list()
        self._bm25 = BM25Index(self._workflows)
        self._workflow_embeddings = dict(
            zip(
                [workflow.workflow_id for workflow in self._workflows],
                self._intelligence.embed_texts([workflow.searchable_text() for workflow in self._workflows]),
                strict=True,
            )
        )

    def match(
        self,
        intent: IntermediateRequestRepresentation,
        top_k: int = 5,
        restrict_to: set[str] | None = None,
    ) -> list[RankedWorkflow]:
        query_text = self._intent_to_query(intent)
        logger.info("Matching workflows for action=%s domain=%s entities=%s", intent.action, intent.domain, intent.entities)
        logger.debug("Matcher query text: %s", query_text)
        query_embedding = self._intelligence.embed_texts([query_text])[0]

        semantic_raw = {
            workflow.workflow_id: _cosine_similarity(query_embedding, self._workflow_embeddings[workflow.workflow_id])
            for workflow in self._workflows
        }
        fuzzy_raw = self._bm25.score(query_text)
        structured_raw = {
            workflow.workflow_id: self._structured_score(intent, workflow)
            for workflow in self._workflows
        }

        if restrict_to is not None:
            semantic_raw = {key: value for key, value in semantic_raw.items() if key in restrict_to}
            fuzzy_raw = {key: value for key, value in fuzzy_raw.items() if key in restrict_to}
            structured_raw = {key: value for key, value in structured_raw.items() if key in restrict_to}

        semantic = _normalize_scores(semantic_raw)
        fuzzy = _normalize_scores(fuzzy_raw)
        structured = _normalize_scores(structured_raw)

        semantic_rank = self._rank(semantic_raw)
        fuzzy_rank = self._rank(fuzzy_raw)
        structured_rank = self._rank(structured_raw)

        candidates: list[RankedWorkflow] = []
        all_ids = list(semantic_raw.keys())
        for workflow_id in all_ids:
            workflow = self._registry.get(workflow_id)
            support_count = sum(
                1
                for raw_scores in (semantic_raw, fuzzy_raw, structured_raw)
                if raw_scores.get(workflow_id, 0.0) > 0.0
            )
            rrf_score = 0.0
            for ranks in (semantic_rank, fuzzy_rank, structured_rank):
                if workflow_id in ranks:
                    rrf_score += 1 / (self._settings.rrf_k + ranks[workflow_id])

            confidence = min(
                1.0,
                (semantic.get(workflow_id, 0.0) * 0.45)
                + (fuzzy.get(workflow_id, 0.0) * 0.20)
                + (structured.get(workflow_id, 0.0) * 0.35)
                + (support_count * 0.03),
            )

            reasons = []
            if intent.action and intent.action in workflow.actions:
                reasons.append(f"Action match: {intent.action}")
            if intent.domain and intent.domain == workflow.domain:
                reasons.append(f"Domain match: {intent.domain}")
            shared_entities = sorted(set(intent.entities).intersection(workflow.entities))
            if shared_entities:
                reasons.append(f"Shared entities: {', '.join(shared_entities)}")
            shared_qualifiers = sorted(set(intent.qualifiers).intersection(workflow.qualifiers))
            if shared_qualifiers:
                reasons.append(f"Shared qualifiers: {', '.join(shared_qualifiers)}")

            candidates.append(
                RankedWorkflow(
                    workflow=WorkflowSummary.from_definition(workflow),
                    rrf_score=rrf_score,
                    confidence=round(confidence, 4),
                    semantic_score=round(semantic.get(workflow_id, 0.0), 4),
                    fuzzy_score=round(fuzzy.get(workflow_id, 0.0), 4),
                    structured_score=round(structured.get(workflow_id, 0.0), 4),
                    support_count=support_count,
                    reasons=reasons,
                )
            )

        candidates.sort(key=lambda item: (item.rrf_score, item.confidence), reverse=True)
        logger.info(
            "Top workflow candidates: %s",
            [
                {
                    "workflow_id": candidate.workflow.workflow_id,
                    "confidence": candidate.confidence,
                    "rrf_score": round(candidate.rrf_score, 4),
                }
                for candidate in candidates[: min(top_k, 3)]
            ],
        )
        return candidates[:top_k]

    def should_auto_select(self, candidates: list[RankedWorkflow]) -> bool:
        if not candidates:
            return False
        if len(candidates) == 1:
            return candidates[0].confidence >= 0.45
        top = candidates[0]
        second = candidates[1]
        return (
            top.confidence >= self._settings.auto_select_confidence
            and (top.confidence - second.confidence) >= self._settings.auto_select_margin
            and top.support_count >= 2
        )

    def _intent_to_query(self, intent: IntermediateRequestRepresentation) -> str:
        parts = [
            intent.raw_text,
            intent.action or "",
            intent.domain or "",
            intent.subdomain or "",
            " ".join(intent.entities),
            " ".join(intent.qualifiers),
            intent.context or "",
        ]
        return " ".join(part for part in parts if part).strip()

    def _rank(self, scores: dict[str, float]) -> dict[str, int]:
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return {workflow_id: index + 1 for index, (workflow_id, _) in enumerate(ordered)}

    def _structured_score(
        self,
        intent: IntermediateRequestRepresentation,
        workflow: WorkflowDefinition,
    ) -> float:
        score = 0.0
        if intent.action and intent.action in workflow.actions:
            score += 3.0
        if intent.domain and intent.domain == workflow.domain:
            score += 2.0
        if intent.subdomain and workflow.subdomain and intent.subdomain == workflow.subdomain:
            score += 1.5
        if intent.entities:
            shared_entities = len(set(intent.entities).intersection(workflow.entities))
            score += 4.0 * (shared_entities / len(set(intent.entities)))
        if intent.qualifiers:
            shared_qualifiers = len(set(intent.qualifiers).intersection(workflow.qualifiers))
            score += 2.0 * (shared_qualifiers / len(set(intent.qualifiers)))
        return score
