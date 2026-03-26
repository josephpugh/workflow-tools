from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from app.models.domain import ConversationState

logger = logging.getLogger(__name__)


class ConversationRepository:
    def __init__(self, database_url: str) -> None:
        self._database_path = self._normalize_database_url(database_url)
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @staticmethod
    def _normalize_database_url(database_url: str) -> Path:
        if database_url.startswith("sqlite:///"):
            return Path(database_url.removeprefix("sqlite:///")).resolve()
        return Path(database_url).resolve()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    session_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL
                )
                """
            )
        logger.debug("Conversation repository initialized at %s", self._database_path)

    def load(self, session_id: str) -> ConversationState | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM conversations WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            logger.debug("Conversation state not found for session_id=%s", session_id)
            return None
        logger.debug("Conversation state loaded for session_id=%s", session_id)
        return ConversationState.model_validate_json(row["payload"])

    def save(self, state: ConversationState) -> None:
        payload = json.dumps(state.model_dump(mode="json"))
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO conversations(session_id, payload)
                VALUES(?, ?)
                ON CONFLICT(session_id) DO UPDATE SET payload = excluded.payload
                """,
                (state.session_id, payload),
            )
        logger.debug("Conversation state saved for session_id=%s status=%s", state.session_id, state.status)
