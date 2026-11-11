"""SQLite-backed conversation bindings and bounded, complete-turn history."""

from contextlib import closing, contextmanager
import json
from pathlib import Path
import sqlite3
from threading import Lock
from uuid import UUID, uuid4

from langchain_core.messages import SystemMessage, messages_from_dict, messages_to_dict

MAX_HISTORY_TURNS = 10
MAX_HISTORY_CHARS = 200_000
_locks = {}
_locks_guard = Lock()


class ConversationConflict(Exception):
    pass


@contextmanager
def conversation_lock(db_path, conversation_id):
    """Serialize a conversation in one API process without retaining idle locks."""
    key = (str(Path(db_path).resolve()), str(conversation_id))
    with _locks_guard:
        lock, users = _locks.get(key, (Lock(), 0))
        _locks[key] = (lock, users + 1)
    try:
        with lock:
            yield
    finally:
        with _locks_guard:
            _, users = _locks[key]
            if users == 1:
                del _locks[key]
            else:
                _locks[key] = (lock, users - 1)


class ConversationNotFound(Exception):
    pass


class ConversationDatasetMismatch(Exception):
    pass


def resolve_conversation(db_path: Path, dataset_id: UUID, conversation_id: UUID | None) -> UUID:
    """Create a server-owned ID or validate an existing ID's dataset binding."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db_path)) as connection:
        with connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS conversations ("
                "conversation_id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL, "
                "created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS conversation_turns ("
                "turn_id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "conversation_id TEXT NOT NULL, messages TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS conversation_turns_lookup "
                "ON conversation_turns (conversation_id, turn_id)"
            )
            if conversation_id is None:
                conversation_id = uuid4()
                connection.execute(
                    "INSERT INTO conversations (conversation_id, dataset_id) VALUES (?, ?)",
                    (str(conversation_id), str(dataset_id)),
                )
                return conversation_id
            row = connection.execute(
                "SELECT dataset_id FROM conversations WHERE conversation_id = ?",
                (str(conversation_id),),
            ).fetchone()
            if row is None:
                raise ConversationNotFound
            if row[0] != str(dataset_id):
                raise ConversationDatasetMismatch
            return conversation_id


def load_history(db_path, conversation_id):
    """Return recent complete turns plus a revision for optimistic concurrency."""
    with closing(sqlite3.connect(db_path)) as connection:
        rows = connection.execute(
            "SELECT turn_id, messages FROM conversation_turns "
            "WHERE conversation_id = ? ORDER BY turn_id DESC LIMIT ?",
            (str(conversation_id), MAX_HISTORY_TURNS),
        ).fetchall()
    revision = rows[0][0] if rows else 0
    selected, size = [], 0
    for _, payload in rows:
        if size + len(payload) > MAX_HISTORY_CHARS:
            break
        selected.append(messages_from_dict(json.loads(payload)))
        size += len(payload)
    return [message for turn in reversed(selected) for message in turn], revision


def save_turn(db_path, conversation_id, revision, messages):
    # Recreate the system prompt each invocation; do not persist transient
    # failure-control instructions as part of the user's conversation.
    messages = [m for m in messages if not isinstance(m, SystemMessage)]
    payload = json.dumps(messages_to_dict(messages), ensure_ascii=False, allow_nan=False)
    with closing(sqlite3.connect(db_path)) as connection:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            latest = connection.execute(
                "SELECT COALESCE(MAX(turn_id), 0) FROM conversation_turns WHERE conversation_id = ?",
                (str(conversation_id),),
            ).fetchone()[0]
            if latest != revision:
                raise ConversationConflict
            connection.execute(
                "INSERT INTO conversation_turns (conversation_id, messages) VALUES (?, ?)",
                (str(conversation_id), payload),
            )
            connection.execute(
                "DELETE FROM conversation_turns WHERE conversation_id = ? AND turn_id NOT IN "
                "(SELECT turn_id FROM conversation_turns WHERE conversation_id = ? "
                "ORDER BY turn_id DESC LIMIT ?)",
                (str(conversation_id), str(conversation_id), MAX_HISTORY_TURNS),
            )
