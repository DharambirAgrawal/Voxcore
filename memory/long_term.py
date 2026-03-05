"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                     VOXCORE — memory/long_term.py                               ║
║           LONG-TERM MEMORY — CHROMADB VECTOR STORE FOR CONVERSATIONS            ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Provides persistent, semantic long-term memory using ChromaDB as the
    vector store. Conversation summaries, key facts, and user preferences
    are embedded and stored for later retrieval via similarity search.

    Uses ChromaDB's built-in embedding model (all-MiniLM-L6-v2) for
    vectorization. Collections are per-user/per-session.

═══════════════════════════════════════════════════════════════════════════════════
IMPORTS REQUIRED:
═══════════════════════════════════════════════════════════════════════════════════

import logging
import hashlib
import time
from typing import Optional
from pathlib import Path

import chromadb                             # ChromaDB client
from chromadb.config import Settings        # ChromaDB settings

from core.event_bus import EventBus, EventType

═══════════════════════════════════════════════════════════════════════════════════
CONSTANTS:
═══════════════════════════════════════════════════════════════════════════════════

DEFAULT_PERSIST_DIR = "data/chromadb"        # Local persistence directory
DEFAULT_COLLECTION = "voxcore_memory"        # Default collection name
MAX_RESULTS = 10                             # Max results per query

═══════════════════════════════════════════════════════════════════════════════════
CLASSES:
═══════════════════════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────────────────────
CLASS: LongTermMemory
──────────────────────────────────────────────────────────────────────────────────
    ChromaDB-backed persistent semantic memory.

    CONSTRUCTOR: __init__(self, event_bus: EventBus, config: dict = None)
    ─────────────────────────────────────────────────────────────
        INPUTS:
            - event_bus: EventBus — For subscribing to MEMORY_COMPRESS events
            - config: dict — Memory config from config.yaml["memory"], keys:
                - persist_dir: str (default "data/chromadb")
                - collection_name: str (default "voxcore_memory")
                - embedding_model: str (default "all-MiniLM-L6-v2")
        INITIALIZES:
            self._bus = event_bus
            config = config or {}
            self._persist_dir = config.get("persist_dir", DEFAULT_PERSIST_DIR)
            self._collection_name = config.get("collection_name", DEFAULT_COLLECTION)
            self._client: chromadb.ClientAPI = None
            self._collection: chromadb.Collection = None
            self._logger = logging.getLogger("LongTermMemory")

    METHODS:
    ─────────────────────────────────────────────────────────────

    async def initialize(self) -> None
        INPUTS: None
        OUTPUT: None
        WHAT IT DOES:
            1. Create persist directory: Path(self._persist_dir).mkdir(parents=True, exist_ok=True)
            2. self._client = chromadb.PersistentClient(path=self._persist_dir)
            3. self._collection = self._client.get_or_create_collection(
                   name=self._collection_name,
                   metadata={"hnsw:space": "cosine"}  # cosine similarity
               )
            4. Log f"LongTermMemory initialized with {self._collection.count()} documents"
            5. Subscribe to EventType.MEMORY_COMPRESS → self._on_compress

    async def store(self, text: str, metadata: dict = None) -> str
        INPUTS:
            - text: str — Text content to embed and store
            - metadata: dict — Optional metadata (source, timestamp, type, etc.)
        OUTPUT:
            - str — The document ID (hash-based)
        WHAT IT DOES:
            1. doc_id = hashlib.sha256(text.encode()).hexdigest()[:16]
            2. meta = metadata or {}
            3. meta["stored_at"] = time.time()
            4. meta["text_length"] = len(text)
            5. self._collection.upsert(
                   ids=[doc_id],
                   documents=[text],
                   metadatas=[meta]
               )
            6. Log f"Stored document {doc_id} ({len(text)} chars)"
            7. Return doc_id

    async def query(self, query_text: str, top_k: int = 5,
                    where_filter: dict = None) -> list[dict]
        INPUTS:
            - query_text: str — Natural language query for semantic search
            - top_k: int — Number of results (default 5, max MAX_RESULTS)
            - where_filter: dict — Optional ChromaDB where clause for metadata filtering
        OUTPUT:
            - list[dict] — Each dict has: {"content": str, "distance": float,
              "metadata": dict, "id": str}
        WHAT IT DOES:
            1. top_k = min(top_k, MAX_RESULTS)
            2. kwargs = {"query_texts": [query_text], "n_results": top_k}
            3. If where_filter: kwargs["where"] = where_filter
            4. results = self._collection.query(**kwargs)
            5. Process results into list of dicts:
               For each i in range(len(results["ids"][0])):
                   output.append({
                       "id": results["ids"][0][i],
                       "content": results["documents"][0][i],
                       "distance": results["distances"][0][i],
                       "metadata": results["metadatas"][0][i]
                   })
            6. Return output

    async def _on_compress(self, event) -> None
        INPUTS:
            - event: Event — data["turns"] contains list of Turn objects,
              data.get("summary") may contain pre-summarized text from compressor
        OUTPUT: None
        WHAT IT DOES:
            1. If "summary" in event.data:
               await self.store(event.data["summary"],
                   metadata={"source": "compressor", "turn_count": len(event.data.get("turns", []))})
            2. Else:
               For each turn in event.data.get("turns", []):
                   text = f"{turn.role}: {turn.text}"
                   await self.store(text, metadata={"source": "overflow", "role": turn.role})

    async def delete(self, doc_id: str) -> bool
        INPUTS:
            - doc_id: str — Document ID to delete
        OUTPUT:
            - bool — True if deleted, False if not found
        WHAT IT DOES:
            1. Try: self._collection.delete(ids=[doc_id])
            2. Return True
            3. On exception: Log warning, return False

    def count(self) -> int
        INPUTS: None
        OUTPUT: int — Number of documents in the collection
        WHAT IT DOES:
            return self._collection.count()

    async def clear(self) -> None
        INPUTS: None
        OUTPUT: None
        WHAT IT DOES:
            1. self._client.delete_collection(self._collection_name)
            2. self._collection = self._client.get_or_create_collection(
                   name=self._collection_name,
                   metadata={"hnsw:space": "cosine"}
               )
            3. Log "Long-term memory cleared"

═══════════════════════════════════════════════════════════════════════════════════
EXPORTS:
    - LongTermMemory   (class)
═══════════════════════════════════════════════════════════════════════════════════

NOTES:
    - ChromaDB PersistentClient automatically saves to disk
    - Embedding is handled internally by ChromaDB using the default model
      (all-MiniLM-L6-v2) unless overridden in collection settings
    - All operations are synchronous under the hood (ChromaDB is sync),
      but wrapped in async for consistency with the VoxCore pipeline
    - For production, consider running ChromaDB as a server (HttpClient)
"""


"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                     VOXCORE — memory/long_term.py                               ║
║           LONG-TERM MEMORY — CHROMADB VECTOR STORE FOR CONVERSATIONS            ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

import logging
import hashlib
import time
from datetime import datetime
from math import exp
from typing import Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings

from core.event_bus import EventBus, EventType

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_PERSIST_DIR = "data/chromadb"
DEFAULT_COLLECTION = "voxcore_memory"
MAX_RESULTS = 10

# Recency decay constants — lambda per category (higher = faster decay)
RECENCY_LAMBDA = {
    "event": 0.1,
    "fact": 0.05,
    "preference": 0.03,
    "correction": 0.02,
    "instruction": 0.01,
}


# ═══════════════════════════════════════════════════════════════════════════════
# CLASS: LongTermMemory
# ═══════════════════════════════════════════════════════════════════════════════

class LongTermMemory:
    """ChromaDB-backed persistent semantic memory."""

    def __init__(self, event_bus: EventBus, config: dict = None) -> None:
        self._bus = event_bus
        config = config or {}
        self._persist_dir = config.get("persist_dir", DEFAULT_PERSIST_DIR)
        self._collection_name = config.get("collection_name", DEFAULT_COLLECTION)
        self._client: Optional[chromadb.ClientAPI] = None
        self._collection: Optional[chromadb.Collection] = None
        self._logger = logging.getLogger("LongTermMemory")

    async def initialize(self) -> None:
        """Initialize ChromaDB client, collection, and event subscriptions."""
        Path(self._persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self._persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._logger.info(
            f"LongTermMemory initialized with {self._collection.count()} documents"
        )
        self._bus.subscribe(EventType.MEMORY_COMPRESSED, self._on_compress)

    async def store(
        self,
        text: str,
        memory_type: str = "semantic",
        category: str = "fact",
        importance: float = 0.5,
        metadata: Optional[dict] = None,
    ) -> str:
        """Embed and store a text document with type/category tags."""
        doc_id = hashlib.sha256(text.encode()).hexdigest()[:16]
        meta = {
            "timestamp": datetime.now().isoformat(),
            "memory_type": memory_type,
            "category": category,
            "importance": importance,
            "access_count": 0,
            "last_accessed": datetime.now().isoformat(),
            "stored_at": time.time(),
            "text_length": len(text),
            **(metadata or {}),
        }
        self._collection.upsert(
            ids=[doc_id],
            documents=[text],
            metadatas=[meta],
        )
        self._logger.info(
            "Stored document %s (%d chars) type=%s cat=%s",
            doc_id, len(text), memory_type, category,
        )
        return doc_id

    async def query(
        self,
        query_text: str,
        top_k: int = 5,
        where_filter: dict = None,
        memory_type: Optional[str] = None,
    ) -> list[dict]:
        """Semantic similarity search with re-ranking by recency + importance."""
        top_k = min(top_k, MAX_RESULTS)

        # Fetch 3× candidates for re-ranking
        collection_count = self._collection.count()
        if collection_count == 0:
            return []
        n_fetch = min(top_k * 3, collection_count)

        # Build where clause
        where = {}
        if memory_type:
            where["memory_type"] = {"$eq": memory_type}
        if where_filter:
            where.update(where_filter)

        kwargs = {
            "query_texts": [query_text],
            "n_results": n_fetch,
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        # Build raw result list
        raw_results: list[dict] = []
        for i in range(len(results["ids"][0])):
            raw_results.append(
                {
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "distance": results["distances"][0][i],
                    "metadata": results["metadatas"][0][i],
                }
            )

        # Re-rank with recency decay + importance + similarity
        now = datetime.now()
        for result in raw_results:
            meta = result["metadata"]
            # Calculate days since last access
            try:
                last_acc = datetime.fromisoformat(meta.get("last_accessed", meta.get("timestamp", now.isoformat())))
                days = max((now - last_acc).days, 0)
            except (ValueError, TypeError):
                days = 30  # default fallback

            lam = RECENCY_LAMBDA.get(meta.get("category", "fact"), 0.05)
            recency = exp(-lam * days)
            importance = float(meta.get("importance", 0.5))
            similarity = 1 - result["distance"]

            result["final_score"] = (
                0.6 * similarity + 0.25 * importance + 0.15 * recency
            )

        # Sort by final_score DESC, return top_k
        raw_results.sort(key=lambda r: r["final_score"], reverse=True)
        returned = raw_results[:top_k]

        # Update access tracking for returned results
        for r in returned:
            try:
                self._collection.update(
                    ids=[r["id"]],
                    metadatas=[{
                        **r["metadata"],
                        "access_count": int(r["metadata"].get("access_count", 0)) + 1,
                        "last_accessed": now.isoformat(),
                    }],
                )
            except Exception:
                pass  # non-critical

        return returned

    async def _on_compress(self, event) -> None:
        """Handle MEMORY_COMPRESSED events by storing summaries."""
        if "summary" in event.data:
            await self.store(
                event.data["summary"],
                memory_type="episodic",
                category="event",
                importance=0.4,
                metadata={
                    "source": "compressor",
                    "turn_count": len(event.data.get("turns", [])),
                },
            )
        else:
            for turn in event.data.get("turns", []):
                text = f"{turn.role}: {turn.content}"
                await self.store(
                    text,
                    memory_type="episodic",
                    category="event",
                    importance=0.3,
                    metadata={"source": "overflow", "role": turn.role},
                )

    async def delete(self, doc_id: str) -> bool:
        """Delete a document by ID. Returns True on success, False on failure."""
        try:
            self._collection.delete(ids=[doc_id])
            return True
        except Exception as exc:
            self._logger.warning(f"Failed to delete document {doc_id}: {exc}")
            return False

    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self._collection.count()

    async def clear(self) -> None:
        """Drop and recreate the collection, erasing all stored documents."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._logger.info("Long-term memory cleared")

    async def recall(
        self,
        query: str,
        memory_type: Optional[str] = None,
        top_k: int = 5,
    ) -> list[dict]:
        """Convenience alias for query() used by retriever and memory_recall."""
        return await self.query(
            query_text=query,
            top_k=top_k,
            memory_type=memory_type,
        )

    async def save_session(self, session) -> None:
        """Summarize and persist the current session's history to ChromaDB on shutdown.

        This ensures short sessions (< 20 turns) still get committed to long-term memory,
        capturing names, preferences, and key facts even if compression never triggered.
        """
        from core.session import Turn

        turns = list(session.history)
        if not turns:
            self._logger.info("No turns to persist")
            return

        # Build a plain-text summary of the entire session
        lines: list[str] = []
        for turn in turns:
            lines.append(f"{turn.role}: {turn.content}")
        transcript = "\n".join(lines)

        # Store as a single document with session metadata
        await self.store(
            transcript,
            memory_type="episodic",
            category="event",
            importance=0.3,
            metadata={
                "source": "session_save",
                "session_id": session.session_id,
                "turn_count": len(turns),
                "persona": session.persona_name,
            },
        )
        self._logger.info(
            "Persisted session %s (%d turns) to long-term memory",
            session.session_id,
            len(turns),
        )