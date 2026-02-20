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

    async def store(self, text: str, metadata: dict = None) -> str:
        """Embed and store a text document, returning its hash-based ID."""
        doc_id = hashlib.sha256(text.encode()).hexdigest()[:16]
        meta = metadata or {}
        meta["stored_at"] = time.time()
        meta["text_length"] = len(text)
        self._collection.upsert(
            ids=[doc_id],
            documents=[text],
            metadatas=[meta],
        )
        self._logger.info(f"Stored document {doc_id} ({len(text)} chars)")
        return doc_id

    async def query(
        self,
        query_text: str,
        top_k: int = 5,
        where_filter: dict = None,
    ) -> list[dict]:
        """Semantic similarity search against stored documents."""
        top_k = min(top_k, MAX_RESULTS)
        kwargs = {"query_texts": [query_text], "n_results": top_k}
        if where_filter:
            kwargs["where"] = where_filter
        results = self._collection.query(**kwargs)

        output: list[dict] = []
        for i in range(len(results["ids"][0])):
            output.append(
                {
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "distance": results["distances"][0][i],
                    "metadata": results["metadatas"][0][i],
                }
            )
        return output

    async def _on_compress(self, event) -> None:
        """Handle MEMORY_COMPRESS events by storing summaries or raw turns."""
        if "summary" in event.data:
            await self.store(
                event.data["summary"],
                metadata={
                    "source": "compressor",
                    "turn_count": len(event.data.get("turns", [])),
                },
            )
        else:
            for turn in event.data.get("turns", []):
                text = f"{turn.role}: {turn.text}"
                await self.store(
                    text, metadata={"source": "overflow", "role": turn.role}
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