"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                     VOXCORE — interfaces/api.py                                 ║
║              REST API — HTTP ENDPOINTS FOR CONFIG, STATUS & CONTROL              ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    FastAPI REST API server providing HTTP endpoints for:
    - System health and status monitoring
    - Session info and conversation history
    - Configuration reading/updating at runtime
    - Memory queries (search long-term memory)
    - Safety stats
    - Voice profile management
    - Manual text injection (for testing)

    Designed to share the same FastAPI app instance as the WebSocket server,
    or run standalone for API-only use cases.

═══════════════════════════════════════════════════════════════════════════════════
IMPORTS REQUIRED:
═══════════════════════════════════════════════════════════════════════════════════

import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from core.event_bus import EventBus, EventType
from core.session import Session, TurnState
from memory.long_term import LongTermMemory
from memory.short_term import ShortTermMemory
from safety.guard import SafetyGuard
from output.voice_profile import VoiceProfile

═══════════════════════════════════════════════════════════════════════════════════
REQUEST/RESPONSE MODELS (Pydantic):
═══════════════════════════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    status: str                      # "ok" | "degraded" | "error"
    uptime_seconds: float
    state: str                       # Current TurnState value
    connected_clients: int

class SessionInfoResponse(BaseModel):
    state: str
    turn_count: int
    short_term_size: int
    long_term_count: int
    persona_name: str

class ConversationHistoryResponse(BaseModel):
    turns: list[dict]                # [{role, text, timestamp}, ...]
    total: int

class TextInjectRequest(BaseModel):
    text: str                        # Text to inject as user input
    source: str = "api"              # Source label

class TextInjectResponse(BaseModel):
    success: bool
    message: str

class MemoryQueryRequest(BaseModel):
    query: str                       # Search query
    top_k: int = 5                   # Max results

class MemoryQueryResponse(BaseModel):
    results: list[dict]              # [{content, distance, metadata}, ...]
    count: int

class VoiceProfileUpdateRequest(BaseModel):
    voice: str = None                # New voice name
    speed: float = None              # New speed multiplier
    default_emotion: str = None      # New default emotion

class SafetyStatsResponse(BaseModel):
    checked: int
    flagged: int
    errors: int
    skipped: int
    enabled: bool

═══════════════════════════════════════════════════════════════════════════════════
CLASSES:
═══════════════════════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────────────────────
CLASS: APIServer
──────────────────────────────────────────────────────────────────────────────────
    REST API server for system monitoring, config, and control.

    CONSTRUCTOR: __init__(self, event_bus: EventBus, session: Session,
                          short_term: ShortTermMemory = None,
                          long_term: LongTermMemory = None,
                          safety: SafetyGuard = None,
                          voice_profile: VoiceProfile = None,
                          app: FastAPI = None)
    ─────────────────────────────────────────────────────────────
        INPUTS:
            - event_bus: EventBus
            - session: Session
            - short_term: ShortTermMemory (optional)
            - long_term: LongTermMemory (optional)
            - safety: SafetyGuard (optional)
            - voice_profile: VoiceProfile (optional)
            - app: FastAPI — Shared app or creates new one
        INITIALIZES:
            self._bus = event_bus
            self._session = session
            self._short_term = short_term
            self._long_term = long_term
            self._safety = safety
            self._voice_profile = voice_profile
            self._app = app or FastAPI(title="VoxCore API", version="0.1.0")
            self._start_time = datetime.utcnow()
            self._logger = logging.getLogger("APIServer")

    METHODS:
    ─────────────────────────────────────────────────────────────

    def setup_routes(self) -> None
        INPUTS: None
        OUTPUT: None
        WHAT IT DOES:
            1. Add CORS middleware (allow all origins for development):
               self._app.add_middleware(CORSMiddleware,
                   allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])
            2. Register all route handlers (see ROUTES below)

    async def start(self, host: str = "0.0.0.0", port: int = 8080) -> None
        INPUTS:
            - host: str
            - port: int
        OUTPUT: None (runs uvicorn server)
        WHAT IT DOES:
            1. self.setup_routes()
            2. config = uvicorn.Config(self._app, host=host, port=port)
            3. server = uvicorn.Server(config)
            4. await server.serve()

    ── ROUTE HANDLERS ──────────────────────────────────────────

    async def get_health(self) -> HealthResponse                 # GET /health
        WHAT IT DOES:
            Return system health: uptime, state, connected clients count

    async def get_session(self) -> SessionInfoResponse           # GET /session
        WHAT IT DOES:
            Return current session state, turn count, memory sizes, persona

    async def get_history(self, limit: int = 20) -> ConversationHistoryResponse
        #  GET /history?limit=20
        WHAT IT DOES:
            Return last N turns from session history

    async def inject_text(self, request: TextInjectRequest) -> TextInjectResponse
        # POST /inject
        WHAT IT DOES:
            1. text = request.text.strip()
            2. If not text: raise HTTPException(400, "Empty text")
            3. Publish EventType.TRANSCRIPT with data={
                   "text": text, "is_final": True, "source": request.source
               }
            4. Return success

    async def query_memory(self, request: MemoryQueryRequest) -> MemoryQueryResponse
        # POST /memory/query
        WHAT IT DOES:
            1. If not self._long_term: raise HTTPException(503, "Memory not available")
            2. results = await self._long_term.query(request.query, request.top_k)
            3. Return MemoryQueryResponse(results=results, count=len(results))

    async def get_safety_stats(self) -> SafetyStatsResponse      # GET /safety/stats
        WHAT IT DOES:
            1. If not self._safety: raise HTTPException(503, "Safety not available")
            2. stats = self._safety.get_stats()
            3. Return SafetyStatsResponse(**stats, enabled=self._safety.is_enabled)

    async def update_voice(self, request: VoiceProfileUpdateRequest) -> dict
        # PATCH /voice
        WHAT IT DOES:
            1. If not self._voice_profile: raise HTTPException(503)
            2. updates = {k:v for k,v in request.dict().items() if v is not None}
            3. self._voice_profile.update(**updates)
            4. Return self._voice_profile.to_dict()

    async def get_voice(self) -> dict                            # GET /voice
        WHAT IT DOES:
            Return self._voice_profile.to_dict()

    async def get_events_stats(self) -> dict                     # GET /events/stats
        WHAT IT DOES:
            Return subscriber counts per event type from EventBus

═══════════════════════════════════════════════════════════════════════════════════
ROUTES SUMMARY:
═══════════════════════════════════════════════════════════════════════════════════

    GET    /health           → HealthResponse
    GET    /session          → SessionInfoResponse
    GET    /history          → ConversationHistoryResponse
    POST   /inject           → TextInjectResponse
    POST   /memory/query     → MemoryQueryResponse
    GET    /safety/stats     → SafetyStatsResponse
    GET    /voice            → dict (voice profile)
    PATCH  /voice            → dict (updated voice profile)
    GET    /events/stats     → dict (event subscriber counts)

═══════════════════════════════════════════════════════════════════════════════════
EXPORTS:
    - APIServer                  (class)
    - HealthResponse             (Pydantic model)
    - SessionInfoResponse        (Pydantic model)
    - ConversationHistoryResponse (Pydantic model)
    - TextInjectRequest          (Pydantic model)
    - TextInjectResponse         (Pydantic model)
    - MemoryQueryRequest         (Pydantic model)
    - MemoryQueryResponse        (Pydantic model)
    - VoiceProfileUpdateRequest  (Pydantic model)
    - SafetyStatsResponse        (Pydantic model)
═══════════════════════════════════════════════════════════════════════════════════

NOTES:
    - Shares the FastAPI app with WebSocketServer when both run together
    - CORS is wide open for development — restrict in production
    - All endpoints are async
    - POST /inject enables testing without audio hardware
    - Memory query uses ChromaDB semantic search under the hood
"""


"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                     VOXCORE — interfaces/api.py                                 ║
║              REST API — HTTP ENDPOINTS FOR CONFIG, STATUS & CONTROL              ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from core.event_bus import EventBus, EventType
from core.session import Session, TurnState
from memory.long_term import LongTermMemory
from memory.short_term import ShortTermMemory
from safety.guard import SafetyGuard
from output.voice_profile import VoiceProfile

# ═════════════════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE MODELS
# ═════════════════════════════════════════════════════════════════════════════════


class HealthResponse(BaseModel):
    status: str  # "ok" | "degraded" | "error"
    uptime_seconds: float
    state: str  # Current TurnState value
    connected_clients: int


class SessionInfoResponse(BaseModel):
    state: str
    turn_count: int
    short_term_size: int
    long_term_count: int
    persona_name: str


class ConversationHistoryResponse(BaseModel):
    turns: list[dict]  # [{role, text, timestamp}, ...]
    total: int


class TextInjectRequest(BaseModel):
    text: str  # Text to inject as user input
    source: str = "api"  # Source label


class TextInjectResponse(BaseModel):
    success: bool
    message: str


class MemoryQueryRequest(BaseModel):
    query: str  # Search query
    top_k: int = 5  # Max results


class MemoryQueryResponse(BaseModel):
    results: list[dict]  # [{content, distance, metadata}, ...]
    count: int


class VoiceProfileUpdateRequest(BaseModel):
    voice: Optional[str] = None  # New voice name
    speed: Optional[float] = None  # New speed multiplier
    default_emotion: Optional[str] = None  # New default emotion


class SafetyStatsResponse(BaseModel):
    checked: int
    flagged: int
    errors: int
    skipped: int
    enabled: bool


# ═════════════════════════════════════════════════════════════════════════════════
# API SERVER
# ═════════════════════════════════════════════════════════════════════════════════


class APIServer:
    """REST API server for system monitoring, config, and control."""

    def __init__(
        self,
        event_bus: EventBus,
        session: Session,
        short_term: Optional[ShortTermMemory] = None,
        long_term: Optional[LongTermMemory] = None,
        safety: Optional[SafetyGuard] = None,
        voice_profile: Optional[VoiceProfile] = None,
        app: Optional[FastAPI] = None,
    ) -> None:
        self._bus = event_bus
        self._session = session
        self._short_term = short_term
        self._long_term = long_term
        self._safety = safety
        self._voice_profile = voice_profile
        self._app = app or FastAPI(title="VoxCore API", version="0.1.0")
        self._start_time = datetime.utcnow()
        self._logger = logging.getLogger("APIServer")

    # ── property for external access to the FastAPI app ──────────────────────

    @property
    def app(self) -> FastAPI:
        return self._app

    # ── route setup ──────────────────────────────────────────────────────────

    def setup_routes(self) -> None:
        # CORS — wide open for development
        self._app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Register routes
        self._app.get("/health", response_model=HealthResponse)(self.get_health)
        self._app.get("/session", response_model=SessionInfoResponse)(self.get_session)
        self._app.get("/history", response_model=ConversationHistoryResponse)(self.get_history)
        self._app.post("/inject", response_model=TextInjectResponse)(self.inject_text)
        self._app.post("/memory/query", response_model=MemoryQueryResponse)(self.query_memory)
        self._app.get("/safety/stats", response_model=SafetyStatsResponse)(self.get_safety_stats)
        self._app.get("/voice")(self.get_voice)
        self._app.patch("/voice")(self.update_voice)
        self._app.get("/events/stats")(self.get_events_stats)

    # ── server start ─────────────────────────────────────────────────────────

    async def start(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        self.setup_routes()
        config = uvicorn.Config(self._app, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()

    # ── route handlers ───────────────────────────────────────────────────────

    async def get_health(self) -> HealthResponse:
        uptime = (datetime.utcnow() - self._start_time).total_seconds()
        state = (
            self._session.state.value
            if hasattr(self._session, "state") and hasattr(self._session.state, "value")
            else str(getattr(self._session, "state", "unknown"))
        )
        connected = getattr(self._session, "connected_clients", 0)

        try:
            status = "ok"
        except Exception:
            status = "error"

        return HealthResponse(
            status=status,
            uptime_seconds=uptime,
            state=state,
            connected_clients=connected,
        )

    async def get_session(self) -> SessionInfoResponse:
        state = (
            self._session.state.value
            if hasattr(self._session, "state") and hasattr(self._session.state, "value")
            else str(getattr(self._session, "state", "unknown"))
        )
        turn_count = len(getattr(self._session, "history", []))
        stm_size = (
            len(self._short_term) if self._short_term is not None else 0
        )
        ltm_count = (
            await self._long_term.count()
            if self._long_term is not None and hasattr(self._long_term, "count")
            else 0
        )
        persona = getattr(self._session, "persona_name", "default")

        return SessionInfoResponse(
            state=state,
            turn_count=turn_count,
            short_term_size=stm_size,
            long_term_count=ltm_count,
            persona_name=persona,
        )

    async def get_history(self, limit: int = 20) -> ConversationHistoryResponse:
        history = getattr(self._session, "history", [])
        turns = history[-limit:] if limit else history
        return ConversationHistoryResponse(turns=turns, total=len(history))

    async def inject_text(self, request: TextInjectRequest) -> TextInjectResponse:
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Empty text")
        await self._bus.publish(
            EventType.TRANSCRIPT,
            {"text": text, "is_final": True, "source": request.source},
        )
        return TextInjectResponse(success=True, message=f"Injected {len(text)} chars")

    async def query_memory(self, request: MemoryQueryRequest) -> MemoryQueryResponse:
        if not self._long_term:
            raise HTTPException(status_code=503, detail="Memory not available")
        results = await self._long_term.query(request.query, request.top_k)
        return MemoryQueryResponse(results=results, count=len(results))

    async def get_safety_stats(self) -> SafetyStatsResponse:
        if not self._safety:
            raise HTTPException(status_code=503, detail="Safety not available")
        stats = self._safety.get_stats()
        return SafetyStatsResponse(**stats, enabled=self._safety.is_enabled)

    async def update_voice(self, request: VoiceProfileUpdateRequest) -> dict:
        if not self._voice_profile:
            raise HTTPException(status_code=503, detail="Voice profile not available")
        updates = {k: v for k, v in request.dict().items() if v is not None}
        if updates:
            self._voice_profile.update(**updates)
        return self._voice_profile.to_dict()

    async def get_voice(self) -> dict:
        if not self._voice_profile:
            raise HTTPException(status_code=503, detail="Voice profile not available")
        return self._voice_profile.to_dict()

    async def get_events_stats(self) -> dict:
        stats: dict = {}
        if hasattr(self._bus, "_subscribers"):
            for event_type, handlers in self._bus._subscribers.items():
                key = event_type.value if hasattr(event_type, "value") else str(event_type)
                stats[key] = len(handlers)
        return {"subscriber_counts": stats}