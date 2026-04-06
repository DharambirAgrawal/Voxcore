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
from core.session import Session
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
    language: Optional[str] = None  # New language
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
        try:
            ltm_count = (
                self._long_term.count()
                if self._long_term is not None and hasattr(self._long_term, "count")
                else 0
            )
        except Exception:
            ltm_count = 0
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
            EventType.TRANSCRIPT_READY,
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