"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                    VOXCORE v4 — websocket/audio_bridge.py                       ║
║        PER-USER AUDIO QUEUES — REPLACES LOCAL MIC + SPEAKER FOR WEBSOCKET      ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Two asyncio queues per user session.
    Inbound:  WebSocket binary frames → mic_stream pipeline
    Outbound: audio_player TTS chunks → WebSocket binary frames

    No processing, no conversion. Pure plumbing.
"""

import asyncio
import logging
import time

logger = logging.getLogger("AudioBridge")


class AudioBridge:
    """Two asyncio queues per user session replacing local mic + speaker."""

    def __init__(self, user_id: str = "unknown"):
        self.user_id = user_id
        self.mic_queue: asyncio.Queue = asyncio.Queue()
        # maxsize=60 ≈ 1.5s of 24kHz audio at 512 samples/chunk
        # If full: drop oldest (stale audio is worse than a gap)
        self.speaker_queue: asyncio.Queue = asyncio.Queue(maxsize=60)
        self._last_log_time: float = 0.0

    def push_mic_audio(self, pcm_bytes: bytes) -> None:
        """Called by server.py when binary frame arrives from client.

        Args:
            pcm_bytes: Raw 16kHz 16-bit signed mono PCM bytes.
        """
        self.mic_queue.put_nowait(pcm_bytes)

    async def get_tts_chunk(self) -> bytes:
        """Called by server.py to get next TTS chunk to send to client.

        Returns:
            Raw PCM bytes to stream to client speaker.
        """
        return await self.speaker_queue.get()

    def push_tts_chunk(self, pcm_bytes: bytes) -> None:
        """Called indirectly via audio_player's sink queue.

        If the speaker queue is full, drops the oldest chunk to keep
        audio flowing — stale audio is worse than a brief gap.
        """
        try:
            self.speaker_queue.put_nowait(pcm_bytes)
        except asyncio.QueueFull:
            # Drop oldest, keep queue moving
            try:
                self.speaker_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self.speaker_queue.put_nowait(pcm_bytes)
            except asyncio.QueueFull:
                pass

    def log_depths(self) -> None:
        """Log queue depths every 5 seconds during active session."""
        now = time.time()
        if now - self._last_log_time < 5.0:
            return
        self._last_log_time = now
        mic_depth = self.mic_queue.qsize()
        speaker_depth = self.speaker_queue.qsize()
        logger.info(
            "[%s] mic_queue_depth=%d  speaker_queue_depth=%d",
            self.user_id, mic_depth, speaker_depth,
        )
        if mic_depth > 20:
            logger.warning(
                "[%s] Pipeline stalled — mic audio backing up (%d)",
                self.user_id, mic_depth,
            )
        if speaker_depth > 40:
            logger.warning(
                "[%s] Client not consuming TTS — network slow? (%d)",
                self.user_id, speaker_depth,
            )
