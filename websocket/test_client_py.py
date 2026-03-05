#!/usr/bin/env python3
"""
VoxCore v4 — Python WebSocket Test Client
==========================================
Captures mic via sounddevice (16kHz mono), sends raw PCM over WebSocket,
receives 24kHz TTS PCM back and plays it through speakers.

Usage:
    python websocket/test_client_py.py                          # defaults
    python websocket/test_client_py.py --url ws://192.168.1.5:8765
    python websocket/test_client_py.py --token my-secret-key --user dharambir

Works exactly like main.py from the user's perspective (talk → hear reply)
but the pipeline runs on the WebSocket server, not locally.

Requirements (already in requirements.txt):
    pip install websockets sounddevice numpy
"""

import argparse
import asyncio
import json
import struct
import sys
import signal
import time
import threading
from collections import deque

import numpy as np
import sounddevice as sd

# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════
MIC_RATE       = 16000      # Hz  — what the server expects
MIC_CHANNELS   = 1
MIC_BLOCKSIZE  = 512        # samples per frame (~32 ms)
MIC_DTYPE      = "int16"

SPEAKER_RATE   = 24000      # Hz  — what the server sends (Kokoro TTS)
SPEAKER_CHANNELS = 1

# ═══════════════════════════════════════════════════════════════════════
# COLOUR HELPERS (ANSI)
# ═══════════════════════════════════════════════════════════════════════
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RED    = "\033[91m"
CYAN   = "\033[96m"
MAGENTA = "\033[95m"

STATE_COLOURS = {
    "listening":   GREEN,
    "thinking":    YELLOW,
    "speaking":    BLUE,
    "interrupted": RED,
    "pending":     MAGENTA,
    "paused":      DIM,
}


def ts():
    return time.strftime("%H:%M:%S")


def log(msg, colour=CYAN):
    print(f"{DIM}[{ts()}]{RESET} {colour}{msg}{RESET}")


def log_state(state):
    c = STATE_COLOURS.get(state.lower(), RESET)
    bar = "█" * 3
    print(f"\n{DIM}[{ts()}]{RESET}  {c}{bar}  {BOLD}{state.upper()}{RESET}{c}  {bar}{RESET}\n")


# ═══════════════════════════════════════════════════════════════════════
# AUDIO PLAYBACK
# ═══════════════════════════════════════════════════════════════════════
class SpeakerPlayer:
    """Queues incoming PCM chunks and plays them through the default output device."""

    def __init__(self):
        self._queue: deque[np.ndarray] = deque()
        self._lock = threading.Lock()
        self._stream: sd.OutputStream | None = None
        self._paused = False
        self._running = False

    def start(self):
        self._running = True
        self._stream = sd.OutputStream(
            samplerate=SPEAKER_RATE,
            channels=SPEAKER_CHANNELS,
            dtype="float32",
            blocksize=1024,
            callback=self._callback,
        )
        self._stream.start()
        log("Speaker stream opened (24 kHz)", DIM)

    def _callback(self, outdata, frames, time_info, status):
        if self._paused:
            outdata[:] = 0
            return

        written = 0
        buf = outdata[:, 0] if outdata.ndim == 2 else outdata.reshape(-1)

        with self._lock:
            while written < frames and self._queue:
                chunk = self._queue[0]
                avail = len(chunk)
                need = frames - written

                if avail <= need:
                    buf[written : written + avail] = chunk
                    written += avail
                    self._queue.popleft()
                else:
                    buf[written : written + need] = chunk[:need]
                    self._queue[0] = chunk[need:]
                    written += need

        if written < frames:
            buf[written:] = 0

    def enqueue(self, pcm_bytes: bytes):
        """Push raw int16 PCM bytes into the play queue."""
        int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        float32 = int16.astype(np.float32) / 32768.0
        with self._lock:
            self._queue.append(float32)

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def flush(self):
        with self._lock:
            self._queue.clear()
        self._paused = False

    def stop(self):
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None


# ═══════════════════════════════════════════════════════════════════════
# WEBSOCKET CLIENT
# ═══════════════════════════════════════════════════════════════════════
class VoxClient:
    def __init__(self, url: str):
        self.url = url
        self.ws = None
        self.speaker = SpeakerPlayer()
        self.mic_muted = False
        self._mic_task: asyncio.Task | None = None
        self._recv_task: asyncio.Task | None = None
        self._ping_task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self._mic_queue: asyncio.Queue = asyncio.Queue()
        self._state = "disconnected"

    # ─── lifecycle ────────────────────────────────────────────────────
    async def run(self):
        try:
            import websockets
        except ImportError:
            print(f"{RED}ERROR: 'websockets' package not installed.  pip install websockets{RESET}")
            return

        log(f"Connecting to {self.url} ...")
        try:
            self.ws = await websockets.connect(
                self.url,
                max_size=2 ** 20,       # 1 MB max message
                ping_interval=None,     # we send our own keepalive
            )
        except Exception as e:
            log(f"Connection failed: {e}", RED)
            return

        log("Connected!", GREEN)
        self.speaker.start()

        self._recv_task = asyncio.create_task(self._recv_loop())
        self._ping_task = asyncio.create_task(self._ping_loop())

        # Wait until told to stop
        await self._stop_event.wait()

        # Tear down
        await self._shutdown()

    async def _shutdown(self):
        log("Shutting down...", YELLOW)

        if self._mic_task and not self._mic_task.done():
            self._mic_task.cancel()
        if self._recv_task and not self._recv_task.done():
            self._recv_task.cancel()
        if self._ping_task and not self._ping_task.done():
            self._ping_task.cancel()

        self.speaker.stop()

        if self.ws:
            await self.ws.close()

        log("Bye!", GREEN)

    def request_stop(self):
        self._stop_event.set()

    # ─── mic capture (runs in thread → pushes to async queue) ────────
    def _start_mic(self):
        """Start mic capture in a background thread via sounddevice."""
        loop = asyncio.get_event_loop()

        def mic_callback(indata, frames, time_info, status):
            if status:
                log(f"Mic status: {status}", YELLOW)
            if self.mic_muted:
                return
            # indata is (frames, 1) int16
            pcm = indata[:, 0].tobytes()
            loop.call_soon_threadsafe(self._mic_queue.put_nowait, pcm)

        self._mic_stream = sd.InputStream(
            samplerate=MIC_RATE,
            channels=MIC_CHANNELS,
            dtype=MIC_DTYPE,
            blocksize=MIC_BLOCKSIZE,
            callback=mic_callback,
        )
        self._mic_stream.start()
        log(f"Mic capture started ({MIC_RATE} Hz, {MIC_BLOCKSIZE} samples/frame)", GREEN)

        # Async task that drains the queue and sends to WS
        self._mic_task = asyncio.create_task(self._mic_send_loop())

    async def _mic_send_loop(self):
        """Drain mic queue and send binary frames to the server."""
        try:
            while True:
                pcm = await self._mic_queue.get()
                if self.ws:
                    try:
                        await self.ws.send(pcm)
                    except Exception:
                        break  # connection gone, exit loop
        except asyncio.CancelledError:
            pass
        finally:
            if hasattr(self, '_mic_stream') and self._mic_stream:
                self._mic_stream.stop()
                self._mic_stream.close()
                log("Mic stopped", DIM)

    # ─── receive loop ────────────────────────────────────────────────
    async def _recv_loop(self):
        """Receive messages from the WebSocket server."""
        try:
            async for message in self.ws:
                if isinstance(message, bytes):
                    # Binary = TTS audio (24 kHz int16 PCM)
                    self.speaker.enqueue(message)
                else:
                    # JSON control
                    msg = json.loads(message)
                    self._handle_json(msg)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log(f"Receive error: {e}", RED)
            self.request_stop()

    def _handle_json(self, msg: dict):
        mtype = msg.get("type", "")

        if mtype == "ready":
            log("Server ready — starting mic capture", GREEN)
            self._state = "listening"
            log_state("listening")
            self._start_mic()

        elif mtype == "pong":
            pass  # keepalive ack

        elif mtype == "state_change":
            state = msg.get("state", "unknown")
            self._state = state.lower()
            log_state(state)

        elif mtype == "command":
            self._handle_command(msg.get("action", ""))

        elif mtype == "error":
            log(f"Server error [{msg.get('code')}]: {msg.get('message')}", RED)

        elif mtype == "text":
            text = msg.get("content", "")
            log(f"AI: {text}", BLUE)

        else:
            log(f"← {json.dumps(msg)}", DIM)

    def _handle_command(self, action: str):
        log(f"CMD ← {action}", MAGENTA)

        if action == "led_listening":
            pass  # terminal doesn't have LEDs, state_change shows it
        elif action == "led_thinking":
            pass
        elif action == "led_speaking":
            pass
        elif action == "led_off":
            pass
        elif action == "speaker_pause":
            self.speaker.pause()
        elif action == "speaker_resume":
            self.speaker.resume()
        elif action == "speaker_flush":
            self.speaker.flush()
        elif action == "mute_mic":
            self.mic_muted = True
            log("Mic MUTED by server", YELLOW)
        elif action == "unmute_mic":
            self.mic_muted = False
            log("Mic UNMUTED by server", GREEN)

    # ─── keepalive ───────────────────────────────────────────────────
    async def _ping_loop(self):
        try:
            while True:
                await asyncio.sleep(10)
                if self.ws:
                    try:
                        await self.ws.send(json.dumps({"type": "ping", "ts": time.time()}))
                    except Exception:
                        break
        except asyncio.CancelledError:
            pass


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="VoxCore v4 — Python WebSocket Test Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python websocket/test_client_py.py
  python websocket/test_client_py.py --url ws://192.168.1.10:8765
  python websocket/test_client_py.py --token my-key --user dharambir
  python websocket/test_client_py.py --list-devices
        """,
    )
    parser.add_argument("--url", default=None, help="WebSocket URL (default: ws://localhost:8765)")
    parser.add_argument("--host", default="localhost", help="Server host (default: localhost)")
    parser.add_argument("--port", type=int, default=8765, help="Server port (default: 8765)")
    parser.add_argument("--token", default="voxcore-dev-key-change-me", help="Auth token")
    parser.add_argument("--user", default="py_test_client", help="User ID")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    # Build URL
    if args.url:
        url = args.url
        # Append query params if not already present
        if "?" not in url:
            url += f"?user_id={args.user}&token={args.token}"
    else:
        url = f"ws://{args.host}:{args.port}/?user_id={args.user}&token={args.token}"

    # Banner
    print(f"""
{CYAN}╔══════════════════════════════════════════════════════════════╗
║         {BOLD}VoxCore v4 — Python WebSocket Test Client{RESET}{CYAN}         ║
║                                                              ║
║  Talk into your mic → pipeline on server → hear TTS reply    ║
║  Press Ctrl+C to quit                                        ║
╚══════════════════════════════════════════════════════════════╝{RESET}

  Server : {url}
  Mic    : {MIC_RATE} Hz, {MIC_BLOCKSIZE} samples/frame
  Speaker: {SPEAKER_RATE} Hz
""")

    client = VoxClient(url)

    # Handle Ctrl+C
    loop = asyncio.new_event_loop()

    def _signal_handler():
        log("Ctrl+C — stopping...", YELLOW)
        client.request_stop()

    loop.add_signal_handler(signal.SIGINT, _signal_handler)
    loop.add_signal_handler(signal.SIGTERM, _signal_handler)

    try:
        loop.run_until_complete(client.run())
    except KeyboardInterrupt:
        loop.run_until_complete(client._shutdown())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
