# VoxCore — Complete System Guide v3

> Full-Duplex Conversational AI | PersonaPlex-Inspired Architecture
> Groq APIs + Local Kokoro-ONNX TTS | February 2026

---

## What v3 Is About

**v3 is entirely focused on making the voice pipeline feel like a real person.**
Memory improvements (fact_store, extractor, retrieval re-ranking) are deferred to v4.

### What Changed From v2

| Area | Change |
|---|---|
| **Startup warmup** | Aria's voice synthesized silently at boot — AriaVoiceFilter + Gate 0 both pre-warmed before first word. Zero gap. |
| **Gate 0 (new)** | Spectral + temporal echo check runs before Gate 1. Catches loud echo residual that survived denoiser + AriaVoiceFilter. |
| **Layer 4 — V5 playback ref correlation (new)** | Cross-correlates mic chunks against recently played TTS audio. Works on 30ms chunks (unlike AriaFilter's 200ms minimum). Catches echo that AriaFilter misses for InterruptionDetector's raw consumer path. |
| **Neural echo removal** | `denoiser` + `AriaVoiceFilter` — before VAD, works on all hardware, any room |
| **Full PAUSE on PENDING** | TTS fully stops at PENDING — not ducked |
| **Instant filler at PENDING** | Pre-generated clip plays immediately while Gate 3 classifies in background |
| **Gate 3 acts on VAD-end** | Gate 3 reads early at 600ms but only acts when user finishes speaking |
| **No gates from PAUSED** | User speech from PAUSED bypasses gates — goes straight to normal STT path |
| **allam-2-7b = classify only** | Outputs DECISION + TYPE only. Never generates speech. |
| **Primary LLM bridge** | Filler clips (sure/got_it) play while llama-3.1-8b starts for real interrupts |
| **Two-path Gate 1** | Path A: short sharp commands at ~150ms; Path B: normal speech at 600ms |
| **Rolling re-fire on VAD-end** | Gate 3 re-fires when user finishes, not on fixed debounce |
| **7 interrupt sub-types** | STOP, PAUSE, CORRECTION, NEW_QUESTION, SAME_TOPIC_REDIRECT, URGENCY, DONE_LISTENING |

---

## Table of Contents

1. [What is VoxCore?](#1-what-is-voxcore)
2. [Architecture Overview](#2-architecture-overview)
3. [Echo Suppression — Four Layers](#3-echo-suppression)
4. [Startup Warmup — Zero Gap Protection](#4-startup-warmup)
5. [The Interrupt Pipeline — Complete](#5-the-interrupt-pipeline)
6. [AriaVoiceFilter — Exact State Rules](#6-ariavoicefilter-exact-state-rules)
7. [Conversation Dynamics](#7-conversation-dynamics)
8. [Tool and Agent System](#8-tool-and-agent-system)
9. [MCP Servers](#9-mcp-servers)
10. [Configuration Reference](#10-configuration-reference)
11. [API and Integration](#11-api-and-integration)
12. [Pipeline Walkthrough — Every Case](#12-pipeline-walkthrough)
13. [Changes Summary for Coding Agent](#13-changes-summary-for-coding-agent)

---

## 1. What is VoxCore?

VoxCore is a **voice-first, full-duplex conversational AI**. The user can speak naturally at any point — short commands, long explanations, mid-sentence additions — and the system responds correctly every time.

### Core Capabilities

| Feature | Description |
|---|---|
| **Five-layer echo suppression** | denoiser → AriaVoiceFilter → Gate 0 → V5 playback reference correlation — each layer catches what the previous missed |
| **Zero-gap warmup** | All echo protection layers pre-warmed at startup — fully ready before first word |
| **Two-path Gate 1** | Short commands (~150ms) on Path A (13× energy); sustained speech at 750ms on Path B (6.0× energy). Entry gate at 4.0× noise floor. |
| **Instant filler at PENDING** | Pre-generated clip plays within ~3ms of Gate 2 pass — fills silence while Gate 3 classifies |
| **VAD-end Gate 3 action** | Gate 3 reads early but only acts when user finishes speaking — prevents loop |
| **PAUSED state** | TTS frozen in RAM — resumes without LLM if user says "go ahead" |
| **7 interrupt sub-types** | Each routes to exact behavior — no secondary LLM call |
| **allam-2-7b = classifier only** | One word output. Never generates speech. |
| **Primary LLM bridge** | Filler clips bridge gap while llama-3.1-8b starts responding |
| **Natural yield points** | `[...]` inserts 300ms clean handoff — better than interrupting |
| **Backchanneling** | "mm-hmm", "right" clips during user pauses in LISTENING |

### What Powers It

| Component | Technology | Cost |
|---|---|---|
| STT | Groq Whisper Large V3 Turbo | Free tier |
| LLM (primary) | Groq llama-3.1-8b-instant | Free tier |
| LLM (classifier) | Groq allam-2-7b | Free tier |
| LLM (smart) | Groq llama-4-scout-17b | Free tier |
| LLM (agentic) | Groq kimi-k2-instruct | Free tier |
| LLM (deep) | Groq qwen/qwen3-32b | Free tier |
| TTS | Kokoro-ONNX (local) | Free, CPU |
| Echo layer 1 | denoiser (local) | Free, CPU |
| Echo layer 2 | AriaVoiceFilter / resemblyzer (local) | Free, CPU |
| Echo layer 3 | Gate 0 spectral check (local) | Free, CPU, ~0.5ms |
| Echo layer 4 | V5 playback reference correlation (local) | Free, CPU, ~1ms |
| Memory | ChromaDB (local) | Free |
| Safety | Groq Llama Guard 4 | Free tier |

---

## 2. Architecture Overview

### Full Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RAW MIC INPUT                                │
│  Microphone → 16kHz PCM chunks (32ms each)                         │
└──────────────────────────┬──────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   AUDIO CLEANING LAYER                              │
│                                                                     │
│  denoiser — ALWAYS runs regardless of state                         │
│    Removes: nonlinear echo, room noise, HVAC, background sounds     │
│    Enhances: human voice (cleaner, not removed)                     │
│    Latency: ~5ms per 32ms chunk                                     │
│                                                                     │
│  AriaVoiceFilter — ONLY while TTS real speech is playing           │
│    Rejects: mic chunks that sound like Aria (speaker identity)      │
│    Passes: all other voices unchanged                               │
│    Pre-warmed at startup — ready from word one                      │
└──────────────────────────┬──────────────────────────────────────────┘
                           ▼ clean audio
         ┌─────────────────┴──────────────────┐
         │                                    │
    LISTENING state                      SPEAKING state
    Filter OFF, gates IDLE               Filter ON, gates ACTIVE
         │                                    │
    VAD → STT → LLM → TTS          GATE 0 — Spectral echo check
                                         (~0.5ms, only SPEAKING)
                                    Does mic sound spectrally like
                                    recent TTS output?
                                    YES → drop (echo residual)
                                    NO  → continue
                                         │
                                    V5 — Playback ref correlation
                                         (~1ms, only SPEAKING)
                                    Cross-correlate mic chunk against
                                    recently played TTS audio buffer
                                    Peak correlation > 0.45 → drop
                                    NO  → continue
                                         │
                                    GATE 1 — Energy + Duration
                                    Path A (150ms, 13× energy) or
                                    Path B (750ms, 6.0× energy) or DROP
                                         │
                                    GATE 2 — Filler word check
                                    Filler → FILLER_REACTION, drop
                                    Real  → continue
                                         │
                                    PENDING (~3ms):
                                      TTS fully pauses
                                      Filter OFF
                                      Filler clip plays instantly
                                      Gate 3 starts in background
                                         │
                                    Gate 3 classifies
                                    Acts when user finishes speaking
                                         │
                                    interrupt_router routes
                                    on DECISION + TYPE
```

### Turn State Machine — 6 States

```
                   ┌──────────┐
                   │ LISTENING │ ← VAD active, backchannel active
                   │          │   AriaVoiceFilter OFF
                   └────┬─────┘   All gates IDLE
                        │ SPEECH_END + 500ms silence
                        ▼
                   ┌──────────┐
                   │ THINKING │ ← STT done, primary LLM streaming
                   └────┬─────┘   Nothing else active
                        │ first TTS chunk ready
                        ▼
                   ┌──────────┐
                   │ SPEAKING │ ← TTS playing
                   │          │   AriaVoiceFilter ON
                   └────┬─────┘   Gate 0 + Gate 1 + Gate 2 ACTIVE
                        │
         ┌──────────────┼─────────────────┬──────────────┐
         ▼              ▼                 ▼              ▼
   LISTENING      ┌──────────┐      ┌──────────┐  ┌────────────┐
   (natural end)  │SOFT_INJECT│     │  PAUSED  │  │INTERRUPTED │
                  └────┬─────┘      └────┬─────┘  └─────┬──────┘
                       │                 │               │
                  TTS resumes       TTS frozen      filler clip plays
                  filter ON         resumable       → primary LLM
                  context injected  no LLM needed
```

### State Transition Rules

```
LISTENING   → THINKING      SPEECH_END from VAD
THINKING    → SPEAKING      first TTS chunk ready from Kokoro
SPEAKING    → LISTENING     PLAYBACK_DONE (natural end)
SPEAKING    → SOFT_INJECT   CLASSIFIED → INJECT
SOFT_INJECT → SPEAKING      immediately after text_injector.inject()
SPEAKING    → PAUSED        CLASSIFIED → INTERRUPT:PAUSE
PAUSED      → SPEAKING      resume phrase detected OR 3s auto-resume
PAUSED      → THINKING      user asks something new (no gates involved)
SPEAKING    → INTERRUPTED   CLASSIFIED → INTERRUPT (any type except PAUSE)
INTERRUPTED → THINKING      immediately (filler clip plays, LLM starts)
```

### File Structure

```
voxcore/
├── main.py                              # [UPDATED] — startup warmup sequence
├── config.yaml                          # [UPDATED] — new blocks
├── .env
├── requirements.txt                     # [UPDATED] — resemblyzer, denoiser, torch
│
├── core/
│   ├── session.py                       # [UPDATED] — 6 TurnStates
│   ├── event_bus.py                     # [UPDATED] — new event types
│   └── turn_manager.py                  # [UPDATED] — PAUSED logic, routing
│
├── input/
│   ├── mic_stream.py                    # [UPDATED] — denoiser + gate0 + V5 playback ref correlation
│   ├── aria_voice_filter.py             # [NEW] — resemblyzer, self-learning
│   ├── gate0_echo_check.py              # [NEW] — spectral + temporal echo gate
│   ├── vad.py                           # unchanged — receives clean audio only
│   ├── stt.py                           # unchanged
│   ├── text_injector.py                 # [UPDATED] — inject_point support
│   ├── interruption_detector.py         # [UPDATED] — two-path Gate 1
│   ├── filler_detector.py               # Gate 2 — unchanged logic
│   └── speaking_monitor.py              # [UPDATED] — VAD-end action, Gate 3
│
├── brain/
│   ├── llm_client.py                    # unchanged
│   ├── prompt_builder.py                # [UPDATED] — memory block injection, [ref:] handling
│   ├── response_parser.py               # [UPDATED] — [ref:sc_...] stripping, inject_point
│   ├── interrupt_router.py              # [NEW] — all 7 sub-type actions
│   ├── emotion_tagger.py                # unchanged
│   └── router.py                        # [UPDATED] — background_llm.create_task()
│
├── output/
│   ├── tts_client.py                    # [UPDATED] — pause/resume + synthesize_silent()
│   ├── audio_player.py                  # [UPDATED] — pause/resume/flush, filter control,
│   │                                    #   V5 playback ref feeding + clearing
│   ├── voice_profile.py                 # unchanged
│   └── pre_pause_clips/                 # [NEW]
│       ├── breath.wav                   # ~200ms soft breath
│       └── mm.wav                       # ~300ms acknowledgment
│
├── memory/                              # [REBUILT v3.6] — five-tier memory system
│   ├── __init__.py                      # [UPDATED] — exports all new classes
│   ├── short_term.py                    # Tier 1: RAM deque, 20 turns + emotions
│   ├── session_cache.py                 # [NEW] Tier 2: RAM dict, full tool results
│   ├── fact_store.py                    # [NEW] Tier 3: SQLite, permanent user facts
│   ├── procedural.py                    # [NEW] Tier 4: SQLite, instructions/reminders
│   ├── long_term.py                     # [UPDATED] Tier 5: ChromaDB, typed + re-ranked
│   ├── compressor.py                    # [UPDATED] — bug fix: clears working memory
│   ├── background_llm.py                # [NEW] — fire-and-forget extraction
│   ├── loader.py                        # [NEW] — session-start memory block assembly
│   └── retriever.py                     # [NEW] — parallel recall across all tiers
│
├── safety/
│   └── guard.py                         # unchanged
│
├── backchannel/
│   ├── cue_detector.py                  # unchanged
│   ├── selector.py                      # [UPDATED] — FILLER_REACTION handling
│   ├── generator.py                     # [UPDATED] — generates pre-pause clips
│   └── clips/heart/
│       ├── [existing 15 backchannel clips]
│       ├── sure.wav                     # bridge clip for NEW_QUESTION/REDIRECT
│       ├── got_it.wav                   # bridge clip for CORRECTION
│       ├── laughs.wav
│       ├── chuckles.wav
│       ├── light_laugh.wav
│       └── sighs.wav
│
├── agent/
│   ├── __init__.py
│   ├── text_out.py                      # Parses <agent> JSON tags
│   ├── tool_router.py                   # [UPDATED] — memory_recall + session_cache_qa
│   ├── slow_llm.py                      # Background heavy model calls
│   └── tools/
│       ├── __init__.py
│       ├── base_tool.py                 # Abstract base for all tools
│       ├── web_search.py                # Tavily / DuckDuckGo
│       ├── article_fetch.py             # [UPDATED] — session cache integration
│       ├── memory_recall.py             # [NEW] — search across memory tiers
│       └── session_cache_qa.py          # [NEW] — QA over cached content
│
└── interfaces/
    ├── cli.py
    ├── websocket_server.py
    └── api.py
```

---

## 3. Echo Suppression — Five Layers

The mic picks up a mix during SPEAKING: Aria's voice from the speaker (echo), room reflections, and background noise. Echo must be removed before it reaches Gate 1. If it reaches Gate 1 with high energy, it can pass all the way to Gate 3, get classified as INJECT or INTERRUPT, and break the conversation.

Five independent layers each catch what the previous one missed.

### Why Previous Approaches Failed

NLMS and correlation-based AEC assume echo is a **linear** delayed copy of speaker output. Real speaker hardware introduces nonlinear harmonic distortion. The echo at the mic is a physically distorted version of what was played. Linear filters partially cancel it, leaving residual that triggers gates. No amount of threshold tuning fixes this because the math is wrong for the physics.

### Layer 1 — denoiser (Always On)

Facebook Research's real-time speech enhancement model. Trained on thousands of real room and speaker combinations. Handles nonlinear distortions because it learned from real nonlinear data. Works identically on any hardware.

```bash
pip install denoiser
```

Latency: ~5ms per 32ms chunk on CPU. Always runs regardless of state. Enhances human voice, removes non-speech signals. Output: cleaner human speech with much of the echo removed.

### Layer 2 — AriaVoiceFilter (During Real TTS Playback Only)

`resemblyzer` computes a 256-dim speaker embedding that uniquely identifies a voice. `AriaVoiceFilter` holds Aria's embedding (pre-warmed at startup) and rejects mic chunks that have high cosine similarity to Aria's voice.

No enrollment needed. Works for any user — any human voice has a fundamentally different embedding from a TTS-generated voice.

```bash
pip install resemblyzer
```

**File:** `input/aria_voice_filter.py`

```python
import numpy as np
from collections import deque
from resemblyzer import VoiceEncoder, preprocess_wav

class AriaVoiceFilter:
    """
    Holds Aria's speaker embedding.
    Pre-warmed at startup via feed_aria_audio() from synthesize_silent().
    Updated continuously from live TTS output during conversation.
    Rejects mic chunks that sound like Aria.
    Passes all other voices unchanged.
    """

    def __init__(self, similarity_threshold: float = 0.75):
        self._encoder = VoiceEncoder()
        self._threshold = similarity_threshold
        self._aria_chunks: deque = deque(maxlen=50)   # ~10s rolling
        self._aria_embedding: np.ndarray | None = None
        self._embedding_ready = False
        self._samples_since_update = 0

    @property
    def is_ready(self) -> bool:
        return self._embedding_ready

    def feed_aria_audio(self, chunk: np.ndarray):
        """
        Called by:
          1. startup_warmup() — from synthesize_silent() output (pre-warm)
          2. audio_player._play_tts_chunk() — from live TTS during conversation

        NOT called for: breath.wav, mm.wav, laughs.wav, backchannel clips,
        bridge clips (sure.wav, got_it.wav). Only real Kokoro TTS speech.
        """
        self._aria_chunks.append(chunk)
        self._samples_since_update += len(chunk)

        # Recompute embedding every 2s of new Aria speech
        if self._samples_since_update >= 32000:
            self._samples_since_update = 0
            combined = np.concatenate(list(self._aria_chunks))
            wav = preprocess_wav(combined, source_sr=16000)
            self._aria_embedding = self._encoder.embed_utterance(wav)
            self._embedding_ready = True

    def is_aria_echo(self, chunk: np.ndarray) -> bool:
        """
        True  → sounds like Aria → drop (echo)
        False → sounds different from Aria → pass to Gate 0
        
        Before is_ready: always returns False.
        This should never happen after startup warmup.
        """
        if not self._embedding_ready or len(chunk) < 3200:
            return False
        wav = preprocess_wav(chunk, source_sr=16000)
        mic_embedding = self._encoder.embed_utterance(wav)
        return float(np.dot(self._aria_embedding, mic_embedding)) > self._threshold

    def reset(self):
        """Call on persona/voice change. Re-warmup required after reset."""
        self._aria_chunks.clear()
        self._aria_embedding = None
        self._embedding_ready = False
        self._samples_since_update = 0
```

### Layer 3 — Gate 0 Spectral Echo Check (During SPEAKING Only)

This layer catches the case you identified: loud echo residual that survived denoiser and AriaVoiceFilter. When speakers are loud or the room has strong acoustics, echo residual can be energetic enough to pass Gate 1's energy threshold. Gate 0 catches it before Gate 1 runs.

**The insight:** Room acoustics change amplitude and add reverb, but do NOT fundamentally change the frequency profile of a voice. Aria's formants (the frequency peaks that define her voice character) survive room bounce. A mic chunk that sounds like Aria's echo will have a spectral profile very similar to what TTS just outputted.

**File:** `input/gate0_echo_check.py`

```python
import numpy as np
from collections import deque

class Gate0EchoCheck:
    """
    Spectral + temporal echo gate.
    Runs before Gate 1, only during SPEAKING state.
    Pre-warmed at startup from synthesize_silent() output.

    Maintains a rolling exponential moving average of recent TTS
    output spectra. Compares each mic chunk's spectrum against it.
    High similarity = mic sounds like TTS = echo → drop.
    """

    def __init__(self,
                 spectral_threshold: float = 0.85,
                 temporal_gate_ms: int = 80,
                 ema_alpha: float = 0.1):
        self._threshold = spectral_threshold
        self._temporal_gate_ms = temporal_gate_ms
        self._ema_alpha = ema_alpha           # exponential moving average weight
        self._tts_spectrum_ema: np.ndarray | None = None
        self._is_ready = False
        self._last_tts_chunk_time: float = 0.0

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def feed_tts_spectrum(self, chunk: np.ndarray):
        """
        Called by:
          1. startup_warmup() — from synthesize_silent() output (pre-warm)
          2. audio_player._play_tts_chunk() — from live TTS output

        NOT called for clips. Same exclusion rule as AriaVoiceFilter.
        Updates rolling spectral fingerprint of Aria's TTS output.
        """
        import time
        self._last_tts_chunk_time = time.time()

        spectrum = np.abs(np.fft.rfft(chunk.astype(np.float32)))

        if self._tts_spectrum_ema is None:
            self._tts_spectrum_ema = spectrum
        else:
            # Exponential moving average — weights recent chunks more
            self._tts_spectrum_ema = (
                self._ema_alpha * spectrum +
                (1 - self._ema_alpha) * self._tts_spectrum_ema
            )
        self._is_ready = True

    def is_echo(self, mic_chunk: np.ndarray) -> bool:
        """
        True  → mic sounds like recent TTS output → drop (echo)
        False → mic sounds spectrally different → pass to Gate 1

        Two independent checks — either alone can catch echo:
          Check 1: Temporal — was TTS playing very recently?
          Check 2: Spectral — does mic frequency profile match TTS?
        """
        import time

        if not self._is_ready:
            return False

        # Check 1: Temporal gate
        # Echo physically cannot arrive before TTS plays.
        # If TTS chunk played < 80ms ago, any mic audio is likely echo.
        # 80ms covers speaker-to-mic travel time in most rooms (5-50ms)
        # plus a safety margin.
        ms_since_tts = (time.time() - self._last_tts_chunk_time) * 1000
        if ms_since_tts < self._temporal_gate_ms:
            return True

        # Check 2: Spectral similarity
        # Compare mic frequency profile against rolling TTS spectrum.
        mic_spectrum = np.abs(np.fft.rfft(mic_chunk.astype(np.float32)))

        # Normalize both to unit vectors (cosine similarity)
        mic_norm = mic_spectrum / (np.linalg.norm(mic_spectrum) + 1e-8)
        tts_norm = self._tts_spectrum_ema / (
            np.linalg.norm(self._tts_spectrum_ema) + 1e-8
        )

        similarity = float(np.dot(mic_norm, tts_norm))
        return similarity > self._threshold
```

**Why exponential moving average for the TTS spectrum:**
A simple last-chunk approach would only represent what Aria just said. The EMA weights recent chunks more but keeps a memory of the last few seconds. This means Gate 0 correctly identifies echo even when the room reverb includes reflections from a few sentences ago.

### Layer 4 — V5 Playback Reference Correlation (During SPEAKING Only)

**The problem Layers 2 and 3 leave unsolved:**

AriaVoiceFilter (`is_aria_echo()`) requires `len(chunk) >= 3200` samples (200ms of audio at 16kHz). But InterruptionDetector's raw mic consumer receives chunks of only 480 samples (30ms). AriaVoiceFilter **never fires** for the InterruptionDetector path. Gate 0's spectral check at 0.92 threshold is unreliable through acoustic path distortion. Result: echo leaks through to Gate 1, causing false interrupts during AI speech.

**The solution:** Cross-correlate each mic chunk against a rolling buffer of recently played TTS audio.

**File:** `input/mic_stream.py` — new methods:

```python
# Constructor additions:
self._playback_ref: collections.deque = collections.deque(maxlen=8000)  # ~500ms at 16kHz
self._playback_ref_lock: threading.Lock = threading.Lock()
self._playback_corr_threshold: float = config.get("playback_correlation_threshold", 0.45)

def feed_playback_reference(self, chunk: np.ndarray):
    """Called by AudioPlayer per frame during real TTS playback.
    Extends the rolling reference buffer with resampled 16kHz TTS audio."""
    with self._playback_ref_lock:
        self._playback_ref.extend(chunk.astype(np.float32).tolist())

def clear_playback_reference(self):
    """Called when TTS stops (pause/flush/playback-end). Clears stale reference."""
    with self._playback_ref_lock:
        self._playback_ref.clear()

def _check_playback_correlation(self, mic_chunk: np.ndarray) -> bool:
    """Normalized cross-correlation of mic chunk against reference buffer.
    Pure echo correlates at 0.6–0.9; user speaking over echo drops to ~0.25–0.35.
    Returns True if peak correlation exceeds threshold (echo detected)."""
    with self._playback_ref_lock:
        if len(self._playback_ref) < len(mic_chunk):
            return False
        ref = np.array(self._playback_ref, dtype=np.float32)
    mic = mic_chunk.astype(np.float32)
    mic_power = float(np.dot(mic, mic))
    if mic_power < 1e-8:
        return False
    # Sliding window correlation using cumsum for efficiency
    corr = np.correlate(ref, mic, mode='valid')
    cum = np.cumsum(np.concatenate(([0.0], ref ** 2)))
    seg_len = len(mic)
    seg_power = cum[seg_len:] - cum[:len(corr)]
    denom = np.sqrt(seg_power * mic_power) + 1e-12
    norm_corr = corr / denom
    return float(np.max(np.abs(norm_corr))) > self._playback_corr_threshold
```

**How it's called in `check_echo()`:**

```python
def check_echo(self, chunk) -> bool:
    # V5: Playback reference correlation — works on 30ms chunks
    if self._check_playback_correlation(chunk):
        return True
    # Layer 2: AriaVoiceFilter (needs 200ms — may not fire on short chunks)
    if self._aria_filter and self._aria_filter.is_aria_echo(chunk):
        return True
    # Layer 3: Gate 0 spectral check
    if self._gate0 and self._gate0.is_echo(chunk):
        return True
    return False
```

**How AudioPlayer feeds the reference:**

```python
# In audio_player._play_audio(), after existing AriaFilter + Gate0 feeds:
self._mic_stream.feed_playback_reference(feed_frame)

# In audio_player.pause(), flush(), and _playback_loop finally:
self._mic_stream.clear_playback_reference()
```

**Why this works:**

| Scenario | Peak Correlation | Result |
|---|---|---|
| Pure echo (user silent) | 0.6–0.9 | **Blocked** (above 0.45) |
| User speaking over echo | 0.2–0.35 | **Passes** (below 0.45) |
| User speaking, no echo | ~0.0 | **Passes** |

The playback reference buffer holds ~500ms of recently played TTS audio (8000 samples at 16kHz). This is enough to catch echo at any reasonable room delay. When TTS stops (pause/flush/end), the buffer is cleared immediately to prevent stale data from blocking real user speech.

**File:** `input/mic_stream.py` — the complete processing chain (updated):

```python
async def _process_chunk(self, raw: bytes) -> bytes | None:

    # --- Layer 1: denoiser — ALWAYS runs ---
    audio = torch.frombuffer(raw, dtype=torch.int16).float() / 32768.0
    audio = audio.unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        clean = self._enhancer(audio)[0]
    clean_np = (clean.squeeze().numpy() * 32768).astype('int16')

    # --- Layers 2 + 3 + 4: Only during SPEAKING (filter_active=True) ---
    if self._filter_active:

        # Layer 2: AriaVoiceFilter — speaker identity check
        if self._aria_filter.is_aria_echo(clean_np):
            return None   # sounds like Aria → drop

        # Layer 3: Gate 0 — spectral echo check
        if self._gate0.is_echo(clean_np):
            return None   # spectrally matches recent TTS → drop

    # --- Raw consumer path uses check_echo() which includes Layer 4 ---
    # InterruptionDetector's raw consumer calls check_echo(chunk)
    # which runs V5 playback correlation first (works on 30ms chunks),
    # then falls back to AriaVoiceFilter + Gate 0

    return clean_np.tobytes()
```

---

## 4. Startup Warmup — Zero Gap Protection

Without warmup, AriaVoiceFilter needs 2 seconds of live Aria speech before its embedding is ready. During those 2 seconds there is no Layer 2 or Layer 3 protection — echo could reach Gate 1 and break the very first conversation turn.

The fix: synthesize Aria's voice silently at boot and use it to pre-warm both systems. No speaker output. ~200ms compute. Done before the first conversation word.

**File:** `main.py` — new `startup_warmup()` function:

```python
async def startup_warmup(aria_filter, gate0, tts_client, config):
    """
    Runs at boot after TTS model loads and before first conversation.
    Synthesizes ~3s of Aria's voice silently (no speaker output).
    Feeds audio to both AriaVoiceFilter and Gate0EchoCheck.
    Both systems are fully ready before word one.

    Cost: ~200ms CPU (Kokoro-ONNX synthesis, local).
    No API calls. No speaker output. No microphone needed.
    """
    warmup_text = (
        "Hello, I am getting ready for our conversation. "
        "Just a moment while I initialize. "
        "Almost ready now, thank you for waiting."
    )

    # Synthesize without playing — returns raw int16 numpy array
    warmup_audio: np.ndarray = await tts_client.synthesize_silent(
        text=warmup_text,
        voice=config["persona"]["voice"]
    )

    # Feed to both systems in the same pass
    chunk_size = 512   # 32ms at 16kHz
    for i in range(0, len(warmup_audio) - chunk_size, chunk_size):
        chunk = warmup_audio[i : i + chunk_size]

        aria_filter.feed_aria_audio(chunk)   # builds speaker embedding
        gate0.feed_tts_spectrum(chunk)       # builds spectral fingerprint

    assert aria_filter.is_ready, "[Warmup] AriaVoiceFilter not ready — check synthesis output"
    assert gate0.is_ready, "[Warmup] Gate0 not ready — check synthesis output"

    print("[VoxCore Warmup] Echo protection fully ready. Starting conversation.")
```

**File:** `output/tts_client.py` — new `synthesize_silent()` method:

```python
async def synthesize_silent(self, text: str, voice: str) -> np.ndarray:
    """
    Synthesizes text using Kokoro-ONNX.
    Returns raw int16 numpy array at 16kHz.
    Does NOT play audio. Does NOT interact with AudioPlayer.
    Used only by startup_warmup().
    """
    # Same synthesis path as normal TTS but returns array instead of queuing
    audio_chunks = []
    async for chunk in self._kokoro.synthesize_stream(text, voice=voice):
        # Resample from 24kHz (Kokoro output) to 16kHz (filter input)
        resampled = resample_16k(chunk)
        audio_chunks.append(resampled)

    if not audio_chunks:
        raise RuntimeError("synthesize_silent: Kokoro returned no audio")

    return np.concatenate(audio_chunks)
```

**Boot sequence:**

```
python main.py starts
  │
  ▼ (~1-2s)
Kokoro-ONNX model loads
  │
  ▼ (~1s)
resemblyzer VoiceEncoder loads
  │
  ▼ (~200ms) ← NEW
startup_warmup():
  synthesize 3s of Aria speech silently
  feed all chunks to aria_filter.feed_aria_audio()
  feed all chunks to gate0.feed_tts_spectrum()
  assert both is_ready == True
  │
  ▼
First conversation starts
AriaVoiceFilter: is_ready = True  ← fully protected from word one
Gate0:           is_ready = True  ← fully protected from word one
No warmup period. No fallback. No gap.
```

**On persona/voice change:**

```python
# In main.py on persona switch:
aria_filter.reset()       # clear old embedding, is_ready → False
gate0_check = Gate0EchoCheck(...)   # create fresh Gate0 (or add reset() method)
mic_stream.set_filter_active(False) # protection off until re-warmed

await startup_warmup(aria_filter, gate0_check, tts_client, new_config)
# Both re-warmed with new voice in ~200ms
# Protection back on before new persona speaks first word
```

---

## 5. The Interrupt Pipeline — Complete

The full gate chain runs **only during SPEAKING state**. During LISTENING, THINKING, and PAUSED it is completely idle.

### Gate 0 — Spectral + Temporal Echo (New, Before Gate 1)

**File:** `input/gate0_echo_check.py` (documented in Section 3)

Catches loud echo residual that survived denoiser and AriaVoiceFilter. Two checks:

```
Check 1 — Temporal:
  Was a TTS chunk played in the last 400ms?
  YES → almost certainly echo → drop
  (400ms covers inter-sentence synthesis gaps and room echo tail.)

Check 2 — Spectral:
  Does mic chunk's frequency profile match rolling TTS spectrum EMA?
  Cosine similarity > 0.92 → spectrally matches Aria's voice → drop
  (Raised from 0.85 to 0.92 with full 257-bin FFT for precision.)
```

If either check returns True → drop chunk.

### V5 Layer — Playback Reference Correlation (Before Gate 1)

**File:** `input/mic_stream.py` (documented in Section 3)

Catches echo that passes Gate 0 by cross-correlating mic audio against recently played TTS. Works on 30ms chunks — critical for InterruptionDetector's raw consumer path where AriaVoiceFilter's 200ms minimum means it never fires.

```
Peak normalized cross-correlation > 0.45 → echo → drop
(Pure echo: 0.6–0.9, user+echo: 0.2–0.35)
```

If correlation exceeds threshold → drop chunk. Gate 1 never sees it.

### Gate 1 — Two Paths (Energy + Duration)

**File:** `input/interruption_detector.py`

After echo is removed, what remains is genuine audio from the room. Gate 1 separates real user speech from incidental sounds (coughs, bumps, distant noise).

**Entry gate:** Before entering any accumulation window, chunk RMS must exceed `noise_floor × 4.0` (configurable via `gate1_entry_multiplier`). This prevents low-energy echo residual from even starting the accumulation timer.

```
PATH A — High Energy Burst
  Catches: "stop!", "no!", "wait!" — short sharp command words
  Trigger: energy > noise_floor × 13  AND  duration ≥ 150ms
  Next: skip Gate 2 (urgency — don't risk dropping command as filler)
        go straight to Gate 3 with urgency_hint=True

  (Threshold raised 8→13: observed echo ceiling was 8.6×;
   real near-field speech easily exceeds 20×)

PATH B — Normal Conversational Speech
  Catches: questions, sentences, explanations
  Trigger: duration ≥ 750ms  AND  energy > noise_floor × 6.0
  Next: proceed to Gate 2

  (Duration raised 600→750ms: echo bursts are shorter than real speech.
   Threshold raised 3→6.0: observed echo ceiling was 4.9×;
   real speech is comfortably above 6×)

DROP
  Everything else: coughs, bumps, breaths, residual sounds
  Action: discard silently
```

```python
# input/interruption_detector.py
def check(self, chunk: bytes, duration_ms: int) -> str:
    """Returns: 'HIGH_ENERGY_BURST' | 'NORMAL_SPEECH' | 'DROP'"""
    rms = compute_rms(chunk)
    ratio = rms / (self._noise_floor + 1e-8)

    # Entry gate: must exceed noise floor × entry_multiplier
    if ratio < self._energy_multiplier:  # default 4.0
        return "DROP"

    if duration_ms >= 150 and ratio > 13.0:
        return "HIGH_ENERGY_BURST"

    if duration_ms >= 750 and ratio > 6.0:
        return "NORMAL_SPEECH"

    return "DROP"
```

### Gate 2 — Filler Word Check

**File:** `input/filler_detector.py`

Path A skips this. Path B checks partial transcript against filler list.

```python
FILLER_LIST = {
    "yeah", "yep", "okay", "ok", "right", "sure", "uh-huh",
    "mm-hmm", "mm", "hmm", "haha", "lol", "wow", "oh", "nice",
    "cool", "great", "interesting", "really", "totally", "exactly"
}

def check(self, partial_transcript: str) -> bool:
    """True = filler (stop), False = real speech (continue to PENDING)."""
    text = partial_transcript.lower().strip().rstrip(".")
    return text in FILLER_LIST
```

When Gate 2 returns True: fire `FILLER_REACTION` event with sentiment. AI continues speaking uninterrupted. TTS never pauses.

### PENDING — What Happens in ~3ms

When Gate 2 passes (real speech confirmed) or Path A fires (urgency), these steps execute in order:

```
1. audio_player.pause()
   TTS freezes at current 20ms frame boundary
   Full stop — not a volume duck
   Remaining buffered TTS chunks stay in RAM (not discarded yet)
   
2. mic_stream.set_filter_active(False)
   AriaVoiceFilter and Gate 0 both turn OFF
   Nothing playing from speaker → nothing to filter
   User's voice passes to Gate 3 transcription completely clean
   
3. audio_player.play_clip_looping(pre_pause_clip, gate3_done_event)
   breath.wav or mm.wav starts playing immediately
   Does NOT call aria_filter.feed_aria_audio()
   Does NOT call gate0.feed_tts_spectrum()
   Does NOT change filter_active
   Loops once if Gate 3 takes longer than clip length
   Stops automatically when gate3_done_event is set
   
4. Event.PENDING fires
   WebSocket clients notified
   
5. asyncio.create_task(_run_gate3(audio_buffer, urgency_hint))
   Gate 3 starts in background — does NOT block
   PENDING handler returns immediately
```

```python
# input/speaking_monitor.py
async def _on_gate_passed(self, audio_buffer: bytes, urgency_hint: bool = False):
    self._gate3_done_event.clear()

    await self._audio_player.pause()
    self._mic_stream.set_filter_active(False)

    asyncio.create_task(
        self._audio_player.play_clip_looping(
            self._config["speaking_monitor"]["pre_pause_clip"] + ".wav",
            stop_event=self._gate3_done_event
        )
    )
    await self._event_bus.publish(Event.PENDING, {"urgency_hint": urgency_hint})
    asyncio.create_task(self._run_gate3(audio_buffer, urgency_hint))
```

### Gate 3 — allam-2-7b Classification (Background)

**allam-2-7b does exactly one thing: output DECISION + TYPE. It never generates speech.**

Two API calls in sequence:

```
Step 1: whisper-turbo transcribes captured audio (~250ms, Groq)
Step 2: allam-2-7b classifies with context (~150ms, Groq)
Total: ~400-600ms — runs while filler clip plays
```

**Classifier prompt:**

```
System:
You are a real-time speech intent classifier for a voice AI.
The AI was speaking. It paused because the user spoke.
Classify the user's intent.

DECISION:
  IGNORE    — reaction only ("haha", "wow", "yeah exactly") — resume AI
  INJECT    — meaningful addition, AI should NOT stop and restart
  INTERRUPT — user needs a real response from the AI

If INTERRUPT, TYPE:
  STOP              — wants silence, no response
  PAUSE             — brief hold, AI content may resume
  CORRECTION        — fixing something AI said
  NEW_QUESTION      — new unrelated topic
  SAME_TOPIC_REDIRECT — follow-up on current topic
  URGENCY           — safety, distress, critical issue
  DONE_LISTENING    — user has heard enough

Context:
AI was saying: "{last_30_words}"
User said: "{transcript}"
Speech position: "{sentence_start|mid_sentence|after_yield|sentence_boundary}"
Urgency hint: {True|False}

Reply ONLY:
DECISION: IGNORE|INJECT|INTERRUPT
TYPE: STOP|PAUSE|CORRECTION|NEW_QUESTION|SAME_TOPIC_REDIRECT|URGENCY|DONE_LISTENING|null
```

**Timeout:** If Gate 3 takes longer than 3 seconds (API cold start, rate limit) → default to `INTERRUPT:PAUSE`. User retains control. Never default to IGNORE — resuming when user clearly spoke is wrong.

### VAD-End Action — The Loop Prevention Fix

Gate 3 reads early on the first 600ms (Path B) or 450ms (Path A) but **only acts when the user has finished speaking**.

```python
async def _wait_for_user_to_finish(
    self, initial_audio: bytes, initial_result: dict
) -> dict:
    """
    Holds the Gate 3 result until VAD confirms user stopped.
    Accumulates audio and re-classifies every 1.5s if user keeps speaking.
    This prevents the main loop: AI resumes mid-user-sentence → re-interrupt.
    """
    accumulated = bytearray(initial_audio)
    current_result = initial_result
    silence_start: float | None = None

    while True:
        await asyncio.sleep(0.05)

        # Lower VAD threshold in interrupt context:
        # TTS is paused → less background noise → be more sensitive
        user_speaking = self._vad.check_energy(
            self._mic_stream.last_200ms(),
            threshold=0.4   # vs normal 0.65 in LISTENING
        )

        if user_speaking:
            silence_start = None
            accumulated.extend(self._mic_stream.last_200ms())

            # Re-classify every 1.5s of new speech — handles escalation
            # e.g. "actually" (INJECT) → "actually I want to ask about X" (NEW_QUESTION)
            if len(accumulated) >= 24000:
                transcript = await self._transcribe(bytes(accumulated))
                current_result = await self._classify(
                    transcript,
                    initial_result.get("urgency_hint", False)
                )
                accumulated = bytearray()
        else:
            if silence_start is None:
                silence_start = time.time()

            # 300ms silence = user finished
            # Shorter than normal 500ms because TTS is already paused —
            # we are waiting, not trying to catch end of turn
            if time.time() - silence_start >= 0.3:
                return current_result
```

**What this means for different utterance lengths:**

```
"Stop!" (Path A, ~200ms):
  PENDING at ~200ms → Gate 3 at ~450ms
  User already silent → _wait exits immediately → act

"What's the weather in London?" (~1200ms):
  PENDING at 600ms → Gate 3 early read on 600ms window
  User still speaking "...in London?" → _wait accumulates
  User finishes → 300ms silence at ~1500ms → act with full audio

Long explanation (4+ seconds):
  PENDING at 600ms → early read holds
  Re-classifies every 1.5s as audio accumulates
  User finishes → 300ms silence → act with complete utterance
  TTS remained paused the entire time — no re-interrupt loop
```

---

## 6. AriaVoiceFilter — Exact State Rules

```
State                TTS Real Speech?   filter_active   Reason
─────────────────────────────────────────────────────────────────────
LISTENING            No                 False           Nothing to filter
THINKING             No                 False           No audio output
SPEAKING             Yes                True            Filter + Gate 0 active
SPEAKING + PENDING   No (paused)        False           TTS paused, nothing playing
PAUSED               No (frozen)        False           "go ahead" must be clean
SOFT_INJECT          Yes (resumed)      True            TTS resumed
Post-IGNORE          Yes (resumed)      True            TTS resumed
[...] yield (300ms)  No                 False           Nothing playing
Clip playing         No                 False           Clips ≠ real TTS
(breath/mm/laughs/
sure/got_it/
backchannel)
```

**Code in `audio_player.py`:**

```python
async def pause(self):
    self._stream.stop()
    self._paused = True
    self._mic_stream.set_filter_active(False)   # Gate 0 + AriaFilter OFF

async def resume(self):
    self._paused = False
    # Restore unplayed remainder of current chunk
    if self._current_chunk is not None:
        remainder = self._current_chunk[self._current_position:]
        if len(remainder) > 0:
            self._pending_chunks.appendleft(remainder)
        self._current_chunk = None
        self._current_position = 0
    self._stream.start()
    self._mic_stream.set_filter_active(True)    # Gate 0 + AriaFilter ON

async def flush(self):
    """Hard stop. Discard all buffered TTS. Used for INTERRUPT."""
    self._pending_chunks.clear()
    self._current_chunk = None
    self._current_position = 0
    self._stream.stop()
    self._paused = False
    self._mic_stream.set_filter_active(False)

async def _play_tts_chunk(self, chunk: np.ndarray):
    """Real Kokoro TTS speech — updates both filter systems."""
    self._aria_filter.feed_aria_audio(chunk)     # Layer 2 — update embedding
    self._gate0.feed_tts_spectrum(chunk)         # Layer 3 — update spectrum EMA
    self._stream.write(chunk.tobytes())

async def play_clip(self, clip_name: str):
    """
    All clips: breath, mm, laughs, chuckles, sighs,
               sure, got_it, all backchannel clips.
    Does NOT call feed_aria_audio() or feed_tts_spectrum().
    Does NOT change filter_active.
    """
    audio = self._load_clip(clip_name)
    self._stream.write(audio)

async def play_clip_looping(self, clip_name: str, stop_event: asyncio.Event):
    """Plays clip. Loops once if stop_event not set when clip ends."""
    await self.play_clip(clip_name)
    if not stop_event.is_set():
        await self.play_clip(clip_name)   # maximum one loop
```

**After PAUSED state resumes:**

AriaVoiceFilter's rolling buffer still has recent Aria audio. Gate 0's EMA still has recent TTS spectra. Both embeddings are still valid. `set_filter_active(True)` is called and both systems protect from the first chunk of resumed TTS output — no relearning delay.

---

## 7. Conversation Dynamics

### interrupt_router — All 7 Sub-Typesa

**File:** `brain/interrupt_router.py`

`turn_manager.py` calls only `await interrupt_router.route(classified_event)`. All routing logic lives here. Primary LLM (llama-3.1-8b-instant) handles actual responses — allam-2-7b only classified.

```
IGNORE
  gate3_done_event.set() → filler clip stops
  audio_player.resume()
  mic_stream.set_filter_active(True)
  if sentiment positive → POSITIVE_REACTION event for backchannel selector
  State → SPEAKING
  No LLM call

INJECT
  gate3_done_event.set() → filler clip stops
  audio_player.resume()
  mic_stream.set_filter_active(True)
  text_injector.inject(transcript, inject_point)
  inject_point: sentence_start | mid_sentence | after_yield | sentence_boundary
  AI naturally weaves content in at the correct point
  State → SOFT_INJECT briefly → SPEAKING
  No LLM call now — surfaces in next primary LLM turn

INTERRUPT:STOP
  gate3_done_event.set() → filler clip stops
  audio_player.flush() ← full discard
  Complete silence — no bridge clip, no response
  State → LISTENING
  No LLM call — user wants silence, give silence

INTERRUPT:PAUSE
  gate3_done_event.set() → filler clip stops
  TTS already paused from PENDING — do NOT flush
  State → PAUSED
  3s auto-resume timer starts
  No LLM call
  Exit routes:
    User says resume phrase → audio_player.resume(), SPEAKING
    User asks something new → audio_player.flush(), THINKING
    3s silence → auto-resume check → SPEAKING

INTERRUPT:CORRECTION
  gate3_done_event.set() → filler clip stops
  audio_player.flush()
  audio_player.play_clip("got_it.wav") ← bridge clip, ~500ms
  While got_it.wav plays: primary LLM (llama-3.1-8b) starts
  State → INTERRUPTED → THINKING
  LLM prompt includes: correction_context=True, what_ai_was_saying

INTERRUPT:NEW_QUESTION
  gate3_done_event.set() → filler clip stops
  audio_player.flush()
  audio_player.play_clip("sure.wav") ← bridge clip, ~300ms
  While sure.wav plays: primary LLM starts with new query
  State → INTERRUPTED → THINKING

INTERRUPT:SAME_TOPIC_REDIRECT
  gate3_done_event.set() → filler clip stops
  audio_player.flush()
  audio_player.play_clip("sure.wav") ← bridge clip
  Primary LLM starts with prior topic context active
  State → INTERRUPTED → THINKING

INTERRUPT:URGENCY
  gate3_done_event.set() → filler clip stops
  audio_player.flush() ← immediate, no bridge clip, zero delay
  State → INTERRUPTED → THINKING with urgency=True
  Primary LLM at maximum priority, no bridge

INTERRUPT:DONE_LISTENING
  gate3_done_event.set() → filler clip stops
  audio_player.stop at sentence boundary (no mid-word cut)
  State → LISTENING
  No LLM call — user has heard enough
```

**Bridge clips (sure.wav, got_it.wav) serve one purpose:** fill the ~100-200ms gap between `flush()` completing and the primary LLM's first token arriving. Without them there is an audible silence gap. With them the conversation feels continuous.

### PAUSED State — Full Logic

```
From PAUSED, user speaks:

  VAD detects speech
        │
        ├── Is it a resume phrase?
        │   ("go ahead", "continue", "okay", "resume",
        │    "keep going", "please continue", "go on")
        │   Detection: local string match, no API
        │     YES → audio_player.resume()
        │            mic_stream.set_filter_active(True)
        │            State → SPEAKING
        │            TTS continues from exact freeze point
        │            No LLM call
        │
        └── Is it something new?
            Detection: any speech that is not a resume phrase
              → audio_player.flush() — discard frozen TTS
              → Normal STT path → primary LLM → new turn
              → State → THINKING
              Note: NO gates involved — PAUSED state, gates are idle

From PAUSED, 3s silence:
  Before resuming: check VAD energy on last 200ms
  If user is speaking → delay 1s → check again (prevents race)
  If still silent → audio_player.resume()
                    mic_stream.set_filter_active(True)
                    State → SPEAKING
```

### Natural Yield Points

LLM outputs `[...]` every 2-3 sentences. `response_parser.py` fires `PAUSE_MARKER`. `audio_player.py` inserts 300ms silence (not via pause() — just empty audio frames).

During `[...]`:
- Nothing playing → `set_filter_active` already reflects this
- VAD fully active
- All gates IDLE (this is a handoff, not an interrupt)
- If user speaks → clean VAD → normal STT path → no gates

This is always the preferred path. System prompt instructs LLM to use it.

### Laughter and Emotional Clips

Tags `[laughs]`, `[chuckles]`, `[sighs]` in LLM output → `response_parser.py` fires `PLAY_CLIP` → `audio_player.play_clip()`. Never `_play_tts_chunk()`. Never updates filters.

---

## 8. Tool and Agent System

### How Tools Work

```
User: "What's the weather in London?"
  ↓
Primary LLM (llama-3.1-8b-instant):
  "[curious] Let me check. [...]
   <agent>{"action":"web_search","params":{"query":"London weather"}}</agent>"
  ↓
response_parser splits:
  "[curious] Let me check." → TTS → _play_tts_chunk() → updates filters
  [...]                     → 300ms silence, filter_active unaffected
  <agent>                   → AGENT_JSON_OUT event
  ↓
ToolRouter → WebSearchTool.execute() → async, non-blocking
  ↓
Result injected → next primary LLM call
  ↓
"[cheerful] It's 12 degrees in London, cloudy with light rain."
```

### Built-in Tools

| Tool | Key Params | Output |
|---|---|---|
| `web_search` | `query` | Result text → main LLM synthesizes spoken response |
| `article_fetch` | `url`, `instruction` | Spoken summary → TTS directly; full text → session cache |
| `memory_recall` | `query`, optional `memory_type` | Facts/episodic results → main LLM synthesizes |
| `session_cache_qa` | `cache_id`, `question` | Answer from cached content → TTS directly |

**New in v3.6:** `memory_recall` searches across the five-tier memory system (fact store, procedural, long-term ChromaDB). `session_cache_qa` answers follow-up questions about articles or search results from the current session using the RAM session cache — no re-fetch needed, ~250ms latency.

---

## 9. MCP Servers

```yaml
mcp_servers:
  github:
    command: ["npx", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "ghp_..."
  filesystem:
    command: ["npx", "@modelcontextprotocol/server-filesystem", "/path"]
  slack:
    command: ["npx", "@anthropic/mcp-server-slack"]
    env:
      SLACK_BOT_TOKEN: "xoxb-..."
```

Auto-registers at startup in `main.py`:

```python
for name, cfg in config.get("mcp_servers", {}).items():
    client = MCPClient(cfg["command"], name)
    await client.start()
    for tool_schema in await client.list_tools():
        tool_router.register_tool(MCPBridgeTool(client, tool_schema))
```

---

## 10. Configuration Reference

```yaml
persona:
  name: "Aria"
  system_prompt: |
    You are Aria, warm and capable.
    Keep responses under 3 sentences before a [...] yield point.
    Start every response with an emotion tag:
      [cheerful] [calm] [concerned] [excited] [empathetic]
    Use [...] every 2-3 sentences.
    Use [laughs] when genuinely funny.
    Use [chuckles] for mild amusement.
    Use [sighs] when thoughtful.
    One paralinguistic tag per response maximum.
    For tool tasks: <agent>{"action":"tool_name","params":{...}}</agent>
  voice: "af_heart"
  language: "en"

models:
  stt_primary: "whisper-large-v3-turbo"
  stt_fallback: "https://your-hf-space.hf.space/transcribe/monitor"
  llm_fast: "llama-3.1-8b-instant"          # primary LLM — all spoken responses
  llm_monitor: "allam-2-7b"                  # classifier only — never speaks
  llm_smart: "meta-llama/llama-4-scout-17b"
  llm_agentic: "moonshotai/kimi-k2-instruct"
  llm_deep: "qwen/qwen3-32b"
  llm_power: "openai/gpt-oss-120b"
  tts_model: "models/kokoro-v1.0.onnx"
  tts_voices: "models/voices-v1.0.bin"
  safety: "meta-llama/llama-guard-4-12b"

audio:
  sample_rate: 16000
  output_sample_rate: 24000
  chunk_ms: 32
  vad_speech_threshold: 0.65            # LISTENING state
  vad_interrupt_threshold: 0.4          # interrupt context (TTS paused, quieter)
  vad_silence_duration_ms: 500          # normal end-of-turn
  vad_interrupt_silence_ms: 300         # interrupt context — user finished speaking
  noise_floor_calibration_s: 0.5

echo_suppression:
  # Layer 1 — denoiser
  denoiser_model: "dns64"               # facebook denoiser model name

  # Layer 2 — AriaVoiceFilter
  aria_similarity_threshold: 0.75       # cosine similarity above = Aria echo

  # Layer 3 — Gate 0
  gate0_spectral_threshold: 0.92        # cosine similarity above = TTS echo
                                        # (raised 0.85→0.92: 257-bin FFT for precision)
  gate0_temporal_gate_ms: 400           # TTS played < N ms ago → echo
                                        # (raised 80→400: covers inter-sentence gaps)
  gate0_ema_alpha: 0.1                  # EMA weight for spectral fingerprint

  # Layer 4 — V5 Playback reference correlation
  playback_correlation_threshold: 0.45  # Peak cross-correlation above this = echo
                                        # Pure echo: 0.6–0.9, user+echo: 0.2–0.35
                                        # Works on 30ms chunks (unlike AriaFilter's 200ms)

  # Startup warmup
  warmup_text: "Hello, how are you doing today? I hope you're having a wonderful day."

speaking_monitor:
  enabled: true
  model: "llama-3.1-8b-instant"          # Gate 3 classifier (was allam-2-7b)

  # Gate 1 — Entry gate
  gate1_entry_multiplier: 4.0            # RMS must exceed noise_floor × this
                                         # before any accumulation starts

  # Gate 1 — Path A (short command words)
  gate1_path_a_min_duration_ms: 150
  gate1_path_a_energy_ratio: 13          # 13× noise floor (raised from 8×)

  # Gate 1 — Path B (normal speech)
  gate1_path_b_min_duration_ms: 750      # (raised from 600ms)
  gate1_path_b_energy_ratio: 6.0         # 6.0× noise floor (raised from 3×)

  # Gate 2
  gate2_filler_list:
    - "yeah" - "yep" - "okay" - "ok" - "right" - "sure"
    - "uh-huh" - "mm-hmm" - "mm" - "hmm" - "haha" - "lol"
    - "wow" - "oh" - "nice" - "cool" - "great" - "interesting"
    - "really" - "totally" - "exactly"

  # Gate 3
  gate3_timeout_s: 3.0                  # → default INTERRUPT:PAUSE on timeout
  gate3_audio_window_ms: 600
  gate3_context_words: 30
  gate3_reclassify_every_samples: 24000 # re-classify every 1.5s if user keeps speaking

  # PENDING behavior
  pre_pause_clip: "breath"              # "breath" or "mm" — appends .wav automatically
  pre_pause_enabled: true

  # PAUSED state
  paused_auto_resume_s: 3.0
  paused_resume_phrases:
    - "go ahead" - "continue" - "okay" - "resume"
    - "keep going" - "please continue" - "go on" - "yes go ahead"

backchannel:
  enabled: true
  min_pause_ms: 350
  min_gap_between_s: 8
  min_user_speech_s: 2
  clips_dir: "backchannel/clips/heart/"
  energy_threshold: 0.02
  emotional_clips:
    laughs: "laughs.wav"
    chuckles: "chuckles.wav"
    light_laugh: "light_laugh.wav"
    sighs: "sighs.wav"
  interrupt_bridge_clips:
    correction: "got_it.wav"
    new_question: "sure.wav"
    same_topic_redirect: "sure.wav"

memory:
  # Working memory (Tier 1)
  short_term_turns: 20
  compress_after_turns: 20

  # ChromaDB long-term (Tier 5)
  long_term_enabled: true
  long_term_db: "./data/memory.db"
  retrieval_top_k: 5
  embedding_model: "default"
  similarity_threshold: 0.65

  # SQLite stores (Tiers 3 & 4)
  fact_store_db: "data/memory/facts.db"     # Tier 3: permanent user facts
  procedural_db: "data/memory/procedural.db" # Tier 4: instructions & reminders

  # Session-start loader
  loader_max_tokens: 350             # Hard cap on memory block in system prompt
  relationship_recency_days: 14
  reminder_window_days: 7

  # Background LLM extraction
  background_enabled: true

agent:
  enabled: true
  max_concurrent_tools: 2
  tool_timeout_s: 30
  tools:
    - web_search
    - article_fetch
    - memory_recall                  # Search past sessions / user facts
    - session_cache_qa               # QA over cached content this session

safety:
  enabled: true
  check_input: true
  check_output: true

server:
  host: "0.0.0.0"
  port: 8000
  websocket_enabled: true
```

### Environment Variables

```bash
GROQ_API_KEY=gsk_...
TAVILY_API_KEY=tvly_...
HF_STT_URL=https://your-hf-space.hf.space
HF_API_KEY=hf_...
```

---

## 11. API and Integration

### WebSocket Events

```json
// From VoxCore:
{"type": "warmup_complete",
 "aria_filter_ready": true, "gate0_ready": true}

{"type": "pending",
 "urgency_hint": false, "timestamp": 1234567890}

{"type": "classified",
 "decision": "INTERRUPT", "interrupt_type": "NEW_QUESTION",
 "transcript": "what about solar panels",
 "confidence": 0.91, "latency_ms": 480}

{"type": "classified",
 "decision": "INJECT", "interrupt_type": null,
 "transcript": "oh and I'm vegetarian",
 "inject_point": "mid_sentence"}

{"type": "classified",
 "decision": "IGNORE", "interrupt_type": null,
 "transcript": "haha yeah", "sentiment": "positive"}

{"type": "filler_reaction",
 "word": "haha", "sentiment": "positive"}

{"type": "state", "old": "speaking", "new": "paused"}
{"type": "state", "old": "paused", "new": "speaking", "reason": "resume_phrase"}
{"type": "state", "old": "paused", "new": "thinking", "reason": "new_question"}

{"type": "filter_status",
 "active": true, "aria_ready": true, "gate0_ready": true}

// To VoxCore:
{"type": "force_resume"}
{"type": "force_stop"}
{"type": "inject", "content": "...", "priority": "high"}
```

---

## 12. Pipeline Walkthrough — Every Case

### Case 1: Normal Turn (LISTENING — gates idle)

```
Clean audio (denoiser, filter OFF) → Silero VAD
SPEECH_END after 500ms → Groq Whisper → TRANSCRIPT_READY
State → THINKING → PromptBuilder → primary LLM streams
response_parser:
  sentences → _play_tts_chunk() → speaker + feeds aria_filter + gate0
  [...] → 300ms silence (filter_active already True, nothing changes)
  [laughs] → play_clip() → speaker (no filter feed)
  <agent> → AGENT_JSON_OUT → ToolRouter async
PLAYBACK_DONE → State → LISTENING
audio_player signals: set_filter_active(False)
```

### Case 2: Echo Arrives at Mic (Aria Speaking, User Silent)

```
Aria speaking, echo hits mic

FILTERED CONSUMER PATH (VAD/STT):
  → denoiser: removes most of it (Layer 1)
  → AriaVoiceFilter: similarity > 0.75 → drop (Layer 2)
  If echo residual survives Layer 2:
    → Gate 0 temporal: TTS chunk played < 400ms ago → drop (Layer 3)
    OR
    → Gate 0 spectral: similarity to TTS EMA > 0.92 → drop (Layer 3)

RAW CONSUMER PATH (InterruptionDetector):
  → V5 playback reference correlation: mic chunk cross-correlated against
    recently played TTS audio buffer. Peak correlation > 0.45 → drop (Layer 4)
    (This is critical: AriaVoiceFilter needs 200ms chunks but raw consumer
     receives 30ms chunks — AriaFilter never fires on this path.
     V5 works on 30ms chunks and catches what AriaFilter cannot.)

Gate 1 never sees echo on either path.
User never gets falsely interrupted.
AI speaks without looping.
```

### Case 3: Short Command — "Stop!" (~200ms)

```
t=0      User: "stop!" — high energy
t=200ms  Gate 0: TTS played recently → temporal check drops?
         Actually no — the user's voice is high energy, different spectrum
         Gate 0 spectral: user's voice ≠ Aria's TTS spectrum → passes Gate 0
         V5 correlation: user's voice ≠ recently played TTS → passes
         Gate 1 Path A: duration=200ms, energy=20×noise > 13× → HIGH_ENERGY_BURST
         Skip Gate 2 (urgency)
t=203ms  PENDING:
           audio_player.pause() — TTS freezes
           set_filter_active(False) — Gate 0 + AriaFilter OFF
           breath.wav starts playing
           Gate 3 starts (urgency_hint=True)
t=450ms  Gate 3: "stop" → INTERRUPT:STOP
         _wait_for_user_to_finish: user silent → act immediately
t=453ms  gate3_done_event.set() → breath clip stops
         interrupt_router:STOP
           audio_player.flush() — discard TTS
           State → LISTENING
           Silence — no bridge clip, no response
```

### Case 4: Normal Sentence — "What's the weather in London?"

```
t=0      User starts speaking
t=0-750ms  Gate 0 checks each chunk:
           User voice spectrum ≠ Aria TTS spectrum → passes Gate 0
           V5 correlation: user voice ≠ recently played TTS → passes
t=750ms  Gate 1 Path B: sustained 750ms, energy > 6.0× noise → NORMAL_SPEECH
t=602ms  Gate 2: partial transcript not filler → passes
t=603ms  PENDING: TTS pauses, filter OFF, breath.wav starts
         Gate 3 starts on first 600ms audio
t=850ms  Gate 3 early read: "what's the weather" → looks like NEW_QUESTION
         _wait_for_user_to_finish: user still speaking "...in London?"
         → HOLD, keep accumulating
t=1200ms User finishes "...London?"
t=1500ms 300ms silence → _wait exits
         Gate 3 re-fires: full "what's the weather in London?" → NEW_QUESTION
t=1503ms gate3_done_event.set() → breath clip stops
         interrupt_router:NEW_QUESTION
           audio_player.flush()
           play_clip("sure.wav") bridge
           While sure.wav plays (~300ms): primary LLM starts
           State → INTERRUPTED → THINKING → SPEAKING
```

### Case 5: Long Explanation (User Speaks for 4+ Seconds)

```
t=0      User: "actually I was thinking about something different,
                I've been wondering about X, and what if..."
t=750ms  Path B → PENDING → TTS pauses → filler clip starts
t=1000ms Gate 3 early read: "actually I was thinking" → INJECT
         _wait: user still speaking → HOLD
t=2500ms Accumulated 1.5s → re-classifies:
         "actually thinking about something different, wondering about X"
         → escalates to NEW_QUESTION
         User still speaking → HOLD
t=4200ms User finishes
t=4500ms 300ms silence → _wait exits → acts on NEW_QUESTION
         TTS was paused for 3.9 seconds — user never got re-interrupted
         interrupt_router: flush, "sure.wav", primary LLM, answer X
```

### Case 6: Filler — "Haha yeah" (Normal Volume)

```
t=0      User: "haha yeah" — ~400ms, normal energy
t=400ms  Path A check: energy=2×noise → not 13× → not Path A
t=750ms  Path B check: duration only 400ms < 750ms → DROP
         Audio discarded — AI never paused, never interrupted
         (Note: very loud "HAHA!" at 13× energy might hit Path A
          → Gate 3 with urgency_hint → IGNORE → resume, no interruption)
```

### Case 7: "Hold On" → PAUSED State

```
t=0      User: "hold on" — ~500ms, normal energy
t=600ms  Gate 0 passes (user's voice ≠ Aria's TTS spectrum)
         Gate 1 Path B passes
t=602ms  Gate 2: "hold on" not in filler list → passes
t=603ms  PENDING: TTS pauses, filter OFF, breath.wav starts
t=1000ms Gate 3: INTERRUPT:PAUSE
         _wait: user silent → act
t=1003ms gate3_done_event.set() → breath stops
         interrupt_router:PAUSE
           TTS already paused — do NOT flush (RAM preserved)
           State → PAUSED
           3s timer starts

         Option A — User says "okay go ahead" at t=2000ms:
           Resume phrase detected (local, no API)
           audio_player.resume() — TTS from exact freeze point
           set_filter_active(True)
           State → SPEAKING

         Option B — User asks new question at t=2000ms:
           VAD detects speech, not a resume phrase
           audio_player.flush() — discard frozen TTS
           Normal STT path → primary LLM
           State → THINKING (no gates — PAUSED state)

         Option C — 3s silence:
           VAD check: user not speaking?
           audio_player.resume()
           set_filter_active(True)
           State → SPEAKING
```

### Case 8: Persona Voice Switch

```
User triggers switch to "Max" (am_adam voice)
  → aria_filter.reset() — embedding cleared, is_ready=False
  → gate0 reset (new instance or reset() method)
  → set_filter_active(False) — protection off temporarily

  await startup_warmup(aria_filter, gate0, tts_client, new_config)
  → synthesizes 3s of Max's voice silently via Kokoro
  → feeds to aria_filter (builds Max's embedding)
  → feeds to gate0 (builds Max's spectral fingerprint)
  → both is_ready = True in ~200ms

  set_filter_active(True) on first Max TTS chunk
  Full protection for Max's voice from his first word
```

### Latency Summary

```
Startup:
  Kokoro model load                     ~1-2s (existing, unchanged)
  resemblyzer VoiceEncoder load         ~1s (new, one-time)
  startup_warmup() synthesis + feeding  ~200ms (new)

Per-chunk audio cleaning:
  denoiser                              ~5ms  (always)
  AriaVoiceFilter check                 ~10ms (filter_active only)
  Gate 0 spectral check                 ~0.5ms (filter_active only)
  V5 playback ref correlation           ~1ms  (raw consumer, SPEAKING only)

Gate timings from speech start:
  Path A fires at                       ~150-200ms
  Path B fires at                       ~600ms
  Gate 2 check                          ~2ms
  PENDING + pause + filter off + clip   ~3ms from Gate 2 pass

Gate 3 background:
  whisper-turbo (Groq)                  ~250ms
  allam-2-7b (Groq)                     ~150ms
  CLASSIFIED fires                      ~400-600ms from PENDING

User perception:
  "AI heard me"                         ~3ms from PENDING
  "AI responded smartly"                ~400-600ms (Gate 3)
  "AI said something"                   ~100-300ms after CLASSIFIED
                                        (bridge clip + LLM first token)

Normal turn end-to-end:
  VAD end-of-turn                       ~500ms
  STT                                   ~200ms
  Primary LLM first token               ~100-200ms
  First sentence TTS                    ~50-100ms
  Playback start                        ~10ms
  Total                                 ~860-1010ms
```

---

## 13. Changes Summary for Coding Agent

This section is the complete implementation reference. Read all of it before modifying any file.

### Critical Rules — Read First

```
allam-2-7b (llm_monitor):
  DOES: receive transcript + context → output DECISION + TYPE (two words)
  DOES NOT: generate speech, play audio, call primary LLM, access memory

Primary LLM (llama-3.1-8b-instant, llm_fast):
  DOES: generate all spoken responses for normal turns AND interrupt responses
  DOES NOT: classify interrupts (that is allam's job)

Bridge clips (sure.wav, got_it.wav):
  PURPOSE: fill 100-200ms gap between flush() and primary LLM first token
  RULE: play_clip() only — never _play_tts_chunk()
  RULE: never call feed_aria_audio() or feed_tts_spectrum()

Pre-pause clips (breath.wav, mm.wav):
  PURPOSE: instant acoustic response at PENDING while Gate 3 runs
  RULE: play_clip_looping() with gate3_done_event
  RULE: never call feed_aria_audio() or feed_tts_spectrum()

Emotional clips (laughs, chuckles, sighs) and backchannel clips:
  RULE: play_clip() only — same exclusion as above

_play_tts_chunk() is the ONLY function that calls:
  aria_filter.feed_aria_audio(chunk)
  gate0.feed_tts_spectrum(chunk)
  mic_stream.feed_playback_reference(chunk)   # V5 — feeds playback ref buffer

set_filter_active(True) is called from:
  audio_player.resume()
  audio_player._play_tts_chunk() (ensures it is on during playback)

set_filter_active(False) is called from:
  audio_player.pause()
  audio_player.flush()
  audio_player.play_clip() does NOT change filter_active

mic_stream.clear_playback_reference() is called from:
  audio_player.pause()    # TTS paused — clear stale reference
  audio_player.flush()    # TTS discarded — clear reference
  audio_player._playback_loop finally  # playback ended — clear reference
```

### New Files to Create

| File | Purpose |
|---|---|
| `input/aria_voice_filter.py` | resemblyzer speaker embedding — pre-warmed at startup, updated live |
| `input/gate0_echo_check.py` | Spectral + temporal echo gate — pre-warmed at startup, updated live |
| `brain/interrupt_router.py` | All 7 interrupt_type routing actions — single source of truth |
| `output/pre_pause_clips/breath.wav` | Generated by `backchannel.generator` |
| `output/pre_pause_clips/mm.wav` | Generated by `backchannel.generator` |
| `memory/fact_store.py` | SQLite-backed permanent user facts (name, city, prefs) |
| `memory/procedural.py` | SQLite-backed standing instructions & reminders |
| `memory/session_cache.py` | RAM dict for full tool results (articles, search) |
| `memory/background_llm.py` | Fire-and-forget extraction from turns into memory |
| `memory/loader.py` | Session-start memory block assembly (350 token cap) |
| `memory/retriever.py` | Parallel recall across all memory tiers |
| `agent/tools/memory_recall.py` | Tool wrapper over retriever for on-demand recall |
| `agent/tools/session_cache_qa.py` | QA over cached content via llm_fast |

### Files to Update

| File | Key Changes |
|---|---|
| `requirements.txt` | Add: `resemblyzer`, `denoiser`, `torch`, `torchaudio` |
| `core/event_bus.py` | Add events: `PENDING`, `CLASSIFIED`, `FILLER_REACTION`, `FILTER_ACTIVE_CHANGE`, `WARMUP_COMPLETE` |
| `core/session.py` | Add to TurnState enum: `PAUSED`, `SOFT_INJECT` |
| `core/turn_manager.py` | PAUSED state handler + 3s auto-resume with VAD check + resume phrase detection + interrupt_type routing |
| `input/mic_stream.py` | Add denoiser pipeline; add Gate 0 call after AriaVoiceFilter; add `set_filter_active()`; add V5 playback reference correlation (`feed_playback_reference()`, `clear_playback_reference()`, `_check_playback_correlation()`) |
| `input/interruption_detector.py` | Two-path Gate 1: `HIGH_ENERGY_BURST` (150ms, 13×) and `NORMAL_SPEECH` (750ms, 6.0×); entry gate at 4.0× (configurable) |
| `input/speaking_monitor.py` | Full pause (not duck) on PENDING; `play_clip_looping` with `gate3_done_event`; Gate 3 as `asyncio.create_task`; `_wait_for_user_to_finish` with 300ms interrupt silence; 1.5s re-classify; 3s timeout → PAUSE |
| `output/audio_player.py` | `pause()`/`resume()` with position tracking; `flush()`; separate `_play_tts_chunk()` vs `play_clip()` vs `play_clip_looping()`; `set_filter_active()` calls; `feed_aria_audio()` + `feed_tts_spectrum()` + `feed_playback_reference()` in `_play_tts_chunk()` only; `clear_playback_reference()` in pause/flush/playback-end |
| `output/tts_client.py` | Add `synthesize_silent(text, voice) → np.ndarray` for startup warmup; mid-stream pause/resume support |
| `input/text_injector.py` | Add `inject_point` parameter |
| `brain/response_parser.py` | Track current sentence position → `inject_point` value; detect `sentence_boundary` when buffer empty; strip `[ref:sc_...]` from TTS output while keeping in stored turn |
| `brain/router.py` | Add `background_llm.create_task()` after each turn; pass session_cache to article_fetch |
| `brain/prompt_builder.py` | Remove per-turn long_term.recall(); add `set_memory_block()` for session-start injection |
| `backchannel/generator.py` | Add `breath.wav` and `mm.wav` generation pass |
| `backchannel/selector.py` | Handle `FILLER_REACTION` event sentiment |
| `config.yaml` | Add `echo_suppression` block (incl. V5 `playback_correlation_threshold`); update `speaking_monitor` with gate params and timeout; add memory block (five tiers); add `memory_recall` + `session_cache_qa` to agent tools |
| `main.py` | Add `startup_warmup()` function; call it after TTS + VoiceEncoder load; wire `gate3_done_event` as shared `asyncio.Event`; wire `FactStore`, `ProceduralStore`, `BackgroundLLM`, `SessionCache`, `MemoryLoader`, `MemoryRetriever` |

### Shared Objects — Wire in main.py

These objects are created once and passed by reference:

```python
# Create once
aria_filter = AriaVoiceFilter(
    similarity_threshold=config["echo_suppression"]["aria_similarity_threshold"]
)
gate0 = Gate0EchoCheck(
    spectral_threshold=config["echo_suppression"]["gate0_spectral_threshold"],
    temporal_gate_ms=config["echo_suppression"]["gate0_temporal_gate_ms"],
    ema_alpha=config["echo_suppression"]["gate0_ema_alpha"]
)
gate3_done_event = asyncio.Event()

# Warmup both before conversation starts
await startup_warmup(aria_filter, gate0, tts_client, config)

# Pass to modules that need them
mic_stream = MicStream(config, event_bus, aria_filter, gate0)
audio_player = AudioPlayer(config, mic_stream, aria_filter, gate0, gate3_done_event)
interrupt_router = InterruptRouter(config, audio_player, mic_stream, llm_client, event_bus)
speaking_monitor = SpeakingMonitor(
    config, event_bus, audio_player, mic_stream,
    interrupt_router, gate3_done_event
)
```

### Build Order (Strict)

```
Step 1   requirements.txt
         pip install resemblyzer denoiser torch torchaudio

Step 2   core/event_bus.py
         New events required by all downstream modules

Step 3   core/session.py
         New TurnStates required by turn_manager

Step 4   input/aria_voice_filter.py
         Standalone — no VoxCore deps

Step 5   input/gate0_echo_check.py
         Standalone — numpy only

Step 6   output/tts_client.py
         Add synthesize_silent() — needed by startup_warmup in main.py
         Add pause/resume support

Step 7   input/mic_stream.py
         Uses aria_filter and gate0
         Exposes set_filter_active()

Step 8   output/audio_player.py
         pause() / resume() / flush() / play_clip() / play_clip_looping()
         _play_tts_chunk() calls feed_aria_audio() + feed_tts_spectrum()
         set_filter_active() calls mic_stream
         Receives gate3_done_event

Step 9   input/interruption_detector.py
         Two-path Gate 1

Step 10  input/filler_detector.py
         Verify FILLER_REACTION event fires correctly

Step 11  input/speaking_monitor.py
         Full PENDING logic
         _wait_for_user_to_finish
         Gate 3 task + timeout
         Uses gate3_done_event

Step 12  brain/interrupt_router.py
         All 7 sub-types
         Bridge clips for CORRECTION/NEW_QUESTION/SAME_TOPIC_REDIRECT
         Calls audio_player, mic_stream, llm_client

Step 13  core/turn_manager.py
         PAUSED state
         Auto-resume with VAD check
         Routes to interrupt_router

Step 14  input/text_injector.py
         inject_point parameter

Step 15  brain/response_parser.py
         inject_point tracking
         sentence_boundary detection

Step 16  backchannel/generator.py
         Add breath.wav and mm.wav

Step 17  backchannel/selector.py
         FILLER_REACTION handler

Step 18  config.yaml
         All new blocks from Section 10

Step 19  main.py
         startup_warmup() function
         Shared object wiring
         startup_warmup() call in boot sequence
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
# New: resemblyzer denoiser torch torchaudio

# 2. Environment
cp .env.example .env
# Fill: GROQ_API_KEY, TAVILY_API_KEY, HF_STT_URL, HF_API_KEY

# 3. TTS model files
# Place in models/:
#   kokoro-v1.0.onnx
#   voices-v1.0.bin

# 4. Generate audio clips (one-time)
python -m backchannel.generator
# Generates: all backchannel clips, emotional clips, breath.wav, mm.wav

# 5. Run
python main.py

# Boot sequence:
#   Kokoro-ONNX loads
#   resemblyzer VoiceEncoder loads
#   startup_warmup() runs (~200ms):
#     synthesizes 3s of Aria silently
#     feeds to AriaVoiceFilter → embedding ready
#     feeds to Gate0 → spectral fingerprint ready
#   [VoxCore Warmup] Echo protection fully ready. Starting conversation.
#   First word is spoken → full protection active from frame one
```

---

*VoxCore System Guide v3 — Voice Pipeline Focus — February 2026*