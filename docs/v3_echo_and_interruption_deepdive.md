# VoxCore V3 Audio & Interruption Architecture

This document maps out the current state of the audio pipeline in the VoxCore project, specifically focusing on how microphone audio is processed, how echo is suppressed, and how user interruptions are detected when the AI is speaking.

## 1. Core Audio Capture (`input/mic_stream.py`)

The `MicStream` class is the foundation of the audio pipeline. It captures audio from the microphone at 16kHz, mono, and chunks it into 30ms segments (`chunk_ms: 30`).

### Audio Fan-Out (Consumers)
The mic stream does NOT process the audio itself (other than formatting it to `float32`). Instead, it pushes every 30ms chunk to multiple independent `asyncio.Queue` consumers. 

Current consumers:
1. **`VADProcessor`** (Active during LISTENING window)
2. **`CueDetector`** (Backchannel phrase boundary detection)
3. **`InterruptionDetector`** (Active during SPEAKING window)

### V3 Echo Suppression (Upstream)
Before a chunk is sent to the consumer queues, it passes through two echo-filtering layers IF `_filter_active` is true (meaning the TTS is actively playing real speech).

- **Layer 2: `AriaVoiceFilter`**: Checks if the audio sounds like the AI's specific speaker identity (Resemblyzer cosine similarity > 0.75).
- **Layer 3: `Gate0EchoCheck`**: Checks if the audio spectrum matches recently played TTS audio (spectral cosine similarity > 0.85).

*If either filter detects echo, the chunk is DROPPED and never reaches the consumers.*

---

## 2. Interruption Detection Flow

When the AI is speaking (TurnState = `SPEAKING`), the user should be able to interrupt. This uses a three-gate system to prevent false positives from throat clears, noises, or unsuppressed echo.

### Gate 1: Duration & Energy (`input/interruption_detector.py`)

The `InterruptionDetector` consumes chunks from the `MicStream` but is only active when `session.state == TurnState.SPEAKING` (or `PENDING`).

**The Logic:**
1. **Energy Gate**: The RMS energy of the chunk must exceed `ambient_gate` (which is `noise_floor * 2.5`). 
2. **VAD Gate**: It runs its own internal instance of the Silero VAD model. The speech probability must be `> 0.60`.
3. **Timer**: If both pass, a timer (`duration_ms`) starts accumulating.
4. **Two-Path Trigger**:
    - **Path A (Burst)**: If `duration_ms >= 150` AND `RMS >= noise_floor * 8`, it triggers instantly as a loud command.
    - **Path B (Normal)**: If `duration_ms >= 600` AND `RMS >= noise_floor * 3`, it triggers as normal sustained speech.

When either path passes, it fires a `GATE1_PASSED` event to the EventBus. It then sleeps for 0.5s to prevent immediate re-triggering.

### Gate 2 & 3: Semantic Verification (`input/speaking_monitor.py`)

The `SpeakingMonitor` listens for `GATE1_PASSED`. When received:

1. **Instant Pause**: It immediately pauses the `AudioPlayer` and enters the `PENDING` state. It also plays a short "mm_hmm" filler clip to acknowledge the user.
2. **Audio Window Gathering**: It pulls the last 2000ms of audio from its own rolling buffer.
3. **Gate 3a (Fast STT)**: It sends that 2000ms window to a fast Whisper endpoint (`/transcribe/monitor`).
4. **Keyword Checks**:
    - If the transcript contains a STOP word ("stop", "wait", etc.), it instantly routes `INTERRUPT` -> `STOP`.
    - If it contains an ATTENTION word ("hey", "can you", etc.), it instantly routes `INTERRUPT`.
5. **Echo Check**: Checks if >60% of the transcript words overlap with what the AI was just saying. If so -> `RESUME` (false alarm).
6. **Gate 2 (Filler Check)**: (Skipped for Path A). Uses `FillerDetector`. If the user just said "uh-huh" or "yeah", it fires a `POSITIVE_REACTION` and `RESUME`s the audio.
7. **Gate 3b (LLM Semantic Check)**: If none of the above caught it, the transcript is sent to a fast LLM (`llama-3.1-8b-instant`). The LLM decides:
    - `IGNORE` (meaningless sounds) -> `RESUME`
    - `INJECT` (meaningful reaction, but don't stop) -> `RESUME`
    - `INTERRUPT` (question/command) -> `INTERRUPT`

---

## 3. The Root Cause: Why Interruptions Are Failing

Based on the code review, there is a **critical bug in `Gate0EchoCheck`** that is completely muting the microphone while the AI is speaking.

In `input/gate0_echo_check.py`:
```python
# Check 1: Temporal gate
ms_since_tts = (time.time() - self._last_tts_chunk_time) * 1000
if ms_since_tts < self._temporal_gate_ms:
    return True # BLOCKED (echo)
```
Where `_temporal_gate_ms` is `80`.

**The Problem:**
When the `AudioPlayer` is playing TTS, it continuously feeds 30ms chunks to `Gate0` (via `audio_player._play_tts_chunk` -> `mic_stream.feed_tts_spectrum`). Because chunks are fed back-to-back while speech is happening, `ms_since_tts` will *always* be less than `80ms`. 

Therefore, **`Gate0`'s temporal check blocks 100% of all microphone audio while the AI is speaking.** 

Since `Gate0` runs in `mic_stream.py` *before* the audio reaches the `InterruptionDetector`, the `InterruptionDetector` receives absolute silence while the AI speaks. It can never trigger an interruption because it hears nothing.

### Proposed Fix: 
The temporal gate in `Gate0` was designed for systems where TTS is played in discrete bursts, not continuously streamed chunk-by-chunk. To fix the interruption system:
1. We must **remove or significantly alter the temporal check** in `Gate0EchoCheck` so it relies on the spectral similarity check instead, allowing user voice to pass through even if TTS is currently playing.
### Implemented Fix
The problem was resolved by fundamentally changing `Gate0EchoCheck`'s behavior:
1. **Removed the temporal gate:** The code that unconditionally blocked all audio for 80ms `ms_since_tts < 80` was deleted.
2. **Relies purely on the spectral check:** `Gate0` now compares the *frequency spectrum* of the mic input against the known spectrum of the TTS output. If the mic picks up double-talk (AI + User), the spectrum will be a mix, and if it exceeds the `0.85` similarity threshold, it drops it. 
3. **Threshold tuning:** In `InterruptionDetector`, the default energy multiplier for Path B (normal speech) was lowered from `3.0` to `2.5` to make it easier for average-volume speech to trigger an interrupt while the TTS is playing.

With these changes, the user should be able to interrupt the AI normally. The V3 filters will reject pure echo but allow mixed audio (double-talk) through to the `InterruptionDetector`.
