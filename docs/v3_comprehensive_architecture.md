# VoxCore V3 Comprehensive Architecture & Documentation

## Overview
VoxCore is a low-latency, full-duplex conversational voice AI system. The V3 architecture focuses on a highly decoupled, event-driven design that mimics natural human turn-taking, including the ability to handle interruptions, backchanneling ("uh-huh"), and seamless streaming responses.

This document maps out the entire project structure, the responsibilities of each module, and provides a deep dive into the audio and interruption pipelines.

---

## 1. Project Structure & Core Modules

The project is divided into distinct functional directories. All communication between modules happens via the centralized `EventBus`.

### `main.py`
The entry point of the application. It loads configurations (`config.yaml`), initializes all modules in the correct order, wires them to the `EventBus`, and starts the async event loop.

### `core/` (The Backbone)
Handles state management and communication.
- **`event_bus.py`**: A centralized publish/subscribe system. Modules do not call each other directly; they publish events (e.g., `SPEECH_START`, `LLM_TOKEN`, `TTS_READY`) and subscribe to events they care about.
- **`session.py`**: Holds the current `TurnState` (e.g., `LISTENING`, `THINKING`, `SPEAKING`, `PENDING`). All modules behave differently based on this global state.
- **`turn_manager.py`**: The state machine. It listens to events (like VAD detecting speech, or TTS finishing) and transitions the `Session` state accordingly.

### `input/` (Hearing & Listening)
Handles microphone capture, voice activity, and converting speech to text.
- **`mic_stream.py`**: Captures continuous 16kHz audio. Runs echo filters (`AriaVoiceFilter` & `Gate0EchoCheck`) and distributes clean audio chunks to consumers (VAD, InterruptionDetector).
- **`vad.py`**: Voice Activity Detection. Monitors audio during `LISTENING` state. Fires `SPEECH_START` and `SPEECH_END`.
- **`stt.py`**: Speech-to-Text. Consumes audio during active speech and uses Deepgram for fast transcription.
- **`interruption_detector.py`**: **Gate 1** of the interruption system. Monitors audio specifically during the `SPEAKING` state to detect if the user talks over the AI.
- **`speaking_monitor.py`**: **Gates 2 & 3** of the interruption system. When Gate 1 triggers, this module analyzes the user's transcript using a fast LLM to decide if it's an `IGNORE`, `INJECT` (reaction), or `INTERRUPT`.
- **`filler_detector.py`**: Used by the Speaking Monitor to detect if the user just said a filler word (e.g., "uh-huh").
- **`aria_voice_filter.py` & `gate0_echo_check.py`**: The upstream echo suppression filters that reject the AI's own voice from the microphone feed.

### `brain/` (Thinking & Routing)
Handles the LLM responses, prompts, and deciding what to do with user input.
- **`router.py`**: The main brain. Receives complete transcripts from STT and routes them to the correct LLM.
- **`llm_client.py`**: Streams responses from the primary LLM (Groq/Llama-3). Fires tokens individually to the EventBus.
- **`prompt_builder.py`**: Constructs the system prompt dynamically based on short-term memory and tools.
- **`emotion_tagger.py`**: Analyzes the LLM response to determine the emotional tone for the TTS engine.
- **`interrupt_router.py`**: Specifically handles how the AI responds *immediately* after being interrupted (e.g., apologizing, or answering a sudden question).

### `output/` (Speaking & Playing)
Handles converting text to speech and playing audio.
- **`tts_client.py`**: Converts incoming LLM tokens into audio streams using Kokoro-ONNX.
- **`audio_player.py`**: Manages the sounddevice output stream. Plays TTS audio, handles pausing/resuming, and plays short MP3/WAV clips.
- **`voice_profile.py`**: Manages the voice ID and speeds for the TTS.

### `agent/` (Tools & Slower Actions)
- **`tool_router.py`**: Detects if the user's request requires an external tool (e.g., fetching calendar, web search).
- **`slow_llm.py`**: Handles tasks that require deep thinking but are too slow for the main conversational loop.
- **`text_out.py`**: Interacts with the user via CLI instead of Voice.

### `memory/` & `backchannel/`
- **`memory/short_term.py` & `long_term.py`**: Manages conversational history and vector databases for recall.
- **`backchannel/cue_detector.py` & `selector.py`**: Analyzes user speech in real-time to play affirmative sounds ("hmm", "yeah") *while* the user is still talking, creating a very natural conversational feel.

---

## 2. The Core State Machine (`core/turn_manager.py`)

The system revolves around `TurnState`. Understanding this is critical to understanding how modules behave.

1. **`LISTENING`**: Default state. Microphone is capturing, VAD is analyzing for user speech.
2. **`USER_SPEAKING`**: VAD detected speech. STT is actively transcribing. Backchannel may occasionally fire.
3. **`THINKING`**: User finished speaking. LLM is generating the response.
4. **`SPEAKING`**: AI is talking (TTS is playing audio). Interruption detection is active.
5. **`PENDING`**: A unique state triggered during an interruption attempt. The AI instantly pauses its speech to verify if the user truly wanted to interrupt, or just sneezed/said "uh-huh".
6. **`INTERRUPTED`**: The AI officially stops its current thought and yields the floor back to the user or processes the new input.

---

## 3. Deep Dive: The V3 Audio & Interruption Pipeline

A critical feature of VoxCore is full-duplex capability: the user can interrupt the AI at any time. Because VoxCore does not use hardware Acoustic Echo Cancellation (AEC), the mic picks up the AI's own voice from the speakers. The V3 pipeline handles this via a multi-layered filtering and gating system.

### A. Echo Suppression (Mic Stream)
When the AI is in `SPEAKING` state, `MicStream` activates two echo filters before passing chunks to any modules:

1. **`Gate0EchoCheck` (Spectral Gate):** Compares the frequency spectrum of the mic audio to the known spectrum of the recent TTS audio. If it's a >85% match, the chunk is dropped. *Note: As of the recent fix, this allows mixed audio (double-talk) to pass.*
2. **`AriaVoiceFilter` (Identity Gate):** Compares the mic audio against an embedding of the AI's standard voice. If it sounds like the AI (>75% similarity), the chunk is dropped.

### B. The Three-Gate Interruption System

If user speech makes it past the echo filters during the `SPEAKING` state, it hits the Interruption Pipeline.

#### **Gate 1: Energy & Duration (`interruption_detector.py`)**
This module constantly analyzes the mic feed while the AI plays audio. It requires sustained, energetic speech to trigger, which prevents false positives from background noise.

- **Path A (The Burst):** Wait for 150ms of audio that is 8x louder than the noise floor (e.g., slamming a door, or yelling "STOP!").
- **Path B (Normal Speech):** Wait for 600ms of audio that is 2.5x louder than the noise floor (e.g., saying "Actually, can you tell me...").

Once either path is met, it publishes `GATE1_PASSED`.

#### **Gate 2 & 3: Semantic Verification (`speaking_monitor.py`)**
Upon receiving `GATE1_PASSED`, the Speaking Monitor takes over:
1. **Instant Pause:** It immediately forces `audio_player` to pause the AI's speech and puts the system in the `PENDING` state.
2. **Filler Acknowledgment:** It plays a quick "mm-hmm" audio clip so the user knows they were heard.
3. **Transcription:** It takes the last 2 seconds of audio and sends it to a fast transcription endpoint.
4. **Gate 2 (Filler / False Alarm):** It checks the transcript. If the user just said "yeah", "uh-huh" or "right" (Filler words), or if it was just echo, it instantly resumes the AI's speech.
5. **Gate 3 (LLM Intent):** It sends the transcript to a fast LLM (`llama-3.1-8b-instant`) to classify the intent:
    - **`IGNORE`**: Reflexive sounds. -> Resumes AI speech.
    - **`INJECT`**: Meaningful reaction ("Oh wow!"). -> Logs it to memory, but resumes AI speech.
    - **`INTERRUPT`**: Questions, commands, topic changes. -> Fully stops the AI, transitions to `INTERRUPTED`, and routes the prompt to the brain.

This system ensures the AI stops instantly when it hears a noise, but gracefully resumes if it realizes the user wasn't actually trying to take the floor.
