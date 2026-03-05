# VoxCore Technical Documentation

## Overview
VoxCore is a voice-first, full-duplex conversational AI system capable of running seamlessly via Groq's APIs. This document is a technical deep dive into the architecture, component logic, latency structure, state logic, and the overall system design. 

For the general vision, capabilities, install instructions, and models used, please see `README.md`. 

---

## 🏗️ System Architecture & Data Flow

### 1. Data Flow

**Inputs:**
- 🎙 **Microphone:** Continuous 16kHz mono PCM stream.
- 📝 **Text-In:** Injected text messages representing context, tool results, notifications, or RAG inputs.
- 📄 **System Prompt:** Sets the persona at initialization.

**Internal Pipeline:**
1. **Audio Input Layer:** Silero VAD runs continuously detecting voice activity.
2. **Speech-to-Text (STT):** Audio chunk delivered to Groq's `whisper-large-v3-turbo` model. Backed up natively by a local huggingface `faster-whisper`.
3. **Brain & Logic Structure:**
   - **Fast Brain (`llama-3.1-8b-instant`)**: Routes conversational text, issues agent instructions (`<agent>` tags).
   - **Simulated Backchannel**: Parallel execution loop scanning phrase boundaries to inject natural voice breaks (e.g. "uh-huh").
   - **Text Injection**: Results or logs inserted directly into the context buffer of the LLM. 
4. **Output Generation:** Structured agent requests stream to routing mechanics, conversational answers stream directly to TTS.

**Outputs:**
- 🔊 **Voice-Out:** Streaming conversational audio through Groq Orpheus API sentence-by-sentence.
- 📤 **Text-Out:** Emitted valid JSON logs or parsed function execution calls.

---

## ⚙️ Core Turn State Machine

VoxCore continuously operates in a 6-state asynchronous loop responding to VAD triggers and LLM inference.
- **`LISTENING`**: VAD active. Active buffer collecting audio. **Exit Trigger:** 400ms of VAD silence.
- **`THINKING`**: VAD loop completes. LLM context processing begins. Text starts streaming. **Exit Trigger:** First sentence returned from the Fast Brain LLM.
- **`SPEAKING`**: Audio output streams to speaker. VAD continues running in the background. Five-layer echo suppression active (denoiser, AriaVoiceFilter, Gate 0 spectral, V5 playback reference correlation). Multi-gate interrupt pipeline (Gate 1 → Gate 2 → Gate 3) monitors for user interrupts. **Exit Trigger:** LLM playback finishes `OR` user interrupts (speaks over audio).
- **`PAUSED`**: TTS frozen in RAM at exact frame boundary. Resumable without LLM call if user says "go ahead". 3s auto-resume timer. **Exit Trigger:** Resume phrase detected `OR` new user question `OR` 3s silence.
- **`SOFT_INJECT`**: Brief state during text injection while TTS continues. AI weaves injected content at correct sentence position. **Exit Trigger:** Immediately transitions to SPEAKING.
- **`INTERRUPTED`**: Audio loop killed. Filler clip plays as bridge. User input directed back to the STT model. **Exit Trigger:** STT finishes transcript → primary LLM starts → `THINKING`.

---

## 🧠 Brains & Logic Flow

VoxCore employs multiple LLMs handling complex and simplistic routines synchronously avoiding I/O blocking.
- `Fast Brain` (`llama-3.1-8b-instant`): Responsible strictly for realtime interactive conversations. Used primarily due to the minimal sub-200ms Time To First Token (TTFT). Fast Brain identifies actions and outputs `<agent>{"action": "web_search", ...}</agent>`.
- `Smart Brain` (`llama-4-scout-17b`): Routed internally for multi-step reasoning capabilities.
- `Agentic Brain` (`kimi-k2-instruct`): Leveraged specifically for executing tools async. Tool execution loops back directly into the Context Buffer logic via Text-In injections.
- `Safety Brain` (`llama-guard-4-12b`): Realtime passive safety checking to override inappropriate inferences. 

---

## 🗂 File & Directory Deep-Dive Analysis

During the system inspection phase, the internal files mapped successfully matched the logical structure.

### `core/` (The Brain Stem)
- **`session.py`**: The god-object. Master state dictating context rules. Last 20 terms and Text-in priorities.
- **`event_bus.py`**: A light-weight pub/sub engine executing zero-tight coupling commands (`SPEECH_START`, `INTERRUPT_DETECTED`).
- **`turn_manager.py`**: Holds state flow instructions managing transitions between the 4 native states listed above.

### `input/` (Ears)
- **`mic_stream.py`**: Runs SoundDevice bindings natively managing constant PCM data. Includes five-layer echo suppression: denoiser (always on), AriaVoiceFilter (speaker identity), Gate 0 (spectral + temporal), V5 playback reference correlation (cross-correlates mic against recent TTS audio). Maintains filtered consumers (clean audio for VAD/STT) and raw consumers (for InterruptionDetector, with `check_echo()` providing V5 protection).
- **`aria_voice_filter.py`**: Resemblyzer-based speaker embedding — identifies and rejects mic chunks that sound like Aria's voice. Pre-warmed at startup.
- **`gate0_echo_check.py`**: Spectral cosine similarity + temporal proximity check against rolling TTS spectrum EMA.
- **`vad.py`**: Pysilero-vad wrappers checking for the 400ms buffer silence gap.
- **`stt.py`**: Converts soundwaves to Groq STT APIs. Handles the fallback execution loops.
- **`interruption_detector.py`**: Two-path Gate 1 interrupt detection — Path A (short energy bursts, 150ms/13×) and Path B (sustained speech, 750ms/6.0×). Entry gate at noise_floor × 4.0×.
- **`text_injector.py`**: Appends async commands to the Session.

### `backchannel/` (Natural Conversation)
- **`cue_detector.py`**: Uses `librosa` parallel processing to evaluate drop-off RMS volumes. Injects "uh-huh" logic.
- **`selector.py`**: Random weighted event selector checking if conditions (e.g., >8s gap) are met.
- **`generator.py`**: Configuration script run prior to bot-execution generating offline TTS WAV clips.

### `brain/` (The Mind)
- **`llm_client.py`**: Custom AsyncGroq API implementation parsing `yield` tokens.
- **`prompt_builder.py`**: Prepares prompt structure (Persona + Memory Block + History + Injection + Input). Memory block injected once at session start via `set_memory_block()`.
- **`response_parser.py`**: Splits XML logic. Conversational data → `output/tts_client.py`, Agent parameters → `agent/text_out.py`. Strips `[ref:sc_...]` tags from TTS output while keeping them in stored turns for session cache reference.
- **`router.py`**: Orchestrates turn processing. Fires `background_llm.create_task()` after each turn for memory extraction.
- **`emotion_tagger.py`**: Prepends emotion instructions matching the text.

### `output/` (The Mouth)
- **`tts_client.py`**: Streams raw text into Kokoro-ONNX TTS engine (local, 24kHz). Supports `synthesize_silent()` for startup warmup.
- **`audio_player.py`**: SoundDevice executor with pause/resume/flush. Feeds TTS chunks to AriaVoiceFilter, Gate 0, and V5 playback reference buffer. Clears playback reference on pause/flush/end.

### `memory/` (Five-Tier Memory System)
- **`short_term.py`**: Tier 1 — RAM deque with last 20 turns + emotion history. Instant access, session-only.
- **`session_cache.py`**: Tier 2 — RAM dict keyed by cache_id. Stores full article text, search results. Wiped on shutdown.
- **`fact_store.py`**: Tier 3 — SQLite-backed permanent user facts (name, city, preferences, relationships).
- **`procedural.py`**: Tier 4 — SQLite-backed standing instructions, pending reminders, relay messages.
- **`long_term.py`**: Tier 5 — ChromaDB with typed entries (`memory_type` + `category`), recency-weighted re-ranking, access tracking.
- **`compressor.py`**: Threshold-triggered compression — summarizes and stores to ChromaDB, then clears working memory (keeps last 3 turns).
- **`background_llm.py`**: Fire-and-forget `asyncio.create_task()` extraction from recent turns. Signal pre-filter avoids API calls on filler turns.
- **`loader.py`**: Session-start memory block assembly (350 token hard cap). Loads facts, instructions, relays, recent relationships.
- **`retriever.py`**: Parallel recall across fact_store, procedural, and long_term via `asyncio.gather()`.

### `agent/` (Logic Executions)
- **`text_out.py`**: Parses `<agent>` loops, tests valid JSON syntax.
- **`tool_router.py`**: Receives correct logic blocks mapping execution routes. Registered tools: `web_search`, `article_fetch`, `memory_recall`, `session_cache_qa`.
- **`slow_llm.py`**: Non-blocking asynchronous inference to the `kimi-k2` and `scout-17b` models returning JSON to the session queue.
- **`tools/memory_recall.py`**: On-demand recall across all memory tiers. Result → text_injector → main LLM synthesizes spoken response.
- **`tools/session_cache_qa.py`**: QA over cached content (articles, search results) from current session. Uses `llm_fast` for ~250ms latency. Result → TTS directly via SPOKEN_TOOL_OUTPUT.

---

## 📈 Latency Expectations
Expected execution duration targets end-to-end processing below `1.0s`. 
| Step | Action | Duration |
| ----------- | ----------- | ----------- |
| 1 | VAD End Detection | `~400ms` |
| 2 | STT Engine Processing | `~200ms` |
| 3 | Fast-Brain TTFT Logic | `~150ms` |
| 4 | First Sentence Formatting | `~50ms` |
| 5 | TTS Audio Fetching TTFB | `~150ms` |
| 6 | Device Output Processing | `~10ms` |
| **Total** | | **`~960ms`** |
