# VoxCore — Complete System Guide v2

> Full-Duplex Conversational AI | PersonaPlex-Inspired Architecture  
> Groq APIs + Local Kokoro-ONNX TTS | February 2026

---

## Changelog from v1

| Area | What Changed |
|---|---|
| **New module** | `input/speaking_monitor.py` — LLM-powered semantic interrupt gate |
| **New module** | `input/filler_detector.py` — Filters "yeah"/"uh-huh" from real interrupts |
| **New files** | `backchannel/clips/heart/laughs.wav`, `chuckles.wav`, `light_laugh.wav` |
| **Updated** | `input/interruption_detector.py` — Now defers to Speaking Monitor |
| **Updated** | `brain/response_parser.py` — Handles `[laughs]`, `[chuckles]`, `[...]` pause markers |
| **Updated** | `brain/prompt_builder.py` — Yield point instructions added to system prompt |
| **Updated** | `output/tts_client.py` — Intercepts laugh/sigh tags → plays clips instead of TTS |
| **Updated** | `output/audio_player.py` — 300ms silence injection for `[...]` markers |
| **Updated** | `core/turn_manager.py` — New SOFT_INJECT state added |
| **Updated** | `config.yaml` — Speaking monitor config block added |
| **Updated** | File structure, pipeline walkthrough, latency budget |

---

## Table of Contents

1. [What is VoxCore?](#1-what-is-voxcore)
2. [Architecture Overview](#2-architecture-overview)
3. [Conversation Dynamics — The Full Picture](#3-conversation-dynamics)
4. [Memory System](#4-memory-system)
5. [Tool & Agent System](#5-tool--agent-system)
6. [Connecting MCP Servers](#6-connecting-mcp-servers)
7. [Adding Custom Tools](#7-adding-custom-tools)
8. [Configuration Reference](#8-configuration-reference)
9. [API & Integration](#9-api--integration)
10. [How the Pipeline Works Step-by-Step](#10-pipeline-walkthrough)
11. [Extending VoxCore](#11-extending-voxcore)

---

## 1. What is VoxCore?

VoxCore is a **voice-first, full-duplex conversational AI** system. You speak to it, it speaks back — in real-time, with intelligent interruption handling, natural backchanneling, laughter, emotional responses, and agentic tool execution.

### Core Capabilities

| Feature | How It Works |
|---|---|
| **Full-duplex conversation** | VAD + Speaking Monitor simulates natural turn-taking |
| **Semantic interruption** | allam-2-7b classifies user intent before deciding to interrupt |
| **Soft injection** | Meaningful user additions mid-speech are injected, AI continues |
| **Backchanneling** | Pre-generated clips ("mm-hmm", "right") play during user pauses |
| **Laughter & sighs** | LLM outputs `[laughs]` tags → real audio clips play, not synthesized |
| **Natural yield points** | LLM inserts `[...]` pauses so user can speak without interrupting |
| **Persistent memory** | ChromaDB stores conversations across sessions |
| **Tool execution** | LLM generates `<agent>` tags → tools run async → results injected back |
| **Safety layer** | Llama Guard 4 filters input/output asynchronously |
| **Text injection** | External systems inject context mid-conversation via API |

### What Powers It

| Component | Technology | Cost |
|---|---|---|
| STT | Groq Whisper Large V3 Turbo | Free tier |
| LLM (fast) | Groq llama-3.1-8b-instant | Free tier |
| LLM (speaking monitor) | Groq allam-2-7b | Free tier |
| LLM (smart) | Groq llama-4-scout-17b | Free tier |
| LLM (agentic) | Groq kimi-k2-instruct | Free tier |
| LLM (deep) | Groq qwen/qwen3-32b | Free tier |
| TTS | Kokoro-ONNX (local) | Free, runs on CPU |
| Memory | ChromaDB (local) | Free, local storage |
| Safety | Groq Llama Guard 4 | Free tier |

---

## 2. Architecture Overview

### Data Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                           INPUTS                                  │
│  🎙 Microphone ──► 16kHz mono PCM audio stream                   │
│  📝 Text-In    ──► Context / tool results / notifications         │
│  📄 Sys Prompt  ──► Persona definition (set once at boot)         │
└──────────────────────┬───────────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                          PIPELINE                                 │
│                                                                    │
│  MicStream ──► VAD(Silero) ──► STT(Whisper) ──► Session          │
│       │                                              │             │
│       │◄── [LISTENING mode] ──────────────────────── │             │
│       │                                              │             │
│       │    Backchannel                 text-in ──────┤             │
│       │    Detector                                  ▼             │
│       │       │                           PromptBuilder            │
│       │  play clip ◄──                        │                    │
│       │                                       ▼                    │
│       │                            LLM (llama-3.1-8b)             │
│       │                         ↙         ↓          ↘            │
│       │                  <speech>     <[...]pause>  <agent JSON>   │
│       │                     │              │              │        │
│       │◄── [SPEAKING mode]  │         AudioPlayer   ToolRouter     │
│       │         │           ▼              │              │        │
│       │   SpeakingMonitor  TTS        300ms silence  Tools run     │
│       │   (allam-2-7b)  (Kokoro)           │          async        │
│       │         │           │              │              │        │
│       │    ┌────┴────┐      ▼              │         inject back   │
│       │    │IGNORE   │ AudioPlayer ◄───────┘              │        │
│       │    │INJECT   │ + laugh clips                      │        │
│       │    │INTERRUPT│                                    │        │
│       │    └────┬────┘                                    │        │
│       │         │                                         │        │
└───────┼─────────┼─────────────────────────────────────────────────┘
        │         │
        │    ┌────▼─────────────────────────────────┐
        │    │ IGNORE  → do nothing, note as filler  │
        │    │ INJECT  → text_injector.inject()       │
        │    │           AI continues speaking        │
        │    │ INTERRUPT → kill TTS, THINKING state   │
        │    └──────────────────────────────────────┘
        │
┌───────▼──────────────────────────────────────────────────────────┐
│                           OUTPUTS                                  │
│  🔊 Voice-Out  ◄── Streaming audio (sentence-by-sentence)         │
│  📤 Text-Out   ◄── JSON tool calls / agent events                  │
│  💾 Memory     ◄── ChromaDB persistent storage                     │
└──────────────────────────────────────────────────────────────────┘
```

### Turn State Machine

VoxCore now runs a **5-state machine**:

```
                     ┌──────────┐
          VAD end    │ LISTENING │ ◄── VAD running, backchannel active
          of speech  └─────┬────┘
                           ▼
                     ┌──────────┐
          First LLM  │ THINKING │ ◄── STT done, LLM starting
          token      └─────┬────┘
                           ▼
                     ┌──────────┐
          LLM done   │ SPEAKING │ ◄── TTS playing, SpeakingMonitor active
          or empty   └─────┬────┘
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
      Back to        ┌──────────┐   ┌─────────────┐
      LISTENING      │SOFT_INJECT│   │ INTERRUPTED  │
                     └────┬─────┘   └──────┬──────┘
                          │                ▼
                    AI continues    Back to THINKING
                    speaking,       with interrupt
                    context added   context prepended
```

**New: SOFT_INJECT state** — triggered when Speaking Monitor classifies user speech as INJECT. The AI does not stop. The user's words are silently added to the context buffer. The AI finishes its current sentence naturally and the new context surfaces in its next response. From the user's perspective it feels like the AI heard them even while talking.

### File Structure

```
voxcore/
├── main.py                            # Entry point, boots all async tasks
├── config.yaml                        # All configuration
├── .env                               # GROQ_API_KEY
│
├── core/
│   ├── session.py                     # Master state: history, context, turn state
│   ├── event_bus.py                   # Async pub/sub connecting all modules
│   └── turn_manager.py                # 5-state machine orchestrator [UPDATED]
│
├── input/
│   ├── mic_stream.py                  # 16kHz mic capture with noise floor
│   ├── vad.py                         # Silero VAD with echo suppression
│   ├── stt.py                         # Groq Whisper STT
│   ├── text_injector.py               # Context injection API
│   ├── interruption_detector.py       # Gate 1: Duration + energy [UPDATED]
│   ├── filler_detector.py             # Gate 2: Word-list filler check [NEW]
│   └── speaking_monitor.py            # Gate 3: allam-2-7b intent classifier [NEW]
│
├── brain/
│   ├── llm_client.py                  # Groq streaming LLM wrapper
│   ├── prompt_builder.py              # System + memory + yield instructions [UPDATED]
│   ├── response_parser.py             # Handles speech, agent, [laughs], [...] [UPDATED]
│   ├── emotion_tagger.py              # Ensures emotion tags in output
│   └── router.py                      # Routes to fast/smart/agentic brain
│
├── output/
│   ├── tts_client.py                  # Kokoro-ONNX local TTS [UPDATED]
│   ├── audio_player.py                # Speaker output + pause injection [UPDATED]
│   └── voice_profile.py               # Voice configuration
│
├── agent/
│   ├── text_out.py                    # Emits validated agent JSON
│   ├── tool_router.py                 # Routes tool calls to handlers
│   ├── slow_llm.py                    # Background calls to powerful models
│   └── tools/
│       ├── base_tool.py               # Abstract base class
│       ├── web_search.py              # Web search (Tavily / DuckDuckGo)
│       ├── memory_tool.py             # Read/write ChromaDB memory
│       └── calendar_tool.py           # Local SQLite calendar
│
├── memory/
│   ├── short_term.py                  # Rolling 20-turn in-memory window
│   ├── long_term.py                   # ChromaDB persistent vector store
│   └── compressor.py                  # Summarizes old turns via qwen3-32b
│
├── safety/
│   └── guard.py                       # Llama Guard 4 async safety filter
│
├── backchannel/
│   ├── cue_detector.py                # Phrase boundary detection
│   ├── selector.py                    # Context-aware clip selection [UPDATED]
│   ├── generator.py                   # One-time clip generation script [UPDATED]
│   └── clips/heart/
│       ├── uh_huh.wav
│       ├── yeah.wav
│       ├── i_see.wav
│       ├── go_on.wav
│       ├── interesting.wav
│       ├── right.wav
│       ├── mm_hmm.wav
│       ├── okay.wav
│       ├── sure.wav
│       ├── oh_wow.wav
│       ├── got_it.wav
│       ├── tell_me_more.wav
│       ├── really.wav
│       ├── makes_sense.wav
│       ├── understood.wav
│       ├── laughs.wav          # [NEW] Natural laugh
│       ├── chuckles.wav        # [NEW] Light chuckle
│       ├── light_laugh.wav     # [NEW] Subtle amused sound
│       └── sighs.wav           # [NEW] Thoughtful sigh
│
└── interfaces/
    ├── cli.py                         # Terminal debug interface
    ├── websocket_server.py            # WebSocket for web/mobile clients
    └── api.py                         # REST API endpoints
```

---

## 3. Conversation Dynamics — The Full Picture

This is the section that changed most significantly in v2. The core problem with v1 was that interruption was purely mechanical — any audio energy above threshold killed the AI mid-sentence. This section documents the complete new approach.

---

### 3.1 The Three-Gate Interrupt System

When the user makes any sound while the AI is speaking, it passes through three sequential gates before anything happens:

```
User audio detected while AI speaking
              │
              ▼
┌─────────────────────────────────────┐
│  GATE 1 — Duration + Energy         │  ~1ms (local, no API)
│  interruption_detector.py           │
│                                     │
│  Is speech > 600ms AND              │
│  energy > noise floor baseline?     │
│                                     │
│  NO  ──► drop silently (cough,      │
│          breath, bump)              │
│  YES ──► continue to Gate 2         │
└─────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  GATE 2 — Filler Word Check         │  ~2ms (local word list)
│  filler_detector.py                 │
│                                     │
│  Partial transcript available?      │
│  Is it in the filler list?          │
│  ("yeah", "uh-huh", "okay",         │
│   "right", "sure", "mm", "haha",    │
│   "lol", "wow", "oh", "nice")       │
│                                     │
│  YES ──► note as implicit           │
│          backchannel, do nothing    │
│  NO  ──► continue to Gate 3         │
└─────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  GATE 3 — Speaking Monitor LLM      │  ~400-600ms total
│  speaking_monitor.py                │  (whisper ~250ms +
│                                     │   allam ~150ms)
│  Transcribe 400-600ms of audio      │
│  (Groq whisper-turbo)               │
│                                     │
│  Classify with allam-2-7b:          │
│  ┌────────────────────────────┐     │
│  │ IGNORE — pure filler or   │     │
│  │   reaction, AI continues  │     │
│  │                            │     │
│  │ INJECT — meaningful but   │     │
│  │   not urgent, add to      │     │
│  │   context silently        │     │
│  │                            │     │
│  │ INTERRUPT — clear intent  │     │
│  │   to stop/redirect, kill  │     │
│  │   TTS, go to THINKING     │     │
│  └────────────────────────────┘     │
└─────────────────────────────────────┘
```

**Why this works:** The allam-2-7b classifier sees both what the AI was saying AND what the user said. "No way!" means something completely different depending on whether the AI just told a joke vs delivered bad news. The classifier has that context.

---

### 3.2 The Speaking Monitor — allam-2-7b

**File:** `input/speaking_monitor.py`

The Speaking Monitor is a dedicated module that runs **only** during the SPEAKING state. It is completely idle during LISTENING and THINKING, so it adds zero overhead to the normal turn flow.

#### Activation / Deactivation

```
Event: STATE → SPEAKING   ──► SpeakingMonitor.activate()
Event: STATE → LISTENING  ──► SpeakingMonitor.deactivate()
Event: STATE → THINKING   ──► SpeakingMonitor.deactivate()
```

#### The Classifier Prompt

The prompt is deliberately minimal — one word output only. Token cost per call is approximately 150-200 tokens including context.

```
System:
You are a real-time speech intent classifier for a voice AI system.
The AI assistant is currently speaking. The user has said something.
Classify the user's intent as exactly one of three words:

IGNORE — The user is reacting naturally without wanting to take over.
  Examples: "yeah", "right", "uh-huh", "wow", "haha", "oh interesting",
            laughter, short affirmations, single word reactions.

INJECT — The user added something meaningful but doesn't need the AI to stop.
  Examples: "oh and also...", "by the way...", "actually I forgot to mention",
            adding context, minor corrections that don't change the topic.

INTERRUPT — The user clearly wants the AI to stop and respond to them.
  Examples: "wait", "stop", "hold on", "no", "actually...", "I have a question",
            asking a new question, contradicting, expressing urgency.

Context:
AI was saying: "{last_30_words_of_ai_speech}"
User said: "{transcript}"

Reply with ONLY one word: IGNORE, INJECT, or INTERRUPT
```

#### Rate Limit Management

allam-2-7b has 30 RPM on your free tier. Each call is ~150-200 tokens. The monitor has a built-in debounce: it will not fire more than once every 3 seconds regardless of how much audio comes in. In a typical conversation where the user speaks over the AI 3-5 times, this uses approximately 5-10 of your 30 RPM — well within limits.

#### Rolling Window Strategy

The monitor does not wait for SPEECH_END to fire. It fires on a rolling 400-600ms audio window. This means:

- At 400ms of user speech → Gate 1 + 2 pass → whisper transcription starts
- At 600ms → transcript arrives → allam classification fires
- Total decision time from user starting to speak: ~700-900ms
- This is perceptually natural — humans also take 300-500ms to yield

If the user keeps speaking after a classification, the monitor re-fires every 3 seconds with updated audio. Classifications can change — IGNORE can escalate to INJECT can escalate to INTERRUPT as the user says more.

---

### 3.3 The Three Outcomes

#### IGNORE

Nothing happens. The AI keeps speaking without interruption. The user's words are noted internally as an implicit engagement signal. If the filler was reactive to something funny ("haha"), this is passed to the backchannel selector as a `POSITIVE_REACTION` event which may influence the AI's next emotional tag.

#### INJECT

This is the most powerful new behavior. The AI does **not** stop. Instead:

1. `text_injector.inject(transcript, priority="high", source="user_mid_speech")` is called silently
2. Session state moves briefly to `SOFT_INJECT` to record the event
3. TTS continues playing without interruption
4. The injected content appears in the next LLM prompt as `[USER ADDED MID-SPEECH]: ...`
5. The AI's next sentence naturally incorporates what the user said

**Example experience:**

> AI: "So the main thing to know about black holes is that they form when a massive star—"
> User: "oh and I read something about neutron stars too"
> AI: "—collapses under its own gravity. [cheerful] And neutron stars are actually the intermediate step — great that you mentioned them."

The AI heard the user. It didn't stop. It wove the addition in naturally. This feels like a real conversation.

#### INTERRUPT

Full interrupt. In order:

1. `audio_player.flush()` — TTS queue emptied, audio cuts cleanly at the current 20ms frame boundary (no mid-word cut)
2. Partial AI speech is saved to session as `[INTERRUPTED AT]: ...`
3. The monitor's partial transcript is pre-loaded into the STT result buffer (head start — no full STT cycle needed)
4. State → THINKING with context: `"User interrupted saying: {transcript}"`
5. Fast brain responds naturally to the interruption

---

### 3.4 Natural Yield Points

**Files:** `brain/prompt_builder.py`, `output/audio_player.py`, `brain/response_parser.py`

PersonaPlex's model learns naturally when to pause mid-speech to let users respond. VoxCore approximates this by instructing the LLM to insert explicit pause markers.

#### System Prompt Addition (in `prompt_builder.py`)

```
CONVERSATION RHYTHM RULES:
- Keep each spoken response to 2-3 sentences maximum before pausing.
- After completing a thought, insert [...] to create a natural breath point.
- Use [...] before asking a question, after delivering information,
  or when transitioning topics. Example:
  "So the first thing to know is that black holes form from collapsing stars. [...]
   The really fascinating part is what happens at the event horizon."
- Use [laughs] when something is genuinely funny.
- Use [chuckles] for mild amusement.
- Use [sighs] when being thoughtful or empathetic.
- Never use these tags back-to-back. One per response maximum.
```

#### How `[...]` is Processed

`response_parser.py` watches the token stream for `[...]`. When detected:

1. It fires `PAUSE_MARKER` event to `audio_player.py`
2. `audio_player.py` inserts 300ms of silence into the audio queue at that position
3. VAD is still running during this silence — if the user speaks, it's a clean natural handoff, not an interruption at all
4. If user stays silent, audio continues after the 300ms

This single change dramatically improves turn-taking feel. The AI naturally creates moments where the user *can* speak without having to interrupt.

---

### 3.5 Laughter and Emotional Reactions

**Files:** `output/tts_client.py`, `backchannel/clips/heart/`

Kokoro-ONNX cannot synthesize convincing laughter. Attempting to TTS the word "haha" produces robotic output. The solution is identical to the backchannel approach: pre-generated audio clips.

#### New Clips (generated once via Groq Orpheus, then local forever)

| Tag | Clip File | Description |
|---|---|---|
| `[laughs]` | `laughs.wav` | Full natural laugh, ~1.5 seconds |
| `[chuckles]` | `chuckles.wav` | Light chuckle, ~0.8 seconds |
| `[light_laugh]` | `light_laugh.wav` | Brief amused sound, ~0.5 seconds |
| `[sighs]` | `sighs.wav` | Thoughtful sigh, ~0.6 seconds |

#### How They Work at Runtime

`response_parser.py` detects these tags in the LLM stream. Instead of passing them to Kokoro-ONNX for synthesis, it fires a `PLAY_CLIP` event with the clip filename. `tts_client.py` intercepts this and routes to `audio_player.py` directly — same as backchannel clips but in-line with the response.

**Example LLM output:**

```
[cheerful] Oh that's a great question about penguins! [chuckles]
They actually can't fly, but they're incredible swimmers —
up to 25 miles per hour underwater. [...]
Pretty remarkable for a bird that looks like it's wearing a tuxedo.
```

**What the user hears:**
- Cheerful TTS: "Oh that's a great question about penguins!"
- Pre-recorded chuckle clip plays
- TTS: "They actually can't fly, but they're incredible swimmers — up to 25 miles per hour underwater."
- 300ms natural pause (the `[...]`)
- TTS: "Pretty remarkable for a bird that looks like it's wearing a tuxedo."

#### Generator Update

`backchannel/generator.py` now also generates laugh/sigh clips. Add these to `BACKCHANNEL_PHRASES`:

```python
# Emotional reaction clips (added to BACKCHANNEL_PHRASES in generator.py)
EMOTIONAL_CLIPS = {
    "laughs":       "Ha, that's great",          # Short natural laugh phrase
    "chuckles":     "Heh",                        # Light amusement
    "light_laugh":  "Hm heh",                     # Very brief
    "sighs":        "Mm",                         # Thoughtful pause sound
}
```

> **Note:** These are generated with Kokoro-ONNX locally, not Groq Orpheus, so they cost nothing and match the AI's voice exactly. The generator script handles both backchannel and emotional clip generation in one pass.

---

### 3.6 Echo Suppression Improvement

**File:** `input/mic_stream.py`

The v1 approach was a simple time-gate: mute VAD during SPEAKING + 700ms cooldown. This sometimes caused the Speaking Monitor to trigger on the AI's own audio leaking through the mic.

The improved approach adds a **reference signal subtraction**:

- `audio_player.py` maintains a rolling buffer of the last 2 seconds of audio it has played
- `mic_stream.py` receives this buffer and checks cross-correlation between mic input and playback buffer
- If correlation > 0.7, the chunk is classified as echo and dropped before VAD sees it
- This is not full AEC (which requires `webrtcvad` or `pyaec`) but catches >90% of echo leakage
- Full AEC via `pyaec` can be enabled via config for environments with loud speakers

---

## 4. Memory System

VoxCore has a **three-tier memory architecture** that gives the AI persistent, context-aware recall across sessions.

### 4.1 Short-Term Memory (RAM)

**File:** `memory/short_term.py`

- Rolling deque of the last 20 conversation turns
- Each turn: `{role, content, timestamp, emotion, turn_id}`
- Subscribes to `TURN_COMPLETE` events from Session
- When turns overflow (> 20), publishes `MEMORY_COMPRESS` with the overflow batch

```
Session.add_turn() → TURN_COMPLETE → ShortTermMemory._on_turn_complete()
                                          │
                                   if overflow ≥ 5 turns:
                                          │
                                     MEMORY_COMPRESS
```

### 4.2 Memory Compressor (Summarization)

**File:** `memory/compressor.py`

- Subscribes to `MEMORY_COMPRESS` events
- Batches turns (5-10 at a time)
- Calls `qwen/qwen3-32b` via Groq to produce a concise summary capturing:
  - Key facts (names, preferences, dates)
  - Decisions and action items
  - Emotional context
  - Explicit "remember this" requests
- Publishes `MEMORY_COMPRESSED` with the summary

### 4.3 Long-Term Memory (ChromaDB)

**File:** `memory/long_term.py`

- ChromaDB persistent vector store at `data/chromadb/`
- Stores summaries from the compressor AND full session transcripts on shutdown
- Semantic similarity search using cosine distance
- Default embedding model: all-MiniLM-L6-v2 (built into ChromaDB)

#### How Memory Gets Saved

| Trigger | What Gets Stored | When |
|---|---|---|
| **Compression overflow** | Summarized batch of old turns | After 25+ turns in a session |
| **Session shutdown** | Full session transcript | Every time VoxCore exits (Ctrl+C) |
| **Explicit memory tool** | User-requested content | When user says "Remember that..." |

#### Memory Flow Diagram

```
┌──────────────┐     TURN_COMPLETE      ┌──────────────────┐
│   Session    │ ─────────────────────► │ ShortTermMemory   │
│  (add_turn)  │                        │  (20-turn deque)  │
└──────────────┘                        └────────┬──────────┘
                                                 │
                                        overflow ≥ 5 turns
                                                 │
                                          MEMORY_COMPRESS
                                                 │
                                                 ▼
                                       ┌──────────────────┐
                                       │ MemoryCompressor  │
                                       │ (qwen/qwen3-32b)  │
                                       └────────┬──────────┘
                                                 │
                                        MEMORY_COMPRESSED
                                                 │
                                   ┌─────────────┼─────────────┐
                                   ▼                           ▼
                           ┌──────────────┐         ┌──────────────────┐
                           │   Session    │         │  LongTermMemory   │
                           │ (compressed_ │         │   (ChromaDB)      │
                           │  summary)   │         │   store()         │
                           └─────────────┘         └──────────────────┘
```

---

## 5. Tool & Agent System

### How Tools Work End-to-End

```
1. User: "What's the weather in London?"
          │
2. LLM output: "[curious] Let me check that for you. [...]
                <agent>{"action": "web_search", "params": {"query": "London weather"}}</agent>"
          │
3. ResponseParser splits:
   ├── Speech: "Let me check that for you." → TTS → Speaker
   ├── Pause: [...] → 300ms silence injected
   └── Agent JSON: {"action": "web_search", ...} → AGENT_JSON_OUT event
          │
4. TextOut validates JSON → publishes AGENT_JSON_OUT
          │
5. ToolRouter receives event → WebSearchTool.execute() → runs async
          │
6. Tool returns: "London: 12°C, cloudy, light rain expected"
          │
7. TextInjector.inject_tool_result() → "[TOOL RESULT — web_search]: London: 12°C..."
          │
8. Next LLM call: "[cheerful] It's 12 degrees in London right now, cloudy with light rain."
```

### Built-in Tools

#### 1. Web Search (`web_search`)

```python
<agent>{"action": "web_search", "params": {"query": "latest AI news"}}</agent>
```

| Param | Type | Required | Description |
|---|---|---|---|
| `query` | string | Yes | Search query |

#### 2. Memory (`memory`)

```python
<agent>{"action": "memory", "params": {"operation": "write", "content": "User's birthday is March 15"}}</agent>
<agent>{"action": "memory", "params": {"operation": "read", "query": "user birthday"}}</agent>
```

| Param | Type | Required | Description |
|---|---|---|---|
| `operation` | string | Yes | `"read"` or `"write"` |
| `content` | string | For write | Content to store |
| `query` | string | For read | Semantic search query |
| `top_k` | int | No | Number of results (default: 5) |

#### 3. Calendar (`calendar`)

```python
<agent>{"action": "calendar", "params": {"action": "create", "title": "Meeting", "event_time": "2026-02-21T14:00:00"}}</agent>
<agent>{"action": "calendar", "params": {"action": "list", "hours_ahead": 48}}</agent>
```

| Param | Type | Required | Description |
|---|---|---|---|
| `action` | string | Yes | `"create"`, `"list"`, or `"delete"` |
| `title` | string | For create | Event title |
| `event_time` | string | For create | ISO 8601 datetime |
| `description` | string | No | Event description |
| `hours_ahead` | int | No | List window (default: 24) |
| `event_id` | int | For delete | Event ID to cancel |

### The Slow LLM — Background Intelligence

| Task Type | Model Used | When |
|---|---|---|
| Tool execution | kimi-k2-instruct | Multi-step agentic workflows |
| Complex reasoning | llama-4-scout-17b | User asks hard questions |
| Document analysis | qwen/qwen3-32b | Long text summarization |
| Maximum intelligence | gpt-oss-120b | Hardest tasks (rate-limited) |

---

## 6. Connecting MCP Servers

**Model Context Protocol (MCP)** connects VoxCore to external tools — databases, APIs, file systems, cloud services. VoxCore's tool architecture integrates with MCP servers seamlessly.

### Architecture: VoxCore + MCP

```
┌──────────────────────────────────────────────────────┐
│                    VoxCore Pipeline                    │
│                                                       │
│  LLM → ResponseParser → TextOut → ToolRouter          │
│                                       │               │
│                          ┌────────────┴──────────┐    │
│                          │    Tool Registry       │    │
│                          │  web_search  (built-in)│    │
│                          │  memory      (built-in)│    │
│                          │  calendar    (built-in)│    │
│                          │  mcp:github  (MCP)     │    │
│                          │  mcp:slack   (MCP)     │    │
│                          └────────────┬──────────┘    │
└───────────────────────────────────────┼───────────────┘
                                        │
                              ┌─────────┴─────────┐
                              │   MCP Client       │
                              │  (JSON-RPC/stdio)  │
                              └─────────┬─────────┘
                                        │
                    ┌───────────────────┼─────────────────┐
                    │                   │                  │
              ┌─────┴─────┐    ┌────────┴──────┐   ┌──────┴──────┐
              │ GitHub MCP │    │   Slack MCP   │   │  Email MCP  │
              └────────────┘    └───────────────┘   └─────────────┘
```

### How to Connect an MCP Server

#### Step 1: Install

```bash
npx @modelcontextprotocol/server-github
npx @modelcontextprotocol/server-filesystem /path/to/allowed/dir
npx @modelcontextprotocol/server-sqlite --db-path ./data/mydb.sqlite
```

#### Step 2: Add to `config.yaml`

```yaml
mcp_servers:
  github:
    command: ["npx", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "ghp_..."

  filesystem:
    command: ["npx", "@modelcontextprotocol/server-filesystem", "/Users/you/Documents"]

  slack:
    command: ["npx", "@anthropic/mcp-server-slack"]
    env:
      SLACK_BOT_TOKEN: "xoxb-..."
```

#### Step 3: Wire into `main.py`

```python
mcp_config = config.get("mcp_servers", {})
for server_name, server_cfg in mcp_config.items():
    client = MCPClient(server_cfg["command"], server_name)
    await client.start()
    tools = await client.list_tools()
    for tool_schema in tools:
        bridge = MCPBridgeTool(client, tool_schema)
        tool_router.register_tool(bridge.name, bridge)
```

### MCP Server Directory

| Server | What It Does | Install |
|---|---|---|
| `@modelcontextprotocol/server-github` | GitHub issues, PRs, repos | `npx ...` |
| `@modelcontextprotocol/server-filesystem` | File read/write/search | `npx ...` |
| `@modelcontextprotocol/server-sqlite` | SQLite queries | `npx ...` |
| `@modelcontextprotocol/server-postgres` | PostgreSQL queries | `npx ...` |
| `@anthropic/mcp-server-slack` | Slack messages | `npx ...` |
| `@anthropic/mcp-server-google-maps` | Location & directions | `npx ...` |
| `@anthropic/mcp-server-brave-search` | Brave web search | `npx ...` |
| `mcp-server-fetch` | HTTP fetch any URL | `npx ...` |
| `@modelcontextprotocol/server-puppeteer` | Browser automation | `npx ...` |
| `mcp-server-home-assistant` | Smart home control | `pip install ...` |
| `mcp-server-spotify` | Music playback/search | `npx ...` |
| `mcp-server-notion` | Notion pages, databases | `npx ...` |

---

## 7. Adding Custom Tools

### Creating a New Built-in Tool

1. **Create** `agent/tools/your_tool.py`:

```python
import logging
from agent.tools.base_tool import BaseTool

class YourTool(BaseTool):
    name = "your_tool"
    description = "Does something useful"
    required_params = ["input_text"]
    optional_params = ["option_a"]

    def __init__(self) -> None:
        super().__init__()
        self._logger = logging.getLogger("Tool.your_tool")

    async def execute(self, params: dict) -> str:
        input_text = params["input_text"]
        return f"Result: {input_text}"
```

2. **Register** in `agent/tool_router.py` → add to `TOOL_MAP`

3. **Enable** in `config.yaml` → add to `agent.tools`

4. **Tell the LLM** → update system prompt in `config.yaml`

---

## 8. Configuration Reference

### config.yaml — Complete Schema

```yaml
persona:
  name: "Aria"
  system_prompt: |
    You are Aria, a warm, focused, and capable assistant.
    Keep all spoken responses under 3 sentences before a yield point.
    Always start with an emotion tag: [cheerful] [calm] [concerned] [excited] [empathetic]
    Use [...] to create natural pause points where the user can speak.
    Use [laughs] when something is genuinely funny. [chuckles] for mild humor. [sighs] when thoughtful.
    Only one paralinguistic tag per response maximum.
    When performing a task output alongside speech:
    <agent>{"action": "tool_name", "params": {...}}</agent>
  voice: "af_heart"
  language: "en"

models:
  stt_primary: "whisper-large-v3-turbo"
  stt_fallback: "https://your-hf-space.hf.space/transcribe"
  llm_fast: "llama-3.1-8b-instant"
  llm_monitor: "allam-2-7b"                        # Speaking Monitor classifier
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
  vad_speech_threshold: 0.65
  vad_silence_threshold: 0.6
  vad_silence_duration_ms: 500
  noise_floor_calibration_s: 0.5
  echo_correlation_threshold: 0.7      # Echo suppression sensitivity

speaking_monitor:
  enabled: true
  model: "allam-2-7b"
  gate1_min_duration_ms: 600           # Gate 1: min speech duration
  gate2_filler_list:                   # Gate 2: word-list fillers
    - "yeah"
    - "yep"
    - "okay"
    - "ok"
    - "right"
    - "sure"
    - "uh-huh"
    - "mm-hmm"
    - "haha"
    - "lol"
    - "wow"
    - "oh"
    - "nice"
    - "cool"
    - "great"
  gate3_debounce_s: 3                  # Gate 3: min seconds between LLM calls
  gate3_audio_window_ms: 600           # Audio to send to whisper for classification
  context_words: 30                    # Last N words of AI speech sent as context

backchannel:
  enabled: true
  min_pause_ms: 350
  min_gap_between_s: 8
  min_user_speech_s: 2
  clips_dir: "backchannel/clips/heart/"
  energy_threshold: 0.02
  emotional_clips:                     # Laugh/sigh clips config
    laughs: "laughs.wav"
    chuckles: "chuckles.wav"
    light_laugh: "light_laugh.wav"
    sighs: "sighs.wav"

memory:
  short_term_turns: 20
  compress_after_turns: 20
  long_term_enabled: true
  long_term_db: "./data/memory.db"
  retrieval_top_k: 5
  embedding_model: "default"

agent:
  enabled: true
  text_out_webhook: null
  max_concurrent_tools: 3
  tool_timeout_s: 30
  tools:
    - web_search
    - memory
    - calendar

safety:
  enabled: true
  check_input: true
  check_output: true

server:
  host: "0.0.0.0"
  port: 8000
  websocket_enabled: true
```

### Environment Variables (`.env`)

```bash
GROQ_API_KEY=gsk_...
TAVILY_API_KEY=tvly_...
HF_API_KEY=hf_...
```

---

## 9. API & Integration

### REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/inject` | Inject text context into conversation |
| `GET` | `/transcript` | Get current session transcript |
| `GET` | `/status` | Current turn state + session info |
| `POST` | `/config` | Hot-reload persona or voice |
| `WS` | `/text-out` | Stream of agent JSON events |

### Text Injection API

```python
await text_injector.inject(
    content="Meeting with John at 3pm tomorrow",
    priority="high",
    source="calendar"
)

# POST /inject
# {"content": "Email from boss: quarterly review moved to Friday", "priority": "high"}
```

### WebSocket Protocol

```json
// From VoxCore:
{"type": "transcript", "text": "Hello there", "role": "user"}
{"type": "speech", "text": "Hi! How can I help?", "emotion": "cheerful"}
{"type": "monitor", "classification": "INJECT", "user_text": "oh and also..."}
{"type": "agent", "action": "web_search", "params": {"query": "..."}}
{"type": "state", "old": "listening", "new": "thinking"}

// To VoxCore:
{"type": "inject", "content": "User opened the weather app", "priority": "normal"}
{"type": "audio", "data": "<base64 audio>"}
```

---

## 10. Pipeline Walkthrough

### Normal Turn (unchanged from v1)

**Step 1 — Audio Capture:** `sounddevice` captures 16kHz mono PCM in 32ms chunks → pushed to VAD and SpeakingMonitor queues.

**Step 2 — VAD:** Silero runs per chunk (~0.5ms). Speech probability > 0.65 → `SPEECH_START`. Silence for 500ms → `SPEECH_END` + buffered audio.

**Step 3 — STT:** Buffered audio → Groq whisper-large-v3-turbo → `TRANSCRIPT_READY`.

**Step 4 — Turn Management:** State → THINKING. Session updated. `PromptBuilder.build()` assembles full context. LLM streaming starts.

**Step 5 — LLM + Parser:** llama-3.1-8b streams tokens. `ResponseParser` handles:
- Complete sentences → `LLM_SPEECH_TOKEN` → TTS
- `[laughs]` / `[chuckles]` / `[sighs]` → `PLAY_CLIP` event → audio clip
- `[...]` → `PAUSE_MARKER` → 300ms silence in audio queue
- `<agent>` JSON → `LLM_AGENT_TAG` → tool execution

**Step 6 — TTS:** Each sentence → Kokoro-ONNX → ~50-100ms → `TTS_CHUNK_READY`.

**Step 7 — Playback:** `AudioPlayer` queues and plays chunks. Handles clip playback, pause insertion, and interrupt flush. SpeakingMonitor is now active.

**Step 8 — Return:** `PLAYBACK_DONE` → state → LISTENING. 700ms VAD cooldown. Cycle repeats.

---

### Interrupt Turn (new in v2)

**User speaks while AI is mid-sentence:**

1. MicStream chunk arrives → Gate 1 checks duration + energy (~1ms local)
2. If Gate 1 passes → Gate 2 checks word-list fillers (~2ms local)
3. If Gate 2 passes → Gate 3 fires: whisper transcribes 600ms audio (~250ms Groq), allam-2-7b classifies (~150ms Groq)
4. **If IGNORE:** nothing happens, AI continues
5. **If INJECT:** `text_injector.inject()` silently, state briefly SOFT_INJECT, AI continues speaking, context appears next turn
6. **If INTERRUPT:** audio flushed cleanly, partial transcript pre-loaded, state → THINKING, AI responds to interruption naturally

---

### Latency Budget

| Step | Component | Latency |
|---|---|---|
| VAD end-of-turn | Silero (local) | ~500ms |
| STT | Groq whisper-turbo | ~200ms |
| LLM first token | llama-3.1-8b | ~100-200ms |
| First sentence buffer | ResponseParser | ~50ms |
| TTS synthesis | Kokoro-ONNX (local) | ~50-100ms |
| Playback start | sounddevice | ~10ms |
| **Normal turn total** | | **~910-1060ms** |
| | | |
| Gate 1 (duration check) | Local | ~1ms |
| Gate 2 (filler check) | Local | ~2ms |
| Gate 3 (whisper + allam) | Groq | ~400-600ms |
| **Interrupt decision total** | | **~700-900ms from speech start** |

---

## 11. Extending VoxCore

### Adding a New Persona

```yaml
persona:
  name: "Max"
  system_prompt: |
    You are Max, a witty and sarcastic tech expert.
    You love puns. Keep responses under 3 sentences before [...].
    Use [laughs] when puns land particularly well.
  voice: "am_adam"
```

### Adding a New Voice

Available Kokoro-ONNX voices:
- Female US: `af_heart`, `af_bella`, `af_nicole`, `af_sarah`, `af_sky`
- Male US: `am_adam`, `am_michael`
- British: `bf_emma`, `bm_george`

After changing voice, regenerate all clips (backchannel + emotional):

```bash
python -m backchannel.generator --force
```

### Running as a Server

```bash
python main.py --interface server --port 8000
```

### Running Headless (No Mic)

```bash
python main.py --no-mic --interface cli
```

---

## Quick Start

```bash
# 1. Clone and install
git clone <your-repo-url> voxcore
cd voxcore
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Add GROQ_API_KEY to .env

# 3. Download TTS models
# Place kokoro-v1.0.onnx + voices-v1.0.bin in models/

# 4. Generate backchannel + emotional clips (one-time)
python -m backchannel.generator

# 5. Run
python main.py
```

---

## New Files Summary for Coding Agent

These are the net-new files that need to be created in v2:

| File | Purpose |
|---|---|
| `input/speaking_monitor.py` | Three-gate interrupt system with allam-2-7b classifier |
| `input/filler_detector.py` | Gate 2 word-list filler detection, returns bool + matched word |

These existing files need to be updated:

| File | What Changes |
|---|---|
| `core/turn_manager.py` | Add SOFT_INJECT state and its transition logic |
| `input/interruption_detector.py` | Remove standalone logic, now just Gate 1, defers to speaking_monitor |
| `brain/prompt_builder.py` | Add yield point and emotional tag instructions to system prompt assembly |
| `brain/response_parser.py` | Handle `[laughs]`, `[chuckles]`, `[sighs]`, `[...]` as special tokens |
| `output/tts_client.py` | Route PLAY_CLIP events to audio_player instead of Kokoro synthesis |
| `output/audio_player.py` | Add 300ms silence injection on PAUSE_MARKER, add clip playback path |
| `input/mic_stream.py` | Add reference signal buffer + correlation-based echo suppression |
| `backchannel/generator.py` | Add emotional clips (laughs, chuckles, sighs) to generation pass |
| `backchannel/selector.py` | Add POSITIVE_REACTION event handler for IGNORE classifications |
| `config.yaml` | Add speaking_monitor block, llm_monitor model, emotional_clips block |

---

*VoxCore System Guide v2 — February 2026*