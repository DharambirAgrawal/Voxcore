


VOXCORE
Full-Duplex Conversational AI System

PersonaPlex-Inspired Architecture Using Groq APIs + Local Kokoro-ONNX TTS


Complete Engineering Blueprint  |  February 2026




 1. What Are We Building?
VoxCore is a voice-first, full-duplex conversational AI system inspired by NVIDIA PersonaPlex. It lets you speak naturally to an AI that listens and responds in real-time, supports interruptions, backchannels ("uh-huh", "go on"), injects context mid-conversation, and outputs structured JSON for agentic tasks — all powered entirely by Groq's free-tier APIs.

The Core Goal
A black box you speak into. It speaks back. It understands, reasons, acts, and never blocks. Everything — STT, LLM, TTS, backchannels, agentic output — runs inside it. You inject text in. You get voice and JSON out.

What VoxCore Does That PersonaPlex Doesn't
Feature	PersonaPlex (NVIDIA)	VoxCore (Ours)
Full-duplex conversation	✅ Native (joint model)	✅ Simulated (VAD + interrupt)
Backchanneling	✅ Native	✅ Simulated (parallel track)
Persona via text prompt	✅	✅
Voice cloning	✅	❌ (fixed voice for now)
Agentic JSON output	❌	✅
Text-in injection	❌	✅
Tool calling / agent tasks	❌	✅
Memory across sessions	❌	✅
Free to run	❌ (7B GPU required)	✅ (Groq free tier + local Kokoro TTS)
Emotional TTS tags	Native	Via punctuation & voice style (Kokoro)


 2. Your Groq Resources & How We Use Each
You have all the components needed across Groq's three service categories. Here is exactly what each model does in VoxCore:

2.1 Speech-to-Text (STT)
Model	Role in VoxCore	Why This One
whisper-large-v3-turbo	PRIMARY — Real-time streaming STT	Faster than v3, same quality. 20 RPM / 7.2K sec/hr limit is fine for voice.
whisper-large-v3	FALLBACK — if turbo hits rate limit	Higher accuracy for important segments
HuggingFace faster-whisper (small)	LOCAL fallback / dev testing	Zero API cost, no rate limits, runs on CPU

Strategy: Default to whisper-large-v3-turbo on Groq. If rate limit hit, fall back to your HuggingFace self-hosted faster-whisper small model. This gives you unlimited fallback at zero cost.

2.2 Language Models (LLM)
Model	Role in VoxCore	Why / When
llama-3.1-8b-instant	FAST BRAIN — Primary conversational LLM	Fastest on Groq. 6K TPM. Used for all real-time voice responses. Sub-200ms TTFT.
meta-llama/llama-4-scout-17b-16e-instruct	SMART BRAIN — Complex reasoning	30K TPM. Used when fast brain routes complex queries out. Better reasoning.
moonshotai/kimi-k2-instruct	AGENTIC BRAIN — Tool use / long tasks	10K req/day, 300K TPD. Best for multi-step agentic workflows.
qwen/qwen3-32b	DEEP BRAIN — Final synthesis / summaries	6K TPM, 500K TPD. Used for summarizing memory or complex document analysis.
openai/gpt-oss-120b	POWER BRAIN — Max intelligence tasks	8K TPD. Reserved for the hardest tasks only. Rate limited so used sparingly.
meta-llama/llama-guard-4-12b	SAFETY LAYER — Input/output filtering	Runs async on every user utterance. Flags harmful content before LLM sees it.

LLM Routing Strategy
The Fast Brain (llama-3.1-8b) handles ALL real-time speech responses — it's the only model in the hot path. When it detects a task in its output (<agent> tag), it routes to Kimi-K2 for tool execution. Complex reasoning goes to Scout-17B. The user never waits for the slower models — they run asynchronously and inject results back via text-in.

2.3 Text-to-Speech (TTS) — Local Kokoro-ONNX
Engine	Role in VoxCore	Details
Kokoro-ONNX (local)	PRIMARY TTS — All speech output	Free, no API key, runs locally on CPU. 24kHz WAV output. 28 voices available. Model: kokoro-v1.0.onnx + voices-v1.0.bin

Kokoro-ONNX is a local ONNX-based TTS engine. Unlike cloud APIs, it runs entirely on your machine with zero cost and no rate limits. It does NOT use bracket-style emotion tags ([cheerful] etc.) — instead, it infers prosody naturally from punctuation, phrasing, and the selected voice style. The LLM still generates emotion tags for analytics/logging, but TTSClient strips them before synthesis.

Available voices: af_heart (default), af_bella, af_nicole, af_sarah, af_sky, am_adam, am_michael, bf_emma, bm_george, and 19 more.
Download models from: https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files


 3. Full System Architecture
3.1 The Black Box — Data Flow
Everything below is what happens inside VoxCore from the moment you speak to the moment it responds:

INPUTS
  🎙  Microphone  ──►  [Continuous audio stream, 16kHz mono PCM]
  📝  Text-In     ──►  [Context / notifications / tool results / RAG]
  📄  Sys Prompt  ──►  [Persona definition, set once at startup]

PIPELINE
  [VAD/Silero]  ──►  [Groq whisper-turbo STT]  ──►  [Context Buffer]
        │                                              │
  [Backchannel]            text-in ──────────────────►│
  [Detector]                                          │
        │                                             ▼
  play clip ◄───────────────────── [llama-3.1-8b-instant / Groq]
                                         ↙            ↘
                             <speech text>      <agent JSON>
                                  │                   │
                        [Kokoro-ONNX TTS]      [Tool Router]

OUTPUTS
  🔊  Voice-Out   ◄──  [Streaming audio, sentence-by-sentence]
  📤  Text-Out    ◄──  [JSON tool calls, agentic task requests]

3.2 The Turn State Machine
VoxCore is always in one of four states. Transitions are driven by VAD events and LLM output:

State	What's Happening	Exit Condition
LISTENING	VAD running, STT accumulating partial transcript, backchannel detector active	VAD detects end of user speech (400ms silence)
THINKING	STT finalizes. LLM streaming starts. TTS buffer warming up.	First sentence token arrives from LLM
SPEAKING	TTS playing audio. LLM still streaming. VAD still running in background.	LLM finishes OR user interrupts
INTERRUPTED	TTS killed. Partial LLM output saved. User speech captured.	New STT finalized → back to THINKING

3.3 Backchannel System (Simulated Full-Duplex)
This is how we fake the "uh-huh" behavior from PersonaPlex without a joint model. It runs on a completely separate async track from the main pipeline and never blocks it:

•	Cue Detector: A secondary audio analysis loop watches the user mic stream continuously
•	Trigger Condition: When energy drops for 300-400ms mid-sentence (natural phrase boundary), it fires a backchannel event
•	Selector: The Selector picks a context-appropriate clip based on conversation state — e.g. "mm-hmm" vs "oh interesting" vs "go on"
•	Clips Bank: Pre-generated WAV clips using Kokoro-ONNX with your chosen persona voice, stored locally per-voice in backchannel/clips/<voice_shortname>/. Zero API calls needed at runtime.
•	Guard Conditions: Clips only fire if: agent is NOT speaking, user has been talking > 2 seconds, last backchannel was > 8 seconds ago

Backchannel Clip Generation (One-Time Setup)
Run the clip generator script once at startup/config time. It uses local Kokoro-ONNX to generate 15-20 short clips ("uh-huh", "yeah", "I see", "right", "okay go on", "interesting", "mm-hmm") in your chosen persona voice and saves them as WAV files in backchannel/clips/<voice_shortname>/. Completely free, no API calls, no rate limits.

3.4 Emotion-Aware TTS
VoxCore's LLM prompt instructs the fast brain to prefix emotional context to its speech output. Example LLM output:

[concerned] I see what you mean, that does sound frustrating. [calm] Let me check that for you right now.

The TTSClient strips these bracket tags before passing text to Kokoro-ONNX (Kokoro does not use inline emotion tags). Kokoro infers prosody naturally from punctuation and voice style. The tags are still useful for logging, analytics, and the EmotionTagger fallback system.


 4. Complete Folder Structure

voxcore/
│
├── main.py                         # Entry point — boots all async tasks
├── config.yaml                     # Persona, model choices, API keys
├── requirements.txt                # All Python dependencies
├── .env                            # GROQ_API_KEY, HF_API_KEY
│
├── core/
│   ├── __init__.py
│   ├── session.py                  # Master state: history, context buffer, turn count
│   ├── event_bus.py                # AsyncIO pub/sub — connects all layers
│   └── turn_manager.py             # State machine: LISTENING/THINKING/SPEAKING/INTERRUPTED
│
├── input/
│   ├── __init__.py
│   ├── mic_stream.py               # Continuous mic capture (sounddevice, 16kHz)
│   ├── vad.py                      # Silero VAD wrapper — fires speech_start/speech_end
│   ├── stt.py                      # Groq Whisper-turbo STT (with HF fallback)
│   ├── text_injector.py            # text-in API — injects context into buffer
│   └── interruption_detector.py    # Monitors VAD while SPEAKING → fires interrupt
│
├── backchannel/
│   ├── __init__.py
│   ├── cue_detector.py             # Phrase boundary detection (energy + pause)
│   ├── selector.py                 # Picks clip by context (agreement/surprise/filler)
│   ├── generator.py                # One-time Kokoro-ONNX clip generation script
│   └── clips/                      # Pre-generated WAV files (per-voice subdirs)
│       └── heart/                  # Clips for af_heart voice
│           ├── uh_huh.wav
│           ├── yeah.wav
│           ├── i_see.wav
│           ├── go_on.wav
│           ├── interesting.wav
│           ├── right.wav
│           └── mm_hmm.wav
│
├── brain/
│   ├── __init__.py
│   ├── llm_client.py               # Groq async streaming LLM wrapper
│   ├── prompt_builder.py            # Builds: system_prompt + memory + text-in + transcript
│   ├── response_parser.py           # Splits stream → <speech> tokens + <agent> JSON
│   ├── emotion_tagger.py            # Ensures emotion tags in speech output
│   └── router.py                   # Decides: fast brain / smart brain / agentic brain
│
├── output/
│   ├── __init__.py
│   ├── tts_client.py               # Kokoro-ONNX local TTS client
│   ├── audio_player.py             # Streams WAV chunks to speaker (interruptible)
│   └── voice_profile.py            # Holds Kokoro voice ID, emotion defaults, sample rate
│
├── agent/
│   ├── __init__.py
│   ├── text_out.py                 # Emits structured JSON to external consumers
│   ├── tool_router.py              # Routes tool calls to handlers
│   ├── slow_llm.py                 # Calls Kimi-K2 / Scout-17B for complex tasks
│   └── tools/
│       ├── __init__.py
│       ├── web_search.py           # Web search tool
│       ├── memory_tool.py          # Read/write long-term memory
│       ├── calendar_tool.py        # Calendar integration
│       └── base_tool.py            # Abstract base class for tools
│
├── memory/
│   ├── __init__.py
│   ├── short_term.py               # Rolling window of last 20 turns (in-memory)
│   ├── long_term.py                # ChromaDB persistent memory across sessions
│   └── compressor.py               # Summarizes old turns using Qwen3-32B
│
├── safety/
│   ├── __init__.py
│   └── guard.py                    # Async Llama-Guard-4-12B input/output filtering
│
└── interfaces/
    ├── __init__.py
    ├── cli.py                      # Terminal debug interface
    ├── websocket_server.py         # WebSocket server for web/mobile clients
    └── api.py                      # FastAPI REST — text-in injection + text-out webhook


 5. Every File — What It Does & What's Inside
core/ — The Brain Stem
core/session.py
Holds the single source of truth for the entire conversation. Contains: the rolling short-term history (last 20 turns), the text-in injection queue (a deque that gets flushed into every prompt), current persona system prompt, session metadata (start time, turn count), and the current TurnState enum. Every other module reads from or writes to the session.
core/event_bus.py
A lightweight async pub/sub system built on asyncio.Queue. All inter-module communication goes through here. Events: SPEECH_START, SPEECH_END, TRANSCRIPT_READY, LLM_SPEECH_TOKEN, LLM_AGENT_TAG, TTS_CHUNK_READY, INTERRUPT_DETECTED, BACKCHANNEL_FIRE, TEXT_INJECTED, TOOL_RESULT_READY. This means zero tight coupling between modules — any module can subscribe to any event without knowing who published it.
core/turn_manager.py
The 4-state machine (LISTENING / THINKING / SPEAKING / INTERRUPTED). Subscribes to VAD events and LLM events. On SPEECH_END fires THINKING. On first LLM token fires SPEAKING. On INTERRUPT_DETECTED fires INTERRUPTED, cancels the TTS audio player task, saves partial LLM output to session, and transitions back to THINKING with the interruption appended to history.

input/ — Ears
input/mic_stream.py
Opens a sounddevice input stream at 16kHz, mono, int16. Produces 30ms audio chunks continuously into a queue. Never blocks. On startup, does a 0.5s silence check to set noise floor for VAD calibration.
input/vad.py
Wraps Silero VAD (pysilero-vad package). Consumes 30ms chunks from mic_stream. Fires SPEECH_START when speech probability > 0.5 and SPEECH_END when silence probability > 0.7 for 400ms. Also maintains a 3-second audio buffer of the last speech segment for STT submission.
input/stt.py
On SPEECH_END event, takes the buffered audio, creates a temporary WAV file, and sends to Groq whisper-large-v3-turbo via the Groq SDK. Returns transcript text. If Groq rate limit is hit (429), automatically falls back to your HuggingFace faster-whisper-small API endpoint. Fires TRANSCRIPT_READY event with the final text.
input/text_injector.py
Exposes a simple inject(text, priority='normal') method. Adds text to a queue in the session's context buffer. The prompt_builder reads this queue on every LLM call and prepends it as a [CONTEXT] block before the user message. High-priority injections (tool results, alerts) are prepended before normal context. This is how external systems, notifications, tool results, and RAG data get into the conversation without the user saying anything.
input/interruption_detector.py
Runs a parallel VAD check specifically during the SPEAKING state. If speech probability > 0.6 for more than 300ms while audio is playing, fires INTERRUPT_DETECTED. Does NOT send to STT yet — just kills the audio. The main STT loop picks up the interrupted speech normally on SPEECH_END.

backchannel/ — The Naturalness Layer
backchannel/cue_detector.py
Runs a secondary energy analysis on the raw mic stream (not STT transcripts — this runs before STT for speed). Uses librosa to detect: RMS energy drops below threshold mid-utterance (phrase boundary), intonation fall patterns, pauses of 300-600ms that are not end-of-turn. When these patterns appear during LISTENING state and enough time has passed since the last backchannel, fires a BACKCHANNEL_OPPORTUNITY event with context (is_question, energy_level, speech_duration).
backchannel/selector.py
Subscribes to BACKCHANNEL_OPPORTUNITY. Uses a weighted random selection to pick from the clips bank based on context: after a question → "mm-hmm" or "right", after surprising info → "oh interesting" or "wow", during long monologue → "yeah" or "go on", general filler → "uh-huh" or "I see". Guards: minimum 8s between backchannels, never during SPEAKING or THINKING, never in first 2 seconds of user speech.
backchannel/generator.py
A one-time setup script. Uses local Kokoro-ONNX with the configured persona voice to generate all 15-20 backchannel clips. Saves them as WAV files in backchannel/clips/<voice_shortname>/ (e.g. clips/heart/ for af_heart). Re-run if you change the persona voice. This script is not part of the runtime — it just populates the clips directory. Free, no API calls, no rate limits.

brain/ — The Mind
brain/llm_client.py
Async wrapper around the Groq Python SDK. Handles streaming completions. Exposes a stream_response(messages, model) async generator that yields tokens as they arrive. Manages retry logic and rate limit backoff. Tracks token usage per call and per session to avoid hitting daily limits.
brain/prompt_builder.py
Assembles the full message array for every LLM call. Structure: (1) System message with persona, current datetime, instructions for emotion tags and agent output format. (2) Compressed old memory summary if history > 20 turns. (3) Last 20 turns of conversation history. (4) Any queued text-in injections as [CONTEXT] blocks. (5) Current user message. Keeps total tokens under 5K for llama-3.1-8b to maintain speed.
brain/response_parser.py
Consumes the streaming token output from llm_client. Maintains two output buffers simultaneously. Speech buffer: accumulates tokens for TTS, fires LLM_SPEECH_TOKEN events sentence-by-sentence (split on . ? !). Agent buffer: watches for <agent> opening tag, captures everything until </agent>, parses as JSON, fires LLM_AGENT_TAG event. Both buffers are flushed in real-time — TTS starts before the LLM finishes.
brain/emotion_tagger.py
A lightweight post-processor on speech tokens. If the LLM output does not contain an emotion tag in the first sentence, infers one based on content keywords and appends it. Example: "I'm sorry to hear that" → prepend "[empathetic]". Note: Kokoro-ONNX does not process these tags — TTSClient strips them before synthesis. The tags are still useful for analytics and logging.
brain/router.py
Decides which LLM to use based on the detected task type. Conversational reply → llama-3.1-8b-instant (always). Tool execution → kimi-k2-instruct (via agent/slow_llm.py, async). Multi-step reasoning → llama-4-scout-17b. Long document analysis → qwen3-32b. Maximum intelligence → gpt-oss-120b (rate-limited, reserved). The fast brain always responds first with a brief spoken acknowledgment; heavy tasks run in background.

output/ — The Voice
output/tts_client.py
Local Kokoro-ONNX TTS client. On each LLM_SPEECH_TOKEN event (one sentence at a time), strips any bracket emotion tags with regex, then synthesizes via Kokoro locally. Uses asyncio.run_in_executor() for CPU-bound synthesis to avoid blocking the event loop. Outputs 24kHz WAV audio. No API calls, no rate limits, no network latency.
output/audio_player.py
An asyncio-driven audio player using sounddevice output stream. Consumes WAV chunks from tts_client as they arrive. Maintains a play queue. Critical: subscribes to INTERRUPT_DETECTED events — on interrupt, immediately empties the play queue and cancels the current chunk. Does NOT kill mid-word; waits for the current 20ms audio frame to finish for clean cutoff.
output/voice_profile.py
Simple config holder: chosen Kokoro voice ID (e.g. "af_heart"), default emotion for LLM prompts, sample rate (24kHz for Kokoro-ONNX), response format ("wav"), model/voices file paths. Load from config.yaml at startup.

agent/ — The Hands
agent/text_out.py
Subscribes to LLM_AGENT_TAG events. Validates the JSON, adds metadata (timestamp, session_id, turn_id), and emits it to all registered text-out consumers. Consumers can be: WebSocket clients, REST webhook endpoints, internal tool_router, or file log. This is the agentic output channel — external systems connect here to receive structured task requests from VoxCore.
agent/tool_router.py
Receives validated agent JSON from text_out. Reads the "action" field and routes to the appropriate tool handler in agent/tools/. After tool execution, injects the result back into the conversation via text_injector.inject(result, priority='high'). The fast brain will pick it up in the next turn and incorporate it naturally into the spoken response.
agent/slow_llm.py
For tasks that need more intelligence than llama-3.1-8b. Called asynchronously by tool_router when the agent tag specifies a complex reasoning task. Uses kimi-k2-instruct for tool-use tasks (it has the best tool-calling ability in your roster) or scout-17b for reasoning. Always runs in the background — the fast brain already said "let me look into that" before slow_llm even starts.
agent/tools/
Each tool is a class extending BaseTool with an execute(params) async method. web_search.py uses a search API. memory_tool.py reads/writes ChromaDB. calendar_tool.py integrates with Google Calendar or a local SQLite store. base_tool.py defines the interface. Adding a new tool means creating one file and registering it in tool_router.py.

memory/ — The Long-Term
memory/short_term.py
A deque of the last 20 conversation turns stored in RAM. Each entry: {role, content, timestamp, emotion, turn_id}. Cleared on session end but persisted to long-term if configured. Provides the immediate conversational context to every LLM call.
memory/long_term.py
ChromaDB vector store. On session end (or every N turns), encodes conversation summary and key facts into embeddings and stores them. On session start, queries for relevant past context based on the initial topic. Inject retrieved memories into prompt via text_injector at session start. This gives VoxCore persistent memory across conversations.
memory/compressor.py
When short-term history exceeds 20 turns, calls qwen3-32b to produce a concise summary of the oldest 10 turns. Replaces those 10 turns with the summary in the prompt. This keeps the fast brain's context window under 5K tokens permanently while preserving conversational continuity.

safety/guard.py
Calls llama-guard-4-12b asynchronously on every user transcript before it reaches the LLM. Also runs on every LLM speech output before it reaches TTS. On unsafe classification, replaces the response with a safe refusal template and logs the incident. Runs in parallel — does not add latency to the hot path because it fires at the same time as the LLM, not before it. If guard flags the LLM output, the TTS queue is flushed before the unsafe audio plays.

interfaces/ — The Ports
interfaces/websocket_server.py
FastAPI + WebSockets. Streams audio from a web browser or mobile app to mic_stream, and streams TTS audio and text-out JSON back to the client. Enables VoxCore to run as a server with a remote frontend. Also handles text-in injection via WebSocket messages.
interfaces/api.py
FastAPI REST endpoints. POST /inject — adds text to the context buffer. GET /transcript — returns current session transcript. GET /status — returns current TurnState. POST /config — hot-reload persona or voice. WebSocket /text-out — stream of agent JSON for external system consumption.
interfaces/cli.py
Terminal debug interface. Shows live VAD state, STT transcript, LLM tokens, TTS status, and agent JSON output in real-time. Allows manual text-in injection by typing. Used for development and testing.


 6. config.yaml — Full Schema

persona:
  name: "Aria"
  system_prompt: |
    You are Aria, a warm, focused, and capable assistant.
    Keep all spoken responses under 3 sentences unless asked for more.
    Always start responses with an emotion tag: [cheerful] [calm] [concerned] [excited] [empathetic]
    When you need to perform a task, include ONLY in your output:
    <agent>{"action": "tool_name", "params": {...}}</agent>
    Do NOT describe the tool call in your speech. Just say 'On it' or 'Let me check'.
  voice: "af_heart"                # Kokoro voice ID (af_heart, af_bella, am_adam, etc.)
  language: "en"

models:
  stt_primary: "whisper-large-v3-turbo"
  stt_fallback: "https://your-hf-space.hf.space/transcribe"   # Your HF endpoint
  llm_fast: "llama-3.1-8b-instant"
  llm_smart: "meta-llama/llama-4-scout-17b-16e-instruct"
  llm_agentic: "moonshotai/kimi-k2-instruct"
  llm_deep: "qwen/qwen3-32b"
  llm_power: "openai/gpt-oss-120b"
  tts_model: "models/kokoro-v1.0.onnx"       # Local Kokoro-ONNX model
  tts_voices: "models/voices-v1.0.bin"        # Kokoro voice embeddings
  safety: "meta-llama/llama-guard-4-12b"

audio:
  sample_rate: 16000              # STT input sample rate
  chunk_ms: 30                    # VAD chunk size
  vad_speech_threshold: 0.5
  vad_silence_threshold: 0.7
  vad_silence_duration_ms: 400    # How long silence before end-of-turn

backchannel:
  enabled: true
  min_pause_ms: 350               # Phrase boundary detection sensitivity
  min_gap_between_s: 8            # Minimum seconds between backchannels
  min_user_speech_s: 2            # User must speak this long before backchannel fires
  clips_dir: "backchannel/clips/heart/"   # Per-voice subdirectory

memory:
  short_term_turns: 20
  compress_after_turns: 20
  long_term_enabled: true
  long_term_db: "./data/memory.db"   # ChromaDB path

agent:
  enabled: true
  text_out_webhook: null           # Optional: POST results here
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


 7. Dependencies & Installation

Package	Version	Purpose
groq	latest	Groq SDK — STT + LLM (not TTS)
kokoro-onnx	0.4+	Local ONNX-based TTS engine (free, no API key)
soundfile	0.12+	WAV file I/O for Kokoro TTS output
sounddevice	0.4.x	Mic capture and audio playback
silero-vad / pysilero-vad	latest	Voice activity detection (CPU, <1ms/frame)
librosa	0.10.x	Audio analysis for backchannel cue detection
chromadb	0.5.x	Long-term vector memory store
fastapi	0.110+	REST API and WebSocket server
uvicorn	latest	ASGI server for FastAPI
pyyaml	6.x	config.yaml parsing
python-dotenv	1.x	Load .env secrets
numpy	1.x	Audio array manipulation
scipy	1.x	WAV file I/O
asyncio	stdlib	Async everything
aiohttp	3.x	Async HTTP for tool calls
torch	2.x (CPU)	Required for Silero VAD

Installation
git clone https://github.com/yourname/voxcore
cd voxcore && pip install -r requirements.txt
cp .env.example .env  # Add your GROQ_API_KEY
python -m backchannel.generator   # One-time: generate backchannel clips
python main.py                    # Launch VoxCore


 8. Latency Budget Analysis
Target: User stops speaking → first audio word playing in under 700ms. Here is the breakdown:

Step	Component	Expected Latency
VAD end-of-turn detection	Silero VAD (local)	~400ms (silence window)
STT transcription	Groq whisper-turbo	~200ms (Groq is fast)
LLM first token	llama-3.1-8b-instant on Groq	~100-200ms TTFT
First sentence buffer	response_parser.py	~50ms (5-8 words)
TTS synthesis	Kokoro-ONNX (local)	~50-100ms (no network)
Audio playback start	sounddevice	~10ms
TOTAL end-to-end	Full pipeline	~910-1060ms

This is competitive with cascade systems (~800ms-1.2s is industry standard). To get below 700ms: reduce silence window to 300ms (more aggressive cutoff), or use a local faster-whisper small model for STT (saves ~150ms at cost of some accuracy).


 9. Agentic Flow — How Tools Work
This is the flow when VoxCore needs to perform a task rather than just talk:

Step	What Happens	Model Used
1. User speaks	"What's the weather like in London?"	STT: whisper-turbo
2. Fast brain responds	Speaks: "[curious] Sure, let me check that for you." Outputs: <agent>{"action": "web_search", "query": "London weather today"}</agent>	llama-3.1-8b-instant
3. TTS plays acknowledgment	User hears response immediately, no waiting	Kokoro-ONNX TTS (local)
4. Tool executes async	tool_router.py calls web_search.py in background	N/A
5. Result injected	text_injector.inject("London: 12°C, cloudy, light rain", priority='high')	N/A
6. Next turn picks it up	Fast brain gets the context and speaks the result naturally	llama-3.1-8b-instant
7. text-out emitted	JSON emitted to external consumers if any	N/A

Key Insight
The user never waits for the tool to complete. The agent already acknowledged and started speaking BEFORE the tool ran. The result comes in the very next natural turn. This is why the fast brain must always produce a spoken acknowledgment alongside every agent tag.


 10. What to Build First (Recommended Order)
Hand this document to your coding agent with this build order:

Phase	Build This	Goal
1 — Core Loop	mic_stream + vad + stt + llm_client + tts_client + audio_player	Get basic voice → voice working end-to-end
2 — State Machine	session + event_bus + turn_manager + response_parser	Add interruption handling and clean turn-taking
3 — Text I/O	text_injector + text_out + prompt_builder	Enable text-in context injection and JSON output
4 — Naturalness	backchannel (generator + cue_detector + selector) + emotion_tagger	Add backchannels and emotional TTS
5 — Memory	short_term + long_term + compressor	Add persistent memory and context compression
6 — Agent	tool_router + slow_llm + tools/	Add tool execution and agentic capabilities
7 — Safety	guard.py	Add Llama-Guard async safety checking
8 — Interfaces	cli + websocket_server + api	Expose to web clients and external systems


VoxCore Architecture Document — February 2026
