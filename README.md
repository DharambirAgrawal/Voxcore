# VoxCore

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Status](https://img.shields.io/badge/status-active--development-orange)

A full-duplex voice AI pipeline that listens and speaks at the same time — handling interruptions, backchanneling, and agentic tool calls over an async, multi-model architecture.

## Overview

VoxCore is a voice-first conversational AI system, loosely inspired by NVIDIA's PersonaPlex. Rather than the usual turn-based "record → transcribe → respond → play" loop, it runs speech capture, transcription, LLM reasoning, and speech synthesis as concurrent asyncio tasks connected by a shared event bus — so the assistant can be interrupted mid-sentence, react to what you're saying while it's still talking, and inject external context (tool results, notifications) into the conversation without stopping to ask.

It's built entirely on Groq's free-tier APIs for STT/LLM/safety, plus a local ONNX text-to-speech engine, so it runs at low latency without a GPU or a paid API bill.

## Features

- **Full-duplex conversation loop** — voice activity detection (Silero VAD) and an event-driven turn state machine (`LISTENING` → `THINKING` → `SPEAKING` → `INTERRUPTED`) simulate simultaneous listening and speaking instead of strict turn-taking.
- **Multi-gate interrupt classification** — layered gates (energy burst detection, filler-word filtering, and an LLM-based semantic classifier) decide whether a sound during playback is a real interruption, a backchannel ("uh-huh"), or noise, before deciding whether to pause or keep talking.
- **Echo suppression** — a multi-layer pipeline (speech enhancement, speaker-embedding similarity via `resemblyzer`, and spectral/temporal gating) to stop the assistant from hearing and reacting to its own TTS output through the mic.
- **Backchanneling** — a separate async track detects natural phrase pauses in the user's speech and plays short pre-generated audio cues ("mm-hmm", "go on") without interrupting the main pipeline.
- **Agentic tool calls** — the LLM emits structured `<agent>{"action": ..., "params": ...}</agent>` tags to invoke tools (web search, calendar, memory) whose results are injected back into the live conversation.
- **Multi-model brain routing** — different Groq-hosted models are assigned to different jobs: a fast model handles real-time replies, larger models are reserved for complex reasoning, tool use, and deep summarization.
- **Persistent memory** — short-term rolling context plus long-term semantic memory backed by ChromaDB, with periodic LLM-driven compression of conversation history.
- **Safety filtering** — input and output are checked against Llama Guard before being spoken or acted on.
- **Local TTS** — speech synthesis runs on-device via Kokoro-ONNX (no API key, no per-request cost), with emotion tags parsed from LLM output for logging/analytics.
- **CLI and network interfaces** — a colored terminal debug interface, plus optional FastAPI REST + WebSocket servers for remote clients.

## Tech Stack

- **Language:** Python 3.11+ (asyncio throughout)
- **STT / LLM / Safety:** [Groq API](https://groq.com/) — Whisper (`whisper-large-v3-turbo`), multiple Llama/Qwen/GPT-OSS models for routed reasoning, Llama Guard for safety
- **TTS:** [Kokoro-ONNX](https://github.com/thewh1teagle/kokoro-onnx) (local, ONNX runtime)
- **VAD:** Silero VAD via `torch` / `torchaudio` (CPU-only)
- **Echo suppression:** `resemblyzer` (speaker embeddings), `denoiser` (Facebook Research speech enhancement)
- **Audio I/O & analysis:** `sounddevice`, `soundfile`, `numpy`, `scipy`, `librosa`
- **Memory:** `chromadb` for vector-based long-term memory
- **Networking:** `fastapi`, `uvicorn`, `websockets`
- **Config:** YAML-based single source of truth (`config.yaml`), `python-dotenv` for secrets

## Getting Started

### Prerequisites

- Python 3.11+
- A [Groq API key](https://console.groq.com/keys) (free tier)
- Kokoro-ONNX model files (`kokoro-v1.0.onnx`, `voices-v1.0.bin`) — download from the [Kokoro-ONNX releases page](https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files) and place them under `models/` (gitignored, not bundled in this repo)

### Installation

```bash
git clone https://github.com/DharambirAgrawal/Voxcore.git
cd Voxcore
pip install -r requirements.txt
```

Set up your environment variables:

```bash
cp .env.example .env
# then edit .env and set GROQ_API_KEY (required)
```

Generate the backchannel audio clips (one-time setup, uses local Kokoro TTS):

```bash
python -m backchannel.generator
```

## Usage

Run the assistant with the terminal interface (default):

```bash
python main.py
```

Other run modes:

```bash
python main.py --interface server        # FastAPI REST + WebSocket server only
python main.py --interface both           # CLI + server
python main.py --no-mic --interface cli   # Text-only mode, no microphone required
python main.py --config custom.yaml --log-level DEBUG
```

All behavior — persona, model selection, audio thresholds, memory, safety, and interrupt tuning — is controlled from `config.yaml`.

## How It Works

VoxCore's pipeline is built around an `EventBus` that decouples its modules: microphone capture, VAD, STT, the LLM brain, TTS, and interrupt/backchannel detection all run as independent asyncio tasks that publish and subscribe to events rather than calling each other directly.

Conversation state is tracked as a small state machine — `LISTENING`, `THINKING`, `SPEAKING`, `INTERRUPTED` — driven by VAD and LLM events. While the assistant is speaking, a separate "speaking monitor" continuously classifies incoming audio through multiple gates (energy burst detection, then filler-word filtering, then an LLM-based semantic check) to decide whether the user is backchanneling, genuinely interrupting, or just making noise — and only then does it pause or reroute the response. Because the mic keeps listening while the TTS is playing, a dedicated echo-suppression stack (speaker-embedding similarity + spectral/temporal gating) is used to stop the assistant from mistaking its own voice for user input.

## Documentation

Further design notes and architecture deep-dives live in [`docs/`](./docs), including [`ARCHITECTURE.md`](./docs/ARCHITECTURE.md) and [`DOCUMENTATION.md`](./docs/DOCUMENTATION.md).

## Status

This is an actively evolving personal project (currently on its third architectural iteration, per `docs/`). Some features — such as the HuggingFace STT fallback and the web search tool — expect additional self-hosted or third-party API keys not included in `.env.example` by default.
