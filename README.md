# VoxCore: Full-Duplex Conversational AI System

**VoxCore** is a voice-first, full-duplex conversational AI system inspired by NVIDIA PersonaPlex. It allows users to speak naturally to an AI that listens and responds in real-time, supports interruptions, backchannels, injects context mid-conversation, and outputs structured JSON for agentic tasks. All of this runs entirely on Groq's free-tier APIs.

## 🌟 The Core Goal
A black box you speak into. It speaks back. It understands, reasons, acts, and never blocks. Everything from Speech-to-Text (STT), Language Model (LLM) reasoning, Text-to-Speech (TTS), backchanneling, and agentic output runs concurrently inside it. You inject text in; you get voice and JSON out.

## ✨ Features

- **Full-duplex conversation**: Simulated via VAD + interrupt handling.
- **Backchanneling**: Simulated natural conversational cues (e.g., "uh-huh", "go on") via a parallel track.
- **Agentic JSON output**: Emits JSON tool calls dynamically based on voice requests.
- **Text-in injection**: Inject real-time external data (e.g., tool results, notifications) effortlessly into the ongoing conversation.
- **Tool calling / agent tasks**: LLM is equipped to trigger async tool execution.
- **Memory across sessions**: Powered by ChromaDB for persistent, long-term memory.
- **Free to run**: Leverages Groq’s free-tier models natively.
- **Emotional TTS tags**: Expressive speech via prompt emotion tagging (e.g., `[cheerful]`, `[sad]`).

## 🧠 Brain Routing Strategy

VoxCore uses a multi-LLM architecture:
1. **Primary Conversational LLM (Fast Brain):** `llama-3.1-8b-instant`. Handles ALL real-time speech responses.
2. **Complex Reasoning (Smart Brain):** `meta-llama/llama-4-scout-17b-16e-instruct`.
3. **Agentic/Tool Use (Agentic Brain):** `moonshotai/kimi-k2-instruct`.
4. **Deep Synthesis (Deep Brain):** `qwen/qwen3-32b`.
5. **Max Intelligence (Power Brain):** `openai/gpt-oss-120b`.
6. **Safety Filtering:** `meta-llama/llama-guard-4-12b`.

## ⚙️ Installation & Setup

1. Clone the repository and navigate into the project directory:
   ```bash
   git clone https://github.com/your-username/voxcore.git
   cd voxcore
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Setup environment variables:
   Create a `.env` file in the root based on `.env.example` (or set the variables manually):
   ```env
   GROQ_API_KEY=your_groq_api_key
   ```

4. Generate backchannel clips (One-time setup):
   ```bash
   python -m backchannel.generator
   ```

5. Launch VoxCore:
   ```bash
   python main.py
   ```

## 📖 Documentation
Detailed architectural designs, data flows, and module responsibilities can be found in the [docs/DOCUMENTATION.md](./docs/DOCUMENTATION.md) and [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) files.
