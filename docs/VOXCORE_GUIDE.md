# VoxCore — Complete System Guide

> Full-Duplex Conversational AI | PersonaPlex-Inspired Architecture  
> Groq APIs + Local Kokoro-ONNX TTS | February 2026

---

## Table of Contents

1. [What is VoxCore?](#1-what-is-voxcore)
2. [Architecture Overview](#2-architecture-overview)
3. [Memory System — How It Works](#3-memory-system)
4. [Tool & Agent System](#4-tool--agent-system)
5. [Connecting MCP Servers](#5-connecting-mcp-servers)
6. [Adding Custom Tools](#6-adding-custom-tools)
7. [Configuration Reference](#7-configuration-reference)
8. [API & Integration](#8-api--integration)
9. [How the Pipeline Works Step-by-Step](#9-pipeline-walkthrough)
10. [Extending VoxCore](#10-extending-voxcore)

---

## 1. What is VoxCore?

VoxCore is a **voice-first, full-duplex conversational AI** system. You speak to it, it speaks back — in real-time, with interruption support, backchanneling ("uh-huh", "go on"), and agentic tool execution.

### Core Capabilities

| Feature | How It Works |
|---|---|
| **Full-duplex conversation** | VAD + interrupt detection simulates natural turn-taking |
| **Backchanneling** | Pre-generated clips ("mm-hmm", "right") play during user pauses |
| **Persistent memory** | ChromaDB stores conversations across sessions |
| **Tool execution** | LLM generates `<agent>` tags → tools run async → results injected back |
| **Emotional TTS** | LLM prefixes emotions like `[cheerful]`, Kokoro infers prosody |
| **Safety layer** | Llama Guard 4 filters input/output asynchronously |
| **Text injection** | External systems inject context mid-conversation via API |

### What Powers It

| Component | Technology | Cost |
|---|---|---|
| STT | Groq Whisper Large V3 Turbo | Free tier |
| LLM (fast) | Groq llama-3.1-8b-instant | Free tier |
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
┌─────────────────────────────────────────────────────────────┐
│                        INPUTS                                │
│  🎙 Microphone ──► 16kHz mono PCM audio stream              │
│  📝 Text-In    ──► Context / tool results / notifications    │
│  📄 Sys Prompt  ──► Persona definition (set once at boot)    │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                     PIPELINE                                 │
│                                                              │
│  MicStream → VAD(Silero) → STT(Whisper) → Session           │
│       │                                      │               │
│  Backchannel                   text-in ──────┤               │
│  Detector                                    ▼               │
│       │                           PromptBuilder              │
│  play clip ◄──                        │                      │
│                                       ▼                      │
│                            LLM (llama-3.1-8b)                │
│                              ↙            ↘                  │
│                     <speech text>    <agent JSON>             │
│                         │                 │                   │
│                   TTSClient          ToolRouter               │
│                  (Kokoro-ONNX)            │                   │
│                         │           ┌─────┼──────┐           │
│                   AudioPlayer       │  Tools     │           │
│                                     │ web_search │           │
│                                     │ memory     │           │
│                                     │ calendar   │           │
│                                     │ MCP...     │           │
│                                     └────────────┘           │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                       OUTPUTS                                │
│  🔊 Voice-Out  ◄── Streaming audio (sentence-by-sentence)   │
│  📤 Text-Out   ◄── JSON tool calls / agent events            │
│  💾 Memory     ◄── ChromaDB persistent storage               │
└─────────────────────────────────────────────────────────────┘
```

### Turn State Machine

VoxCore runs a 4-state machine at all times:

```
                    ┌──────────┐
         VAD end    │ LISTENING │ ◄── VAD running, mic active
         of speech  └─────┬────┘
                          ▼
                    ┌──────────┐
         First LLM  │ THINKING │ ◄── STT finalizing, LLM starting
         token       └─────┬────┘
                          ▼
                    ┌──────────┐
         LLM done   │ SPEAKING │ ◄── TTS playing, VAD still monitoring
         or empty    └─────┬────┘
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
        Back to               ┌─────────────┐
        LISTENING             │ INTERRUPTED  │ ◄── User spoke over AI
                              └──────┬──────┘
                                     ▼
                              Back to THINKING
```

### File Structure

```
voxcore/
├── main.py                          # Entry point, boots all async tasks
├── config.yaml                      # All configuration
├── .env                             # GROQ_API_KEY
│
├── core/
│   ├── session.py                   # Master state: history, context, turn state
│   ├── event_bus.py                 # Async pub/sub connecting all modules
│   └── turn_manager.py              # State machine orchestrator
│
├── input/
│   ├── mic_stream.py                # 16kHz mic capture with noise floor
│   ├── vad.py                       # Silero VAD with echo suppression
│   ├── stt.py                       # Groq Whisper STT
│   ├── text_injector.py             # Context injection API
│   └── interruption_detector.py     # Detects user speaking over AI
│
├── brain/
│   ├── llm_client.py                # Groq streaming LLM wrapper
│   ├── prompt_builder.py            # Assembles system + memory + history + context
│   ├── response_parser.py           # Splits LLM output → speech + agent JSON
│   ├── emotion_tagger.py            # Ensures emotion tags in output
│   └── router.py                    # Routes to fast/smart/agentic brain
│
├── output/
│   ├── tts_client.py                # Kokoro-ONNX local TTS
│   ├── audio_player.py              # Speaker output with interrupt support
│   └── voice_profile.py             # Voice configuration
│
├── agent/
│   ├── text_out.py                  # Emits validated agent JSON
│   ├── tool_router.py               # Routes tool calls to handlers
│   ├── slow_llm.py                  # Background calls to powerful models
│   └── tools/
│       ├── base_tool.py             # Abstract base class
│       ├── web_search.py            # Web search (Tavily / DuckDuckGo)
│       ├── memory_tool.py           # Read/write ChromaDB memory
│       └── calendar_tool.py         # Local SQLite calendar
│
├── memory/
│   ├── short_term.py                # Rolling 20-turn in-memory window
│   ├── long_term.py                 # ChromaDB persistent vector store
│   └── compressor.py                # Summarizes old turns via qwen3-32b
│
├── safety/
│   └── guard.py                     # Llama Guard 4 async safety filter
│
├── backchannel/
│   ├── cue_detector.py              # Phrase boundary detection
│   ├── selector.py                  # Context-aware clip selection
│   ├── generator.py                 # One-time clip generation script
│   └── clips/heart/                 # Pre-generated WAV clips
│
└── interfaces/
    ├── cli.py                       # Terminal debug interface
    ├── websocket_server.py          # WebSocket for web/mobile clients
    └── api.py                       # REST API endpoints
```

---

## 3. Memory System

VoxCore has a **three-tier memory architecture** that gives the AI persistent, context-aware recall across sessions.

### 3.1 Short-Term Memory (RAM)

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

### 3.2 Memory Compressor (Summarization)

**File:** `memory/compressor.py`

- Subscribes to `MEMORY_COMPRESS` events
- Batches turns (5-10 at a time)
- Calls `qwen/qwen3-32b` via Groq to produce a concise summary capturing:
  - Key facts (names, preferences, dates)
  - Decisions and action items
  - Emotional context
  - Explicit "remember this" requests
- Publishes `MEMORY_COMPRESSED` with the summary

```
MEMORY_COMPRESS → MemoryCompressor._compress_batch()
                       │
                  qwen/qwen3-32b summarizes
                       │
                  MEMORY_COMPRESSED
                       │
               ┌───────┴───────┐
               ▼               ▼
        Session updates    LongTermMemory
        compressed_summary   stores in ChromaDB
```

### 3.3 Long-Term Memory (ChromaDB)

**File:** `memory/long_term.py`

- ChromaDB persistent vector store at `data/chromadb/`
- Stores summaries from the compressor AND full session transcripts on shutdown
- Semantic similarity search using cosine distance
- Default embedding model: all-MiniLM-L6-v2 (built into ChromaDB)

#### How Memory Gets Saved

There are **three paths** into long-term storage:

| Trigger | What Gets Stored | When |
|---|---|---|
| **Compression overflow** | Summarized batch of old turns | After 25+ turns in a session |
| **Session shutdown** | Full session transcript | Every time VoxCore exits (Ctrl+C) |
| **Explicit memory tool** | User-requested content | When user says "Remember that..." |

#### How Memory Gets Retrieved

On every new session startup:

1. `LongTermMemory.initialize()` connects to ChromaDB
2. Queries for past context: `"user name preferences conversation summary"`
3. Top 3 results are injected into `session.compressed_summary`
4. `PromptBuilder` includes this as a `[MEMORY SUMMARY]` block in every LLM call

```python
# What the LLM sees in its prompt:
[MEMORY SUMMARY]: user: My name is Dharambir.
assistant: Nice to meet you, Dharambir.
user: Can you tell me a joke?
assistant: Here's a joke: Why couldn't the bicycle stand up by itself?
...
```

#### Memory Flow Diagram

```
┌──────────────┐     TURN_COMPLETE      ┌─────────────────┐
│   Session    │ ──────────────────────► │ ShortTermMemory  │
│  (add_turn)  │                         │  (20-turn deque) │
└──────────────┘                         └────────┬────────┘
                                                  │
                                         overflow ≥ 5 turns
                                                  │
                                          MEMORY_COMPRESS
                                                  │
                                                  ▼
                                        ┌─────────────────┐
                                        │ MemoryCompressor │
                                        │ (qwen/qwen3-32b)│
                                        └────────┬────────┘
                                                  │
                                         MEMORY_COMPRESSED
                                                  │
                                    ┌─────────────┼─────────────┐
                                    ▼                           ▼
                            ┌──────────────┐          ┌────────────────┐
                            │   Session    │          │ LongTermMemory │
                            │ (compressed_ │          │  (ChromaDB)    │
                            │  summary)    │          │  store()       │
                            └──────────────┘          └────────────────┘

On shutdown:
┌──────────────┐                          ┌────────────────┐
│   Session    │ ──── save_session() ───► │ LongTermMemory │
│  (history)   │     full transcript      │  (ChromaDB)    │
└──────────────┘                          └────────────────┘

On startup:
┌────────────────┐                        ┌──────────────┐
│ LongTermMemory │ ──── query() ────────► │   Session    │
│  (ChromaDB)    │   top 3 past contexts  │ (compressed_ │
└────────────────┘                        │  summary)    │
                                          └──────────────┘
```

---

## 4. Tool & Agent System

### How Tools Work End-to-End

When VoxCore needs to **do** something (search the web, save a memory, check a calendar), it follows this flow:

```
1. User: "What's the weather in London?"
          │
2. LLM output: "[curious] Let me check that for you.
                <agent>{"action": "web_search", "params": {"query": "London weather"}}</agent>"
          │
3. ResponseParser splits:
   ├── Speech: "Let me check that for you." → TTS → Speaker
   └── Agent JSON: {"action": "web_search", ...} → AGENT_JSON_OUT event
          │
4. TextOut validates JSON → publishes AGENT_JSON_OUT
          │
5. ToolRouter receives event:
   ├── Looks up "web_search" in tool registry
   ├── Calls WebSearchTool.execute({"query": "London weather"})
   └── Runs with timeout (30s default)
          │
6. Tool returns: "London: 12°C, cloudy, light rain expected"
          │
7. TextInjector.inject_tool_result():
   └── Adds to session context queue: "[TOOL RESULT — web_search]: London: 12°C..."
          │
8. Next LLM call picks up the context:
   └── LLM: "[cheerful] It's 12 degrees in London right now, cloudy with light rain."
```

**Key insight:** The user never waits for the tool. The AI says "Let me check" immediately, the tool runs in the background, and the result appears naturally in the next response.

### Built-in Tools

#### 1. Web Search (`web_search`)

**File:** `agent/tools/web_search.py`

```python
# LLM generates:
<agent>{"action": "web_search", "params": {"query": "latest AI news"}}</agent>

# Tool uses Tavily API (if TAVILY_API_KEY set) or DuckDuckGo fallback
# Returns formatted results with titles, snippets, and URLs
```

| Param | Type | Required | Description |
|---|---|---|---|
| `query` | string | Yes | Search query |

#### 2. Memory (`memory`)

**File:** `agent/tools/memory_tool.py`

```python
# Write to memory:
<agent>{"action": "memory", "params": {"operation": "write", "content": "User's birthday is March 15"}}</agent>

# Read from memory:
<agent>{"action": "memory", "params": {"operation": "read", "query": "user birthday"}}</agent>
```

| Param | Type | Required | Description |
|---|---|---|---|
| `operation` | string | Yes | `"read"` or `"write"` |
| `content` | string | For write | Content to store |
| `query` | string | For read | Semantic search query |
| `top_k` | int | No | Number of results (default: 5) |

#### 3. Calendar (`calendar`)

**File:** `agent/tools/calendar_tool.py`

```python
# Create event:
<agent>{"action": "calendar", "params": {"action": "create", "title": "Meeting", "event_time": "2026-02-21T14:00:00"}}</agent>

# List upcoming events:
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

**File:** `agent/slow_llm.py`

For tasks requiring more intelligence than llama-3.1-8b:

| Task Type | Model Used | When |
|---|---|---|
| Tool execution | kimi-k2-instruct | Multi-step agentic workflows |
| Complex reasoning | llama-4-scout-17b | User asks hard questions |
| Document analysis | qwen/qwen3-32b | Long text summarization |
| Maximum intelligence | gpt-oss-120b | Hardest tasks (rate-limited) |

The fast brain always responds first. Slow models run async and inject results back.

---

## 5. Connecting MCP Servers

**Model Context Protocol (MCP)** is the industry-standard way to connect AI systems to external tools — databases, APIs, file systems, cloud services, and more. VoxCore's tool architecture is designed to integrate with MCP servers seamlessly.

### What is MCP?

MCP servers expose tools as JSON-RPC endpoints. Instead of building custom integrations for every service, you connect to an MCP server and its tools become available.

Examples of MCP servers:
- **Filesystem MCP** — read/write files, create directories
- **GitHub MCP** — create issues, PRs, search repos
- **Slack MCP** — send messages, read channels
- **Database MCP** — query PostgreSQL, MySQL, SQLite
- **Email MCP** — send/receive emails
- **Google MCP** — Calendar, Drive, Gmail, Maps
- **Spotify MCP** — playback control, search music
- **Home Assistant MCP** — smart home control

### Architecture: VoxCore + MCP

```
┌─────────────────────────────────────────────────────┐
│                    VoxCore Pipeline                   │
│                                                      │
│  LLM → ResponseParser → TextOut → ToolRouter         │
│                                       │              │
│                          ┌────────────┴──────────┐   │
│                          │    Tool Registry       │   │
│                          │                        │   │
│                          │  web_search  (built-in)│   │
│                          │  memory      (built-in)│   │
│                          │  calendar    (built-in)│   │
│                          │                        │   │
│                          │  mcp:github  (MCP)     │   │
│                          │  mcp:slack   (MCP)     │   │
│                          │  mcp:email   (MCP)     │   │
│                          │  mcp:fs      (MCP)     │   │
│                          └────────────┬──────────┘   │
│                                       │              │
└───────────────────────────────────────┼──────────────┘
                                        │
                              ┌─────────┴─────────┐
                              │   MCP Client       │
                              │  (JSON-RPC over    │
                              │   stdio / SSE)     │
                              └─────────┬─────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    │                   │                   │
              ┌─────┴─────┐     ┌──────┴──────┐    ┌──────┴──────┐
              │ GitHub MCP │     │  Slack MCP  │    │  Email MCP  │
              │  Server    │     │   Server    │    │   Server    │
              └────────────┘     └─────────────┘    └─────────────┘
```

### How to Connect an MCP Server

#### Step 1: Install an MCP server

```bash
# Example: GitHub MCP server
npx @modelcontextprotocol/server-github

# Example: Filesystem MCP server
npx @modelcontextprotocol/server-filesystem /path/to/allowed/dir

# Example: SQLite MCP server
npx @modelcontextprotocol/server-sqlite --db-path ./data/mydb.sqlite
```

#### Step 2: Create the MCP bridge tool

Create `agent/tools/mcp_bridge.py`:

```python
"""
MCP Bridge — connects VoxCore's tool system to any MCP server.

Each MCP server connection becomes a tool that the LLM can call.
The bridge handles JSON-RPC communication, tool discovery, and
result formatting.
"""

import asyncio
import json
import logging
import subprocess
from typing import Any, Optional

from agent.tools.base_tool import BaseTool


class MCPClient:
    """Manages a connection to a single MCP server via stdio."""

    def __init__(self, server_command: list[str], name: str) -> None:
        self.name = name
        self._command = server_command
        self._process: Optional[subprocess.Popen] = None
        self._request_id = 0
        self._logger = logging.getLogger(f"MCP.{name}")

    async def start(self) -> None:
        """Launch the MCP server subprocess."""
        self._process = await asyncio.create_subprocess_exec(
            *self._command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        # Send initialize request
        await self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "voxcore", "version": "0.1.0"},
        })
        self._logger.info("MCP server '%s' started", self.name)

    async def list_tools(self) -> list[dict]:
        """Discover available tools from the MCP server."""
        response = await self._send_request("tools/list", {})
        return response.get("tools", [])

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Call a tool on the MCP server and return the result."""
        response = await self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments,
        })
        # Extract text content from MCP response
        content = response.get("content", [])
        texts = [c["text"] for c in content if c.get("type") == "text"]
        return "\n".join(texts) if texts else json.dumps(response)

    async def _send_request(self, method: str, params: dict) -> dict:
        """Send a JSON-RPC request to the MCP server."""
        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }
        line = json.dumps(request) + "\n"
        self._process.stdin.write(line.encode())
        await self._process.stdin.drain()

        response_line = await self._process.stdout.readline()
        return json.loads(response_line).get("result", {})

    async def stop(self) -> None:
        """Shut down the MCP server subprocess."""
        if self._process:
            self._process.terminate()
            await self._process.wait()


class MCPBridgeTool(BaseTool):
    """A VoxCore tool that proxies calls to an MCP server."""

    def __init__(self, mcp_client: MCPClient, tool_schema: dict) -> None:
        self.name = f"mcp:{mcp_client.name}:{tool_schema['name']}"
        self.description = tool_schema.get("description", "MCP tool")
        self.required_params = list(
            tool_schema.get("inputSchema", {}).get("required", [])
        )
        self.optional_params = [
            k for k in tool_schema.get("inputSchema", {}).get("properties", {})
            if k not in self.required_params
        ]
        super().__init__()
        self._client = mcp_client
        self._mcp_tool_name = tool_schema["name"]
        self._logger = logging.getLogger(f"Tool.{self.name}")

    async def execute(self, params: dict) -> str:
        """Execute the tool via MCP server."""
        self._logger.info("Calling MCP tool '%s'", self._mcp_tool_name)
        return await self._client.call_tool(self._mcp_tool_name, params)
```

#### Step 3: Add MCP configuration to `config.yaml`

```yaml
mcp_servers:
  github:
    command: ["npx", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "ghp_..."
    description: "GitHub — create issues, PRs, search repos"

  filesystem:
    command: ["npx", "@modelcontextprotocol/server-filesystem", "/Users/you/Documents"]
    description: "Read and write files in Documents folder"

  slack:
    command: ["npx", "@anthropic/mcp-server-slack"]
    env:
      SLACK_BOT_TOKEN: "xoxb-..."
    description: "Send and read Slack messages"

  database:
    command: ["npx", "@modelcontextprotocol/server-sqlite", "--db-path", "./data/app.sqlite"]
    description: "Query and modify the application database"
```

#### Step 4: Wire MCP into ToolRouter at startup

Add to `main.py` initialization:

```python
# After tool_router is created:
mcp_config = config.get("mcp_servers", {})
for server_name, server_cfg in mcp_config.items():
    try:
        client = MCPClient(server_cfg["command"], server_name)
        await client.start()

        # Discover all tools exposed by this MCP server
        tools = await client.list_tools()
        for tool_schema in tools:
            bridge = MCPBridgeTool(client, tool_schema)
            tool_router.register_tool(bridge.name, bridge)
            logger.info("Registered MCP tool: %s", bridge.name)

    except Exception as exc:
        logger.warning("Failed to start MCP server '%s': %s", server_name, exc)
```

#### Step 5: Update the system prompt

Add MCP tools to the persona's tool awareness:

```yaml
persona:
  system_prompt: |
    ...existing prompt...

    Available tools:
    - web_search: Search the web for information
    - memory: Read/write persistent memory
    - calendar: Manage calendar events
    - mcp:github:create_issue: Create a GitHub issue
    - mcp:github:search_repos: Search GitHub repositories
    - mcp:slack:send_message: Send a Slack message
    - mcp:filesystem:read_file: Read a file from disk
```

### MCP Server Directory

Here are popular MCP servers you can connect:

| Server | What It Does | Install |
|---|---|---|
| **@modelcontextprotocol/server-github** | GitHub issues, PRs, repos | `npx @modelcontextprotocol/server-github` |
| **@modelcontextprotocol/server-filesystem** | File read/write/search | `npx @modelcontextprotocol/server-filesystem /path` |
| **@modelcontextprotocol/server-sqlite** | SQLite database queries | `npx @modelcontextprotocol/server-sqlite --db-path ./db` |
| **@modelcontextprotocol/server-postgres** | PostgreSQL queries | `npx @modelcontextprotocol/server-postgres` |
| **@anthropic/mcp-server-slack** | Slack messages & channels | `npx @anthropic/mcp-server-slack` |
| **@anthropic/mcp-server-google-maps** | Location & directions | `npx @anthropic/mcp-server-google-maps` |
| **@anthropic/mcp-server-brave-search** | Brave web search | `npx @anthropic/mcp-server-brave-search` |
| **mcp-server-fetch** | HTTP fetch any URL | `npx mcp-server-fetch` |
| **@modelcontextprotocol/server-puppeteer** | Browser automation | `npx @modelcontextprotocol/server-puppeteer` |
| **mcp-server-home-assistant** | Smart home control | `pip install mcp-server-home-assistant` |
| **mcp-server-spotify** | Music playback/search | `npx mcp-server-spotify` |
| **mcp-server-notion** | Notion pages, databases | `npx mcp-server-notion` |

---

## 6. Adding Custom Tools

### Creating a New Built-in Tool

1. **Create the tool file** — `agent/tools/your_tool.py`:

```python
import logging
from agent.tools.base_tool import BaseTool


class YourTool(BaseTool):
    name = "your_tool"
    description = "Does something useful"
    required_params = ["input_text"]
    optional_params = ["option_a", "option_b"]

    def __init__(self) -> None:
        super().__init__()
        self._logger = logging.getLogger("Tool.your_tool")

    async def execute(self, params: dict) -> str:
        input_text = params["input_text"]
        # ... your logic here ...
        return f"Result: {input_text}"
```

2. **Register in ToolRouter** — `agent/tool_router.py`:

```python
from agent.tools.your_tool import YourTool

# Add to TOOL_MAP in _register_tools():
TOOL_MAP = {
    "web_search": WebSearchTool,
    "memory": MemoryTool,
    "calendar": CalendarTool,
    "your_tool": YourTool,  # ← Add here
}
```

3. **Enable in config** — `config.yaml`:

```yaml
agent:
  tools:
    - web_search
    - memory
    - calendar
    - your_tool    # ← Add here
```

4. **Tell the LLM about it** — Update the system prompt in `config.yaml`:

```yaml
persona:
  system_prompt: |
    ...
    Available tools: web_search, memory, calendar, your_tool
    Use your_tool when the user asks to [describe when to use it].
    Format: <agent>{"action": "your_tool", "params": {"input_text": "..."}}</agent>
```

That's it. The tool will be auto-registered, auto-routed, and results auto-injected back into the conversation.

### Tool Ideas

| Tool | Description | Complexity |
|---|---|---|
| **email** | Send/read emails via IMAP/SMTP | Medium |
| **weather** | Current weather via OpenWeatherMap | Easy |
| **timer** | Set countdown timers, reminders | Easy |
| **calculator** | Evaluate math expressions | Easy |
| **news** | Fetch latest headlines | Easy |
| **translate** | Translate text between languages | Easy |
| **image_gen** | Generate images via DALL-E / Stable Diffusion | Medium |
| **code_run** | Execute Python code in sandbox | Medium |
| **smart_home** | Control lights, thermostat, etc. | Medium (MCP) |
| **music** | Spotify playback control | Medium (MCP) |

---

## 7. Configuration Reference

### config.yaml — Complete Schema

```yaml
persona:
  name: "Aria"                       # AI persona name
  system_prompt: |                   # LLM system prompt (personality + rules)
    ...
  voice: "af_heart"                  # Kokoro voice ID
  language: "en"                     # Language code

models:
  stt_primary: "whisper-large-v3-turbo"        # Groq STT
  stt_fallback: "https://..."                   # HuggingFace fallback
  llm_fast: "llama-3.1-8b-instant"             # Fast brain (real-time)
  llm_smart: "meta-llama/llama-4-scout-17b"    # Smart brain (reasoning)
  llm_agentic: "moonshotai/kimi-k2-instruct"   # Agentic brain (tools)
  llm_deep: "qwen/qwen3-32b"                   # Deep brain (summaries)
  llm_power: "openai/gpt-oss-120b"             # Power brain (rare)
  tts_model: "models/kokoro-v1.0.onnx"         # Local TTS model
  tts_voices: "models/voices-v1.0.bin"          # Voice embeddings
  safety: "meta-llama/llama-guard-4-12b"        # Safety filter

audio:
  sample_rate: 16000                 # Mic input sample rate (Hz)
  output_sample_rate: 24000          # TTS output sample rate (Hz)
  chunk_ms: 32                       # VAD chunk size (ms)
  vad_speech_threshold: 0.65         # VAD sensitivity (0-1, higher = less sensitive)
  vad_silence_threshold: 0.6         # Silence detection threshold
  vad_silence_duration_ms: 500       # Silence duration for end-of-turn (ms)
  noise_floor_calibration_s: 0.5     # Ambient noise calibration window (s)

backchannel:
  enabled: true                      # Enable backchanneling
  min_pause_ms: 350                  # Phrase boundary detection (ms)
  min_gap_between_s: 8              # Min gap between backchannels (s)
  min_user_speech_s: 2              # User must speak this long first (s)
  clips_dir: "backchannel/clips/heart/"
  energy_threshold: 0.02

memory:
  short_term_turns: 20               # Rolling history size
  compress_after_turns: 20           # Compression trigger threshold
  long_term_enabled: true            # Enable ChromaDB persistence
  long_term_db: "./data/memory.db"   # ChromaDB path
  retrieval_top_k: 5                 # Past memories to retrieve at startup
  embedding_model: "default"         # ChromaDB embedding model

agent:
  enabled: true                      # Enable tool execution
  text_out_webhook: null             # Optional webhook for agent JSON
  max_concurrent_tools: 3            # Max parallel tool executions
  tool_timeout_s: 30                 # Tool execution timeout
  tools:                             # Enabled tool names
    - web_search
    - memory
    - calendar

safety:
  enabled: true                      # Enable safety filtering
  check_input: true                  # Filter user input
  check_output: true                 # Filter LLM output

server:
  host: "0.0.0.0"                    # Server bind host
  port: 8000                         # Server bind port
  websocket_enabled: true            # Enable WebSocket
```

### Environment Variables (`.env`)

```bash
GROQ_API_KEY=gsk_...                 # Required: Groq API key
TAVILY_API_KEY=tvly_...              # Optional: Tavily web search API key
HF_API_KEY=hf_...                    # Optional: HuggingFace fallback STT
```

---

## 8. API & Integration

### REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/inject` | Inject text context into conversation |
| `GET` | `/transcript` | Get current session transcript |
| `GET` | `/status` | Current turn state + session info |
| `POST` | `/config` | Hot-reload persona or voice |
| `WS` | `/text-out` | Stream of agent JSON events |

### Text Injection API

External systems push context into VoxCore via text injection:

```python
# From Python:
await text_injector.inject(
    content="Meeting with John at 3pm tomorrow",
    priority="high",        # "high" or "normal"
    source="calendar"
)

# Via REST API:
# POST /inject
# {"content": "Email from boss: quarterly review moved to Friday", "priority": "high"}
```

The injected context appears in the next LLM prompt as a `[CONTEXT]` block, and the AI naturally incorporates it into the conversation.

### WebSocket Protocol

Connect via `ws://localhost:8000/ws` for real-time streaming:

```json
// Incoming (from VoxCore):
{"type": "transcript", "text": "Hello there", "role": "user"}
{"type": "speech", "text": "Hi! How can I help?", "emotion": "cheerful"}
{"type": "agent", "action": "web_search", "params": {"query": "..."}}
{"type": "state", "old": "listening", "new": "thinking"}

// Outgoing (to VoxCore):
{"type": "inject", "content": "User just opened the weather app", "priority": "normal"}
{"type": "audio", "data": "<base64 audio>"}
```

---

## 9. Pipeline Walkthrough

Here's exactly what happens from when you say "Hello" to when VoxCore responds:

### Step 1: Audio Capture (MicStream)
- `sounddevice` captures 16kHz mono PCM audio in 32ms chunks
- Each chunk is pushed to all registered consumer queues (VAD, InterruptionDetector)

### Step 2: Voice Activity Detection (VAD)
- Silero VAD model runs inference on each chunk (~0.5ms)
- When speech probability > 0.65: fires `SPEECH_START` event
- Accumulates audio in buffer while speaking
- When silence probability < 0.35 for 500ms: fires `SPEECH_END` with buffered audio
- **Echo suppression:** Completely ignores audio during SPEAKING state + 700ms cooldown after

### Step 3: Speech-to-Text (STT)
- On `SPEECH_END`, takes buffered audio bytes
- Converts to WAV, sends to Groq Whisper Large V3 Turbo
- Fires `TRANSCRIPT_READY` with the text

### Step 4: Turn Management
- On `TRANSCRIPT_READY`, switches state to THINKING
- Adds user turn to session history
- Calls `PromptBuilder.build()` to assemble the full message array
- Starts LLM streaming via `llm_client.stream()`

### Step 5: LLM Streaming + Response Parsing
- llama-3.1-8b-instant streams tokens via Groq
- `ResponseParser` accumulates tokens, splits sentences on `.?!`
- Each complete sentence fires `LLM_SPEECH_TOKEN` for TTS
- If `<agent>` tag detected, captures JSON and fires `LLM_AGENT_TAG`
- State moves to SPEAKING on first sentence

### Step 6: TTS Synthesis (Local)
- On each `LLM_SPEECH_TOKEN`, strips emotion tags
- Kokoro-ONNX synthesizes locally (~50-100ms per sentence)
- Fires `TTS_CHUNK_READY` with WAV audio data

### Step 7: Audio Playback
- `AudioPlayer` receives TTS chunks, queues them
- Plays through speakers via `sounddevice` output stream
- On `INTERRUPT_DETECTED`: immediately stops playback, flushes queue
- When all chunks played: fires `PLAYBACK_DONE`

### Step 8: Back to Listening
- On `PLAYBACK_DONE`, state returns to LISTENING
- VAD resumes processing after 700ms cooldown
- Cycle repeats

### Latency Budget

| Step | Component | Expected Latency |
|---|---|---|
| VAD end-of-turn | Silero VAD (local) | ~500ms (silence window) |
| STT transcription | Groq whisper-turbo | ~200ms |
| LLM first token | llama-3.1-8b on Groq | ~100-200ms |
| First sentence buffer | ResponseParser | ~50ms |
| TTS synthesis | Kokoro-ONNX (local) | ~50-100ms |
| Audio playback start | sounddevice | ~10ms |
| **Total** | **Full pipeline** | **~910-1060ms** |

---

## 10. Extending VoxCore

### Adding a New Persona

Edit `config.yaml`:

```yaml
persona:
  name: "Max"
  system_prompt: |
    You are Max, a witty and sarcastic tech expert.
    You love puns and always slip one into your responses.
    Keep responses under 3 sentences. Be helpful but funny.
    ...
  voice: "am_adam"   # Male voice
```

### Adding a New Voice

See available Kokoro-ONNX voices:
- Female: `af_heart`, `af_bella`, `af_nicole`, `af_sarah`, `af_sky`
- Male: `am_adam`, `am_michael`
- British: `bf_emma`, `bm_george`
- And 19 more

After changing voice, regenerate backchannel clips:
```bash
python -m backchannel.generator
```

### Running as a Server

```bash
python main.py --interface server --port 8000
```

Enables WebSocket + REST API for web/mobile clients.

### Running in Headless Mode (No Mic)

```bash
python main.py --no-mic --interface cli
```

Text-only mode for testing. Type messages instead of speaking.

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
# Edit .env → add GROQ_API_KEY

# 3. Download TTS models
# Place kokoro-v1.0.onnx + voices-v1.0.bin in models/

# 4. Generate backchannel clips
python -m backchannel.generator

# 5. Run
python main.py
```

---

*VoxCore — February 2026*
