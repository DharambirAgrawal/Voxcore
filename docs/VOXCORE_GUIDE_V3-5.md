# VoxCore v3.5 — Agent Folder Structure & Tool Architecture

*Companion to VoxCore_Agent_Upgrade_v5.md and VoxCore_Agent_v5_Scenarios.md*

---

## Folder Structure

```
voxcore/agent/
│
├── __init__.py                  # Package exports — BaseTool + all tool classes
├── text_out.py                  # Validates <agent> JSON tags from LLM output
├── tool_router.py               # Dispatches tool calls, handles spoken output flag
├── slow_llm.py                  # Background calls to heavier models (unchanged)
│
└── tools/
    ├── __init__.py              # Exports all tool classes
    ├── base_tool.py             # Abstract base — every tool implements this
    ├── web_search.py            # Search internet via Tavily / DuckDuckGo fallback
    └── article_fetch.py         # Fetch URL → clean text → spoken summary via Groq
```

**Rule:** One tool = one file. Each file is fully self-contained. Adding a new tool means adding one file here and one line in `tool_router.py` and `config.yaml`. Nothing else changes.

---

## `base_tool.py` — The Contract Every Tool Signs

```python
from abc import ABC, abstractmethod

class BaseTool(ABC):
    name: str = ""                       # Unique action name — used in <agent> JSON
    description: str = ""               # LLM reads this to decide when to call the tool
    required_params: list[str] = []     # Must be present — router validates before calling
    optional_params: list[str] = []     # Can be absent — tool handles defaults internally
    produces_spoken_output: bool = False # If True → result goes to TTS directly, skips main LLM

    @abstractmethod
    async def execute(self, params: dict) -> str:
        """
        Run the tool. Always returns a plain string.
        If produces_spoken_output is True: return natural spoken text ready for TTS.
        If produces_spoken_output is False: return factual result for main LLM to synthesize.
        Never raise — catch all errors internally and return a spoken-friendly error string.
        """
        ...
```

**The four rules every tool must follow:**

1. `name` must be unique across all tools — it's the routing key
2. `description` must be precise and mutually exclusive from other tools — this is how the LLM routes
3. `execute()` must never raise — all errors caught internally, returned as strings
4. Return value is always a plain string — no markdown, no JSON, no structured objects

---

## `tool_router.py` — The Only File That Knows All Tools Exist

This is the single registration point. To add a new tool: import it here, add one line to `TOOL_MAP`.

```python
from agent.tools.web_search import WebSearchTool
from agent.tools.article_fetch import ArticleFetchTool
# Add new tools here — one import per tool

TOOL_MAP = {
    "web_search":    WebSearchTool,
    "article_fetch": ArticleFetchTool,
    # Add new tools here — one line per tool
}
```

**Nothing else in the codebase needs to know about individual tools.** `text_out.py` doesn't import them. `prompt_builder.py` doesn't import them. The router is the only junction.

---

## `__init__.py` — Clean Public Interface

```python
# agent/__init__.py
from agent.text_out import TextOut
from agent.tool_router import ToolRouter
from agent.slow_llm import SlowLLM

# agent/tools/__init__.py
from agent.tools.base_tool import BaseTool
from agent.tools.web_search import WebSearchTool
from agent.tools.article_fetch import ArticleFetchTool
# Add new tool exports here
```

---

## The `instruction` Parameter — Why Mode Enums Were Removed

### The old design (removed):

```json
{"action": "article_fetch", "params": {"url": "...", "mode": "key_points"}}
```

The tool would map `"key_points"` to a hardcoded prompt string internally. The main LLM's understanding of what the user wanted got squashed into one of three fixed options. Any nuance was lost.

### The new design:

```json
{"action": "article_fetch", "params": {"url": "...", "instruction": "user wants the three main takeaways spoken quickly, no background context"}}
```

The main LLM writes a plain English instruction. The tool's internal Groq model reads that instruction directly — verbatim, no translation. The full context of what the user wanted flows all the way through to the model doing the work.

**Why this is better:**

| | Mode enum | `instruction` field |
|---|---|---|
| Expressiveness | 3 fixed options | Unlimited — whatever the LLM writes |
| Nuance preserved | No — squashed to enum | Yes — full intent passes through |
| New behavior needs | Code change to add mode | No code change — LLM writes it |
| Tool complexity | Switch statement inside tool | One prompt field, always |

**How `article_fetch` uses it:**

```python
instruction = params.get("instruction", "Summarize the article naturally in about 150 spoken words.")

prompt = (
    f"You are summarizing an article to be spoken aloud by a voice AI.\n\n"
    f"Rules:\n"
    f"- Write for speech only. No markdown, no bullet points, no headers.\n"
    f"- Natural flowing sentences. First person as the AI narrator.\n"
    f"- Do not start with 'This article' or 'The author'.\n\n"
    f"Task: {instruction}\n\n"  # ← main LLM's intent, passed straight through
    f"Article:\n{cleaned_text}"
)
```

The `instruction` becomes the task line. The internal model gets the full picture. No information lost between the main LLM understanding the user and the summarizing model executing on it.

**Examples of what the main LLM might write as `instruction`:**

- `"Give a quick 2-sentence spoken overview, user is in a hurry"`
- `"Walk through the full article section by section, user wants the complete picture"`
- `"Focus only on the methodology section, user is a researcher"`
- `"Extract only the statistics and numbers mentioned, user wants data points"`
- `"Explain it simply as if for someone unfamiliar with the topic"`

None of these would be possible with a mode enum. All of them work with the `instruction` field and zero code changes.

---

## Adding a New Tool — Exact Steps

### Step 1 — Create the tool file

```python
# agent/tools/email_tool.py

from agent.tools.base_tool import BaseTool

class EmailTool(BaseTool):
    name = "email"
    description = (
        "Send an email to a contact in the user's address book. "
        "Use when the user explicitly asks to send, draft, or compose an email. "
        "Do not use for searching or reading content."
    )
    required_params = ["to", "subject", "body"]
    optional_params = ["cc"]
    produces_spoken_output = False  # result is context for main LLM to confirm

    def __init__(self, config: dict):
        self.smtp_host = config.get("email.smtp_host")
        # ... setup

    async def execute(self, params: dict) -> str:
        try:
            to = params["to"]
            subject = params["subject"]
            body = params["body"]
            # ... send logic
            return f"Email sent to {to} with subject '{subject}'."
        except Exception as e:
            return f"Failed to send email: {str(e)}"
```

### Step 2 — Register in `tool_router.py`

```python
from agent.tools.email_tool import EmailTool   # ← add this

TOOL_MAP = {
    "web_search":    WebSearchTool,
    "article_fetch": ArticleFetchTool,
    "email":         EmailTool,                 # ← add this
}
```

### Step 3 — Add to `config.yaml`

```yaml
agent:
  tools:
    - web_search
    - article_fetch
    - email           # ← add this

  email:              # ← add tool config block
    smtp_host: "smtp.gmail.com"
    smtp_port: 587
```

### Step 4 — Export in `tools/__init__.py`

```python
from agent.tools.email_tool import EmailTool   # ← add this
```

**That's it. Four lines across four files. No other file in the codebase changes.**

---

## Tool Description Writing Guide

The description field is load-bearing — it's how the LLM decides which tool to call. Write it badly and the LLM mis-routes. Write it well and routing is automatic.

**Formula:**
```
[What the tool does] + [When to use it] + [When NOT to use it — explicit contrast with other tools]
```

**Good example:**
```python
description = (
    "Search the internet for current information, news, facts, prices, "
    "or events that may have changed recently. Use when you need to FIND "
    "information and no URL has been provided. "
    "Do NOT use when the user has given a specific URL — use article_fetch for that."
)
```

**Bad example:**
```python
description = "Searches the web for information."
```

The bad version gives the LLM no signal about when NOT to use it. The good version creates a clean boundary between tools.

**Rules:**
- Always mention at least one other tool by name in the "do not use" clause
- Use uppercase for the most important routing word (FIND, NOT, ONLY)
- Keep under 60 words — longer descriptions dilute the routing signal

---

## `produces_spoken_output` — How It Flows

```
Tool returns string
        │
        ▼
tool_router._execute_tool() checks tool.produces_spoken_output
        │
        ├── False (default)
        │   └── text_injector.inject_tool_result(action, result)
        │       → "[TOOL RESULT — web_search]: Reuters: Fed holds rates..."
        │       → Main LLM reads this on next turn → synthesizes spoken answer
        │
        └── True (article_fetch only, currently)
            └── event_bus.publish(SPOKEN_TOOL_OUTPUT, {text: result})
                → turn_manager picks up event
                → feeds result directly to TTS pipeline
                → Aria speaks it — main LLM never sees it
                → No re-narration, no latency, no paraphrasing drift
```

**When to set `produces_spoken_output = True`:**
- Tool's output IS the answer — not context for further reasoning
- Tool has its own internal LLM call that produces polished spoken text
- Re-narrating through the main LLM would add latency and degrade quality

**When to leave it `False`:**
- Tool returns raw data that the main LLM should synthesize and contextualize
- Tool result is one input among several the LLM should reason about
- Tool confirms an action — main LLM should acknowledge naturally

---

## Config Pattern — Every Tool Gets Its Own Block

```yaml
agent:
  tools:                          # Only listed tools get registered at boot
    - web_search
    - article_fetch
  
  tool_timeout_seconds: 30        # Global fallback timeout
  max_concurrent: 2               # Max simultaneous tool executions

  web_search:                     # Tool-specific config
    default_num_results: 3
    search_depth: "basic"

  article_fetch:                  # Tool-specific config
    fetch_timeout_seconds: 10
    max_article_words: 8000
    long_article_threshold: 3000
    summary_model: "llama-3.1-8b-instant"
    long_article_model: "qwen3-32b"
    default_summary_max_words: 150
```

Each tool reads only its own config block. Tools are fully independent — changing article_fetch config cannot affect web_search behavior.

---

## Current Tool Registry (v5)

| Tool | File | `produces_spoken_output` | Internal LLM | Use Case |
|---|---|---|---|---|
| `web_search` | `web_search.py` | False | None | Find current info |
| `article_fetch` | `article_fetch.py` | True | llama-3.1-8b or qwen3-32b | Read a URL aloud |

**Removed in v5:**
- `calendar_tool.py` — deleted, not needed
- `memory_tool.py` — deleted, memory handled by `memory/` system automatically

---

*VoxCore Agent Folder Architecture — v3.5 — March 2026*