# VoxCore — Memory System Rebuild
## Implementation Spec — Aligned to config.yaml v3.6

*This document supersedes all previous memory specs.
It is written against the ACTUAL config.yaml v3.6 and actual code.
Read the entire document before writing any code.*

---

## What Exists Right Now — Actual State

```
memory/
├── __init__.py         exports ShortTermMemory, LongTermMemory, MemoryCompressor
├── short_term.py       deque(maxlen=10), emotion history, session context
├── long_term.py        ChromaDB PersistentClient, cosine similarity recall
└── compressor.py       threshold-triggered compression — HAS CRITICAL BUG
```

**Config values currently in use (memory block):**
```yaml
memory:
  short_term_turns: 20
  compress_after_turns: 20
  long_term_enabled: true
  long_term_db: "./data/memory.db"
  retrieval_top_k: 5
  embedding_model: "default"
```

**Agent tools currently enabled:**
```yaml
agent:
  tools:
    - web_search
    - article_fetch
```

---

## Bugs to Fix Before Building Anything New

### Bug 1 — Compressor never clears working memory (critical)

In `compressor.py._compress()`:
```python
# Current — stores to ChromaDB then returns:
await self._long_term.store(content=summary, metadata={...})
# ← MISSING: never clears short_term after this

# Result: after turn 20, compression runs on EVERY subsequent turn.
# ChromaDB fills with duplicate summaries forever.
```

Fix: after storing, keep only the last 3 turns in working memory.
See compressor section below for exact fix.

### Bug 2 — Compressor stores noise

Raw string concatenation stores "hi", "okay", "yeah" in ChromaDB permanently.
Fix: filter turns under 5 words before storing.

### Bug 3 — Per-turn ChromaDB recall in prompt_builder.py

`long_term.recall(user_input)` fires on every single user turn.
This is a vector search on every turn even when context hasn't changed.
Fix: remove this. Replace with session-start load + on-demand tool recall.

### Bug 4 — Everything stored as one type

All ChromaDB entries: `metadata={"type": "conversation_summary"}`.
No category filtering possible.
Fix: every stored memory gets `memory_type` + `category` fields.

---

## Model Allocation — Maps to config.yaml Model Names

**Read this table before any implementation that makes a model call.**
Use the config key names — never hardcode model strings in code.

```
TASK                              CONFIG KEY          REASON
──────────────────────────────────────────────────────────────────────────
background extraction             models.llm_fast     Runs after every turn.
                                                      Simple extraction from
                                                      2-3 turns. Must be cheap.

compressor summarization          models.llm_fast     Runs every 20 turns.
                                                      Moderate task, infrequent.

article_fetch short (<3000w)      article_fetch       Uses its own config block:
                                  .summary_model      llama-4-scout currently.
                                                      Inherits fallback_models.

article_fetch long (>3000w)       article_fetch       Uses its own config block:
                                  .long_article_model kimi-k2 currently.
                                                      Inherits fallback_models.

session_cache Q&A                 models.llm_fast     Answering from already-
                                                      extracted content.
                                                      Fast is correct here.

fact conflict detection           models.llm_fast     Inline binary check
                                                      (see background_llm section).
                                                      No separate model needed.

memory_recall formatting          NO MODEL            Pure string assembly.

loader.py session block           NO MODEL            Pure SQLite + data assembly.

long_term re-ranking              NO MODEL            Math only.
──────────────────────────────────────────────────────────────────────────
```

**Rules:**
- `models.llm_smart` (llama-4-scout) — only for article summarization via config
- `models.llm_agentic` (kimi-k2) — only for long article analysis via config
- `models.llm_deep` / `models.llm_power` — not used in memory system
- No model where pure data retrieval or math suffices
- All model references in code: `config["models"]["llm_fast"]` — never raw strings

---

## The Five Memory Tiers

```
TIER 1  WORKING MEMORY     short_term.py (exists, keep)
        What:  Last 20 turns + emotions (matches short_term_turns: 20)
        Where: RAM deque
        Speed: Instant
        Lives: Current session only

TIER 2  SESSION CACHE      session_cache.py (new)
        What:  Full tool results — article text, search results
        Where: RAM dict keyed by cache_id
        Speed: Instant
        Lives: Current session only — wiped on shutdown
        Why:   User asks follow-up about article → answer from RAM, not re-fetch

TIER 3  FACT STORE         fact_store.py (new)
        What:  Hard user facts: name, city, preferences, relationships
        Where: SQLite — data/memory/facts.db
        Speed: ~2ms
        Lives: Permanent until overwritten

TIER 4  PROCEDURAL         procedural.py (new)
        What:  Standing instructions + pending reminders
        Where: SQLite — data/memory/procedural.db
        Speed: ~2ms
        Lives: Until complete() called or expires_at passes

TIER 5  LONG-TERM          long_term.py (exists, update)
        What:  Episodic memories, semantic facts, saved links
        Where: ChromaDB — ./data/memory.db (matches long_term_db in config)
        Speed: 50-150ms vector search
        Lives: Permanent with recency decay
```

**The one rule that prevents all latency problems:**
The main LLM never waits for a memory operation mid-turn.
Tiers 1 and 2 are RAM — zero I/O.
Tiers 3 and 4 load once at session start.
Tier 5 loads only on-demand via agent tool call.

---

## The Session Cache — Tier 2 in Detail

### The problem it solves

article_fetch reads a 4000-word paper and speaks a 150-word summary.
User then asks: "what did it say about the methodology?"

Options:
- Re-fetch the article: 2-3 second network latency — terrible for voice
- Store full text in working memory: pollutes every subsequent LLM context
- Store full text in ChromaDB: permanent, stale after session, wasteful
- Store full text in RAM session cache, pointer in working memory: ✅

### What goes where

```
SESSION CACHE stores:
  full_content: full trafilatura-extracted article text (up to 8000 words)
  summary: the 150-word spoken text that was spoken aloud
  source_url: the URL
  title: extracted or inferred
  cache_id: "sc_{timestamp}_{4hex}"

WORKING MEMORY stores (as assistant turn):
  "[read article: {title} — ref:{cache_id}]
   {spoken_150_word_summary}"

  This is ~60-80 words. Same size as any other assistant response.
  The ref:{cache_id} tag is NOT spoken — response_parser strips [ref:...]
  from TTS output but keeps it in the stored turn text.

LONG-TERM stores: nothing automatically.
  Only if user says "save this" or "remember this link".
```

### Schema

```python
# session_cache.py
class SessionCache:
    def __init__(self):
        self._store: dict[str, dict] = {}
        # {
        #   cache_id: {
        #     content_type: "article" | "search_results",
        #     full_content: str,
        #     summary: str,
        #     source_url: Optional[str],
        #     title: Optional[str],
        #     word_count: int,
        #     stored_at: float,
        #     access_count: int
        #   }
        # }

    def store(self, content_type: str, full_content: str,
              summary: str, source_url: str = None,
              title: str = None) -> str:
        # cache_id = f"sc_{int(time.time())}_{uuid4().hex[:4]}"
        # Returns cache_id

    def get_content(self, cache_id: str) -> Optional[str]:
        # Returns full_content, increments access_count
        # Returns None if not found — never raises

    def get(self, cache_id: str) -> Optional[dict]:
        # Returns full entry

    def list_all(self) -> list[dict]:
        # Returns [{cache_id, content_type, title, source_url, summary}]
        # Without full_content — for AI to see what's cached this session

    def clear(self) -> None:
        # Called at session end — wipes everything
```

### Search results follow the same pattern

```
web_search returns 3 formatted results.

SESSION CACHE: full formatted results text
WORKING MEMORY: "[searched: {query} — ref:{cache_id}]
                 Top result: {one-line summary of top result}"

Follow-up about search results → session_cache_qa fires
No re-search needed
```

### How the AI answers follow-up questions from cache

```
User: "what did that paper say about the methodology?"

Working memory has: "[read article: X — ref:sc_1741_a3f2] ..."
AI recognizes: question about cached content this session
AI emits:
<agent>{"action": "session_cache_qa",
        "params": {"cache_id": "sc_1741_a3f2",
                   "question": "what does the paper say about methodology"}}</agent>

session_cache_qa tool:
  1. session_cache.get_content("sc_1741_a3f2") → full article text (RAM)
  2. Groq call: config["models"]["llm_fast"]
     Prompt: "Answer this question from the content below.
              Spoken answer, 2-3 sentences max, no markdown.
              Question: {question}
              Content: {full_content[:6000]}"
  3. produces_spoken_output = True → answer goes directly to TTS

Total latency: ~250ms (one fast LLM call, no network fetch)
```

---

## File: `memory/fact_store.py` (NEW)

```python
# Schema
CREATE TABLE IF NOT EXISTS facts (
    key             TEXT PRIMARY KEY,
    value           TEXT NOT NULL,
    confidence      REAL DEFAULT 0.7,
    source          TEXT DEFAULT 'extracted',
    last_mentioned  INTEGER DEFAULT 0,
    created_at      INTEGER NOT NULL,
    updated_at      INTEGER NOT NULL
)

# db path from config: memory.fact_store_db
# default if not in config: "data/memory/facts.db"
# os.makedirs(parent_dir, exist_ok=True) on init
```

### Key naming convention

```
user_name               user_city               user_state
user_occupation         user_employer           user_timezone
pref_response_length    pref_technical_depth    pref_communication_style
pref_paper_year         pref_units              pref_language
research_field_primary  project_current         project_status
relationship_{name}     schedule_{label}
```

### Methods

```python
def set(self, key: str, value: str,
        confidence: float = 0.7,
        source: str = "extracted") -> None:
    # Check if key exists first — preserve created_at if updating
    # INSERT OR REPLACE with correct created_at handling

def get(self, key: str) -> Optional[str]:
    # Returns None if not found — never raises

def get_with_meta(self, key: str) -> Optional[dict]:
    # Full row as dict

def get_all(self) -> dict[str, str]:
    # {key: value} — empty dict if none

def get_by_prefix(self, prefix: str) -> dict[str, str]:
    # SELECT WHERE key LIKE 'prefix%'

def touch(self, key: str) -> None:
    # UPDATE SET last_mentioned = int(time.time()) WHERE key = ?
    # Called by background_llm when fact referenced in conversation

def delete(self, key: str) -> bool:
    # Returns True if deleted, False if not found
```

### Selective loading rules (used by loader.py)

```
ALWAYS LOAD — core identity, small, stable:
  user_name, user_city, user_occupation
  pref_response_length, pref_technical_depth, pref_communication_style
  → ~40-50 tokens

LOAD IF RECENT — last_mentioned within 14 days, max 5 entries:
  relationship_* prefix — sorted by last_mentioned DESC
  project_current, research_field_primary
  → ~40-60 tokens if present

NEVER AUTO-LOAD — on-demand via memory_recall only:
  All other keys not listed above
  (saved links, schedules, detailed facts)
```

---

## File: `memory/procedural.py` (NEW)

```python
# Schema
CREATE TABLE IF NOT EXISTS instructions (
    id          TEXT PRIMARY KEY,
    content     TEXT NOT NULL,
    category    TEXT NOT NULL DEFAULT 'general',
    importance  REAL DEFAULT 0.9,
    source      TEXT DEFAULT 'user_explicit',
    active      INTEGER DEFAULT 1,
    created_at  INTEGER NOT NULL,
    expires_at  INTEGER DEFAULT NULL
)

# db path from config: memory.procedural_db
# default if not in config: "data/memory/procedural.db"
```

**category values:**
`general` | `communication` | `topic` | `reminder` | `relay`

**relay** = message from third party, surfaces proactively at session start,
first thing before any other content.

### Methods

```python
def add_instruction(self, content: str, category: str = "general",
                    importance: float = 0.9,
                    source: str = "user_explicit",
                    expires_at: Optional[int] = None) -> str:
    # id = f"inst_{int(time.time())}_{uuid4().hex[:6]}"
    # Returns id

def get_active(self, category: Optional[str] = None) -> list[dict]:
    # WHERE active=1 AND (expires_at IS NULL OR expires_at > now)
    # ORDER BY importance DESC

def get_pending_relays(self) -> list[dict]:
    # get_active(category='relay')

def complete(self, instruction_id: str, outcome: str = "") -> bool:
    # UPDATE SET active=0
    # Returns True if found and updated

def get_active_count(self) -> int:
    # Quick count for loader.py token budget check
```

---

## File: `memory/background_llm.py` (NEW)

### Model: `config["models"]["llm_fast"]`

Simple extraction from 2-3 short turns. Fast and cheap.
Runs after every turn as `asyncio.create_task()`. Never awaited.

### Pre-filter — runs before any API call

```python
SIGNAL_PHRASES = [
    "my name is", "i am", "i live", "i work", "i like", "i hate",
    "i prefer", "i always", "i never", "remember", "don't forget",
    "actually", "no that's wrong", "you keep saying", "it's not",
    "my friend", "my colleague", "my boss", "next week", "last week",
    "i told you", "make sure", "always remember", "by the way",
    "remind me", "i need to", "i have to", "deadline", "meeting",
    "appointment", "don't ever", "save this", "note that", "save that"
]

def has_signal(turns: list[dict]) -> bool:
    combined = " ".join(
        t["content"].lower() for t in turns if t["role"] == "user"
    )
    return any(phrase in combined for phrase in SIGNAL_PHRASES)
```

If `has_signal()` is False → return immediately. Zero API calls.

### Conflict detection — inline with llm_fast, no separate model

When background_llm finds a potential UPDATE to an existing fact,
add a second message to the SAME extraction call:

```python
# In the extraction prompt, when an existing fact is found:
system_addendum = f"""
CONFLICT CHECK:
Existing stored value for '{key}': "{existing_value}"
New value extracted: "{new_value}"
Are these contradictory? If yes, set operation to CONFLICT.
If the new value is just more detail or a correction, set to UPDATE.
"""
```

No second API call. The same llm_fast call handles it.
CONFLICT means store both in ChromaDB with a conflict flag.
UPDATE means overwrite fact_store.

### Extraction prompt

```
Extract memorable information from these conversation turns.
Output one JSON object per line. No preamble. No explanation.
If nothing worth storing: output exactly: NOTHING

Each JSON must have exactly these fields:
{
  "operation": "ADD" | "UPDATE" | "CONFLICT" | "IGNORE",
  "store":     "fact" | "procedural" | "episodic" | "semantic",
  "key":       "snake_case_key_if_fact_store_else_null",
  "content":   "single clean third-person sentence",
  "importance": 0.0-1.0,
  "category":  "fact|preference|relationship|instruction|relay|event|correction",
  "expires_at": "ISO timestamp or null"
}

Rules:
- Greetings, filler, one-word responses → NOTHING
- "hi" "okay" "thanks" "got it" "yeah" "sure" → NOTHING
- Explicit correction ("actually X not Y") → UPDATE, importance 1.0
- "remember"/"don't forget"/"always"/"never"/"make sure" → procedural
- User name/city/job → fact, key=user_name/user_city/user_occupation
- Preferences → fact, key=pref_*
- Relationships → fact, key=relationship_{firstname_lowercase}
- Future events, deadlines → episodic + procedural reminder
- Past events → episodic, long_term
- Do NOT store: AI's words, tool results, prices, weather, news
- Do NOT store: what already exists unchanged
```

### Routing extracted items

```python
async def process_turn_background(turns, fact_store, procedural,
                                  long_term, existing_summary) -> None:

    if not has_signal(turns):
        return

    extracted = await _call_extraction(turns, existing_summary)
    if not extracted:
        return

    for item in extracted:
        op = item["operation"]
        store = item["store"]

        if op == "IGNORE":
            continue

        elif store == "fact":
            if op in ("UPDATE", "CONFLICT"):
                await long_term.store(
                    content=f"Previous: '{fact_store.get(item['key'])}' "
                            f"→ Updated: '{item['content']}'",
                    memory_type="semantic",
                    category="correction",
                    importance=0.8
                )
            fact_store.set(
                item["key"], item["content"],
                confidence=1.0 if op == "UPDATE" else item["importance"],
                source="user_correction" if op == "UPDATE" else "extracted"
            )

        elif store == "procedural":
            expires = _parse_iso_to_unix(item.get("expires_at"))
            procedural.add_instruction(
                content=item["content"],
                category=item["category"],
                importance=item["importance"],
                expires_at=expires
            )

        elif store in ("episodic", "semantic"):
            await long_term.store(
                content=item["content"],
                memory_type=store,
                category=item["category"],
                importance=item["importance"]
            )
```

### How it's called from BrainRouter

```python
# End of _process_input(), after short_term.add_turn() calls:
recent_turns = self._short_term.get_recent_turns(3)
asyncio.create_task(
    self._background_llm.process_turn_background(
        turns=recent_turns,
        fact_store=self._fact_store,
        procedural=self._procedural,
        long_term=self._long_term,
        existing_summary=self._loader.get_stored_summary()
    )
)
# No await. Fire and forget.
```

---

## File: `memory/loader.py` (NEW)

### No LLM involved. Pure data assembly from SQLite stores.

Runs once at session start. Returns a formatted string block
that `prompt_builder.py` injects into the system prompt.

### Selective loading — what gets loaded

```
ALWAYS:
  Pending relays (procedural category=relay)     → [PENDING MESSAGES]
  Active instructions (procedural active=1)      → [STANDING INSTRUCTIONS]
  Core identity facts (user_name, user_city,
    user_occupation, pref_* keys)                → [USER CONTEXT]

IF RECENT (last_mentioned within 14 days, max 5):
  relationship_* from fact_store                 → [RECENT RELATIONSHIPS]

IF WITHIN 7 DAYS:
  procedural category=reminder with              → [UPCOMING]
  expires_at within next 7 days

NEVER AUTO-LOAD:
  link_* keys, detailed episodic history,
  old relationships — on-demand via memory_recall
```

### Token budget: 350 tokens hard cap

Estimate: `len(block_text) // 4`

Drop order when over budget:
1. Drop [RECENT RELATIONSHIPS] oldest entries first
2. Drop [UPCOMING] entries lowest importance first
3. Truncate low-confidence pref_* entries from [USER CONTEXT]
4. **Never drop [PENDING MESSAGES] or [STANDING INSTRUCTIONS]**

### Output format

```
[USER CONTEXT]
Name: Dharambir
Location: Shreveport, Louisiana
Occupation: Professor, LSU
Communication: Concise and technical

[PENDING MESSAGES]
- Rahul needs the dataset by Friday

[STANDING INSTRUCTIONS]
- Cite papers from 2018 or later unless asked
- Use bullet points when listing items

[UPCOMING]
- Paper deadline: March 15

[RECENT RELATIONSHIPS]
- Rahul: colleague, former PhD student
- Dr. Patel: department chair (new)
```

### Methods

```python
class MemoryLoader:
    def __init__(self, fact_store, procedural, long_term, config):
        self._max_tokens = config.get("memory", {}).get("loader_max_tokens", 350)
        self._relationship_days = config.get("memory", {}).get("relationship_recency_days", 14)
        self._reminder_days = config.get("memory", {}).get("reminder_window_days", 7)
        self._cached_summary: str = ""

    async def build_session_block(self) -> str:
        # 1. relays = procedural.get_pending_relays()
        # 2. instructions = procedural.get_active(category != 'relay')
        # 3. core_facts = fact_store.get core identity keys
        # 4. relationships = fact_store.get_by_prefix("relationship_")
        #    filtered by last_mentioned > now - 14 days, max 5
        # 5. upcoming = procedural.get_active(category='reminder')
        #    filtered by expires_at within 7 days
        # 6. assemble → enforce_budget() → cache → return

    def get_stored_summary(self) -> str:
        return self._cached_summary
```

### Where it's called

```python
# In VoxCoreSession.initialize() or equivalent startup:
memory_block = await self._loader.build_session_block()
self._prompt_builder.set_memory_block(memory_block)
# Called once. Not per-turn.
```

---

## File: `memory/retriever.py` (NEW)

### No LLM. Pure data retrieval + string formatting.

```python
class MemoryRetriever:

    async def recall(self, query: str,
                     memory_type: Optional[str] = None,
                     top_k: int = 5) -> dict:
        # Three paths in PARALLEL via asyncio.gather():
        #
        # Path 1: working memory — last 5 turns, direct deque access
        # Path 2: long_term.recall() — ChromaDB vector search
        # Path 3: fact_store signal check — if query mentions a person
        #         or preference keyword, include relevant fact_store entries
        #
        # Returns:
        # {
        #   "working": [last 5 turns],
        #   "memories": [chromadb results],
        #   "facts": {relevant fact_store entries},
        #   "token_estimate": int
        # }

    async def recall_for_person(self, name: str) -> dict:
        # fact_store.get(f"relationship_{name.lower()}")
        # + long_term.recall(name, memory_type="semantic")
        # + long_term.recall(name, memory_type="episodic")

    def format_for_injection(self, result: dict) -> str:
        # Plain text, no markdown
        # ~100-200 words max
        # This goes through text_injector as tool result context
```

---

## File: `memory/long_term.py` (UPDATED)

### Change 1 — Type tagging on every store() call

```python
async def store(self,
                content: str,
                memory_type: str = "semantic",
                category: str = "fact",
                importance: float = 0.5,
                metadata: Optional[dict] = None) -> str:
    meta = {
        "timestamp": datetime.now().isoformat(),
        "memory_type": memory_type,
        "category": category,
        "importance": importance,
        "access_count": 0,
        "last_accessed": datetime.now().isoformat(),
        **(metadata or {})
    }
    # rest of existing store logic unchanged
```

### Change 2 — Re-ranking in recall()

```python
from math import exp

RECENCY_LAMBDA = {
    "event": 0.1,
    "fact": 0.05,
    "preference": 0.03,
    "correction": 0.02,
    "instruction": 0.01
}

# After getting raw ChromaDB results:
for result in raw_results:
    days = (datetime.now() -
            datetime.fromisoformat(result["metadata"]["last_accessed"])).days
    lam = RECENCY_LAMBDA.get(result["metadata"].get("category", "fact"), 0.05)
    recency = exp(-lam * days)
    importance = result["metadata"].get("importance", 0.5)
    similarity = 1 - result["distance"]

    result["final_score"] = (0.6 * similarity +
                             0.25 * importance +
                             0.15 * recency)

# Sort by final_score DESC, return top_k
# Fetch 3× candidates from ChromaDB, re-rank to top_k
```

### Change 3 — Increment access tracking

```python
# After returning results, update ChromaDB metadata:
for r in returned:
    self._collection.update(
        ids=[r["id"]],
        metadatas=[{
            **r["metadata"],
            "access_count": r["metadata"].get("access_count", 0) + 1,
            "last_accessed": datetime.now().isoformat()
        }]
    )
```

### Change 4 — memory_type filter

```python
where = {}
if memory_type:
    where["memory_type"] = {"$eq": memory_type}

results = self._collection.query(
    query_texts=[query],
    n_results=min(top_k * 3, self._collection.count()),
    where=where if where else None
)
```

Note: `long_term_db` config key is `./data/memory.db` — keep using this path.
Do not change the ChromaDB path.

---

## File: `memory/compressor.py` (UPDATED)

### The bug fix

```python
async def _compress(self) -> None:
    self._is_compressing = True
    try:
        turns = self._short_term.get_recent_turns()
        if not turns:
            return

        # Filter noise — only store turns with real content
        meaningful = [
            t for t in turns
            if len(t["content"].split()) > 5
            and t["content"].lower().strip() not in {
                "okay", "yes", "no", "thanks", "got it", "sure",
                "right", "yeah", "alright", "fine", "understood"
            }
        ]

        if meaningful:
            summary = await self._build_summary_llm(meaningful)
            if summary:
                await self._long_term.store(
                    content=summary,
                    memory_type="episodic",
                    category="event",
                    importance=0.4,
                    metadata={
                        "turn_count": len(turns),
                        "compressed_at": datetime.now().isoformat()
                    }
                )

        # ✅ THE FIX — keep last 3 turns, clear the rest
        turns_list = list(self._short_term._turns)
        self._short_term._turns.clear()
        for turn in turns_list[-3:]:
            self._short_term._turns.append(turn)

        logger.info(f"Compressed. Kept last 3 of {len(turns)} turns.")

    except Exception as e:
        logger.error(f"Compression failed: {e}")
    finally:
        self._is_compressing = False

async def _build_summary_llm(self, turns: list[dict]) -> Optional[str]:
    # Groq call: config["models"]["llm_fast"]
    # Prompt:
    # "Summarize this conversation excerpt in 2-3 sentences.
    #  Third person. Facts only. No filler. No AI speech.
    #  Conversation: {formatted_turns}"
    # Returns summary string or None on failure

async def compress_session_end(self) -> None:
    # Called from VoxCoreSession.shutdown()
    remaining = self._short_term.get_recent_turns()
    if len(remaining) > 2:
        await self._compress()
```

**`compress_after_turns`** is read from config: `config["memory"]["compress_after_turns"]`.
Currently 20. The compressor uses this value — never hardcode 10 or 8.

---

## Agent Tools — New Tools to Add

### `agent/tools/session_cache_qa.py` (NEW)

```python
class SessionCacheQATool(BaseTool):
    name = "session_cache_qa"
    description = (
        "Answer a question from an article, paper, or search result that was "
        "read or searched EARLIER IN THIS CONVERSATION. Use when the user asks "
        "a follow-up question about content from the current session. "
        "Requires the cache_id from the earlier result (visible in conversation). "
        "Do NOT use for new searches. Do NOT use for past sessions."
    )
    required_params = ["cache_id", "question"]
    optional_params = []
    produces_spoken_output = True

    def __init__(self, config: dict, session_cache: SessionCache):
        self._session_cache = session_cache
        self._model = config["models"]["llm_fast"]
        self._groq = AsyncGroq()

    async def execute(self, params: dict) -> str:
        content = self._session_cache.get_content(params["cache_id"])
        if not content:
            return (
                "I don't seem to have that content cached anymore. "
                "Would you like me to fetch it again?"
            )

        response = await self._groq.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": (
                f"Answer this question based only on the content below.\n"
                f"Spoken answer, 2-3 sentences max, no markdown, no bullet points.\n"
                f"Question: {params['question']}\n\n"
                f"Content:\n{content[:6000]}"
            )}],
            max_tokens=200,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
```

### `agent/tools/memory_recall.py` (NEW)

```python
class MemoryRecallTool(BaseTool):
    name = "memory_recall"
    description = (
        "Search memory for information about the user, relationships, preferences, "
        "or topics discussed in PREVIOUS SESSIONS. Use when user references "
        "something from the past, mentions a person needing context, or asks "
        "what you remember about something. "
        "Do NOT use for current internet info — use web_search. "
        "Do NOT use for content from this session — use session_cache_qa."
    )
    required_params = ["query"]
    optional_params = ["memory_type", "top_k"]
    produces_spoken_output = False   # context for main LLM to synthesize

    async def execute(self, params: dict) -> str:
        result = await self._retriever.recall(
            query=params["query"],
            memory_type=params.get("memory_type"),
            top_k=params.get("top_k", 5)
        )
        return self._retriever.format_for_injection(result)
```

### Updated `tool_router.py` TOOL_MAP

```python
from agent.tools.web_search import WebSearchTool
from agent.tools.article_fetch import ArticleFetchTool
from agent.tools.memory_recall import MemoryRecallTool
from agent.tools.session_cache_qa import SessionCacheQATool

TOOL_MAP = {
    "web_search":       WebSearchTool,
    "article_fetch":    ArticleFetchTool,
    "memory_recall":    MemoryRecallTool,
    "session_cache_qa": SessionCacheQATool,
}
```

### How article_fetch stores to session cache

In `article_fetch.py`, after generating the spoken summary:

```python
cache_id = self._session_cache.store(
    content_type="article",
    full_content=raw_text,
    summary=spoken_text,
    source_url=url,
    title=self._extract_title(raw_text)
)

# Append cache reference to return value
# response_parser.py will strip [ref:...] from TTS but keep in working memory
return f"{spoken_text} [ref:{cache_id}]"
```

`response_parser.py` must:
- Strip `[ref:sc_...]` from TTS output (not spoken aloud)
- Keep `[ref:sc_...]` in the text stored as assistant turn in working memory

The AI sees the ref tag in working memory on subsequent turns and uses it
when it needs to call session_cache_qa.

---

## Updated `brain/prompt_builder.py`

```python
class PromptBuilder:
    def __init__(self, config, short_term, long_term):
        # existing init unchanged
        self._memory_block: str = ""   # set once at session start

    def set_memory_block(self, block: str) -> None:
        self._memory_block = block
        logger.debug(f"Memory block set: ~{len(block)//4} tokens")

    async def build_messages(self, user_input, emotion=None,
                             injected_context=None) -> list[dict]:
        # REMOVE: recalled_memories = await self._long_term.recall(user_input)
        # That per-turn ChromaDB call is gone.

        system_content = self._build_system_prompt(emotion)
        messages = [{"role": "system", "content": system_content}]
        messages.extend(self._short_term.get_messages_for_llm())

        if injected_context:
            messages.append({
                "role": "user",
                "content": f"[Context Update]: {injected_context}"
            })

        messages.append({"role": "user", "content": user_input})
        return messages

    def _build_system_prompt(self, emotion: Optional[str]) -> str:
        parts = [self._system_prompt_template]  # from config persona.system_prompt

        if self._memory_block:
            parts.append(f"\n\n{self._memory_block}")

        emotion_ctx = self._short_term.get_emotion_context()
        if emotion_ctx:
            parts.append(f"\n\n[Recent Emotions]: {emotion_ctx}")
        if emotion:
            parts.append(f"\n[Current User Emotion]: {emotion}")

        return "".join(parts)
```

---

## System Prompt Additions

The existing system prompt in `config.yaml` is good and its length is
justified — the examples are load-bearing for correct tool routing.

Add this block to the TOOL USAGE section, after the existing article_fetch example:

```
- User asks a follow-up question about an article or search from THIS conversation
  → USE session_cache_qa with the cache_id visible in conversation history

- User asks what you remember about them, past conversations, or previous sessions
  → USE memory_recall

- User says "remember this", "save this link", "don't forget"
  → USE memory_recall with operation hint (this triggers background storage)

HOW TO USE NEW TOOLS:
<agent>{"action": "session_cache_qa", "params": {"cache_id": "sc_XXXX", "question": "QUESTION"}}</agent>
<agent>{"action": "memory_recall", "params": {"query": "WHAT TO RECALL"}}</agent>

EXAMPLES:
User: "what did that paper say about the dataset?"
You: [calm] Let me check what we read. <agent>{"action": "session_cache_qa", "params": {"cache_id": "sc_1741_a3f2", "question": "what does the paper say about the dataset"}}</agent>

User: "do you remember my research focus?"
You: [calm] Let me check. <agent>{"action": "memory_recall", "params": {"query": "user research focus and field"}}</agent>
```

**Keep all existing examples. Add new ones after.**
Do not shorten or remove existing system prompt content.

---

## config.yaml — Memory Block Update

Replace the existing `memory:` block with:

```yaml
memory:
  # Working memory
  short_term_turns: 20               # Keep existing value
  compress_after_turns: 20           # Keep existing value

  # ChromaDB — keep existing path
  long_term_enabled: true
  long_term_db: "./data/memory.db"   # Keep existing path — do not change
  retrieval_top_k: 5
  embedding_model: "default"
  similarity_threshold: 0.65

  # New SQLite stores
  fact_store_db: "data/memory/facts.db"
  procedural_db: "data/memory/procedural.db"

  # Session-start loader
  loader_max_tokens: 350
  relationship_recency_days: 14
  reminder_window_days: 7

  # Background LLM
  background_enabled: true           # Set false to disable during testing
```

Add to `agent.tools` list:

```yaml
agent:
  tools:
    - web_search
    - article_fetch
    - memory_recall       # new
    - session_cache_qa    # new
```

---

## Files Changed Outside `memory/` and `agent/`

```
brain/prompt_builder.py    Remove per-turn long_term.recall() call
                           Add set_memory_block() method
                           Remove recalled_memories parameter from _build_system_prompt

brain/router.py            Add background_llm.create_task() after each turn
                           Add session_cache reference passed to article_fetch

brain/response_parser.py   Add [ref:sc_...] tag stripping from TTS text
                           Keep [ref:sc_...] in stored assistant turn text

core/session.py            Add: fact_store, procedural, background_llm,
                                session_cache, loader, retriever
                           Add initialize(): build_session_block()
                           Add shutdown(): compress_session_end() + cache.clear()

core/event_bus.py          Add SPOKEN_TOOL_OUTPUT to EventType enum

core/turn_manager.py       Subscribe to SPOKEN_TOOL_OUTPUT
                           Feed result text directly to TTS pipeline

memory/__init__.py         Export all new classes
```

---

## Build Order

```
Step 1   Fix compressor.py bug        MUST BE FIRST. Test: no duplicate ChromaDB
                                       entries after turn 20.

Step 2   memory/fact_store.py         SQLite + all methods.
                                       Test: set/get/overwrite/prefix/touch.

Step 3   memory/procedural.py         SQLite + all methods.
                                       Test: add/get_active/complete lifecycle.

Step 4   memory/session_cache.py      RAM dict, no persistence.
                                       Test: store → get_content → clear.

Step 5   memory/background_llm.py     Build signal pre-filter FIRST.
                                       Test: pure filler turns = zero API calls.
                                       Then build extraction + routing.

Step 6   memory/loader.py             Session block assembly + budget enforcement.
                                       Test: populate stores, print the block.

Step 7   Update memory/long_term.py   Type fields + re-ranking + access tracking.
                                       Test: typed store → typed recall.

Step 8   memory/retriever.py          Parallel asyncio.gather() recall.
                                       Test: all three paths return correctly.

Step 9   agent/tools/memory_recall.py Thin wrapper over retriever.

Step 10  agent/tools/session_cache_qa.py QA over cached content.
                                       Test: store article → ask question from it.

Step 11  Update article_fetch.py      Add session_cache.store() + [ref:id] append.

Step 12  Update response_parser.py    Handle [ref:...] tags.

Step 13  Update prompt_builder.py     Remove per-turn recall + set_memory_block().

Step 14  Update router.py             Add background_llm.create_task().

Step 15  Update session.py            Wire all new components.

Step 16  Update tool_router.py        Add new tools to TOOL_MAP.

Step 17  config.yaml                  Update memory block + agent.tools list.

Step 18  memory/__init__.py           Export all new classes.
```

---

## Hard Rules

```
1. Main LLM never waits for memory mid-turn.
   RAM tiers (working, session cache) are instant.
   SQLite tiers load once at session start.
   ChromaDB loads on-demand via tool call only.

2. background_llm: asyncio.create_task() only. Never awaited.

3. fact_store and procedural: SQLite only. Not ChromaDB.
   Instructions cannot miss due to vector similarity threshold.

4. Compressor MUST clear working memory after compressing.
   This is Bug 1. Fix it in Step 1 before anything else.

5. loader.py: 350 token hard cap. Never drop relays or instructions.

6. background_llm: signal pre-filter before any API call.
   Filler turns cost zero API calls.

7. Article full text: session cache only. Never in working memory or ChromaDB.
   Re-fetching next session is cheaper than stale permanent storage.

8. All model references in code use config keys.
   Never hardcode model name strings.
   config["models"]["llm_fast"] not "llama-3.1-8b-instant"

9. memory_recall result → text_injector → main LLM synthesizes.
   session_cache_qa result → SPOKEN_TOOL_OUTPUT → TTS directly.

10. ChromaDB path stays as ./data/memory.db (existing config value).
    Do not change paths that already have data.
```

---

*VoxCore Memory System — v5 Final — Aligned to config.yaml v3.6 — March 2026*
*Supersedes all previous memory spec documents*