# JARVIS v2 - Simplified MVP Plan

**Goal:** A working Tauri app that demonstrates deep understanding of AI/ML - when to use what technique.

**Core Demo:** Click on a conversation → Get 3 smart, natural reply suggestions

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         TAURI APP                               │
│  (Svelte frontend - reuse/simplify from v1)                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FASTAPI BACKEND                            │
│                    (10-15 endpoints only)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  GET  /health                                                   │
│  GET  /conversations                                            │
│  GET  /conversations/{id}/messages                              │
│  POST /generate/replies     ← Main feature                      │
│  POST /search               ← Hybrid search (Phase 2)           │
│  GET  /settings                                                 │
│  PUT  /settings                                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       CORE SERVICES                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  iMessage   │  │    MLX      │  │   Reply Generator       │  │
│  │   Reader    │  │   Loader    │  │   (prompts + analysis)  │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Style     │  │  Context    │  │   Intent Classifier     │  │
│  │  Analyzer   │  │  Analyzer   │  │   (for chat mode)       │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                               │
├─────────────────────────────────────────────────────────────────┤
│  iMessage DB (read-only)  │  Settings (~/.jarvis/config.json)   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Folder Structure

```
v2/
├── api/
│   ├── __init__.py
│   ├── main.py              # FastAPI app, CORS, startup
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── health.py        # GET /health
│   │   ├── conversations.py # GET /conversations, /conversations/{id}/messages
│   │   ├── generate.py      # POST /generate/replies
│   │   ├── search.py        # POST /search (Phase 2)
│   │   └── settings.py      # GET/PUT /settings
│   └── schemas.py           # Pydantic models (minimal)
│
├── core/
│   ├── __init__.py
│   ├── imessage/
│   │   ├── __init__.py
│   │   └── reader.py        # Copy/simplify from v1
│   ├── models/
│   │   ├── __init__.py
│   │   ├── loader.py        # MLX model loading
│   │   └── registry.py      # Available models
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── reply_generator.py   # Main reply generation logic
│   │   ├── style_analyzer.py    # Analyze user's texting style
│   │   ├── context_analyzer.py  # Analyze conversation context
│   │   └── prompts.py           # Prompt templates
│   └── config.py            # Settings management
│
├── tests/
│   ├── __init__.py
│   ├── test_reader.py
│   ├── test_generator.py
│   └── test_api.py
│
├── scripts/
│   ├── test_generation.py   # Manual testing script
│   └── benchmark_models.py  # Compare different LLMs
│
├── pyproject.toml           # Dependencies (minimal)
├── Makefile                 # dev commands
└── README.md
```

**Desktop app:** Keep in existing `desktop/` folder, just simplify components.

---

## Phase 1: Core Reply Generation (MVP)

### 1.1 Style Analyzer

Analyzes the user's texting patterns from their sent messages.

```python
# v2/core/generation/style_analyzer.py

from dataclasses import dataclass

@dataclass
class UserStyle:
    avg_word_count: float       # e.g., 8.5
    avg_char_count: float       # e.g., 42
    uses_emoji: bool            # True
    emoji_frequency: float      # 0.3 (30% of messages)
    capitalization: str         # "lowercase" | "normal" | "all_caps"
    uses_abbreviations: bool    # True (u, ur, lol, etc.)
    punctuation_style: str      # "minimal" | "normal" | "expressive!!!"
    common_phrases: list[str]   # ["haha", "sounds good", "omg"]
    enthusiasm_level: str       # "high" | "medium" | "low"
    response_speed: str         # "quick" | "delayed" (from timestamps)


def analyze_user_style(user_messages: list[dict]) -> UserStyle:
    """Analyze user's texting style from their sent messages.

    Args:
        user_messages: List of messages sent BY the user
                      [{"text": "...", "timestamp": ...}, ...]

    Returns:
        UserStyle with detected patterns
    """
    texts = [m["text"] for m in user_messages if m.get("text")]

    if not texts:
        return _default_style()

    # Word/char counts
    word_counts = [len(t.split()) for t in texts]
    char_counts = [len(t) for t in texts]

    # Emoji detection
    emoji_messages = [t for t in texts if _has_emoji(t)]

    # Capitalization
    lowercase_ratio = sum(1 for t in texts if t == t.lower()) / len(texts)

    # Abbreviations
    abbrevs = {"u", "ur", "r", "lol", "lmao", "omg", "idk", "tbh", "ngl", "rn"}
    uses_abbrevs = any(
        any(word.lower() in abbrevs for word in t.split())
        for t in texts
    )

    # Common phrases (most frequent 2-3 word sequences)
    common = _extract_common_phrases(texts)

    # Enthusiasm (exclamation marks, caps, emoji density)
    enthusiasm = _detect_enthusiasm(texts)

    return UserStyle(
        avg_word_count=sum(word_counts) / len(word_counts),
        avg_char_count=sum(char_counts) / len(char_counts),
        uses_emoji=len(emoji_messages) > 0,
        emoji_frequency=len(emoji_messages) / len(texts),
        capitalization="lowercase" if lowercase_ratio > 0.7 else "normal",
        uses_abbreviations=uses_abbrevs,
        punctuation_style=_detect_punctuation_style(texts),
        common_phrases=common[:5],
        enthusiasm_level=enthusiasm,
        response_speed="quick",  # TODO: analyze from timestamps
    )


def style_to_prompt_instructions(style: UserStyle) -> str:
    """Convert style analysis to prompt instructions."""

    instructions = []

    # Length
    if style.avg_word_count < 6:
        instructions.append("Keep replies very short (under 6 words)")
    elif style.avg_word_count < 12:
        instructions.append("Keep replies brief (under 12 words)")
    else:
        instructions.append("Medium length replies are okay (under 20 words)")

    # Capitalization
    if style.capitalization == "lowercase":
        instructions.append("Use lowercase (no capitals)")

    # Emoji
    if style.uses_emoji and style.emoji_frequency > 0.3:
        instructions.append("Use emojis occasionally")
    elif not style.uses_emoji:
        instructions.append("Don't use emojis")

    # Abbreviations
    if style.uses_abbreviations:
        instructions.append("Casual abbreviations okay (u, ur, lol)")

    # Enthusiasm
    if style.enthusiasm_level == "high":
        instructions.append("Enthusiastic, use exclamation marks")
    elif style.enthusiasm_level == "low":
        instructions.append("Calm, understated tone")

    return "\n".join(f"- {i}" for i in instructions)
```

### 1.2 Context Analyzer

Analyzes the current conversation to understand what kind of reply is needed.

```python
# v2/core/generation/context_analyzer.py

from dataclasses import dataclass
from enum import Enum

class MessageIntent(Enum):
    YES_NO_QUESTION = "yes_no_question"      # "Want to grab dinner?"
    OPEN_QUESTION = "open_question"          # "What time works?"
    CHOICE_QUESTION = "choice_question"      # "Italian or Mexican?"
    STATEMENT = "statement"                   # "The meeting is at 3"
    EMOTIONAL = "emotional"                   # "I'm so stressed"
    GREETING = "greeting"                     # "Hey! How are you?"
    LOGISTICS = "logistics"                   # "I'm running 10 min late"
    SHARING = "sharing"                       # "Check out this restaurant"


class RelationshipType(Enum):
    CLOSE_FRIEND = "close_friend"
    CASUAL_FRIEND = "casual_friend"
    FAMILY = "family"
    WORK = "work"
    ACQUAINTANCE = "acquaintance"
    UNKNOWN = "unknown"


@dataclass
class ConversationContext:
    last_message: str
    last_message_intent: MessageIntent
    conversation_topic: str          # "dinner plans", "work", "catching up"
    relationship_type: RelationshipType
    urgency: str                     # "high" | "normal" | "low"
    awaiting_response: bool          # True if they asked something
    conversation_mood: str           # "positive" | "neutral" | "negative"
    thread_summary: str              # Brief summary of recent exchange


def analyze_context(messages: list[dict]) -> ConversationContext:
    """Analyze conversation to understand context for reply.

    Args:
        messages: Recent messages [{"sender": "them"|"me", "text": "...", ...}]

    Returns:
        ConversationContext with analysis
    """
    if not messages:
        return _default_context()

    last_msg = messages[-1]
    last_text = last_msg.get("text", "")

    # Detect intent of last message
    intent = _detect_intent(last_text)

    # Detect relationship from conversation patterns
    relationship = _detect_relationship(messages)

    # Detect topic
    topic = _detect_topic(messages[-10:])  # Last 10 messages

    # Check if we need to respond
    awaiting = last_msg.get("sender") != "me" and intent in [
        MessageIntent.YES_NO_QUESTION,
        MessageIntent.OPEN_QUESTION,
        MessageIntent.CHOICE_QUESTION,
    ]

    # Mood detection
    mood = _detect_mood(messages[-5:])

    # Brief summary
    summary = _summarize_thread(messages[-10:])

    return ConversationContext(
        last_message=last_text,
        last_message_intent=intent,
        conversation_topic=topic,
        relationship_type=relationship,
        urgency=_detect_urgency(last_text),
        awaiting_response=awaiting,
        conversation_mood=mood,
        thread_summary=summary,
    )


def _detect_intent(text: str) -> MessageIntent:
    """Detect the intent of a message."""
    text_lower = text.lower().strip()

    # Question detection
    if text.endswith("?"):
        # Yes/No patterns
        yes_no_starters = [
            "do you", "are you", "can you", "will you", "would you",
            "want to", "wanna", "could you", "should we", "shall we",
            "is it", "are we", "did you", "have you", "has ",
        ]
        if any(text_lower.startswith(s) for s in yes_no_starters):
            return MessageIntent.YES_NO_QUESTION

        # Choice patterns
        if " or " in text_lower:
            return MessageIntent.CHOICE_QUESTION

        return MessageIntent.OPEN_QUESTION

    # Greeting patterns
    greetings = ["hey", "hi", "hello", "what's up", "how are", "how's it"]
    if any(text_lower.startswith(g) for g in greetings):
        return MessageIntent.GREETING

    # Emotional patterns
    emotional_words = ["stressed", "sad", "happy", "excited", "worried", "anxious", "love", "hate", "ugh", "omg"]
    if any(w in text_lower for w in emotional_words):
        return MessageIntent.EMOTIONAL

    # Logistics patterns
    logistics_words = ["running late", "on my way", "be there", "arrived", "leaving now", "eta"]
    if any(w in text_lower for w in logistics_words):
        return MessageIntent.LOGISTICS

    # Sharing patterns (links, recommendations)
    if "http" in text_lower or "check out" in text_lower or "you should try" in text_lower:
        return MessageIntent.SHARING

    return MessageIntent.STATEMENT


def context_to_reply_strategy(context: ConversationContext) -> dict:
    """Convert context to reply generation strategy."""

    strategy = {
        "reply_types": [],
        "tone": "casual",
        "include_question": False,
        "max_length": 15,
    }

    # Based on intent, decide what types of replies to generate
    match context.last_message_intent:
        case MessageIntent.YES_NO_QUESTION:
            strategy["reply_types"] = ["enthusiastic_yes", "polite_no", "conditional"]

        case MessageIntent.OPEN_QUESTION:
            strategy["reply_types"] = ["direct_answer", "answer_with_followup", "deflect"]

        case MessageIntent.CHOICE_QUESTION:
            strategy["reply_types"] = ["pick_first", "pick_second", "suggest_alternative"]

        case MessageIntent.STATEMENT:
            strategy["reply_types"] = ["acknowledge", "react", "ask_followup"]
            strategy["include_question"] = True

        case MessageIntent.EMOTIONAL:
            strategy["reply_types"] = ["supportive", "empathetic", "offer_help"]
            strategy["tone"] = "warm"

        case MessageIntent.GREETING:
            strategy["reply_types"] = ["greeting_back", "greeting_with_news", "greeting_with_question"]

        case MessageIntent.LOGISTICS:
            strategy["reply_types"] = ["acknowledge", "helpful_response", "brief_confirm"]
            strategy["max_length"] = 8

        case MessageIntent.SHARING:
            strategy["reply_types"] = ["positive_reaction", "interested_question", "thanks"]

    # Adjust for relationship
    if context.relationship_type == RelationshipType.WORK:
        strategy["tone"] = "professional"
        strategy["max_length"] = 20
    elif context.relationship_type == RelationshipType.CLOSE_FRIEND:
        strategy["tone"] = "very_casual"

    return strategy
```

### 1.3 Prompt Templates

```python
# v2/core/generation/prompts.py

REPLY_GENERATION_PROMPT = """You are helping draft iMessage replies. Generate exactly 3 reply options.

## Conversation Context
{thread_summary}

## Recent Messages
{formatted_messages}

## Last Message (needs reply)
{last_message}

## Your Texting Style
{style_instructions}

## Reply Strategy
Generate these types of replies:
{reply_types}

Tone: {tone}
Max length: {max_length} words each

## Rules
- Sound natural, like a real text message
- Match the conversation's energy and tone
- No quotes around the replies
- No explanations, just the reply text
- Each reply should be meaningfully different

## Examples of good iMessage replies

For "Want to grab dinner tonight?":
- yes! where were you thinking?
- can't tonight, rain check?
- what time works?

For "The meeting got moved to 3pm":
- got it, thanks for the heads up
- works for me 👍
- okay sounds good

For "I'm so stressed about this deadline":
- ugh that sucks, anything i can help with?
- you got this! almost done
- want to talk about it?

## Generate 3 replies for the last message:

1."""


FEW_SHOT_EXAMPLES = {
    "yes_no_question": [
        ("Want to grab coffee tomorrow?", [
            "yes! morning or afternoon?",
            "can't tomorrow, wednesday work?",
            "sure, where were you thinking?"
        ]),
        ("Are you coming to the party?", [
            "definitely! what time?",
            "probably not, super tired",
            "maybe, who's going?"
        ]),
        ("Do you have the notes from class?", [
            "yeah i'll send them over",
            "i missed that one too lol",
            "let me check, one sec"
        ]),
    ],
    "open_question": [
        ("What time works for you?", [
            "anytime after 5 works",
            "how about 7?",
            "pretty flexible, you pick"
        ]),
        ("Where should we eat?", [
            "that thai place on main?",
            "down for anything tbh",
            "somewhere with outdoor seating?"
        ]),
    ],
    "emotional": [
        ("I'm so stressed about this deadline", [
            "ugh that sucks, you got this tho",
            "anything i can help with?",
            "when's it due? want to talk it out?"
        ]),
        ("I got the job!!!", [
            "YESSS congrats!!! 🎉",
            "omg that's amazing!! so happy for you",
            "knew you would! we gotta celebrate"
        ]),
    ],
    "statement": [
        ("The meeting got moved to 3pm", [
            "got it, thanks for the heads up",
            "works for me 👍",
            "okay cool, same room?"
        ]),
        ("I'll be there in 10", [
            "sounds good, see you soon",
            "perfect, i'll grab us a table",
            "no rush!"
        ]),
    ],
}


def build_reply_prompt(
    messages: list[dict],
    style: "UserStyle",
    context: "ConversationContext",
    strategy: dict,
) -> str:
    """Build the full prompt for reply generation."""

    # Format messages
    formatted = []
    for msg in messages[-15:]:  # Last 15 messages
        sender = "You" if msg.get("is_from_me") else msg.get("sender", "Them")
        formatted.append(f"{sender}: {msg.get('text', '')}")

    # Get relevant few-shot examples
    intent_key = context.last_message_intent.value.replace("_question", "")
    if intent_key not in FEW_SHOT_EXAMPLES:
        intent_key = "statement"
    examples = FEW_SHOT_EXAMPLES.get(intent_key, FEW_SHOT_EXAMPLES["statement"])

    # Format reply types
    reply_types = "\n".join(f"- {rt}" for rt in strategy["reply_types"])

    return REPLY_GENERATION_PROMPT.format(
        thread_summary=context.thread_summary,
        formatted_messages="\n".join(formatted),
        last_message=context.last_message,
        style_instructions=style_to_prompt_instructions(style),
        reply_types=reply_types,
        tone=strategy["tone"],
        max_length=strategy["max_length"],
    )
```

### 1.4 Reply Generator (Main Orchestrator)

```python
# v2/core/generation/reply_generator.py

from dataclasses import dataclass
from .style_analyzer import analyze_user_style, UserStyle
from .context_analyzer import analyze_context, context_to_reply_strategy, ConversationContext
from .prompts import build_reply_prompt


@dataclass
class GeneratedReply:
    text: str
    reply_type: str          # e.g., "enthusiastic_yes"
    confidence: float        # How confident we are this is good
    generation_time_ms: float


@dataclass
class ReplyGenerationResult:
    replies: list[GeneratedReply]
    context: ConversationContext
    style: UserStyle
    model_used: str
    total_time_ms: float


class ReplyGenerator:
    """Generates contextual reply suggestions for iMessage conversations."""

    def __init__(self, model_loader):
        self.model_loader = model_loader
        self._style_cache: dict[str, UserStyle] = {}  # chat_id -> style

    def generate_replies(
        self,
        messages: list[dict],
        chat_id: str,
        num_replies: int = 3,
    ) -> ReplyGenerationResult:
        """Generate reply suggestions for a conversation.

        Args:
            messages: Recent messages from the conversation
            chat_id: Conversation identifier (for style caching)
            num_replies: Number of replies to generate (default 3)

        Returns:
            ReplyGenerationResult with generated replies and analysis
        """
        import time
        start = time.time()

        # 1. Analyze user's texting style (cached per conversation)
        style = self._get_or_analyze_style(messages, chat_id)

        # 2. Analyze conversation context
        context = analyze_context(messages)

        # 3. Determine reply strategy
        strategy = context_to_reply_strategy(context)

        # 4. Build prompt
        prompt = build_reply_prompt(messages, style, context, strategy)

        # 5. Generate with LLM
        raw_output = self.model_loader.generate(
            prompt=prompt,
            max_tokens=150,
            temperature=0.8,  # Some creativity
            stop=["\n\n", "4."],  # Stop after 3 replies
        )

        # 6. Parse and validate replies
        replies = self._parse_replies(raw_output, strategy)

        total_time = (time.time() - start) * 1000

        return ReplyGenerationResult(
            replies=replies,
            context=context,
            style=style,
            model_used=self.model_loader.current_model,
            total_time_ms=total_time,
        )

    def _get_or_analyze_style(self, messages: list[dict], chat_id: str) -> UserStyle:
        """Get cached style or analyze from messages."""
        if chat_id in self._style_cache:
            return self._style_cache[chat_id]

        # Filter to only user's messages
        user_messages = [m for m in messages if m.get("is_from_me")]
        style = analyze_user_style(user_messages)

        self._style_cache[chat_id] = style
        return style

    def _parse_replies(self, raw_output: str, strategy: dict) -> list[GeneratedReply]:
        """Parse LLM output into structured replies."""
        replies = []

        # Split by numbered lines
        lines = raw_output.strip().split("\n")

        for i, line in enumerate(lines):
            # Remove numbering (1., 2., 3., -, *)
            text = line.strip()
            for prefix in ["1.", "2.", "3.", "-", "*", "1)", "2)", "3)"]:
                if text.startswith(prefix):
                    text = text[len(prefix):].strip()
                    break

            # Skip empty or too long
            if not text or len(text) > 200:
                continue

            # Remove quotes if present
            if text.startswith('"') and text.endswith('"'):
                text = text[1:-1]

            reply_type = strategy["reply_types"][i] if i < len(strategy["reply_types"]) else "general"

            replies.append(GeneratedReply(
                text=text,
                reply_type=reply_type,
                confidence=0.8,  # TODO: Could score based on length/style match
                generation_time_ms=0,  # Set per-reply if needed
            ))

            if len(replies) >= 3:
                break

        return replies

    def clear_style_cache(self, chat_id: str = None):
        """Clear cached style analysis."""
        if chat_id:
            self._style_cache.pop(chat_id, None)
        else:
            self._style_cache.clear()
```

---

## Phase 2: Hybrid Search (After MVP)

### Components to Add

```
v2/core/search/
├── __init__.py
├── indexer.py       # Build FTS5 + vector indexes
├── bm25.py          # SQLite FTS5 wrapper
├── vector.py        # sqlite-vec wrapper
├── hybrid.py        # RRF fusion
└── reranker.py      # Cross-encoder re-ranking
```

### Index Schema

```sql
-- FTS5 for BM25
CREATE VIRTUAL TABLE messages_fts USING fts5(
    text,
    sender,
    content='messages',
    content_rowid='rowid'
);

-- Vector index (sqlite-vec)
CREATE VIRTUAL TABLE messages_vec USING vec0(
    rowid INTEGER PRIMARY KEY,
    embedding FLOAT[384]  -- or 768 depending on model
);
```

---

## What to Copy from v1

| Component | v1 Location | Action |
|-----------|-------------|--------|
| iMessage reader | `integrations/imessage/reader.py` | Copy & simplify |
| MLX loader | `models/loader.py` | Copy & simplify |
| Model registry | `models/registry.py` | Copy & simplify |
| Config management | `jarvis/config.py` | Copy & simplify |

**DO NOT COPY:**
- CLI (`jarvis/cli.py`)
- Complex templates (`models/templates.py`)
- 130+ API endpoints
- Benchmarks, experiments, insights, metrics
- Most Svelte components (keep ~6)

---

## Tauri App Simplification

### Keep These Components

```
desktop/src/lib/components/
├── Sidebar.svelte           # Navigation
├── ConversationList.svelte  # List of chats
├── MessageView.svelte       # Messages + reply suggestions
├── GlobalSearch.svelte      # Search (Phase 2)
├── Settings.svelte          # Basic settings
└── LoadingSpinner.svelte    # UI helper
```

### Remove These Components

```
- AIDraftPanel.svelte
- AttachmentGallery.svelte
- ConversationInsights.svelte
- ConversationStats.svelte
- Dashboard.svelte
- DigestView.svelte
- EventDetection.svelte
- ExperimentDashboard.svelte
- FeedbackCollector.svelte
- HealthStatus.svelte
- PDFExportModal.svelte
- PriorityInbox.svelte
- QualityDashboard.svelte
- SmartReplyChips.svelte (merge into MessageView)
- StreamingMessage.svelte
- SummaryModal.svelte
- TemplateBuilder.svelte
- ThreadedView.svelte
```

---

## API Endpoints (v2)

```python
# v2/api/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="JARVIS v2", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["tauri://localhost", "http://localhost:*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health
@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}

# Conversations
@app.get("/conversations")
async def list_conversations(limit: int = 50):
    """List recent conversations."""
    pass

@app.get("/conversations/{chat_id}/messages")
async def get_messages(chat_id: str, limit: int = 50):
    """Get messages for a conversation."""
    pass

# Generation
@app.post("/generate/replies")
async def generate_replies(chat_id: str, num_replies: int = 3):
    """Generate reply suggestions for a conversation."""
    pass

# Search (Phase 2)
@app.post("/search")
async def search_messages(query: str, limit: int = 20):
    """Hybrid search across all messages."""
    pass

# Settings
@app.get("/settings")
async def get_settings():
    pass

@app.put("/settings")
async def update_settings(settings: dict):
    pass
```

Total: **7 endpoints** (vs 136 in v1)

---

## Development Phases

### Phase 1: Core Reply Generation (Week 1)
- [ ] Set up v2 folder structure
- [ ] Copy/simplify iMessage reader
- [ ] Copy/simplify MLX loader
- [ ] Implement StyleAnalyzer
- [ ] Implement ContextAnalyzer
- [ ] Implement ReplyGenerator
- [ ] Create FastAPI endpoints
- [ ] Test with real conversations

### Phase 2: Tauri Integration (Week 2)
- [ ] Simplify Svelte components
- [ ] Connect to v2 API
- [ ] Reply suggestions UI in MessageView
- [ ] Basic settings

### Phase 3: Hybrid Search (Week 3)
- [ ] FTS5 indexing
- [ ] Vector indexing (sqlite-vec)
- [ ] Hybrid fusion (RRF)
- [ ] Re-ranker integration
- [ ] Search UI

### Phase 4: Polish (Week 4)
- [ ] Test different LLMs
- [ ] Tune prompts
- [ ] Performance optimization
- [ ] README + demo video

---

## Success Criteria

1. **Reply Quality**: Generated replies sound natural and match user's style
2. **Speed**: Replies generate in <2 seconds on M2 MacBook Air
3. **Memory**: Fits in 8GB RAM
4. **Demo-Ready**: Can show to hiring manager and explain every decision

---

## Model Stack (Final)

| Component | Model | Size | Notes |
|-----------|-------|------|-------|
| **Embeddings** | `bge-small-en-v1.5` | ~130MB | Fast, good quality |
| **Re-ranker** | `bge-reranker-base` or Qwen3-Reranker | ~400MB-2GB | Phase 2 |
| **Generation** | Test: Qwen2.5-3B, Phi-3, Gemma-3 | ~2-3GB | Pick best |
