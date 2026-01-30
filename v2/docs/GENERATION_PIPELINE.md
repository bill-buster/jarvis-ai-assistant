# Reply Generation Pipeline

This document explains how JARVIS generates contextual reply suggestions.

## Overview

The generation pipeline transforms a conversation into reply suggestions through multiple stages:

```
Messages → Analysis → Context Building → Generation → Post-Processing → Replies
```

Each stage is optimized with fast-paths to skip expensive operations when possible.

## Pipeline Stages

### 1. Coherence Filtering

**Purpose**: Extract relevant recent messages, ignoring topic shifts.

**For 1:1 Chats**:
- Find your last reply
- Include what you were replying to (context)
- Limit to 6-8 messages max

**For Group Chats**:
- Use last 5 messages
- Multiple senders detected via sender analysis

```python
# Example output
[
    {"text": "Want to grab dinner?", "is_from_me": False},
    {"text": "Sure, when?", "is_from_me": True},
    {"text": "Tomorrow at 7?", "is_from_me": False}  # ← Reply to this
]
```

### 2. Template Fast-Path

**Purpose**: Skip all analysis for common response patterns.

**How it works**:
1. Match incoming message against learned templates (1500+ triggers)
2. If confidence > 0.75, return template response immediately
3. ~2ms latency vs ~1000ms for full generation

**Example**:
```
Incoming: "sounds good to me"
Template match: "Thanks!" (confidence: 0.82)
→ Return immediately, skip LLM
```

### 3. Style Analysis

**Purpose**: Learn user's texting style from their messages.

**Analyzes**:
- Average message length
- Emoji usage and frequency
- Capitalization pattern (lowercase/normal/ALL CAPS)
- Abbreviation usage (u, ur, lol, tbh, etc.)
- Punctuation style (minimal/normal/expressive)
- Common phrases

**Output**: `UserStyle` object used for prompt instructions.

```python
UserStyle(
    avg_word_count=5.2,
    uses_emoji=True,
    capitalization="lowercase",
    uses_abbreviations=True,
    punctuation_style="minimal"
)
```

### 4. Context Analysis

**Purpose**: Understand the incoming message's intent and context.

**Detects**:
- **Intent**: yes/no question, open question, greeting, thanks, emotional, etc.
- **Mood**: positive, neutral, negative
- **Urgency**: high, normal, low
- **Relationship**: close friend, family, work, etc.

**Example**:
```python
ConversationContext(
    last_message="Want to grab dinner tomorrow?",
    last_sender="John",
    intent=MessageIntent.YES_NO_QUESTION,
    mood="positive",
    urgency="normal",
    relationship=RelationshipType.CLOSE_FRIEND
)
```

### 5. Reply Strategy

**Purpose**: Determine appropriate response type, tone, and length.

Based on detected intent:

| Intent | Reply Types | Tone | Max Length |
|--------|-------------|------|------------|
| yes_no_question | affirmative, negative, deferred | casual | 10 words |
| open_question | informative, questioning | matches context | 20 words |
| greeting | greeting, casual | friendly | 5 words |
| emotional | supportive, empathetic | warm | 15 words |

### 6. Past Replies Lookup

**Purpose**: Find YOUR similar past responses for style learning.

**Process**:
1. Embed incoming message with all-MiniLM-L6-v2
2. Search FAISS index for similar messages from others
3. For each match, find YOUR reply that followed
4. Apply time-weighting (recent replies scored higher)
5. Filter by availability signal (busy/free)

**Time-Weighting Formula**:
```python
final_score = (
    semantic_similarity * 0.85 +
    recency_factor * 0.15 +
    time_of_day_boost +  # +0.1 if same time window
    day_type_boost       # +0.05 if same weekday/weekend
)
```

**Example Output**:
```python
[
    ("Want to get dinner?", "sure, what time?", 0.89),
    ("Dinner tomorrow?", "down!", 0.85),
    ("Let's eat out", "sounds good", 0.82)
]
```

### 7. Availability Signal Detection

**Purpose**: Detect if you're busy/free from recent messages.

**Scans YOUR recent messages for**:

**Busy Signals**:
- "busy", "can't", "exhausted", "swamped", "packed"
- "tired", "slammed", "hectic", "no time"

**Free Signals**:
- "free", "down", "available", "let's do"
- "sounds good", "i'm in", "count me in"

**Usage**:
- If "busy" → boost decline-type past replies (+0.1)
- If "free" → boost accept-type past replies (+0.1)
- Add context hint to prompt: `[Context: You've been busy lately]`

### 8. Contact Profile Loading

**Purpose**: Get comprehensive communication patterns with this contact.

**Provides**:
- Message statistics (sent/received counts)
- Tone analysis (casual/formal)
- Emoji/slang usage patterns
- Topics discussed
- Average response times

**Used for**: Better style instructions in the prompt.

### 9. Style Instructions Building

**Purpose**: Convert analysis into prompt instructions.

**Combines**:
- Contact profile data (if available)
- Style analysis results

**Example Output**:
```
"brief replies (under 10 words), lowercase only, emojis okay, casual tone"
```

### 10. Context Refresh

**Purpose**: For long conversations, re-query based on current topic.

**When**: Conversation has >10 messages in context.

**How**:
1. Extract topic from last 3 messages
2. Query embeddings for relevant historical messages
3. Inject as "[Earlier relevant message]" markers

**Prevents**: Context drift when original topic has shifted.

### 11. Prompt Building

**Purpose**: Construct the final prompt for the LLM.

**Template**:
```
Text message conversation. Reply briefly as {user_name}.

Your past replies to similar messages:
- They said: "Want dinner?" → You: "sure!"
- They said: "Lunch tomorrow?" → You: "down"

[Context: You've been busy lately]

{user_name}: Previous message
Them: Current message to reply to
```

### 12. LLM Generation

**Purpose**: Generate the actual reply text.

**Parameters**:
- `max_tokens`: 30 (short replies)
- `temperature`: 0.2-0.9 (scales up on regeneration)
- `stop`: ["\n", "2.", "##", "Note:", "---"]

**Temperature Scaling**:
| Regeneration # | Temperature |
|----------------|-------------|
| 1st | 0.2 (consistent) |
| 2nd | 0.4 |
| 3rd | 0.6 |
| 4th | 0.8 |
| 5th+ | 0.9 (max variety) |

### 13. Reply Parsing

**Purpose**: Extract clean reply text from model output.

**Steps**:
1. Take first line only
2. Remove prefixes: "Reply:", "Response:", etc.
3. Strip surrounding quotes
4. Remove emojis if profile says no emojis
5. Validate length (2-150 chars)

### 14. Repetition Filtering

**Purpose**: Avoid suggesting recently-used replies.

**Tracks**: Last 5 replies per conversation.

**Filters**: Exact matches (case-insensitive).

### 15. Fallback Handling

**Purpose**: Ensure valid replies even on failure.

**Fallback Sources** (in order):
1. Personal templates from THIS conversation
2. Generic templates by intent type

**Generic Templates**:
```python
fallbacks = {
    "yes_no_question": ["sounds good!", "can't right now", "let me check"],
    "greeting": ["hey!", "hi there", "what's up"],
    "thanks": ["no problem!", "anytime", "you're welcome"],
}
```

## Performance Timing

Each stage is timed and logged:

```
Generation completed in 1250ms -
  template:2ms, coherence:5ms, style:45ms, context:30ms,
  past_replies:120ms, profile:25ms, refresh:50ms,
  LLM:950ms, parse:3ms
```

## Fast-Path Optimization

The pipeline has multiple early-exit points:

```
┌─────────────────────────────────────────────────┐
│ 1. Template Match (>0.75 confidence)            │
│    → Return immediately (~2ms)                   │
├─────────────────────────────────────────────────┤
│ 2. Past Reply Template Match                    │
│    (2+ high-confidence consistent responses)    │
│    → Return immediately (~150ms)                │
├─────────────────────────────────────────────────┤
│ 3. Full LLM Generation                          │
│    → Complete pipeline (~1000-2000ms)           │
└─────────────────────────────────────────────────┘
```

## Example Walkthrough

**Scenario**: John asks "Want to grab dinner tomorrow?"

1. **Coherence**: Extract last 4 messages of conversation
2. **Template**: No match (confidence 0.45)
3. **Style**: You use lowercase, minimal punctuation, occasional emojis
4. **Context**: Intent=YES_NO_QUESTION, Mood=positive, Relationship=close_friend
5. **Strategy**: Reply types=[affirmative, negative, deferred], Tone=casual, MaxLength=10
6. **Past Replies**:
   - "Dinner tonight?" → "sure!" (0.89)
   - "Lunch tomorrow?" → "down" (0.85)
7. **Availability**: "unknown" (no recent busy/free signals)
8. **Profile**: John - close friend, 1542 messages, casual tone
9. **Style Instructions**: "brief replies, lowercase, casual"
10. **Prompt**: Built with past examples and style
11. **LLM**: Generates "sounds good, what time?"
12. **Parse**: Clean text extracted
13. **Filter**: Not recently used
14. **Result**: "sounds good, what time?" (confidence: 0.92)

## Debugging

Enable the debug panel in the frontend to see:
- Full prompt sent to LLM
- Style instructions used
- Past replies found with similarity scores
- Timing breakdown per stage
- Availability signal detected

## Configuration

Key parameters in `reply_generator.py`:

```python
# Template matching
MIN_TEMPLATE_CONFIDENCE = 0.75

# Past replies
PAST_REPLIES_LIMIT = 5
MIN_SIMILARITY = 0.55

# Time weighting
RECENCY_WEIGHT = 0.15
TIME_WINDOW_BOOST = 0.1
DAY_TYPE_BOOST = 0.05

# Availability adjustments
BUSY_ACCEPT_PENALTY = -0.05
BUSY_DECLINE_BOOST = 0.1
FREE_ACCEPT_BOOST = 0.1

# LLM
MAX_TOKENS = 30
INITIAL_TEMPERATURE = 0.2
MAX_TEMPERATURE = 0.9
```
