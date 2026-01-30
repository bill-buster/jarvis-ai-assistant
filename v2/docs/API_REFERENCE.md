# JARVIS v2 API Reference

Base URL: `http://localhost:8000`

## Health

### GET /health
Check system health status.

**Response**
```json
{
  "status": "ok",
  "version": "2.0.0",
  "model_loaded": true,
  "imessage_accessible": true
}
```

### GET /health/cache
Get embedding cache statistics.

**Response**
```json
{
  "total_entries": 5432,
  "hits": 12500,
  "misses": 2100,
  "hit_rate": 0.856
}
```

---

## Conversations

### GET /conversations
List recent conversations.

**Query Parameters**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| limit | int | 50 | Max conversations to return |

**Response**
```json
[
  {
    "chat_id": "iMessage;+;chat123456",
    "display_name": "John Doe",
    "participants": ["+15551234567"],
    "last_message_text": "See you tomorrow!",
    "last_message_date": "2024-01-15T10:30:00",
    "message_count": 1542,
    "is_group": false
  }
]
```

### GET /conversations/{chat_id}/messages
Get messages for a conversation.

**Path Parameters**
| Parameter | Type | Description |
|-----------|------|-------------|
| chat_id | string | URL-encoded chat identifier |

**Query Parameters**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| limit | int | 50 | Max messages to return |
| before | string | null | Pagination: get messages before this timestamp |

**Response**
```json
[
  {
    "id": 12345,
    "text": "Hey, what's up?",
    "sender": "+15551234567",
    "sender_name": "John Doe",
    "is_from_me": false,
    "timestamp": "2024-01-15T10:25:00",
    "chat_id": "iMessage;+;chat123456"
  }
]
```

### GET /conversations/{chat_id}/profile
Get contact profile for a conversation.

**Response**
```json
{
  "chat_id": "iMessage;+;chat123456",
  "display_name": "John Doe",
  "relationship_type": "close_friend",
  "relationship_confidence": 0.85,
  "total_messages": 1542,
  "you_sent": 720,
  "they_sent": 822,
  "avg_your_length": 24.5,
  "avg_their_length": 31.2,
  "tone": "casual",
  "uses_emoji": true,
  "uses_slang": true,
  "is_playful": true,
  "topics": [
    {
      "name": "weekend plans",
      "keywords": ["saturday", "dinner", "movie"],
      "message_count": 145,
      "percentage": 9.4
    }
  ],
  "most_active_hours": [19, 20, 21],
  "their_common_phrases": ["what's up", "sounds good"],
  "your_common_phrases": ["lol", "yeah for sure"],
  "summary": "Close friend. Casual, playful tone. Often discuss weekend plans."
}
```

### POST /conversations/{chat_id}/send
Send a message (experimental, may not work reliably).

**Request Body**
```json
{
  "text": "Hey, what's up?"
}
```

**Response**
```json
{
  "success": true,
  "message": "Message sent"
}
```

### POST /conversations/preload-indices
Preload FAISS indices for faster search.

**Request Body**
```json
{
  "chat_ids": ["iMessage;+;chat123456", "iMessage;+;chat789"]
}
```

**Response**
```json
{
  "preloaded": 2
}
```

---

## Generation

### POST /generate/replies
Generate reply suggestions for a conversation.

**Request Body**
```json
{
  "chat_id": "iMessage;+;chat123456",
  "num_replies": 3
}
```

**Response**
```json
{
  "replies": [
    {
      "text": "sounds good!",
      "reply_type": "affirmative",
      "confidence": 0.92
    },
    {
      "text": "let me check my schedule",
      "reply_type": "deferred",
      "confidence": 0.85
    }
  ],
  "model_used": "lfm2.5-1.2b",
  "generation_time_ms": 1250,
  "context": {
    "last_message": "Want to grab dinner tomorrow?",
    "last_sender": "John",
    "intent": "yes_no_question",
    "tone": "casual"
  },
  "debug_info": {
    "style_instructions": "brief replies, lowercase only, emojis okay",
    "past_replies_count": 3,
    "template_match": false,
    "availability_signal": "unknown",
    "timing_breakdown": {
      "template_check_ms": 2,
      "style_analysis_ms": 45,
      "context_analysis_ms": 30,
      "past_replies_ms": 120,
      "llm_generation_ms": 950
    }
  }
}
```

---

## Search

### GET /search
Semantic search across messages.

**Query Parameters**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| q | string | required | Search query |
| limit | int | 20 | Max results |
| chat_id | string | null | Filter to specific conversation |

**Response**
```json
[
  {
    "message_id": 12345,
    "chat_id": "iMessage;+;chat123456",
    "text": "Let's get dinner on Friday",
    "sender": "John Doe",
    "timestamp": "2024-01-10T18:30:00",
    "is_from_me": false,
    "similarity": 0.87
  }
]
```

---

## Settings

### GET /settings
Get current settings.

**Response**
```json
{
  "model": "lfm2.5-1.2b",
  "auto_suggest": true,
  "max_replies": 3
}
```

### PUT /settings
Update settings.

**Request Body**
```json
{
  "model": "qwen3-1.7b",
  "auto_suggest": false,
  "max_replies": 5
}
```

**Response**
```json
{
  "model": "qwen3-1.7b",
  "auto_suggest": false,
  "max_replies": 5
}
```

### GET /settings/models
List available models.

**Response**
```json
[
  {
    "id": "lfm2.5-1.2b",
    "display_name": "LFM 2.5 1.2B",
    "size_gb": 0.5,
    "quality": "excellent",
    "description": "Fast and high quality"
  },
  {
    "id": "qwen3-4b",
    "display_name": "Qwen3 4B",
    "size_gb": 2.1,
    "quality": "best",
    "description": "Maximum quality, slower"
  }
]
```

---

## WebSocket

### GET /ws
WebSocket endpoint for real-time updates.

**Connection**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
```

**Client → Server Messages**

#### GENERATE_REPLIES
```json
{
  "type": "GENERATE_REPLIES",
  "chat_id": "iMessage;+;chat123456",
  "num_replies": 3
}
```

#### WATCH_MESSAGES
```json
{
  "type": "WATCH_MESSAGES",
  "chat_ids": ["iMessage;+;chat123456"]
}
```

#### PING
```json
{
  "type": "PING"
}
```

#### CANCEL
```json
{
  "type": "CANCEL"
}
```

**Server → Client Messages**

#### CONNECTED
```json
{
  "type": "CONNECTED",
  "client_id": "uuid-here"
}
```

#### GENERATION_START
```json
{
  "type": "GENERATION_START",
  "generation_id": "gen-uuid"
}
```

#### REPLY (streaming)
```json
{
  "type": "REPLY",
  "generation_id": "gen-uuid",
  "reply": {
    "text": "sounds good!",
    "reply_type": "affirmative",
    "confidence": 0.92
  }
}
```

#### GENERATION_COMPLETE
```json
{
  "type": "GENERATION_COMPLETE",
  "generation_id": "gen-uuid",
  "total_replies": 3,
  "generation_time_ms": 1250
}
```

#### NEW_MESSAGE
```json
{
  "type": "NEW_MESSAGE",
  "chat_id": "iMessage;+;chat123456",
  "message": {
    "id": 12346,
    "text": "On my way!",
    "sender": "+15551234567",
    "is_from_me": false,
    "timestamp": "2024-01-15T10:35:00"
  }
}
```

#### ERROR
```json
{
  "type": "ERROR",
  "message": "Failed to generate replies",
  "code": "GENERATION_FAILED"
}
```

---

## Error Responses

All endpoints return errors in a consistent format:

```json
{
  "detail": "Error message here"
}
```

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad request (invalid parameters) |
| 404 | Resource not found |
| 500 | Internal server error |
| 503 | Service unavailable (permissions issue) |

### Common Errors

**Permission Denied (503)**
```json
{
  "detail": "Cannot access iMessage database. Grant Full Disk Access in System Settings."
}
```

**Conversation Not Found (404)**
```json
{
  "detail": "Conversation not found: iMessage;+;chat123456"
}
```

**Invalid Model (400)**
```json
{
  "detail": "Unknown model: invalid-model. Available: lfm2.5-1.2b, qwen3-4b, ..."
}
```
