# JARVIS API Reference

Complete REST API documentation for the JARVIS iMessage Assistant backend.

**Base URL:** `http://localhost:8742`

**Documentation URLs:**
- Swagger UI: http://localhost:8742/docs
- ReDoc: http://localhost:8742/redoc
- OpenAPI JSON: http://localhost:8742/openapi.json

---

## Table of Contents

- [Authentication](#authentication)
- [Rate Limiting](#rate-limiting)
- [Error Responses](#error-responses)
- [Endpoints](#endpoints)
  - [Health](#health)
  - [Conversations](#conversations)
  - [Drafts (AI Generation)](#drafts-ai-generation)
  - [Suggestions (Quick Replies)](#suggestions-quick-replies)
  - [Settings](#settings)

---

## Authentication

This API is designed for local use by the JARVIS desktop application. **No authentication is required** as the API only binds to localhost.

## Rate Limiting

No rate limiting is applied. The API is designed for single-user local access.

## Error Responses

All errors return a JSON response with the following structure:

```json
{
    "error": "Brief error message",
    "detail": "Detailed explanation of the error",
    "code": "MACHINE_READABLE_CODE"
}
```

### Common Error Codes

| HTTP Status | Code | Description |
|-------------|------|-------------|
| 400 | `VALIDATION_ERROR` | Invalid request parameters |
| 403 | `PERMISSION_DENIED` | Full Disk Access not granted |
| 404 | `NOT_FOUND` | Resource not found |
| 500 | `INTERNAL_ERROR` | Server error |
| 503 | `SERVICE_UNAVAILABLE` | Model not loaded |

---

## Endpoints

### Health

System health monitoring and status checks.

---

#### GET /

Root endpoint - simple health ping.

**Response:**
```json
{
    "status": "ok",
    "service": "jarvis-api"
}
```

**curl:**
```bash
curl http://localhost:8742/
```

---

#### GET /health

Get comprehensive system health status.

**Response:**
```json
{
    "status": "healthy",
    "imessage_access": true,
    "memory_available_gb": 12.5,
    "memory_used_gb": 3.5,
    "memory_mode": "FULL",
    "model_loaded": true,
    "permissions_ok": true,
    "jarvis_rss_mb": 256.5,
    "jarvis_vms_mb": 1024.0,
    "model": {
        "id": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        "display_name": "Qwen 0.5B (Fast)",
        "loaded": true,
        "memory_usage_mb": 450.5,
        "quality_tier": "basic"
    },
    "recommended_model": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    "system_ram_gb": 16.0
}
```

**Health Status Values:**
- `healthy`: All systems operational
- `degraded`: Running with reduced capability (low memory)
- `unhealthy`: Critical issue (no iMessage access)

**Memory Modes:**
- `FULL`: >= 4GB available
- `LITE`: 2-4GB available
- `MINIMAL`: < 2GB available

**curl:**
```bash
curl http://localhost:8742/health
```

---

### Conversations

iMessage conversation and message management.

---

#### GET /conversations

List recent conversations sorted by last message date.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | int | 50 | Max conversations (1-500) |
| `since` | datetime | - | Only convos with messages after this date |
| `before` | datetime | - | Pagination cursor |

**Response:**
```json
[
    {
        "chat_id": "chat123456789",
        "participants": ["+15551234567"],
        "display_name": "John Doe",
        "last_message_date": "2024-01-15T10:30:00Z",
        "message_count": 150,
        "is_group": false,
        "last_message_text": "See you later!"
    }
]
```

**curl:**
```bash
# List 50 most recent conversations
curl http://localhost:8742/conversations

# With pagination
curl "http://localhost:8742/conversations?limit=20&before=2024-01-10T08:00:00Z"

# Only conversations since a date
curl "http://localhost:8742/conversations?since=2024-01-01T00:00:00Z"
```

---

#### GET /conversations/{chat_id}/messages

Get messages for a specific conversation.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `chat_id` | string | Unique conversation identifier |

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | int | 100 | Max messages (1-1000) |
| `before` | datetime | - | Only messages before this date |

**Response:**
```json
[
    {
        "id": 12345,
        "chat_id": "chat123456789",
        "sender": "+15551234567",
        "sender_name": "John Doe",
        "text": "Hey, are you free for lunch?",
        "date": "2024-01-15T10:30:00Z",
        "is_from_me": false,
        "attachments": [],
        "reply_to_id": null,
        "reactions": [
            {
                "type": "love",
                "sender": "+15559876543",
                "sender_name": "Jane",
                "date": "2024-01-15T10:31:00Z"
            }
        ],
        "is_system_message": false
    }
]
```

**curl:**
```bash
# Get latest 100 messages
curl http://localhost:8742/conversations/chat123456789/messages

# Get older messages (pagination)
curl "http://localhost:8742/conversations/chat123456789/messages?limit=50&before=2024-01-10T00:00:00Z"
```

---

#### GET /conversations/search

Search messages across all conversations.

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `q` | string | Yes | Search query (min 1 char) |
| `limit` | int | No | Max results (1-500, default 50) |
| `sender` | string | No | Filter by sender phone/email |
| `after` | datetime | No | Messages after this date |
| `before` | datetime | No | Messages before this date |
| `chat_id` | string | No | Filter to specific conversation |
| `has_attachments` | bool | No | Filter by attachment presence |

**Response:**
```json
[
    {
        "id": 12345,
        "chat_id": "chat123456789",
        "sender": "+15551234567",
        "sender_name": "John Doe",
        "text": "Let's meet for dinner tomorrow at 7pm",
        "date": "2024-01-15T10:30:00Z",
        "is_from_me": false,
        "attachments": [],
        "reactions": []
    }
]
```

**curl:**
```bash
# Basic search
curl "http://localhost:8742/conversations/search?q=dinner"

# Search with filters
curl "http://localhost:8742/conversations/search?q=meeting&sender=+15551234567&after=2024-01-01T00:00:00Z"

# Search for messages with attachments
curl "http://localhost:8742/conversations/search?q=photo&has_attachments=true"
```

---

#### POST /conversations/{chat_id}/send

Send a text message to a conversation.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `chat_id` | string | Conversation identifier |

**Request Body:**
```json
{
    "text": "Hey, are you free for lunch?",
    "recipient": "+15551234567",
    "is_group": false
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | Yes | Message text (1-10000 chars) |
| `recipient` | string | For individual | Recipient phone/email |
| `is_group` | bool | No | True for group chats |

**Response:**
```json
{
    "success": true,
    "error": null
}
```

**curl:**
```bash
# Send to individual
curl -X POST http://localhost:8742/conversations/chat123/send \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello!", "recipient": "+15551234567"}'

# Send to group
curl -X POST http://localhost:8742/conversations/chat123/send \
  -H "Content-Type: application/json" \
  -d '{"text": "Hey everyone!", "is_group": true}'
```

---

#### POST /conversations/{chat_id}/send-attachment

Send a file attachment to a conversation.

**Request Body:**
```json
{
    "file_path": "/Users/john/Documents/photo.jpg",
    "recipient": "+15551234567",
    "is_group": false
}
```

**curl:**
```bash
curl -X POST http://localhost:8742/conversations/chat123/send-attachment \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/Users/john/photo.jpg", "recipient": "+15551234567"}'
```

---

### Drafts (AI Generation)

AI-powered draft generation using the local MLX language model.

---

#### POST /drafts/reply

Generate AI-powered reply suggestions for a conversation.

**Request Body:**
```json
{
    "chat_id": "chat123456789",
    "instruction": "accept enthusiastically",
    "num_suggestions": 3,
    "context_messages": 20
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `chat_id` | string | Required | Conversation to generate replies for |
| `instruction` | string | null | Tone/content guidance |
| `num_suggestions` | int | 3 | Number of suggestions (1-5) |
| `context_messages` | int | 20 | Messages for context (5-50) |

**Instruction Examples:**
- `"accept enthusiastically"` - Positive, excited response
- `"politely decline"` - Courteous refusal
- `"ask for more details"` - Clarifying questions
- `"be brief"` - Short, concise reply
- `"be formal"` - Professional tone

**Response:**
```json
{
    "suggestions": [
        {"text": "Yes, I'd love to! What time works for you?", "confidence": 0.9},
        {"text": "Absolutely! Count me in!", "confidence": 0.8},
        {"text": "Sure thing! Looking forward to it!", "confidence": 0.7}
    ],
    "context_used": {
        "num_messages": 20,
        "participants": ["John Doe"],
        "last_message": "Are you free for dinner tonight?"
    }
}
```

**curl:**
```bash
curl -X POST http://localhost:8742/drafts/reply \
  -H "Content-Type: application/json" \
  -d '{
    "chat_id": "chat123456789",
    "instruction": "accept enthusiastically",
    "num_suggestions": 3
  }'
```

---

#### POST /drafts/summarize

Summarize a conversation using AI.

**Request Body:**
```json
{
    "chat_id": "chat123456789",
    "num_messages": 50
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `chat_id` | string | Required | Conversation to summarize |
| `num_messages` | int | 50 | Messages to include (10-200) |

**Response:**
```json
{
    "summary": "Discussion about planning a weekend trip to the beach.",
    "key_points": [
        "Decided on Saturday departure at 9am",
        "Meeting at John's place",
        "Everyone bringing snacks and sunscreen",
        "Return planned for Sunday evening"
    ],
    "date_range": {
        "start": "2024-01-10",
        "end": "2024-01-15"
    }
}
```

**curl:**
```bash
curl -X POST http://localhost:8742/drafts/summarize \
  -H "Content-Type: application/json" \
  -d '{"chat_id": "chat123456789", "num_messages": 100}'
```

---

### Suggestions (Quick Replies)

Fast pattern-based reply suggestions (no model required).

---

#### POST /suggestions

Get smart reply suggestions based on the last message.

**Request Body:**
```json
{
    "last_message": "Want to grab dinner tonight?",
    "num_suggestions": 3
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `last_message` | string | Required | Last received message |
| `num_suggestions` | int | 3 | Number of suggestions (1-5) |

**Response:**
```json
{
    "suggestions": [
        {"text": "I'm in! Where were you thinking?", "score": 0.85},
        {"text": "Sounds good!", "score": 0.3},
        {"text": "Got it!", "score": 0.25}
    ]
}
```

**Score Interpretation:**
- 0.9-1.0: Strong keyword match
- 0.7-0.9: Partial word match
- 0.3 or below: Generic fallback

**Supported Patterns:**
- Time/scheduling: "what time", "are you free", "when"
- Affirmative: "sounds good", "yes", "okay"
- Gratitude: "thanks", "thank you"
- Social: "dinner", "lunch", "coffee"
- Running late: "omw", "on my way"

**curl:**
```bash
curl -X POST http://localhost:8742/suggestions \
  -H "Content-Type: application/json" \
  -d '{"last_message": "Thanks for your help!", "num_suggestions": 3}'
```

---

### Settings

Application configuration management.

---

#### GET /settings

Get current settings including model, generation, behavior, and system info.

**Response:**
```json
{
    "model_id": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    "generation": {
        "temperature": 0.7,
        "max_tokens_reply": 150,
        "max_tokens_summary": 500
    },
    "behavior": {
        "auto_suggest_replies": true,
        "suggestion_count": 3,
        "context_messages_reply": 20,
        "context_messages_summary": 50
    },
    "system": {
        "system_ram_gb": 16.0,
        "current_memory_usage_gb": 8.5,
        "model_loaded": true,
        "model_memory_usage_gb": 0.5,
        "imessage_access": true
    }
}
```

**curl:**
```bash
curl http://localhost:8742/settings
```

---

#### PUT /settings

Update settings (partial update - only provided fields change).

**Request Body:**
```json
{
    "model_id": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    "generation": {
        "temperature": 0.8,
        "max_tokens_reply": 200
    },
    "behavior": {
        "auto_suggest_replies": false
    }
}
```

All fields are optional. Only provided fields are updated.

**curl:**
```bash
# Update just temperature
curl -X PUT http://localhost:8742/settings \
  -H "Content-Type: application/json" \
  -d '{"generation": {"temperature": 0.8}}'

# Update multiple settings
curl -X PUT http://localhost:8742/settings \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    "behavior": {"suggestion_count": 5}
  }'
```

---

#### GET /settings/models

List available models with their status.

**Response:**
```json
[
    {
        "model_id": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        "name": "Qwen 0.5B (Fast)",
        "size_gb": 0.4,
        "quality_tier": "basic",
        "ram_requirement_gb": 4.0,
        "is_downloaded": true,
        "is_loaded": true,
        "is_recommended": false,
        "description": "Fastest responses, good for simple tasks"
    },
    {
        "model_id": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        "name": "Qwen 1.5B (Balanced)",
        "size_gb": 1.0,
        "quality_tier": "good",
        "ram_requirement_gb": 8.0,
        "is_downloaded": true,
        "is_loaded": false,
        "is_recommended": true,
        "description": "Balanced speed and quality"
    }
]
```

**Quality Tiers:**
- `basic`: Fastest responses (0.5B parameters)
- `good`: Balanced speed/quality (1.5B parameters)
- `best`: Highest quality (3B parameters)

**curl:**
```bash
curl http://localhost:8742/settings/models
```

---

#### POST /settings/models/{model_id}/download

Download a model from HuggingFace Hub.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_id` | string | Model ID (e.g., `mlx-community/Qwen2.5-1.5B-Instruct-4bit`) |

**Response:**
```json
{
    "model_id": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    "status": "completed",
    "progress": 100.0
}
```

**Status Values:**
- `downloading`: In progress
- `completed`: Successfully downloaded
- `failed`: Download failed (check `error` field)

**curl:**
```bash
curl -X POST "http://localhost:8742/settings/models/mlx-community/Qwen2.5-1.5B-Instruct-4bit/download"
```

---

#### POST /settings/models/{model_id}/activate

Switch to a different model.

**Response:**
```json
{
    "success": true,
    "model_id": "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
}
```

**Error Response (not downloaded):**
```json
{
    "success": false,
    "model_id": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    "error": "Model not downloaded. Please download first."
}
```

**curl:**
```bash
curl -X POST "http://localhost:8742/settings/models/mlx-community/Qwen2.5-1.5B-Instruct-4bit/activate"
```

---

## Quick Start Examples

### Check System Status

```bash
# Verify API is running
curl http://localhost:8742/

# Get detailed health status
curl http://localhost:8742/health
```

### Browse Conversations

```bash
# List recent conversations
curl http://localhost:8742/conversations

# Get messages from a conversation
curl http://localhost:8742/conversations/chat123456789/messages

# Search for messages
curl "http://localhost:8742/conversations/search?q=dinner&limit=10"
```

### Generate AI Replies

```bash
# Generate reply suggestions
curl -X POST http://localhost:8742/drafts/reply \
  -H "Content-Type: application/json" \
  -d '{"chat_id": "chat123456789", "num_suggestions": 3}'

# Get quick suggestions (no AI)
curl -X POST http://localhost:8742/suggestions \
  -H "Content-Type: application/json" \
  -d '{"last_message": "Thanks!"}'
```

### Manage Models

```bash
# List available models
curl http://localhost:8742/settings/models

# Download a model
curl -X POST "http://localhost:8742/settings/models/mlx-community/Qwen2.5-1.5B-Instruct-4bit/download"

# Activate a model
curl -X POST "http://localhost:8742/settings/models/mlx-community/Qwen2.5-1.5B-Instruct-4bit/activate"
```

---

## Privacy

All data processing happens locally on your Mac. No conversation data, messages, or personal information is ever sent to external servers. The MLX language models run entirely on Apple Silicon.
