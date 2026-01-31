# JARVIS CLI Guide

Complete reference guide for the JARVIS command-line interface.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Commands](#commands)
  - [chat](#chat---interactive-chat-mode)
  - [search-messages](#search-messages---search-imessage-conversations)
  - [reply](#reply---generate-reply-suggestions)
  - [summarize](#summarize---conversation-summarization)
  - [export](#export---export-conversations)
  - [health](#health---system-health-status)
  - [benchmark](#benchmark---run-performance-benchmarks)
  - [serve](#serve---start-api-server)
  - [mcp-serve](#mcp-serve---start-mcp-server)
  - [version](#version---show-version-information)
- [Configuration](#configuration)
- [Environment Variables](#environment-variables)
- [Keyboard Shortcuts](#keyboard-shortcuts)
- [Troubleshooting](#troubleshooting)
- [Shell Completion](#shell-completion)

## Installation

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.11 or later
- Full Disk Access permission (for iMessage features)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/jwalinshah/jarvis-ai-assistant.git
cd jarvis-ai-assistant

# Install with uv (recommended)
make setup

# Or install with pip
pip install -e .
```

### Verify Installation

```bash
# Check version
jarvis --version

# Run setup wizard to validate environment
uv run python -m jarvis.setup

# Check system health
jarvis health
```

## Quick Start

```bash
# Start an interactive chat session
jarvis chat

# Search for messages
jarvis search-messages "dinner plans"

# Get reply suggestions for a conversation
jarvis reply John

# Summarize a conversation
jarvis summarize Sarah

# Check system status
jarvis health
```

## Commands

### Global Options

These options apply to all commands:

| Option | Description |
|--------|-------------|
| `-v, --verbose` | Enable debug logging for troubleshooting |
| `--version` | Show version information and exit |
| `--examples` | Show detailed usage examples for all commands |
| `-h, --help` | Show help message and exit |

### `chat` - Interactive Chat Mode

Start an interactive chat session with JARVIS. The assistant uses intent-aware routing to understand your requests and provide contextual responses.

**Usage:**
```bash
jarvis chat
```

**Features:**
- Natural language processing with intent classification
- Automatic routing to reply, summarize, or search functions
- Memory-aware operation (adjusts to system resources)
- Template matching for common queries (fast responses)

**Interactive Commands:**
- Type your message and press Enter to send
- Type `quit`, `exit`, or `q` to leave chat mode
- Press `Ctrl+C` to interrupt and exit

**Examples:**
```bash
$ jarvis chat
╭──────────────────────── Chat Mode ────────────────────────╮
│ JARVIS Chat                                               │
│ Type your message and press Enter. Type 'quit' to leave.  │
╰───────────────────────────────────────────────────────────╯
Operating in FULL mode

You: What did John say about the meeting?
JARVIS: Found 3 messages matching 'meeting':
[Jan 15, 14:30] John: Let's reschedule the meeting to 3pm
...

You: Reply to John
JARVIS: [Replying to John]
Their message: "Let's reschedule the meeting to 3pm"

Suggested reply: Sounds good, 3pm works for me!

You: quit
Goodbye!
```

### `search-messages` - Search iMessage Conversations

Search through your iMessage conversations with powerful filtering options.

**Usage:**
```bash
jarvis search-messages <query> [options]
```

**Arguments:**
| Argument | Description |
|----------|-------------|
| `query` | Search query string (required) |

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `-l, --limit <n>` | Maximum number of results | 20 |
| `--start-date <date>` | Filter messages after this date | None |
| `--end-date <date>` | Filter messages before this date | None |
| `--sender <name>` | Filter by sender (use 'me' for your messages) | None |
| `--has-attachment` | Show only messages with attachments | None |
| `--no-attachment` | Show only messages without attachments | None |

**Date Formats:**
- `YYYY-MM-DD` (e.g., `2024-01-15`)
- `YYYY-MM-DD HH:MM` (e.g., `2024-01-15 14:30`)

**Examples:**
```bash
# Basic search
jarvis search-messages "dinner"

# Search with limit
jarvis search-messages "project update" --limit 50

# Filter by date range
jarvis search-messages "meeting" --start-date 2024-01-01 --end-date 2024-01-31

# Filter by sender
jarvis search-messages "lunch" --sender "John"

# Search your own messages
jarvis search-messages "I'll be there" --sender me

# Find messages with attachments
jarvis search-messages "photo" --has-attachment

# Combine filters
jarvis search-messages "birthday" --sender "Mom" --start-date 2024-06-01 --limit 10
```

**Output:**
```
Searching messages for: dinner

Filters: after 2024-01-01, from John

           Search Results (5 messages)
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Date             ┃ Sender ┃ Message                  ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 2024-01-15 18:30 │ John   │ Want to grab dinner?     │
│ 2024-01-15 18:45 │ Me     │ Sure, where?             │
│ 2024-01-15 19:00 │ John   │ How about the Italian... │
└──────────────────┴────────┴──────────────────────────┘
```

### `reply` - Generate Reply Suggestions

Generate intelligent reply suggestions for a conversation. This is an advanced command that analyzes conversation context to suggest appropriate responses.

**Usage:**
```bash
jarvis reply <person> [options]
```

**Arguments:**
| Argument | Description |
|----------|-------------|
| `person` | Name of the person to reply to (required) |

**Options:**
| Option | Description |
|--------|-------------|
| `-i, --instruction <text>` | Optional instruction to guide the reply |

**Examples:**
```bash
# Basic reply generation
jarvis reply John

# Reply with specific instruction
jarvis reply Sarah -i "say yes but ask about timing"

# Reply agreeing to a request
jarvis reply Mom --instruction "accept the invitation warmly"

# Reply declining politely
jarvis reply Boss -i "decline politely, mention prior commitment"
```

**Output:**
```
Generating reply for conversation with John...

Last message from John:
  "Are you free for coffee tomorrow?"

Generating suggestions...

Suggested replies:

  1. Sure, I'd love to! What time works for you?

  2. Tomorrow sounds great! Morning or afternoon?

  3. Yes, I'm free! Should we meet at the usual place?
```

### `summarize` - Conversation Summarization

Generate a summary of a conversation with a specific person.

**Usage:**
```bash
jarvis summarize <person> [options]
```

**Arguments:**
| Argument | Description |
|----------|-------------|
| `person` | Name of the person/conversation to summarize (required) |

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `-n, --messages <n>` | Number of messages to include | 50 |

**Examples:**
```bash
# Summarize recent conversation
jarvis summarize John

# Summarize more messages
jarvis summarize Sarah -n 100

# Get a quick summary
jarvis summarize Mom --messages 20
```

**Output:**
```
Summarizing conversation with John...

Analyzing 50 messages from January 10, 2024 to January 15, 2024

Generating summary...

╭────────────────────── Summary: John ──────────────────────╮
│ Over the past week, you and John have discussed:          │
│                                                           │
│ • Project deadline moved to Friday                        │
│ • Coffee meeting scheduled for Tuesday at 3pm             │
│ • He recommended a new restaurant downtown                │
│ • You both agreed to review the proposal together         │
╰────────────── 50 messages | Jan 10 - Jan 15 ──────────────╯
```

### `export` - Export Conversations

Export a conversation to a file in various formats (JSON, CSV, or plain text).

**Usage:**
```bash
jarvis export --chat-id <id> [options]
```

**Required Options:**
| Option | Description |
|--------|-------------|
| `--chat-id <id>` | Conversation ID to export (required) |

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `-f, --format <fmt>` | Export format: `json`, `csv`, or `txt` | `json` |
| `-o, --output <file>` | Output file path | Auto-generated |
| `-l, --limit <n>` | Maximum messages to export | 1000 |
| `--include-attachments` | Include attachment info in export (CSV only) | Disabled |

**Examples:**
```bash
# Export to JSON (default format)
jarvis export --chat-id chat123456

# Export to CSV format
jarvis export --chat-id chat123456 -f csv

# Export to plain text
jarvis export --chat-id chat123456 --format txt

# Export with custom output path
jarvis export --chat-id chat123456 -o ~/Desktop/conversation.json

# Export limited messages
jarvis export --chat-id chat123456 --limit 500

# Export CSV with attachment information
jarvis export --chat-id chat123456 -f csv --include-attachments

# Full example with all options
jarvis export --chat-id chat123456 -f csv -o export.csv -l 200 --include-attachments
```

**Finding Chat IDs:**

To find the chat ID for a conversation, you can list available conversations when an invalid ID is provided:

```bash
# This will show available conversations if the ID doesn't exist
jarvis export --chat-id unknown
```

**Output Formats:**

| Format | Description | Best For |
|--------|-------------|----------|
| `json` | Structured JSON with full metadata | Programmatic access, backups |
| `csv` | Comma-separated values | Spreadsheet analysis, Excel |
| `txt` | Plain text, human-readable | Reading, printing, sharing |

**Output:**
```
Exporting conversation: chat123456

Exporting 150 messages...

Successfully exported to: conversation_chat123456_20240115.json
Format: JSON
Messages: 150
```

### `health` - System Health Status

Display comprehensive system health information including memory status, feature availability, and model status.

**Usage:**
```bash
jarvis health
```

**Output:**
```
╭───────────────────── Health Check ─────────────────────╮
│                   JARVIS System Health                 │
╰────────────────────────────────────────────────────────╯

           Memory Status
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Metric              ┃ Value       ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ Available Memory    │ 6,234 MB    │
│ Used Memory         │ 1,766 MB    │
│ Operating Mode      │ FULL        │
│ Pressure Level      │ normal      │
│ Model Loaded        │ No          │
└─────────────────────┴─────────────┘

           Feature Status
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Feature          ┃ Status   ┃ Details                  ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ chat             │ HEALTHY  │ OK                       │
│ imessage         │ HEALTHY  │ OK                       │
└──────────────────┴──────────┴──────────────────────────┘

           Model Status
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Metric           ┃ Value       ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ Loaded           │ No          │
└──────────────────┴─────────────┘
```

### `benchmark` - Run Performance Benchmarks

Run various performance benchmarks to evaluate system capabilities.

**Usage:**
```bash
jarvis benchmark <type> [options]
```

**Arguments:**
| Argument | Description |
|----------|-------------|
| `type` | Benchmark type: `memory`, `latency`, or `hhem` (required) |

**Options:**
| Option | Description |
|--------|-------------|
| `-o, --output <file>` | Output file for results (JSON format) |

**Benchmark Types:**

| Type | Description | Gate Criteria |
|------|-------------|---------------|
| `memory` | Profile model memory usage | Pass: <5.5GB, Fail: >6.5GB |
| `latency` | Measure response times (cold/warm/hot) | Pass: warm <3s, cold <15s |
| `hhem` | Evaluate hallucination scores | Pass: mean >=0.5 |

**Examples:**
```bash
# Run memory benchmark
jarvis benchmark memory

# Run latency benchmark and save results
jarvis benchmark latency --output latency_results.json

# Run hallucination evaluation
jarvis benchmark hhem -o hhem_results.json
```

**Note:** Memory and latency benchmarks require Apple Silicon (MLX).

### `serve` - Start API Server

Start the FastAPI REST server for integration with the Tauri desktop application or other clients.

**Usage:**
```bash
jarvis serve [options]
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--host <addr>` | Host address to bind to | 127.0.0.1 |
| `-p, --port <n>` | Port number to bind to | 8000 |
| `--reload` | Enable auto-reload for development | Disabled |

**Examples:**
```bash
# Start with defaults (localhost:8000)
jarvis serve

# Start on all interfaces
jarvis serve --host 0.0.0.0

# Start on custom port
jarvis serve --port 3000

# Development mode with auto-reload
jarvis serve --reload

# Production setup
jarvis serve --host 0.0.0.0 --port 8080
```

**API Endpoints:**
Once running, the API provides endpoints at `http://localhost:8000`:
- `GET /health` - Health check
- `POST /chat` - Send chat messages
- `GET /messages/search` - Search messages
- `GET /docs` - Interactive API documentation (Swagger UI)

### `mcp-serve` - Start MCP Server

Start the Model Context Protocol (MCP) server for integration with Claude Code or other MCP-compatible clients.

**Usage:**
```bash
jarvis mcp-serve [options]
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--transport <mode>` | Transport mode: `stdio` or `http` | `stdio` |
| `--host <addr>` | Host address for HTTP transport | `127.0.0.1` |
| `-p, --port <n>` | Port number for HTTP transport | `8765` |

**Transport Modes:**

| Mode | Description | Use Case |
|------|-------------|----------|
| `stdio` | Standard input/output communication | Claude Code integration, local CLI tools |
| `http` | HTTP server with JSON-RPC | Web clients, remote access |

**Examples:**
```bash
# Start MCP server in stdio mode (default, for Claude Code)
jarvis mcp-serve

# Start MCP server in HTTP mode
jarvis mcp-serve --transport http

# Start HTTP server on custom port
jarvis mcp-serve --transport http --port 9000

# Start HTTP server accessible from other machines
jarvis mcp-serve --transport http --host 0.0.0.0 --port 8765
```

**Available MCP Tools:**

The MCP server exposes the following tools:

| Tool | Description |
|------|-------------|
| `search_messages` | Search iMessage conversations with filters |
| `get_conversations` | List recent conversations |
| `get_messages` | Get messages from a specific conversation |
| `generate_reply` | Generate AI-powered reply suggestions |
| `summarize_conversation` | Generate conversation summaries |

**Claude Code Integration:**

Add to your Claude Code settings (`~/.claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "jarvis": {
      "command": "jarvis",
      "args": ["mcp-serve"]
    }
  }
}
```

For detailed documentation, see [docs/MCP_INTEGRATION.md](MCP_INTEGRATION.md).

### `version` - Show Version Information

Display the current JARVIS version.

**Usage:**
```bash
jarvis version
# or
jarvis --version
```

**Output:**
```
JARVIS AI Assistant v1.0.0
```

## Configuration

JARVIS stores configuration in `~/.jarvis/config.json`. The setup wizard creates this file automatically.

### Configuration File Structure

```json
{
  "config_version": 7,
  "model_path": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
  "template_similarity_threshold": 0.7,
  "memory_thresholds": {
    "full_mode_mb": 8000,
    "lite_mode_mb": 4000
  },
  "ui": {
    "theme": "system",
    "font_size": 14,
    "show_timestamps": true,
    "compact_mode": false
  },
  "search": {
    "default_limit": 50,
    "default_date_range_days": null
  },
  "chat": {
    "stream_responses": true,
    "show_typing_indicator": true
  },
  "routing": {
    "template_threshold": 0.9,
    "context_threshold": 0.7,
    "generate_threshold": 0.5,
    "ab_test_group": "control",
    "ab_test_thresholds": {}
  },
  "model": {
    "model_id": "qwen-1.5b",
    "auto_select": true,
    "max_tokens_reply": 150,
    "max_tokens_summary": 500,
    "temperature": 0.7
  }
}
```

### Configuration Options

#### Model Settings

| Option | Description | Default |
|--------|-------------|---------|
| `model.model_id` | Model identifier (`qwen-0.5b`, `qwen-1.5b`, `qwen-3b`) | `qwen-1.5b` |
| `model.auto_select` | Auto-select model based on available RAM | `true` |
| `model.max_tokens_reply` | Maximum tokens for reply generation | `150` |
| `model.max_tokens_summary` | Maximum tokens for summarization | `500` |
| `model.temperature` | Sampling temperature (0.0-2.0) | `0.7` |

#### Memory Settings

| Option | Description | Default |
|--------|-------------|---------|
| `memory_thresholds.full_mode_mb` | RAM threshold for FULL mode | `8000` |
| `memory_thresholds.lite_mode_mb` | RAM threshold for LITE mode | `4000` |

#### Search Settings

| Option | Description | Default |
|--------|-------------|---------|
| `search.default_limit` | Default search result limit | `50` |
| `search.default_date_range_days` | Default date range (days) | `null` |

#### Template Matching (Deprecated)

| Option | Description | Default |
|--------|-------------|---------|
| `template_similarity_threshold` | **DEPRECATED** - use `routing.template_threshold` instead. Kept for backwards compatibility; non-default values are migrated to `routing.template_threshold` on load. | `0.7` |

#### Routing Thresholds

| Option | Description | Default |
|--------|-------------|---------|
| `routing.template_threshold` | Template decision threshold (0-1) | `0.90` |
| `routing.context_threshold` | Context threshold (0-1) | `0.70` |
| `routing.generate_threshold` | Generation threshold (0-1) | `0.50` |
| `routing.ab_test_group` | A/B group name | `control` |
| `routing.ab_test_thresholds` | Threshold overrides by group | `{}` |

### Running the Setup Wizard

```bash
# Full setup (validates environment, creates config)
uv run python -m jarvis.setup

# Check status only (no modifications)
uv run python -m jarvis.setup --check
```

## Environment Variables

JARVIS respects the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `JARVIS_CONFIG_PATH` | Custom config file path | `~/.jarvis/config.json` |
| `JARVIS_LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | `INFO` |
| `JARVIS_MODEL_CACHE` | Model cache directory | `~/.cache/huggingface` |
| `NO_COLOR` | Disable colored output when set | Not set |

**Examples:**
```bash
# Use custom config
JARVIS_CONFIG_PATH=/path/to/config.json jarvis health

# Enable debug logging
JARVIS_LOG_LEVEL=DEBUG jarvis chat

# Disable colors for piping
NO_COLOR=1 jarvis search-messages "test" | less
```

## Keyboard Shortcuts

### Chat Mode

| Shortcut | Action |
|----------|--------|
| `Enter` | Send message |
| `Ctrl+C` | Cancel current operation / Exit |
| `Ctrl+D` | Exit chat mode |
| `Ctrl+L` | Clear screen (terminal dependent) |
| `Up Arrow` | Previous input (terminal dependent) |
| `Down Arrow` | Next input (terminal dependent) |

### Interactive Commands

In chat mode, you can also type these commands:

| Command | Action |
|---------|--------|
| `quit`, `exit`, `q` | Exit chat mode |
| `help` | Show available commands |
| `clear` | Clear conversation history |

## Troubleshooting

### Common Issues

#### "Cannot access iMessage database"

**Problem:** JARVIS cannot read your iMessage conversations.

**Solution:**
1. Open System Settings > Privacy & Security > Full Disk Access
2. Add your terminal application (Terminal, iTerm2, etc.)
3. Restart your terminal
4. Run `jarvis health` to verify access

#### "Model system not available"

**Problem:** The ML model cannot be loaded.

**Solution:**
1. Verify you're on Apple Silicon: `uname -m` should show `arm64`
2. Check available memory: `jarvis health`
3. Ensure MLX is installed: `pip install mlx mlx-lm`
4. Try a smaller model in config: set `model.model_id` to `qwen-0.5b`

#### "Could not find conversation with 'Name'"

**Problem:** JARVIS cannot find a contact by name.

**Solution:**
1. Check the exact name as it appears in Messages.app
2. Try using a partial name match
3. Run `jarvis search-messages "name"` to find exact contact names
4. Group chats may use custom names different from contact names

#### "Memory pressure: MINIMAL mode"

**Problem:** System is running in reduced functionality mode due to low memory.

**Solution:**
1. Close other memory-intensive applications
2. Check memory usage: `jarvis health`
3. Consider using a smaller model in configuration
4. Restart the system if memory is fragmented

#### "Error running benchmark: ..."

**Problem:** Benchmarks fail to run.

**Solution:**
1. Ensure you're on Apple Silicon for memory/latency benchmarks
2. Install benchmark dependencies: `pip install -e ".[benchmarks]"`
3. Check system resources: `jarvis health`
4. Run with verbose logging: `jarvis -v benchmark memory`

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
# Run any command with -v flag
jarvis -v chat
jarvis -v search-messages "test"
jarvis -v health
```

Debug logs show:
- Intent classification results
- Database queries executed
- Memory controller decisions
- Model loading status
- Circuit breaker state

### Getting Help

If issues persist:

1. Check the health status: `jarvis health`
2. Run setup validation: `uv run python -m jarvis.setup --check`
3. Review debug logs: `jarvis -v <command>`
4. Check the [GitHub Issues](https://github.com/jwalinshah/jarvis-ai-assistant/issues)

## Shell Completion

JARVIS supports shell completion for bash, zsh, and fish shells via `argcomplete`.

### Installation

```bash
# Install argcomplete
pip install argcomplete

# Enable global completion (one-time setup)
activate-global-python-argcomplete
```

### Bash

Add to `~/.bashrc`:
```bash
eval "$(register-python-argcomplete jarvis)"
```

### Zsh

Add to `~/.zshrc`:
```zsh
autoload -U bashcompinit
bashcompinit
eval "$(register-python-argcomplete jarvis)"
```

### Fish

Create `~/.config/fish/completions/jarvis.fish`:
```fish
register-python-argcomplete --shell fish jarvis | source
```

### Usage

After setup, press `Tab` to complete:
```bash
jarvis <Tab>          # Shows: chat, search-messages, health, ...
jarvis search-<Tab>   # Completes to: search-messages
jarvis benchmark <Tab> # Shows: memory, latency, hhem
```

## Examples

### Daily Workflow

```bash
# Morning: Check what messages need responses
jarvis chat
> What messages do I have from yesterday?

# Reply to important messages
jarvis reply Boss -i "confirm attendance at meeting"
jarvis reply Mom

# Search for specific information
jarvis search-messages "address" --sender "John"

# End of day: Get summaries
jarvis summarize Boss -n 100
```

### Batch Operations

```bash
# Export search results
jarvis search-messages "project" --limit 100 > project_messages.txt

# Run all benchmarks
jarvis benchmark memory -o memory.json
jarvis benchmark latency -o latency.json
jarvis benchmark hhem -o hhem.json
```

### Integration with Scripts

```bash
#!/bin/bash
# health_check.sh - Monitor JARVIS health

if ! jarvis health | grep -q "HEALTHY"; then
    echo "Warning: JARVIS health check failed"
    exit 1
fi
echo "JARVIS is healthy"
```

---

For more information, see the [main documentation](../CLAUDE.md) or run `jarvis --help`.
