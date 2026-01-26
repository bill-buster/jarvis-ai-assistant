"""CLI examples module for JARVIS.

Provides detailed usage examples displayed via `jarvis --examples`.
"""

from rich.console import Console
from rich.markdown import Markdown

console = Console()


EXAMPLES_TEXT = """
# JARVIS CLI Examples

## Interactive Chat

Start an interactive chat session:

```bash
jarvis chat
```

In chat mode, JARVIS understands natural language requests:

```
╭──────────────────────── Chat Mode ────────────────────────╮
│ JARVIS Chat                                               │
│ Type your message and press Enter. Type 'quit' to leave.  │
╰───────────────────────────────────────────────────────────╯
Operating in FULL mode

You: What did John say about the project deadline?
JARVIS: Found 3 messages matching 'project deadline':
[Jan 12, 09:15] John: The project deadline is moved to Friday
[Jan 12, 09:20] You: Got it, thanks!
[Jan 13, 14:30] John: Don't forget the deadline is tomorrow

You: Reply to John
JARVIS: [Replying to John]
Their message: "Don't forget the deadline is tomorrow"

Suggested reply: Thanks for the reminder! I'm on track to finish tonight.

You: quit
Goodbye!
```

---

## Message Search

### Basic Search
```bash
jarvis search-messages "dinner"
```

### Search with Limit
```bash
jarvis search-messages "project update" --limit 50
```

### Filter by Sender
```bash
# Messages from a specific person
jarvis search-messages "meeting" --sender "John"

# Your own messages
jarvis search-messages "I'll call you" --sender me
```

### Date Range Filters
```bash
# Messages after a date
jarvis search-messages "birthday" --start-date 2024-06-01

# Messages in a range
jarvis search-messages "vacation" --start-date 2024-07-01 --end-date 2024-07-31
```

### Attachment Filters
```bash
# Messages with attachments
jarvis search-messages "photo" --has-attachment

# Messages without attachments
jarvis search-messages "link" --no-attachment
```

### Combining Filters
```bash
jarvis search-messages "report" \\
  --sender "Boss" \\
  --start-date 2024-01-01 \\
  --limit 100
```

---

## Reply Generation

### Basic Reply
```bash
jarvis reply John
```

Output:
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

### Guided Reply
```bash
# Accept something
jarvis reply Sarah -i "say yes enthusiastically"

# Decline politely
jarvis reply Boss -i "decline, mention prior commitment"

# Ask for more info
jarvis reply Mom --instruction "ask about timing"
```

---

## Conversation Summary

### Basic Summary
```bash
jarvis summarize John
```

Output:
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

### Extended Summary
```bash
jarvis summarize Sarah -n 100
```

---

## Health Check

```bash
jarvis health
```

Output:
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
```

---

## Benchmarks

### Memory Benchmark
```bash
jarvis benchmark memory

# Save results
jarvis benchmark memory -o memory_results.json
```

### Latency Benchmark
```bash
jarvis benchmark latency
jarvis benchmark latency --output latency.json
```

### Hallucination Evaluation
```bash
jarvis benchmark hhem
jarvis benchmark hhem -o hhem_results.json
```

---

## API Server

### Start with Defaults
```bash
jarvis serve
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Custom Configuration
```bash
# Different port
jarvis serve -p 3000

# All interfaces (for network access)
jarvis serve --host 0.0.0.0

# Development mode
jarvis serve --reload
```

---

## Debugging

### Verbose Mode
Add `-v` before any command for debug logging:

```bash
jarvis -v chat
jarvis -v search-messages "test"
jarvis -v health
```

### Setup Validation
```bash
# Full setup
python -m jarvis.setup

# Check only (no changes)
python -m jarvis.setup --check
```

---

## Tips & Tricks

1. **Partial Name Matching**: If a full name doesn't match, try partial:
   ```bash
   jarvis reply "John"  # Instead of "John Smith"
   ```

2. **Quote Queries with Spaces**:
   ```bash
   jarvis search-messages "I'll be there tomorrow"
   ```

3. **Check Permissions**:
   ```bash
   jarvis health  # Shows iMessage access status
   ```

4. **Run Setup Wizard** for first-time configuration:
   ```bash
   python -m jarvis.setup
   ```

For complete documentation, see: docs/CLI_GUIDE.md
"""


def get_examples_text() -> str:
    """Get the examples text for display.

    Returns:
        Markdown-formatted examples text.
    """
    return EXAMPLES_TEXT


def print_examples() -> None:
    """Print examples to the console using rich formatting."""
    md = Markdown(EXAMPLES_TEXT)
    console.print(md)


if __name__ == "__main__":
    print_examples()
