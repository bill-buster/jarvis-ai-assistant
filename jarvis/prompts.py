"""Prompt builders for RAG-based generation.

Provides functions to build prompts with conversation context
for reply and summary generation.
"""

# Few-shot examples for reply generation (input, output pairs)
REPLY_EXAMPLES: list[tuple[str, str]] = [
    (
        "Context: [Jan 15, 10:30] John: Hey, are we still meeting for lunch today?\n"
        "Last message: Hey, are we still meeting for lunch today?\n"
        "Generate a reply:",
        "Yes! Looking forward to it. What time works for you?",
    ),
    (
        "Context: [Jan 15, 14:00] Sarah: Thanks so much for helping with the project!\n"
        "[Jan 15, 14:01] You: Happy to help!\n"
        "[Jan 15, 14:05] Sarah: Would you be free to review one more section?\n"
        "Last message: Would you be free to review one more section?\n"
        "Generate a reply:",
        "Of course! Send it over and I'll take a look.",
    ),
    (
        "Context: [Jan 16, 09:00] Mom: Don't forget dinner at 6pm on Sunday!\n"
        "Last message: Don't forget dinner at 6pm on Sunday!\n"
        "Generate a reply:",
        "Got it, thanks for the reminder! See you then.",
    ),
]

# Few-shot examples for summary generation
SUMMARY_EXAMPLES: list[tuple[str, str]] = [
    (
        "Summarize this conversation:\n"
        "[Jan 10, 10:00] John: Hey, want to grab lunch this week?\n"
        "[Jan 10, 10:05] You: Sure! How about Wednesday?\n"
        "[Jan 10, 10:07] John: Wednesday works. 12:30 at the usual place?\n"
        "[Jan 10, 10:10] You: Perfect, see you then!\n",
        "You and John made plans to meet for lunch on Wednesday at 12:30 at your usual spot.",
    ),
    (
        "Summarize this conversation:\n"
        "[Jan 12, 15:00] Sarah: The deadline moved to Friday\n"
        "[Jan 12, 15:02] You: Oh no, that's tight. What still needs to be done?\n"
        "[Jan 12, 15:05] Sarah: Just the final review and formatting\n"
        "[Jan 12, 15:08] You: I can help with formatting if needed\n"
        "[Jan 12, 15:10] Sarah: That would be great! I'll send you the doc\n",
        "Sarah informed you the project deadline moved to Friday. The remaining "
        "tasks are final review and formatting. You offered to help with formatting, "
        "and Sarah will send you the document.",
    ),
]


def build_reply_prompt(
    context: str,
    last_message: str,
    instruction: str | None = None,
) -> str:
    """Build a prompt for generating a reply.

    Args:
        context: Formatted conversation history.
        last_message: The specific message to reply to.
        instruction: Optional user instruction for the reply style.

    Returns:
        Formatted prompt string.
    """
    parts = [
        "You are a helpful assistant generating reply suggestions for iMessage conversations.",
        "Generate a natural, friendly reply based on the conversation context.",
        "Keep replies concise and match the conversational tone.",
        "",
        "Conversation history:",
        context,
        "",
        f"Last message to reply to: {last_message}",
    ]

    if instruction:
        parts.append(f"\nUser instruction: {instruction}")

    parts.append("\nGenerate a reply:")

    return "\n".join(parts)


def build_summary_prompt(
    context: str,
    focus: str | None = None,
) -> str:
    """Build a prompt for generating a conversation summary.

    Args:
        context: Formatted conversation history.
        focus: Optional focus area for the summary (e.g., "action items").

    Returns:
        Formatted prompt string.
    """
    parts = [
        "You are a helpful assistant summarizing iMessage conversations.",
        "Provide a clear, concise summary of the key points discussed.",
        "Include any decisions made, action items, or important information mentioned.",
        "",
        "Conversation to summarize:",
        context,
    ]

    if focus:
        parts.append(f"\nFocus on: {focus}")

    parts.append("\nSummary:")

    return "\n".join(parts)


def build_search_response_prompt(
    query: str,
    results_context: str,
) -> str:
    """Build a prompt for responding about search results.

    Args:
        query: The original search query.
        results_context: Formatted search results.

    Returns:
        Formatted prompt string.
    """
    parts = [
        "You are a helpful assistant analyzing iMessage search results.",
        f"The user searched for: {query}",
        "",
        "Search results:",
        results_context,
        "",
        "Provide a brief summary of what was found:",
    ]

    return "\n".join(parts)
