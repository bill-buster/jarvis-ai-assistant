"""Prompt templates and builders for iMessage reply generation.

Provides well-engineered prompts optimized for small local LLMs (Qwen2.5-0.5B/1.5B)
with clear structure, few-shot examples, and tone-aware generation.

This module is the SINGLE SOURCE OF TRUTH for all prompts in the JARVIS system.
Import prompts from here, not from other modules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from jarvis.threading import ThreadContext, ThreadedReplyConfig

# =============================================================================
# Prompt Metadata & Versioning
# =============================================================================

PROMPT_VERSION = "1.0.0"
PROMPT_LAST_UPDATED = "2026-01-26"

# =============================================================================
# Few-Shot Examples
# =============================================================================


@dataclass
class PromptMetadata:
    """Metadata for a prompt or example set.

    Attributes:
        name: Human-readable name for the prompt/example set
        version: Semantic version string (e.g., "1.0.0")
        last_updated: ISO date string of last update
        description: Brief description of the prompt's purpose
    """

    name: str
    version: str = PROMPT_VERSION
    last_updated: str = PROMPT_LAST_UPDATED
    description: str = ""


@dataclass
class FewShotExample:
    """A few-shot example for prompt engineering.

    Attributes:
        context: The conversation context
        output: The expected output/reply
        tone: The tone of the example (casual/professional)
    """

    context: str
    output: str
    tone: Literal["casual", "professional"] = "casual"


# Reply generation examples - casual tone
CASUAL_REPLY_EXAMPLES: list[FewShotExample] = [
    FewShotExample(
        context="[10:30] John: Want to grab lunch?",
        output="Sure! What time works for you?",
        tone="casual",
    ),
    FewShotExample(
        context="[14:00] Mom: Don't forget dinner Sunday",
        output="I'll be there! Should I bring anything?",
        tone="casual",
    ),
    FewShotExample(
        context="[09:15] Alex: Running 10 min late",
        output="No worries, see you soon!",
        tone="casual",
    ),
    FewShotExample(
        context="[18:30] Sam: Did you see the game last night?",
        output="Yes! That ending was incredible",
        tone="casual",
    ),
    FewShotExample(
        context="[11:00] Lisa: Can you pick up milk on your way home?",
        output="Sure thing, anything else?",
        tone="casual",
    ),
]

# Reply generation examples - professional tone
PROFESSIONAL_REPLY_EXAMPLES: list[FewShotExample] = [
    FewShotExample(
        context="[09:00] Manager: Can you send the Q4 report by EOD?",
        output="Absolutely, I'll have it ready by 5 PM.",
        tone="professional",
    ),
    FewShotExample(
        context="[14:30] Client: When can we schedule a follow-up meeting?",
        output="I'm available Thursday or Friday afternoon. Would either work for you?",
        tone="professional",
    ),
    FewShotExample(
        context="[10:15] HR: Please complete your annual review form",
        output="Thank you for the reminder. I'll complete it today.",
        tone="professional",
    ),
    FewShotExample(
        context="[16:00] Colleague: Could you review my proposal draft?",
        output="Happy to help. I'll review it and send feedback by tomorrow morning.",
        tone="professional",
    ),
    FewShotExample(
        context="[11:30] Vendor: The shipment will be delayed by 2 days",
        output="Thanks for letting me know. Please send the updated tracking info when available.",
        tone="professional",
    ),
]

# Summarization examples
SUMMARIZATION_EXAMPLES: list[tuple[str, str]] = [
    (
        """[Mon 9:00] John: Hey, want to meet for coffee Tuesday?
[Mon 9:05] You: Sure, what time?
[Mon 9:10] John: How about 3pm at Blue Bottle?
[Mon 9:12] You: Perfect, see you then""",
        """Summary:
- Coffee meeting planned for Tuesday at 3pm
- Location: Blue Bottle
- Confirmed with John""",
    ),
    (
        """[Tue 14:00] Boss: Can you handle the client presentation Friday?
[Tue 14:05] You: Yes, I can do that
[Tue 14:10] Boss: Great, they want to see the Q3 numbers
[Tue 14:15] Boss: Also include the growth projections
[Tue 14:20] You: Got it, I'll prepare both""",
        """Summary:
- Action item: Prepare client presentation for Friday
- Include: Q3 numbers and growth projections
- Commitment made to prepare materials""",
    ),
    (
        """[Wed 10:00] Mom: Are you coming for Thanksgiving?
[Wed 10:15] You: Yes, I'll be there
[Wed 10:20] Mom: Can you bring your famous pie?
[Wed 10:25] You: Of course!
[Wed 10:30] Mom: Dinner is at 4pm""",
        """Summary:
- Thanksgiving plans confirmed
- Bringing pie (action item)
- Dinner time: 4pm""",
    ),
]

# Search/question answering examples
SEARCH_ANSWER_EXAMPLES: list[tuple[str, str, str]] = [
    (
        """[Mon] John: Let's meet at 123 Main St
[Tue] John: Actually, let's do 456 Oak Ave instead""",
        "Where are we meeting?",
        "You're meeting at 456 Oak Ave (John changed it from the original 123 Main St).",
    ),
    (
        """[10:00] Sarah: The project deadline is March 15th
[10:30] Sarah: We need the draft by March 10th for review""",
        "When is the deadline?",
        "The project deadline is March 15th, with a draft due March 10th for review.",
    ),
    (
        """[Mon] Alex: I recommend trying Sushi Palace
[Mon] Alex: They have great lunch specials
[Tue] Mom: That new Italian place is good too""",
        "What restaurants were recommended?",
        "Two restaurants were recommended: Sushi Palace (by Alex, noted for lunch specials) "
        "and a new Italian place (by Mom).",
    ),
]


# =============================================================================
# Thread-Specific Few-Shot Examples
# =============================================================================

# Examples for logistics/coordination threads - concise responses
LOGISTICS_THREAD_EXAMPLES: list[FewShotExample] = [
    FewShotExample(
        context="[14:00] Sarah: What time works for pickup?\n[14:05] Me: How about 5pm?",
        output="5pm works! See you then.",
        tone="casual",
    ),
    FewShotExample(
        context="[10:00] John: Running late, stuck in traffic\n[10:05] John: Be there in 15",
        output="No worries, take your time!",
        tone="casual",
    ),
    FewShotExample(
        context="[18:30] Mom: Which entrance should I use?",
        output="Use the main entrance on Oak Street. I'll meet you there.",
        tone="casual",
    ),
]

# Examples for emotional support threads - empathetic responses
EMOTIONAL_SUPPORT_THREAD_EXAMPLES: list[FewShotExample] = [
    FewShotExample(
        context=(
            "[20:00] Sarah: Had the worst day at work today\n"
            "[20:05] Sarah: My project got cancelled"
        ),
        output=(
            "I'm so sorry to hear that. That's really frustrating, "
            "especially after all your hard work. Want to talk about it?"
        ),
        tone="casual",
    ),
    FewShotExample(
        context=(
            "[15:00] John: Just found out I didn't get the job\n"
            "[15:02] John: Feeling pretty down"
        ),
        output=(
            "That's really disappointing, I know you were excited about it. "
            "Their loss though - something better will come along. I'm here if you need to vent."
        ),
        tone="casual",
    ),
    FewShotExample(
        context="[09:00] Mom: Missing dad a lot today",
        output=(
            "I know, I miss him too. It's okay to have those days. "
            "Want to look through some old photos together later?"
        ),
        tone="casual",
    ),
]

# Examples for planning threads - includes action items
PLANNING_THREAD_EXAMPLES: list[FewShotExample] = [
    FewShotExample(
        context=(
            "[14:00] Alex: Want to do dinner Saturday?\n"
            "[14:05] Me: Sure!\n"
            "[14:10] Alex: Any restaurant preferences?"
        ),
        output=(
            "How about Italian? There's a new place downtown I've been wanting to try. "
            "I can make a reservation for 7pm if that works?"
        ),
        tone="casual",
    ),
    FewShotExample(
        context=(
            "[10:00] Sarah: We should plan a birthday surprise for Mike\n"
            "[10:05] Sarah: Any ideas?"
        ),
        output=(
            "Definitely! We could do a small surprise at his favorite bar. "
            "I can coordinate with Lisa on decorations. Should we aim for next Friday?"
        ),
        tone="casual",
    ),
    FewShotExample(
        context=(
            "[16:00] John: Thinking about a camping trip next month\n"
            "[16:05] John: You interested?"
        ),
        output=(
            "Count me in! I can bring the tent and cooler. "
            "What dates are you thinking? Should I check with the others too?"
        ),
        tone="casual",
    ),
]

# Examples for catching up threads - warm, conversational
CATCHING_UP_THREAD_EXAMPLES: list[FewShotExample] = [
    FewShotExample(
        context="[19:00] Lisa: Haven't heard from you in ages! How are things?",
        output=(
            "I know, it's been too long! Things are good - busy with work but can't complain. "
            "How about you? Still enjoying the new job?"
        ),
        tone="casual",
    ),
    FewShotExample(
        context=(
            "[11:00] College Friend: Just saw your post about the promotion!\n"
            "[11:02] College Friend: So proud of you!"
        ),
        output=(
            "Thanks so much! It's been a wild ride. "
            "We should catch up properly soon - coffee this weekend?"
        ),
        tone="casual",
    ),
]

# Examples for quick exchange threads - brief responses
QUICK_EXCHANGE_THREAD_EXAMPLES: list[FewShotExample] = [
    FewShotExample(
        context="[12:00] Tom: Got the tickets!",
        output="Awesome, thanks!",
        tone="casual",
    ),
    FewShotExample(
        context="[09:00] Boss: Can you join the 2pm call?",
        output="Yes, I'll be there.",
        tone="professional",
    ),
]

# Thread examples organized by topic for the registry
THREAD_EXAMPLES: dict[str, list[FewShotExample]] = {
    "logistics": LOGISTICS_THREAD_EXAMPLES,
    "emotional_support": EMOTIONAL_SUPPORT_THREAD_EXAMPLES,
    "planning": PLANNING_THREAD_EXAMPLES,
    "catching_up": CATCHING_UP_THREAD_EXAMPLES,
    "quick_exchange": QUICK_EXCHANGE_THREAD_EXAMPLES,
}


# =============================================================================
# Prompt Templates
# =============================================================================

# Token limit guidance for small models
MAX_PROMPT_TOKENS = 1500  # Reserve space for generation
MAX_CONTEXT_CHARS = 4000  # Approximate, ~4 chars per token


@dataclass
class PromptTemplate:
    """A prompt template with placeholders.

    Attributes:
        name: Template identifier
        system_message: Role/context for the model
        template: Format string with {placeholders}
        max_output_tokens: Suggested max tokens for response
    """

    name: str
    system_message: str
    template: str
    max_output_tokens: int = 100


REPLY_TEMPLATE = PromptTemplate(
    name="reply_generation",
    system_message="You are helping draft a text message reply. Keep replies natural, concise, "
    "and friendly. Match the conversation's tone.",
    template="""### Conversation Context:
{context}

### Instructions:
Generate a natural reply to the last message. The reply should:
- Match the tone of the conversation ({tone})
- Be concise (1-2 sentences for texts)
- Sound like something a real person would send
{custom_instruction}

### Examples:
{examples}

### Last message to reply to:
{last_message}

### Your reply:""",
    max_output_tokens=50,
)


SUMMARY_TEMPLATE = PromptTemplate(
    name="conversation_summary",
    system_message="You are summarizing a text message conversation. Extract key information "
    "concisely and highlight any action items or commitments.",
    template="""### Conversation:
{context}

### Instructions:
Summarize this conversation. Include:
- Key points discussed
- Any action items or commitments made
- Important dates, times, or locations mentioned
{focus_instruction}

### Examples:
{examples}

### Summary:""",
    max_output_tokens=150,
)


SEARCH_ANSWER_TEMPLATE = PromptTemplate(
    name="search_answer",
    system_message="You are answering a question about a text message conversation. "
    "Base your answer only on the provided messages.",
    template="""### Messages:
{context}

### Question:
{question}

### Instructions:
Answer the question based only on the messages above. Be specific and cite relevant details.
If the answer isn't in the messages, say so.

### Examples:
{examples}

### Answer:""",
    max_output_tokens=100,
)


THREADED_REPLY_TEMPLATE = PromptTemplate(
    name="threaded_reply",
    system_message=(
        "You are helping draft a text message reply based on the conversation thread context. "
        "Match the tone and respond appropriately to the thread type."
    ),
    template="""### Thread Context:
Topic: {thread_topic}
State: {thread_state}
Your role: {user_role}
{participants_info}

### Relevant Messages:
{context}

### Instructions:
Generate a natural reply that:
- Matches the thread's {response_style} tone
- Is {length_guidance}
{additional_instructions}
{custom_instruction}

### Examples:
{examples}

### Last message to reply to:
{last_message}

### Your reply:""",
    max_output_tokens=100,
)


# =============================================================================
# Tone Detection
# =============================================================================

# Casual indicators
CASUAL_INDICATORS: set[str] = {
    # Emoji patterns are detected separately
    "lol",
    "haha",
    "hehe",
    "lmao",
    "omg",
    "btw",
    "brb",
    "ttyl",
    "idk",
    "ikr",
    "nvm",
    "tbh",
    "imo",
    "fyi",
    "np",
    "k",
    "kk",
    "ok",
    "yeah",
    "yep",
    "nope",
    "yup",
    "gonna",
    "wanna",
    "gotta",
    "cuz",
    "bc",
    "u",
    "ur",
    "r",
    "y",
    "thx",
    "ty",
    "pls",
    "plz",
    "omw",
    "wya",
    "wassup",
    "sup",
    "hey",
    "yo",
    "dude",
    "bro",
    "sis",
    "fam",
    "lit",
    "chill",
    "cool",
    "nice",
    "sick",
    "dope",
    "yay",
    "ooh",
    "ahh",
    "hmm",
    "meh",
    "ugh",
    "whoa",
    "wow",
    "aww",
    "oops",
    "whoops",
}

# Professional indicators
PROFESSIONAL_INDICATORS: set[str] = {
    "regarding",
    "pursuant",
    "attached",
    "please",
    "kindly",
    "sincerely",
    "regards",
    "cordially",
    "respectfully",
    "appreciate",
    "opportunity",
    "discussed",
    "confirmed",
    "scheduled",
    "deadline",
    "deliverable",
    "milestone",
    "stakeholder",
    "proposal",
    "presentation",
    "quarterly",
    "annual",
    "fiscal",
    "eod",
    "eow",
    "asap",
    "fyi",
    "cc",
    "per",
    "via",
    "ensure",
    "verify",
    "confirm",
    "acknowledge",
    "proceed",
    "follow-up",
    "followup",
    "meeting",
    "conference",
    "agenda",
    "minutes",
    "action item",
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
    "dear",
    "hello",
    "good morning",
    "good afternoon",
    "good evening",
}

# Emoji regex pattern
EMOJI_PATTERN = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map symbols
    "\U0001f1e0-\U0001f1ff"  # flags
    "\U00002702-\U000027b0"  # dingbats
    "\U000024c2-\U0001f251"  # enclosed characters
    "]+"
)


def detect_tone(messages: list[str]) -> Literal["casual", "professional", "mixed"]:
    """Detect the conversational tone from a list of messages.

    Analyzes messages for casual indicators (slang, emoji, informal greetings)
    and professional indicators (formal language, business terminology).

    Args:
        messages: List of message strings to analyze

    Returns:
        "casual" if predominantly informal language
        "professional" if predominantly formal language
        "mixed" if both styles are present or unclear
    """
    if not messages:
        return "casual"  # Default to casual for empty input

    # Combine all messages for analysis
    combined = " ".join(messages).lower()
    words = set(re.findall(r"\b\w+\b", combined))

    # Count indicators
    casual_count = len(words & CASUAL_INDICATORS)
    professional_count = len(words & PROFESSIONAL_INDICATORS)

    # Check for emoji (strong casual indicator)
    emoji_count = len(EMOJI_PATTERN.findall(combined))
    casual_count += emoji_count * 2  # Weight emoji heavily

    # Check for exclamation marks (casual indicator)
    exclamation_count = combined.count("!")
    if exclamation_count > 2:
        casual_count += 1

    # Check for multi-character expressions like "hahahaha" or "looool"
    if re.search(r"(.)\1{3,}", combined):  # 4+ repeated chars
        casual_count += 2

    # Determine tone based on counts
    total = casual_count + professional_count

    if total == 0:
        return "casual"  # Default to casual when no strong indicators

    casual_ratio = casual_count / total

    if casual_ratio >= 0.7:
        return "casual"
    elif casual_ratio <= 0.3:
        return "professional"
    else:
        return "mixed"


# =============================================================================
# Prompt Builder Functions
# =============================================================================


def _format_examples(examples: list[FewShotExample]) -> str:
    """Format few-shot examples for prompt inclusion.

    Args:
        examples: List of FewShotExample objects

    Returns:
        Formatted string with examples
    """
    formatted = []
    for ex in examples:
        formatted.append(f"Context: {ex.context}\nReply: {ex.output}")
    return "\n\n".join(formatted)


def _format_summary_examples(examples: list[tuple[str, str]]) -> str:
    """Format summarization examples for prompt inclusion.

    Args:
        examples: List of (conversation, summary) tuples

    Returns:
        Formatted string with examples
    """
    formatted = []
    for conversation, summary in examples:
        formatted.append(f"Conversation:\n{conversation}\n\n{summary}")
    return "\n\n---\n\n".join(formatted)


def _format_search_examples(examples: list[tuple[str, str, str]]) -> str:
    """Format search/QA examples for prompt inclusion.

    Args:
        examples: List of (messages, question, answer) tuples

    Returns:
        Formatted string with examples
    """
    formatted = []
    for messages, question, answer in examples:
        formatted.append(f"Messages:\n{messages}\nQuestion: {question}\nAnswer: {answer}")
    return "\n\n---\n\n".join(formatted)


def _truncate_context(context: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Truncate context to fit within token limits.

    Keeps the most recent messages when truncating.

    Args:
        context: The conversation context
        max_chars: Maximum characters to keep

    Returns:
        Truncated context string
    """
    if len(context) <= max_chars:
        return context

    # Keep the most recent messages (end of context)
    truncated = context[-max_chars:]

    # Try to start at a message boundary (newline)
    first_newline = truncated.find("\n")
    if first_newline != -1 and first_newline < 200:
        truncated = truncated[first_newline + 1 :]

    return f"[Earlier messages truncated]\n{truncated}"


def build_reply_prompt(
    context: str,
    last_message: str,
    instruction: str | None = None,
    tone: Literal["casual", "professional", "mixed"] = "casual",
) -> str:
    """Build a prompt for generating iMessage replies.

    Args:
        context: The conversation history
        last_message: The most recent message to reply to
        instruction: Optional custom instruction for the reply
        tone: The desired tone for the reply

    Returns:
        Formatted prompt string ready for model input
    """
    # Select appropriate examples based on tone
    if tone == "professional":
        examples = PROFESSIONAL_REPLY_EXAMPLES[:3]
        tone_str = "professional/formal"
    else:
        examples = CASUAL_REPLY_EXAMPLES[:3]
        tone_str = "casual/friendly"

    # Format custom instruction
    custom_instruction = ""
    if instruction:
        custom_instruction = f"- Additional guidance: {instruction}"

    # Truncate context if needed
    truncated_context = _truncate_context(context)

    # Build the prompt
    prompt = REPLY_TEMPLATE.template.format(
        context=truncated_context,
        tone=tone_str,
        custom_instruction=custom_instruction,
        examples=_format_examples(examples),
        last_message=last_message,
    )

    return prompt


def build_summary_prompt(
    context: str,
    focus: str | None = None,
) -> str:
    """Build a prompt for summarizing conversations.

    Args:
        context: The conversation to summarize
        focus: Optional focus area (e.g., "action items", "decisions", "dates")

    Returns:
        Formatted prompt string ready for model input
    """
    # Format focus instruction
    focus_instruction = ""
    if focus:
        focus_instruction = f"- Focus especially on: {focus}"

    # Select a subset of examples
    examples = SUMMARIZATION_EXAMPLES[:2]

    # Truncate context if needed
    truncated_context = _truncate_context(context)

    # Build the prompt
    prompt = SUMMARY_TEMPLATE.template.format(
        context=truncated_context,
        focus_instruction=focus_instruction,
        examples=_format_summary_examples(examples),
    )

    return prompt


def build_search_answer_prompt(
    context: str,
    question: str,
) -> str:
    """Build a prompt for answering questions about conversations.

    Args:
        context: The relevant messages/conversation context
        question: The question to answer

    Returns:
        Formatted prompt string ready for model input
    """
    # Select a subset of examples
    examples = SEARCH_ANSWER_EXAMPLES[:2]

    # Truncate context if needed
    truncated_context = _truncate_context(context)

    # Build the prompt
    prompt = SEARCH_ANSWER_TEMPLATE.template.format(
        context=truncated_context,
        question=question,
        examples=_format_search_examples(examples),
    )

    return prompt


def _get_thread_examples(topic_name: str) -> list[FewShotExample]:
    """Get few-shot examples for a thread topic.

    Args:
        topic_name: The thread topic name (e.g., "logistics", "emotional_support")

    Returns:
        List of relevant FewShotExample instances
    """
    # Map topic enum values to example keys
    topic_map = {
        "logistics": "logistics",
        "planning": "planning",
        "catching_up": "catching_up",
        "emotional_support": "emotional_support",
        "quick_exchange": "quick_exchange",
        "information": "logistics",  # Use logistics as fallback
        "decision_making": "planning",  # Similar to planning
        "celebration": "catching_up",  # Similar tone
        "unknown": "catching_up",  # Default to conversational
    }

    key = topic_map.get(topic_name, "catching_up")
    return THREAD_EXAMPLES.get(key, CATCHING_UP_THREAD_EXAMPLES)


def _format_thread_context(messages: list[object]) -> str:
    """Format thread messages for prompt context.

    Args:
        messages: List of Message objects

    Returns:
        Formatted string of messages
    """
    lines = []
    for msg in messages:
        # Handle Message objects (duck typing)
        if hasattr(msg, "date") and hasattr(msg, "text"):
            timestamp = msg.date.strftime("%H:%M") if hasattr(msg.date, "strftime") else ""
            sender = "Me" if getattr(msg, "is_from_me", False) else getattr(
                msg, "sender_name", None
            ) or getattr(msg, "sender", "Unknown")
            text = msg.text or ""
            lines.append(f"[{timestamp}] {sender}: {text}")
        # Handle raw strings
        elif isinstance(msg, str):
            lines.append(msg)

    return "\n".join(lines)


def _get_length_guidance(response_style: str, max_length: int) -> str:
    """Get length guidance based on response style.

    Args:
        response_style: The recommended response style
        max_length: Maximum response length

    Returns:
        Human-readable length guidance string
    """
    if response_style in ("concise", "brief"):
        return "brief and to the point (1-2 sentences)"
    elif response_style == "empathetic":
        return "warm and supportive (2-3 sentences, show you care)"
    elif response_style == "detailed":
        return "complete but not lengthy (2-3 sentences with relevant details)"
    elif response_style == "enthusiastic":
        return "upbeat and celebratory (1-2 sentences)"
    else:
        return "natural and conversational (1-2 sentences)"


def _get_additional_instructions(
    topic_name: str,
    state_name: str,
    config: ThreadedReplyConfig,
) -> str:
    """Get additional instructions based on thread context.

    Args:
        topic_name: Thread topic name
        state_name: Thread state name
        config: Thread response configuration

    Returns:
        Additional instruction string
    """
    instructions = []

    # State-specific instructions
    if state_name == "open_question":
        instructions.append("- Answer the question directly")
    elif state_name == "awaiting_response":
        instructions.append("- Acknowledge their message and respond appropriately")

    # Topic-specific instructions
    if topic_name == "emotional_support":
        instructions.append("- Show empathy and understanding")
        instructions.append("- Offer support without being preachy")
    elif topic_name == "logistics":
        instructions.append("- Be clear and specific about times/places")
        instructions.append("- Confirm key details")
    elif topic_name == "planning":
        instructions.append("- Be constructive and suggest next steps")
        if config.include_action_items:
            instructions.append("- Include any commitments you're making")
    elif topic_name == "decision_making":
        instructions.append("- Provide your input clearly")
        instructions.append("- Help move the decision forward")

    # Config-specific instructions
    if config.suggest_follow_up:
        instructions.append("- End with a question or invitation to continue")

    return "\n".join(instructions) if instructions else ""


def build_threaded_reply_prompt(
    thread_context: ThreadContext,
    config: ThreadedReplyConfig,
    instruction: str | None = None,
    tone: Literal["casual", "professional", "mixed"] = "casual",
) -> str:
    """Build a prompt for thread-aware reply generation.

    Constructs a prompt that includes thread context, topic-specific examples,
    and appropriate instructions for the thread type and state.

    Args:
        thread_context: Analyzed thread context from ThreadAnalyzer
        config: Response configuration based on thread type
        instruction: Optional custom instruction for the reply
        tone: Overall tone preference (default: casual)

    Returns:
        Formatted prompt string ready for model input
    """
    # Get topic and state names
    topic_name = thread_context.topic.value
    state_name = thread_context.state.value
    user_role = thread_context.user_role.value

    # Get appropriate examples for this thread type
    examples = _get_thread_examples(topic_name)[:2]

    # Format the relevant messages (not all messages)
    relevant_msgs = thread_context.relevant_messages or thread_context.messages[-5:]
    context = _format_thread_context(relevant_msgs)

    # Get the last message to reply to
    last_message = ""
    if thread_context.messages:
        last_msg = thread_context.messages[-1]
        if hasattr(last_msg, "text"):
            last_message = last_msg.text or ""
        elif isinstance(last_msg, str):
            last_message = last_msg

    # Build participants info for group chats
    participants_info = ""
    if thread_context.participants_count > 1:
        participants_info = f"Group chat with {thread_context.participants_count} participants"

    # Get length guidance
    length_guidance = _get_length_guidance(config.response_style, config.max_response_length)

    # Get additional instructions
    additional_instructions = _get_additional_instructions(topic_name, state_name, config)

    # Format custom instruction
    custom_instruction = ""
    if instruction:
        custom_instruction = f"- Additional guidance: {instruction}"

    # Truncate context if needed
    truncated_context = _truncate_context(context, max_chars=2000)

    # Build the prompt
    prompt = THREADED_REPLY_TEMPLATE.template.format(
        thread_topic=topic_name.replace("_", " ").title(),
        thread_state=state_name.replace("_", " ").title(),
        user_role=user_role.replace("_", " ").title(),
        participants_info=participants_info,
        context=truncated_context,
        response_style=config.response_style,
        length_guidance=length_guidance,
        additional_instructions=additional_instructions,
        custom_instruction=custom_instruction,
        examples=_format_examples(examples),
        last_message=last_message,
    )

    return prompt


def get_thread_max_tokens(config: ThreadedReplyConfig) -> int:
    """Get max tokens for generation based on thread config.

    Args:
        config: Thread response configuration

    Returns:
        Recommended max tokens for generation
    """
    # Rough estimate: ~4 chars per token
    base_tokens = config.max_response_length // 4

    # Add buffer for safety
    return max(30, min(base_tokens + 20, 150))


# =============================================================================
# Utility Functions
# =============================================================================


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text.

    Uses a simple heuristic of ~4 characters per token.
    This is an approximation; actual token counts vary by model.

    Args:
        text: The text to estimate tokens for

    Returns:
        Estimated token count
    """
    return len(text) // 4


def is_within_token_limit(prompt: str, limit: int = MAX_PROMPT_TOKENS) -> bool:
    """Check if a prompt is within the token limit.

    Args:
        prompt: The prompt to check
        limit: Maximum allowed tokens (default: MAX_PROMPT_TOKENS)

    Returns:
        True if prompt is within limit, False otherwise
    """
    return estimate_tokens(prompt) <= limit


# =============================================================================
# Compatibility Exports
# =============================================================================

# Convert FewShotExample lists to tuple format for GenerationRequest compatibility
REPLY_EXAMPLES: list[tuple[str, str]] = [(ex.context, ex.output) for ex in CASUAL_REPLY_EXAMPLES]

SUMMARY_EXAMPLES: list[tuple[str, str]] = SUMMARIZATION_EXAMPLES


# =============================================================================
# API-Style Prompt Examples
# =============================================================================

# These examples use the instruction-based format for API endpoints
# (e.g., the drafts router). They include explicit instructions.

API_REPLY_EXAMPLES_METADATA = PromptMetadata(
    name="api_reply_examples",
    version=PROMPT_VERSION,
    last_updated=PROMPT_LAST_UPDATED,
    description="Few-shot examples for API reply generation with explicit instructions",
)

API_REPLY_EXAMPLES: list[tuple[str, str]] = [
    (
        "Last message: 'Hey, are you free for dinner tomorrow?'\n"
        "Instruction: accept enthusiastically",
        "Yes, absolutely! I'd love to! What time works for you?",
    ),
    (
        "Last message: 'Can you review this document by EOD?'\n"
        "Instruction: confirm and ask for details",
        "Sure, I can take a look. Which sections should I focus on?",
    ),
    (
        "Last message: 'Thanks for your help yesterday!'\nInstruction: None",
        "You're welcome! Happy I could help.",
    ),
]

API_SUMMARY_EXAMPLES_METADATA = PromptMetadata(
    name="api_summary_examples",
    version=PROMPT_VERSION,
    last_updated=PROMPT_LAST_UPDATED,
    description="Few-shot examples for API conversation summarization",
)

API_SUMMARY_EXAMPLES: list[tuple[str, str]] = [
    (
        "Conversation about planning a birthday party with 5 messages "
        "discussing date, venue, and guest list.",
        "Summary: Planning discussion for a birthday party.\n"
        "Key points:\n- Deciding on date and venue\n- Creating guest list",
    ),
]


# =============================================================================
# Prompt Registry
# =============================================================================


class PromptRegistry:
    """Registry for dynamic prompt management.

    Provides centralized access to all prompts, examples, and templates
    with metadata tracking and versioning support.

    Example:
        >>> registry = PromptRegistry()
        >>> examples = registry.get_examples("casual_reply")
        >>> template = registry.get_template("reply_generation")
        >>> metadata = registry.get_metadata("casual_reply")
    """

    def __init__(self) -> None:
        """Initialize the prompt registry with all registered prompts."""
        self._examples: dict[str, list[tuple[str, str]]] = {
            "casual_reply": REPLY_EXAMPLES,
            "professional_reply": [(ex.context, ex.output) for ex in PROFESSIONAL_REPLY_EXAMPLES],
            "summarization": SUMMARIZATION_EXAMPLES,
            "search_answer": [
                (f"Messages:\n{msgs}\nQuestion: {q}", a) for msgs, q, a in SEARCH_ANSWER_EXAMPLES
            ],
            "api_reply": API_REPLY_EXAMPLES,
            "api_summary": API_SUMMARY_EXAMPLES,
            "thread_logistics": [(ex.context, ex.output) for ex in LOGISTICS_THREAD_EXAMPLES],
            "thread_emotional_support": [
                (ex.context, ex.output) for ex in EMOTIONAL_SUPPORT_THREAD_EXAMPLES
            ],
            "thread_planning": [(ex.context, ex.output) for ex in PLANNING_THREAD_EXAMPLES],
            "thread_catching_up": [(ex.context, ex.output) for ex in CATCHING_UP_THREAD_EXAMPLES],
            "thread_quick_exchange": [
                (ex.context, ex.output) for ex in QUICK_EXCHANGE_THREAD_EXAMPLES
            ],
        }

        self._templates: dict[str, PromptTemplate] = {
            "reply_generation": REPLY_TEMPLATE,
            "conversation_summary": SUMMARY_TEMPLATE,
            "search_answer": SEARCH_ANSWER_TEMPLATE,
            "threaded_reply": THREADED_REPLY_TEMPLATE,
        }

        self._metadata: dict[str, PromptMetadata] = {
            "casual_reply": PromptMetadata(
                name="casual_reply",
                description="Few-shot examples for casual iMessage replies",
            ),
            "professional_reply": PromptMetadata(
                name="professional_reply",
                description="Few-shot examples for professional iMessage replies",
            ),
            "summarization": PromptMetadata(
                name="summarization",
                description="Few-shot examples for conversation summarization",
            ),
            "search_answer": PromptMetadata(
                name="search_answer",
                description="Few-shot examples for question answering over messages",
            ),
            "api_reply": API_REPLY_EXAMPLES_METADATA,
            "api_summary": API_SUMMARY_EXAMPLES_METADATA,
            "reply_generation": PromptMetadata(
                name="reply_generation",
                description="Template for generating iMessage replies",
            ),
            "conversation_summary": PromptMetadata(
                name="conversation_summary",
                description="Template for summarizing conversations",
            ),
            "search_answer_template": PromptMetadata(
                name="search_answer_template",
                description="Template for answering questions about conversations",
            ),
            "threaded_reply": PromptMetadata(
                name="threaded_reply",
                description="Template for thread-aware reply generation",
            ),
            "thread_logistics": PromptMetadata(
                name="thread_logistics",
                description="Examples for logistics/coordination thread replies",
            ),
            "thread_emotional_support": PromptMetadata(
                name="thread_emotional_support",
                description="Examples for emotional support thread replies",
            ),
            "thread_planning": PromptMetadata(
                name="thread_planning",
                description="Examples for planning thread replies with action items",
            ),
            "thread_catching_up": PromptMetadata(
                name="thread_catching_up",
                description="Examples for catching up/casual thread replies",
            ),
            "thread_quick_exchange": PromptMetadata(
                name="thread_quick_exchange",
                description="Examples for quick exchange thread replies",
            ),
        }

    def get_examples(self, name: str) -> list[tuple[str, str]]:
        """Get few-shot examples by name.

        Args:
            name: The example set name (e.g., "casual_reply", "api_reply")

        Returns:
            List of (input, output) tuples for few-shot prompting

        Raises:
            KeyError: If the example set doesn't exist
        """
        if name not in self._examples:
            available = ", ".join(sorted(self._examples.keys()))
            raise KeyError(f"Unknown example set '{name}'. Available: {available}")
        return self._examples[name]

    def get_template(self, name: str) -> PromptTemplate:
        """Get a prompt template by name.

        Args:
            name: The template name (e.g., "reply_generation")

        Returns:
            The PromptTemplate instance

        Raises:
            KeyError: If the template doesn't exist
        """
        if name not in self._templates:
            available = ", ".join(sorted(self._templates.keys()))
            raise KeyError(f"Unknown template '{name}'. Available: {available}")
        return self._templates[name]

    def get_metadata(self, name: str) -> PromptMetadata:
        """Get metadata for a prompt or example set.

        Args:
            name: The prompt/example set name

        Returns:
            The PromptMetadata instance

        Raises:
            KeyError: If the metadata doesn't exist
        """
        if name not in self._metadata:
            available = ", ".join(sorted(self._metadata.keys()))
            raise KeyError(f"Unknown prompt '{name}'. Available: {available}")
        return self._metadata[name]

    def list_examples(self) -> list[str]:
        """List all available example set names.

        Returns:
            Sorted list of example set names
        """
        return sorted(self._examples.keys())

    def list_templates(self) -> list[str]:
        """List all available template names.

        Returns:
            Sorted list of template names
        """
        return sorted(self._templates.keys())

    def register_examples(
        self,
        name: str,
        examples: list[tuple[str, str]],
        metadata: PromptMetadata | None = None,
    ) -> None:
        """Register a new example set.

        Args:
            name: Unique name for the example set
            examples: List of (input, output) tuples
            metadata: Optional metadata for the example set
        """
        self._examples[name] = examples
        if metadata:
            self._metadata[name] = metadata
        else:
            self._metadata[name] = PromptMetadata(
                name=name,
                description=f"Custom example set: {name}",
            )

    def register_template(
        self,
        template: PromptTemplate,
        metadata: PromptMetadata | None = None,
    ) -> None:
        """Register a new prompt template.

        Args:
            template: The PromptTemplate to register
            metadata: Optional metadata for the template
        """
        self._templates[template.name] = template
        if metadata:
            self._metadata[template.name] = metadata
        else:
            self._metadata[template.name] = PromptMetadata(
                name=template.name,
                description=f"Custom template: {template.name}",
            )

    @property
    def version(self) -> str:
        """Get the prompt system version."""
        return PROMPT_VERSION

    @property
    def last_updated(self) -> str:
        """Get the last update date."""
        return PROMPT_LAST_UPDATED


# Global registry instance
_registry: PromptRegistry | None = None


def get_prompt_registry() -> PromptRegistry:
    """Get the global PromptRegistry instance.

    Returns:
        The shared PromptRegistry instance
    """
    global _registry
    if _registry is None:
        _registry = PromptRegistry()
    return _registry


def reset_prompt_registry() -> None:
    """Reset the global PromptRegistry instance.

    Useful for testing or when prompts need to be reloaded.
    """
    global _registry
    _registry = None
