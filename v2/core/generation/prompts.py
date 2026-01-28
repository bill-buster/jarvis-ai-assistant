"""Prompt templates for reply generation in JARVIS v2.

Contains the main prompt template and few-shot examples.
"""

from __future__ import annotations

# Main prompt template for reply generation
# Minimal instructions - just continue naturally
REPLY_PROMPT = '''Text message conversation. Reply briefly.

{conversation}
{user_name}:'''


# Few-shot examples organized by intent
FEW_SHOT_EXAMPLES = {
    "yes_no_question": [
        {
            "message": "Want to grab dinner tonight?",
            "replies": [
                "yes! where were you thinking?",
                "can't tonight, rain check?",
                "what time works for you?",
            ],
        },
        {
            "message": "Are you coming to the party?",
            "replies": [
                "definitely! what time?",
                "probably not, super tired lately",
                "maybe, who else is going?",
            ],
        },
        {
            "message": "Do you have the notes from class?",
            "replies": [
                "yeah i'll send them over",
                "i missed that one too lol",
                "let me check, one sec",
            ],
        },
        {
            "message": "Can you pick me up?",
            "replies": [
                "yeah, what time?",
                "can't today, sorry!",
                "where from?",
            ],
        },
    ],
    "open_question": [
        {
            "message": "How've you been?",
            "replies": [
                "good! been super busy with work",
                "doing well, how about you?",
                "can't complain! what's new with you?",
            ],
        },
        {
            "message": "What time works for you?",
            "replies": [
                "anytime after 5",
                "how about 7?",
                "pretty flexible, you pick",
            ],
        },
        {
            "message": "What did you think of the movie?",
            "replies": [
                "loved it! the ending was wild",
                "eh, it was okay",
                "so good, we should see the sequel",
            ],
        },
    ],
    "choice_question": [
        {
            "message": "Italian or Mexican?",
            "replies": [
                "italian for sure",
                "mexican! been craving tacos",
                "either works for me",
            ],
        },
        {
            "message": "Friday or Saturday?",
            "replies": [
                "friday works better",
                "saturday for me",
                "either day is fine",
            ],
        },
    ],
    "statement": [
        {
            "message": "The meeting got moved to 3pm",
            "replies": [
                "got it, thanks for the heads up",
                "works for me 👍",
                "okay cool, same room?",
            ],
        },
        {
            "message": "I'll be there in 10 minutes",
            "replies": [
                "sounds good, see you soon",
                "perfect, i'll grab us a table",
                "no rush!",
            ],
        },
        {
            "message": "Just finished the project",
            "replies": [
                "nice! how'd it turn out?",
                "finally! that took forever",
                "congrats! 🎉",
            ],
        },
    ],
    "emotional": [
        {
            "message": "I'm so stressed about this deadline",
            "replies": [
                "ugh that sucks, you got this tho",
                "anything i can help with?",
                "when's it due? want to talk it out?",
            ],
        },
        {
            "message": "I got the job!!!",
            "replies": [
                "YESSS congrats!!! 🎉",
                "omg that's amazing!! so happy for you",
                "knew you would! we gotta celebrate",
            ],
        },
        {
            "message": "Today was rough",
            "replies": [
                "sorry to hear that, what happened?",
                "that sucks, wanna talk about it?",
                "hope tomorrow is better 💙",
            ],
        },
    ],
    "greeting": [
        {
            "message": "Hey! How are you?",
            "replies": [
                "hey! good, how about you?",
                "doing well! what's up?",
                "hey! been a minute, how've you been?",
            ],
        },
        {
            "message": "What's up?",
            "replies": [
                "not much, you?",
                "just chilling, what's good?",
                "hey! just got off work",
            ],
        },
    ],
    "logistics": [
        {
            "message": "I'm running 10 minutes late",
            "replies": [
                "no worries, take your time",
                "all good, see you soon",
                "thanks for letting me know!",
            ],
        },
        {
            "message": "Just parked",
            "replies": [
                "cool, i'm inside by the window",
                "see you in a sec!",
                "perfect timing, just got here too",
            ],
        },
    ],
    "sharing": [
        {
            "message": "I got you something from the trip",
            "replies": [
                "omg you didn't have to! what is it??",
                "aww thank you!! can't wait to see",
                "you're the best! when can i see it?",
            ],
        },
        {
            "message": "I finally brought the painting for you",
            "replies": [
                "yay finally!! can't wait to see it",
                "omg thank you! where should we hang it?",
                "you're amazing, thank you!",
            ],
        },
        {
            "message": "Check out this restaurant I found",
            "replies": [
                "ooh looks good! where is it?",
                "we should try it!",
                "adding to my list",
            ],
        },
        {
            "message": "Look at this meme lol",
            "replies": [
                "hahaha",
                "lmaooo so accurate",
                "dead",
            ],
        },
    ],
    "thanks": [
        {
            "message": "Thanks so much!",
            "replies": [
                "of course!",
                "anytime!",
                "no problem 👍",
            ],
        },
        {
            "message": "Thanks for your help",
            "replies": [
                "happy to help!",
                "no worries!",
                "anytime, let me know if you need anything else",
            ],
        },
    ],
    "farewell": [
        {
            "message": "Talk to you later!",
            "replies": [
                "later!",
                "sounds good, bye!",
                "talk soon!",
            ],
        },
        {
            "message": "Gotta run, bye!",
            "replies": [
                "bye! 👋",
                "later!",
                "see ya!",
            ],
        },
    ],
}


def get_examples_for_intent(intent_value: str) -> str:
    """Get formatted examples for an intent type.

    Args:
        intent_value: The intent value (e.g., "yes_no_question")

    Returns:
        Formatted examples string for the prompt
    """
    # Map intent to example key
    key_mapping = {
        "yes_no_question": "yes_no_question",
        "open_question": "open_question",
        "choice_question": "choice_question",
        "statement": "statement",
        "emotional": "emotional",
        "greeting": "greeting",
        "logistics": "logistics",
        "sharing": "sharing",
        "thanks": "thanks",
        "farewell": "farewell",
    }

    key = key_mapping.get(intent_value, "statement")
    examples = FEW_SHOT_EXAMPLES.get(key, FEW_SHOT_EXAMPLES["statement"])

    # Format examples - show 1 example with replies in numbered format
    ex = examples[0]
    lines = [
        f"Example for \"{ex['message']}\":",
        f"1. {ex['replies'][0]}",
        f"2. {ex['replies'][1]}",
        f"3. {ex['replies'][2]}",
    ]

    return "\n".join(lines)


def _get_display_name(msg: dict, fallback: str = "Them") -> str:
    """Get display name for a message sender, preferring name over phone number."""
    if msg.get("is_from_me"):
        return None  # Will use user_name instead
    # Prefer sender_name (contact name) over sender (phone number)
    name = msg.get("sender_name") or msg.get("sender") or fallback
    # If it looks like a phone number, use fallback
    if name and name.startswith("+"):
        return fallback
    return name


def _build_messages_array(messages: list[dict], max_messages: int = 6) -> tuple[list[str], str]:
    """Build simple messages array for JSON prompt.

    Uses ">" prefix for their messages, no prefix for yours.

    Args:
        messages: Recent messages
        max_messages: Max messages to include

    Returns:
        Tuple of (messages list, reply_to string)
    """
    if not messages:
        return [], ""

    # Take last N messages for context
    recent = messages[-max_messages:] if len(messages) > max_messages else messages

    # Prefix their messages with ">", yours with no prefix
    texts = []
    for msg in recent:
        text = msg.get("text", "")
        if not text:
            continue
        # Skip attachment placeholders and very short messages
        text = text.replace("\ufffc", "").strip()
        if len(text) < 2:
            continue
        if msg.get("is_from_me"):
            texts.append(text)
        else:
            texts.append(f">{text}")

    # What we're replying to
    last_msg = messages[-1]
    reply_to = last_msg.get("text", "") if not last_msg.get("is_from_me") else ""

    return texts, reply_to


def format_past_replies(past_replies: list[tuple[str, str, float]] | None) -> str:
    """Format past replies as examples for the prompt.

    Args:
        past_replies: List of (their_message, your_reply, similarity) tuples

    Returns:
        Formatted string showing how user replied before
    """
    if not past_replies:
        return ""

    lines = ["Your past replies to similar messages:"]
    for their_msg, your_reply, _sim in past_replies[:3]:  # Top 3 examples
        # Truncate long messages
        their_msg = their_msg[:50] + "..." if len(their_msg) > 50 else their_msg
        your_reply = your_reply[:50] + "..." if len(your_reply) > 50 else your_reply
        lines.append(f'- They said: "{their_msg}" → You: "{your_reply}"')

    return "\n".join(lines)


# Prompt template with past replies section
REPLY_PROMPT_WITH_HISTORY = '''Text message conversation. Reply briefly.
{past_replies_section}
{conversation}
{user_name}:'''


def build_reply_prompt(
    messages: list[dict],
    last_message: str,
    last_sender: str,
    style_instructions: str,
    reply_types: list[str],
    tone: str,
    max_length: int,
    intent_value: str,
    past_replies: list[tuple[str, str, float]] | None = None,
    user_name: str = "me",
    recent_topics: list[str] | None = None,
) -> str:
    """Build the complete reply generation prompt.

    Args:
        messages: Recent conversation messages
        last_message: The message to reply to
        last_sender: Who sent the last message
        style_instructions: User style instructions
        reply_types: Types of replies to generate
        tone: Tone for replies
        max_length: Max words per reply
        intent_value: Detected intent
        past_replies: User's past replies to similar messages
        user_name: User's name for personalization
        recent_topics: Recent conversation topics for context

    Returns:
        Complete prompt string (JSON format for Qwen3)
    """
    # Build simple conversation format
    lines = []

    # Add topic hint if available (helps maintain conversation context)
    if recent_topics:
        topic_hint = f"[Recent topics: {', '.join(recent_topics[:3])}]"
        lines.append(topic_hint)

    for msg in messages[-6:]:  # Last 6 messages
        text = msg.get("text", "").replace("\ufffc", "").strip()
        if not text or len(text) < 2:
            continue
        if msg.get("is_from_me"):
            lines.append(f"{user_name}: {text}")
        else:
            lines.append(f"Them: {text}")

    conversation = "\n".join(lines)

    # Format past replies section
    past_replies_section = format_past_replies(past_replies)
    if past_replies_section:
        past_replies_section = "\n" + past_replies_section + "\n"

    return REPLY_PROMPT_WITH_HISTORY.format(
        user_name=user_name,
        conversation=conversation,
        past_replies_section=past_replies_section,
    )
