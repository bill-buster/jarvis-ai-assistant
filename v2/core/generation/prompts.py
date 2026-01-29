"""Prompt templates for reply generation in JARVIS v2.

Contains the main prompt template and few-shot examples.
"""

from __future__ import annotations

# Main prompt template for reply generation
# Minimal instructions - let the model continue naturally
# Note: No trailing "user:" - ChatML handles turn structure via <|im_start|>assistant
REPLY_PROMPT = '''Text message conversation. Reply briefly as {user_name}.

{conversation}'''


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


def _get_display_name(msg: dict, fallback: str = "Them") -> str | None:
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
    """Format past replies as few-shot examples for the prompt.

    This is the KEY learning signal - showing the model how you ACTUALLY reply.

    Args:
        past_replies: List of (their_message, your_reply, similarity) tuples

    Returns:
        Formatted string showing how user replied before
    """
    if not past_replies:
        return ""

    # Format as clear input->output examples
    lines = ["How you replied to similar messages:"]
    for their_msg, your_reply, _sim in past_replies[:4]:  # Top 4 examples
        # Truncate long messages but keep enough context
        their_msg = their_msg[:60] + "..." if len(their_msg) > 60 else their_msg
        your_reply = your_reply[:40] if len(your_reply) <= 40 else your_reply[:40] + "..."
        lines.append(f"Them: {their_msg}")
        lines.append(f"You: {your_reply}")
        lines.append("")  # Blank line between examples

    return "\n".join(lines).strip()


def format_style_samples(messages: list[dict], limit: int = 5) -> str:
    """Extract message samples showing the user's actual reply style.

    Only includes messages that are RESPONSES (following a message from them),
    not isolated messages or reactions. These are more useful as examples.

    Args:
        messages: Recent conversation messages
        limit: Max samples to include

    Returns:
        Formatted string with style samples
    """
    # Get user's messages that are actual RESPONSES (following their message)
    response_samples = []
    prev_was_theirs = False

    for msg in messages:
        is_from_me = msg.get("is_from_me", False)
        text = (msg.get("text") or "").replace("\ufffc", "").strip()

        if not is_from_me:
            prev_was_theirs = True
        elif is_from_me and prev_was_theirs and text:
            # This is a response to their message
            # Filter: 3-40 chars, not a reaction, not just emoji
            if 3 <= len(text) <= 40:
                # Skip reactions like "Loved an image" or "Laughed at..."
                text_lower = text.lower()
                if not any(r in text_lower for r in ["loved", "liked", "emphasized", "laughed at"]):
                    # Skip pure emoji messages
                    if any(c.isalpha() for c in text):
                        response_samples.append(text)
            prev_was_theirs = False

    # Get most recent samples
    recent = response_samples[-limit:] if len(response_samples) > limit else response_samples

    if not recent:
        return ""

    return "How you typically reply: " + " | ".join(recent)


# Simple conversation continuation prompt
# Shows actual conversation and lets model continue naturally
# This works better than complex few-shot examples
CONVERSATION_PROMPT = '''[{style_hint}]

{conversation}
me:'''

# Generic few-shot examples (fallback when no conversation history)
CASUAL_FEW_SHOT = """casual texts between friends:

them: wanna hang?
me: ya sure

them: you coming tonight?
me: prob not

them: what time?
me: like 7ish

them: how was it?
me: pretty good tbh

"""

# Legacy prompt template (kept for backwards compatibility)
REPLY_PROMPT_WITH_HISTORY = '''{few_shot}{past_replies_section}{availability_hint}them: {last_message}
me:'''


def build_conversation_prompt(
    messages: list[dict],
    style_hint: str = "brief, casual",
    max_messages: int = 10,
) -> str:
    """Build a simple conversation-continuation prompt.

    This is the NEW approach - just show the conversation and let the model continue.
    Works better than complex few-shot examples for most models.

    Args:
        messages: Recent conversation messages
        style_hint: Brief style instruction
        max_messages: Max messages to include

    Returns:
        Prompt string ending with "me:" for completion
    """
    if not messages:
        return f"[{style_hint}]\n\nthem: hey\nme:"

    # Take last N messages
    recent = messages[-max_messages:] if len(messages) > max_messages else messages

    # Format as simple conversation
    lines = []
    for msg in recent:
        text = (msg.get("text") or "").replace("\ufffc", "").strip()
        if not text or len(text) < 1:
            continue
        if msg.get("is_from_me"):
            lines.append(f"me: {text}")
        else:
            lines.append(f"them: {text}")

    conversation = "\n".join(lines)

    return CONVERSATION_PROMPT.format(
        style_hint=style_hint,
        conversation=conversation,
    )


def build_balanced_few_shot() -> str:
    """Build a balanced few-shot example set covering multiple intents.

    Selects 1 example from each major intent category to ensure the model
    sees a diverse range of response types (questions, statements, reactions).
    """
    selected = []
    
    # Priority intents to include
    intents = [
        "yes_no_question", "open_question", "statement", 
        "emotional", "greeting", "logistics"
    ]
    
    lines = ["casual texts between friends:\n"]
    
    for intent in intents:
        examples = FEW_SHOT_EXAMPLES.get(intent, [])
        if examples:
            # deterministic selection (always take first one) for consistency
            ex = examples[0]
            lines.append(f"them: {ex['message']}")
            lines.append(f"me: {ex['replies'][0]}") # Take first reply variation
            lines.append("")
            
    return "\n".join(lines)


def build_reply_prompt(
    messages: list[dict],
    last_message: str,
    last_sender: str,
    style_instructions: str,
    past_replies: list[tuple[str, str, float]] | None = None,
    user_name: str = "me",
    recent_topics: list[str] | None = None,
    availability: str | None = None,
    your_phrases: list[str] | None = None,
    global_style=None,
    contact_profile=None,
) -> str:
    """Build the complete reply generation prompt.

    Uses few-shot examples to teach Llama 3.2 the casual texting style.
    The model learns from examples better than from instructions.

    Args:
        messages: Recent conversation messages
        last_message: The message to reply to
        last_sender: Who sent the last message
        style_instructions: User style instructions (used to build custom examples)
        past_replies: User's past replies to similar messages
        user_name: User's name for personalization
        recent_topics: Recent conversation topics for context
        availability: User's current availability ("busy", "free", or None)
        your_phrases: Common phrases the user uses (for personalization)
        global_style: Optional GlobalUserStyle with personality info
        contact_profile: Optional ContactProfile with relationship info

    Returns:
        Complete prompt string for Llama 3.2
    """
    # Start with balanced few-shot examples (better than generic static ones)
    few_shot = build_balanced_few_shot()

    # If we have user's past replies, use those instead (more personalized)
    if past_replies and len(past_replies) >= 2:
        lines = ["casual texts:\n"]
        for their_msg, your_reply, _ in past_replies[:4]:
            their_msg = their_msg[:50]
            your_reply = your_reply[:30]
            lines.append(f"them: {their_msg}")
            lines.append(f"me: {your_reply}")
            lines.append("")
        few_shot = "\n".join(lines)

    # Format availability hint
    availability_hint = ""
    if availability == "busy":
        availability_hint = "(busy) "
    elif availability == "free":
        availability_hint = "(free) "

    # Build past replies section if we have some but not enough for full few-shot
    past_replies_section = ""
    if past_replies and len(past_replies) < 2:
        past_replies_section = format_past_replies(past_replies) + "\n\n"

    # Get the last message to reply to
    # Clean it up
    clean_message = (last_message or "").replace("\ufffc", "").strip()
    if not clean_message:
        # Fall back to last message in conversation
        for msg in reversed(messages):
            if not msg.get("is_from_me"):
                clean_message = (msg.get("text") or "").replace("\ufffc", "").strip()
                if clean_message:
                    break

    return REPLY_PROMPT_WITH_HISTORY.format(
        few_shot=few_shot,
        past_replies_section=past_replies_section,
        availability_hint=availability_hint,
        last_message=clean_message,
    )
