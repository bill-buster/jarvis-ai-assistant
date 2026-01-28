"""Prompt templates for reply generation in JARVIS v2.

Contains the main prompt template and few-shot examples.
"""

from __future__ import annotations

# Main prompt template for reply generation
REPLY_PROMPT = '''Reply to this iMessage conversation. Write 3 different short replies.

{conversation_history}
Them: {last_message}

{examples_section}

Your 3 replies (casual, under {max_length} words each):
1.'''


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
            "message": "Check out this restaurant I found",
            "replies": [
                "ooh looks good! where is it?",
                "we should try it!",
                "adding to my list 📝",
            ],
        },
        {
            "message": "Look at this meme lol",
            "replies": [
                "hahaha 😂",
                "lmaooo so accurate",
                "dead 💀",
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


def build_reply_prompt(
    messages: list[dict],
    last_message: str,
    last_sender: str,
    style_instructions: str,
    reply_types: list[str],
    tone: str,
    max_length: int,
    intent_value: str,
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

    Returns:
        Complete prompt string
    """
    # Format conversation history
    history_lines = []
    for msg in messages[-8:]:  # Last 8 messages
        sender = "You" if msg.get("is_from_me") else "Them"
        text = msg.get("text", "")[:80]  # Truncate long messages
        if text:
            history_lines.append(f"{sender}: {text}")

    conversation_history = "\n".join(history_lines) if history_lines else "(New conversation)"

    # Get examples
    examples_section = get_examples_for_intent(intent_value)

    return REPLY_PROMPT.format(
        conversation_history=conversation_history,
        last_message=last_message,
        tone=tone,
        max_length=max_length,
        style_instructions=style_instructions,
        examples_section=examples_section,
    )
