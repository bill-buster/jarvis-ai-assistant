"""Reply generation examples for prompts module.

This module contains few-shot examples for reply generation,
organized by tone (casual, professional) and thread context.
"""

from jarvis.prompts.models import FewShotExample

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
            "[15:00] John: Just found out I didn't get the job\n[15:02] John: Feeling pretty down"
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
            "[10:00] Sarah: We should plan a birthday surprise for Mike\n[10:05] Sarah: Any ideas?"
        ),
        output=(
            "Definitely! We could do a small surprise at his favorite bar. "
            "I can coordinate with Lisa on decorations. Should we aim for next Friday?"
        ),
        tone="casual",
    ),
    FewShotExample(
        context=(
            "[16:00] John: Thinking about a camping trip next month\n[16:05] John: You interested?"
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
