"""Email reply templates for coverage analysis.

Workstream 3: Template Coverage Analyzer

Templates capture INTENT, not exact wording. Each template should match
many variations of the same response type.
"""

# Category 1: Acknowledgments (6 templates)
ACKNOWLEDGMENT_TEMPLATES = [
    "Got it, I'll take a look",
    "Thanks for sending this over",
    "Received, thanks for the update",
    "Thanks for letting me know",
    "Acknowledged, I'm on it",
    "Thanks for the heads up",
]

# Category 2: Scheduling/Timing (6 templates)
SCHEDULING_TEMPLATES = [
    "I'll get back to you by tomorrow",
    "Let's schedule a call to discuss",
    "I'm available this afternoon, does that work?",
    "Can we push this to next week?",
    "I need more time to review this",
    "Let me check my calendar and get back to you",
]

# Category 3: Requests for Information (5 templates)
INFORMATION_REQUEST_TEMPLATES = [
    "Could you send me more details?",
    "What's the deadline for this?",
    "Who else should be involved?",
    "Can you clarify what you mean?",
    "What's the priority on this?",
]

# Category 4: Affirmative Responses (5 templates)
AFFIRMATIVE_TEMPLATES = [
    "Yes, I can help with that",
    "Sounds good, let's do it",
    "I'm happy to assist",
    "That works for me",
    "Approved, go ahead",
]

# Category 5: Declines/Deferrals (5 templates)
DECLINE_TEMPLATES = [
    "I can't make it, sorry",
    "I'm not the right person for this",
    "I'll have to pass on this one",
    "Unfortunately I'm not available",
    "This doesn't align with my priorities right now",
]

# Category 6: Follow-ups (5 templates)
FOLLOWUP_TEMPLATES = [
    "Just following up on my last message",
    "Any updates on this?",
    "Wanted to check in on the status",
    "Did you get a chance to look at this?",
    "Circling back on this request",
]

# Category 7: Gratitude (5 templates)
GRATITUDE_TEMPLATES = [
    "Thank you so much for your help",
    "Really appreciate you handling this",
    "Thanks for the quick turnaround",
    "Grateful for your support",
    "This is exactly what I needed, thanks",
]

# Category 8: Confirmations (5 templates)
CONFIRMATION_TEMPLATES = [
    "Confirmed, see you then",
    "I'll be there",
    "Meeting accepted",
    "Booking confirmed",
    "I've completed the task",
]

# Category 9: Questions/Clarifications (5 templates)
QUESTION_TEMPLATES = [
    "Quick question about the project",
    "I have a few questions before proceeding",
    "Can you walk me through this?",
    "I'm not sure I understand, can you explain?",
    "What are the next steps?",
]

# Category 10: Closings/Sign-offs (5 templates)
CLOSING_TEMPLATES = [
    "Let me know if you need anything else",
    "Talk soon",
    "Looking forward to hearing from you",
    "Don't hesitate to reach out",
    "Have a great day",
]

# Master list of all templates (exported)
TEMPLATES: list[str] = [
    *ACKNOWLEDGMENT_TEMPLATES,
    *SCHEDULING_TEMPLATES,
    *INFORMATION_REQUEST_TEMPLATES,
    *AFFIRMATIVE_TEMPLATES,
    *DECLINE_TEMPLATES,
    *FOLLOWUP_TEMPLATES,
    *GRATITUDE_TEMPLATES,
    *CONFIRMATION_TEMPLATES,
    *QUESTION_TEMPLATES,
    *CLOSING_TEMPLATES,
]

# For WS8 integration - export as DEFAULT_TEMPLATES
DEFAULT_TEMPLATES = TEMPLATES


def get_templates_by_category() -> dict[str, list[str]]:
    """Return templates organized by category.

    Returns:
        Dictionary mapping category names to template lists
    """
    return {
        "acknowledgment": ACKNOWLEDGMENT_TEMPLATES,
        "scheduling": SCHEDULING_TEMPLATES,
        "information_request": INFORMATION_REQUEST_TEMPLATES,
        "affirmative": AFFIRMATIVE_TEMPLATES,
        "decline": DECLINE_TEMPLATES,
        "followup": FOLLOWUP_TEMPLATES,
        "gratitude": GRATITUDE_TEMPLATES,
        "confirmation": CONFIRMATION_TEMPLATES,
        "question": QUESTION_TEMPLATES,
        "closing": CLOSING_TEMPLATES,
    }
