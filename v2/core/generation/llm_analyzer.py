"""LLM-based conversation analyzer.

Uses an LLM to analyze conversation context instead of keyword heuristics.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from core.models.loader import ModelLoader


@dataclass
class ConversationAnalysis:
    """LLM-generated conversation analysis."""

    relationship: str  # family, close_friend, casual_friend, work, romantic, acquaintance
    formality: str  # very_casual, casual, neutral, formal
    their_tone: str  # playful, serious, warm, distant, excited, upset, neutral
    my_style: str  # description of how I typically respond in this convo
    topics: list[str]  # what they're talking about
    response_type: str  # what kind of response is expected
    raw_analysis: str  # full LLM output


ANALYSIS_PROMPT = '''/no_think
Analyze this iMessage conversation. Output ONLY valid JSON, nothing else.

CONVERSATION:
{conversation}

Output this JSON:
{{"relationship": "<family|close_friend|casual_friend|work|romantic|acquaintance>", "formality": "<very_casual|casual|neutral|formal>", "their_tone": "<playful|serious|warm|excited|upset|neutral>", "my_style_notes": "<how 'me' texts - length, punctuation, slang>", "topics": ["<topic>"], "response_type": "<short ack|answer question|continue chat|emotional support>"}}

JSON:'''


class LLMAnalyzer:
    """Analyzes conversations using an LLM."""

    def __init__(self, model_id: str = "lfm2.5-1.2b"):
        """Initialize with a fast model for analysis."""
        self._loader = ModelLoader(model_id)
        self._loader.preload()

    def analyze(self, messages: list[dict]) -> ConversationAnalysis:
        """Analyze conversation context.

        Args:
            messages: List of {"text": "...", "is_from_me": bool}

        Returns:
            ConversationAnalysis with LLM-generated insights
        """
        # Format conversation
        lines = []
        for m in messages[-15:]:  # Last 15 messages
            prefix = "me:" if m.get("is_from_me") else "them:"
            text = m.get("text", "").strip()
            if text:
                lines.append(f"{prefix} {text}")

        conversation = "\n".join(lines)
        prompt = ANALYSIS_PROMPT.format(conversation=conversation)

        # Generate analysis
        result = self._loader.generate(
            prompt=prompt,
            max_tokens=200,
            temperature=0.1,  # Low temp for consistent output
            stop=["}\n\n", "}\n}", "\n\nCONVERSATION", "<think>", "\n\n"],
        )

        raw = result.text.strip()

        # Parse JSON
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback defaults
            data = {
                "relationship": "casual_friend",
                "formality": "casual",
                "their_tone": "neutral",
                "my_style_notes": "unknown",
                "topics": ["general"],
                "response_type": "continue discussion",
            }

        return ConversationAnalysis(
            relationship=data.get("relationship", "casual_friend"),
            formality=data.get("formality", "casual"),
            their_tone=data.get("their_tone", "neutral"),
            my_style=data.get("my_style_notes", ""),
            topics=data.get("topics", ["general"]),
            response_type=data.get("response_type", "respond"),
            raw_analysis=raw,
        )

    def analyze_batch(self, conversations: list[list[dict]]) -> list[ConversationAnalysis]:
        """Analyze multiple conversations."""
        return [self.analyze(msgs) for msgs in conversations]

    def unload(self):
        """Unload the model."""
        self._loader.unload()


# Quick test
if __name__ == "__main__":
    analyzer = LLMAnalyzer()

    test_msgs = [
        {"text": "yo what are you up to", "is_from_me": False},
        {"text": "nm just working", "is_from_me": True},
        {"text": "wanna grab dinner later", "is_from_me": False},
        {"text": "yea sure where", "is_from_me": True},
        {"text": "idk maybe that new ramen place", "is_from_me": False},
    ]

    result = analyzer.analyze(test_msgs)
    print(f"Relationship: {result.relationship}")
    print(f"Formality: {result.formality}")
    print(f"Their tone: {result.their_tone}")
    print(f"My style: {result.my_style}")
    print(f"Topics: {result.topics}")
    print(f"Response type: {result.response_type}")
    print(f"\nRaw: {result.raw_analysis}")

    analyzer.unload()
