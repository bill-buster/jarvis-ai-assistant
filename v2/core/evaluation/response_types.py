"""Response type classification and clustering.

Classifies responses into functional types so we can evaluate
whether the model got the RIGHT TYPE of response, even if the
exact words differ.

Types:
- ACCEPT: Agreeing to do something ("yeah", "down", "sure", "bet")
- DECLINE: Refusing/can't do something ("nah", "can't", "busy")
- ACKNOWLEDGE: Simple acknowledgment ("ok", "got it", "word")
- REACT_POSITIVE: Positive reaction ("lol", "nice", "fire", "damn")
- REACT_NEGATIVE: Negative reaction ("rip", "oof", "that sucks")
- QUESTION: Asking for info ("what time?", "where?", "who?")
- INFORM: Providing information ("at the library", "around 5")
- EMPATHY: Showing concern ("sorry to hear", "that's rough")
- THANKS: Gratitude ("thanks", "appreciate it")
- GREETING: Hello/bye ("hey", "sup", "later")
- CONTINUE: Continuing conversation (doesn't fit other categories)
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class ResponseType(Enum):
    """Functional type of a response."""
    ACCEPT = "accept"
    DECLINE = "decline"
    ACKNOWLEDGE = "acknowledge"
    REACT_POSITIVE = "react_positive"
    REACT_NEGATIVE = "react_negative"
    QUESTION = "question"
    INFORM = "inform"
    EMPATHY = "empathy"
    THANKS = "thanks"
    GREETING = "greeting"
    CONTINUE = "continue"


@dataclass
class TypeClassification:
    """Result of classifying a response."""
    response_type: ResponseType
    confidence: float
    matched_pattern: str | None = None


# Pattern-based classification rules
# Order matters - first match wins
RESPONSE_PATTERNS = {
    ResponseType.ACCEPT: [
        r"^(yeah|yea|ya|ye|yes|yep|yup|yuh)\b",
        r"^(sure|ok|okay|k|kk)\b.*\b(down|in|bet|sounds)\b",
        r"^(down|bet|im down|i'm down|im in|i'm in)\b",
        r"^(sounds good|sounds great|sounds fun)\b",
        r"^(lets go|let's go|lesgo|lessgo)\b",
        r"^(for sure|fs|fasho)\b",
        r"^(say less)\b",
    ],
    ResponseType.DECLINE: [
        r"^(no|nah|nope|na)\b",
        r"^(can't|cant|cannot)\b",
        r"^(sorry|sry)\b.*(can't|cant|busy|not)",
        r"^(not today|not rn|not right now)\b",
        r"^(i'm good|im good)\b",  # As in "no thanks"
        r"\b(busy|swamped|slammed)\b",
        r"^(rain check|another time)\b",
    ],
    ResponseType.ACKNOWLEDGE: [
        r"^(ok|okay|k|kk|okie|oki)\s*$",
        r"^(got it|gotcha|gotchu)\b",
        r"^(bet)\s*$",
        r"^(word|werd)\s*$",
        r"^(aight|ight|alright)\s*$",
        r"^(copy|copy that)\b",
        r"^(noted)\b",
        r"^(understood)\b",
    ],
    ResponseType.REACT_POSITIVE: [
        r"^(lol|lmao|lmfao|haha|hahaha|ha)+\b",
        r"^(nice|noice)\b",
        r"^(fire|lit|dope|sick|cool|awesome)\b",
        r"^(damn|dang|dayum)\b(?!.*sorry|bad|rough)",
        r"^(yooo|yoo|yo)\b",
        r"^(lessgo|let's go|ayy|ayyy)\b",
        r"^(w|dub|big w)\b",
        r"^(goat|goated)\b",
    ],
    ResponseType.REACT_NEGATIVE: [
        r"^(rip|f|oof|yikes)\b",
        r"^(damn|dang)\b.*(sorry|bad|rough|sucks)",
        r"^(that sucks|that's rough|thats rough)\b",
        r"^(bruh|bro)\b.*\b(moment|really|seriously)",
        r"^(pain|brutal)\b",
        r"^(L|big L|huge L)\b",
    ],
    ResponseType.QUESTION: [
        r"\?\s*$",  # Ends with question mark
        r"^(what|when|where|who|why|how|which)\b",
        r"^(do you|are you|can you|will you|would you)\b",
        r"^(u |you )(good|ok|sure|down|free|coming)\?",
        r"^(wanna|want to|tryna)\b",
    ],
    ResponseType.INFORM: [
        r"^(i'm at|im at|at the|at my)\b",
        r"^(i'm |im )(on my way|omw|coming|heading)\b",
        r"^(around|about|like)\s+\d",
        r"^\d{1,2}(:\d{2})?\s*(am|pm|ish)?\s*$",
        r"^(running late|be there)\b",
        r"^(eta|omw)\b",
    ],
    ResponseType.EMPATHY: [
        r"^(sorry to hear|sorry about)\b",
        r"^(that's tough|thats tough|that's hard)\b",
        r"^(hope (you're|ur|you are) ok)\b",
        r"^(feel better|get well)\b",
        r"^(oh no|aw|aww)\b",
        r"^(sending (love|hugs|good vibes))\b",
    ],
    ResponseType.THANKS: [
        r"^(thanks|thank you|thx|ty|tysm)\b",
        r"^(appreciate it|appreciate that)\b",
        r"^(you're the best|ur the best)\b",
        r"^(legend|goat)\b",
        r"^(ily|love you|luv u)\b.*\b(for|thanks)\b",
    ],
    ResponseType.GREETING: [
        r"^(hey|hi|hello|yo|sup|what's up|whats up)\b",
        r"^(good morning|good night|gm|gn)\b",
        r"^(later|bye|peace|cya|see ya|ttyl)\b",
    ],
}


def classify_response(text: str) -> TypeClassification:
    """Classify a response into a functional type.

    Args:
        text: The response text to classify

    Returns:
        TypeClassification with type, confidence, and matched pattern
    """
    if not text:
        return TypeClassification(ResponseType.CONTINUE, 0.0)

    text_lower = text.lower().strip()

    # Try pattern matching first
    for response_type, patterns in RESPONSE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                # Confidence based on how specific the pattern is
                confidence = 0.9 if len(text_lower) < 20 else 0.7
                return TypeClassification(
                    response_type=response_type,
                    confidence=confidence,
                    matched_pattern=pattern,
                )

    # Default to CONTINUE for anything that doesn't match
    return TypeClassification(
        response_type=ResponseType.CONTINUE,
        confidence=0.5,
    )


def types_match(type1: ResponseType, type2: ResponseType) -> bool:
    """Check if two response types are functionally equivalent."""
    if type1 == type2:
        return True

    # Define equivalent type groups
    equivalents = [
        {ResponseType.ACCEPT, ResponseType.ACKNOWLEDGE},  # "ok" can be either
        {ResponseType.REACT_POSITIVE, ResponseType.ACKNOWLEDGE},  # "nice" can acknowledge
    ]

    for group in equivalents:
        if type1 in group and type2 in group:
            return True

    return False


def get_type_similarity(type1: ResponseType, type2: ResponseType) -> float:
    """Get similarity score between two response types.

    Returns:
        1.0 if same type
        0.7 if related types (e.g., ACCEPT and ACKNOWLEDGE)
        0.3 if opposite types (e.g., ACCEPT and DECLINE)
        0.5 otherwise
    """
    if type1 == type2:
        return 1.0

    # Related types
    related = [
        (ResponseType.ACCEPT, ResponseType.ACKNOWLEDGE),
        (ResponseType.REACT_POSITIVE, ResponseType.ACKNOWLEDGE),
        (ResponseType.REACT_POSITIVE, ResponseType.ACCEPT),
        (ResponseType.EMPATHY, ResponseType.REACT_NEGATIVE),
    ]
    for t1, t2 in related:
        if (type1 == t1 and type2 == t2) or (type1 == t2 and type2 == t1):
            return 0.7

    # Opposite types
    opposites = [
        (ResponseType.ACCEPT, ResponseType.DECLINE),
        (ResponseType.REACT_POSITIVE, ResponseType.REACT_NEGATIVE),
    ]
    for t1, t2 in opposites:
        if (type1 == t1 and type2 == t2) or (type1 == t2 and type2 == t1):
            return 0.3

    return 0.5


class ResponseTypeEvaluator:
    """Evaluates responses using type-based matching."""

    def __init__(self):
        self._embedding_model = None
        self._type_centroids: dict[ResponseType, np.ndarray] = {}

    def evaluate(
        self,
        generated: str,
        gold: str,
        include_semantic: bool = True,
    ) -> dict:
        """Evaluate a generated response against gold.

        Args:
            generated: The model's generated response
            gold: The actual response (ground truth)
            include_semantic: Whether to include semantic similarity

        Returns:
            Dict with type_match, type_similarity, and optionally semantic_sim
        """
        gen_class = classify_response(generated)
        gold_class = classify_response(gold)

        result = {
            "generated_type": gen_class.response_type.value,
            "gold_type": gold_class.response_type.value,
            "type_match": gen_class.response_type == gold_class.response_type,
            "type_similarity": get_type_similarity(
                gen_class.response_type,
                gold_class.response_type
            ),
            "gen_confidence": gen_class.confidence,
            "gold_confidence": gold_class.confidence,
        }

        if include_semantic:
            result["semantic_sim"] = self._compute_semantic_sim(generated, gold)

        # Combined score: 60% type, 40% semantic
        if include_semantic:
            result["combined_score"] = (
                0.6 * result["type_similarity"] +
                0.4 * result["semantic_sim"]
            )

        return result

    def _compute_semantic_sim(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                return 0.0

        embeddings = self._embedding_model.encode(
            [text1, text2],
            normalize_embeddings=True
        )
        return float(np.dot(embeddings[0], embeddings[1]))

    def evaluate_batch(
        self,
        generated_list: list[str],
        gold_list: list[str],
    ) -> dict:
        """Evaluate a batch of responses.

        Returns:
            Aggregated metrics across all samples
        """
        results = []
        for gen, gold in zip(generated_list, gold_list):
            results.append(self.evaluate(gen, gold))

        # Aggregate
        n = len(results)
        return {
            "n_samples": n,
            "type_match_rate": sum(r["type_match"] for r in results) / n,
            "avg_type_similarity": sum(r["type_similarity"] for r in results) / n,
            "avg_semantic_sim": sum(r["semantic_sim"] for r in results) / n,
            "avg_combined_score": sum(r["combined_score"] for r in results) / n,
            "type_distribution": self._get_type_distribution(results),
        }

    def _get_type_distribution(self, results: list[dict]) -> dict:
        """Get distribution of response types."""
        from collections import Counter
        gen_types = Counter(r["generated_type"] for r in results)
        gold_types = Counter(r["gold_type"] for r in results)
        return {
            "generated": dict(gen_types),
            "gold": dict(gold_types),
        }


# Singleton
_evaluator: ResponseTypeEvaluator | None = None


def get_evaluator() -> ResponseTypeEvaluator:
    """Get singleton evaluator instance."""
    global _evaluator
    if _evaluator is None:
        _evaluator = ResponseTypeEvaluator()
    return _evaluator


# Quick test
if __name__ == "__main__":
    test_cases = [
        ("yeah down", ResponseType.ACCEPT),
        ("sure sounds good", ResponseType.ACCEPT),
        ("nah can't", ResponseType.DECLINE),
        ("sorry busy rn", ResponseType.DECLINE),
        ("lol", ResponseType.REACT_POSITIVE),
        ("nice", ResponseType.REACT_POSITIVE),
        ("ok", ResponseType.ACKNOWLEDGE),
        ("got it", ResponseType.ACKNOWLEDGE),
        ("what time?", ResponseType.QUESTION),
        ("where are you?", ResponseType.QUESTION),
        ("at the library", ResponseType.INFORM),
        ("sorry to hear that", ResponseType.EMPATHY),
        ("thanks!", ResponseType.THANKS),
        ("hey what's up", ResponseType.GREETING),
    ]

    print("Response Type Classification Test")
    print("=" * 60)

    correct = 0
    for text, expected in test_cases:
        result = classify_response(text)
        match = "✓" if result.response_type == expected else "✗"
        if result.response_type == expected:
            correct += 1
        print(f"{match} '{text}' -> {result.response_type.value} (expected {expected.value})")

    print(f"\nAccuracy: {correct}/{len(test_cases)} ({correct/len(test_cases)*100:.1f}%)")
