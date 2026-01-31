#!/usr/bin/env python3
"""Test framework for classifier-first reply pipeline.

Compares three approaches:
1. Current simple_reply (single call, hopes model asks when uncertain)
2. Reference prompt (single call with hard ASK/RESPOND gate in prompt)
3. Classifier pipeline (two calls: classifier decides, generator executes)

Run: uv run python scripts/experiments/test_classifier_pipeline.py
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results/classifier_pipeline")


# =============================================================================
# CLASSIFIER LABELS
# =============================================================================


class ReplyLabel(Enum):
    RESPOND = "RESPOND"
    ASK = "ASK"
    DO_NOT_REPLY = "DO_NOT_REPLY"
    NEEDS_CONTEXT = "NEEDS_CONTEXT"

    # Sub-labels for ASK (optional, helps generator)
    ASK_TIME = "ASK:TIME"
    ASK_INTENT = "ASK:INTENT"
    ASK_LOCATION = "ASK:LOCATION"
    ASK_WHICH = "ASK:WHICH"


# =============================================================================
# PROMPTS
# =============================================================================

# Classifier prompt - forces single label output
CLASSIFIER_PROMPT = """Return exactly one label: RESPOND, ASK, or DO_NOT_REPLY.

Rules:
- RESPOND: Clear message, enough info to reply
- ASK: Missing info (time, intent, which thing, unclear reference)
- DO_NOT_REPLY: Just "lol", "ok", emoji, or FYI with no question

If unsure between RESPOND and ASK, choose ASK.

Conversation:
{conversation}

Label:"""


# Classifier with reason codes
CLASSIFIER_PROMPT_WITH_REASON = """Return exactly one label.

Labels:
- RESPOND (enough info to reply)
- ASK:TIME (missing when/what time)
- ASK:INTENT (unclear what they want)
- ASK:WHICH (unclear which thing)
- ASK:LOCATION (missing where)
- DO_NOT_REPLY (no response needed)

Choose ASK:* if ANY key detail is missing.

Conversation:
{conversation}

Label:"""


# Generator prompt - constrained by mode
GENERATOR_PROMPT_ASK = """Mode: ASK
Output ONLY one short question (max 10 words). No other text.

Conversation:
{conversation}

Question:"""


GENERATOR_PROMPT_RESPOND = """Mode: RESPOND
Output ONLY the reply message (max 2 sentences). No other text.

You are NOT an assistant. You are impersonating the message author.
Match their texting style from the conversation.

Never use: "Sure", "I can help", "Let me know", "Happy to"

Conversation:
{conversation}

Reply:"""


# Reference implementation - single call with hard gate (for Instruct models)
REFERENCE_PROMPT_INSTRUCT = """You are NOT an assistant.
You are NOT helping a user.

You are impersonating the message author.
Your output will be sent verbatim as a personal message.

If you behave like an assistant, the output is invalid.

Before writing, decide exactly one:
[RESPOND] – enough info to reply naturally
[ASK] – missing info required

You MUST choose [ASK] if:
- Intent is ambiguous
- A decision, time, or commitment is unclear
- Multiple interpretations exist
- You would ask a follow-up in real life

If [ASK], output ONLY one short question.
If [RESPOND], output ONLY the reply.

Never combine asking and responding.
Never explain anything.

Never use:
"Sure", "I can help", "Let me know", explanations, advice.

Maximum length:
- Replies: 2 sentences
- Questions: 1 sentence

Conversation:
{conversation}

Output:"""

# Base model prompt - minimal completion style
REFERENCE_PROMPT_BASE = """[Them]: wanna hang out?
[Me]: when?

[Them]: tomorrow at 6?
[Me]: yeah works

[Them]: did you see it?
[Me]: see what?

[Them]: dinner at 7?
[Me]: sure

[Them]: can you do that thing?
[Me]: which thing?

[Them]: lol
[Me]: lol

[Them]: the usual?
[Me]: usual what?

[Them]: you free?
[Me]: for what?

[Them]: are you coming saturday at 8?
[Me]: yeah

{conversation}
[Me]:"""

# Classifier prompt for Instruct - number format (harder to misinterpret)
CLASSIFIER_PROMPT_INSTRUCT = """Classify message. Output 1, 2, or 3 only.

1 = need to ask (missing info)
2 = can respond (clear)
3 = no reply needed (lol, ok, emoji)

"wanna grab lunch?" → 1
"tomorrow at 6?" → 2
"did you see it?" → 1
"lol" → 3
"dinner saturday at 8?" → 2
"can you do that thing?" → 1
"you free?" → 1
"yeah sounds good" → 2
"👍" → 3
"the usual?" → 1

"{last_message}" →"""

# Classifier prompt for Base - ultra minimal completion
CLASSIFIER_PROMPT_BASE = """[Them]: wanna grab lunch?
ASK

[Them]: tomorrow at 2pm?
RESPOND

[Them]: did you see it?
ASK

[Them]: lol
NO_REPLY

[Them]: coming saturday at 8?
RESPOND

[Them]: can you do that thing?
ASK

[Them]: 👍
NO_REPLY

[Them]: the usual?
ASK

[Them]: you free?
ASK

[Them]: yeah sounds good
RESPOND

{conversation}
"""

# Generator prompts for Base model - ultra minimal
GENERATOR_PROMPT_ASK_BASE = """[Them]: wanna hang?
[Me]: when?

[Them]: did you see it?
[Me]: see what?

[Them]: can you do that thing?
[Me]: which thing?

[Them]: the usual?
[Me]: usual what?

{conversation}
[Me]:"""

GENERATOR_PROMPT_RESPOND_BASE = """[Them]: dinner at 7?
[Me]: sounds good

[Them]: coming saturday?
[Me]: yeah

[Them]: did you finish the report?
[Me]: yep

[Them]: how about thai at 7?
[Me]: works for me

{conversation}
[Me]:"""


# =============================================================================
# TEST CASES
# =============================================================================

# Test cases designed to probe ASK vs RESPOND behavior
TEST_CASES = [
    # Should ASK - vague time references
    {
        "id": "time_vague_1",
        "conversation": [
            ("them", "wanna grab lunch?"),
        ],
        "expected_action": "ASK",
        "expected_ask_type": "TIME",
        "notes": "No time specified - should ask when",
    },
    {
        "id": "time_vague_2",
        "conversation": [
            ("them", "tomorrow works"),
        ],
        "expected_action": "ASK",
        "expected_ask_type": "TIME",
        "notes": "Tomorrow but no time - should ask what time",
    },
    {
        "id": "time_vague_3",
        "conversation": [
            ("them", "let's do it later"),
        ],
        "expected_action": "ASK",
        "expected_ask_type": "TIME",
        "notes": "'Later' is vague - should clarify",
    },
    # Should ASK - unclear intent
    {
        "id": "intent_vague_1",
        "conversation": [
            ("them", "can you do that thing?"),
        ],
        "expected_action": "ASK",
        "expected_ask_type": "INTENT",
        "notes": "'That thing' is ambiguous",
    },
    {
        "id": "intent_vague_2",
        "conversation": [
            ("them", "so what do you think?"),
        ],
        "expected_action": "ASK",
        "expected_ask_type": "INTENT",
        "notes": "Think about what? Need context",
    },
    {
        "id": "intent_vague_3",
        "conversation": [
            ("them", "did you see it?"),
        ],
        "expected_action": "ASK",
        "expected_ask_type": "WHICH",
        "notes": "'It' is unclear reference",
    },
    # Should ASK - multiple interpretations
    {
        "id": "ambiguous_1",
        "conversation": [
            ("them", "the usual?"),
        ],
        "expected_action": "ASK",
        "expected_ask_type": "INTENT",
        "notes": "Usual what? Could be anything",
    },
    {
        "id": "ambiguous_2",
        "conversation": [
            ("them", "you free?"),
        ],
        "expected_action": "ASK",
        "expected_ask_type": "TIME",
        "notes": "Free when? For what?",
    },
    # Should RESPOND - clear questions
    {
        "id": "clear_1",
        "conversation": [
            ("them", "are you coming to the party saturday at 8?"),
        ],
        "expected_action": "RESPOND",
        "notes": "Clear yes/no question with all details",
    },
    {
        "id": "clear_2",
        "conversation": [
            ("them", "what's your address?"),
        ],
        "expected_action": "RESPOND",
        "notes": "Direct factual question",
    },
    {
        "id": "clear_3",
        "conversation": [
            ("them", "did you finish the report?"),
        ],
        "expected_action": "RESPOND",
        "notes": "Clear yes/no about specific thing",
    },
    {
        "id": "clear_4",
        "conversation": [
            ("me", "want to grab dinner?"),
            ("them", "sure! how about thai at 7?"),
        ],
        "expected_action": "RESPOND",
        "notes": "Proposal with specifics - can accept/decline",
    },
    # Should DO_NOT_REPLY
    {
        "id": "no_reply_1",
        "conversation": [
            ("them", "lol"),
        ],
        "expected_action": "DO_NOT_REPLY",
        "notes": "Just lol - no response needed",
    },
    {
        "id": "no_reply_2",
        "conversation": [
            ("them", "👍"),
        ],
        "expected_action": "DO_NOT_REPLY",
        "notes": "Just emoji acknowledgment",
    },
    {
        "id": "no_reply_3",
        "conversation": [
            ("them", "fyi meeting moved to 3pm"),
        ],
        "expected_action": "DO_NOT_REPLY",
        "notes": "FYI message, no question asked",
    },
    # Edge cases - conversation context matters
    {
        "id": "context_1",
        "conversation": [
            ("me", "want to grab coffee?"),
            ("them", "yes"),
        ],
        "expected_action": "ASK",
        "expected_ask_type": "TIME",
        "notes": "They said yes but no time set",
    },
    {
        "id": "context_2",
        "conversation": [
            ("me", "2pm work?"),
            ("them", "yep"),
        ],
        "expected_action": "RESPOND",
        "notes": "Confirmed time - can acknowledge",
    },
]


# =============================================================================
# QUALITY DETECTORS
# =============================================================================

ASSISTANT_PHRASES = [
    "sure",
    "i can help",
    "let me know",
    "happy to",
    "i'd be happy",
    "absolutely",
    "of course",
    "no problem",
    "certainly",
    "definitely",
    "i understand",
    "that makes sense",
    "great question",
    "good question",
    "i see",
    "got it",
    "understood",
    "sounds good to me",
    "feel free",
    "don't hesitate",
    "if you need anything",
    "is there anything else",
    "how can i help",
    "what can i do",
]

GREETING_STARTERS = ["hey!", "hi!", "hello!", "hey there", "hi there"]


def has_assistant_language(text: str) -> bool:
    """Detect assistant-style language."""
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in ASSISTANT_PHRASES)


def has_greeting(text: str) -> bool:
    """Detect assistant-style greetings."""
    text_lower = text.lower().strip()
    return any(text_lower.startswith(g) for g in GREETING_STARTERS)


def is_question(text: str) -> bool:
    """Check if text is a question."""
    return "?" in text or text.lower().startswith(
        (
            "what",
            "when",
            "where",
            "which",
            "who",
            "how",
            "why",
            "do ",
            "did ",
            "are ",
            "is ",
            "can ",
            "could ",
            "would ",
        )
    )


def word_count(text: str) -> int:
    """Count words in text."""
    return len(text.split())


# =============================================================================
# RESULT DATACLASSES
# =============================================================================


@dataclass
class ClassifierResult:
    """Result from classifier step."""

    label: str
    raw_output: str
    time_ms: float


@dataclass
class GeneratorResult:
    """Result from generator step."""

    text: str
    raw_output: str
    time_ms: float


@dataclass
class PipelineResult:
    """Full result from a pipeline run."""

    approach: str
    test_id: str
    expected_action: str

    # What the model did
    actual_action: str  # RESPOND, ASK, DO_NOT_REPLY
    output_text: str

    # Quality metrics
    correct_action: bool
    has_assistant_lang: bool
    has_greeting: bool
    is_question_output: bool
    word_count: int

    # Timing
    total_time_ms: float
    classifier_time_ms: float | None = None
    generator_time_ms: float | None = None

    # Raw outputs for debugging
    raw_outputs: dict = field(default_factory=dict)


# =============================================================================
# PIPELINE IMPLEMENTATIONS
# =============================================================================


class BasePipeline:
    """Base class for reply pipelines."""

    def __init__(self, model_id: str | None = None):
        self._generator = None
        self._model_id = model_id

    @property
    def generator(self):
        if self._generator is None:
            from models import get_generator

            self._generator = get_generator(model_id=self._model_id, skip_templates=True)
        return self._generator

    @property
    def is_base_model(self) -> bool:
        """Check if using a base (non-instruct) model."""
        return self._model_id is not None and "base" in self._model_id.lower()

    def reset_generator(self):
        """Reset generator to force reload with different model."""
        if self._generator is not None:
            self._generator.unload()
            self._generator = None

    def format_conversation(self, messages: list[tuple[str, str]]) -> str:
        """Format conversation for prompt."""
        lines = []
        for speaker, text in messages:
            name = "Me" if speaker.lower() == "me" else "Them"
            lines.append(f"[{name}]: {text}")
        return "\n".join(lines)

    def generate(self, prompt: str, max_tokens: int = 30) -> tuple[str, float]:
        """Generate text and return (output, time_ms)."""
        from contracts.models import GenerationRequest

        start = time.time()
        request = GenerationRequest(
            prompt=prompt,
            context_documents=[],
            few_shot_examples=[],
            max_tokens=max_tokens,
            temperature=0.6,
            top_p=0.9,
        )
        response = self.generator.generate(request)
        elapsed_ms = (time.time() - start) * 1000

        return response.text.strip(), elapsed_ms

    def run(self, test_case: dict) -> PipelineResult:
        """Run pipeline on test case. Override in subclasses."""
        raise NotImplementedError


class SimpleReplyPipeline(BasePipeline):
    """Current simple_reply approach - single call, hopes model asks."""

    name = "simple_reply"

    def __init__(self, model_id: str | None = None):
        super().__init__(model_id)

    PROMPT = """Look at how "Me" texts in this conversation:

{conversation}

Now write what "Me" would say next. Match their exact style - same length, \
same slang, no greetings like "Hey!" or encouraging phrases like "Let's go!". \
Just a normal reply."""

    def run(self, test_case: dict) -> PipelineResult:
        conv_text = self.format_conversation(test_case["conversation"])
        prompt = self.PROMPT.format(conversation=conv_text)

        output, time_ms = self.generate(prompt, max_tokens=25)

        # Determine action from output
        if is_question(output):
            actual_action = "ASK"
        elif not output or output.lower() in ["", "ok", "👍"]:
            actual_action = "DO_NOT_REPLY"
        else:
            actual_action = "RESPOND"

        return PipelineResult(
            approach=self.name,
            test_id=test_case["id"],
            expected_action=test_case["expected_action"],
            actual_action=actual_action,
            output_text=output,
            correct_action=(actual_action == test_case["expected_action"]),
            has_assistant_lang=has_assistant_language(output),
            has_greeting=has_greeting(output),
            is_question_output=is_question(output),
            word_count=word_count(output),
            total_time_ms=time_ms,
            raw_outputs={"full": output},
        )


class ReferencePipeline(BasePipeline):
    """Reference implementation - single call with hard ASK/RESPOND gate."""

    name = "reference"

    def __init__(self, model_id: str | None = None):
        super().__init__(model_id)

    def run(self, test_case: dict) -> PipelineResult:
        conv_text = self.format_conversation(test_case["conversation"])

        # Use model-appropriate prompt
        if self.is_base_model:
            prompt = REFERENCE_PROMPT_BASE.format(conversation=conv_text)
        else:
            prompt = REFERENCE_PROMPT_INSTRUCT.format(conversation=conv_text)

        output, time_ms = self.generate(prompt, max_tokens=40)

        # Determine action from output
        if is_question(output):
            actual_action = "ASK"
        elif not output or output.lower() in ["", "(no reply)", "no reply"]:
            actual_action = "DO_NOT_REPLY"
        else:
            actual_action = "RESPOND"

        return PipelineResult(
            approach=self.name,
            test_id=test_case["id"],
            expected_action=test_case["expected_action"],
            actual_action=actual_action,
            output_text=output,
            correct_action=(actual_action == test_case["expected_action"]),
            has_assistant_lang=has_assistant_language(output),
            has_greeting=has_greeting(output),
            is_question_output=is_question(output),
            word_count=word_count(output),
            total_time_ms=time_ms,
            raw_outputs={"full": output},
        )


class ClassifierPipeline(BasePipeline):
    """Two-call pipeline: classifier decides, generator executes."""

    name = "classifier"

    def __init__(self, model_id: str | None = None):
        super().__init__(model_id)

    def parse_label(self, raw: str) -> str:
        """Parse classifier output to label."""
        raw_clean = raw.strip()

        # Check for numeric format (1/2/3)
        if raw_clean.startswith("1") or " 1" in raw_clean:
            return "ASK"
        if raw_clean.startswith("2") or " 2" in raw_clean:
            return "RESPOND"
        if raw_clean.startswith("3") or " 3" in raw_clean:
            return "DO_NOT_REPLY"

        raw_upper = raw_clean.upper()

        # Check for NO_REPLY (base model format)
        if "NO_REPLY" in raw_upper or "NOREPLY" in raw_upper:
            return "DO_NOT_REPLY"

        # Check for ASK with reason
        if raw_upper.startswith("ASK"):
            return "ASK"

        # Direct matches
        for label in ["RESPOND", "DO_NOT_REPLY", "NEEDS_CONTEXT"]:
            if label in raw_upper:
                return label

        # Fallback - if it's short and has ASK or RESPOND
        if "ASK" in raw_upper:
            return "ASK"
        if "RESPOND" in raw_upper:
            return "RESPOND"

        # Default to RESPOND if unclear
        return "RESPOND"

    def run(self, test_case: dict) -> PipelineResult:
        conv_text = self.format_conversation(test_case["conversation"])
        last_message = test_case["conversation"][-1][1]

        # Step 1: Classify - use model-appropriate prompt
        if self.is_base_model:
            classifier_prompt = CLASSIFIER_PROMPT_BASE.format(conversation=conv_text)
        else:
            classifier_prompt = CLASSIFIER_PROMPT_INSTRUCT.format(last_message=last_message)

        classifier_raw, classifier_time = self.generate(classifier_prompt, max_tokens=5)
        label = self.parse_label(classifier_raw)

        # Debug: log raw classifier output
        logger.debug(f"Classifier raw: '{classifier_raw}' -> {label}")

        # Step 2: Generate based on label - use model-appropriate prompts
        if label == "DO_NOT_REPLY":
            output = "(no reply)"
            generator_time = 0.0
            generator_raw = ""
        elif label == "ASK":
            if self.is_base_model:
                gen_prompt = GENERATOR_PROMPT_ASK_BASE.format(conversation=conv_text)
            else:
                gen_prompt = GENERATOR_PROMPT_ASK.format(conversation=conv_text)
            output, generator_time = self.generate(gen_prompt, max_tokens=20)
            generator_raw = output
        else:  # RESPOND
            if self.is_base_model:
                gen_prompt = GENERATOR_PROMPT_RESPOND_BASE.format(conversation=conv_text)
            else:
                gen_prompt = GENERATOR_PROMPT_RESPOND.format(conversation=conv_text)
            output, generator_time = self.generate(gen_prompt, max_tokens=30)
            generator_raw = output

        total_time = classifier_time + generator_time

        return PipelineResult(
            approach=self.name,
            test_id=test_case["id"],
            expected_action=test_case["expected_action"],
            actual_action=label,
            output_text=output,
            correct_action=(label == test_case["expected_action"]),
            has_assistant_lang=has_assistant_language(output),
            has_greeting=has_greeting(output),
            is_question_output=is_question(output),
            word_count=word_count(output),
            total_time_ms=total_time,
            classifier_time_ms=classifier_time,
            generator_time_ms=generator_time,
            raw_outputs={
                "classifier": classifier_raw,
                "generator": generator_raw,
            },
        )


class ClassifierWithRulesPipeline(ClassifierPipeline):
    """Classifier pipeline with rule-based pre-filtering."""

    name = "classifier_rules"

    # Rules that catch common patterns
    TIME_WORDS = ["tomorrow", "later", "soon", "this weekend", "next week", "sometime"]
    NO_REPLY_PATTERNS = ["lol", "haha", "👍", "😂", "k", "kk"]
    # Short affirmatives that might need follow-up (not auto-no-reply)
    AFFIRMATIVE_PATTERNS = ["yes", "yep", "yeah", "yea", "sure", "ok", "okay"]

    def apply_rules(self, last_message: str) -> str | None:
        """Apply rules to short-circuit classifier. Returns label or None."""
        msg_lower = last_message.lower().strip()

        # DO_NOT_REPLY patterns - pure acknowledgments with no content
        if msg_lower in self.NO_REPLY_PATTERNS:
            return "DO_NOT_REPLY"

        # Single emoji
        if len(msg_lower) <= 2 and not msg_lower.isalpha():
            return "DO_NOT_REPLY"

        # TIME vagueness
        if any(word in msg_lower for word in self.TIME_WORDS):
            # Check if time is actually specified
            time_pattern = r"\d{1,2}(:\d{2})?\s*(am|pm)?|\d{1,2}\s*(am|pm)"
            if not re.search(time_pattern, msg_lower):
                return "ASK"

        return None  # Fall through to classifier

    def run(self, test_case: dict) -> PipelineResult:
        # Get last message
        last_msg = test_case["conversation"][-1][1]

        # Try rules first
        rule_label = self.apply_rules(last_msg)

        if rule_label:
            # Rules matched - skip classifier
            conv_text = self.format_conversation(test_case["conversation"])

            if rule_label == "DO_NOT_REPLY":
                output = "(no reply)"
                gen_time = 0.0
            elif rule_label == "ASK":
                if self.is_base_model:
                    gen_prompt = GENERATOR_PROMPT_ASK_BASE.format(conversation=conv_text)
                else:
                    gen_prompt = GENERATOR_PROMPT_ASK.format(conversation=conv_text)
                output, gen_time = self.generate(gen_prompt, max_tokens=20)
            else:
                if self.is_base_model:
                    gen_prompt = GENERATOR_PROMPT_RESPOND_BASE.format(conversation=conv_text)
                else:
                    gen_prompt = GENERATOR_PROMPT_RESPOND.format(conversation=conv_text)
                output, gen_time = self.generate(gen_prompt, max_tokens=30)

            return PipelineResult(
                approach=self.name,
                test_id=test_case["id"],
                expected_action=test_case["expected_action"],
                actual_action=rule_label,
                output_text=output,
                correct_action=(rule_label == test_case["expected_action"]),
                has_assistant_lang=has_assistant_language(output),
                has_greeting=has_greeting(output),
                is_question_output=is_question(output),
                word_count=word_count(output),
                total_time_ms=gen_time,
                classifier_time_ms=0.0,
                generator_time_ms=gen_time,
                raw_outputs={"rule": rule_label, "generator": output},
            )

        # Fall through to full classifier
        return super().run(test_case)


# =============================================================================
# TEST RUNNER
# =============================================================================


def run_tests(
    pipelines: list[BasePipeline],
    test_cases: list[dict],
    verbose: bool = False,
) -> dict[str, list[PipelineResult]]:
    """Run all test cases through all pipelines."""
    results: dict[str, list[PipelineResult]] = {p.name: [] for p in pipelines}

    for i, test_case in enumerate(test_cases):
        if (i + 1) % 5 == 0:
            logger.info(f"Progress: {i + 1}/{len(test_cases)}")

        for pipeline in pipelines:
            try:
                result = pipeline.run(test_case)
                results[pipeline.name].append(result)

                if verbose:
                    status = "✓" if result.correct_action else "✗"
                    logger.info(
                        f"[{pipeline.name}] {test_case['id']}: "
                        f"{status} expected={result.expected_action} "
                        f"got={result.actual_action} "
                        f"output='{result.output_text[:40]}...'"
                    )
            except Exception as e:
                logger.error(f"[{pipeline.name}] {test_case['id']} failed: {e}")

    return results


def compute_stats(results: list[PipelineResult]) -> dict[str, Any]:
    """Compute aggregate statistics."""
    if not results:
        return {}

    n = len(results)

    return {
        "total": n,
        "correct_action_rate": sum(1 for r in results if r.correct_action) / n,
        "ask_when_should_ask": sum(
            1 for r in results if r.expected_action == "ASK" and r.actual_action == "ASK"
        )
        / max(1, sum(1 for r in results if r.expected_action == "ASK")),
        "respond_when_should_respond": sum(
            1 for r in results if r.expected_action == "RESPOND" and r.actual_action == "RESPOND"
        )
        / max(1, sum(1 for r in results if r.expected_action == "RESPOND")),
        "assistant_lang_rate": sum(1 for r in results if r.has_assistant_lang) / n,
        "greeting_rate": sum(1 for r in results if r.has_greeting) / n,
        "avg_word_count": sum(r.word_count for r in results) / n,
        "avg_time_ms": sum(r.total_time_ms for r in results) / n,
    }


def print_results(all_results: dict[str, list[PipelineResult]]) -> None:
    """Print comparison of pipelines."""
    print("\n" + "=" * 80)
    print("CLASSIFIER PIPELINE EXPERIMENT RESULTS")
    print("=" * 80)

    # Compute stats for each pipeline
    stats = {name: compute_stats(results) for name, results in all_results.items()}

    # Header
    header = (
        f"{'Pipeline':<20} {'Correct':>8} {'ASK↑':>8} {'RESP↑':>8} "
        f"{'Asst%':>8} {'Words':>8} {'Time':>8}"
    )
    print(f"\n{header}")
    print("-" * 80)

    for name, s in stats.items():
        if not s:
            continue
        print(
            f"{name:<20} "
            f"{s['correct_action_rate'] * 100:>7.1f}% "
            f"{s['ask_when_should_ask'] * 100:>7.1f}% "
            f"{s['respond_when_should_respond'] * 100:>7.1f}% "
            f"{s['assistant_lang_rate'] * 100:>7.1f}% "
            f"{s['avg_word_count']:>8.1f} "
            f"{s['avg_time_ms']:>7.0f}ms"
        )

    print("-" * 80)

    # Show failures
    print("\n--- FAILURES (expected != actual) ---\n")
    for name, results in all_results.items():
        failures = [r for r in results if not r.correct_action]
        if failures:
            print(f"\n[{name}] {len(failures)} failures:")
            for f in failures[:5]:  # Show first 5
                print(f"  {f.test_id}: expected={f.expected_action} got={f.actual_action}")
                print(f"    output: '{f.output_text[:60]}'")

    # Show example outputs for same test case
    print("\n--- EXAMPLE OUTPUTS (same test case) ---\n")
    test_id = TEST_CASES[0]["id"]
    print(f"Test: {test_id}")
    print(f"Conversation: {TEST_CASES[0]['conversation']}")
    print(f"Expected: {TEST_CASES[0]['expected_action']}\n")

    for name, results in all_results.items():
        for r in results:
            if r.test_id == test_id:
                status = "✓" if r.correct_action else "✗"
                print(f"[{name}] {status} {r.actual_action}: '{r.output_text}'")
                break


def save_results(all_results: dict[str, list[PipelineResult]], output_dir: Path) -> None:
    """Save results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Convert to serializable format
    data = {}
    for name, results in all_results.items():
        data[name] = {
            "stats": compute_stats(results),
            "results": [
                {
                    "test_id": r.test_id,
                    "expected": r.expected_action,
                    "actual": r.actual_action,
                    "correct": r.correct_action,
                    "output": r.output_text,
                    "has_assistant_lang": r.has_assistant_lang,
                    "word_count": r.word_count,
                    "time_ms": r.total_time_ms,
                }
                for r in results
            ],
        }

    path = output_dir / f"results_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved results to {path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test classifier pipeline approaches")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--pipelines",
        nargs="+",
        choices=["simple_reply", "reference", "classifier", "classifier_rules"],
        default=["simple_reply", "reference", "classifier", "classifier_rules"],
    )
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--dry-run", action="store_true", help="Just show test cases")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model ID to use (e.g., lfm-1.2b, lfm-1.2b-thinking, lfm-1.2b-base)",
    )

    args = parser.parse_args()

    if args.dry_run:
        print("Test cases:")
        for tc in TEST_CASES:
            print(f"  {tc['id']}: {tc['expected_action']} - {tc.get('notes', '')}")
        return 0

    if args.model:
        logger.info(f"Using model: {args.model}")

    # Initialize pipelines
    pipeline_map = {
        "simple_reply": SimpleReplyPipeline,
        "reference": ReferencePipeline,
        "classifier": ClassifierPipeline,
        "classifier_rules": ClassifierWithRulesPipeline,
    }

    pipelines = [pipeline_map[name](model_id=args.model) for name in args.pipelines]

    logger.info(f"Testing {len(pipelines)} pipelines on {len(TEST_CASES)} test cases")

    # Run tests
    results = run_tests(pipelines, TEST_CASES, verbose=args.verbose)

    # Save and print
    save_results(results, Path(args.output_dir))
    print_results(results)

    return 0


if __name__ == "__main__":
    exit(main())
