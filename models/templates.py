"""Template Matcher for semantic template matching.

Bypasses model generation for common request patterns using
semantic similarity with sentence embeddings.
"""

from __future__ import annotations

import gc
import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Lazy-loaded sentence transformer
_sentence_model: SentenceTransformer | None = None


class SentenceModelError(Exception):
    """Raised when sentence transformer model cannot be loaded."""


def _get_sentence_model() -> Any:
    """Lazy-load the sentence transformer model.

    Returns:
        The loaded SentenceTransformer model

    Raises:
        SentenceModelError: If model cannot be loaded (network issues, etc.)
    """
    global _sentence_model
    if _sentence_model is None:
        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading sentence transformer: all-MiniLM-L6-v2")
            _sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            logger.exception("Failed to load sentence transformer")
            msg = f"Failed to load sentence transformer: {e}"
            raise SentenceModelError(msg) from e
    return _sentence_model


def unload_sentence_model() -> None:
    """Unload the sentence transformer model to free memory.

    Call this when template matching is no longer needed and you want
    to reclaim memory for other operations (e.g., loading the MLX model).
    """
    global _sentence_model
    if _sentence_model is not None:
        logger.info("Unloading sentence transformer model")
        _sentence_model = None
        gc.collect()


def is_sentence_model_loaded() -> bool:
    """Check if the sentence transformer model is currently loaded.

    Returns:
        True if model is loaded, False otherwise
    """
    return _sentence_model is not None


@dataclass
class ResponseTemplate:
    """A template for common response patterns."""

    name: str
    patterns: list[str]  # Example prompts that match this template
    response: str  # The response to return


@dataclass
class TemplateMatch:
    """Result of template matching."""

    template: ResponseTemplate
    similarity: float
    matched_pattern: str


def _get_minimal_fallback_templates() -> list[ResponseTemplate]:
    """Minimal templates for development when WS3 not available."""
    return [
        ResponseTemplate(
            name="thank_you_acknowledgment",
            patterns=[
                "Thanks for sending the report",
                "Thank you for the update",
                "Thanks for letting me know",
                "Thank you for your email",
                "Thanks for the information",
            ],
            response="You're welcome! Let me know if you need anything else.",
        ),
        ResponseTemplate(
            name="meeting_confirmation",
            patterns=[
                "Confirming our meeting tomorrow",
                "Just confirming our call",
                "Confirming the meeting time",
                "See you at the meeting",
                "Looking forward to our meeting",
            ],
            response="Confirmed! Looking forward to it.",
        ),
        ResponseTemplate(
            name="schedule_request",
            patterns=[
                "Can we schedule a meeting",
                "When are you free to meet",
                "Let's set up a call",
                "What times work for you",
                "Can we find a time to talk",
            ],
            response="I'd be happy to meet. Could you share a few time options that work for you?",
        ),
        ResponseTemplate(
            name="acknowledgment",
            patterns=[
                "Got it",
                "Understood",
                "Makes sense",
                "Sounds good",
                "Perfect",
            ],
            response="Great, thanks for confirming!",
        ),
        ResponseTemplate(
            name="file_receipt",
            patterns=[
                "I've attached the file",
                "Please find attached",
                "Here's the document",
                "Attached is the file you requested",
                "I'm sending over the file",
            ],
            response="Thanks for sending this over! I'll review it shortly.",
        ),
        ResponseTemplate(
            name="deadline_reminder",
            patterns=[
                "Just a reminder about the deadline",
                "Don't forget the deadline",
                "Reminder: deadline approaching",
                "The deadline is coming up",
                "Final reminder about the due date",
            ],
            response="Thanks for the reminder! I'm on track to complete this by the deadline.",
        ),
        ResponseTemplate(
            name="greeting",
            patterns=[
                "Hi, how are you",
                "Hello, hope you're doing well",
                "Good morning",
                "Hey, hope all is well",
                "Hi there",
            ],
            response="Hi! I'm doing well, thanks for asking. How can I help you today?",
        ),
        ResponseTemplate(
            name="out_of_office",
            patterns=[
                "I'll be out of office",
                "I'm on vacation",
                "I'll be unavailable",
                "Out of the office until",
                "Taking some time off",
            ],
            response="Thanks for letting me know! Enjoy your time off.",
        ),
        ResponseTemplate(
            name="follow_up",
            patterns=[
                "Just following up",
                "Wanted to check in",
                "Any updates on this",
                "Circling back on this",
                "Following up on my previous email",
            ],
            response="Thanks for following up! Let me check on this and get back to you shortly.",
        ),
        ResponseTemplate(
            name="apology",
            patterns=[
                "Sorry for the delay",
                "Apologies for the late response",
                "Sorry I missed your message",
                "My apologies for not responding sooner",
                "Sorry for the wait",
            ],
            response="No worries at all! I appreciate you getting back to me.",
        ),
        # iMessage-specific templates for quick text patterns
        ResponseTemplate(
            name="quick_ok",
            patterns=[
                "ok",
                "k",
                "kk",
                "okay",
                "okie",
                "okey",
                "alright",
            ],
            response="Got it!",
        ),
        ResponseTemplate(
            name="quick_affirmative",
            patterns=[
                "sure",
                "yep",
                "yup",
                "yeah",
                "ya",
                "yes",
                "definitely",
                "for sure",
            ],
            response="Sounds good!",
        ),
        ResponseTemplate(
            name="quick_thanks",
            patterns=[
                "thx",
                "ty",
                "tysm",
                "thanks",
                "thank u",
                "thank you",
                "thanks!",
            ],
            response="You're welcome!",
        ),
        ResponseTemplate(
            name="quick_no_problem",
            patterns=[
                "np",
                "no prob",
                "no problem",
                "no worries",
                "nw",
                "all good",
            ],
            response="Glad I could help!",
        ),
        ResponseTemplate(
            name="on_my_way",
            patterns=[
                "omw",
                "on my way",
                "leaving now",
                "heading out",
                "be there in 10",
                "coming now",
                "just left",
            ],
            response="See you soon!",
        ),
        ResponseTemplate(
            name="be_there_soon",
            patterns=[
                "be there soon",
                "almost there",
                "5 mins away",
                "pulling up",
                "around the corner",
                "just about there",
            ],
            response="Great, see you in a bit!",
        ),
        ResponseTemplate(
            name="running_late",
            patterns=[
                "running late",
                "gonna be late",
                "running behind",
                "stuck in traffic",
                "be there in a bit",
                "sorry running late",
            ],
            response="No worries, take your time!",
        ),
        ResponseTemplate(
            name="where_are_you",
            patterns=[
                "where are you",
                "where r u",
                "where you at",
                "wya",
                "where u at",
                "you here yet",
            ],
            response="On my way! Be there soon.",
        ),
        ResponseTemplate(
            name="what_time",
            patterns=[
                "what time",
                "when",
                "what time works",
                "when should we meet",
                "what time is good",
                "when are you free",
            ],
            response="Let me check my schedule and get back to you!",
        ),
        ResponseTemplate(
            name="time_proposal",
            patterns=[
                "how about 3",
                "does 5 work",
                "is 7 ok",
                "maybe around 2",
                "let's say noon",
                "how's 6pm",
                "works for me at 4",
            ],
            response="That time works for me!",
        ),
        ResponseTemplate(
            name="hang_out_invite",
            patterns=[
                "wanna hang",
                "want to hang out",
                "down to hang",
                "u free",
                "you free",
                "wanna chill",
                "want to chill",
            ],
            response="Yeah, I'm down! What did you have in mind?",
        ),
        ResponseTemplate(
            name="dinner_plans",
            patterns=[
                "down for dinner",
                "wanna grab dinner",
                "want to get food",
                "hungry?",
                "wanna eat",
                "let's get food",
                "dinner tonight?",
            ],
            response="I'm in! Where were you thinking?",
        ),
        ResponseTemplate(
            name="free_tonight",
            patterns=[
                "free tonight",
                "doing anything tonight",
                "busy tonight",
                "plans tonight",
                "what are you doing tonight",
                "got plans",
            ],
            response="Let me check! What's up?",
        ),
        ResponseTemplate(
            name="coffee_drinks",
            patterns=[
                "let's grab coffee",
                "wanna get coffee",
                "coffee sometime",
                "grab a drink",
                "wanna get drinks",
                "drinks later",
            ],
            response="Sounds great! When works for you?",
        ),
        ResponseTemplate(
            name="laughter",
            patterns=[
                "lol",
                "lmao",
                "haha",
                "hahaha",
                "lolol",
                "rofl",
                "dying",
            ],
            response="Haha right?!",
        ),
        ResponseTemplate(
            name="emoji_reaction",
            patterns=[
                "😂",
                "🤣",
                "😭",
                "💀",
                "😆",
                "🙃",
            ],
            response="😊",
        ),
        ResponseTemplate(
            name="positive_reaction",
            patterns=[
                "nice",
                "nice!",
                "awesome",
                "amazing",
                "love it",
                "so cool",
                "that's great",
                "dope",
                "sick",
            ],
            response="Thanks! 😊",
        ),
        ResponseTemplate(
            name="check_in",
            patterns=[
                "you there",
                "you there?",
                "u there",
                "hello?",
                "hey?",
                "you alive",
                "earth to you",
            ],
            response="I'm here! What's up?",
        ),
        ResponseTemplate(
            name="did_you_see",
            patterns=[
                "did you see my text",
                "did you get my message",
                "see my last message",
                "did you read that",
                "u see that",
                "did u see",
            ],
            response="Just saw it! Let me respond.",
        ),
        ResponseTemplate(
            name="talk_later",
            patterns=[
                "ttyl",
                "talk later",
                "talk to you later",
                "catch you later",
                "later",
                "laters",
                "chat soon",
            ],
            response="Talk soon!",
        ),
        ResponseTemplate(
            name="goodnight",
            patterns=[
                "gn",
                "goodnight",
                "good night",
                "night",
                "nite",
                "sleep well",
                "sweet dreams",
            ],
            response="Goodnight! Sleep well!",
        ),
        ResponseTemplate(
            name="goodbye",
            patterns=[
                "bye",
                "bye!",
                "cya",
                "see ya",
                "see you",
                "peace",
                "take care",
            ],
            response="Bye! Take care!",
        ),
        ResponseTemplate(
            name="appreciation",
            patterns=[
                "you're the best",
                "ur the best",
                "appreciate it",
                "thanks so much",
                "really appreciate it",
                "you rock",
                "ily",
                "love you",
            ],
            response="Aw, thanks! You're the best too!",
        ),
        ResponseTemplate(
            name="question_response",
            patterns=[
                "idk",
                "i don't know",
                "dunno",
                "not sure",
                "no idea",
                "beats me",
            ],
            response="No worries, we can figure it out!",
        ),
        ResponseTemplate(
            name="agreement",
            patterns=[
                "same",
                "same here",
                "me too",
                "i agree",
                "totally",
                "exactly",
                "fr",
                "for real",
            ],
            response="Right?!",
        ),
        ResponseTemplate(
            name="brb",
            patterns=[
                "brb",
                "be right back",
                "one sec",
                "gimme a sec",
                "hold on",
                "one minute",
                "give me a minute",
            ],
            response="No rush, take your time!",
        ),
        # iMessage Assistant Scenarios - queries to the AI assistant about messages
        ResponseTemplate(
            name="summarize_conversation",
            patterns=[
                "summarize my conversation with",
                "give me a summary of my chat with",
                "what did I talk about with",
                "summarize the messages from",
                "recap my conversation with",
                "what's the summary of my texts with",
                "sum up my chat with",
            ],
            response=(
                "I'll analyze your conversation and provide a summary of the key points, "
                "topics discussed, and any action items mentioned."
            ),
        ),
        ResponseTemplate(
            name="summarize_recent_messages",
            patterns=[
                "summarize my recent messages",
                "what have I been texting about",
                "recap my recent conversations",
                "summarize my texts from today",
                "what's been happening in my messages",
                "give me a summary of recent chats",
                "summarize today's messages",
            ],
            response=(
                "I'll review your recent messages and provide a summary of conversations, "
                "key topics, and any items that may need your attention."
            ),
        ),
        ResponseTemplate(
            name="find_messages_from_person",
            patterns=[
                "find messages from",
                "show me texts from",
                "what did say",
                "messages from",
                "show messages from",
                "get messages from",
                "search messages from",
            ],
            response=(
                "I'll search your messages and show you the conversations "
                "from that person. You can specify a time range if needed."
            ),
        ),
        ResponseTemplate(
            name="find_unread_messages",
            patterns=[
                "show me unread messages",
                "what messages haven't I read",
                "do I have unread texts",
                "any unread messages",
                "show unread",
                "unread messages",
                "messages I haven't seen",
            ],
            response=(
                "I'll check for any messages you haven't read yet "
                "and show you a summary of who they're from."
            ),
        ),
        ResponseTemplate(
            name="unread_message_recap",
            patterns=[
                "recap my unread messages",
                "summarize unread texts",
                "what did I miss",
                "catch me up on messages",
                "what messages did I miss",
                "summarize what I haven't read",
                "what's new in my messages",
            ],
            response=(
                "I'll provide a recap of your unread messages, "
                "highlighting important ones and summarizing the rest."
            ),
        ),
        ResponseTemplate(
            name="find_dates_times",
            patterns=[
                "find messages about dates",
                "when did we plan to meet",
                "search for times mentioned",
                "find messages with dates",
                "what dates were mentioned",
                "find scheduled times",
                "search for meeting times",
            ],
            response=(
                "I'll search your messages for mentions of dates, times, "
                "and scheduled events to help you find what you're looking for."
            ),
        ),
        ResponseTemplate(
            name="find_shared_links",
            patterns=[
                "find links in messages",
                "show shared links",
                "what links did they send",
                "find urls in my texts",
                "search for shared links",
                "show me links from",
                "find websites shared",
            ],
            response=(
                "I'll search your messages for shared links and URLs, "
                "and show you when and who shared them."
            ),
        ),
        ResponseTemplate(
            name="find_shared_photos",
            patterns=[
                "find photos in messages",
                "show shared photos",
                "what pictures did they send",
                "find images in my texts",
                "search for photos from",
                "show me pictures from",
                "find shared images",
            ],
            response=(
                "I'll search your messages for shared photos and images, "
                "showing you who sent them and when."
            ),
        ),
        ResponseTemplate(
            name="find_attachments",
            patterns=[
                "find attachments in messages",
                "show shared files",
                "what files did they send",
                "find documents in my texts",
                "search for attachments from",
                "show me files from",
                "find shared documents",
            ],
            response=(
                "I'll search your messages for attachments and files, "
                "showing you the type, sender, and date for each."
            ),
        ),
        ResponseTemplate(
            name="search_topic",
            patterns=[
                "find messages about",
                "search for texts about",
                "show messages mentioning",
                "find conversations about",
                "search my messages for",
                "find texts mentioning",
                "look for messages about",
            ],
            response=(
                "I'll search your messages for that topic and show you relevant conversations."
            ),
        ),
        ResponseTemplate(
            name="search_keyword",
            patterns=[
                "search for keyword",
                "find texts containing",
                "search messages for word",
                "find messages with word",
                "look for keyword in messages",
                "search for specific word",
                "find word in my texts",
            ],
            response=(
                "I'll search your messages for that keyword and show you all matches with context."
            ),
        ),
        ResponseTemplate(
            name="recent_conversations",
            patterns=[
                "who have I texted recently",
                "show recent conversations",
                "who messaged me lately",
                "my recent chats",
                "show my latest conversations",
                "who have I been talking to",
                "list recent contacts",
            ],
            response=(
                "I'll show you a list of your most recent conversations, "
                "sorted by the last message time."
            ),
        ),
        ResponseTemplate(
            name="messages_from_today",
            patterns=[
                "show today's messages",
                "what messages did I get today",
                "today's texts",
                "messages from today",
                "show me today's conversations",
                "who texted me today",
                "today's chats",
            ],
            response=(
                "I'll show you all the messages you've received today, organized by conversation."
            ),
        ),
        ResponseTemplate(
            name="messages_from_yesterday",
            patterns=[
                "show yesterday's messages",
                "what messages did I get yesterday",
                "yesterday's texts",
                "messages from yesterday",
                "show me yesterday's conversations",
                "who texted me yesterday",
                "yesterday's chats",
            ],
            response=("I'll show you all the messages from yesterday, organized by conversation."),
        ),
        ResponseTemplate(
            name="messages_this_week",
            patterns=[
                "show this week's messages",
                "messages from this week",
                "what texts did I get this week",
                "this week's conversations",
                "show me messages since monday",
                "weekly message summary",
                "recap of this week's texts",
            ],
            response=(
                "I'll provide a summary of your messages from this week, "
                "highlighting key conversations and topics."
            ),
        ),
        ResponseTemplate(
            name="find_address_location",
            patterns=[
                "find addresses in messages",
                "search for locations shared",
                "what addresses were sent",
                "find location in texts",
                "search for places mentioned",
                "find shared locations",
                "where did they say to meet",
            ],
            response=(
                "I'll search your messages for addresses and locations "
                "that were shared or mentioned."
            ),
        ),
        ResponseTemplate(
            name="find_phone_numbers",
            patterns=[
                "find phone numbers in messages",
                "search for numbers shared",
                "what phone numbers were sent",
                "find contact numbers in texts",
                "search for phone numbers",
                "find shared phone numbers",
                "numbers mentioned in messages",
            ],
            response=(
                "I'll search your messages for phone numbers and show you who shared them and when."
            ),
        ),
        ResponseTemplate(
            name="message_count",
            patterns=[
                "how many messages from",
                "count messages from",
                "how many texts did I get",
                "message count with",
                "how many times did they text",
                "count my messages",
                "how many texts today",
            ],
            response=(
                "I'll count the messages matching your criteria "
                "and provide you with the statistics."
            ),
        ),
        ResponseTemplate(
            name="last_message_from",
            patterns=[
                "when did I last hear from",
                "last message from",
                "when did they last text",
                "last time I heard from",
                "most recent message from",
                "when was the last text from",
                "how long since I heard from",
            ],
            response=(
                "I'll find the most recent message from that person and tell you when it was sent."
            ),
        ),
        ResponseTemplate(
            name="find_plans_events",
            patterns=[
                "find plans in messages",
                "what events are mentioned",
                "search for plans we made",
                "find scheduled events",
                "what did we plan",
                "search for upcoming plans",
                "find events in my texts",
            ],
            response=(
                "I'll search your messages for mentions of plans, events, and scheduled activities."
            ),
        ),
        ResponseTemplate(
            name="find_recommendations",
            patterns=[
                "find recommendations in messages",
                "what did they recommend",
                "search for suggestions",
                "find recommended places",
                "what restaurants were suggested",
                "find movie recommendations",
                "search for recommendations",
            ],
            response=(
                "I'll search your messages for recommendations and suggestions "
                "that were shared with you."
            ),
        ),
        ResponseTemplate(
            name="group_chat_summary",
            patterns=[
                "summarize the group chat",
                "what happened in the group",
                "recap group conversation",
                "group chat summary",
                "what did I miss in group",
                "summarize group messages",
                "catch me up on group chat",
            ],
            response=(
                "I'll provide a summary of the group chat, including key discussions, "
                "decisions made, and any action items."
            ),
        ),
        ResponseTemplate(
            name="who_mentioned_me",
            patterns=[
                "who mentioned me",
                "find messages mentioning my name",
                "was I mentioned in any chats",
                "search for mentions of me",
                "who talked about me",
                "find where I was mentioned",
                "any messages about me",
            ],
            response=(
                "I'll search your messages for mentions of your name "
                "and show you the relevant conversations."
            ),
        ),
        ResponseTemplate(
            name="important_messages",
            patterns=[
                "show important messages",
                "find urgent texts",
                "what messages need attention",
                "priority messages",
                "find important conversations",
                "urgent messages",
                "messages that need reply",
            ],
            response=(
                "I'll identify messages that may need your attention based on "
                "content, sender, and conversation context."
            ),
        ),
        ResponseTemplate(
            name="conversation_history",
            patterns=[
                "show full conversation with",
                "entire chat history with",
                "all messages with",
                "complete conversation with",
                "full message history with",
                "show all texts with",
                "entire chat with",
            ],
            response=(
                "I'll show you the complete conversation history with that person, "
                "starting from the earliest message."
            ),
        ),
    ]


def _load_templates() -> list[ResponseTemplate]:
    """Load templates from WS3, fallback to minimal set.

    WS3 templates are organized by category with response strings.
    We convert them to ResponseTemplate objects, using the response
    as both pattern and response for simple matching.
    """
    try:
        from benchmarks.coverage.templates import get_templates_by_category

        category_templates = get_templates_by_category()
        templates = []

        for category, responses in category_templates.items():
            # Convert WS3 format: use responses as both patterns and responses
            templates.append(
                ResponseTemplate(
                    name=category,
                    patterns=responses,  # Use responses as matching patterns
                    response=responses[0],  # Use first response as default
                )
            )

        logger.info("Loaded %d template categories from WS3", len(templates))
        return templates
    except ImportError:
        logger.warning("WS3 templates not available, using minimal fallback set")
        return _get_minimal_fallback_templates()


class TemplateMatcher:
    """Semantic template matcher using sentence embeddings.

    Computes cosine similarity between input prompt and template patterns.
    Returns best matching template if similarity exceeds threshold.
    """

    SIMILARITY_THRESHOLD = 0.7

    def __init__(self, templates: list[ResponseTemplate] | None = None) -> None:
        """Initialize the template matcher.

        Args:
            templates: List of templates to use. Loads defaults if not provided.
        """
        self.templates = templates or _load_templates()
        self._pattern_embeddings: np.ndarray | None = None
        self._pattern_to_template: list[tuple[str, ResponseTemplate]] = []
        self._embeddings_lock = threading.Lock()

    def _ensure_embeddings(self) -> None:
        """Compute and cache embeddings for all template patterns.

        Uses double-check locking for thread-safe lazy initialization.
        """
        # Fast path: embeddings already computed
        if self._pattern_embeddings is not None:
            return

        # Slow path: acquire lock and double-check
        with self._embeddings_lock:
            # Double-check after acquiring lock
            if self._pattern_embeddings is not None:
                return

            model = _get_sentence_model()

            # Collect all patterns with their templates
            all_patterns = []
            pattern_to_template: list[tuple[str, ResponseTemplate]] = []
            for template in self.templates:
                for pattern in template.patterns:
                    all_patterns.append(pattern)
                    pattern_to_template.append((pattern, template))

            # Compute embeddings in batch
            embeddings = model.encode(all_patterns, convert_to_numpy=True)

            # Assign both atomically (pattern_to_template first, then embeddings)
            self._pattern_to_template = pattern_to_template
            self._pattern_embeddings = embeddings
            logger.info("Computed embeddings for %d patterns", len(all_patterns))

    def match(self, query: str) -> TemplateMatch | None:
        """Find best matching template for a query.

        Args:
            query: Input prompt to match against templates

        Returns:
            TemplateMatch if similarity >= threshold, None otherwise.
            Returns None if sentence model fails to load (falls back to model generation).
        """
        try:
            self._ensure_embeddings()

            # Type guard: _ensure_embeddings guarantees this is not None
            pattern_embeddings = self._pattern_embeddings
            if pattern_embeddings is None:
                return None

            model = _get_sentence_model()
            query_embedding = model.encode([query], convert_to_numpy=True)[0]

            # Compute cosine similarities
            similarities = np.dot(pattern_embeddings, query_embedding) / (
                np.linalg.norm(pattern_embeddings, axis=1) * np.linalg.norm(query_embedding)
            )

            best_idx = int(np.argmax(similarities))
            best_similarity = float(similarities[best_idx])

            if best_similarity >= self.SIMILARITY_THRESHOLD:
                matched_pattern, template = self._pattern_to_template[best_idx]
                # Log template match without user query content (privacy)
                logger.debug(
                    "Template match: %s (similarity: %.3f)",
                    template.name,
                    best_similarity,
                )
                return TemplateMatch(
                    template=template,
                    similarity=best_similarity,
                    matched_pattern=matched_pattern,
                )

            logger.debug(
                "No template match (best similarity: %.3f, threshold: %.3f)",
                best_similarity,
                self.SIMILARITY_THRESHOLD,
            )
            return None

        except SentenceModelError:
            # Fall back to model generation if sentence model unavailable
            logger.warning("Template matching unavailable, falling back to model generation")
            return None

    def clear_cache(self) -> None:
        """Clear cached embeddings.

        Call this after unload_sentence_model() to ensure embeddings
        are recomputed when the model is reloaded.
        """
        self._pattern_embeddings = None
        self._pattern_to_template = []
        logger.debug("Template matcher cache cleared")
