"""Tests for jarvis/text_normalizer.py - Text cleaning and feature extraction."""

from jarvis.text_normalizer import (
    extract_temporal_refs,
    extract_text_features,
    get_attachment_token,
    is_acknowledgment_only,
    is_emoji_only,
    is_question,
    is_reaction,
    normalize_text,
    starts_new_topic,
    trigger_expects_content,
)


class TestNormalizeText:
    """Tests for the normalize_text function."""

    def test_basic_normalization(self) -> None:
        """Test basic whitespace normalization."""
        assert normalize_text("  hello   world  ") == "hello world"
        assert normalize_text("hello\n\nworld") == "hello\nworld"

    def test_reaction_removal(self) -> None:
        """Test that reactions return empty string."""
        assert normalize_text('Liked "hey there"') == ""
        assert normalize_text('Loved "great job!"') == ""
        assert normalize_text('Laughed at "haha"') == ""
        assert normalize_text('Emphasized "important!"') == ""
        assert normalize_text('Questioned "really?"') == ""

    def test_removed_reactions(self) -> None:
        """Test removal reactions."""
        assert normalize_text('Removed a like from "message"') == ""
        assert normalize_text('Removed a heart from "message"') == ""

    def test_signature_stripping(self) -> None:
        """Test auto-signature removal."""
        text = "Hello!\nSent from my iPhone"
        assert normalize_text(text) == "Hello!"

        text = "Message\n--\nJohn Doe\nSome Company"
        assert normalize_text(text) == "Message"

    def test_emoji_collapse(self) -> None:
        """Test collapsing repeated emojis."""
        # 3+ emojis collapse to 2
        assert "😂😂😂😂" not in normalize_text("haha 😂😂😂😂")
        result = normalize_text("haha 😂😂😂😂")
        assert "😂😂" in result

    def test_preserve_newlines(self) -> None:
        """Test that meaningful newlines are preserved."""
        text = "Line one\nLine two\nLine three"
        result = normalize_text(text)
        assert result.count("\n") == 2

    def test_empty_input(self) -> None:
        """Test empty input handling."""
        assert normalize_text("") == ""
        assert normalize_text(None) == ""  # type: ignore

    def test_collapse_emojis_disabled(self) -> None:
        """Test disabling emoji collapse."""
        text = "haha 😂😂😂😂😂"
        result = normalize_text(text, collapse_emojis=False)
        assert "😂😂😂😂😂" in result


class TestIsReaction:
    """Tests for reaction detection."""

    def test_positive_cases(self) -> None:
        """Test recognized reaction patterns."""
        assert is_reaction('Liked "test"')
        assert is_reaction('Loved "message"')
        assert is_reaction('Laughed at "joke"')
        assert is_reaction('Emphasized "important"')
        assert is_reaction('Questioned "what?"')
        assert is_reaction('Disliked "bad"')

    def test_negative_cases(self) -> None:
        """Test non-reaction text."""
        assert not is_reaction("I liked that movie")
        assert not is_reaction("Hello there")
        assert not is_reaction("")


class TestIsAcknowledgmentOnly:
    """Tests for acknowledgment detection."""

    def test_positive_cases(self) -> None:
        """Test recognized acknowledgments."""
        assert is_acknowledgment_only("ok")
        assert is_acknowledgment_only("OK")
        assert is_acknowledgment_only("okay")
        assert is_acknowledgment_only("k")
        assert is_acknowledgment_only("thanks")
        assert is_acknowledgment_only("thank you")
        assert is_acknowledgment_only("sure")
        assert is_acknowledgment_only("lol")
        assert is_acknowledgment_only("haha")
        assert is_acknowledgment_only("sounds good")
        assert is_acknowledgment_only("got it")
        assert is_acknowledgment_only("omw")
        assert is_acknowledgment_only("bet")

    def test_with_punctuation(self) -> None:
        """Test acknowledgments with trailing punctuation."""
        assert is_acknowledgment_only("ok!")
        assert is_acknowledgment_only("thanks.")
        assert is_acknowledgment_only("sure?")

    def test_negative_cases(self) -> None:
        """Test non-acknowledgment text."""
        assert not is_acknowledgment_only("I'm on my way to the store")
        assert not is_acknowledgment_only("That sounds good to me")
        assert not is_acknowledgment_only("Hello")


class TestIsEmojiOnly:
    """Tests for emoji-only detection."""

    def test_positive_cases(self) -> None:
        """Test emoji-only strings."""
        assert is_emoji_only("😀")
        assert is_emoji_only("😂😂😂")
        assert is_emoji_only("👍 👌")
        assert is_emoji_only("🎉🎊🥳")

    def test_negative_cases(self) -> None:
        """Test strings with non-emoji content."""
        assert not is_emoji_only("hello 😀")
        assert not is_emoji_only("ok")
        assert not is_emoji_only("")


class TestStartsNewTopic:
    """Tests for topic-shift marker detection."""

    def test_positive_cases(self) -> None:
        """Test recognized topic-shift markers."""
        assert starts_new_topic("btw, did you see that?")
        assert starts_new_topic("by the way, I forgot to mention")
        assert starts_new_topic("anyway, what's the plan?")
        assert starts_new_topic("also, I need to ask you something")
        assert starts_new_topic("speaking of, have you talked to John?")
        assert starts_new_topic("random but are you free?")

    def test_negative_cases(self) -> None:
        """Test non-topic-shift text."""
        assert not starts_new_topic("That's a great idea")
        assert not starts_new_topic("I agree")
        assert not starts_new_topic("")


class TestIsQuestion:
    """Tests for question detection."""

    def test_punctuation_questions(self) -> None:
        """Test questions by punctuation."""
        assert is_question("What time?")
        assert is_question("Are you coming?")
        assert is_question("Really?")

    def test_question_words(self) -> None:
        """Test questions by question words."""
        assert is_question("What do you think")
        assert is_question("Where should we go")
        assert is_question("When is it happening")
        assert is_question("How does that work")
        assert is_question("Who is coming")
        assert is_question("Why not")
        assert is_question("Which one do you prefer")

    def test_negative_cases(self) -> None:
        """Test non-questions."""
        assert not is_question("I'll be there")
        assert not is_question("Sounds good")
        assert not is_question("")


class TestExtractTemporalRefs:
    """Tests for temporal reference extraction."""

    def test_relative_times(self) -> None:
        """Test relative time references."""
        refs = extract_temporal_refs("Let's meet tomorrow")
        assert any("tomorrow" in str(r).lower() for r in refs)

        refs = extract_temporal_refs("See you later today")
        assert any("today" in str(r).lower() or "later" in str(r).lower() for r in refs)

    def test_specific_times(self) -> None:
        """Test specific time references."""
        refs = extract_temporal_refs("Let's meet at 3pm")
        assert len(refs) > 0

    def test_days_of_week(self) -> None:
        """Test day references."""
        refs = extract_temporal_refs("Are you free on Monday?")
        assert any("monday" in str(r).lower() for r in refs)

    def test_no_temporal_refs(self) -> None:
        """Test text without temporal references."""
        refs = extract_temporal_refs("Hello there")
        assert refs == []


class TestGetAttachmentToken:
    """Tests for attachment token generation."""

    def test_image_types(self) -> None:
        """Test image MIME types."""
        assert get_attachment_token("image/jpeg") == "<ATTACHMENT:image>"
        assert get_attachment_token("image/png") == "<ATTACHMENT:image>"
        assert get_attachment_token("jpg") == "<ATTACHMENT:image>"
        assert get_attachment_token("HEIC") == "<ATTACHMENT:image>"

    def test_video_types(self) -> None:
        """Test video MIME types."""
        assert get_attachment_token("video/mp4") == "<ATTACHMENT:video>"
        assert get_attachment_token("mov") == "<ATTACHMENT:video>"

    def test_audio_types(self) -> None:
        """Test audio MIME types."""
        assert get_attachment_token("audio/mp3") == "<ATTACHMENT:audio>"
        assert get_attachment_token("m4a") == "<ATTACHMENT:audio>"

    def test_pdf_type(self) -> None:
        """Test PDF type."""
        assert get_attachment_token("application/pdf") == "<ATTACHMENT:pdf>"
        assert get_attachment_token("pdf") == "<ATTACHMENT:pdf>"

    def test_generic_type(self) -> None:
        """Test generic/unknown types."""
        assert get_attachment_token("application/zip") == "<ATTACHMENT:file>"
        assert get_attachment_token(None) == "<ATTACHMENT:file>"
        assert get_attachment_token("") == "<ATTACHMENT:file>"


class TestExtractTextFeatures:
    """Tests for feature extraction."""

    def test_all_features(self) -> None:
        """Test extracting all features."""
        features = extract_text_features("btw, are you free tomorrow at 3pm?")

        assert features.is_question
        assert features.starts_new_topic
        assert features.word_count == 7
        assert len(features.temporal_refs) > 0

    def test_empty_input(self) -> None:
        """Test empty input."""
        features = extract_text_features("")
        assert features.word_count == 0
        assert not features.is_question
        assert not features.is_reaction

    def test_reaction_features(self) -> None:
        """Test reaction detection in features."""
        features = extract_text_features('Liked "message"')
        assert features.is_reaction

    def test_to_dict(self) -> None:
        """Test feature dict conversion."""
        features = extract_text_features("Hello?")
        d = features.to_dict()
        assert "is_question" in d
        assert d["is_question"] is True
        assert "word_count" in d


class TestTriggerExpectsContent:
    """Tests for trigger content expectation detection."""

    def test_questions(self) -> None:
        """Test that questions expect content."""
        assert trigger_expects_content("What do you think?")
        assert trigger_expects_content("Are you free tonight?")
        assert trigger_expects_content("How was your day?")

    def test_requests(self) -> None:
        """Test that requests expect content."""
        assert trigger_expects_content("Can you help me?")
        assert trigger_expects_content("Could you send that?")
        assert trigger_expects_content("Let me know what you think")
        assert trigger_expects_content("Tell me about it")

    def test_proposals(self) -> None:
        """Test that proposals expect content."""
        assert trigger_expects_content("How about dinner?")
        assert trigger_expects_content("What about tomorrow?")
        assert trigger_expects_content("Should we go?")
        assert trigger_expects_content("Let's meet up")

    def test_statements(self) -> None:
        """Test that simple statements don't expect content."""
        assert not trigger_expects_content("I'm here")
        assert not trigger_expects_content("Sounds good")
        assert not trigger_expects_content("ok")
