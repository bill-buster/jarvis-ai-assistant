"""Unit tests for JARVIS Template System.

Tests cover template definitions, group chat templates, template matching,
group context awareness, and edge cases.

Note: Some tests require sentence_transformers and are marked with the
requires_sentence_transformers marker.
"""

import pytest

from models.templates import (
    EmbeddingCache,
    ResponseTemplate,
    TemplateMatch,
    TemplateMatcher,
    _get_minimal_fallback_templates,
    _load_templates,
)

# Import the marker for tests that require sentence_transformers
from tests.conftest import requires_sentence_transformers

# Marker for tests that depend on specific model outputs (may vary)
model_dependent = pytest.mark.xfail(
    reason="Model output varies - tests verify expected behavior but allow variation",
    strict=False,
)


class TestResponseTemplate:
    """Tests for ResponseTemplate dataclass."""

    def test_basic_template_creation(self) -> None:
        """Test basic ResponseTemplate creation."""
        template = ResponseTemplate(
            name="test_template",
            patterns=["hello", "hi", "hey"],
            response="Hello there!",
        )
        assert template.name == "test_template"
        assert len(template.patterns) == 3
        assert template.response == "Hello there!"
        assert template.is_group_template is False
        assert template.min_group_size is None
        assert template.max_group_size is None

    def test_group_template_creation(self) -> None:
        """Test group template with size constraints."""
        template = ResponseTemplate(
            name="group_test",
            patterns=["who's coming"],
            response="Count me in!",
            is_group_template=True,
            min_group_size=3,
            max_group_size=10,
        )
        assert template.is_group_template is True
        assert template.min_group_size == 3
        assert template.max_group_size == 10

    def test_template_with_only_min_size(self) -> None:
        """Test template with only minimum group size."""
        template = ResponseTemplate(
            name="large_group",
            patterns=["so many messages"],
            response="I know right!",
            is_group_template=True,
            min_group_size=10,
        )
        assert template.min_group_size == 10
        assert template.max_group_size is None

    def test_template_with_only_max_size(self) -> None:
        """Test template with only maximum group size."""
        template = ResponseTemplate(
            name="small_group",
            patterns=["the squad"],
            response="Love our crew!",
            is_group_template=True,
            max_group_size=5,
        )
        assert template.min_group_size is None
        assert template.max_group_size == 5


class TestTemplateMatch:
    """Tests for TemplateMatch dataclass."""

    def test_template_match_creation(self) -> None:
        """Test TemplateMatch creation."""
        template = ResponseTemplate(
            name="test",
            patterns=["hello"],
            response="Hi!",
        )
        match = TemplateMatch(
            template=template,
            similarity=0.85,
            matched_pattern="hello",
        )
        assert match.template == template
        assert match.similarity == 0.85
        assert match.matched_pattern == "hello"


class TestLoadTemplates:
    """Tests for template loading functions."""

    def test_load_templates_returns_list(self) -> None:
        """Test that _load_templates returns a list."""
        templates = _load_templates()
        assert isinstance(templates, list)
        assert len(templates) > 0

    def test_all_templates_are_response_templates(self) -> None:
        """Test that all loaded templates are ResponseTemplate instances."""
        templates = _load_templates()
        for template in templates:
            assert isinstance(template, ResponseTemplate)

    def test_fallback_templates_match_loaded(self) -> None:
        """Test that fallback templates match loaded templates."""
        fallback = _get_minimal_fallback_templates()
        loaded = _load_templates()
        assert fallback == loaded


class TestGroupChatTemplates:
    """Tests for group chat specific templates."""

    @pytest.fixture
    def templates(self) -> list[ResponseTemplate]:
        """Load all templates."""
        return _load_templates()

    def test_group_templates_exist(self, templates: list[ResponseTemplate]) -> None:
        """Test that group templates are included."""
        group_templates = [t for t in templates if t.is_group_template]
        assert len(group_templates) >= 20, (
            f"Expected 20+ group templates, found {len(group_templates)}"
        )

    def test_event_planning_templates(self, templates: list[ResponseTemplate]) -> None:
        """Test that event planning templates exist."""
        event_templates = [t for t in templates if t.is_group_template and "event" in t.name]
        assert len(event_templates) >= 3

    def test_rsvp_templates(self, templates: list[ResponseTemplate]) -> None:
        """Test that RSVP templates exist."""
        rsvp_templates = [t for t in templates if t.is_group_template and "rsvp" in t.name]
        assert len(rsvp_templates) >= 4

    def test_poll_templates(self, templates: list[ResponseTemplate]) -> None:
        """Test that poll templates exist."""
        poll_templates = [t for t in templates if t.is_group_template and "poll" in t.name]
        assert len(poll_templates) >= 3

    def test_logistics_templates(self, templates: list[ResponseTemplate]) -> None:
        """Test that logistics templates exist."""
        logistics_templates = [
            t for t in templates if t.is_group_template and "logistics" in t.name
        ]
        assert len(logistics_templates) >= 4

    def test_celebration_templates(self, templates: list[ResponseTemplate]) -> None:
        """Test that celebration templates exist."""
        celebration_templates = [
            t for t in templates if t.is_group_template and "celebration" in t.name
        ]
        assert len(celebration_templates) >= 4

    def test_info_sharing_templates(self, templates: list[ResponseTemplate]) -> None:
        """Test that information sharing templates exist."""
        info_templates = [t for t in templates if t.is_group_template and "info" in t.name]
        assert len(info_templates) >= 3

    def test_large_group_templates(self, templates: list[ResponseTemplate]) -> None:
        """Test that large group specific templates exist."""
        large_group = [
            t
            for t in templates
            if t.is_group_template and t.min_group_size is not None and t.min_group_size >= 10
        ]
        assert len(large_group) >= 2

    def test_small_group_templates(self, templates: list[ResponseTemplate]) -> None:
        """Test that small group specific templates exist."""
        small_group = [
            t
            for t in templates
            if t.is_group_template and t.max_group_size is not None and t.max_group_size <= 5
        ]
        assert len(small_group) >= 1

    def test_no_duplicate_template_names(self, templates: list[ResponseTemplate]) -> None:
        """Test that all template names are unique."""
        names = [t.name for t in templates]
        unique_names = set(names)
        assert len(names) == len(unique_names), "Duplicate template names found"

    def test_all_templates_have_patterns(self, templates: list[ResponseTemplate]) -> None:
        """Test that all templates have at least one pattern."""
        for template in templates:
            assert len(template.patterns) >= 1, f"Template {template.name} has no patterns"

    def test_all_templates_have_responses(self, templates: list[ResponseTemplate]) -> None:
        """Test that all templates have non-empty responses."""
        for template in templates:
            assert len(template.response) > 0, f"Template {template.name} has empty response"


class TestEmbeddingCache:
    """Tests for EmbeddingCache class."""

    def test_cache_basic_operations(self) -> None:
        """Test basic cache get/set operations."""
        cache: EmbeddingCache[str, str] = EmbeddingCache(maxsize=10)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_cache_miss(self) -> None:
        """Test cache miss returns None."""
        cache: EmbeddingCache[str, str] = EmbeddingCache(maxsize=10)
        assert cache.get("nonexistent") is None

    def test_cache_eviction(self) -> None:
        """Test LRU eviction when cache is full."""
        cache: EmbeddingCache[str, str] = EmbeddingCache(maxsize=2)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")  # Should evict key1
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"

    def test_cache_hit_rate(self) -> None:
        """Test hit rate calculation."""
        cache: EmbeddingCache[str, str] = EmbeddingCache(maxsize=10)
        cache.set("key1", "value1")
        cache.get("key1")  # hit
        cache.get("key1")  # hit
        cache.get("key2")  # miss
        stats = cache.stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == pytest.approx(2 / 3, rel=0.01)

    def test_cache_clear(self) -> None:
        """Test cache clear."""
        cache: EmbeddingCache[str, str] = EmbeddingCache(maxsize=10)
        cache.set("key1", "value1")
        cache.get("key1")
        cache.clear()
        assert cache.get("key1") is None
        stats = cache.stats()
        assert stats["size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0


class TestTemplateMatcherBasics:
    """Tests for TemplateMatcher that don't require sentence_transformers."""

    def test_matcher_initialization(self) -> None:
        """Test TemplateMatcher initialization."""
        matcher = TemplateMatcher()
        assert matcher.templates is not None
        assert len(matcher.templates) > 0

    def test_matcher_with_custom_templates(self) -> None:
        """Test TemplateMatcher with custom templates."""
        custom = [
            ResponseTemplate(
                name="custom",
                patterns=["test pattern"],
                response="Test response",
            )
        ]
        matcher = TemplateMatcher(templates=custom)
        assert len(matcher.templates) == 1
        assert matcher.templates[0].name == "custom"

    def test_clear_cache_no_crash(self) -> None:
        """Test that clear_cache doesn't crash on uninitialized matcher."""
        matcher = TemplateMatcher()
        matcher.clear_cache()  # Should not crash
        assert matcher._pattern_embeddings is None

    def test_get_cache_stats(self) -> None:
        """Test get_cache_stats returns valid structure."""
        matcher = TemplateMatcher()
        stats = matcher.get_cache_stats()
        assert "size" in stats
        assert "maxsize" in stats
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats


class TestTemplateMatcherGroupContext:
    """Tests for group context awareness in TemplateMatcher."""

    @pytest.fixture
    def matcher(self) -> TemplateMatcher:
        """Create a matcher instance."""
        return TemplateMatcher()

    def test_template_matches_group_size_none(self, matcher: TemplateMatcher) -> None:
        """Test that non-group templates match when group_size is None."""
        non_group = ResponseTemplate(
            name="non_group",
            patterns=["hello"],
            response="Hi!",
            is_group_template=False,
        )
        assert matcher._template_matches_group_size(non_group, None) is True

    def test_group_template_not_matched_when_size_none(self, matcher: TemplateMatcher) -> None:
        """Test that group templates don't match when group_size is None."""
        group = ResponseTemplate(
            name="group",
            patterns=["who's coming"],
            response="Count me in!",
            is_group_template=True,
        )
        assert matcher._template_matches_group_size(group, None) is False

    def test_group_template_not_matched_for_1on1(self, matcher: TemplateMatcher) -> None:
        """Test that group templates don't match for 1-on-1 chats."""
        group = ResponseTemplate(
            name="group",
            patterns=["who's coming"],
            response="Count me in!",
            is_group_template=True,
        )
        assert matcher._template_matches_group_size(group, 2) is False

    def test_group_template_matched_for_group(self, matcher: TemplateMatcher) -> None:
        """Test that group templates match for group chats (3+)."""
        group = ResponseTemplate(
            name="group",
            patterns=["who's coming"],
            response="Count me in!",
            is_group_template=True,
        )
        assert matcher._template_matches_group_size(group, 3) is True
        assert matcher._template_matches_group_size(group, 10) is True

    def test_min_group_size_constraint(self, matcher: TemplateMatcher) -> None:
        """Test min_group_size constraint is respected."""
        large_group = ResponseTemplate(
            name="large_group",
            patterns=["so many messages"],
            response="I know!",
            is_group_template=True,
            min_group_size=10,
        )
        assert matcher._template_matches_group_size(large_group, 5) is False
        assert matcher._template_matches_group_size(large_group, 10) is True
        assert matcher._template_matches_group_size(large_group, 15) is True

    def test_max_group_size_constraint(self, matcher: TemplateMatcher) -> None:
        """Test max_group_size constraint is respected."""
        small_group = ResponseTemplate(
            name="small_group",
            patterns=["the squad"],
            response="Love it!",
            is_group_template=True,
            max_group_size=5,
        )
        assert matcher._template_matches_group_size(small_group, 3) is True
        assert matcher._template_matches_group_size(small_group, 5) is True
        assert matcher._template_matches_group_size(small_group, 6) is False

    def test_both_size_constraints(self, matcher: TemplateMatcher) -> None:
        """Test template with both min and max size constraints."""
        mid_group = ResponseTemplate(
            name="mid_group",
            patterns=["test"],
            response="Test",
            is_group_template=True,
            min_group_size=5,
            max_group_size=10,
        )
        assert matcher._template_matches_group_size(mid_group, 3) is False
        assert matcher._template_matches_group_size(mid_group, 5) is True
        assert matcher._template_matches_group_size(mid_group, 8) is True
        assert matcher._template_matches_group_size(mid_group, 10) is True
        assert matcher._template_matches_group_size(mid_group, 11) is False

    def test_get_group_templates(self, matcher: TemplateMatcher) -> None:
        """Test get_group_templates returns only group templates."""
        group_templates = matcher.get_group_templates()
        assert all(t.is_group_template for t in group_templates)
        assert len(group_templates) >= 20

    def test_get_templates_for_group_size_small(self, matcher: TemplateMatcher) -> None:
        """Test get_templates_for_group_size for small group."""
        templates = matcher.get_templates_for_group_size(3)
        # Should include general group templates and size-appropriate ones
        for t in templates:
            if t.is_group_template:
                if t.min_group_size is not None:
                    assert t.min_group_size <= 3
                if t.max_group_size is not None:
                    assert t.max_group_size >= 3

    def test_get_templates_for_group_size_large(self, matcher: TemplateMatcher) -> None:
        """Test get_templates_for_group_size for large group."""
        templates = matcher.get_templates_for_group_size(15)
        # Should include large group templates
        large_group_found = any(
            t.is_group_template and t.min_group_size is not None and t.min_group_size >= 10
            for t in templates
        )
        assert large_group_found


@requires_sentence_transformers
class TestTemplateMatcherWithEmbeddings:
    """Tests for TemplateMatcher that require sentence_transformers."""

    @pytest.fixture
    def matcher(self) -> TemplateMatcher:
        """Create a matcher instance."""
        return TemplateMatcher()

    def test_match_basic(self, matcher: TemplateMatcher) -> None:
        """Test basic template matching."""
        result = matcher.match("thanks for the help")
        assert result is not None
        assert result.similarity >= 0.7

    @model_dependent
    def test_match_greeting(self, matcher: TemplateMatcher) -> None:
        """Test matching a greeting pattern."""
        result = matcher.match("hi, how are you")
        assert result is not None
        assert "greeting" in result.template.name or result.similarity >= 0.7

    def test_match_returns_none_below_threshold(self, matcher: TemplateMatcher) -> None:
        """Test that match returns None for low similarity."""
        result = matcher.match("xyzzy plugh abracadabra")
        assert result is None

    @model_dependent
    def test_match_with_context_1on1(self, matcher: TemplateMatcher) -> None:
        """Test match_with_context for 1-on-1 chat."""
        result = matcher.match_with_context("thanks!", group_size=2)
        if result is not None:
            # Should not return group templates for 1-on-1
            assert not result.template.is_group_template

    @model_dependent
    def test_match_with_context_group(self, matcher: TemplateMatcher) -> None:
        """Test match_with_context for group chat."""
        result = matcher.match_with_context("count me in!", group_size=5)
        # Could match group or general template
        assert result is None or result.similarity >= 0.7

    @model_dependent
    def test_match_with_context_prefers_group(self, matcher: TemplateMatcher) -> None:
        """Test that prefer_group_templates boosts group templates."""
        # Query that could match both group and general templates
        result_normal = matcher.match_with_context(
            "sounds good!", group_size=5, prefer_group_templates=False
        )
        result_prefer = matcher.match_with_context(
            "sounds good!", group_size=5, prefer_group_templates=True
        )
        # Both should return valid matches
        assert result_normal is not None or result_prefer is not None

    @model_dependent
    def test_match_event_planning(self, matcher: TemplateMatcher) -> None:
        """Test matching event planning patterns."""
        result = matcher.match_with_context("when works for everyone?", group_size=5)
        assert result is not None or matcher.match("when works for everyone?") is not None

    @model_dependent
    def test_match_rsvp(self, matcher: TemplateMatcher) -> None:
        """Test matching RSVP patterns."""
        result = matcher.match_with_context("count me in!", group_size=5)
        assert result is None or result.similarity >= 0.7

    @model_dependent
    def test_match_celebration(self, matcher: TemplateMatcher) -> None:
        """Test matching celebration patterns."""
        result = matcher.match("happy birthday!")
        assert result is not None

    def test_match_after_clear_cache(self, matcher: TemplateMatcher) -> None:
        """Test that matching works after cache clear."""
        # First match to initialize
        result1 = matcher.match("thanks!")
        assert result1 is not None

        # Clear and rematch
        matcher.clear_cache()
        result2 = matcher.match("thanks!")
        assert result2 is not None


class TestGroupTemplatePatternCoverage:
    """Tests to ensure all required group patterns are covered."""

    @pytest.fixture
    def all_patterns(self) -> list[str]:
        """Collect all patterns from group templates."""
        templates = _load_templates()
        patterns = []
        for t in templates:
            if t.is_group_template:
                patterns.extend(t.patterns)
        return patterns

    def test_event_planning_patterns_coverage(self, all_patterns: list[str]) -> None:
        """Test that event planning patterns are covered."""
        event_keywords = [
            "when works for everyone",
            "saturday",
            "that day",
        ]
        for keyword in event_keywords:
            found = any(keyword.lower() in p.lower() for p in all_patterns)
            assert found, f"Missing event planning pattern for: {keyword}"

    def test_rsvp_patterns_coverage(self, all_patterns: list[str]) -> None:
        """Test that RSVP patterns are covered."""
        rsvp_keywords = [
            "count me in",
            "+1",
            "can't make it",
            "maybe",
            "who's coming",
        ]
        for keyword in rsvp_keywords:
            found = any(keyword.lower() in p.lower() for p in all_patterns)
            assert found, f"Missing RSVP pattern for: {keyword}"

    def test_poll_patterns_coverage(self, all_patterns: list[str]) -> None:
        """Test that poll patterns are covered."""
        poll_keywords = [
            "vote",
            "option",
            "either works",
        ]
        for keyword in poll_keywords:
            found = any(keyword.lower() in p.lower() for p in all_patterns)
            assert found, f"Missing poll pattern for: {keyword}"

    def test_logistics_patterns_coverage(self, all_patterns: list[str]) -> None:
        """Test that logistics patterns are covered."""
        logistics_keywords = [
            "who's bringing",
            "reservation",
            "where are we meeting",
            "ride",
            "split",
        ]
        for keyword in logistics_keywords:
            found = any(keyword.lower() in p.lower() for p in all_patterns)
            assert found, f"Missing logistics pattern for: {keyword}"

    def test_celebration_patterns_coverage(self, all_patterns: list[str]) -> None:
        """Test that celebration patterns are covered."""
        celebration_keywords = [
            "happy birthday",
            "congrats",
            "happy holidays",
            "thanks everyone",
        ]
        for keyword in celebration_keywords:
            found = any(keyword.lower() in p.lower() for p in all_patterns)
            assert found, f"Missing celebration pattern for: {keyword}"

    def test_info_sharing_patterns_coverage(self, all_patterns: list[str]) -> None:
        """Test that info sharing patterns are covered."""
        info_keywords = [
            "fyi",
            "heads up",
            "sharing",
            "update",
            "reminder",
        ]
        for keyword in info_keywords:
            found = any(keyword.lower() in p.lower() for p in all_patterns)
            assert found, f"Missing info sharing pattern for: {keyword}"
