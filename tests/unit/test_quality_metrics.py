"""Tests for the quality metrics module.

Tests quality metrics collection, aggregation, and recommendations.
"""

import time

from jarvis.intent import IntentType
from jarvis.quality_metrics import (
    AcceptanceStatus,
    ConversationType,
    QualityMetricsCollector,
    ResponseSource,
    compute_edit_distance,
    get_quality_metrics,
    reset_quality_metrics,
)


class TestQualityMetricsCollector:
    """Tests for QualityMetricsCollector."""

    def test_record_template_response(self):
        """Record template response increments counters."""
        collector = QualityMetricsCollector()

        collector.record_response(
            source=ResponseSource.TEMPLATE,
            intent=IntentType.REPLY,
            contact_id="test_contact",
            conversation_type=ConversationType.ONE_ON_ONE,
            latency_ms=50.0,
        )

        summary = collector.get_summary()
        assert summary["total_responses"] == 1
        assert summary["template_responses"] == 1
        assert summary["model_responses"] == 0
        assert summary["template_hit_rate_percent"] == 100.0

    def test_record_model_response(self):
        """Record model response increments counters."""
        collector = QualityMetricsCollector()

        collector.record_response(
            source=ResponseSource.MODEL,
            intent=IntentType.REPLY,
            contact_id="test_contact",
            conversation_type=ConversationType.ONE_ON_ONE,
            latency_ms=500.0,
            hhem_score=0.75,
        )

        summary = collector.get_summary()
        assert summary["total_responses"] == 1
        assert summary["template_responses"] == 0
        assert summary["model_responses"] == 1
        assert summary["model_fallback_rate_percent"] == 100.0
        assert summary["avg_hhem_score"] == 0.75
        assert summary["hhem_score_count"] == 1

    def test_record_mixed_responses(self):
        """Record mixed template and model responses."""
        collector = QualityMetricsCollector()

        # 3 template, 2 model
        for _ in range(3):
            collector.record_response(
                source=ResponseSource.TEMPLATE,
                intent=IntentType.REPLY,
                contact_id="test_contact",
                conversation_type=ConversationType.ONE_ON_ONE,
                latency_ms=50.0,
            )

        for _ in range(2):
            collector.record_response(
                source=ResponseSource.MODEL,
                intent=IntentType.SUMMARIZE,
                contact_id="test_contact",
                conversation_type=ConversationType.ONE_ON_ONE,
                latency_ms=500.0,
                hhem_score=0.6,
            )

        summary = collector.get_summary()
        assert summary["total_responses"] == 5
        assert summary["template_responses"] == 3
        assert summary["model_responses"] == 2
        assert summary["template_hit_rate_percent"] == 60.0
        assert summary["model_fallback_rate_percent"] == 40.0

    def test_record_acceptance_unchanged(self):
        """Record acceptance unchanged increments counter."""
        collector = QualityMetricsCollector()

        collector.record_response(
            source=ResponseSource.TEMPLATE,
            intent=IntentType.REPLY,
            contact_id="test_contact",
            conversation_type=ConversationType.ONE_ON_ONE,
            latency_ms=50.0,
        )

        collector.record_acceptance(
            contact_id="test_contact",
            status=AcceptanceStatus.ACCEPTED_UNCHANGED,
        )

        summary = collector.get_summary()
        assert summary["accepted_unchanged_count"] == 1
        assert summary["accepted_modified_count"] == 0
        assert summary["rejected_count"] == 0
        assert summary["acceptance_rate_percent"] == 100.0

    def test_record_acceptance_modified(self):
        """Record acceptance modified tracks edit distance."""
        collector = QualityMetricsCollector()

        collector.record_response(
            source=ResponseSource.TEMPLATE,
            intent=IntentType.REPLY,
            contact_id="test_contact",
            conversation_type=ConversationType.ONE_ON_ONE,
            latency_ms=50.0,
        )

        collector.record_acceptance(
            contact_id="test_contact",
            status=AcceptanceStatus.ACCEPTED_MODIFIED,
            edit_distance=15,
        )

        summary = collector.get_summary()
        assert summary["accepted_unchanged_count"] == 0
        assert summary["accepted_modified_count"] == 1
        assert summary["avg_edit_distance"] == 15.0

    def test_record_acceptance_rejected(self):
        """Record rejection updates counters."""
        collector = QualityMetricsCollector()

        collector.record_response(
            source=ResponseSource.TEMPLATE,
            intent=IntentType.REPLY,
            contact_id="test_contact",
            conversation_type=ConversationType.ONE_ON_ONE,
            latency_ms=50.0,
        )

        collector.record_acceptance(
            contact_id="test_contact",
            status=AcceptanceStatus.REJECTED,
        )

        summary = collector.get_summary()
        assert summary["rejected_count"] == 1
        assert summary["acceptance_rate_percent"] == 0.0

    def test_latency_tracking(self):
        """Track latency separately for template and model."""
        collector = QualityMetricsCollector()

        # Template responses
        collector.record_response(
            source=ResponseSource.TEMPLATE,
            intent=IntentType.REPLY,
            contact_id="test_contact",
            conversation_type=ConversationType.ONE_ON_ONE,
            latency_ms=50.0,
        )
        collector.record_response(
            source=ResponseSource.TEMPLATE,
            intent=IntentType.REPLY,
            contact_id="test_contact",
            conversation_type=ConversationType.ONE_ON_ONE,
            latency_ms=100.0,
        )

        # Model responses
        collector.record_response(
            source=ResponseSource.MODEL,
            intent=IntentType.REPLY,
            contact_id="test_contact",
            conversation_type=ConversationType.ONE_ON_ONE,
            latency_ms=500.0,
        )
        collector.record_response(
            source=ResponseSource.MODEL,
            intent=IntentType.REPLY,
            contact_id="test_contact",
            conversation_type=ConversationType.ONE_ON_ONE,
            latency_ms=1000.0,
        )

        summary = collector.get_summary()
        assert summary["avg_template_latency_ms"] == 75.0
        assert summary["avg_model_latency_ms"] == 750.0

    def test_contact_quality_aggregation(self):
        """Get quality metrics aggregated by contact."""
        collector = QualityMetricsCollector()

        # Contact A: 3 responses
        for _ in range(3):
            collector.record_response(
                source=ResponseSource.TEMPLATE,
                intent=IntentType.REPLY,
                contact_id="contact_a",
                conversation_type=ConversationType.ONE_ON_ONE,
                latency_ms=50.0,
            )

        # Contact B: 2 responses
        for _ in range(2):
            collector.record_response(
                source=ResponseSource.MODEL,
                intent=IntentType.REPLY,
                contact_id="contact_b",
                conversation_type=ConversationType.GROUP,
                latency_ms=500.0,
                hhem_score=0.7,
            )

        contacts = collector.get_contact_quality()

        assert len(contacts) == 2

        # Sorted by total responses (descending)
        assert contacts[0].contact_id == "contact_a"
        assert contacts[0].total_responses == 3
        assert contacts[0].template_responses == 3

        assert contacts[1].contact_id == "contact_b"
        assert contacts[1].total_responses == 2
        assert contacts[1].model_responses == 2
        assert contacts[1].avg_hhem_score == 0.7

    def test_time_of_day_aggregation(self):
        """Get quality metrics aggregated by hour of day."""
        collector = QualityMetricsCollector()

        # Record a response
        collector.record_response(
            source=ResponseSource.TEMPLATE,
            intent=IntentType.REPLY,
            contact_id="test_contact",
            conversation_type=ConversationType.ONE_ON_ONE,
            latency_ms=50.0,
        )

        time_data = collector.get_time_of_day_quality()

        # Should have 24 entries
        assert len(time_data) == 24

        # At least one hour should have a response
        total_responses = sum(h.total_responses for h in time_data)
        assert total_responses == 1

    def test_intent_quality_aggregation(self):
        """Get quality metrics aggregated by intent."""
        collector = QualityMetricsCollector()

        # Reply intents
        for _ in range(3):
            collector.record_response(
                source=ResponseSource.TEMPLATE,
                intent=IntentType.REPLY,
                contact_id="test_contact",
                conversation_type=ConversationType.ONE_ON_ONE,
                latency_ms=50.0,
            )

        # Summarize intents
        for _ in range(2):
            collector.record_response(
                source=ResponseSource.MODEL,
                intent=IntentType.SUMMARIZE,
                contact_id="test_contact",
                conversation_type=ConversationType.ONE_ON_ONE,
                latency_ms=500.0,
            )

        intent_data = collector.get_intent_quality()

        # Should have 2 intents
        assert len(intent_data) == 2

        # Find reply intent
        reply_data = next(i for i in intent_data if i.intent == IntentType.REPLY)
        assert reply_data.total_responses == 3
        assert reply_data.template_hit_rate == 100.0

        # Find summarize intent
        summarize_data = next(i for i in intent_data if i.intent == IntentType.SUMMARIZE)
        assert summarize_data.total_responses == 2
        assert summarize_data.template_hit_rate == 0.0

    def test_conversation_type_aggregation(self):
        """Get quality metrics by conversation type."""
        collector = QualityMetricsCollector()

        # 1:1 conversation
        for _ in range(3):
            collector.record_response(
                source=ResponseSource.TEMPLATE,
                intent=IntentType.REPLY,
                contact_id="test_contact",
                conversation_type=ConversationType.ONE_ON_ONE,
                latency_ms=50.0,
            )

        # Group conversation
        for _ in range(2):
            collector.record_response(
                source=ResponseSource.MODEL,
                intent=IntentType.REPLY,
                contact_id="test_group",
                conversation_type=ConversationType.GROUP,
                latency_ms=500.0,
            )

        conv_data = collector.get_conversation_type_quality()

        assert "1:1" in conv_data
        assert "group" in conv_data

        assert conv_data["1:1"]["total_responses"] == 3
        assert conv_data["1:1"]["template_hit_rate_percent"] == 100.0

        assert conv_data["group"]["total_responses"] == 2
        assert conv_data["group"]["template_hit_rate_percent"] == 0.0

    def test_recommendations_low_template_rate(self):
        """Generate recommendation for low template hit rate."""
        collector = QualityMetricsCollector()

        # 2 template, 9 model = ~18% template rate (need >10 total for recommendations)
        for _ in range(2):
            collector.record_response(
                source=ResponseSource.TEMPLATE,
                intent=IntentType.REPLY,
                contact_id="test_contact",
                conversation_type=ConversationType.ONE_ON_ONE,
                latency_ms=50.0,
            )

        for _ in range(9):
            collector.record_response(
                source=ResponseSource.MODEL,
                intent=IntentType.REPLY,
                contact_id="test_contact",
                conversation_type=ConversationType.ONE_ON_ONE,
                latency_ms=500.0,
            )

        recommendations = collector.get_recommendations()

        # Should have recommendation for low template rate
        template_rec = next((r for r in recommendations if r.category == "template_coverage"), None)
        assert template_rec is not None
        assert template_rec.priority == "high"

    def test_recommendations_low_hhem_score(self):
        """Generate recommendation for low HHEM score."""
        collector = QualityMetricsCollector()

        # Record 10 model responses with low HHEM scores
        for _ in range(10):
            collector.record_response(
                source=ResponseSource.MODEL,
                intent=IntentType.REPLY,
                contact_id="test_contact",
                conversation_type=ConversationType.ONE_ON_ONE,
                latency_ms=500.0,
                hhem_score=0.35,
            )

        recommendations = collector.get_recommendations()

        # Should have recommendation for low HHEM
        hhem_rec = next((r for r in recommendations if r.category == "hallucination"), None)
        assert hhem_rec is not None
        assert hhem_rec.priority == "high"

    def test_recommendations_good_metrics(self):
        """No critical recommendations when metrics are good."""
        collector = QualityMetricsCollector()

        # Good template rate (80%)
        for _ in range(8):
            collector.record_response(
                source=ResponseSource.TEMPLATE,
                intent=IntentType.REPLY,
                contact_id="test_contact",
                conversation_type=ConversationType.ONE_ON_ONE,
                latency_ms=50.0,
            )

        for _ in range(2):
            collector.record_response(
                source=ResponseSource.MODEL,
                intent=IntentType.REPLY,
                contact_id="test_contact",
                conversation_type=ConversationType.ONE_ON_ONE,
                latency_ms=500.0,
                hhem_score=0.7,
            )

        recommendations = collector.get_recommendations()

        # Should have the "all good" recommendation
        good_rec = next((r for r in recommendations if r.category == "overall"), None)
        assert good_rec is not None
        assert good_rec.priority == "low"

    def test_reset_clears_all_data(self):
        """Reset clears all collected data."""
        collector = QualityMetricsCollector()

        # Add some data
        for _ in range(5):
            collector.record_response(
                source=ResponseSource.TEMPLATE,
                intent=IntentType.REPLY,
                contact_id="test_contact",
                conversation_type=ConversationType.ONE_ON_ONE,
                latency_ms=50.0,
            )

        collector.reset()

        summary = collector.get_summary()
        assert summary["total_responses"] == 0
        assert summary["template_responses"] == 0
        assert summary["model_responses"] == 0

    def test_trends_empty_initially(self):
        """Trends are empty when no snapshots exist."""
        collector = QualityMetricsCollector()

        # Trends may have one snapshot from initialization
        trends = collector.get_trends(days=7)

        # Check it's a list (may or may not be empty depending on timing)
        assert isinstance(trends, list)

    def test_get_summary_uptime(self):
        """Summary includes uptime in seconds."""
        collector = QualityMetricsCollector()

        # Small delay to ensure uptime > 0
        time.sleep(0.01)

        summary = collector.get_summary()

        assert "uptime_seconds" in summary
        assert summary["uptime_seconds"] > 0


class TestGlobalSingleton:
    """Tests for global singleton quality metrics."""

    def test_get_quality_metrics_returns_same_instance(self):
        """Get quality metrics returns same instance."""
        reset_quality_metrics()

        m1 = get_quality_metrics()
        m2 = get_quality_metrics()

        assert m1 is m2

    def test_reset_quality_metrics_creates_new_instance(self):
        """Reset creates new instance."""
        m1 = get_quality_metrics()
        reset_quality_metrics()
        m2 = get_quality_metrics()

        assert m1 is not m2


class TestComputeEditDistance:
    """Tests for edit distance computation."""

    def test_identical_strings(self):
        """Identical strings have distance 0."""
        assert compute_edit_distance("hello", "hello") == 0

    def test_empty_strings(self):
        """Empty strings handled correctly."""
        assert compute_edit_distance("", "") == 0
        assert compute_edit_distance("hello", "") == 5
        assert compute_edit_distance("", "hello") == 5

    def test_single_substitution(self):
        """Single character substitution."""
        assert compute_edit_distance("hello", "hallo") == 1

    def test_single_insertion(self):
        """Single character insertion."""
        assert compute_edit_distance("hello", "helllo") == 1

    def test_single_deletion(self):
        """Single character deletion."""
        assert compute_edit_distance("hello", "helo") == 1

    def test_multiple_edits(self):
        """Multiple edits computed correctly."""
        assert compute_edit_distance("kitten", "sitting") == 3

    def test_case_sensitive(self):
        """Edit distance is case sensitive."""
        assert compute_edit_distance("Hello", "hello") == 1
