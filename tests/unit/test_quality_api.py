"""Tests for the quality metrics API endpoints.

Tests the REST API for quality metrics dashboard.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from jarvis.intent import IntentType
from jarvis.quality_metrics import (
    ConversationType,
    ResponseSource,
    get_quality_metrics,
    reset_quality_metrics,
)


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_metrics():
    """Reset quality metrics before each test."""
    reset_quality_metrics()
    yield
    reset_quality_metrics()


class TestQualitySummaryEndpoint:
    """Tests for GET /quality/summary."""

    def test_get_summary_empty(self, client):
        """Get summary returns zeros when no data."""
        response = client.get("/quality/summary")

        assert response.status_code == 200
        data = response.json()

        assert data["total_responses"] == 0
        assert data["template_responses"] == 0
        assert data["model_responses"] == 0
        assert data["template_hit_rate_percent"] == 0.0

    def test_get_summary_with_data(self, client):
        """Get summary returns correct values."""
        metrics = get_quality_metrics()

        # Add some data
        metrics.record_response(
            source=ResponseSource.TEMPLATE,
            intent=IntentType.REPLY,
            contact_id="test_contact",
            conversation_type=ConversationType.ONE_ON_ONE,
            latency_ms=50.0,
        )

        response = client.get("/quality/summary")

        assert response.status_code == 200
        data = response.json()

        assert data["total_responses"] == 1
        assert data["template_responses"] == 1
        assert data["template_hit_rate_percent"] == 100.0


class TestQualityTrendsEndpoint:
    """Tests for GET /quality/trends."""

    def test_get_trends_default(self, client):
        """Get trends with default parameters."""
        response = client.get("/quality/trends")

        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, list)

    def test_get_trends_custom_days(self, client):
        """Get trends with custom days parameter."""
        response = client.get("/quality/trends?days=7")

        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, list)


class TestContactQualityEndpoint:
    """Tests for GET /quality/contact/{contact_id}."""

    def test_get_contact_quality_unknown(self, client):
        """Get quality for unknown contact returns zeros."""
        response = client.get("/quality/contact/unknown_contact")

        assert response.status_code == 200
        data = response.json()

        assert data["contact_id"] == "unknown_contact"
        assert data["total_responses"] == 0

    def test_get_contact_quality_with_data(self, client):
        """Get quality for known contact returns correct data."""
        metrics = get_quality_metrics()

        metrics.record_response(
            source=ResponseSource.TEMPLATE,
            intent=IntentType.REPLY,
            contact_id="test_contact",
            conversation_type=ConversationType.ONE_ON_ONE,
            latency_ms=50.0,
        )

        response = client.get("/quality/contact/test_contact")

        assert response.status_code == 200
        data = response.json()

        assert data["contact_id"] == "test_contact"
        assert data["total_responses"] == 1
        assert data["template_responses"] == 1


class TestAllContactsEndpoint:
    """Tests for GET /quality/contacts."""

    def test_get_all_contacts_empty(self, client):
        """Get all contacts when no data."""
        response = client.get("/quality/contacts")

        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, list)
        assert len(data) == 0

    def test_get_all_contacts_with_data(self, client):
        """Get all contacts returns sorted list."""
        metrics = get_quality_metrics()

        # Add data for multiple contacts
        for _ in range(3):
            metrics.record_response(
                source=ResponseSource.TEMPLATE,
                intent=IntentType.REPLY,
                contact_id="contact_a",
                conversation_type=ConversationType.ONE_ON_ONE,
                latency_ms=50.0,
            )

        for _ in range(2):
            metrics.record_response(
                source=ResponseSource.MODEL,
                intent=IntentType.REPLY,
                contact_id="contact_b",
                conversation_type=ConversationType.GROUP,
                latency_ms=500.0,
            )

        response = client.get("/quality/contacts")

        assert response.status_code == 200
        data = response.json()

        assert len(data) == 2
        # Should be sorted by total responses (descending)
        assert data[0]["contact_id"] == "contact_a"
        assert data[1]["contact_id"] == "contact_b"


class TestTimeOfDayEndpoint:
    """Tests for GET /quality/time-of-day."""

    def test_get_time_of_day(self, client):
        """Get time of day returns 24 hours."""
        response = client.get("/quality/time-of-day")

        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, list)
        assert len(data) == 24

        # Check hours are 0-23
        hours = [d["hour"] for d in data]
        assert hours == list(range(24))


class TestIntentQualityEndpoint:
    """Tests for GET /quality/by-intent."""

    def test_get_by_intent_empty(self, client):
        """Get intent quality when no data."""
        response = client.get("/quality/by-intent")

        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, list)
        assert len(data) == 0

    def test_get_by_intent_with_data(self, client):
        """Get intent quality returns aggregated data."""
        metrics = get_quality_metrics()

        metrics.record_response(
            source=ResponseSource.TEMPLATE,
            intent=IntentType.REPLY,
            contact_id="test_contact",
            conversation_type=ConversationType.ONE_ON_ONE,
            latency_ms=50.0,
        )

        metrics.record_response(
            source=ResponseSource.MODEL,
            intent=IntentType.SUMMARIZE,
            contact_id="test_contact",
            conversation_type=ConversationType.ONE_ON_ONE,
            latency_ms=500.0,
        )

        response = client.get("/quality/by-intent")

        assert response.status_code == 200
        data = response.json()

        assert len(data) == 2

        # Check intent values
        intents = [d["intent"] for d in data]
        assert "reply" in intents
        assert "summarize" in intents


class TestConversationTypeEndpoint:
    """Tests for GET /quality/by-conversation-type."""

    def test_get_by_conversation_type(self, client):
        """Get conversation type quality."""
        response = client.get("/quality/by-conversation-type")

        assert response.status_code == 200
        data = response.json()

        assert "1:1" in data
        assert "group" in data


class TestRecommendationsEndpoint:
    """Tests for GET /quality/recommendations."""

    def test_get_recommendations_empty(self, client):
        """Get recommendations when no data."""
        response = client.get("/quality/recommendations")

        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, list)
        # Should have "all good" recommendation
        assert len(data) >= 1

    def test_get_recommendations_with_issues(self, client):
        """Get recommendations when there are issues."""
        metrics = get_quality_metrics()

        # Create low template hit rate (10 model, 1 template)
        for _ in range(10):
            metrics.record_response(
                source=ResponseSource.MODEL,
                intent=IntentType.REPLY,
                contact_id="test_contact",
                conversation_type=ConversationType.ONE_ON_ONE,
                latency_ms=500.0,
            )

        metrics.record_response(
            source=ResponseSource.TEMPLATE,
            intent=IntentType.REPLY,
            contact_id="test_contact",
            conversation_type=ConversationType.ONE_ON_ONE,
            latency_ms=50.0,
        )

        response = client.get("/quality/recommendations")

        assert response.status_code == 200
        data = response.json()

        # Should have a template coverage recommendation
        categories = [r["category"] for r in data]
        assert "template_coverage" in categories


class TestDashboardEndpoint:
    """Tests for GET /quality/dashboard."""

    def test_get_dashboard(self, client):
        """Get dashboard returns all data."""
        response = client.get("/quality/dashboard")

        assert response.status_code == 200
        data = response.json()

        # Check all expected fields
        assert "summary" in data
        assert "trends" in data
        assert "top_contacts" in data
        assert "time_of_day" in data
        assert "by_intent" in data
        assert "by_conversation_type" in data
        assert "recommendations" in data

    def test_get_dashboard_custom_params(self, client):
        """Get dashboard with custom parameters."""
        response = client.get("/quality/dashboard?trend_days=7&top_contacts_limit=5")

        assert response.status_code == 200
        data = response.json()

        assert "summary" in data


class TestRecordResponseEndpoint:
    """Tests for POST /quality/record/response."""

    def test_record_template_response(self, client):
        """Record template response via API."""
        response = client.post(
            "/quality/record/response",
            json={
                "source": "template",
                "intent": "reply",
                "contact_id": "test_contact",
                "conversation_type": "1:1",
                "latency_ms": 50.0,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

        # Verify data was recorded
        summary_response = client.get("/quality/summary")
        summary = summary_response.json()
        assert summary["template_responses"] == 1

    def test_record_model_response(self, client):
        """Record model response with HHEM score."""
        response = client.post(
            "/quality/record/response",
            json={
                "source": "model",
                "intent": "summarize",
                "contact_id": "test_contact",
                "conversation_type": "group",
                "latency_ms": 500.0,
                "hhem_score": 0.75,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_record_response_invalid_source(self, client):
        """Record response with invalid source."""
        response = client.post(
            "/quality/record/response",
            json={
                "source": "invalid",
                "intent": "reply",
                "contact_id": "test_contact",
                "conversation_type": "1:1",
                "latency_ms": 50.0,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"


class TestRecordAcceptanceEndpoint:
    """Tests for POST /quality/record/acceptance."""

    def test_record_acceptance_unchanged(self, client):
        """Record acceptance unchanged via API."""
        response = client.post(
            "/quality/record/acceptance",
            json={
                "contact_id": "test_contact",
                "status": "accepted_unchanged",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_record_acceptance_modified(self, client):
        """Record acceptance modified with edit distance."""
        response = client.post(
            "/quality/record/acceptance",
            json={
                "contact_id": "test_contact",
                "status": "accepted_modified",
                "edit_distance": 15,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_record_acceptance_rejected(self, client):
        """Record rejection via API."""
        response = client.post(
            "/quality/record/acceptance",
            json={
                "contact_id": "test_contact",
                "status": "rejected",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


class TestResetEndpoint:
    """Tests for POST /quality/reset."""

    def test_reset_quality_metrics(self, client):
        """Reset quality metrics via API."""
        metrics = get_quality_metrics()

        # Add some data
        metrics.record_response(
            source=ResponseSource.TEMPLATE,
            intent=IntentType.REPLY,
            contact_id="test_contact",
            conversation_type=ConversationType.ONE_ON_ONE,
            latency_ms=50.0,
        )

        # Reset
        response = client.post("/quality/reset")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

        # Verify data was reset
        summary_response = client.get("/quality/summary")
        summary = summary_response.json()
        assert summary["total_responses"] == 0
