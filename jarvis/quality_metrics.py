"""Quality Metrics Dashboard for JARVIS.

Tracks and aggregates quality metrics for response generation performance:
- Template hit rate (% of queries matched by templates vs model generation)
- HHEM scores for model-generated responses (track over time)
- User acceptance rate (sent unchanged / total suggestions)
- Edit distance when users modify suggestions
- Response latency (template vs model)

Aggregations supported:
- By contact (who gets best/worst suggestions?)
- By time of day (are evening responses different?)
- By conversation type (1:1 vs group)
- By intent (reply vs summarize vs search)
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from jarvis.intent import IntentType


class ResponseSource(Enum):
    """Source of the generated response."""

    TEMPLATE = "template"
    MODEL = "model"


class ConversationType(Enum):
    """Type of conversation."""

    ONE_ON_ONE = "1:1"
    GROUP = "group"


class AcceptanceStatus(Enum):
    """Status of user acceptance of a suggestion."""

    ACCEPTED_UNCHANGED = "accepted_unchanged"
    ACCEPTED_MODIFIED = "accepted_modified"
    REJECTED = "rejected"


@dataclass
class ResponseEvent:
    """A single response generation event."""

    timestamp: datetime
    source: ResponseSource
    intent: IntentType
    contact_id: str
    conversation_type: ConversationType
    latency_ms: float
    hhem_score: float | None = None  # Only for model-generated responses
    acceptance_status: AcceptanceStatus | None = None
    edit_distance: int | None = None  # Levenshtein distance if modified
    hour_of_day: int = field(init=False)

    def __post_init__(self) -> None:
        """Compute derived fields."""
        self.hour_of_day = self.timestamp.hour


@dataclass
class QualitySnapshot:
    """Point-in-time quality metrics snapshot."""

    timestamp: datetime
    template_hit_rate: float
    model_hit_rate: float
    avg_hhem_score: float | None
    acceptance_rate: float
    avg_edit_distance: float | None
    avg_template_latency_ms: float
    avg_model_latency_ms: float
    total_responses: int


@dataclass
class ContactQuality:
    """Quality metrics for a specific contact."""

    contact_id: str
    total_responses: int
    template_responses: int
    model_responses: int
    acceptance_rate: float
    avg_hhem_score: float | None
    avg_edit_distance: float | None
    avg_latency_ms: float


@dataclass
class TimeOfDayQuality:
    """Quality metrics aggregated by hour of day."""

    hour: int
    total_responses: int
    template_hit_rate: float
    acceptance_rate: float
    avg_latency_ms: float


@dataclass
class IntentQuality:
    """Quality metrics aggregated by intent type."""

    intent: IntentType
    total_responses: int
    template_hit_rate: float
    acceptance_rate: float
    avg_hhem_score: float | None
    avg_latency_ms: float


@dataclass
class Recommendation:
    """Actionable recommendation for improving quality."""

    category: str
    priority: str  # "high", "medium", "low"
    title: str
    description: str
    metric_value: float | None = None
    target_value: float | None = None


class QualityMetricsCollector:
    """Thread-safe collector for quality metrics.

    Tracks response generation events and provides aggregated metrics
    for the quality dashboard.
    """

    MAX_EVENTS = 10000  # Maximum events to retain
    MAX_SNAPSHOTS = 1000  # Maximum daily snapshots to retain

    def __init__(self) -> None:
        """Initialize the quality metrics collector."""
        self._lock = threading.Lock()
        self._start_time = datetime.now(UTC)

        # Event storage
        self._events: list[ResponseEvent] = []

        # Daily snapshots for trends
        self._daily_snapshots: list[QualitySnapshot] = []
        self._last_snapshot_date: datetime | None = None

        # Counters for quick access
        self._template_count = 0
        self._model_count = 0
        self._accepted_unchanged_count = 0
        self._accepted_modified_count = 0
        self._rejected_count = 0

        # Running sums for averages
        self._total_hhem_scores = 0.0
        self._hhem_score_count = 0
        self._total_edit_distance = 0
        self._edit_distance_count = 0
        self._total_template_latency = 0.0
        self._template_latency_count = 0
        self._total_model_latency = 0.0
        self._model_latency_count = 0

    def record_response(
        self,
        source: ResponseSource,
        intent: IntentType,
        contact_id: str,
        conversation_type: ConversationType,
        latency_ms: float,
        hhem_score: float | None = None,
    ) -> None:
        """Record a response generation event.

        Args:
            source: Whether response came from template or model
            intent: The classified intent type
            contact_id: ID of the contact/conversation
            conversation_type: Whether 1:1 or group chat
            latency_ms: Response generation latency in milliseconds
            hhem_score: HHEM hallucination score (0-1, model only)
        """
        event = ResponseEvent(
            timestamp=datetime.now(UTC),
            source=source,
            intent=intent,
            contact_id=contact_id,
            conversation_type=conversation_type,
            latency_ms=latency_ms,
            hhem_score=hhem_score,
        )

        with self._lock:
            self._events.append(event)

            # Update counters
            if source == ResponseSource.TEMPLATE:
                self._template_count += 1
                self._total_template_latency += latency_ms
                self._template_latency_count += 1
            else:
                self._model_count += 1
                self._total_model_latency += latency_ms
                self._model_latency_count += 1

                if hhem_score is not None:
                    self._total_hhem_scores += hhem_score
                    self._hhem_score_count += 1

            # Trim events if at capacity
            if len(self._events) > self.MAX_EVENTS:
                self._events = self._events[-self.MAX_EVENTS :]

            # Create daily snapshot if needed
            self._maybe_create_snapshot()

    def record_acceptance(
        self,
        contact_id: str,
        status: AcceptanceStatus,
        edit_distance: int | None = None,
    ) -> None:
        """Record user acceptance of a suggestion.

        This should be called after a response is shown to the user
        and they take action on it.

        Args:
            contact_id: ID of the contact/conversation
            status: Whether the suggestion was accepted/modified/rejected
            edit_distance: Levenshtein distance if modified
        """
        with self._lock:
            # Update counters
            if status == AcceptanceStatus.ACCEPTED_UNCHANGED:
                self._accepted_unchanged_count += 1
            elif status == AcceptanceStatus.ACCEPTED_MODIFIED:
                self._accepted_modified_count += 1
                if edit_distance is not None:
                    self._total_edit_distance += edit_distance
                    self._edit_distance_count += 1
            else:
                self._rejected_count += 1

            # Update most recent event for this contact
            for event in reversed(self._events):
                if event.contact_id == contact_id and event.acceptance_status is None:
                    event.acceptance_status = status
                    event.edit_distance = edit_distance
                    break

    def _maybe_create_snapshot(self) -> None:
        """Create a daily snapshot if it's a new day.

        Called internally while holding the lock.
        """
        now = datetime.now(UTC)
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)

        if self._last_snapshot_date is None or self._last_snapshot_date < today:
            snapshot = self._create_snapshot_unlocked()
            self._daily_snapshots.append(snapshot)

            # Trim snapshots if at capacity
            if len(self._daily_snapshots) > self.MAX_SNAPSHOTS:
                self._daily_snapshots = self._daily_snapshots[-self.MAX_SNAPSHOTS :]

            self._last_snapshot_date = today

    def _create_snapshot_unlocked(self) -> QualitySnapshot:
        """Create a snapshot from current metrics.

        Must be called while holding the lock.
        """
        total = self._template_count + self._model_count
        total_accepted = (
            self._accepted_unchanged_count + self._accepted_modified_count + self._rejected_count
        )

        return QualitySnapshot(
            timestamp=datetime.now(UTC),
            template_hit_rate=(self._template_count / total * 100 if total > 0 else 0.0),
            model_hit_rate=self._model_count / total * 100 if total > 0 else 0.0,
            avg_hhem_score=(
                self._total_hhem_scores / self._hhem_score_count
                if self._hhem_score_count > 0
                else None
            ),
            acceptance_rate=(
                (self._accepted_unchanged_count + self._accepted_modified_count)
                / total_accepted
                * 100
                if total_accepted > 0
                else 0.0
            ),
            avg_edit_distance=(
                self._total_edit_distance / self._edit_distance_count
                if self._edit_distance_count > 0
                else None
            ),
            avg_template_latency_ms=(
                self._total_template_latency / self._template_latency_count
                if self._template_latency_count > 0
                else 0.0
            ),
            avg_model_latency_ms=(
                self._total_model_latency / self._model_latency_count
                if self._model_latency_count > 0
                else 0.0
            ),
            total_responses=total,
        )

    def get_summary(self) -> dict[str, Any]:
        """Get overall quality metrics summary.

        Returns:
            Dictionary with key quality metrics
        """
        with self._lock:
            total = self._template_count + self._model_count
            total_accepted = (
                self._accepted_unchanged_count
                + self._accepted_modified_count
                + self._rejected_count
            )
            elapsed = (datetime.now(UTC) - self._start_time).total_seconds()

            return {
                "total_responses": total,
                "template_responses": self._template_count,
                "model_responses": self._model_count,
                "template_hit_rate_percent": (
                    round(self._template_count / total * 100, 2) if total > 0 else 0.0
                ),
                "model_fallback_rate_percent": (
                    round(self._model_count / total * 100, 2) if total > 0 else 0.0
                ),
                "avg_hhem_score": (
                    round(self._total_hhem_scores / self._hhem_score_count, 4)
                    if self._hhem_score_count > 0
                    else None
                ),
                "hhem_score_count": self._hhem_score_count,
                "acceptance_rate_percent": (
                    round(
                        (self._accepted_unchanged_count + self._accepted_modified_count)
                        / total_accepted
                        * 100,
                        2,
                    )
                    if total_accepted > 0
                    else 0.0
                ),
                "accepted_unchanged_count": self._accepted_unchanged_count,
                "accepted_modified_count": self._accepted_modified_count,
                "rejected_count": self._rejected_count,
                "avg_edit_distance": (
                    round(self._total_edit_distance / self._edit_distance_count, 2)
                    if self._edit_distance_count > 0
                    else None
                ),
                "avg_template_latency_ms": (
                    round(self._total_template_latency / self._template_latency_count, 2)
                    if self._template_latency_count > 0
                    else 0.0
                ),
                "avg_model_latency_ms": (
                    round(self._total_model_latency / self._model_latency_count, 2)
                    if self._model_latency_count > 0
                    else 0.0
                ),
                "uptime_seconds": round(elapsed, 2),
                "responses_per_second": (round(total / elapsed, 4) if elapsed > 0 else 0.0),
            }

    def get_trends(self, days: int = 30) -> list[dict[str, Any]]:
        """Get quality metrics over time.

        Args:
            days: Number of days of history to return

        Returns:
            List of daily snapshots
        """
        cutoff = datetime.now(UTC) - timedelta(days=days)

        with self._lock:
            # Ensure we have a snapshot for today
            self._maybe_create_snapshot()

            trends = []
            for snapshot in self._daily_snapshots:
                if snapshot.timestamp >= cutoff:
                    trends.append(
                        {
                            "date": snapshot.timestamp.date().isoformat(),
                            "template_hit_rate_percent": round(snapshot.template_hit_rate, 2),
                            "model_hit_rate_percent": round(snapshot.model_hit_rate, 2),
                            "avg_hhem_score": (
                                round(snapshot.avg_hhem_score, 4)
                                if snapshot.avg_hhem_score is not None
                                else None
                            ),
                            "acceptance_rate_percent": round(snapshot.acceptance_rate, 2),
                            "avg_edit_distance": (
                                round(snapshot.avg_edit_distance, 2)
                                if snapshot.avg_edit_distance is not None
                                else None
                            ),
                            "avg_template_latency_ms": round(snapshot.avg_template_latency_ms, 2),
                            "avg_model_latency_ms": round(snapshot.avg_model_latency_ms, 2),
                            "total_responses": snapshot.total_responses,
                        }
                    )

            return trends

    def get_contact_quality(self, contact_id: str | None = None) -> list[ContactQuality]:
        """Get quality metrics aggregated by contact.

        Args:
            contact_id: Specific contact to get metrics for, or None for all

        Returns:
            List of ContactQuality objects
        """
        with self._lock:
            # Aggregate by contact
            contact_data: dict[str, dict[str, Any]] = defaultdict(
                lambda: {
                    "total": 0,
                    "template": 0,
                    "model": 0,
                    "accepted": 0,
                    "total_acceptance": 0,
                    "hhem_total": 0.0,
                    "hhem_count": 0,
                    "edit_total": 0,
                    "edit_count": 0,
                    "latency_total": 0.0,
                }
            )

            for event in self._events:
                if contact_id is not None and event.contact_id != contact_id:
                    continue

                data = contact_data[event.contact_id]
                data["total"] += 1
                data["latency_total"] += event.latency_ms

                if event.source == ResponseSource.TEMPLATE:
                    data["template"] += 1
                else:
                    data["model"] += 1
                    if event.hhem_score is not None:
                        data["hhem_total"] += event.hhem_score
                        data["hhem_count"] += 1

                if event.acceptance_status is not None:
                    data["total_acceptance"] += 1
                    if event.acceptance_status in (
                        AcceptanceStatus.ACCEPTED_UNCHANGED,
                        AcceptanceStatus.ACCEPTED_MODIFIED,
                    ):
                        data["accepted"] += 1

                    if event.edit_distance is not None:
                        data["edit_total"] += event.edit_distance
                        data["edit_count"] += 1

            # Build result list
            results = []
            for cid, data in contact_data.items():
                results.append(
                    ContactQuality(
                        contact_id=cid,
                        total_responses=data["total"],
                        template_responses=data["template"],
                        model_responses=data["model"],
                        acceptance_rate=(
                            data["accepted"] / data["total_acceptance"] * 100
                            if data["total_acceptance"] > 0
                            else 0.0
                        ),
                        avg_hhem_score=(
                            data["hhem_total"] / data["hhem_count"]
                            if data["hhem_count"] > 0
                            else None
                        ),
                        avg_edit_distance=(
                            data["edit_total"] / data["edit_count"]
                            if data["edit_count"] > 0
                            else None
                        ),
                        avg_latency_ms=(
                            data["latency_total"] / data["total"] if data["total"] > 0 else 0.0
                        ),
                    )
                )

            # Sort by total responses (most active first)
            results.sort(key=lambda x: x.total_responses, reverse=True)

            return results

    def get_time_of_day_quality(self) -> list[TimeOfDayQuality]:
        """Get quality metrics aggregated by hour of day.

        Returns:
            List of TimeOfDayQuality objects for each hour (0-23)
        """
        with self._lock:
            # Initialize all hours
            hour_data: dict[int, dict[str, Any]] = {
                h: {
                    "total": 0,
                    "template": 0,
                    "accepted": 0,
                    "total_acceptance": 0,
                    "latency_total": 0.0,
                }
                for h in range(24)
            }

            for event in self._events:
                data = hour_data[event.hour_of_day]
                data["total"] += 1
                data["latency_total"] += event.latency_ms

                if event.source == ResponseSource.TEMPLATE:
                    data["template"] += 1

                if event.acceptance_status is not None:
                    data["total_acceptance"] += 1
                    if event.acceptance_status in (
                        AcceptanceStatus.ACCEPTED_UNCHANGED,
                        AcceptanceStatus.ACCEPTED_MODIFIED,
                    ):
                        data["accepted"] += 1

            # Build result list
            results = []
            for hour in range(24):
                data = hour_data[hour]
                results.append(
                    TimeOfDayQuality(
                        hour=hour,
                        total_responses=data["total"],
                        template_hit_rate=(
                            data["template"] / data["total"] * 100 if data["total"] > 0 else 0.0
                        ),
                        acceptance_rate=(
                            data["accepted"] / data["total_acceptance"] * 100
                            if data["total_acceptance"] > 0
                            else 0.0
                        ),
                        avg_latency_ms=(
                            data["latency_total"] / data["total"] if data["total"] > 0 else 0.0
                        ),
                    )
                )

            return results

    def get_intent_quality(self) -> list[IntentQuality]:
        """Get quality metrics aggregated by intent type.

        Returns:
            List of IntentQuality objects for each intent
        """
        with self._lock:
            intent_data: dict[IntentType, dict[str, Any]] = defaultdict(
                lambda: {
                    "total": 0,
                    "template": 0,
                    "accepted": 0,
                    "total_acceptance": 0,
                    "hhem_total": 0.0,
                    "hhem_count": 0,
                    "latency_total": 0.0,
                }
            )

            for event in self._events:
                data = intent_data[event.intent]
                data["total"] += 1
                data["latency_total"] += event.latency_ms

                if event.source == ResponseSource.TEMPLATE:
                    data["template"] += 1
                else:
                    if event.hhem_score is not None:
                        data["hhem_total"] += event.hhem_score
                        data["hhem_count"] += 1

                if event.acceptance_status is not None:
                    data["total_acceptance"] += 1
                    if event.acceptance_status in (
                        AcceptanceStatus.ACCEPTED_UNCHANGED,
                        AcceptanceStatus.ACCEPTED_MODIFIED,
                    ):
                        data["accepted"] += 1

            # Build result list
            results = []
            for intent, data in intent_data.items():
                results.append(
                    IntentQuality(
                        intent=intent,
                        total_responses=data["total"],
                        template_hit_rate=(
                            data["template"] / data["total"] * 100 if data["total"] > 0 else 0.0
                        ),
                        acceptance_rate=(
                            data["accepted"] / data["total_acceptance"] * 100
                            if data["total_acceptance"] > 0
                            else 0.0
                        ),
                        avg_hhem_score=(
                            data["hhem_total"] / data["hhem_count"]
                            if data["hhem_count"] > 0
                            else None
                        ),
                        avg_latency_ms=(
                            data["latency_total"] / data["total"] if data["total"] > 0 else 0.0
                        ),
                    )
                )

            # Sort by total responses (most used first)
            results.sort(key=lambda x: x.total_responses, reverse=True)

            return results

    def get_conversation_type_quality(self) -> dict[str, dict[str, Any]]:
        """Get quality metrics aggregated by conversation type.

        Returns:
            Dictionary with metrics for 1:1 and group conversations
        """
        with self._lock:
            type_data: dict[ConversationType, dict[str, Any]] = {
                ConversationType.ONE_ON_ONE: {
                    "total": 0,
                    "template": 0,
                    "accepted": 0,
                    "total_acceptance": 0,
                    "hhem_total": 0.0,
                    "hhem_count": 0,
                    "latency_total": 0.0,
                },
                ConversationType.GROUP: {
                    "total": 0,
                    "template": 0,
                    "accepted": 0,
                    "total_acceptance": 0,
                    "hhem_total": 0.0,
                    "hhem_count": 0,
                    "latency_total": 0.0,
                },
            }

            for event in self._events:
                data = type_data[event.conversation_type]
                data["total"] += 1
                data["latency_total"] += event.latency_ms

                if event.source == ResponseSource.TEMPLATE:
                    data["template"] += 1
                else:
                    if event.hhem_score is not None:
                        data["hhem_total"] += event.hhem_score
                        data["hhem_count"] += 1

                if event.acceptance_status is not None:
                    data["total_acceptance"] += 1
                    if event.acceptance_status in (
                        AcceptanceStatus.ACCEPTED_UNCHANGED,
                        AcceptanceStatus.ACCEPTED_MODIFIED,
                    ):
                        data["accepted"] += 1

            result = {}
            for conv_type, data in type_data.items():
                result[conv_type.value] = {
                    "total_responses": data["total"],
                    "template_hit_rate_percent": (
                        round(data["template"] / data["total"] * 100, 2)
                        if data["total"] > 0
                        else 0.0
                    ),
                    "acceptance_rate_percent": (
                        round(data["accepted"] / data["total_acceptance"] * 100, 2)
                        if data["total_acceptance"] > 0
                        else 0.0
                    ),
                    "avg_hhem_score": (
                        round(data["hhem_total"] / data["hhem_count"], 4)
                        if data["hhem_count"] > 0
                        else None
                    ),
                    "avg_latency_ms": (
                        round(data["latency_total"] / data["total"], 2)
                        if data["total"] > 0
                        else 0.0
                    ),
                }

            return result

    def get_recommendations(self) -> list[Recommendation]:
        """Get actionable recommendations for improving quality.

        Analyzes current metrics and generates suggestions.

        Returns:
            List of Recommendation objects prioritized by importance
        """
        recommendations: list[Recommendation] = []

        with self._lock:
            total = self._template_count + self._model_count
            total_accepted = (
                self._accepted_unchanged_count
                + self._accepted_modified_count
                + self._rejected_count
            )

            # Check template hit rate
            if total > 10:  # Need enough data
                hit_rate = self._template_count / total * 100
                if hit_rate < 50:
                    recommendations.append(
                        Recommendation(
                            category="template_coverage",
                            priority="high",
                            title="Low Template Hit Rate",
                            description=(
                                f"Only {hit_rate:.1f}% of queries are matched by templates. "
                                "Consider adding more template patterns for common queries."
                            ),
                            metric_value=hit_rate,
                            target_value=70.0,
                        )
                    )
                elif hit_rate < 70:
                    recommendations.append(
                        Recommendation(
                            category="template_coverage",
                            priority="medium",
                            title="Template Hit Rate Could Be Improved",
                            description=(
                                f"Template hit rate is {hit_rate:.1f}%. "
                                "Review missed queries to identify new template opportunities."
                            ),
                            metric_value=hit_rate,
                            target_value=70.0,
                        )
                    )

            # Check HHEM scores
            if self._hhem_score_count > 5:
                avg_hhem = self._total_hhem_scores / self._hhem_score_count
                if avg_hhem < 0.4:
                    recommendations.append(
                        Recommendation(
                            category="hallucination",
                            priority="high",
                            title="High Hallucination Risk",
                            description=(
                                f"Average HHEM score is {avg_hhem:.3f}, indicating potential "
                                "hallucination issues. Consider improving prompts or "
                                "adding more context to queries."
                            ),
                            metric_value=avg_hhem,
                            target_value=0.5,
                        )
                    )
                elif avg_hhem < 0.5:
                    recommendations.append(
                        Recommendation(
                            category="hallucination",
                            priority="medium",
                            title="HHEM Score Below Target",
                            description=(
                                f"Average HHEM score is {avg_hhem:.3f}. "
                                "Review low-scoring responses to identify patterns."
                            ),
                            metric_value=avg_hhem,
                            target_value=0.5,
                        )
                    )

            # Check acceptance rate
            if total_accepted > 10:
                acceptance_rate = (
                    (self._accepted_unchanged_count + self._accepted_modified_count)
                    / total_accepted
                    * 100
                )
                if acceptance_rate < 50:
                    recommendations.append(
                        Recommendation(
                            category="user_acceptance",
                            priority="high",
                            title="Low User Acceptance Rate",
                            description=(
                                f"Only {acceptance_rate:.1f}% of suggestions are accepted. "
                                "Review rejected suggestions to understand user preferences."
                            ),
                            metric_value=acceptance_rate,
                            target_value=70.0,
                        )
                    )
                elif acceptance_rate < 70:
                    recommendations.append(
                        Recommendation(
                            category="user_acceptance",
                            priority="medium",
                            title="User Acceptance Rate Could Be Improved",
                            description=(
                                f"Acceptance rate is {acceptance_rate:.1f}%. "
                                "Analyze common edit patterns to improve suggestions."
                            ),
                            metric_value=acceptance_rate,
                            target_value=70.0,
                        )
                    )

            # Check edit distance
            if self._edit_distance_count > 5:
                avg_edit = self._total_edit_distance / self._edit_distance_count
                if avg_edit > 50:
                    recommendations.append(
                        Recommendation(
                            category="edit_distance",
                            priority="medium",
                            title="High Edit Distance",
                            description=(
                                f"Average edit distance is {avg_edit:.1f} characters. "
                                "Users are making significant modifications to suggestions."
                            ),
                            metric_value=avg_edit,
                            target_value=20.0,
                        )
                    )

            # Check latency
            if self._model_latency_count > 5:
                avg_model_latency = self._total_model_latency / self._model_latency_count
                if avg_model_latency > 5000:  # > 5 seconds
                    recommendations.append(
                        Recommendation(
                            category="latency",
                            priority="high",
                            title="High Model Response Latency",
                            description=(
                                f"Average model response time is {avg_model_latency:.0f}ms. "
                                "Consider using a smaller model or improving template coverage."
                            ),
                            metric_value=avg_model_latency,
                            target_value=3000.0,
                        )
                    )
                elif avg_model_latency > 3000:  # > 3 seconds
                    recommendations.append(
                        Recommendation(
                            category="latency",
                            priority="medium",
                            title="Model Response Latency Above Target",
                            description=(
                                f"Average model response time is {avg_model_latency:.0f}ms. "
                                "Monitor for user experience impact."
                            ),
                            metric_value=avg_model_latency,
                            target_value=3000.0,
                        )
                    )

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda r: priority_order.get(r.priority, 2))

        # If no issues, add a positive recommendation
        if not recommendations:
            recommendations.append(
                Recommendation(
                    category="overall",
                    priority="low",
                    title="Quality Metrics Look Good",
                    description=(
                        "All metrics are within acceptable ranges. "
                        "Continue monitoring for any regressions."
                    ),
                )
            )

        return recommendations

    def reset(self) -> None:
        """Reset all collected metrics."""
        with self._lock:
            self._start_time = datetime.now(UTC)
            self._events.clear()
            self._daily_snapshots.clear()
            self._last_snapshot_date = None

            self._template_count = 0
            self._model_count = 0
            self._accepted_unchanged_count = 0
            self._accepted_modified_count = 0
            self._rejected_count = 0

            self._total_hhem_scores = 0.0
            self._hhem_score_count = 0
            self._total_edit_distance = 0
            self._edit_distance_count = 0
            self._total_template_latency = 0.0
            self._template_latency_count = 0
            self._total_model_latency = 0.0
            self._model_latency_count = 0


# Global singleton instance
_quality_metrics: QualityMetricsCollector | None = None
_quality_lock = threading.Lock()


def get_quality_metrics() -> QualityMetricsCollector:
    """Get the global quality metrics collector instance.

    Returns:
        Shared QualityMetricsCollector instance
    """
    global _quality_metrics
    if _quality_metrics is None:
        with _quality_lock:
            if _quality_metrics is None:
                _quality_metrics = QualityMetricsCollector()
    return _quality_metrics


def reset_quality_metrics() -> None:
    """Reset the global quality metrics collector instance."""
    global _quality_metrics
    with _quality_lock:
        _quality_metrics = None


def compute_edit_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Minimum number of single-character edits to transform s1 into s2
    """
    if len(s1) < len(s2):
        return compute_edit_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # j+1 instead of j since previous_row and current_row are one character longer
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]
