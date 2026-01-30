"""Calendar integration for JARVIS v2.

Provides event detection from messages and macOS Calendar integration.

Usage:
    from core.calendar import (
        get_event_detector,
        get_calendar_reader,
        check_for_conflicts,
    )

    # Detect events in message text
    detector = get_event_detector()
    events = detector.detect_events("Let's meet tomorrow at 3pm")

    # Read calendar events
    reader = get_calendar_reader()
    if reader.check_access():
        calendars = reader.get_calendars()
        events = reader.get_events()
        is_busy, event = reader.is_busy_at(datetime.now())

    # Check for conflicts
    conflicts = check_for_conflicts(
        "dinner tomorrow at 7",
        reference_date=datetime.now()
    )
"""

from datetime import datetime, timedelta
import logging

from .models import (
    Calendar,
    CalendarEvent,
    CalendarAccessError,
    CalendarConflict,
    CreateEventResult,
    DetectedEvent,
)
from .detector import (
    EventDetector,
    get_event_detector,
    reset_event_detector,
)
from .reader import (
    CalendarReader,
    get_calendar_reader,
    reset_calendar_reader,
)

logger = logging.getLogger(__name__)

__all__ = [
    # Models
    "Calendar",
    "CalendarEvent",
    "CalendarAccessError",
    "CalendarConflict",
    "CreateEventResult",
    "DetectedEvent",
    # Event Detection
    "EventDetector",
    "get_event_detector",
    "reset_event_detector",
    # Calendar Reading
    "CalendarReader",
    "get_calendar_reader",
    "reset_calendar_reader",
    # Helpers
    "check_for_conflicts",
    "get_schedule_context",
]


def check_for_conflicts(
    message: str,
    reference_date: datetime | None = None,
    buffer_minutes: int = 30,
) -> list[CalendarConflict]:
    """Check if a message contains events that conflict with your calendar.

    This is the main integration point for reply generation - when someone
    asks "are you free tomorrow at 7?", this detects the proposed time
    and checks against your calendar.

    Args:
        message: Message text to analyze for events.
        reference_date: Reference date for parsing (defaults to now).
        buffer_minutes: Buffer time to consider around events.

    Returns:
        List of conflicts found, empty if no conflicts or no calendar access.

    Example:
        >>> conflicts = check_for_conflicts("dinner tomorrow at 7pm?")
        >>> if conflicts:
        ...     print(f"Conflict: {conflicts[0].suggestion}")
        "Conflict: You have 'Team standup' from 6:30 PM to 7:30 PM"
    """
    conflicts = []

    # Detect events in message
    detector = get_event_detector()
    detected = detector.detect_events(message, reference_date)

    if not detected:
        return []

    # Check calendar access
    reader = get_calendar_reader()
    if not reader.check_access():
        logger.debug("No calendar access, skipping conflict check")
        return []

    # Check each detected event against calendar
    for event in detected:
        is_busy, conflicting = reader.is_busy_at(event.start, buffer_minutes)

        if is_busy and conflicting:
            # Calculate overlap
            overlap_start = max(event.start, conflicting.start)
            overlap_end = min(
                event.end or (event.start + timedelta(hours=1)),
                conflicting.end
            )
            overlap_minutes = int((overlap_end - overlap_start).total_seconds() / 60)

            # Build suggestion
            time_str = conflicting.start.strftime("%I:%M %p").lstrip("0")
            end_str = conflicting.end.strftime("%I:%M %p").lstrip("0")
            suggestion = f"You have '{conflicting.title}' from {time_str} to {end_str}"

            conflicts.append(
                CalendarConflict(
                    proposed_event=event,
                    conflicting_event=conflicting,
                    overlap_minutes=max(0, overlap_minutes),
                    suggestion=suggestion,
                )
            )

    return conflicts


def get_schedule_context(
    start: datetime | None = None,
    end: datetime | None = None,
    max_events: int = 5,
) -> str:
    """Get a natural language summary of your schedule.

    Useful for including in prompts when generating replies.

    Args:
        start: Start of range (defaults to now).
        end: End of range (defaults to end of today).
        max_events: Maximum events to include.

    Returns:
        Human-readable schedule summary, or empty string if no access.

    Example:
        >>> context = get_schedule_context()
        >>> print(context)
        "Today you have: Team standup at 10:00 AM, Lunch with Sarah at 12:30 PM"
    """
    reader = get_calendar_reader()
    if not reader.check_access():
        return ""

    start = start or datetime.now()
    if end is None:
        # Default to end of today
        end = start.replace(hour=23, minute=59, second=59)

    try:
        events = reader.get_events(start=start, end=end, limit=max_events)
    except CalendarAccessError:
        return ""

    if not events:
        return ""

    # Build summary
    today = datetime.now().date()
    tomorrow = today + timedelta(days=1)

    summaries = []
    for event in events:
        time_str = event.start.strftime("%I:%M %p").lstrip("0")

        if event.start.date() == today:
            day_prefix = "today"
        elif event.start.date() == tomorrow:
            day_prefix = "tomorrow"
        else:
            day_prefix = event.start.strftime("%A")

        if event.all_day:
            summaries.append(f"{event.title} ({day_prefix}, all day)")
        else:
            summaries.append(f"{event.title} at {time_str} ({day_prefix})")

    if len(summaries) == 1:
        return f"You have: {summaries[0]}"
    else:
        return "Your schedule: " + ", ".join(summaries)
