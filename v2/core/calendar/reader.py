"""macOS Calendar reader via AppleScript.

Provides read access to macOS Calendar events using AppleScript commands.

Ported from v1: integrations/calendar/reader.py
"""

from __future__ import annotations

import logging
import re
import subprocess
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

from .models import Calendar, CalendarEvent, CalendarAccessError

logger = logging.getLogger(__name__)

# AppleScript to list calendars (simplified - uses name as ID since id fails)
LIST_CALENDARS_SCRIPT = """
tell application "Calendar"
    set output to ""
    repeat with cal in calendars
        set output to output & (name of cal) & "\\n"
    end repeat
    return output
end tell
"""

# AppleScript template to get events (simplified - avoids id property)
GET_EVENTS_SCRIPT_TEMPLATE = """
tell application "Calendar"
    set output to ""
    set startDate to date "{start_date}"
    set endDate to date "{end_date}"

    {calendar_filter}

    repeat with cal in targetCalendars
        set calName to name of cal
        set calEvents to (every event of cal whose start date >= startDate and start date <= endDate)
        repeat with evt in calEvents
            try
                set evtTitle to summary of evt
                set evtStart to start date of evt
                set evtEnd to end date of evt
                set evtAllDay to allday event of evt
                set evtLoc to ""
                try
                    set evtLoc to location of evt
                end try
                set output to output & calName & "||" & evtTitle & "||" & (evtStart as string) & "||" & (evtEnd as string) & "||" & evtAllDay & "||" & evtLoc & "\\n"
            end try
        end repeat
    end repeat
    return output
end tell
"""

# Calendar filter for all calendars
ALL_CALENDARS_FILTER = "set targetCalendars to calendars"

# Calendar filter for specific calendar (uses name since id fails)
SPECIFIC_CALENDAR_FILTER = 'set targetCalendars to (calendars whose name is "{calendar_id}")'


class CalendarReader:
    """Reads events from macOS Calendar.

    Uses AppleScript to interact with the Calendar app.
    """

    def __init__(self) -> None:
        """Initialize the calendar reader."""
        self._access_checked = False
        self._has_access = False

    def check_access(self) -> bool:
        """Check if we have permission to access Calendar.

        Returns:
            True if calendar access is available.
        """
        if self._access_checked:
            return self._has_access

        try:
            # Try a simple AppleScript command
            result = subprocess.run(
                ["osascript", "-e", 'tell application "Calendar" to name of calendars'],
                capture_output=True,
                text=True,
                timeout=10,
            )
            self._has_access = result.returncode == 0
            self._access_checked = True
            return self._has_access
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning("Calendar access check failed: %s", e)
            self._has_access = False
            self._access_checked = True
            return False

    def require_access(self) -> None:
        """Require calendar access, raising error if unavailable.

        Raises:
            CalendarAccessError: If calendar access is not available.
        """
        if not self.check_access():
            raise CalendarAccessError(
                "Calendar access is required. Please grant Calendar permissions.",
                requires_permission=True,
            )

    def get_calendars(self) -> list[Calendar]:
        """Get available calendars.

        Returns:
            List of available calendars, or empty list if access fails.

        Note:
            Requires Calendar automation permission in System Settings >
            Privacy & Security > Automation.
        """
        if not self.check_access():
            return []

        try:
            result = subprocess.run(
                ["osascript", "-e", LIST_CALENDARS_SCRIPT],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                logger.warning("Failed to list calendars: %s", result.stderr)
                return []

            # Parse simplified output: one calendar name per line
            calendars = []
            for line in result.stdout.strip().split("\n"):
                name = line.strip()
                if name:
                    calendars.append(Calendar(
                        id=name,  # Use name as ID (id property fails on some calendars)
                        name=name,
                        is_editable=True,
                    ))
            return calendars

        except subprocess.TimeoutExpired:
            logger.warning("Calendar request timed out")
            return []
        except Exception as e:
            logger.warning("Calendar access error: %s", e)
            return []

    def get_events(
        self,
        calendar_id: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 50,
    ) -> list[CalendarEvent]:
        """Get events from calendar(s).

        Args:
            calendar_id: Optional calendar ID to filter by.
            start: Start of date range (defaults to now).
            end: End of date range (defaults to 7 days from now).
            limit: Maximum number of events to return.

        Returns:
            List of calendar events.

        Raises:
            CalendarAccessError: If calendar access fails.
        """
        self.require_access()

        # Set default date range
        start = start or datetime.now()
        end = end or (start + timedelta(days=7))

        # Format dates for AppleScript
        start_str = start.strftime("%B %d, %Y %I:%M:%S %p")
        end_str = end.strftime("%B %d, %Y %I:%M:%S %p")

        # Build calendar filter
        if calendar_id:
            cal_filter = SPECIFIC_CALENDAR_FILTER.format(calendar_id=calendar_id)
        else:
            cal_filter = ALL_CALENDARS_FILTER

        # Build script
        script = GET_EVENTS_SCRIPT_TEMPLATE.format(
            start_date=start_str,
            end_date=end_str,
            calendar_filter=cal_filter,
        )

        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                logger.warning("Get events failed: %s", result.stderr)
                return []

            # Parse simplified output: "cal_name||title||start||end||all_day||location\n"
            events = []
            for line in result.stdout.strip().split("\n"):
                line = line.strip()
                if "||" in line:
                    parts = line.split("||")
                    if len(parts) >= 4:
                        start = self._parse_date(parts[2])
                        end = self._parse_date(parts[3])
                        if start and end:
                            events.append(CalendarEvent(
                                id=f"{parts[0]}:{parts[1]}:{start.isoformat()}",  # Generate ID
                                calendar_id=parts[0],
                                calendar_name=parts[0],
                                title=parts[1],
                                start=start,
                                end=end,
                                all_day=parts[4].lower() == "true" if len(parts) > 4 else False,
                                location=parts[5] if len(parts) > 5 and parts[5] else None,
                            ))

            # Sort by start time
            events.sort(key=lambda e: e.start)

            # Apply limit
            return events[:limit]

        except subprocess.TimeoutExpired:
            logger.warning("Calendar event request timed out")
            return []
        except Exception as e:
            logger.warning("Failed to get events: %s", e)
            return []

    def get_events_for_day(self, date: datetime | None = None) -> list[CalendarEvent]:
        """Get all events for a specific day.

        Args:
            date: Date to get events for (defaults to today).

        Returns:
            List of calendar events for that day.
        """
        date = date or datetime.now()
        start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        return self.get_events(start=start, end=end, limit=100)

    def is_busy_at(self, when: datetime, buffer_minutes: int = 30) -> tuple[bool, CalendarEvent | None]:
        """Check if there's a calendar event at a specific time.

        Args:
            when: Time to check.
            buffer_minutes: Buffer around the time to check.

        Returns:
            Tuple of (is_busy, conflicting_event or None).
        """
        start = when - timedelta(minutes=buffer_minutes)
        end = when + timedelta(minutes=buffer_minutes)

        events = self.get_events(start=start, end=end, limit=10)

        for event in events:
            # Check for actual overlap
            if event.start <= when < event.end:
                return True, event
            if event.start <= (when + timedelta(minutes=buffer_minutes)) and event.end > when:
                return True, event

        return False, None

    def search_events(
        self,
        query: str,
        calendar_id: str | None = None,
        limit: int = 50,
    ) -> list[CalendarEvent]:
        """Search events by text query.

        Args:
            query: Search text.
            calendar_id: Optional calendar ID to filter by.
            limit: Maximum number of results.

        Returns:
            List of matching events.
        """
        # Get events for the next 90 days and filter locally
        events = self.get_events(
            calendar_id=calendar_id,
            start=datetime.now(),
            end=datetime.now() + timedelta(days=90),
            limit=500,
        )

        query_lower = query.lower()
        matching = []

        for event in events:
            if (
                query_lower in event.title.lower()
                or (event.location and query_lower in event.location.lower())
                or (event.notes and query_lower in event.notes.lower())
            ):
                matching.append(event)
                if len(matching) >= limit:
                    break

        return matching

    def _parse_applescript_list(
        self,
        output: str,
        parser: Callable[[str], Any],
    ) -> list[Any]:
        """Parse AppleScript list output."""
        output = output.strip()
        if not output or output in ("{}", "{{}}"):
            return []

        items = []
        # Simple parsing - split by "}, {" and clean up
        parts = output.split("}, {")

        for part in parts:
            # Clean up braces
            part = part.strip().lstrip("{").rstrip("}")
            if not part:
                continue

            try:
                item = parser(part)
                if item:
                    items.append(item)
            except Exception as e:
                logger.debug("Failed to parse item: %s, error: %s", part[:100], e)

        return items

    def _parse_calendar_dict(self, record: str) -> Calendar | None:
        """Parse a calendar record from AppleScript."""
        try:
            data = self._parse_record(record)
            editable_str = data.get("editable", "true")
            is_editable = (
                editable_str.lower() == "true"
                if isinstance(editable_str, str)
                else bool(editable_str)
            )
            return Calendar(
                id=data.get("id", ""),
                name=data.get("name", "Unknown"),
                color=data.get("color") or None,
                is_editable=is_editable,
            )
        except Exception as e:
            logger.debug("Failed to parse calendar: %s", e)
            return None

    def _parse_event_dict(self, record: str) -> CalendarEvent | None:
        """Parse an event record from AppleScript."""
        try:
            data = self._parse_record(record)

            # Parse dates
            start_str = data.get("start", "")
            end_str = data.get("end", "")

            start = self._parse_date(start_str)
            end = self._parse_date(end_str)

            if not start or not end:
                return None

            return CalendarEvent(
                id=data.get("id", ""),
                calendar_id=data.get("calendar_id", ""),
                calendar_name=data.get("calendar_name", ""),
                title=data.get("title", "Untitled Event"),
                start=start,
                end=end,
                all_day=data.get("all_day", "false").lower() == "true",
                location=data.get("location") or None,
                notes=data.get("notes") or None,
                url=data.get("url") or None,
                attendees=[],
                status="confirmed",
            )
        except Exception as e:
            logger.debug("Failed to parse event: %s", e)
            return None

    def _parse_record(self, record: str) -> dict[str, str]:
        """Parse an AppleScript record into a dictionary."""
        result = {}
        # Match |key|:value patterns
        pattern = r'\|([^|]+)\|:([^,}]+|"[^"]*")'
        for match in re.finditer(pattern, record):
            key = match.group(1)
            value = match.group(2).strip().strip('"')
            if value != "missing value":
                result[key] = value
        return result

    def _parse_date(self, date_str: str) -> datetime | None:
        """Parse date from AppleScript output."""
        if not date_str:
            return None

        date_str = date_str.strip().strip('"')

        # Try ISO format first
        try:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except ValueError:
            pass

        # Try common macOS date formats
        formats = [
            "%Y-%m-%dT%H:%M:%S",
            "%A, %B %d, %Y at %I:%M:%S %p",  # Monday, February 2, 2026 at 10:40:00 AM
            "%A, %B %d, %Y at %H:%M:%S",      # Monday, February 2, 2026 at 10:40:00
            "%B %d, %Y at %I:%M:%S %p",
            "%B %d, %Y at %H:%M:%S",
            "%m/%d/%Y %I:%M:%S %p",
            "%m/%d/%Y, %I:%M:%S %p",
            "%d/%m/%Y %H:%M:%S",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        # Try dateutil as fallback
        try:
            from dateutil import parser as dateutil_parser
            return dateutil_parser.parse(date_str)
        except Exception:
            pass

        logger.debug("Could not parse date: %s", date_str)
        return None


# Module-level singleton
_reader: CalendarReader | None = None


def get_calendar_reader() -> CalendarReader:
    """Get the singleton calendar reader."""
    global _reader
    if _reader is None:
        _reader = CalendarReader()
    return _reader


def reset_calendar_reader() -> None:
    """Reset the singleton calendar reader."""
    global _reader
    _reader = None
