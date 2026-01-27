"""macOS Calendar reader via AppleScript.

Provides read access to macOS Calendar events using AppleScript commands.
"""

from __future__ import annotations

import logging
import re
import subprocess
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from contracts.calendar import Calendar, CalendarEvent
from jarvis.errors import CalendarAccessError

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# AppleScript to list calendars
LIST_CALENDARS_SCRIPT = """
tell application "Calendar"
    set calList to {}
    repeat with cal in calendars
        set calInfo to {|id|:(id of cal), |name|:(name of cal), ¬
            |color|:"", |editable|:(writable of cal)}
        set end of calList to calInfo
    end repeat
    return calList
end tell
"""

# AppleScript template to get events
# Note: AppleScript line continuations use backslash character
GET_EVENTS_SCRIPT_TEMPLATE = """
tell application "Calendar"
    set eventList to {{}}
    set startDate to date "{start_date}"
    set endDate to date "{end_date}"

    {calendar_filter}

    repeat with cal in targetCalendars
        set calEvents to (every event of cal whose start date >= startDate ¬
            and start date <= endDate)
        repeat with evt in calEvents
            try
                set evtInfo to {{|id|:(uid of evt), |calendar_id|:(id of cal), ¬
                    |calendar_name|:(name of cal), |title|:(summary of evt), ¬
                    |start|:(start date of evt as «class isot» as string), ¬
                    |end|:(end date of evt as «class isot» as string), ¬
                    |all_day|:(allday event of evt), |location|:(location of evt), ¬
                    |notes|:(description of evt), |url|:(url of evt)}}
                set end of eventList to evtInfo
            end try
        end repeat
    end repeat
    return eventList
end tell
"""

# Calendar filter for all calendars
ALL_CALENDARS_FILTER = "set targetCalendars to calendars"

# Calendar filter for specific calendar
SPECIFIC_CALENDAR_FILTER = 'set targetCalendars to (calendars whose id is "{calendar_id}")'


class CalendarReaderImpl:
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
            List of available calendars.

        Raises:
            CalendarAccessError: If calendar access fails.
        """
        self.require_access()

        try:
            result = subprocess.run(
                ["osascript", "-e", LIST_CALENDARS_SCRIPT],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                raise CalendarAccessError(f"Failed to list calendars: {result.stderr}")

            # Parse AppleScript output
            calendars = self._parse_applescript_list(result.stdout, self._parse_calendar_dict)
            return calendars

        except subprocess.TimeoutExpired as e:
            raise CalendarAccessError("Calendar request timed out") from e

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
            end: End of date range (defaults to 30 days from now).
            limit: Maximum number of events to return.

        Returns:
            List of calendar events.

        Raises:
            CalendarAccessError: If calendar access fails.
        """
        self.require_access()

        # Set default date range
        start = start or datetime.now()
        end = end or (start + timedelta(days=30))

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

            # Parse events
            events = self._parse_applescript_list(result.stdout, self._parse_event_dict)

            # Sort by start time
            events.sort(key=lambda e: e.start)

            # Apply limit
            return events[:limit]

        except subprocess.TimeoutExpired as e:
            raise CalendarAccessError("Calendar event request timed out") from e

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
        # Get events for the next year and filter locally
        # AppleScript doesn't support text search well
        events = self.get_events(
            calendar_id=calendar_id,
            start=datetime.now(),
            end=datetime.now() + timedelta(days=365),
            limit=1000,
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
        """Parse AppleScript list output.

        AppleScript returns lists in a specific format that we need to parse.

        Args:
            output: Raw AppleScript output.
            parser: Function to parse each item.

        Returns:
            List of parsed items.
        """
        # AppleScript returns records in format: {{key:value, key:value}, {...}}
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
        """Parse a calendar record from AppleScript.

        Args:
            record: AppleScript record string.

        Returns:
            Calendar object or None if parsing fails.
        """
        try:
            data = self._parse_record(record)
            # Convert editable string to bool (AppleScript returns "true"/"false")
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
        """Parse an event record from AppleScript.

        Args:
            record: AppleScript record string.

        Returns:
            CalendarEvent object or None if parsing fails.
        """
        try:
            data = self._parse_record(record)

            # Parse dates (ISO format from AppleScript)
            start_str = data.get("start", "")
            end_str = data.get("end", "")

            start = self._parse_iso_date(start_str)
            end = self._parse_iso_date(end_str)

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
        """Parse an AppleScript record into a dictionary.

        Args:
            record: Record string like "|key|:value, |key2|:value2".

        Returns:
            Parsed dictionary.
        """
        result = {}
        # Match |key|:value patterns
        pattern = r'\|([^|]+)\|:([^,}]+|"[^"]*")'
        for match in re.finditer(pattern, record):
            key = match.group(1)
            value = match.group(2).strip().strip('"')
            if value != "missing value":
                result[key] = value
        return result

    def _parse_iso_date(self, date_str: str) -> datetime | None:
        """Parse ISO date from AppleScript.

        Args:
            date_str: ISO date string.

        Returns:
            Parsed datetime or None.
        """
        if not date_str:
            return None

        try:
            # AppleScript returns dates in ISO format
            date_str = date_str.strip().strip('"')
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except ValueError:
            try:
                # Try parsing common macOS date format
                return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S")
            except ValueError:
                return None


# Module-level singleton
_reader: CalendarReaderImpl | None = None


def get_calendar_reader() -> CalendarReaderImpl:
    """Get the singleton calendar reader.

    Returns:
        CalendarReaderImpl instance.
    """
    global _reader
    if _reader is None:
        _reader = CalendarReaderImpl()
    return _reader


def reset_calendar_reader() -> None:
    """Reset the singleton calendar reader."""
    global _reader
    _reader = None
