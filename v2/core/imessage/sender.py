"""iMessage sender using AppleScript.

Sends iMessages via the Messages app using osascript subprocess calls.
Requires Automation permission (user will be prompted on first use).
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def _escape_applescript_string(s: str) -> str:
    """Escape a string for safe use in AppleScript double-quoted strings.

    AppleScript requires escaping of:
    - Backslash -> \\
    - Double quote -> \"
    - Newlines and carriage returns (convert to space to avoid script issues)
    - Tab characters (convert to space)
    """
    return (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", " ")
        .replace("\r", " ")
        .replace("\t", " ")
    )


@dataclass
class SendResult:
    """Result of sending a message."""
    success: bool
    error: str | None = None


def send_message(
    text: str,
    chat_id: str,
    is_group: bool = False,
    recipient: str | None = None,
) -> SendResult:
    """Send a message to a conversation.

    Args:
        text: Message text to send
        chat_id: The chat ID from the database
        is_group: Whether this is a group chat
        recipient: Phone number or email (for individual chats, extracted from chat_id if not provided)

    Returns:
        SendResult with success status and optional error message
    """
    if not text:
        return SendResult(success=False, error="Message text is required")

    if not chat_id:
        return SendResult(success=False, error="Chat ID is required")

    # Escape special characters for AppleScript
    escaped_text = _escape_applescript_string(text)

    if is_group:
        # For group chats, send directly to the chat
        escaped_chat_id = _escape_applescript_string(chat_id)
        applescript = f'''
tell application "Messages"
    set targetChat to chat id "{escaped_chat_id}"
    send "{escaped_text}" to targetChat
end tell
'''
    else:
        # For individual chats, extract recipient from chat_id or use provided
        # chat_id format is typically "iMessage;-;+1234567890" or "iMessage;-;email@example.com"
        if recipient is None:
            parts = chat_id.split(";")
            recipient = parts[-1] if len(parts) >= 3 else chat_id

        escaped_recipient = _escape_applescript_string(recipient)
        applescript = f'''
tell application "Messages"
    set targetService to 1st account whose service type = iMessage
    set targetBuddy to participant "{escaped_recipient}" of targetService
    send "{escaped_text}" to targetBuddy
end tell
'''

    return _run_applescript(applescript, chat_id)


def _run_applescript(script: str, target_desc: str) -> SendResult:
    """Execute an AppleScript and return the result."""
    try:
        logger.info("Sending iMessage to %s", target_desc)
        subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            check=True,
            text=True,
            timeout=10,
        )
        logger.info("Message sent successfully")
        return SendResult(success=True)

    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.strip() if e.stderr else "AppleScript execution failed"
        logger.error("Failed to send message: %s", error_msg)
        return SendResult(success=False, error=error_msg)

    except subprocess.TimeoutExpired:
        logger.error("Message send timed out")
        return SendResult(success=False, error="Timeout sending message")

    except Exception as e:
        logger.exception("Unexpected error sending message")
        return SendResult(success=False, error=str(e))
