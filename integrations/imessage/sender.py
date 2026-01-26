"""iMessage sender using AppleScript.

Sends iMessages via the Messages app using osascript subprocess calls.
Requires Automation permission (user will be prompted on first use).

.. deprecated::
    This module relies on AppleScript automation which has significant
    limitations imposed by Apple:

    1. **Automation Permission Required**: The user must grant Automation
       permission for the calling application (Terminal/IDE) to control
       Messages.app. This is prompted on first use but can be unreliable.

    2. **SIP Restrictions**: System Integrity Protection may block AppleScript
       from interacting with Messages.app in certain contexts.

    3. **No Background Sending**: Messages can only be sent when Messages.app
       is running and the user is logged in.

    4. **Sandboxing Issues**: Applications running in a sandbox may not be
       able to use AppleScript automation at all.

    5. **API Instability**: Apple may change or restrict AppleScript access
       to Messages.app in future macOS versions without notice.

    For these reasons, this module should be considered experimental and
    unreliable for production use. Consider using alternative communication
    methods or informing users of these limitations in the UI.
"""

from __future__ import annotations

import logging
import subprocess
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# Module-level deprecation warning for developers
_DEPRECATION_MESSAGE = (
    "IMessageSender is deprecated and may not work reliably due to Apple's "
    "AppleScript automation restrictions. See module docstring for details."
)


class TapbackType(Enum):
    """iMessage tapback reaction types."""

    LOVE = 0
    LIKE = 1
    DISLIKE = 2
    LAUGH = 3
    EMPHASIS = 4  # !!
    QUESTION = 5  # ?


@dataclass
class SendResult:
    """Result of sending a message."""

    success: bool
    error: str | None = None


class IMessageSender:
    """Send iMessages via AppleScript.

    .. deprecated::
        This class relies on AppleScript automation which has significant
        Apple-imposed limitations. See module docstring for details.
        Consider this experimental and unreliable for production use.
    """

    def __init__(self) -> None:
        """Initialize the sender.

        Emits a DeprecationWarning due to Apple's AppleScript restrictions.
        """
        warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)

    def send_message(
        self,
        text: str,
        *,
        recipient: str | None = None,
        chat_id: str | None = None,
        is_group: bool = False,
    ) -> SendResult:
        """Send a message to a conversation.

        For individual chats, provide recipient (phone/email).
        For group chats, provide chat_id and set is_group=True.

        Args:
            text: Message text to send
            recipient: Phone number or email (for individual chats)
            chat_id: The chat ID from the database (for group chats)
            is_group: Whether this is a group chat

        Returns:
            SendResult with success status and optional error message
        """
        if not text:
            return SendResult(success=False, error="Message text is required")

        if is_group:
            return self._send_to_group(text, chat_id)
        else:
            return self._send_to_individual(text, recipient)

    def send_attachment(
        self,
        file_path: str | Path,
        *,
        recipient: str | None = None,
        chat_id: str | None = None,
        is_group: bool = False,
    ) -> SendResult:
        """Send a file attachment to a conversation.

        Args:
            file_path: Path to the file to send
            recipient: Phone number or email (for individual chats)
            chat_id: The chat ID from the database (for group chats)
            is_group: Whether this is a group chat

        Returns:
            SendResult with success status and optional error message
        """
        path = Path(file_path)
        if not path.exists():
            return SendResult(success=False, error=f"File not found: {file_path}")

        # Get absolute POSIX path for AppleScript
        abs_path = str(path.resolve())
        escaped_path = abs_path.replace("\\", "\\\\").replace('"', '\\"')

        if is_group:
            if not chat_id:
                return SendResult(success=False, error="Chat ID required for group")
            escaped_chat_id = chat_id.replace("\\", "\\\\").replace('"', '\\"')
            applescript = f'''
tell application "Messages"
    set targetChat to chat id "{escaped_chat_id}"
    set theFile to POSIX file "{escaped_path}"
    send theFile to targetChat
end tell
'''
        else:
            if not recipient:
                return SendResult(success=False, error="Recipient required")
            escaped_recipient = recipient.replace("\\", "\\\\").replace('"', '\\"')
            applescript = f'''
tell application "Messages"
    set targetService to 1st account whose service type = iMessage
    set targetBuddy to participant "{escaped_recipient}" of targetService
    set theFile to POSIX file "{escaped_path}"
    send theFile to targetBuddy
end tell
'''
        return self._run_applescript(applescript, f"attachment to {chat_id or recipient}")

    def _send_to_individual(self, text: str, recipient: str | None) -> SendResult:
        """Send a message to an individual recipient."""
        if not recipient:
            return SendResult(success=False, error="Recipient is required")

        # Escape special characters for AppleScript
        escaped_text = text.replace("\\", "\\\\").replace('"', '\\"')
        escaped_recipient = recipient.replace("\\", "\\\\").replace('"', '\\"')

        applescript = f'''
tell application "Messages"
    set targetService to 1st account whose service type = iMessage
    set targetBuddy to participant "{escaped_recipient}" of targetService
    send "{escaped_text}" to targetBuddy
end tell
'''
        return self._run_applescript(applescript, f"individual: {recipient}")

    def _send_to_group(self, text: str, chat_id: str | None) -> SendResult:
        """Send a message to a group chat."""
        if not chat_id:
            return SendResult(success=False, error="Chat ID is required for group chats")

        # Escape special characters for AppleScript
        escaped_text = text.replace("\\", "\\\\").replace('"', '\\"')
        escaped_chat_id = chat_id.replace("\\", "\\\\").replace('"', '\\"')

        # For group chats, we find the chat by its ID and send directly to it
        applescript = f'''
tell application "Messages"
    set targetChat to chat id "{escaped_chat_id}"
    send "{escaped_text}" to targetChat
end tell
'''
        return self._run_applescript(applescript, f"group: {chat_id}")

    def _run_applescript(self, script: str, target_desc: str) -> SendResult:
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
