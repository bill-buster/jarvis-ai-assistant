"""iMessage integration."""

from .reader import MessageReader, Message, Conversation
from .sender import send_message, SendResult

__all__ = ["MessageReader", "Message", "Conversation", "send_message", "SendResult"]
