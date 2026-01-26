"""PDF generator for conversation exports.

Generates beautifully formatted PDF documents from iMessage conversations,
including headers, styled message bubbles, attachments, and reactions.
"""

import base64
import io
import os
from dataclasses import dataclass
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from contracts.imessage import Conversation, Message


@dataclass
class PDFExportOptions:
    """Options for PDF export."""

    include_attachments: bool = True
    include_reactions: bool = True
    start_date: datetime | None = None
    end_date: datetime | None = None


# Colors for the PDF (grayscale for print-friendly output)
BUBBLE_ME_COLOR = colors.Color(0.85, 0.85, 0.85)  # Light gray for sent
BUBBLE_OTHER_COLOR = colors.Color(0.95, 0.95, 0.95)  # Very light gray for received
HEADER_BG_COLOR = colors.Color(0.2, 0.2, 0.2)  # Dark gray
HEADER_TEXT_COLOR = colors.white
DATE_HEADER_COLOR = colors.Color(0.5, 0.5, 0.5)  # Medium gray
REACTION_BG_COLOR = colors.Color(0.9, 0.9, 0.9)


def _create_styles() -> dict:
    """Create paragraph styles for the PDF."""
    styles = getSampleStyleSheet()

    custom_styles = {
        "title": ParagraphStyle(
            "Title",
            parent=styles["Heading1"],
            fontSize=18,
            spaceAfter=6,
            textColor=colors.white,
            alignment=TA_CENTER,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=styles["Normal"],
            fontSize=10,
            textColor=colors.Color(0.8, 0.8, 0.8),
            alignment=TA_CENTER,
        ),
        "date_header": ParagraphStyle(
            "DateHeader",
            parent=styles["Normal"],
            fontSize=10,
            textColor=DATE_HEADER_COLOR,
            alignment=TA_CENTER,
            spaceBefore=12,
            spaceAfter=8,
        ),
        "sender": ParagraphStyle(
            "Sender",
            parent=styles["Normal"],
            fontSize=9,
            textColor=colors.Color(0.3, 0.3, 0.3),
            spaceBefore=2,
            spaceAfter=2,
        ),
        "message_me": ParagraphStyle(
            "MessageMe",
            parent=styles["Normal"],
            fontSize=11,
            leading=14,
            textColor=colors.black,
        ),
        "message_other": ParagraphStyle(
            "MessageOther",
            parent=styles["Normal"],
            fontSize=11,
            leading=14,
            textColor=colors.black,
        ),
        "timestamp": ParagraphStyle(
            "Timestamp",
            parent=styles["Normal"],
            fontSize=8,
            textColor=colors.Color(0.5, 0.5, 0.5),
        ),
        "attachment": ParagraphStyle(
            "Attachment",
            parent=styles["Normal"],
            fontSize=9,
            textColor=colors.Color(0.4, 0.4, 0.4),
        ),
        "reaction": ParagraphStyle(
            "Reaction",
            parent=styles["Normal"],
            fontSize=8,
            textColor=colors.Color(0.4, 0.4, 0.4),
        ),
        "system_message": ParagraphStyle(
            "SystemMessage",
            parent=styles["Normal"],
            fontSize=10,
            textColor=colors.Color(0.5, 0.5, 0.5),
            alignment=TA_CENTER,
            fontName="Helvetica-Oblique",
        ),
        "footer": ParagraphStyle(
            "Footer",
            parent=styles["Normal"],
            fontSize=8,
            textColor=colors.Color(0.5, 0.5, 0.5),
            alignment=TA_CENTER,
        ),
    }

    return custom_styles


def _format_date_header(date: datetime) -> str:
    """Format date for section headers."""
    today = datetime.now().date()
    msg_date = date.date()

    if msg_date == today:
        return "Today"
    elif (today - msg_date).days == 1:
        return "Yesterday"
    else:
        return date.strftime("%A, %B %d, %Y")


def _format_timestamp(date: datetime) -> str:
    """Format timestamp for messages."""
    return date.strftime("%I:%M %p").lstrip("0")


def _escape_text(text: str) -> str:
    """Escape special characters for ReportLab paragraphs."""
    # Replace special XML/HTML characters
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    # Replace newlines with line breaks
    text = text.replace("\n", "<br/>")
    return text


def _get_image_thumbnail(file_path: str | None, max_width: float = 2 * inch) -> Image | None:
    """Get a thumbnail image if the file exists and is an image."""
    if not file_path:
        return None

    # Expand ~ in path
    expanded_path = os.path.expanduser(file_path)

    if not os.path.exists(expanded_path):
        return None

    # Check if it's an image based on extension
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".heic"}
    ext = os.path.splitext(expanded_path)[1].lower()

    if ext not in image_extensions:
        return None

    try:
        img = Image(expanded_path)

        # Scale to fit within max_width while maintaining aspect ratio
        aspect = img.imageHeight / img.imageWidth
        if img.imageWidth > max_width:
            img.drawWidth = max_width
            img.drawHeight = max_width * aspect

        # Cap height too
        max_height = 2 * inch
        if img.drawHeight > max_height:
            img.drawHeight = max_height
            img.drawWidth = max_height / aspect

        return img
    except Exception:
        # If we can't load the image, return None
        return None


class PDFGenerator:
    """Generates PDF documents from iMessage conversations."""

    def __init__(self) -> None:
        """Initialize the PDF generator."""
        self.styles = _create_styles()
        self.page_width, self.page_height = letter
        self.margin = 0.75 * inch
        self.content_width = self.page_width - 2 * self.margin

    def generate(
        self,
        messages: list[Message],
        conversation: Conversation | None = None,
        options: PDFExportOptions | None = None,
    ) -> bytes:
        """Generate a PDF document from messages.

        Args:
            messages: List of messages to include.
            conversation: Optional conversation metadata.
            options: Export options (attachments, reactions, date range).

        Returns:
            PDF document as bytes.
        """
        if options is None:
            options = PDFExportOptions()

        # Filter messages by date range if specified
        filtered_messages = self._filter_by_date(messages, options)

        # Create PDF buffer
        buffer = io.BytesIO()

        # Create document
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            leftMargin=self.margin,
            rightMargin=self.margin,
            topMargin=self.margin,
            bottomMargin=self.margin,
        )

        # Build content
        story = []

        # Add header
        story.extend(self._create_header(filtered_messages, conversation))

        # Add messages
        story.extend(self._create_messages(filtered_messages, conversation, options))

        # Add footer info
        story.extend(self._create_footer_content(filtered_messages))

        # Build PDF with page numbers
        doc.build(story, onFirstPage=self._add_page_number, onLaterPages=self._add_page_number)

        # Get PDF bytes
        buffer.seek(0)
        return buffer.read()

    def _filter_by_date(
        self, messages: list[Message], options: PDFExportOptions
    ) -> list[Message]:
        """Filter messages by date range."""
        if not options.start_date and not options.end_date:
            return messages

        filtered = []
        for msg in messages:
            if options.start_date and msg.date < options.start_date:
                continue
            if options.end_date and msg.date > options.end_date:
                continue
            filtered.append(msg)

        return filtered

    def _create_header(
        self, messages: list[Message], conversation: Conversation | None
    ) -> list:
        """Create the document header section."""
        elements = []

        # Get conversation info
        if conversation:
            title = conversation.display_name or ", ".join(conversation.participants)
            participants = conversation.participants
            is_group = conversation.is_group
        else:
            title = "Conversation Export"
            participants = []
            is_group = False

        # Get date range
        if messages:
            dates = [m.date for m in messages]
            start_date = min(dates).strftime("%B %d, %Y")
            end_date = max(dates).strftime("%B %d, %Y")
            if start_date == end_date:
                date_range = start_date
            else:
                date_range = f"{start_date} - {end_date}"
        else:
            date_range = "No messages"

        # Create header table
        header_data = [
            [Paragraph(_escape_text(title), self.styles["title"])],
            [
                Paragraph(
                    f"{'Group Chat' if is_group else 'Conversation'} | {len(messages)} messages",
                    self.styles["subtitle"],
                )
            ],
            [Paragraph(date_range, self.styles["subtitle"])],
        ]

        if participants and len(participants) <= 5:
            participant_text = ", ".join(participants)
            header_data.append([Paragraph(f"Participants: {participant_text}", self.styles["subtitle"])])

        header_table = Table(
            header_data,
            colWidths=[self.content_width],
        )

        header_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), HEADER_BG_COLOR),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, 0), 16),
                    ("BOTTOMPADDING", (0, -1), (-1, -1), 16),
                    ("LEFTPADDING", (0, 0), (-1, -1), 12),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ]
            )
        )

        elements.append(header_table)
        elements.append(Spacer(1, 20))

        return elements

    def _create_messages(
        self,
        messages: list[Message],
        conversation: Conversation | None,
        options: PDFExportOptions,
    ) -> list:
        """Create message bubbles for the PDF."""
        elements = []
        current_date = None
        is_group = conversation.is_group if conversation else False

        bubble_width = self.content_width * 0.7  # 70% of content width

        for message in messages:
            # Add date header if date changed
            msg_date = message.date.date()
            if current_date != msg_date:
                current_date = msg_date
                date_text = _format_date_header(message.date)
                elements.append(
                    Paragraph(f"— {date_text} —", self.styles["date_header"])
                )

            # Handle system messages
            if message.is_system_message:
                elements.append(
                    Paragraph(_escape_text(message.text), self.styles["system_message"])
                )
                elements.append(Spacer(1, 8))
                continue

            # Create message content
            msg_elements = []

            # Sender name (for group chats, incoming messages)
            if is_group and not message.is_from_me:
                sender = message.sender_name or message.sender
                msg_elements.append(
                    Paragraph(_escape_text(sender), self.styles["sender"])
                )

            # Message text
            style = self.styles["message_me"] if message.is_from_me else self.styles["message_other"]
            msg_elements.append(Paragraph(_escape_text(message.text), style))

            # Attachments
            if options.include_attachments and message.attachments:
                for attachment in message.attachments:
                    # Try to show image thumbnail
                    img = _get_image_thumbnail(attachment.file_path)
                    if img:
                        msg_elements.append(Spacer(1, 4))
                        msg_elements.append(img)
                    else:
                        # Show attachment filename
                        msg_elements.append(
                            Paragraph(
                                f"[Attachment: {_escape_text(attachment.filename)}]",
                                self.styles["attachment"],
                            )
                        )

            # Timestamp
            timestamp = _format_timestamp(message.date)
            ts_style = ParagraphStyle(
                "ts",
                parent=self.styles["timestamp"],
                alignment=TA_RIGHT if message.is_from_me else TA_LEFT,
            )
            msg_elements.append(Paragraph(timestamp, ts_style))

            # Reactions
            if options.include_reactions and message.reactions:
                reaction_text = " ".join(
                    f"{r.type}" for r in message.reactions
                )
                msg_elements.append(
                    Paragraph(f"Reactions: {reaction_text}", self.styles["reaction"])
                )

            # Create bubble table
            bubble_color = BUBBLE_ME_COLOR if message.is_from_me else BUBBLE_OTHER_COLOR

            # Wrap message elements in a cell
            cell_content = []
            for elem in msg_elements:
                cell_content.append(elem)

            # Create inner table for bubble content
            inner_table = Table(
                [[elem] for elem in cell_content],
                colWidths=[bubble_width - 16],  # Account for padding
            )

            inner_table.setStyle(
                TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("TOPPADDING", (0, 0), (-1, -1), 2),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                    ]
                )
            )

            # Create outer table for positioning
            if message.is_from_me:
                # Right-aligned bubble
                bubble_table = Table(
                    [[Spacer(1, 1), inner_table]],
                    colWidths=[self.content_width - bubble_width, bubble_width],
                )
            else:
                # Left-aligned bubble
                bubble_table = Table(
                    [[inner_table, Spacer(1, 1)]],
                    colWidths=[bubble_width, self.content_width - bubble_width],
                )

            # Apply bubble styling
            if message.is_from_me:
                bubble_table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (1, 0), (1, 0), bubble_color),
                            ("ROUNDEDCORNERS", [10, 10, 10, 10]),
                            ("TOPPADDING", (1, 0), (1, 0), 8),
                            ("BOTTOMPADDING", (1, 0), (1, 0), 8),
                            ("LEFTPADDING", (1, 0), (1, 0), 10),
                            ("RIGHTPADDING", (1, 0), (1, 0), 10),
                        ]
                    )
                )
            else:
                bubble_table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (0, 0), bubble_color),
                            ("ROUNDEDCORNERS", [10, 10, 10, 10]),
                            ("TOPPADDING", (0, 0), (0, 0), 8),
                            ("BOTTOMPADDING", (0, 0), (0, 0), 8),
                            ("LEFTPADDING", (0, 0), (0, 0), 10),
                            ("RIGHTPADDING", (0, 0), (0, 0), 10),
                        ]
                    )
                )

            elements.append(bubble_table)
            elements.append(Spacer(1, 6))

        return elements

    def _create_footer_content(self, messages: list[Message]) -> list:
        """Create footer content at the end of the document."""
        elements = []
        elements.append(Spacer(1, 20))

        export_time = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        elements.append(
            Paragraph(
                f"Exported from JARVIS on {export_time}",
                self.styles["footer"],
            )
        )

        return elements

    def _add_page_number(self, canvas, doc) -> None:  # type: ignore[no-untyped-def]
        """Add page numbers to each page."""
        canvas.saveState()
        page_num = canvas.getPageNumber()
        text = f"Page {page_num}"

        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.Color(0.5, 0.5, 0.5))
        canvas.drawCentredString(
            self.page_width / 2,
            0.5 * inch,
            text,
        )
        canvas.restoreState()


def generate_pdf(
    messages: list[Message],
    conversation: Conversation | None = None,
    options: PDFExportOptions | None = None,
) -> bytes:
    """Generate a PDF document from messages.

    Args:
        messages: List of messages to include.
        conversation: Optional conversation metadata.
        options: Export options.

    Returns:
        PDF document as bytes.
    """
    generator = PDFGenerator()
    return generator.generate(messages, conversation, options)


def generate_pdf_base64(
    messages: list[Message],
    conversation: Conversation | None = None,
    options: PDFExportOptions | None = None,
) -> str:
    """Generate a PDF document and return as base64-encoded string.

    Args:
        messages: List of messages to include.
        conversation: Optional conversation metadata.
        options: Export options.

    Returns:
        Base64-encoded PDF document.
    """
    pdf_bytes = generate_pdf(messages, conversation, options)
    return base64.b64encode(pdf_bytes).decode("utf-8")
