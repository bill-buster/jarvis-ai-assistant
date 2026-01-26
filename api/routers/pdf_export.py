"""PDF Export API endpoints.

Provides endpoints for exporting conversations to PDF format.
"""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from api.dependencies import get_imessage_reader
from integrations.imessage import ChatDBReader
from jarvis.pdf_generator import PDFExportOptions, generate_pdf

router = APIRouter(prefix="/export", tags=["export"])


class PDFExportDateRange(BaseModel):
    """Date range filter for PDF exports."""

    start: datetime | None = Field(default=None, description="Start date (inclusive)")
    end: datetime | None = Field(default=None, description="End date (inclusive)")


class PDFExportRequest(BaseModel):
    """Request to export a conversation to PDF.

    Example:
        ```json
        {
            "include_attachments": true,
            "include_reactions": true,
            "date_range": {"start": "2024-01-01", "end": "2024-01-31"},
            "limit": 1000
        }
        ```
    """

    include_attachments: bool = Field(
        default=True,
        description="Include attachment thumbnails in PDF",
    )
    include_reactions: bool = Field(
        default=True,
        description="Include message reactions in PDF",
    )
    date_range: PDFExportDateRange | None = Field(
        default=None,
        description="Optional date range filter",
    )
    limit: int = Field(
        default=1000,
        ge=1,
        le=10000,
        description="Maximum messages to export",
    )


class PDFExportResponse(BaseModel):
    """Response containing PDF export data.

    Example:
        ```json
        {
            "success": true,
            "filename": "conversation_chat123_20240126_120000.pdf",
            "data": "JVBERi0xLjQK...",
            "message_count": 150,
            "size_bytes": 245760
        }
        ```
    """

    success: bool = Field(..., description="Whether the export succeeded")
    filename: str = Field(..., description="Suggested filename for the PDF")
    data: str = Field(..., description="Base64-encoded PDF data")
    message_count: int = Field(..., description="Number of messages in the PDF")
    size_bytes: int = Field(..., description="Size of the PDF in bytes")


@router.post("/pdf/{chat_id}", response_model=PDFExportResponse)
def export_conversation_pdf(
    chat_id: str,
    request: PDFExportRequest,
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> PDFExportResponse:
    """Export a conversation to PDF format.

    Generates a beautifully formatted PDF document with:
    - Header with conversation name, participants, and date range
    - Messages with styled bubbles (grayscale for printing)
    - Inline image attachment thumbnails
    - Message reactions
    - Page numbers in footer

    Args:
        chat_id: The conversation ID to export.
        request: Export options including attachments, reactions, and date range.

    Returns:
        PDFExportResponse with base64-encoded PDF data.

    Raises:
        HTTPException: If conversation not found or export fails.
    """
    import base64

    try:
        # Get conversation metadata
        conversations = reader.get_conversations(limit=500)
        conversation = None
        for conv in conversations:
            if conv.chat_id == chat_id:
                conversation = conv
                break

        if conversation is None:
            raise HTTPException(
                status_code=404,
                detail=f"Conversation not found: {chat_id}",
            )

        # Get messages with optional date filtering
        before = request.date_range.end if request.date_range else None
        messages = reader.get_messages(
            chat_id=chat_id,
            limit=request.limit,
            before=before,
        )

        # Apply after filter if specified
        if request.date_range and request.date_range.start:
            messages = [m for m in messages if m.date >= request.date_range.start]

        if not messages:
            raise HTTPException(
                status_code=404,
                detail="No messages found in the specified range",
            )

        # Create export options
        options = PDFExportOptions(
            include_attachments=request.include_attachments,
            include_reactions=request.include_reactions,
            start_date=request.date_range.start if request.date_range else None,
            end_date=request.date_range.end if request.date_range else None,
        )

        # Generate PDF
        pdf_bytes = generate_pdf(messages, conversation, options)
        pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_chat_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in chat_id)
        safe_chat_id = safe_chat_id[:30]
        filename = f"conversation_{safe_chat_id}_{timestamp}.pdf"

        return PDFExportResponse(
            success=True,
            filename=filename,
            data=pdf_base64,
            message_count=len(messages),
            size_bytes=len(pdf_bytes),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate PDF: {e}",
        ) from e


@router.post("/pdf/{chat_id}/download")
def download_conversation_pdf(
    chat_id: str,
    request: PDFExportRequest,
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> Response:
    """Download a conversation as a PDF file.

    Returns the PDF file directly for download instead of base64-encoded data.

    Args:
        chat_id: The conversation ID to export.
        request: Export options.

    Returns:
        PDF file response for direct download.
    """
    try:
        # Get conversation metadata
        conversations = reader.get_conversations(limit=500)
        conversation = None
        for conv in conversations:
            if conv.chat_id == chat_id:
                conversation = conv
                break

        if conversation is None:
            raise HTTPException(
                status_code=404,
                detail=f"Conversation not found: {chat_id}",
            )

        # Get messages with optional date filtering
        before = request.date_range.end if request.date_range else None
        messages = reader.get_messages(
            chat_id=chat_id,
            limit=request.limit,
            before=before,
        )

        # Apply after filter if specified
        if request.date_range and request.date_range.start:
            messages = [m for m in messages if m.date >= request.date_range.start]

        if not messages:
            raise HTTPException(
                status_code=404,
                detail="No messages found in the specified range",
            )

        # Create export options
        options = PDFExportOptions(
            include_attachments=request.include_attachments,
            include_reactions=request.include_reactions,
            start_date=request.date_range.start if request.date_range else None,
            end_date=request.date_range.end if request.date_range else None,
        )

        # Generate PDF
        pdf_bytes = generate_pdf(messages, conversation, options)

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_chat_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in chat_id)
        safe_chat_id = safe_chat_id[:30]
        filename = f"conversation_{safe_chat_id}_{timestamp}.pdf"

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate PDF: {e}",
        ) from e
