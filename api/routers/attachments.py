"""Attachments API endpoints.

Provides endpoints for listing, filtering, and analyzing attachments
across iMessage conversations.

Features:
- List attachments with filtering by type, date, and contact
- Get attachment thumbnails
- Calculate storage usage per conversation
"""

import logging
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from api.dependencies import get_imessage_reader
from api.schemas import (
    AttachmentStatsResponse,
    AttachmentTypeEnum,
    AttachmentWithContextResponse,
    ErrorResponse,
    ExtendedAttachmentResponse,
    StorageByConversationResponse,
    StorageSummaryResponse,
)
from integrations.imessage import ChatDBReader

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/attachments", tags=["attachments"])


def format_bytes(size_bytes: int) -> str:
    """Format bytes as human-readable string.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string like "1.5 MB" or "2.3 GB"
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


@router.get(
    "",
    response_model=list[AttachmentWithContextResponse],
    response_model_exclude_unset=True,
    response_description="List of attachments with context",
    summary="List attachments with filtering",
    responses={
        200: {
            "description": "Attachments retrieved successfully",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "attachment": {
                                "filename": "IMG_1234.jpg",
                                "mime_type": "image/jpeg",
                                "file_size": 245760,
                                "width": 1920,
                                "height": 1080,
                            },
                            "message_id": 12345,
                            "message_date": "2024-01-15T10:30:00Z",
                            "chat_id": "chat123456789",
                            "sender": "+15551234567",
                            "is_from_me": False,
                        }
                    ]
                }
            },
        },
        403: {
            "description": "Full Disk Access not granted",
            "model": ErrorResponse,
        },
    },
)
def list_attachments(
    chat_id: str | None = Query(
        default=None,
        description="Filter by conversation ID",
        examples=["chat123456789"],
    ),
    attachment_type: AttachmentTypeEnum = Query(
        default=AttachmentTypeEnum.ALL,
        description="Filter by attachment type",
    ),
    after: datetime | None = Query(
        default=None,
        description="Only attachments after this date",
        examples=["2024-01-01T00:00:00Z"],
    ),
    before: datetime | None = Query(
        default=None,
        description="Only attachments before this date",
        examples=["2024-12-31T23:59:59Z"],
    ),
    limit: int = Query(
        default=100,
        ge=1,
        le=500,
        description="Maximum number of attachments to return",
    ),
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> list[AttachmentWithContextResponse]:
    """List attachments with optional filtering.

    Returns attachments sorted by date (newest first), with full metadata
    including dimensions for images/videos and message context.

    **Filtering Options:**
    - `chat_id`: Filter to a specific conversation
    - `attachment_type`: Filter by type (images, videos, audio, documents)
    - `after`/`before`: Date range filter
    - `limit`: Maximum results (default 100, max 500)

    **Example Requests:**
    - All images: `GET /attachments?attachment_type=images`
    - Videos from a conversation: `GET /attachments?chat_id=chat123&attachment_type=videos`
    """
    # Convert attachment type to filter value
    type_filter = None if attachment_type == AttachmentTypeEnum.ALL else attachment_type.value

    attachments = reader.get_attachments(
        chat_id=chat_id,
        attachment_type=type_filter,
        after=after,
        before=before,
        limit=limit,
    )

    results = []
    for item in attachments:
        attachment = item["attachment"]
        results.append(
            AttachmentWithContextResponse(
                attachment=ExtendedAttachmentResponse(
                    filename=attachment.filename,
                    file_path=attachment.file_path,
                    mime_type=attachment.mime_type,
                    file_size=attachment.file_size,
                    width=attachment.width,
                    height=attachment.height,
                    duration_seconds=attachment.duration_seconds,
                    created_date=attachment.created_date,
                    is_sticker=attachment.is_sticker,
                    uti=attachment.uti,
                ),
                message_id=item["message_id"],
                message_date=item["message_date"],
                chat_id=item["chat_id"],
                sender=item["sender"],
                sender_name=item["sender_name"],
                is_from_me=item["is_from_me"],
            )
        )

    return results


@router.get(
    "/stats/{chat_id}",
    response_model=AttachmentStatsResponse,
    response_model_exclude_unset=True,
    response_description="Attachment statistics for the conversation",
    summary="Get attachment statistics for a conversation",
    responses={
        200: {
            "description": "Statistics retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "chat_id": "chat123456789",
                        "total_count": 150,
                        "total_size_bytes": 524288000,
                        "total_size_formatted": "500.0 MB",
                        "by_type": {"images": 100, "videos": 30},
                        "size_by_type": {"images": 314572800, "videos": 157286400},
                    }
                }
            },
        },
        403: {
            "description": "Full Disk Access not granted",
            "model": ErrorResponse,
        },
    },
)
def get_attachment_stats(
    chat_id: str,
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> AttachmentStatsResponse:
    """Get attachment statistics for a specific conversation.

    Returns the total count and size of attachments, broken down by type
    (images, videos, audio, documents, other).

    **Response includes:**
    - Total attachment count
    - Total size in bytes and human-readable format
    - Breakdown by type (count and size)
    """
    stats = reader.get_attachment_stats(chat_id)

    return AttachmentStatsResponse(
        chat_id=chat_id,
        total_count=stats["total_count"],
        total_size_bytes=stats["total_size_bytes"],
        total_size_formatted=format_bytes(stats["total_size_bytes"]),
        by_type=stats["by_type"],
        size_by_type=stats["size_by_type"],
    )


@router.get(
    "/storage",
    response_model=StorageSummaryResponse,
    response_model_exclude_unset=True,
    response_description="Storage breakdown by conversation",
    summary="Get storage usage by conversation",
    responses={
        200: {
            "description": "Storage summary retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "total_attachments": 1500,
                        "total_size_bytes": 5242880000,
                        "total_size_formatted": "5.0 GB",
                        "by_conversation": [
                            {
                                "chat_id": "chat123",
                                "display_name": "John Doe",
                                "attachment_count": 150,
                                "total_size_bytes": 524288000,
                                "total_size_formatted": "500.0 MB",
                            }
                        ],
                    }
                }
            },
        },
        403: {
            "description": "Full Disk Access not granted",
            "model": ErrorResponse,
        },
    },
)
def get_storage_summary(
    limit: int = Query(
        default=50,
        ge=1,
        le=200,
        description="Maximum number of conversations to return",
    ),
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> StorageSummaryResponse:
    """Get storage usage breakdown by conversation.

    Returns a summary of attachment storage across all conversations,
    sorted by total size (largest first).

    **Use cases:**
    - Identify conversations taking up the most space
    - Get total iMessage attachment storage usage
    - Plan storage cleanup
    """
    storage_data = reader.get_storage_by_conversation(limit=limit)

    total_attachments = 0
    total_size = 0
    by_conversation = []

    for item in storage_data:
        total_attachments += item["attachment_count"]
        total_size += item["total_size_bytes"]

        by_conversation.append(
            StorageByConversationResponse(
                chat_id=item["chat_id"],
                display_name=item["display_name"],
                attachment_count=item["attachment_count"],
                total_size_bytes=item["total_size_bytes"],
                total_size_formatted=format_bytes(item["total_size_bytes"]),
            )
        )

    return StorageSummaryResponse(
        total_attachments=total_attachments,
        total_size_bytes=total_size,
        total_size_formatted=format_bytes(total_size),
        by_conversation=by_conversation,
    )


@router.get(
    "/thumbnail",
    response_class=FileResponse,
    response_description="Thumbnail image file",
    summary="Get attachment thumbnail",
    responses={
        200: {
            "description": "Thumbnail retrieved successfully",
            "content": {"image/jpeg": {}, "image/png": {}},
        },
        404: {
            "description": "Thumbnail not found or not available",
            "model": ErrorResponse,
        },
        403: {
            "description": "Full Disk Access not granted",
            "model": ErrorResponse,
        },
    },
)
def get_thumbnail(
    file_path: str = Query(
        ...,
        description="Path to the original attachment file",
    ),
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> FileResponse:
    """Get a thumbnail for an attachment.

    Attempts to find and return a thumbnail for the specified attachment.
    If no thumbnail exists, returns a 404 error.

    **Note:** Only image and video attachments typically have thumbnails.
    """
    # Expand tilde in path
    if file_path.startswith("~"):
        file_path = str(Path(file_path).expanduser())

    # Security check: ensure the path is within the expected attachments directory
    attachments_base = Path.home() / "Library" / "Messages" / "Attachments"
    try:
        resolved_path = Path(file_path).resolve()
        if not str(resolved_path).startswith(str(attachments_base)):
            raise HTTPException(
                status_code=403,
                detail="Access denied: path outside attachments directory",
            )
    except Exception as e:
        logger.warning(f"Path resolution error: {e}")
        raise HTTPException(
            status_code=400,
            detail="Invalid file path",
        ) from e

    # Check for thumbnail
    thumb_path = reader.get_attachment_thumbnail_path(file_path)

    if thumb_path:
        return FileResponse(
            thumb_path,
            media_type="image/jpeg",
            filename=Path(thumb_path).name,
        )

    # If no thumbnail exists, try to return the original file (for small images)
    original_path = Path(file_path)
    if original_path.exists():
        # Only serve original if it's an image and reasonably sized
        suffix = original_path.suffix.lower()
        if suffix in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".heic"):
            try:
                size = original_path.stat().st_size
                # Only serve if under 2MB
                if size < 2 * 1024 * 1024:
                    media_type = "image/jpeg"
                    if suffix == ".png":
                        media_type = "image/png"
                    elif suffix == ".gif":
                        media_type = "image/gif"
                    elif suffix == ".webp":
                        media_type = "image/webp"

                    return FileResponse(
                        str(original_path),
                        media_type=media_type,
                        filename=original_path.name,
                    )
            except Exception as e:
                logger.debug(f"Error checking file size: {e}")

    raise HTTPException(
        status_code=404,
        detail="Thumbnail not available for this attachment",
    )


@router.get(
    "/file",
    response_class=FileResponse,
    response_description="Attachment file",
    summary="Download attachment file",
    responses={
        200: {
            "description": "File retrieved successfully",
        },
        404: {
            "description": "File not found",
            "model": ErrorResponse,
        },
        403: {
            "description": "Access denied",
            "model": ErrorResponse,
        },
    },
)
def download_attachment(
    file_path: str = Query(
        ...,
        description="Path to the attachment file",
    ),
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> FileResponse:
    """Download an attachment file.

    Returns the original attachment file for download.

    **Security:** Only files within the iMessage attachments directory can be accessed.
    """
    # Expand tilde in path
    if file_path.startswith("~"):
        file_path = str(Path(file_path).expanduser())

    # Security check: ensure the path is within the expected attachments directory
    attachments_base = Path.home() / "Library" / "Messages" / "Attachments"
    try:
        resolved_path = Path(file_path).resolve()
        if not str(resolved_path).startswith(str(attachments_base)):
            raise HTTPException(
                status_code=403,
                detail="Access denied: path outside attachments directory",
            )
    except Exception as e:
        logger.warning(f"Path resolution error: {e}")
        raise HTTPException(
            status_code=400,
            detail="Invalid file path",
        ) from e

    if not resolved_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Attachment file not found",
        )

    # Determine media type from extension
    suffix = resolved_path.suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".heic": "image/heic",
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".m4v": "video/mp4",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".wav": "audio/wav",
        ".pdf": "application/pdf",
        ".doc": "application/msword",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }
    media_type = media_types.get(suffix, "application/octet-stream")

    return FileResponse(
        str(resolved_path),
        media_type=media_type,
        filename=resolved_path.name,
    )
