"""Export API endpoints.

Provides endpoints for exporting conversations, search results, and backups.
"""

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_imessage_reader
from api.schemas import (
    ExportBackupRequest,
    ExportConversationRequest,
    ExportFormatEnum,
    ExportResponse,
    ExportSearchRequest,
)
from contracts.imessage import Conversation, Message
from integrations.imessage import ChatDBReader
from jarvis.export import (
    ExportFormat,
    export_backup,
    export_messages,
    export_search_results,
    get_export_filename,
)

router = APIRouter(prefix="/export", tags=["export"])


def _to_export_format(format_enum: ExportFormatEnum) -> ExportFormat:
    """Convert API format enum to export module format enum."""
    return ExportFormat(format_enum.value)


@router.post("/conversation/{chat_id}", response_model=ExportResponse)
def export_conversation(
    chat_id: str,
    request: ExportConversationRequest,
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> ExportResponse:
    """Export a single conversation.

    Exports all messages from a conversation in the specified format.
    Supports JSON (full data), CSV (flattened), and TXT (human-readable).

    Args:
        chat_id: The conversation ID to export.
        request: Export options including format and date range.

    Returns:
        ExportResponse with exported data.
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

        # Export in requested format
        export_format = _to_export_format(request.format)
        exported_data = export_messages(
            messages=messages,
            format=export_format,
            conversation=conversation,
            include_attachments=request.include_attachments,
        )

        filename = get_export_filename(
            format=export_format,
            prefix="conversation",
            chat_id=chat_id,
        )

        return ExportResponse(
            success=True,
            format=request.format.value,
            filename=filename,
            data=exported_data,
            message_count=len(messages),
            export_type="conversation",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to export conversation: {e}",
        ) from e


@router.post("/search", response_model=ExportResponse)
def export_search(
    request: ExportSearchRequest,
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> ExportResponse:
    """Export search results.

    Searches messages and exports the results in the specified format.

    Args:
        request: Search query and export options.

    Returns:
        ExportResponse with exported search results.
    """
    try:
        # Perform search
        after = request.date_range.start if request.date_range else None
        before = request.date_range.end if request.date_range else None

        messages = reader.search(
            query=request.query,
            limit=request.limit,
            sender=request.sender,
            after=after,
            before=before,
        )

        if not messages:
            raise HTTPException(
                status_code=404,
                detail=f"No messages found matching query: {request.query}",
            )

        # Export in requested format
        export_format = _to_export_format(request.format)
        exported_data = export_search_results(
            messages=messages,
            query=request.query,
            format=export_format,
        )

        filename = get_export_filename(
            format=export_format,
            prefix="search_results",
        )

        return ExportResponse(
            success=True,
            format=request.format.value,
            filename=filename,
            data=exported_data,
            message_count=len(messages),
            export_type="search",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to export search results: {e}",
        ) from e


@router.post("/backup", response_model=ExportResponse)
def export_full_backup(
    request: ExportBackupRequest,
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> ExportResponse:
    """Create a full backup of accessible conversations.

    Exports multiple conversations with their messages in JSON format.
    Only JSON format is supported for full backups.

    Args:
        request: Backup options including limits.

    Returns:
        ExportResponse with backup data.
    """
    try:
        # Get conversations
        conversations = reader.get_conversations(limit=request.conversation_limit)

        if not conversations:
            raise HTTPException(
                status_code=404,
                detail="No conversations found to backup",
            )

        # Collect messages for each conversation
        conversation_data: list[tuple[Conversation, list[Message]]] = []
        total_messages = 0

        for conv in conversations:
            before = request.date_range.end if request.date_range else None
            messages = reader.get_messages(
                chat_id=conv.chat_id,
                limit=request.messages_per_conversation,
                before=before,
            )

            # Apply after filter if specified
            if request.date_range and request.date_range.start:
                messages = [m for m in messages if m.date >= request.date_range.start]

            if messages:
                conversation_data.append((conv, messages))
                total_messages += len(messages)

        if not conversation_data:
            raise HTTPException(
                status_code=404,
                detail="No messages found in the specified range",
            )

        # Export backup (JSON only)
        exported_data = export_backup(conversation_data)

        filename = get_export_filename(
            format=ExportFormat.JSON,
            prefix="backup",
        )

        return ExportResponse(
            success=True,
            format="json",
            filename=filename,
            data=exported_data,
            message_count=total_messages,
            export_type="backup",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create backup: {e}",
        ) from e
