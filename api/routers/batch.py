"""Batch operations API endpoints.

Provides endpoints for batch export, summarization, and reply generation.
All batch operations are executed as background tasks.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from api.routers.tasks import TaskResponse, _task_to_response
from jarvis.tasks import TaskType, get_task_queue, get_worker, start_worker

router = APIRouter(prefix="/batch", tags=["batch"])


# =============================================================================
# Pydantic Schemas
# =============================================================================


class BatchExportRequest(BaseModel):
    """Request for batch export of conversations.

    Example:
        ```json
        {
            "chat_ids": ["chat1", "chat2"],
            "format": "json",
            "output_dir": "/path/to/output"
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chat_ids": ["chat123", "chat456"],
                "format": "json",
                "output_dir": None,
            }
        }
    )

    chat_ids: list[str] = Field(
        ...,
        min_length=1,
        description="List of conversation IDs to export",
    )
    format: str = Field(
        default="json",
        description="Export format (json, csv, txt)",
    )
    output_dir: str | None = Field(
        default=None,
        description="Directory to save export files (optional)",
    )


class BatchExportAllRequest(BaseModel):
    """Request to export all conversations.

    Example:
        ```json
        {
            "format": "json",
            "limit": 50,
            "output_dir": "/path/to/output"
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "format": "json",
                "limit": 50,
                "output_dir": None,
            }
        }
    )

    format: str = Field(
        default="json",
        description="Export format (json, csv, txt)",
    )
    limit: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum number of conversations to export",
    )
    output_dir: str | None = Field(
        default=None,
        description="Directory to save export files (optional)",
    )


class BatchSummarizeRequest(BaseModel):
    """Request for batch summarization of conversations.

    Example:
        ```json
        {
            "chat_ids": ["chat1", "chat2"],
            "num_messages": 50
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chat_ids": ["chat123", "chat456"],
                "num_messages": 50,
            }
        }
    )

    chat_ids: list[str] = Field(
        ...,
        min_length=1,
        description="List of conversation IDs to summarize",
    )
    num_messages: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Number of messages to include in each summary",
    )


class BatchSummarizeRecentRequest(BaseModel):
    """Request to summarize recent conversations.

    Example:
        ```json
        {
            "limit": 10,
            "num_messages": 50
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "limit": 10,
                "num_messages": 50,
            }
        }
    )

    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of most recent conversations to summarize",
    )
    num_messages: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Number of messages to include in each summary",
    )


class BatchGenerateRepliesRequest(BaseModel):
    """Request for batch reply generation.

    Example:
        ```json
        {
            "chat_ids": ["chat1", "chat2"],
            "instruction": "be friendly",
            "num_suggestions": 3
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chat_ids": ["chat123", "chat456"],
                "instruction": None,
                "num_suggestions": 3,
            }
        }
    )

    chat_ids: list[str] = Field(
        ...,
        min_length=1,
        description="List of conversation IDs to generate replies for",
    )
    instruction: str | None = Field(
        default=None,
        description="Optional instruction for reply tone/content",
    )
    num_suggestions: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Number of reply suggestions per conversation",
    )


class BatchResponse(BaseModel):
    """Response for batch operations.

    Returns the created task for tracking progress.

    Example:
        ```json
        {
            "task": {...},
            "message": "Batch export started for 5 conversations"
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "task": {},
                "message": "Batch export started for 5 conversations",
            }
        }
    )

    task: TaskResponse = Field(..., description="Created background task")
    message: str = Field(..., description="Human-readable status message")


# =============================================================================
# Helper Functions
# =============================================================================


def _ensure_worker_running() -> None:
    """Ensure the background worker is running."""
    worker = get_worker()
    if not worker.is_running:
        start_worker()


def _get_all_chat_ids(limit: int = 50) -> list[str]:
    """Get all chat IDs from iMessage database.

    Args:
        limit: Maximum number of conversations to return.

    Returns:
        List of chat IDs.
    """
    from integrations.imessage import ChatDBReader

    with ChatDBReader() as reader:
        conversations = reader.get_conversations(limit=limit)
        return [c.chat_id for c in conversations]


# =============================================================================
# API Endpoints
# =============================================================================


@router.post("/export", response_model=BatchResponse)
def batch_export(request: BatchExportRequest) -> BatchResponse:
    """Export multiple conversations as a background task.

    Creates a task to export the specified conversations. The task runs
    in the background and can be monitored via the tasks API.

    Args:
        request: Batch export request with chat IDs and format.

    Returns:
        Created task for tracking progress.
    """
    _ensure_worker_running()

    queue = get_task_queue()
    task = queue.enqueue(
        task_type=TaskType.BATCH_EXPORT,
        params={
            "chat_ids": request.chat_ids,
            "format": request.format,
            "output_dir": request.output_dir,
        },
    )

    return BatchResponse(
        task=_task_to_response(task),
        message=f"Batch export started for {len(request.chat_ids)} conversations",
    )


@router.post("/export/all", response_model=BatchResponse)
def batch_export_all(request: BatchExportAllRequest) -> BatchResponse:
    """Export all conversations as a background task.

    Fetches all conversation IDs and creates a task to export them.

    Args:
        request: Export options including format and limit.

    Returns:
        Created task for tracking progress.
    """
    try:
        chat_ids = _get_all_chat_ids(limit=request.limit)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get conversations: {e}",
        ) from e

    if not chat_ids:
        raise HTTPException(
            status_code=404,
            detail="No conversations found",
        )

    _ensure_worker_running()

    queue = get_task_queue()
    task = queue.enqueue(
        task_type=TaskType.BATCH_EXPORT,
        params={
            "chat_ids": chat_ids,
            "format": request.format,
            "output_dir": request.output_dir,
        },
    )

    return BatchResponse(
        task=_task_to_response(task),
        message=f"Batch export started for {len(chat_ids)} conversations",
    )


@router.post("/summarize", response_model=BatchResponse)
def batch_summarize(request: BatchSummarizeRequest) -> BatchResponse:
    """Summarize multiple conversations as a background task.

    Creates a task to generate summaries for the specified conversations.

    Args:
        request: Batch summarize request with chat IDs.

    Returns:
        Created task for tracking progress.
    """
    _ensure_worker_running()

    queue = get_task_queue()
    task = queue.enqueue(
        task_type=TaskType.BATCH_SUMMARIZE,
        params={
            "chat_ids": request.chat_ids,
            "num_messages": request.num_messages,
        },
    )

    return BatchResponse(
        task=_task_to_response(task),
        message=f"Batch summarization started for {len(request.chat_ids)} conversations",
    )


@router.post("/summarize/recent", response_model=BatchResponse)
def batch_summarize_recent(request: BatchSummarizeRecentRequest) -> BatchResponse:
    """Summarize recent conversations as a background task.

    Fetches the most recent conversation IDs and creates a task to summarize them.

    Args:
        request: Options including limit and message count.

    Returns:
        Created task for tracking progress.
    """
    try:
        chat_ids = _get_all_chat_ids(limit=request.limit)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get conversations: {e}",
        ) from e

    if not chat_ids:
        raise HTTPException(
            status_code=404,
            detail="No conversations found",
        )

    _ensure_worker_running()

    queue = get_task_queue()
    task = queue.enqueue(
        task_type=TaskType.BATCH_SUMMARIZE,
        params={
            "chat_ids": chat_ids,
            "num_messages": request.num_messages,
        },
    )

    return BatchResponse(
        task=_task_to_response(task),
        message=f"Batch summarization started for {len(chat_ids)} recent conversations",
    )


@router.post("/generate-replies", response_model=BatchResponse)
def batch_generate_replies(request: BatchGenerateRepliesRequest) -> BatchResponse:
    """Generate reply suggestions for multiple conversations as a background task.

    Creates a task to generate reply suggestions for each specified conversation.

    Args:
        request: Batch reply request with chat IDs and options.

    Returns:
        Created task for tracking progress.
    """
    _ensure_worker_running()

    queue = get_task_queue()
    task = queue.enqueue(
        task_type=TaskType.BATCH_GENERATE_REPLIES,
        params={
            "chat_ids": request.chat_ids,
            "instruction": request.instruction,
            "num_suggestions": request.num_suggestions,
        },
    )

    return BatchResponse(
        task=_task_to_response(task),
        message=f"Batch reply generation started for {len(request.chat_ids)} conversations",
    )
