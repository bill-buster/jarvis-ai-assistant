"""Task management API endpoints.

Provides endpoints for managing background tasks in the task queue.
Supports creating, listing, cancelling, and monitoring tasks.
"""

from datetime import datetime
from enum import Enum

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from jarvis.tasks import (
    Task,
    TaskStatus,
    TaskType,
    get_task_queue,
    get_worker,
    start_worker,
)

router = APIRouter(prefix="/tasks", tags=["tasks"])


# =============================================================================
# Pydantic Schemas
# =============================================================================


class TaskStatusEnum(str, Enum):
    """Task status enumeration for API."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskTypeEnum(str, Enum):
    """Task type enumeration for API."""

    BATCH_EXPORT = "batch_export"
    BATCH_SUMMARIZE = "batch_summarize"
    BATCH_GENERATE_REPLIES = "batch_generate_replies"
    SINGLE_EXPORT = "single_export"
    SINGLE_SUMMARIZE = "single_summarize"
    SINGLE_GENERATE_REPLY = "single_generate_reply"


class TaskProgressResponse(BaseModel):
    """Task progress information.

    Example:
        ```json
        {
            "current": 5,
            "total": 10,
            "message": "Exporting chat123",
            "percent": 50.0
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "current": 5,
                "total": 10,
                "message": "Exporting chat123",
                "percent": 50.0,
            }
        }
    )

    current: int = Field(..., ge=0, description="Current item index")
    total: int = Field(..., ge=0, description="Total items to process")
    message: str = Field(default="", description="Progress message")
    percent: float = Field(..., ge=0, le=100, description="Completion percentage")


class TaskResultResponse(BaseModel):
    """Task result information.

    Example:
        ```json
        {
            "success": true,
            "data": {"exports": [...]},
            "error": null,
            "items_processed": 10,
            "items_failed": 0
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "data": {"exports": []},
                "error": None,
                "items_processed": 10,
                "items_failed": 0,
            }
        }
    )

    success: bool = Field(..., description="Whether the task completed successfully")
    data: dict | list | None = Field(default=None, description="Result data")
    error: str | None = Field(default=None, description="Error message if failed")
    items_processed: int = Field(default=0, ge=0, description="Successfully processed items")
    items_failed: int = Field(default=0, ge=0, description="Failed items")


class TaskResponse(BaseModel):
    """Task information response.

    Example:
        ```json
        {
            "id": "abc123",
            "task_type": "batch_export",
            "status": "running",
            "params": {"chat_ids": ["chat1"]},
            "progress": {"current": 5, "total": 10, "percent": 50.0},
            "result": null,
            "created_at": "2024-01-15T10:00:00Z",
            "started_at": "2024-01-15T10:00:01Z",
            "completed_at": null,
            "error_message": null,
            "duration_seconds": 5.5
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "abc123",
                "task_type": "batch_export",
                "status": "running",
                "params": {"chat_ids": ["chat1"]},
                "progress": {"current": 5, "total": 10, "message": "", "percent": 50.0},
                "result": None,
                "created_at": "2024-01-15T10:00:00Z",
                "started_at": "2024-01-15T10:00:01Z",
                "completed_at": None,
                "error_message": None,
                "duration_seconds": 5.5,
            }
        }
    )

    id: str = Field(..., description="Unique task identifier")
    task_type: TaskTypeEnum = Field(..., description="Type of task")
    status: TaskStatusEnum = Field(..., description="Current task status")
    params: dict = Field(default_factory=dict, description="Task parameters")
    progress: TaskProgressResponse = Field(..., description="Progress information")
    result: TaskResultResponse | None = Field(default=None, description="Task result")
    created_at: datetime = Field(..., description="Task creation time")
    started_at: datetime | None = Field(default=None, description="Task start time")
    completed_at: datetime | None = Field(default=None, description="Task completion time")
    error_message: str | None = Field(default=None, description="Error message if failed")
    duration_seconds: float | None = Field(default=None, description="Task duration in seconds")


class TaskCreateRequest(BaseModel):
    """Request to create a new task.

    Example:
        ```json
        {
            "task_type": "batch_export",
            "params": {"chat_ids": ["chat1", "chat2"], "format": "json"}
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "task_type": "batch_export",
                "params": {"chat_ids": ["chat1", "chat2"], "format": "json"},
            }
        }
    )

    task_type: TaskTypeEnum = Field(..., description="Type of task to create")
    params: dict = Field(default_factory=dict, description="Task parameters")


class TaskListResponse(BaseModel):
    """List of tasks response.

    Example:
        ```json
        {
            "tasks": [...],
            "total": 10,
            "pending": 2,
            "running": 1,
            "completed": 5,
            "failed": 2
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "tasks": [],
                "total": 10,
                "pending": 2,
                "running": 1,
                "completed": 5,
                "failed": 2,
            }
        }
    )

    tasks: list[TaskResponse] = Field(default_factory=list, description="List of tasks")
    total: int = Field(..., ge=0, description="Total number of tasks")
    pending: int = Field(default=0, ge=0, description="Number of pending tasks")
    running: int = Field(default=0, ge=0, description="Number of running tasks")
    completed: int = Field(default=0, ge=0, description="Number of completed tasks")
    failed: int = Field(default=0, ge=0, description="Number of failed tasks")


class TaskQueueStatsResponse(BaseModel):
    """Task queue statistics response.

    Example:
        ```json
        {
            "total": 10,
            "by_status": {"pending": 2, "completed": 5},
            "by_type": {"batch_export": 3},
            "worker_running": true
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total": 10,
                "by_status": {"pending": 2, "completed": 5},
                "by_type": {"batch_export": 3},
                "worker_running": True,
            }
        }
    )

    total: int = Field(..., ge=0, description="Total number of tasks")
    by_status: dict[str, int] = Field(default_factory=dict, description="Task counts by status")
    by_type: dict[str, int] = Field(default_factory=dict, description="Task counts by type")
    worker_running: bool = Field(..., description="Whether the worker is running")


# =============================================================================
# Helper Functions
# =============================================================================


def _task_to_response(task: Task) -> TaskResponse:
    """Convert a Task to TaskResponse."""
    progress = TaskProgressResponse(
        current=task.progress.current,
        total=task.progress.total,
        message=task.progress.message,
        percent=task.progress.percent,
    )

    result = None
    if task.result:
        result = TaskResultResponse(
            success=task.result.success,
            data=task.result.data,
            error=task.result.error,
            items_processed=task.result.items_processed,
            items_failed=task.result.items_failed,
        )

    return TaskResponse(
        id=task.id,
        task_type=TaskTypeEnum(task.task_type.value),
        status=TaskStatusEnum(task.status.value),
        params=task.params,
        progress=progress,
        result=result,
        created_at=task.created_at,
        started_at=task.started_at,
        completed_at=task.completed_at,
        error_message=task.error_message,
        duration_seconds=task.duration_seconds,
    )


# =============================================================================
# API Endpoints
# =============================================================================


@router.post("", response_model=TaskResponse)
def create_task(request: TaskCreateRequest) -> TaskResponse:
    """Create a new background task.

    Enqueues a task for background processing. The task will be picked up
    by the worker and processed asynchronously.

    Args:
        request: Task creation request with type and parameters.

    Returns:
        Created task with initial status.
    """
    # Ensure worker is running
    worker = get_worker()
    if not worker.is_running:
        start_worker()

    queue = get_task_queue()
    task = queue.enqueue(
        task_type=TaskType(request.task_type.value),
        params=request.params,
    )

    return _task_to_response(task)


@router.get("", response_model=TaskListResponse)
def list_tasks(
    status: TaskStatusEnum | None = None,
    task_type: TaskTypeEnum | None = None,
    limit: int = 50,
) -> TaskListResponse:
    """List all tasks in the queue.

    Args:
        status: Filter by task status.
        task_type: Filter by task type.
        limit: Maximum number of tasks to return.

    Returns:
        List of tasks with summary counts.
    """
    queue = get_task_queue()

    # Convert enum params
    status_filter = TaskStatus(status.value) if status else None
    type_filter = TaskType(task_type.value) if task_type else None

    tasks = queue.get_all(status=status_filter, task_type=type_filter, limit=limit)

    # Get counts
    all_tasks = queue.get_all()
    pending = sum(1 for t in all_tasks if t.status == TaskStatus.PENDING)
    running = sum(1 for t in all_tasks if t.status == TaskStatus.RUNNING)
    completed = sum(1 for t in all_tasks if t.status == TaskStatus.COMPLETED)
    failed = sum(1 for t in all_tasks if t.status == TaskStatus.FAILED)

    return TaskListResponse(
        tasks=[_task_to_response(t) for t in tasks],
        total=len(all_tasks),
        pending=pending,
        running=running,
        completed=completed,
        failed=failed,
    )


@router.get("/stats", response_model=TaskQueueStatsResponse)
def get_stats() -> TaskQueueStatsResponse:
    """Get task queue statistics.

    Returns:
        Queue statistics including counts by status and type.
    """
    queue = get_task_queue()
    stats = queue.get_stats()
    worker = get_worker()

    return TaskQueueStatsResponse(
        total=stats["total"],
        by_status=stats["by_status"],
        by_type=stats["by_type"],
        worker_running=worker.is_running,
    )


@router.get("/{task_id}", response_model=TaskResponse)
def get_task(task_id: str) -> TaskResponse:
    """Get a specific task by ID.

    Args:
        task_id: The task identifier.

    Returns:
        Task details including status and progress.
    """
    queue = get_task_queue()
    task = queue.get(task_id)

    if task is None:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    return _task_to_response(task)


@router.delete("/{task_id}")
def cancel_task(task_id: str) -> dict:
    """Cancel a pending task.

    Only pending tasks can be cancelled. Running tasks will complete.

    Args:
        task_id: The task identifier.

    Returns:
        Confirmation of cancellation.
    """
    queue = get_task_queue()
    task = queue.get(task_id)

    if task is None:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    if task.status != TaskStatus.PENDING:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel task with status '{task.status.value}'. "
            "Only pending tasks can be cancelled.",
        )

    success = queue.cancel(task_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to cancel task")

    return {"success": True, "message": f"Task {task_id} cancelled"}


@router.post("/{task_id}/retry")
def retry_task(task_id: str) -> TaskResponse:
    """Retry a failed task.

    Only failed tasks can be retried, up to the maximum retry limit.

    Args:
        task_id: The task identifier.

    Returns:
        Updated task with pending status.
    """
    queue = get_task_queue()
    task = queue.get(task_id)

    if task is None:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    if not task.can_retry():
        if task.status != TaskStatus.FAILED:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot retry task with status '{task.status.value}'",
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Task has exceeded maximum retries ({task.max_retries})",
            )

    success = queue.retry(task_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to retry task")

    # Refresh task
    task = queue.get(task_id)
    return _task_to_response(task)  # type: ignore[arg-type]


@router.post("/worker/start")
def start_task_worker() -> dict:
    """Start the background task worker.

    Returns:
        Confirmation that the worker was started.
    """
    worker = get_worker()
    if worker.is_running:
        return {"success": True, "message": "Worker is already running"}

    start_worker()
    return {"success": True, "message": "Worker started"}


@router.post("/worker/stop")
def stop_task_worker() -> dict:
    """Stop the background task worker.

    Returns:
        Confirmation that the worker was stopped.
    """
    from jarvis.tasks import stop_worker

    worker = get_worker()
    if not worker.is_running:
        return {"success": True, "message": "Worker is not running"}

    stop_worker()
    return {"success": True, "message": "Worker stopped"}


@router.delete("/completed/clear")
def clear_completed_tasks() -> dict:
    """Clear all completed tasks from the queue.

    Returns:
        Number of tasks removed.
    """
    queue = get_task_queue()
    count = queue.clear_completed()
    return {"success": True, "tasks_removed": count}
