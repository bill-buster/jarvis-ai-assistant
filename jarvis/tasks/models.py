"""Task data models for the background task queue.

Defines the task status model and related types for tracking background operations
like batch exports, summarizations, and reply generation.

Usage:
    from jarvis.tasks.models import Task, TaskStatus, TaskType

    task = Task(
        task_type=TaskType.BATCH_EXPORT,
        params={"chat_ids": ["chat1", "chat2"]},
    )
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    """Status of a background task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(str, Enum):
    """Type of background task."""

    BATCH_EXPORT = "batch_export"
    BATCH_SUMMARIZE = "batch_summarize"
    BATCH_GENERATE_REPLIES = "batch_generate_replies"
    SINGLE_EXPORT = "single_export"
    SINGLE_SUMMARIZE = "single_summarize"
    SINGLE_GENERATE_REPLY = "single_generate_reply"


@dataclass
class TaskProgress:
    """Progress information for a running task.

    Attributes:
        current: Current item being processed (0-indexed).
        total: Total number of items to process.
        message: Human-readable progress message.
        percent: Calculated percentage (0-100).
    """

    current: int = 0
    total: int = 0
    message: str = ""

    @property
    def percent(self) -> float:
        """Calculate progress percentage."""
        if self.total <= 0:
            return 0.0
        return min(100.0, (self.current / self.total) * 100.0)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "current": self.current,
            "total": self.total,
            "message": self.message,
            "percent": round(self.percent, 1),
        }


@dataclass
class TaskResult:
    """Result of a completed or failed task.

    Attributes:
        success: Whether the task completed successfully.
        data: Result data for successful tasks.
        error: Error message for failed tasks.
        items_processed: Number of items successfully processed.
        items_failed: Number of items that failed.
        partial_results: Results from successfully processed items (for partial failures).
    """

    success: bool = True
    data: Any = None
    error: str | None = None
    items_processed: int = 0
    items_failed: int = 0
    partial_results: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result: dict[str, Any] = {
            "success": self.success,
            "items_processed": self.items_processed,
            "items_failed": self.items_failed,
        }
        if self.data is not None:
            result["data"] = self.data
        if self.error:
            result["error"] = self.error
        if self.partial_results:
            result["partial_results"] = self.partial_results
        return result


@dataclass
class Task:
    """A background task with status and progress tracking.

    Attributes:
        id: Unique task identifier.
        task_type: Type of task being performed.
        status: Current task status.
        params: Task parameters (task-type specific).
        progress: Progress information for running tasks.
        result: Result data for completed/failed tasks.
        created_at: When the task was created.
        started_at: When the task started running.
        completed_at: When the task completed (success or failure).
        error_message: Error message if the task failed.
        retry_count: Number of times this task has been retried.
        max_retries: Maximum number of retries allowed.
    """

    task_type: TaskType
    params: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: TaskStatus = TaskStatus.PENDING
    progress: TaskProgress = field(default_factory=TaskProgress)
    result: TaskResult | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    retry_count: int = 0
    max_retries: int = 3

    def start(self) -> None:
        """Mark the task as running."""
        self.status = TaskStatus.RUNNING
        self.started_at = datetime.now(UTC)
        self.progress = TaskProgress(current=0, total=0, message="Starting...")

    def update_progress(self, current: int, total: int, message: str = "") -> None:
        """Update task progress.

        Args:
            current: Current item index (0-indexed).
            total: Total number of items.
            message: Human-readable progress message.
        """
        self.progress = TaskProgress(current=current, total=total, message=message)

    def complete(self, result: TaskResult) -> None:
        """Mark the task as completed.

        Args:
            result: The task result.
        """
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now(UTC)
        self.result = result
        self.progress = TaskProgress(
            current=result.items_processed,
            total=result.items_processed + result.items_failed,
            message="Completed",
        )

    def fail(self, error: str, result: TaskResult | None = None) -> None:
        """Mark the task as failed.

        Args:
            error: Error message.
            result: Optional partial result.
        """
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.now(UTC)
        self.error_message = error
        self.result = result or TaskResult(success=False, error=error)
        self.progress.message = f"Failed: {error}"

    def cancel(self) -> None:
        """Mark the task as cancelled."""
        self.status = TaskStatus.CANCELLED
        self.completed_at = datetime.now(UTC)
        self.progress.message = "Cancelled"

    def can_retry(self) -> bool:
        """Check if the task can be retried."""
        return self.status == TaskStatus.FAILED and self.retry_count < self.max_retries

    def retry(self) -> None:
        """Prepare the task for retry."""
        if not self.can_retry():
            raise ValueError("Task cannot be retried")
        self.retry_count += 1
        self.status = TaskStatus.PENDING
        self.started_at = None
        self.completed_at = None
        self.error_message = None
        self.progress = TaskProgress()

    @property
    def is_terminal(self) -> bool:
        """Check if the task is in a terminal state."""
        return self.status in (
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        )

    @property
    def duration_seconds(self) -> float | None:
        """Get task duration in seconds."""
        if self.started_at is None:
            return None
        end_time = self.completed_at or datetime.now(UTC)
        return (end_time - self.started_at).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Convert task to dictionary for serialization.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "id": self.id,
            "task_type": self.task_type.value,
            "status": self.status.value,
            "params": self.params,
            "progress": self.progress.to_dict(),
            "result": self.result.to_dict() if self.result else None,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "duration_seconds": self.duration_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Task:
        """Create a Task from a dictionary.

        Args:
            data: Dictionary with task data.

        Returns:
            Task instance.
        """
        # Parse datetime fields
        created_at = datetime.fromisoformat(data["created_at"])
        started_at = datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None
        completed_at = (
            datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None
        )

        # Parse progress
        progress_data = data.get("progress", {})
        progress = TaskProgress(
            current=progress_data.get("current", 0),
            total=progress_data.get("total", 0),
            message=progress_data.get("message", ""),
        )

        # Parse result
        result = None
        if data.get("result"):
            result_data = data["result"]
            result = TaskResult(
                success=result_data.get("success", True),
                data=result_data.get("data"),
                error=result_data.get("error"),
                items_processed=result_data.get("items_processed", 0),
                items_failed=result_data.get("items_failed", 0),
                partial_results=result_data.get("partial_results", []),
            )

        return cls(
            id=data["id"],
            task_type=TaskType(data["task_type"]),
            status=TaskStatus(data["status"]),
            params=data.get("params", {}),
            progress=progress,
            result=result,
            created_at=created_at,
            started_at=started_at,
            completed_at=completed_at,
            error_message=data.get("error_message"),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
        )


# Export all public symbols
__all__ = [
    "Task",
    "TaskProgress",
    "TaskResult",
    "TaskStatus",
    "TaskType",
]
