"""In-memory task queue with persistence support.

Provides a thread-safe task queue for managing background operations.
Tasks are persisted to disk for recovery after restart.

Usage:
    from jarvis.tasks import get_task_queue

    queue = get_task_queue()
    task = queue.enqueue(TaskType.BATCH_EXPORT, {"chat_ids": ["chat1"]})
    status = queue.get(task.id)
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from jarvis.tasks.models import Task, TaskStatus, TaskType

logger = logging.getLogger(__name__)

# Default persistence path
DEFAULT_QUEUE_PATH = Path.home() / ".jarvis" / "task_queue.json"


class TaskQueue:
    """Thread-safe in-memory task queue with persistence.

    Manages background tasks with support for:
    - Adding and retrieving tasks
    - Status updates and progress tracking
    - Persistence to disk
    - Task cancellation
    - Automatic cleanup of old completed tasks

    Attributes:
        max_completed_tasks: Maximum number of completed tasks to keep in memory.
        persistence_path: Path to the persistence file.
    """

    def __init__(
        self,
        persistence_path: Path | None = None,
        max_completed_tasks: int = 100,
        auto_persist: bool = True,
    ) -> None:
        """Initialize the task queue.

        Args:
            persistence_path: Path to persist queue state. Defaults to ~/.jarvis/task_queue.json.
            max_completed_tasks: Maximum completed tasks to retain.
            auto_persist: Whether to auto-persist on changes.
        """
        self._tasks: dict[str, Task] = {}
        self._lock = threading.RLock()
        self._persistence_path = persistence_path or DEFAULT_QUEUE_PATH
        self._max_completed_tasks = max_completed_tasks
        self._auto_persist = auto_persist
        self._callbacks: dict[str, list[Callable[[Task], None]]] = {}

        # Load persisted state
        self._load()

    def enqueue(
        self,
        task_type: TaskType,
        params: dict[str, Any] | None = None,
        max_retries: int = 3,
    ) -> Task:
        """Add a new task to the queue.

        Args:
            task_type: Type of task to create.
            params: Task-specific parameters.
            max_retries: Maximum retry attempts on failure.

        Returns:
            The created task.
        """
        task = Task(
            task_type=task_type,
            params=params or {},
            max_retries=max_retries,
        )

        with self._lock:
            self._tasks[task.id] = task
            self._cleanup_old_tasks()
            if self._auto_persist:
                self._persist()

        logger.info(f"Task enqueued: {task.id} ({task_type.value})")
        return task

    def get(self, task_id: str) -> Task | None:
        """Get a task by ID.

        Args:
            task_id: The task identifier.

        Returns:
            The task if found, None otherwise.
        """
        with self._lock:
            return self._tasks.get(task_id)

    def get_all(
        self,
        status: TaskStatus | None = None,
        task_type: TaskType | None = None,
        limit: int | None = None,
    ) -> list[Task]:
        """Get all tasks, optionally filtered.

        Args:
            status: Filter by status.
            task_type: Filter by task type.
            limit: Maximum number of tasks to return.

        Returns:
            List of matching tasks, newest first.
        """
        with self._lock:
            tasks = list(self._tasks.values())

        # Apply filters
        if status is not None:
            tasks = [t for t in tasks if t.status == status]
        if task_type is not None:
            tasks = [t for t in tasks if t.task_type == task_type]

        # Sort by creation time (newest first)
        tasks.sort(key=lambda t: t.created_at, reverse=True)

        # Apply limit
        if limit is not None:
            tasks = tasks[:limit]

        return tasks

    def get_pending(self) -> list[Task]:
        """Get all pending tasks in order.

        Returns:
            List of pending tasks, oldest first (FIFO order).
        """
        with self._lock:
            pending = [t for t in self._tasks.values() if t.status == TaskStatus.PENDING]
            pending.sort(key=lambda t: t.created_at)  # Oldest first
            return pending

    def get_next_pending(self) -> Task | None:
        """Get the next pending task to process.

        Returns:
            The oldest pending task, or None if no pending tasks.
        """
        pending = self.get_pending()
        return pending[0] if pending else None

    def update(self, task: Task) -> None:
        """Update a task in the queue.

        Args:
            task: The task to update.
        """
        with self._lock:
            if task.id not in self._tasks:
                logger.warning(f"Attempted to update unknown task: {task.id}")
                return
            self._tasks[task.id] = task
            if self._auto_persist:
                self._persist()

        # Notify callbacks
        self._notify_callbacks(task)

    def cancel(self, task_id: str) -> bool:
        """Cancel a pending task.

        Args:
            task_id: The task identifier.

        Returns:
            True if cancelled, False if task not found or not cancellable.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False

            if task.status != TaskStatus.PENDING:
                logger.warning(f"Cannot cancel task {task_id}: status is {task.status.value}")
                return False

            task.cancel()
            if self._auto_persist:
                self._persist()

        self._notify_callbacks(task)
        logger.info(f"Task cancelled: {task_id}")
        return True

    def retry(self, task_id: str) -> bool:
        """Retry a failed task.

        Args:
            task_id: The task identifier.

        Returns:
            True if retry scheduled, False otherwise.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False

            if not task.can_retry():
                logger.warning(f"Cannot retry task {task_id}")
                return False

            task.retry()
            if self._auto_persist:
                self._persist()

        logger.info(f"Task queued for retry: {task_id} (attempt {task.retry_count})")
        return True

    def delete(self, task_id: str) -> bool:
        """Delete a task from the queue.

        Only terminal tasks (completed, failed, cancelled) can be deleted.

        Args:
            task_id: The task identifier.

        Returns:
            True if deleted, False otherwise.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False

            if not task.is_terminal:
                logger.warning(
                    f"Cannot delete non-terminal task {task_id}: status is {task.status.value}"
                )
                return False

            del self._tasks[task_id]
            if self._auto_persist:
                self._persist()

        logger.info(f"Task deleted: {task_id}")
        return True

    def clear_completed(self) -> int:
        """Remove all completed tasks.

        Returns:
            Number of tasks removed.
        """
        with self._lock:
            to_remove = [
                task_id
                for task_id, task in self._tasks.items()
                if task.status == TaskStatus.COMPLETED
            ]
            for task_id in to_remove:
                del self._tasks[task_id]
            if self._auto_persist and to_remove:
                self._persist()

        logger.info(f"Cleared {len(to_remove)} completed tasks")
        return len(to_remove)

    def register_callback(self, task_id: str, callback: Callable[[Task], None]) -> None:
        """Register a callback for task updates.

        Args:
            task_id: The task to watch.
            callback: Function to call on task updates.
        """
        with self._lock:
            if task_id not in self._callbacks:
                self._callbacks[task_id] = []
            self._callbacks[task_id].append(callback)

    def unregister_callback(self, task_id: str, callback: Callable[[Task], None]) -> None:
        """Unregister a callback for task updates.

        Args:
            task_id: The task being watched.
            callback: The callback to remove.
        """
        with self._lock:
            if task_id in self._callbacks:
                try:
                    self._callbacks[task_id].remove(callback)
                    if not self._callbacks[task_id]:
                        del self._callbacks[task_id]
                except ValueError:
                    pass

    def _notify_callbacks(self, task: Task) -> None:
        """Notify registered callbacks of a task update."""
        with self._lock:
            callbacks = list(self._callbacks.get(task.id, []))

        for callback in callbacks:
            try:
                callback(task)
            except Exception as e:
                logger.exception(f"Error in task callback: {e}")

    def _cleanup_old_tasks(self) -> None:
        """Remove old completed tasks to stay within limits."""
        completed = [t for t in self._tasks.values() if t.status == TaskStatus.COMPLETED]

        if len(completed) > self._max_completed_tasks:
            # Sort by completion time, oldest first
            completed.sort(key=lambda t: t.completed_at or t.created_at)
            to_remove = len(completed) - self._max_completed_tasks

            for task in completed[:to_remove]:
                del self._tasks[task.id]

            logger.debug(f"Cleaned up {to_remove} old completed tasks")

    def _persist(self) -> None:
        """Persist queue state to disk."""
        try:
            self._persistence_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "version": 1,
                "tasks": [task.to_dict() for task in self._tasks.values()],
            }

            with self._persistence_path.open("w") as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Persisted {len(self._tasks)} tasks to {self._persistence_path}")

        except Exception as e:
            logger.error(f"Failed to persist task queue: {e}")

    def _load(self) -> None:
        """Load queue state from disk."""
        if not self._persistence_path.exists():
            logger.debug(f"No persisted queue at {self._persistence_path}")
            return

        try:
            with self._persistence_path.open() as f:
                data = json.load(f)

            version = data.get("version", 1)
            if version != 1:
                logger.warning(f"Unknown queue format version: {version}")
                return

            tasks_data = data.get("tasks", [])
            for task_data in tasks_data:
                try:
                    task = Task.from_dict(task_data)

                    # Reset running tasks to pending (they were interrupted)
                    if task.status == TaskStatus.RUNNING:
                        task.status = TaskStatus.PENDING
                        task.started_at = None

                    self._tasks[task.id] = task
                except Exception as e:
                    logger.warning(f"Failed to load task: {e}")

            logger.info(f"Loaded {len(self._tasks)} tasks from {self._persistence_path}")

        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in task queue file: {e}")
        except Exception as e:
            logger.error(f"Failed to load task queue: {e}")

    def persist(self) -> None:
        """Manually persist queue state."""
        with self._lock:
            self._persist()

    def get_stats(self) -> dict[str, Any]:
        """Get queue statistics.

        Returns:
            Dictionary with queue statistics.
        """
        with self._lock:
            tasks = list(self._tasks.values())

        by_status: dict[str, int] = {}
        by_type: dict[str, int] = {}

        for status in TaskStatus:
            count = sum(1 for t in tasks if t.status == status)
            if count > 0:
                by_status[status.value] = count

        for task_type in TaskType:
            count = sum(1 for t in tasks if t.task_type == task_type)
            if count > 0:
                by_type[task_type.value] = count

        return {
            "total": len(tasks),
            "by_status": by_status,
            "by_type": by_type,
        }


# Module-level singleton
_queue: TaskQueue | None = None
_queue_lock = threading.Lock()


def get_task_queue() -> TaskQueue:
    """Get the singleton task queue instance.

    Returns:
        Shared TaskQueue instance.
    """
    global _queue
    if _queue is None:
        with _queue_lock:
            if _queue is None:
                _queue = TaskQueue()
    return _queue


def reset_task_queue() -> None:
    """Reset the singleton task queue (for testing)."""
    global _queue
    with _queue_lock:
        _queue = None


# Export all public symbols
__all__ = [
    "TaskQueue",
    "get_task_queue",
    "reset_task_queue",
    "DEFAULT_QUEUE_PATH",
]
