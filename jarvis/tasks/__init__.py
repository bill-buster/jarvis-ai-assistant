"""Task queue system for background operations.

Provides infrastructure for running batch operations and background tasks
with progress tracking, persistence, and error handling.

Components:
    - Task: Data model for a background task
    - TaskQueue: Thread-safe queue with persistence
    - TaskWorker: Background processor for tasks

Usage:
    from jarvis.tasks import (
        get_task_queue,
        start_worker,
        stop_worker,
        Task,
        TaskStatus,
        TaskType,
    )

    # Start the background worker
    start_worker()

    # Enqueue a batch export task
    queue = get_task_queue()
    task = queue.enqueue(
        TaskType.BATCH_EXPORT,
        {"chat_ids": ["chat1", "chat2"], "format": "json"},
    )

    # Check task status
    task = queue.get(task.id)
    print(f"Status: {task.status}, Progress: {task.progress.percent}%")

    # Stop the worker when done
    stop_worker()
"""

from jarvis.tasks.models import (
    Task,
    TaskProgress,
    TaskResult,
    TaskStatus,
    TaskType,
)
from jarvis.tasks.queue import (
    DEFAULT_QUEUE_PATH,
    TaskQueue,
    get_task_queue,
    reset_task_queue,
)
from jarvis.tasks.worker import (
    TaskWorker,
    get_worker,
    reset_worker,
    start_worker,
    stop_worker,
)

__all__ = [
    # Models
    "Task",
    "TaskProgress",
    "TaskResult",
    "TaskStatus",
    "TaskType",
    # Queue
    "TaskQueue",
    "get_task_queue",
    "reset_task_queue",
    "DEFAULT_QUEUE_PATH",
    # Worker
    "TaskWorker",
    "get_worker",
    "reset_worker",
    "start_worker",
    "stop_worker",
]
