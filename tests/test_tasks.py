"""Tests for the task queue system.

Tests cover:
- Task lifecycle (creation, status updates, completion, failure)
- Queue operations (enqueue, get, cancel, retry)
- Persistence (save and load)
- Worker (task processing)
- Error handling and retries
- Concurrent access
"""

from __future__ import annotations

import json
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from jarvis.tasks.models import (
    Task,
    TaskProgress,
    TaskResult,
    TaskStatus,
    TaskType,
)
from jarvis.tasks.queue import TaskQueue, get_task_queue, reset_task_queue
from jarvis.tasks.worker import TaskWorker, get_worker, reset_worker


class TestTaskModels:
    """Tests for task data models."""

    def test_task_creation(self) -> None:
        """Test creating a new task."""
        task = Task(
            task_type=TaskType.BATCH_EXPORT,
            params={"chat_ids": ["chat1", "chat2"]},
        )

        assert task.id is not None
        assert task.task_type == TaskType.BATCH_EXPORT
        assert task.status == TaskStatus.PENDING
        assert task.params == {"chat_ids": ["chat1", "chat2"]}
        assert task.created_at is not None
        assert task.started_at is None
        assert task.completed_at is None

    def test_task_start(self) -> None:
        """Test starting a task."""
        task = Task(task_type=TaskType.BATCH_EXPORT)
        task.start()

        assert task.status == TaskStatus.RUNNING
        assert task.started_at is not None
        assert task.progress.message == "Starting..."

    def test_task_update_progress(self) -> None:
        """Test updating task progress."""
        task = Task(task_type=TaskType.BATCH_EXPORT)
        task.start()
        task.update_progress(5, 10, "Processing chat 5")

        assert task.progress.current == 5
        assert task.progress.total == 10
        assert task.progress.message == "Processing chat 5"
        assert task.progress.percent == 50.0

    def test_task_complete(self) -> None:
        """Test completing a task."""
        task = Task(task_type=TaskType.BATCH_EXPORT)
        task.start()

        result = TaskResult(
            success=True,
            data={"exports": ["file1.json", "file2.json"]},
            items_processed=2,
            items_failed=0,
        )
        task.complete(result)

        assert task.status == TaskStatus.COMPLETED
        assert task.completed_at is not None
        assert task.result == result
        assert task.result.success is True

    def test_task_fail(self) -> None:
        """Test failing a task."""
        task = Task(task_type=TaskType.BATCH_EXPORT)
        task.start()
        task.fail("Database connection error")

        assert task.status == TaskStatus.FAILED
        assert task.completed_at is not None
        assert task.error_message == "Database connection error"
        assert task.result is not None
        assert task.result.success is False

    def test_task_cancel(self) -> None:
        """Test cancelling a task."""
        task = Task(task_type=TaskType.BATCH_EXPORT)
        task.cancel()

        assert task.status == TaskStatus.CANCELLED
        assert task.completed_at is not None

    def test_task_retry(self) -> None:
        """Test retrying a failed task."""
        task = Task(task_type=TaskType.BATCH_EXPORT, max_retries=3)
        task.start()
        task.fail("Temporary error")

        assert task.can_retry() is True
        assert task.retry_count == 0

        task.retry()

        assert task.status == TaskStatus.PENDING
        assert task.retry_count == 1
        assert task.started_at is None
        assert task.error_message is None

    def test_task_max_retries(self) -> None:
        """Test that retry limit is enforced."""
        task = Task(task_type=TaskType.BATCH_EXPORT, max_retries=2)

        # Fail and retry twice
        for _ in range(2):
            task.start()
            task.fail("Error")
            task.retry()

        # Third failure should not be retryable
        task.start()
        task.fail("Error")

        assert task.can_retry() is False
        assert task.retry_count == 2

    def test_task_is_terminal(self) -> None:
        """Test terminal state detection."""
        task = Task(task_type=TaskType.BATCH_EXPORT)

        assert task.is_terminal is False

        task.status = TaskStatus.RUNNING
        assert task.is_terminal is False

        task.status = TaskStatus.COMPLETED
        assert task.is_terminal is True

        task.status = TaskStatus.FAILED
        assert task.is_terminal is True

        task.status = TaskStatus.CANCELLED
        assert task.is_terminal is True

    def test_task_duration(self) -> None:
        """Test task duration calculation."""
        task = Task(task_type=TaskType.BATCH_EXPORT)

        assert task.duration_seconds is None

        task.start()
        time.sleep(0.1)

        # Should have some duration while running
        assert task.duration_seconds is not None
        assert task.duration_seconds > 0

        task.complete(TaskResult(success=True, items_processed=1))

        # Duration should be frozen after completion
        final_duration = task.duration_seconds
        time.sleep(0.1)
        assert task.duration_seconds == final_duration

    def test_task_serialization(self) -> None:
        """Test task serialization to dict."""
        task = Task(
            task_type=TaskType.BATCH_EXPORT,
            params={"chat_ids": ["chat1"]},
        )
        task.start()
        task.update_progress(1, 2, "In progress")
        task.complete(TaskResult(success=True, items_processed=2))

        data = task.to_dict()

        assert data["id"] == task.id
        assert data["task_type"] == "batch_export"
        assert data["status"] == "completed"
        assert data["params"] == {"chat_ids": ["chat1"]}
        assert data["progress"]["current"] == 2
        assert data["result"]["success"] is True

    def test_task_deserialization(self) -> None:
        """Test task deserialization from dict."""
        original = Task(
            task_type=TaskType.BATCH_SUMMARIZE,
            params={"chat_ids": ["chat1", "chat2"]},
        )
        original.start()
        original.complete(TaskResult(success=True, items_processed=2))

        data = original.to_dict()
        restored = Task.from_dict(data)

        assert restored.id == original.id
        assert restored.task_type == original.task_type
        assert restored.status == original.status
        assert restored.params == original.params
        assert restored.result is not None
        assert restored.result.success is True

    def test_progress_percent_calculation(self) -> None:
        """Test progress percentage calculation."""
        progress = TaskProgress(current=0, total=0)
        assert progress.percent == 0.0

        progress = TaskProgress(current=5, total=10)
        assert progress.percent == 50.0

        progress = TaskProgress(current=10, total=10)
        assert progress.percent == 100.0

        # Should cap at 100
        progress = TaskProgress(current=15, total=10)
        assert progress.percent == 100.0


class TestTaskQueue:
    """Tests for the task queue."""

    @pytest.fixture(autouse=True)
    def reset_singletons(self) -> None:
        """Reset singletons before each test."""
        reset_task_queue()
        reset_worker()

    @pytest.fixture
    def temp_queue(self) -> TaskQueue:
        """Create a queue with a temporary persistence file."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        queue = TaskQueue(persistence_path=temp_path, auto_persist=False)
        yield queue

        # Cleanup
        if temp_path.exists():
            temp_path.unlink()

    def test_enqueue(self, temp_queue: TaskQueue) -> None:
        """Test enqueueing a task."""
        task = temp_queue.enqueue(
            TaskType.BATCH_EXPORT,
            params={"chat_ids": ["chat1"]},
        )

        assert task is not None
        assert task.status == TaskStatus.PENDING
        assert task.params == {"chat_ids": ["chat1"]}

        # Should be retrievable
        retrieved = temp_queue.get(task.id)
        assert retrieved is not None
        assert retrieved.id == task.id

    def test_get_nonexistent(self, temp_queue: TaskQueue) -> None:
        """Test getting a nonexistent task."""
        result = temp_queue.get("nonexistent-id")
        assert result is None

    def test_get_all(self, temp_queue: TaskQueue) -> None:
        """Test getting all tasks."""
        task1 = temp_queue.enqueue(TaskType.BATCH_EXPORT)
        task2 = temp_queue.enqueue(TaskType.BATCH_SUMMARIZE)

        tasks = temp_queue.get_all()
        assert len(tasks) == 2

        # Should be newest first
        assert tasks[0].id == task2.id
        assert tasks[1].id == task1.id

    def test_get_all_filtered_by_status(self, temp_queue: TaskQueue) -> None:
        """Test filtering tasks by status."""
        task1 = temp_queue.enqueue(TaskType.BATCH_EXPORT)
        task2 = temp_queue.enqueue(TaskType.BATCH_EXPORT)

        task1.status = TaskStatus.COMPLETED
        temp_queue.update(task1)

        pending = temp_queue.get_all(status=TaskStatus.PENDING)
        completed = temp_queue.get_all(status=TaskStatus.COMPLETED)

        assert len(pending) == 1
        assert len(completed) == 1
        assert pending[0].id == task2.id
        assert completed[0].id == task1.id

    def test_get_all_filtered_by_type(self, temp_queue: TaskQueue) -> None:
        """Test filtering tasks by type."""
        temp_queue.enqueue(TaskType.BATCH_EXPORT)
        temp_queue.enqueue(TaskType.BATCH_SUMMARIZE)

        exports = temp_queue.get_all(task_type=TaskType.BATCH_EXPORT)
        summaries = temp_queue.get_all(task_type=TaskType.BATCH_SUMMARIZE)

        assert len(exports) == 1
        assert len(summaries) == 1

    def test_get_pending(self, temp_queue: TaskQueue) -> None:
        """Test getting pending tasks in FIFO order."""
        task1 = temp_queue.enqueue(TaskType.BATCH_EXPORT)
        time.sleep(0.01)  # Ensure different timestamps
        task2 = temp_queue.enqueue(TaskType.BATCH_EXPORT)

        pending = temp_queue.get_pending()

        # Should be oldest first
        assert len(pending) == 2
        assert pending[0].id == task1.id
        assert pending[1].id == task2.id

    def test_get_next_pending(self, temp_queue: TaskQueue) -> None:
        """Test getting the next pending task."""
        task1 = temp_queue.enqueue(TaskType.BATCH_EXPORT)
        time.sleep(0.01)
        temp_queue.enqueue(TaskType.BATCH_EXPORT)

        next_task = temp_queue.get_next_pending()
        assert next_task is not None
        assert next_task.id == task1.id

    def test_get_next_pending_empty(self, temp_queue: TaskQueue) -> None:
        """Test getting next pending from empty queue."""
        result = temp_queue.get_next_pending()
        assert result is None

    def test_update(self, temp_queue: TaskQueue) -> None:
        """Test updating a task."""
        task = temp_queue.enqueue(TaskType.BATCH_EXPORT)

        task.start()
        task.update_progress(1, 2, "In progress")
        temp_queue.update(task)

        retrieved = temp_queue.get(task.id)
        assert retrieved is not None
        assert retrieved.status == TaskStatus.RUNNING
        assert retrieved.progress.current == 1

    def test_cancel(self, temp_queue: TaskQueue) -> None:
        """Test cancelling a pending task."""
        task = temp_queue.enqueue(TaskType.BATCH_EXPORT)

        success = temp_queue.cancel(task.id)
        assert success is True

        retrieved = temp_queue.get(task.id)
        assert retrieved is not None
        assert retrieved.status == TaskStatus.CANCELLED

    def test_cancel_nonexistent(self, temp_queue: TaskQueue) -> None:
        """Test cancelling a nonexistent task."""
        success = temp_queue.cancel("nonexistent-id")
        assert success is False

    def test_cancel_running_task(self, temp_queue: TaskQueue) -> None:
        """Test that running tasks cannot be cancelled."""
        task = temp_queue.enqueue(TaskType.BATCH_EXPORT)
        task.start()
        temp_queue.update(task)

        success = temp_queue.cancel(task.id)
        assert success is False

        # Task should still be running
        retrieved = temp_queue.get(task.id)
        assert retrieved is not None
        assert retrieved.status == TaskStatus.RUNNING

    def test_retry(self, temp_queue: TaskQueue) -> None:
        """Test retrying a failed task."""
        task = temp_queue.enqueue(TaskType.BATCH_EXPORT, max_retries=3)
        task.start()
        task.fail("Error")
        temp_queue.update(task)

        success = temp_queue.retry(task.id)
        assert success is True

        retrieved = temp_queue.get(task.id)
        assert retrieved is not None
        assert retrieved.status == TaskStatus.PENDING
        assert retrieved.retry_count == 1

    def test_delete(self, temp_queue: TaskQueue) -> None:
        """Test deleting a terminal task."""
        task = temp_queue.enqueue(TaskType.BATCH_EXPORT)
        task.complete(TaskResult(success=True, items_processed=1))
        temp_queue.update(task)

        success = temp_queue.delete(task.id)
        assert success is True

        retrieved = temp_queue.get(task.id)
        assert retrieved is None

    def test_delete_running_task(self, temp_queue: TaskQueue) -> None:
        """Test that running tasks cannot be deleted."""
        task = temp_queue.enqueue(TaskType.BATCH_EXPORT)
        task.start()
        temp_queue.update(task)

        success = temp_queue.delete(task.id)
        assert success is False

        # Task should still exist
        retrieved = temp_queue.get(task.id)
        assert retrieved is not None

    def test_clear_completed(self, temp_queue: TaskQueue) -> None:
        """Test clearing completed tasks."""
        task1 = temp_queue.enqueue(TaskType.BATCH_EXPORT)
        task2 = temp_queue.enqueue(TaskType.BATCH_EXPORT)
        task3 = temp_queue.enqueue(TaskType.BATCH_EXPORT)

        task1.complete(TaskResult(success=True, items_processed=1))
        task2.complete(TaskResult(success=True, items_processed=1))
        temp_queue.update(task1)
        temp_queue.update(task2)

        count = temp_queue.clear_completed()
        assert count == 2

        # Only pending task should remain
        tasks = temp_queue.get_all()
        assert len(tasks) == 1
        assert tasks[0].id == task3.id

    def test_persistence_save(self, temp_queue: TaskQueue) -> None:
        """Test persisting queue state."""
        task = temp_queue.enqueue(
            TaskType.BATCH_EXPORT,
            params={"chat_ids": ["chat1"]},
        )

        temp_queue.persist()

        # Read the persisted file
        with temp_queue._persistence_path.open() as f:
            data = json.load(f)

        assert data["version"] == 1
        assert len(data["tasks"]) == 1
        assert data["tasks"][0]["id"] == task.id

    def test_persistence_load(self) -> None:
        """Test loading queue state from disk."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Create initial queue and persist
            queue1 = TaskQueue(persistence_path=temp_path)
            task = queue1.enqueue(TaskType.BATCH_EXPORT)
            task.start()
            task.update_progress(5, 10, "Progress")
            queue1.update(task)
            queue1.persist()

            # Create new queue from persisted state
            queue2 = TaskQueue(persistence_path=temp_path)

            loaded = queue2.get(task.id)
            assert loaded is not None
            # Running tasks should be reset to pending on load
            assert loaded.status == TaskStatus.PENDING
            assert loaded.progress.current == 5

        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_cleanup_old_tasks(self) -> None:
        """Test cleanup of old completed tasks.

        Note: Cleanup happens during enqueue, so the sequence matters.
        We first complete tasks, then enqueue new ones to trigger cleanup.
        """
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            queue = TaskQueue(
                persistence_path=temp_path,
                max_completed_tasks=2,
                auto_persist=False,
            )

            # First, create and complete 4 tasks
            completed_tasks = []
            for i in range(4):
                task = queue.enqueue(TaskType.BATCH_EXPORT)
                task.complete(TaskResult(success=True, items_processed=1))
                queue.update(task)
                completed_tasks.append(task)
                time.sleep(0.01)

            # Now we have 4 completed tasks. Enqueue one more to trigger cleanup.
            task5 = queue.enqueue(TaskType.BATCH_EXPORT)
            # Cleanup runs, removing 2 oldest (max_completed=2, had 4, remove 2)
            # Now we have 2 completed + 1 pending = 3 tasks

            completed = queue.get_all(status=TaskStatus.COMPLETED)
            # After cleanup, should have 2 completed tasks
            assert len(completed) == 2

            # The remaining completed tasks should be the most recent ones
            remaining_ids = {t.id for t in completed}
            # completed_tasks[2] and completed_tasks[3] should remain
            assert completed_tasks[2].id in remaining_ids
            assert completed_tasks[3].id in remaining_ids

            # Clean up - delete the pending task
            queue.cancel(task5.id)

        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_get_stats(self, temp_queue: TaskQueue) -> None:
        """Test getting queue statistics."""
        task1 = temp_queue.enqueue(TaskType.BATCH_EXPORT)
        temp_queue.enqueue(TaskType.BATCH_EXPORT)
        temp_queue.enqueue(TaskType.BATCH_SUMMARIZE)

        task1.complete(TaskResult(success=True, items_processed=1))
        temp_queue.update(task1)

        stats = temp_queue.get_stats()

        assert stats["total"] == 3
        assert stats["by_status"]["pending"] == 2
        assert stats["by_status"]["completed"] == 1
        assert stats["by_type"]["batch_export"] == 2
        assert stats["by_type"]["batch_summarize"] == 1

    def test_callback_registration(self, temp_queue: TaskQueue) -> None:
        """Test callback registration and notification."""
        callback_called = threading.Event()
        callback_task = None

        def callback(task: Task) -> None:
            nonlocal callback_task
            callback_task = task
            callback_called.set()

        task = temp_queue.enqueue(TaskType.BATCH_EXPORT)
        temp_queue.register_callback(task.id, callback)

        task.start()
        temp_queue.update(task)

        assert callback_called.wait(timeout=1.0)
        assert callback_task is not None
        assert callback_task.id == task.id
        assert callback_task.status == TaskStatus.RUNNING


class TestTaskWorker:
    """Tests for the background worker."""

    @pytest.fixture(autouse=True)
    def reset_singletons(self) -> None:
        """Reset singletons before each test."""
        reset_worker()
        reset_task_queue()

    @pytest.fixture
    def worker_with_queue(self) -> tuple[TaskWorker, TaskQueue]:
        """Create a worker with a test queue."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        queue = TaskQueue(persistence_path=temp_path, auto_persist=False)
        worker = TaskWorker(queue=queue, poll_interval=0.1)

        yield worker, queue

        worker.stop()
        if temp_path.exists():
            temp_path.unlink()

    def test_worker_start_stop(self, worker_with_queue: tuple[TaskWorker, TaskQueue]) -> None:
        """Test starting and stopping the worker."""
        worker, _queue = worker_with_queue

        assert worker.is_running is False

        worker.start()
        assert worker.is_running is True

        worker.stop()
        assert worker.is_running is False

    def test_worker_processes_task(self, worker_with_queue: tuple[TaskWorker, TaskQueue]) -> None:
        """Test that worker processes tasks."""
        worker, queue = worker_with_queue

        # Register a simple handler
        handler_called = threading.Event()

        def simple_handler(
            task: Task,
            update_progress: MagicMock,
        ) -> TaskResult:
            handler_called.set()
            return TaskResult(success=True, items_processed=1)

        worker.register_handler(TaskType.BATCH_EXPORT, simple_handler)

        # Enqueue and start worker
        task = queue.enqueue(TaskType.BATCH_EXPORT)
        worker.start()

        # Wait for task to be processed
        assert handler_called.wait(timeout=2.0)

        # Task should be completed
        time.sleep(0.2)  # Give time for status update
        updated = queue.get(task.id)
        assert updated is not None
        assert updated.status == TaskStatus.COMPLETED

    def test_worker_handles_error(self, worker_with_queue: tuple[TaskWorker, TaskQueue]) -> None:
        """Test that worker handles task errors."""
        worker, queue = worker_with_queue

        # Register a failing handler
        def failing_handler(
            task: Task,
            update_progress: MagicMock,
        ) -> TaskResult:
            raise ValueError("Handler error")

        worker.register_handler(TaskType.BATCH_EXPORT, failing_handler)

        # Enqueue and start worker
        task = queue.enqueue(TaskType.BATCH_EXPORT, max_retries=0)
        worker.start()

        # Wait for task to be processed
        time.sleep(0.5)

        # Task should be failed
        updated = queue.get(task.id)
        assert updated is not None
        assert updated.status == TaskStatus.FAILED
        assert "Handler error" in str(updated.error_message)

    def test_worker_retries_failed_task(
        self, worker_with_queue: tuple[TaskWorker, TaskQueue]
    ) -> None:
        """Test that worker retries failed tasks."""
        worker, queue = worker_with_queue

        call_count = 0

        def flaky_handler(
            task: Task,
            update_progress: MagicMock,
        ) -> TaskResult:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Temporary error")
            return TaskResult(success=True, items_processed=1)

        worker.register_handler(TaskType.BATCH_EXPORT, flaky_handler)

        # Enqueue and start worker
        task = queue.enqueue(TaskType.BATCH_EXPORT, max_retries=3)
        worker.start()

        # Wait for retries
        time.sleep(1.0)

        # Task should eventually complete
        updated = queue.get(task.id)
        assert updated is not None
        assert updated.status == TaskStatus.COMPLETED
        assert call_count == 2

    def test_worker_progress_updates(self, worker_with_queue: tuple[TaskWorker, TaskQueue]) -> None:
        """Test that worker receives progress updates."""
        worker, queue = worker_with_queue

        progress_updates: list[tuple[int, int, str]] = []

        def tracked_handler(
            task: Task,
            update_progress: MagicMock,
        ) -> TaskResult:
            for i in range(3):
                update_progress(i + 1, 3, f"Step {i + 1}")
                progress_updates.append((i + 1, 3, f"Step {i + 1}"))
                time.sleep(0.05)
            return TaskResult(success=True, items_processed=3)

        worker.register_handler(TaskType.BATCH_EXPORT, tracked_handler)

        queue.enqueue(TaskType.BATCH_EXPORT)
        worker.start()

        time.sleep(1.0)

        assert len(progress_updates) == 3
        assert progress_updates[-1] == (3, 3, "Step 3")


class TestConcurrency:
    """Tests for concurrent access."""

    @pytest.fixture(autouse=True)
    def reset_singletons(self) -> None:
        """Reset singletons before each test."""
        reset_task_queue()
        reset_worker()

    def test_concurrent_enqueue(self) -> None:
        """Test concurrent task enqueueing."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            queue = TaskQueue(persistence_path=temp_path, auto_persist=False)
            task_ids: list[str] = []
            lock = threading.Lock()

            def enqueue_task() -> None:
                task = queue.enqueue(TaskType.BATCH_EXPORT)
                with lock:
                    task_ids.append(task.id)

            # Create 10 threads
            threads = [threading.Thread(target=enqueue_task) for _ in range(10)]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # All tasks should be created
            assert len(task_ids) == 10
            assert len(set(task_ids)) == 10  # All unique

            # All tasks should be retrievable
            for task_id in task_ids:
                assert queue.get(task_id) is not None

        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_concurrent_updates(self) -> None:
        """Test concurrent task updates."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            queue = TaskQueue(persistence_path=temp_path, auto_persist=False)
            task = queue.enqueue(TaskType.BATCH_EXPORT)
            task.start()
            queue.update(task)

            errors: list[Exception] = []

            def update_progress(i: int) -> None:
                try:
                    t = queue.get(task.id)
                    if t:
                        t.update_progress(i, 10, f"Update {i}")
                        queue.update(t)
                except Exception as e:
                    errors.append(e)

            # Create 10 concurrent updates
            threads = [threading.Thread(target=update_progress, args=(i,)) for i in range(10)]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # No errors should occur
            assert len(errors) == 0

        finally:
            if temp_path.exists():
                temp_path.unlink()


class TestSingletons:
    """Tests for singleton access."""

    @pytest.fixture(autouse=True)
    def reset_singletons(self) -> None:
        """Reset singletons before each test."""
        reset_task_queue()
        reset_worker()

    def test_get_task_queue_singleton(self) -> None:
        """Test that get_task_queue returns the same instance."""
        queue1 = get_task_queue()
        queue2 = get_task_queue()

        assert queue1 is queue2

    def test_reset_task_queue(self) -> None:
        """Test that reset_task_queue clears the singleton."""
        queue1 = get_task_queue()
        reset_task_queue()
        queue2 = get_task_queue()

        assert queue1 is not queue2

    def test_get_worker_singleton(self) -> None:
        """Test that get_worker returns the same instance."""
        worker1 = get_worker()
        worker2 = get_worker()

        assert worker1 is worker2

    def test_reset_worker(self) -> None:
        """Test that reset_worker clears the singleton."""
        worker1 = get_worker()
        reset_worker()
        worker2 = get_worker()

        assert worker1 is not worker2
