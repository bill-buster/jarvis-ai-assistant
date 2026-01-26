"""Background worker for processing tasks.

Provides a threaded worker that continuously processes tasks from the queue.
Supports batch operations like export, summarization, and reply generation.

Usage:
    from jarvis.tasks import get_worker, start_worker, stop_worker

    # Start the background worker
    start_worker()

    # Stop the worker
    stop_worker()
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

from jarvis.tasks.models import Task, TaskResult, TaskType
from jarvis.tasks.queue import TaskQueue, get_task_queue

logger = logging.getLogger(__name__)

# Type alias for task handlers
TaskHandler = Callable[[Task, Callable[[int, int, str], None]], TaskResult]


class TaskWorker:
    """Background worker for processing tasks.

    Runs in a separate thread and continuously processes tasks from the queue.
    Supports registering custom handlers for different task types.

    Attributes:
        poll_interval: Seconds between queue polls when idle.
        max_concurrent: Maximum concurrent tasks (currently 1 for simplicity).
    """

    def __init__(
        self,
        queue: TaskQueue | None = None,
        poll_interval: float = 1.0,
    ) -> None:
        """Initialize the worker.

        Args:
            queue: Task queue to process. Defaults to singleton queue.
            poll_interval: Seconds between queue polls.
        """
        self._queue = queue or get_task_queue()
        self._poll_interval = poll_interval
        self._running = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._handlers: dict[TaskType, TaskHandler] = {}

        # Register default handlers
        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        """Register handlers for built-in task types."""
        self._handlers[TaskType.BATCH_EXPORT] = self._handle_batch_export
        self._handlers[TaskType.BATCH_SUMMARIZE] = self._handle_batch_summarize
        self._handlers[TaskType.BATCH_GENERATE_REPLIES] = self._handle_batch_generate_replies
        self._handlers[TaskType.SINGLE_EXPORT] = self._handle_single_export
        self._handlers[TaskType.SINGLE_SUMMARIZE] = self._handle_single_summarize
        self._handlers[TaskType.SINGLE_GENERATE_REPLY] = self._handle_single_generate_reply

    def register_handler(self, task_type: TaskType, handler: TaskHandler) -> None:
        """Register a custom handler for a task type.

        Args:
            task_type: The task type to handle.
            handler: Function that processes the task.
        """
        self._handlers[task_type] = handler

    def start(self) -> None:
        """Start the background worker thread."""
        if self._running:
            logger.warning("Worker is already running")
            return

        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="TaskWorker")
        self._thread.start()
        logger.info("Task worker started")

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the background worker.

        Args:
            timeout: Seconds to wait for graceful shutdown.
        """
        if not self._running:
            return

        self._running = False
        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("Worker thread did not stop gracefully")

        self._thread = None
        logger.info("Task worker stopped")

    @property
    def is_running(self) -> bool:
        """Check if the worker is running."""
        return self._running and self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        """Main worker loop."""
        logger.debug("Worker loop started")

        while not self._stop_event.is_set():
            try:
                # Get next pending task
                task = self._queue.get_next_pending()

                if task is None:
                    # No tasks, wait before polling again
                    self._stop_event.wait(self._poll_interval)
                    continue

                # Process the task
                self._process_task(task)

            except Exception as e:
                logger.exception(f"Error in worker loop: {e}")
                # Brief pause to avoid tight error loops
                self._stop_event.wait(1.0)

        logger.debug("Worker loop exited")

    def _process_task(self, task: Task) -> None:
        """Process a single task.

        Args:
            task: The task to process.
        """
        logger.info(f"Processing task {task.id} ({task.task_type.value})")

        # Get handler for this task type
        handler = self._handlers.get(task.task_type)
        if handler is None:
            logger.error(f"No handler for task type: {task.task_type.value}")
            task.fail(f"No handler for task type: {task.task_type.value}")
            self._queue.update(task)
            return

        # Mark task as running
        task.start()
        self._queue.update(task)

        # Create progress callback
        def update_progress(current: int, total: int, message: str = "") -> None:
            task.update_progress(current, total, message)
            self._queue.update(task)

        # Execute handler
        try:
            result = handler(task, update_progress)
            task.complete(result)
            logger.info(f"Task {task.id} completed successfully")

        except Exception as e:
            logger.exception(f"Task {task.id} failed: {e}")
            task.fail(str(e))

            # Schedule retry if possible
            if task.can_retry():
                task.retry()
                logger.info(f"Task {task.id} scheduled for retry (attempt {task.retry_count})")

        # Update final state
        self._queue.update(task)

    # Default task handlers

    def _handle_batch_export(
        self,
        task: Task,
        update_progress: Callable[[int, int, str], None],
    ) -> TaskResult:
        """Handle batch export task."""
        from integrations.imessage import ChatDBReader
        from jarvis.export import ExportFormat, export_messages

        params = task.params
        chat_ids = params.get("chat_ids", [])
        format_str = params.get("format", "json")
        export_format = ExportFormat(format_str)
        output_dir = params.get("output_dir")

        if not chat_ids:
            return TaskResult(success=False, error="No chat_ids provided")

        results: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        total = len(chat_ids)

        with ChatDBReader() as reader:
            conversations = reader.get_conversations(limit=500)
            conv_map = {c.chat_id: c for c in conversations}

            for i, chat_id in enumerate(chat_ids):
                update_progress(i, total, f"Exporting {chat_id}")

                try:
                    conv = conv_map.get(chat_id)
                    if conv is None:
                        failed.append({"chat_id": chat_id, "error": "Not found"})
                        continue

                    messages = reader.get_messages(chat_id=chat_id, limit=1000)
                    if not messages:
                        failed.append({"chat_id": chat_id, "error": "No messages"})
                        continue

                    exported_data = export_messages(
                        messages=messages,
                        format=export_format,
                        conversation=conv,
                    )

                    result_item: dict[str, Any] = {
                        "chat_id": chat_id,
                        "message_count": len(messages),
                        "format": format_str,
                    }

                    # Save to file if output_dir specified
                    if output_dir:
                        from pathlib import Path

                        from jarvis.export import get_export_filename

                        filename = get_export_filename(
                            format=export_format,
                            prefix="conversation",
                            chat_id=chat_id,
                        )
                        output_path = Path(output_dir) / filename
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        output_path.write_text(exported_data, encoding="utf-8")
                        result_item["output_file"] = str(output_path)
                    else:
                        result_item["data"] = exported_data

                    results.append(result_item)

                except Exception as e:
                    logger.warning(f"Failed to export {chat_id}: {e}")
                    failed.append({"chat_id": chat_id, "error": str(e)})

        update_progress(total, total, "Export complete")

        return TaskResult(
            success=len(failed) == 0,
            data={"exports": results},
            items_processed=len(results),
            items_failed=len(failed),
            partial_results=failed if failed else [],
        )

    def _handle_batch_summarize(
        self,
        task: Task,
        update_progress: Callable[[int, int, str], None],
    ) -> TaskResult:
        """Handle batch summarize task."""
        from contracts.models import GenerationRequest
        from integrations.imessage import ChatDBReader
        from jarvis.context import ContextFetcher
        from jarvis.prompts import SUMMARY_EXAMPLES, build_summary_prompt
        from models import get_generator

        params = task.params
        chat_ids = params.get("chat_ids", [])
        num_messages = params.get("num_messages", 50)

        if not chat_ids:
            return TaskResult(success=False, error="No chat_ids provided")

        results: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        total = len(chat_ids)

        generator = get_generator()

        with ChatDBReader() as reader:
            fetcher = ContextFetcher(reader)
            conversations = reader.get_conversations(limit=500)
            conv_map = {c.chat_id: c for c in conversations}

            for i, chat_id in enumerate(chat_ids):
                conv = conv_map.get(chat_id)
                display_name = conv.display_name if conv else chat_id
                update_progress(i, total, f"Summarizing {display_name}")

                try:
                    context = fetcher.get_summary_context(chat_id, num_messages=num_messages)

                    if len(context.messages) < 3:
                        failed.append({"chat_id": chat_id, "error": "Not enough messages"})
                        continue

                    formatted_prompt = build_summary_prompt(context=context.formatted_context)

                    request = GenerationRequest(
                        prompt=formatted_prompt,
                        context_documents=[context.formatted_context],
                        few_shot_examples=SUMMARY_EXAMPLES,
                        max_tokens=500,
                        temperature=0.5,
                    )
                    response = generator.generate(request)

                    start_date = context.date_range[0].strftime("%Y-%m-%d")
                    end_date = context.date_range[1].strftime("%Y-%m-%d")

                    results.append(
                        {
                            "chat_id": chat_id,
                            "display_name": display_name,
                            "summary": response.text,
                            "message_count": len(context.messages),
                            "date_range": {"start": start_date, "end": end_date},
                        }
                    )

                except Exception as e:
                    logger.warning(f"Failed to summarize {chat_id}: {e}")
                    failed.append({"chat_id": chat_id, "error": str(e)})

        update_progress(total, total, "Summarization complete")

        return TaskResult(
            success=len(failed) == 0,
            data={"summaries": results},
            items_processed=len(results),
            items_failed=len(failed),
            partial_results=failed if failed else [],
        )

    def _handle_batch_generate_replies(
        self,
        task: Task,
        update_progress: Callable[[int, int, str], None],
    ) -> TaskResult:
        """Handle batch reply generation task."""
        from contracts.models import GenerationRequest
        from integrations.imessage import ChatDBReader
        from jarvis.context import ContextFetcher
        from jarvis.prompts import REPLY_EXAMPLES, build_reply_prompt
        from models import get_generator

        params = task.params
        chat_ids = params.get("chat_ids", [])
        instruction = params.get("instruction")
        num_suggestions = params.get("num_suggestions", 3)

        if not chat_ids:
            return TaskResult(success=False, error="No chat_ids provided")

        results: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        total = len(chat_ids)

        generator = get_generator()

        with ChatDBReader() as reader:
            fetcher = ContextFetcher(reader)
            conversations = reader.get_conversations(limit=500)
            conv_map = {c.chat_id: c for c in conversations}

            for i, chat_id in enumerate(chat_ids):
                conv = conv_map.get(chat_id)
                display_name = conv.display_name if conv else chat_id
                update_progress(i, total, f"Generating replies for {display_name}")

                try:
                    context = fetcher.get_reply_context(chat_id, num_messages=20)

                    if not context.last_received_message:
                        failed.append(
                            {"chat_id": chat_id, "error": "No recent messages to reply to"}
                        )
                        continue

                    last_msg = context.last_received_message
                    formatted_prompt = build_reply_prompt(
                        context=context.formatted_context,
                        last_message=last_msg.text,
                        instruction=instruction,
                    )

                    suggestions: list[str] = []
                    for j in range(num_suggestions):
                        request = GenerationRequest(
                            prompt=formatted_prompt,
                            context_documents=[context.formatted_context],
                            few_shot_examples=REPLY_EXAMPLES,
                            max_tokens=150,
                            temperature=0.7 + (j * 0.1),
                        )
                        response = generator.generate(request)
                        suggestions.append(response.text)

                    results.append(
                        {
                            "chat_id": chat_id,
                            "display_name": display_name,
                            "last_message": last_msg.text[:200],
                            "suggestions": suggestions,
                        }
                    )

                except Exception as e:
                    logger.warning(f"Failed to generate replies for {chat_id}: {e}")
                    failed.append({"chat_id": chat_id, "error": str(e)})

        update_progress(total, total, "Reply generation complete")

        return TaskResult(
            success=len(failed) == 0,
            data={"replies": results},
            items_processed=len(results),
            items_failed=len(failed),
            partial_results=failed if failed else [],
        )

    def _handle_single_export(
        self,
        task: Task,
        update_progress: Callable[[int, int, str], None],
    ) -> TaskResult:
        """Handle single export task."""
        # Wrap single item in batch handler
        params = task.params
        chat_id = params.get("chat_id")
        if not chat_id:
            return TaskResult(success=False, error="No chat_id provided")

        task.params["chat_ids"] = [chat_id]
        return self._handle_batch_export(task, update_progress)

    def _handle_single_summarize(
        self,
        task: Task,
        update_progress: Callable[[int, int, str], None],
    ) -> TaskResult:
        """Handle single summarize task."""
        params = task.params
        chat_id = params.get("chat_id")
        if not chat_id:
            return TaskResult(success=False, error="No chat_id provided")

        task.params["chat_ids"] = [chat_id]
        return self._handle_batch_summarize(task, update_progress)

    def _handle_single_generate_reply(
        self,
        task: Task,
        update_progress: Callable[[int, int, str], None],
    ) -> TaskResult:
        """Handle single reply generation task."""
        params = task.params
        chat_id = params.get("chat_id")
        if not chat_id:
            return TaskResult(success=False, error="No chat_id provided")

        task.params["chat_ids"] = [chat_id]
        return self._handle_batch_generate_replies(task, update_progress)


# Module-level singleton
_worker: TaskWorker | None = None
_worker_lock = threading.Lock()


def get_worker() -> TaskWorker:
    """Get the singleton worker instance.

    Returns:
        Shared TaskWorker instance.
    """
    global _worker
    if _worker is None:
        with _worker_lock:
            if _worker is None:
                _worker = TaskWorker()
    return _worker


def start_worker() -> None:
    """Start the background task worker."""
    worker = get_worker()
    worker.start()


def stop_worker() -> None:
    """Stop the background task worker."""
    global _worker
    with _worker_lock:
        if _worker is not None:
            _worker.stop()


def reset_worker() -> None:
    """Reset the singleton worker (for testing)."""
    global _worker
    with _worker_lock:
        if _worker is not None:
            _worker.stop()
        _worker = None


# Export all public symbols
__all__ = [
    "TaskWorker",
    "get_worker",
    "reset_worker",
    "start_worker",
    "stop_worker",
]
