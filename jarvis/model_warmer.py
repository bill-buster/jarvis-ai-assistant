"""Smart model loading with warm model cache and idle timeout.

Keeps the model warm for recent traffic to avoid cold-start latency for active users.

Features:
- Track last request timestamp
- Keep model loaded if recent activity (configurable, default 5 minutes)
- Automatic unload after idle timeout
- Memory-aware loading decisions (respects memory controller modes)
- Memory pressure callback integration for emergency unloads

Usage:
    from jarvis.model_warmer import get_model_warmer

    warmer = get_model_warmer()
    warmer.start()  # Start background monitoring

    # Record activity (call this on every generation request)
    warmer.touch()

    # Check if model should be loaded
    if warmer.should_load():
        generator.load()

    # On shutdown
    warmer.stop()
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from contracts.memory import MemoryMode

if TYPE_CHECKING:
    from models.generator import MLXGenerator

logger = logging.getLogger(__name__)

# Memory mode ordering for comparison (higher = more memory available)
_MEMORY_MODE_ORDER = {
    MemoryMode.MINIMAL: 0,
    MemoryMode.LITE: 1,
    MemoryMode.FULL: 2,
}


@dataclass
class WarmerStats:
    """Statistics for model warmer operations.

    Attributes:
        total_loads: Total number of model loads triggered.
        total_unloads: Total number of model unloads (idle + pressure).
        idle_unloads: Number of unloads due to idle timeout.
        pressure_unloads: Number of unloads due to memory pressure.
        last_load_time: Timestamp of last model load.
        last_unload_time: Timestamp of last model unload.
        current_idle_seconds: Current idle time in seconds.
    """

    total_loads: int = 0
    total_unloads: int = 0
    idle_unloads: int = 0
    pressure_unloads: int = 0
    last_load_time: float | None = None
    last_unload_time: float | None = None
    current_idle_seconds: float = 0.0


@dataclass
class WarmerConfig:
    """Configuration for the model warmer.

    Attributes:
        idle_timeout_seconds: Unload model after this many seconds of inactivity.
            Set to 0 to disable automatic unloading.
        check_interval_seconds: How often to check for idle timeout.
        warm_on_startup: Pre-load model when warmer starts.
        respect_memory_pressure: Unload model when memory pressure is high.
        min_memory_mode: Minimum memory mode required to keep model loaded.
            If system drops below this mode, model will be unloaded.
    """

    idle_timeout_seconds: float = 300.0
    check_interval_seconds: float = 30.0
    warm_on_startup: bool = False
    respect_memory_pressure: bool = True
    min_memory_mode: MemoryMode = field(default=MemoryMode.LITE)


class ModelWarmer:
    """Smart model loader that keeps models warm for active users.

    Tracks request activity and automatically unloads models after idle timeout
    to free memory for inactive users. Integrates with the memory controller
    to respect system memory constraints.

    Thread-safe for concurrent access.
    """

    def __init__(
        self,
        config: WarmerConfig | None = None,
        generator: MLXGenerator | None = None,
    ) -> None:
        """Initialize the model warmer.

        Args:
            config: Warmer configuration. Uses defaults if not provided.
            generator: Generator instance to manage. If not provided,
                uses the singleton generator.
        """
        self._config = config or self._load_config_from_settings()
        self._generator = generator
        self._last_activity: float = 0.0
        self._lock = threading.RLock()  # Use RLock to allow nested acquisition
        self._stop_event = threading.Event()
        self._monitor_thread: threading.Thread | None = None
        self._started = False
        self._stats = WarmerStats()
        self._pressure_callback_registered = False

    def _load_config_from_settings(self) -> WarmerConfig:
        """Load warmer config from JARVIS settings.

        Returns:
            WarmerConfig populated from jarvis.config.
        """
        try:
            from jarvis.config import get_config

            settings = get_config().model
            return WarmerConfig(
                idle_timeout_seconds=settings.idle_timeout_seconds,
                warm_on_startup=settings.warm_on_startup,
            )
        except Exception:
            logger.debug("Could not load config, using defaults")
            return WarmerConfig()

    def _get_generator(self) -> MLXGenerator:
        """Get the generator instance.

        Returns:
            The MLXGenerator instance (provided or singleton).
        """
        if self._generator is not None:
            return self._generator

        from models import get_generator

        return get_generator()

    def _get_memory_controller(self):
        """Get the memory controller instance.

        Returns:
            The DefaultMemoryController singleton.
        """
        from core.memory.controller import get_memory_controller

        return get_memory_controller()

    def start(self) -> None:
        """Start the model warmer.

        Begins background monitoring for idle timeout.
        If warm_on_startup is enabled, loads the model immediately.
        Registers memory pressure callback.

        Thread-safe: can be called multiple times safely.
        """
        with self._lock:
            if self._started:
                logger.debug("Model warmer already started")
                return

            self._started = True
            self._stop_event.clear()

            # Register memory pressure callback
            if self._config.respect_memory_pressure:
                self._register_pressure_callback()

            # Start background monitor thread
            if self._config.idle_timeout_seconds > 0:
                self._monitor_thread = threading.Thread(
                    target=self._monitor_loop,
                    name="ModelWarmerMonitor",
                    daemon=True,
                )
                self._monitor_thread.start()
                logger.info(
                    "Model warmer started (idle_timeout=%.0fs)",
                    self._config.idle_timeout_seconds,
                )
            else:
                logger.info("Model warmer started (idle timeout disabled)")

            # Warm on startup if configured
            if self._config.warm_on_startup:
                self._warm_model()

    def stop(self) -> None:
        """Stop the model warmer.

        Stops background monitoring. Does NOT unload the model.
        Call unload() explicitly if you want to free memory.

        Thread-safe: can be called multiple times safely.
        """
        with self._lock:
            if not self._started:
                return

            self._stop_event.set()
            self._started = False

            # Unregister pressure callback
            self._unregister_pressure_callback()

        # Wait for monitor thread outside lock to avoid deadlock
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None

        logger.info("Model warmer stopped")

    def touch(self) -> None:
        """Record activity to keep the model warm.

        Call this method on every generation request to reset the idle timer.
        """
        with self._lock:
            self._last_activity = time.monotonic()
            logger.debug("Model warmer touched")

    def get_idle_seconds(self) -> float:
        """Get the number of seconds since last activity.

        Returns:
            Seconds since last touch(), or 0 if never touched.
        """
        with self._lock:
            if self._last_activity == 0.0:
                return 0.0
            return time.monotonic() - self._last_activity

    def is_idle(self) -> bool:
        """Check if the model has been idle longer than the timeout.

        Returns:
            True if idle time exceeds configured timeout, False otherwise.
            Always returns False if idle timeout is disabled (set to 0).
        """
        if self._config.idle_timeout_seconds <= 0:
            return False
        return self.get_idle_seconds() > self._config.idle_timeout_seconds

    def should_load(self) -> bool:
        """Check if the model should be loaded based on memory constraints.

        Considers:
        - Current memory mode (FULL/LITE/MINIMAL)
        - Memory pressure level
        - Available memory for model

        Returns:
            True if it's safe to load the model, False otherwise.
        """
        try:
            controller = self._get_memory_controller()

            # Check memory mode
            mode = controller.get_mode()
            current_order = _MEMORY_MODE_ORDER.get(mode, 0)
            min_order = _MEMORY_MODE_ORDER.get(self._config.min_memory_mode, 1)
            if current_order < min_order:
                logger.debug(
                    "Memory mode %s below minimum %s, not loading",
                    mode.value,
                    self._config.min_memory_mode.value,
                )
                return False

            # Check memory pressure
            state = controller.get_state()
            if state.pressure_level in ("red", "critical"):
                logger.debug(
                    "Memory pressure %s too high, not loading",
                    state.pressure_level,
                )
                return False

            # Check if we can load the model
            generator = self._get_generator()
            required_mb = generator.config.estimated_memory_mb
            if not controller.can_load_model(required_mb):
                logger.debug(
                    "Insufficient memory (need %.0fMB), not loading",
                    required_mb,
                )
                return False

            return True

        except Exception as e:
            logger.warning("Error checking load conditions: %s", e)
            # Default to allowing load if we can't check
            return True

    def ensure_warm(self) -> bool:
        """Ensure the model is loaded if conditions allow.

        Convenience method that checks should_load() and loads if appropriate.
        Also touches the warmer to reset idle timer.

        Returns:
            True if model is loaded (was already loaded or just loaded),
            False if model could not be loaded due to constraints.
        """
        self.touch()

        generator = self._get_generator()
        if generator.is_loaded():
            return True

        if not self.should_load():
            return False

        return self._load_model()

    def unload(self) -> None:
        """Unload the model to free memory.

        Records unload in statistics. Does not stop the warmer.
        """
        generator = self._get_generator()
        if generator.is_loaded():
            generator.unload()
            with self._lock:
                self._stats.total_unloads += 1
                self._stats.last_unload_time = time.time()

            # Update memory controller state
            try:
                controller = self._get_memory_controller()
                controller.set_model_loaded(False)
            except Exception as e:
                logger.debug("Could not update memory controller: %s", e)

            logger.info("Model unloaded by warmer")

    def get_stats(self) -> WarmerStats:
        """Get current warmer statistics.

        Returns:
            WarmerStats with current counters and timestamps.
        """
        with self._lock:
            stats = WarmerStats(
                total_loads=self._stats.total_loads,
                total_unloads=self._stats.total_unloads,
                idle_unloads=self._stats.idle_unloads,
                pressure_unloads=self._stats.pressure_unloads,
                last_load_time=self._stats.last_load_time,
                last_unload_time=self._stats.last_unload_time,
                current_idle_seconds=self.get_idle_seconds(),
            )
        return stats

    def _warm_model(self) -> bool:
        """Pre-load the model (warm on startup).

        Returns:
            True if model loaded successfully, False otherwise.
        """
        if not self.should_load():
            logger.info("Skipping warm-on-startup due to memory constraints")
            return False

        logger.info("Warming model on startup")
        return self._load_model()

    def _load_model(self) -> bool:
        """Load the model and update statistics.

        Returns:
            True if model loaded successfully, False otherwise.
        """
        try:
            generator = self._get_generator()
            if generator.load():
                with self._lock:
                    self._stats.total_loads += 1
                    self._stats.last_load_time = time.time()
                    self._last_activity = time.monotonic()

                # Update memory controller state
                try:
                    controller = self._get_memory_controller()
                    controller.set_model_loaded(True)
                except Exception as e:
                    logger.debug("Could not update memory controller: %s", e)

                logger.info("Model loaded by warmer")
                return True
            return False
        except Exception as e:
            logger.error("Failed to load model: %s", e)
            return False

    def _monitor_loop(self) -> None:
        """Background loop that checks for idle timeout.

        Runs until stop() is called.
        """
        logger.debug("Model warmer monitor started")

        while not self._stop_event.wait(self._config.check_interval_seconds):
            self._check_idle_timeout()

        logger.debug("Model warmer monitor stopped")

    def _check_idle_timeout(self) -> None:
        """Check if model should be unloaded due to idle timeout."""
        try:
            generator = self._get_generator()
            if not generator.is_loaded():
                return

            if self.is_idle():
                idle_secs = self.get_idle_seconds()
                logger.info(
                    "Model idle for %.0fs (timeout=%.0fs), unloading",
                    idle_secs,
                    self._config.idle_timeout_seconds,
                )
                self.unload()
                with self._lock:
                    self._stats.idle_unloads += 1

        except Exception as e:
            logger.warning("Error in idle timeout check: %s", e)

    def _register_pressure_callback(self) -> None:
        """Register callback with memory controller for pressure events."""
        if self._pressure_callback_registered:
            return

        try:
            controller = self._get_memory_controller()
            controller.register_pressure_callback(self._on_memory_pressure)
            self._pressure_callback_registered = True
            logger.debug("Registered memory pressure callback")
        except Exception as e:
            logger.warning("Failed to register pressure callback: %s", e)

    def _unregister_pressure_callback(self) -> None:
        """Unregister memory pressure callback."""
        if not self._pressure_callback_registered:
            return

        try:
            controller = self._get_memory_controller()
            controller.unregister_pressure_callback(self._on_memory_pressure)
            self._pressure_callback_registered = False
            logger.debug("Unregistered memory pressure callback")
        except Exception as e:
            logger.warning("Failed to unregister pressure callback: %s", e)

    def _on_memory_pressure(self, pressure_level: str) -> None:
        """Handle memory pressure callback.

        Unloads model if pressure is high.

        Args:
            pressure_level: Current pressure level ("green", "yellow", "red", "critical").
        """
        if pressure_level in ("red", "critical"):
            logger.warning(
                "Memory pressure %s, unloading model",
                pressure_level,
            )
            self.unload()
            with self._lock:
                self._stats.pressure_unloads += 1


# Singleton instance with thread-safe initialization
_warmer: ModelWarmer | None = None
_warmer_lock = threading.Lock()


def get_model_warmer() -> ModelWarmer:
    """Get or create singleton model warmer instance.

    Thread-safe using double-check locking pattern.

    Returns:
        The shared ModelWarmer instance.
    """
    global _warmer
    if _warmer is None:
        with _warmer_lock:
            if _warmer is None:
                _warmer = ModelWarmer()
    return _warmer


def reset_model_warmer() -> None:
    """Reset the singleton model warmer instance.

    Stops the warmer and clears the singleton.
    Use this for testing or to reinitialize with new configuration.
    """
    global _warmer
    with _warmer_lock:
        if _warmer is not None:
            _warmer.stop()
        _warmer = None


def get_warm_generator(
    skip_templates: bool = True,
    model_id: str | None = None,
) -> MLXGenerator:
    """Get the singleton generator with warmer integration.

    Convenience function that:
    1. Gets the model warmer singleton
    2. Touches it to reset idle timer
    3. Returns the generator

    This is the recommended way to get the generator when using
    the model warmer. The warmer should be started separately
    (e.g., at API startup).

    Args:
        skip_templates: If True (default), skip template matching.
        model_id: Optional model ID from registry.

    Returns:
        The shared MLXGenerator instance.

    Example:
        # At startup
        warmer = get_model_warmer()
        warmer.start()

        # On each request
        generator = get_warm_generator()
        response = generator.generate(request)

        # At shutdown
        warmer.stop()
    """
    from models import get_generator

    # Touch the warmer to reset idle timer
    warmer = get_model_warmer()
    warmer.touch()

    return get_generator(skip_templates=skip_templates, model_id=model_id)
