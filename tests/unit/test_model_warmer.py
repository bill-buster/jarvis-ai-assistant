"""Tests for the model warmer module."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from contracts.memory import MemoryMode
from jarvis.model_warmer import (
    ModelWarmer,
    WarmerConfig,
    WarmerStats,
    get_model_warmer,
    get_warm_generator,
    reset_model_warmer,
)


@pytest.fixture
def mock_generator():
    """Create a mock generator."""
    generator = MagicMock()
    generator.is_loaded.return_value = False
    generator.load.return_value = True
    generator.unload.return_value = None
    generator.config.estimated_memory_mb = 800
    return generator


@pytest.fixture
def mock_memory_controller():
    """Create a mock memory controller."""
    controller = MagicMock()
    controller.get_mode.return_value = MemoryMode.FULL
    controller.get_state.return_value = MagicMock(
        pressure_level="green",
        available_mb=8000,
        used_mb=4000,
        model_loaded=False,
        current_mode=MemoryMode.FULL,
    )
    controller.can_load_model.return_value = True
    controller.register_pressure_callback.return_value = None
    controller.unregister_pressure_callback.return_value = None
    controller.set_model_loaded.return_value = None
    return controller


class TestWarmerConfig:
    """Tests for WarmerConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = WarmerConfig()
        assert config.idle_timeout_seconds == 300.0
        assert config.check_interval_seconds == 30.0
        assert config.warm_on_startup is False
        assert config.respect_memory_pressure is True
        assert config.min_memory_mode == MemoryMode.LITE

    def test_custom_config(self):
        """Test custom configuration values."""
        config = WarmerConfig(
            idle_timeout_seconds=600.0,
            check_interval_seconds=60.0,
            warm_on_startup=True,
            respect_memory_pressure=False,
            min_memory_mode=MemoryMode.FULL,
        )
        assert config.idle_timeout_seconds == 600.0
        assert config.check_interval_seconds == 60.0
        assert config.warm_on_startup is True
        assert config.respect_memory_pressure is False
        assert config.min_memory_mode == MemoryMode.FULL


class TestModelWarmer:
    """Tests for ModelWarmer class."""

    def test_touch_updates_last_activity(self, mock_generator, mock_memory_controller):
        """Test that touch() updates the last activity timestamp."""
        with (
            patch(
                "jarvis.model_warmer.ModelWarmer._get_generator",
                return_value=mock_generator,
            ),
            patch(
                "jarvis.model_warmer.ModelWarmer._get_memory_controller",
                return_value=mock_memory_controller,
            ),
        ):
            config = WarmerConfig(idle_timeout_seconds=0)  # Disable auto-unload
            warmer = ModelWarmer(config=config, generator=mock_generator)

            # Initially no activity
            assert warmer.get_idle_seconds() == 0.0

            # Touch and verify activity is recorded
            warmer.touch()
            idle_secs = warmer.get_idle_seconds()
            assert idle_secs >= 0.0
            assert idle_secs < 1.0  # Should be very recent

    def test_is_idle_respects_timeout(self, mock_generator, mock_memory_controller):
        """Test that is_idle() respects the idle timeout setting."""
        with (
            patch(
                "jarvis.model_warmer.ModelWarmer._get_generator",
                return_value=mock_generator,
            ),
            patch(
                "jarvis.model_warmer.ModelWarmer._get_memory_controller",
                return_value=mock_memory_controller,
            ),
        ):
            config = WarmerConfig(idle_timeout_seconds=0.1)  # Very short timeout
            warmer = ModelWarmer(config=config, generator=mock_generator)

            warmer.touch()
            assert warmer.is_idle() is False  # Just touched

            # Wait for timeout
            time.sleep(0.15)
            assert warmer.is_idle() is True  # Should be idle now

    def test_is_idle_disabled_when_timeout_zero(self, mock_generator):
        """Test that is_idle() returns False when timeout is 0."""
        config = WarmerConfig(idle_timeout_seconds=0)
        warmer = ModelWarmer(config=config, generator=mock_generator)

        warmer.touch()
        time.sleep(0.1)
        assert warmer.is_idle() is False  # Should never be idle with timeout=0

    def test_should_load_checks_memory_mode(self, mock_generator, mock_memory_controller):
        """Test that should_load() checks memory mode."""
        with patch(
            "jarvis.model_warmer.ModelWarmer._get_memory_controller",
            return_value=mock_memory_controller,
        ):
            config = WarmerConfig(min_memory_mode=MemoryMode.FULL)
            warmer = ModelWarmer(config=config, generator=mock_generator)

            # FULL mode should allow loading
            mock_memory_controller.get_mode.return_value = MemoryMode.FULL
            assert warmer.should_load() is True

            # LITE mode should not allow loading (below FULL)
            mock_memory_controller.get_mode.return_value = MemoryMode.LITE
            assert warmer.should_load() is False

    def test_should_load_checks_memory_pressure(self, mock_generator, mock_memory_controller):
        """Test that should_load() checks memory pressure."""
        with patch(
            "jarvis.model_warmer.ModelWarmer._get_memory_controller",
            return_value=mock_memory_controller,
        ):
            config = WarmerConfig(min_memory_mode=MemoryMode.LITE)
            warmer = ModelWarmer(config=config, generator=mock_generator)

            # Green pressure should allow loading
            mock_memory_controller.get_state.return_value.pressure_level = "green"
            assert warmer.should_load() is True

            # Red pressure should not allow loading
            mock_memory_controller.get_state.return_value.pressure_level = "red"
            assert warmer.should_load() is False

    def test_ensure_warm_loads_model_if_not_loaded(self, mock_generator, mock_memory_controller):
        """Test that ensure_warm() loads the model if not loaded."""
        with patch(
            "jarvis.model_warmer.ModelWarmer._get_memory_controller",
            return_value=mock_memory_controller,
        ):
            config = WarmerConfig(idle_timeout_seconds=0)
            warmer = ModelWarmer(config=config, generator=mock_generator)

            mock_generator.is_loaded.return_value = False
            mock_generator.load.return_value = True

            result = warmer.ensure_warm()
            assert result is True
            mock_generator.load.assert_called_once()

    def test_ensure_warm_skips_load_if_already_loaded(self, mock_generator, mock_memory_controller):
        """Test that ensure_warm() doesn't reload if already loaded."""
        with patch(
            "jarvis.model_warmer.ModelWarmer._get_memory_controller",
            return_value=mock_memory_controller,
        ):
            config = WarmerConfig(idle_timeout_seconds=0)
            warmer = ModelWarmer(config=config, generator=mock_generator)

            mock_generator.is_loaded.return_value = True

            result = warmer.ensure_warm()
            assert result is True
            mock_generator.load.assert_not_called()

    def test_unload_updates_stats(self, mock_generator, mock_memory_controller):
        """Test that unload() updates statistics."""
        with patch(
            "jarvis.model_warmer.ModelWarmer._get_memory_controller",
            return_value=mock_memory_controller,
        ):
            config = WarmerConfig(idle_timeout_seconds=0)
            warmer = ModelWarmer(config=config, generator=mock_generator)

            mock_generator.is_loaded.return_value = True

            warmer.unload()
            mock_generator.unload.assert_called_once()

            stats = warmer.get_stats()
            assert stats.total_unloads == 1
            assert stats.last_unload_time is not None

    def test_start_stop_lifecycle(self, mock_generator, mock_memory_controller):
        """Test start/stop lifecycle."""
        with patch(
            "jarvis.model_warmer.ModelWarmer._get_memory_controller",
            return_value=mock_memory_controller,
        ):
            config = WarmerConfig(
                idle_timeout_seconds=1.0,
                check_interval_seconds=0.1,
            )
            warmer = ModelWarmer(config=config, generator=mock_generator)

            assert warmer._started is False

            warmer.start()
            assert warmer._started is True
            assert warmer._monitor_thread is not None
            assert warmer._monitor_thread.is_alive()

            warmer.stop()
            assert warmer._started is False
            time.sleep(0.2)  # Give thread time to stop
            assert warmer._monitor_thread is None or not warmer._monitor_thread.is_alive()

    def test_start_is_idempotent(self, mock_generator, mock_memory_controller):
        """Test that calling start() multiple times is safe."""
        with patch(
            "jarvis.model_warmer.ModelWarmer._get_memory_controller",
            return_value=mock_memory_controller,
        ):
            config = WarmerConfig(idle_timeout_seconds=0)  # Disable monitor thread
            warmer = ModelWarmer(config=config, generator=mock_generator)

            warmer.start()
            warmer.start()  # Should not raise
            warmer.start()  # Should not raise
            warmer.stop()

    def test_stop_is_idempotent(self, mock_generator):
        """Test that calling stop() multiple times is safe."""
        config = WarmerConfig(idle_timeout_seconds=0)
        warmer = ModelWarmer(config=config, generator=mock_generator)

        warmer.stop()  # Should not raise even though not started
        warmer.stop()  # Should not raise

    def test_warm_on_startup_loads_model(self, mock_generator, mock_memory_controller):
        """Test that warm_on_startup=True loads the model at start."""
        with patch(
            "jarvis.model_warmer.ModelWarmer._get_memory_controller",
            return_value=mock_memory_controller,
        ):
            config = WarmerConfig(
                idle_timeout_seconds=0,
                warm_on_startup=True,
            )
            warmer = ModelWarmer(config=config, generator=mock_generator)

            mock_generator.is_loaded.return_value = False
            mock_generator.load.return_value = True

            warmer.start()
            mock_generator.load.assert_called_once()
            warmer.stop()

    def test_memory_pressure_callback_unloads_model(self, mock_generator, mock_memory_controller):
        """Test that memory pressure callback unloads the model."""
        with patch(
            "jarvis.model_warmer.ModelWarmer._get_memory_controller",
            return_value=mock_memory_controller,
        ):
            config = WarmerConfig(
                idle_timeout_seconds=0,
                respect_memory_pressure=True,
            )
            warmer = ModelWarmer(config=config, generator=mock_generator)
            warmer.start()

            mock_generator.is_loaded.return_value = True

            # Simulate memory pressure callback
            warmer._on_memory_pressure("critical")

            mock_generator.unload.assert_called_once()
            stats = warmer.get_stats()
            assert stats.pressure_unloads == 1

            warmer.stop()

    def test_get_stats_returns_current_values(self, mock_generator, mock_memory_controller):
        """Test that get_stats() returns current statistics."""
        with patch(
            "jarvis.model_warmer.ModelWarmer._get_memory_controller",
            return_value=mock_memory_controller,
        ):
            config = WarmerConfig(idle_timeout_seconds=0)
            warmer = ModelWarmer(config=config, generator=mock_generator)

            stats = warmer.get_stats()
            assert isinstance(stats, WarmerStats)
            assert stats.total_loads == 0
            assert stats.total_unloads == 0
            assert stats.idle_unloads == 0
            assert stats.pressure_unloads == 0


class TestSingletonFunctions:
    """Tests for singleton functions."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_model_warmer()

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_model_warmer()

    def test_get_model_warmer_returns_same_instance(self):
        """Test that get_model_warmer() returns the same instance."""
        warmer1 = get_model_warmer()
        warmer2 = get_model_warmer()
        assert warmer1 is warmer2

    def test_reset_model_warmer_clears_instance(self):
        """Test that reset_model_warmer() clears the singleton."""
        warmer1 = get_model_warmer()
        reset_model_warmer()
        warmer2 = get_model_warmer()
        assert warmer1 is not warmer2

    def test_get_warm_generator_touches_warmer(self):
        """Test that get_warm_generator() touches the warmer."""
        with (
            patch("jarvis.model_warmer.get_model_warmer") as mock_get_warmer,
            patch("models.get_generator") as mock_get_gen,
        ):
            mock_warmer = MagicMock()
            mock_get_warmer.return_value = mock_warmer
            mock_get_gen.return_value = MagicMock()

            get_warm_generator()

            mock_warmer.touch.assert_called_once()


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_touch_calls(self, mock_generator, mock_memory_controller):
        """Test that concurrent touch() calls are safe."""
        with patch(
            "jarvis.model_warmer.ModelWarmer._get_memory_controller",
            return_value=mock_memory_controller,
        ):
            config = WarmerConfig(idle_timeout_seconds=0)
            warmer = ModelWarmer(config=config, generator=mock_generator)

            errors = []

            def touch_repeatedly():
                try:
                    for _ in range(100):
                        warmer.touch()
                        warmer.get_idle_seconds()
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=touch_repeatedly) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0

    def test_concurrent_start_stop_calls(self, mock_generator, mock_memory_controller):
        """Test that concurrent start/stop calls are safe."""
        with patch(
            "jarvis.model_warmer.ModelWarmer._get_memory_controller",
            return_value=mock_memory_controller,
        ):
            config = WarmerConfig(idle_timeout_seconds=0)
            warmer = ModelWarmer(config=config, generator=mock_generator)

            errors = []

            def start_stop_repeatedly():
                try:
                    for _ in range(10):
                        warmer.start()
                        time.sleep(0.01)
                        warmer.stop()
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=start_stop_repeatedly) for _ in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0
            warmer.stop()  # Ensure stopped
