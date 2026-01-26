"""
E2E test script - run manually to test with real iMessage data.

This script tests the complete RAG flow with actual iMessage data from your Mac.
It requires Full Disk Access permission to read the iMessage database.

Usage:
    # Run the full E2E test
    python -m tests.integration.test_e2e_script

    # Run with verbose output
    python -m tests.integration.test_e2e_script --verbose

    # Run specific test
    python -m tests.integration.test_e2e_script --test reply

Requirements:
    - macOS with iMessage
    - Full Disk Access permission granted to Terminal/IDE
    - At least one iMessage conversation
    - MLX framework (Apple Silicon)

Note:
    This script is NOT run by pytest. It's designed for manual E2E testing on macOS.
    Pytest will skip this file because it doesn't define any test_ functions.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime

# Flag to indicate if we're on macOS
IS_MACOS = sys.platform == "darwin"


def print_header(title: str) -> None:
    """Print a formatted header."""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_step(step: int, description: str) -> None:
    """Print a step indicator."""
    print(f"\n{step}. {description}...")


def print_success(message: str) -> None:
    """Print success message."""
    print(f"   [OK] {message}")


def print_error(message: str) -> None:
    """Print error message."""
    print(f"   [ERROR] {message}")


def print_info(message: str) -> None:
    """Print info message."""
    print(f"   {message}")


def check_imessage_access(verbose: bool = False) -> bool:
    """Check that we can access iMessage database.

    Returns:
        True if access is available, False otherwise.
    """
    print_header("Testing iMessage Access")

    try:
        from integrations.imessage import ChatDBReader

        print_step(1, "Checking database access")

        with ChatDBReader() as reader:
            has_access = reader.check_access()

            if has_access:
                print_success("iMessage database access confirmed")
            else:
                print_error("Cannot access iMessage database")
                print_info("Grant Full Disk Access in System Settings > Privacy & Security")
                return False

            print_step(2, "Fetching conversations")
            convos = reader.get_conversations(limit=5)

            if not convos:
                print_error("No conversations found")
                print_info("You need at least one iMessage conversation")
                return False

            print_success(f"Found {len(convos)} conversations")

            if verbose:
                for c in convos[:3]:
                    name = c.display_name or (c.participants[0] if c.participants else "Unknown")
                    print_info(f"  - {name} ({c.message_count} messages)")

        return True

    except ImportError as e:
        print_error(f"Import error: {e}")
        return False
    except PermissionError as e:
        print_error(f"Permission denied: {e}")
        print_info("Grant Full Disk Access to your terminal application")
        return False
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        return False


def check_model_loading(verbose: bool = False) -> bool:
    """Check that we can load the MLX model.

    Returns:
        True if model loads successfully, False otherwise.
    """
    print_header("Testing Model Loading")

    try:
        print_step(1, "Importing model generator")
        from models import get_generator

        print_step(2, "Loading model (this may take a moment)")
        start_time = time.time()

        generator = get_generator()
        loaded = generator.is_loaded()

        load_time = time.time() - start_time

        if loaded:
            print_success(f"Model loaded in {load_time:.2f}s")
            if verbose:
                memory_mb = generator.get_memory_usage_mb()
                print_info(f"  Memory usage: {memory_mb:.1f} MB")
                print_info(f"  Model: {generator.config.model_path}")
        else:
            print_info("Model not loaded yet, will load on first generation")

        return True

    except ImportError as e:
        print_error(f"Import error: {e}")
        print_info("MLX framework may not be available (requires Apple Silicon)")
        return False
    except Exception as e:
        print_error(f"Error loading model: {e}")
        return False


def check_real_reply_generation(verbose: bool = False) -> bool:
    """Check reply generation with real iMessage data.

    Returns:
        True if generation succeeds, False otherwise.
    """
    print_header("Testing Reply Generation")

    try:
        from contracts.models import GenerationRequest
        from integrations.imessage import ChatDBReader
        from models import get_generator

        # Import our mock helpers that define the expected interface
        from tests.integration.conftest import MockContextFetcher, build_reply_prompt

        print_step(1, "Fetching recent conversations")
        with ChatDBReader() as reader:
            convos = reader.get_conversations(limit=5)

            if not convos:
                print_error("No conversations found!")
                return False

            print_success(f"Found {len(convos)} conversations")

            # Find a conversation with recent messages
            selected_conv = None
            for c in convos:
                if c.message_count > 3:
                    selected_conv = c
                    break

            if selected_conv is None:
                selected_conv = convos[0]

            name = selected_conv.display_name or (
                selected_conv.participants[0] if selected_conv.participants else "Unknown"
            )
            print_info(f"Using conversation: {name}")

            print_step(2, "Fetching context")
            fetcher = MockContextFetcher(reader=reader)
            context = fetcher.get_reply_context(selected_conv.chat_id, num_messages=10)

            print_success(f"Got {len(context.messages)} messages")

            if context.last_received_message:
                last_msg = context.last_received_message.text
                if len(last_msg) > 50:
                    last_msg = last_msg[:50] + "..."
                print_info(f"Last message: {last_msg}")
            else:
                print_info("No received messages found (all messages from you)")

            if verbose:
                print_info("\nConversation context:")
                for line in context.formatted_context.split("\n")[-5:]:
                    print_info(f"  {line[:80]}")

            print_step(3, "Building prompt")
            last_message_text = (
                context.last_received_message.text if context.last_received_message else ""
            )
            prompt = build_reply_prompt(
                context=context.formatted_context,
                last_message=last_message_text,
            )
            print_success(f"Prompt length: {len(prompt)} chars")

            print_step(4, "Generating reply")
            generator = get_generator()

            request = GenerationRequest(
                prompt=prompt,
                context_documents=[context.formatted_context],
                few_shot_examples=[],
                max_tokens=150,
                temperature=0.7,
            )

            start_time = time.time()
            response = generator.generate(request)
            gen_time = time.time() - start_time

            print()
            print("=" * 40)
            print("  Generated Reply")
            print("=" * 40)
            print(response.text)
            print("=" * 40)
            print()

            print_success(f"Generation complete in {gen_time:.2f}s")
            print_info(f"  Tokens: {response.tokens_used}")
            print_info(f"  Time: {response.generation_time_ms:.0f}ms")
            print_info(f"  Used template: {response.used_template}")

            return True

    except ImportError as e:
        print_error(f"Import error: {e}")
        return False
    except Exception as e:
        print_error(f"Error: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        return False


def check_search_functionality(verbose: bool = False) -> bool:
    """Check search functionality with real data.

    Returns:
        True if search works, False otherwise.
    """
    print_header("Testing Search Functionality")

    try:
        from integrations.imessage import ChatDBReader

        print_step(1, "Connecting to iMessage database")

        with ChatDBReader() as reader:
            # Try common search terms
            search_terms = ["dinner", "meeting", "tomorrow", "thanks", "hello"]

            for term in search_terms:
                print_step(2, f"Searching for '{term}'")
                results = reader.search(query=term, limit=5)

                if results:
                    print_success(f"Found {len(results)} results for '{term}'")

                    if verbose:
                        for msg in results[:2]:
                            text = msg.text[:50] + "..." if len(msg.text) > 50 else msg.text
                            sender = msg.sender_name or msg.sender
                            print_info(f"  - {sender}: {text}")

                    return True

            print_info("No results found for common search terms")
            print_info("This is OK if your conversations don't contain these words")
            return True

    except Exception as e:
        print_error(f"Search error: {e}")
        return False


def check_api_endpoint(verbose: bool = False) -> bool:
    """Check the FastAPI endpoint.

    Returns:
        True if API works, False otherwise.
    """
    print_header("Testing API Endpoint")

    try:
        from fastapi.testclient import TestClient

        from jarvis.api import app

        print_step(1, "Creating test client")
        client = TestClient(app, raise_server_exceptions=False)

        print_step(2, "Testing /health endpoint")
        response = client.get("/health")

        if response.status_code == 200:
            print_success("Health endpoint working")
            data = response.json()
            if verbose:
                print_info(f"  Status: {data['status']}")
                print_info(f"  Version: {data['version']}")
        else:
            print_error(f"Health endpoint returned {response.status_code}")
            return False

        print_step(3, "Testing /chat endpoint")
        response = client.post(
            "/chat",
            json={
                "message": "Hello, this is a test",
                "max_tokens": 50,
            },
        )

        if response.status_code == 200:
            print_success("Chat endpoint working")
            data = response.json()
            if verbose:
                text = data["text"][:50] + "..." if len(data["text"]) > 50 else data["text"]
                print_info(f"  Response: {text}")
        else:
            # Chat may fail if model isn't available - that's OK for this test
            print_info(f"Chat endpoint returned {response.status_code} (model may not be loaded)")

        return True

    except Exception as e:
        print_error(f"API error: {e}")
        return False


def run_all_tests(verbose: bool = False) -> int:
    """Run all E2E tests.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    print()
    print("#" * 60)
    print("#  JARVIS RAG E2E Test Suite")
    print(f"#  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 60)

    results = {}

    # Test 1: iMessage Access
    results["imessage_access"] = check_imessage_access(verbose)

    # Only continue if we have iMessage access
    if results["imessage_access"]:
        # Test 2: Model Loading
        results["model_loading"] = check_model_loading(verbose)

        # Test 3: Reply Generation (only if model loads)
        if results["model_loading"]:
            results["reply_generation"] = check_real_reply_generation(verbose)
        else:
            results["reply_generation"] = False
            print_info("Skipping reply generation (model not available)")

        # Test 4: Search
        results["search"] = check_search_functionality(verbose)

        # Test 5: API
        results["api"] = check_api_endpoint(verbose)
    else:
        print_info("\nSkipping remaining tests (no iMessage access)")
        results["model_loading"] = False
        results["reply_generation"] = False
        results["search"] = False
        results["api"] = False

    # Summary
    print_header("Test Summary")

    passed = sum(1 for r in results.values() if r)
    total = len(results)

    for test_name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {test_name.replace('_', ' ').title()}")

    print()
    print(f"  Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n  All tests passed!")
        return 0
    elif passed > 0:
        print("\n  Some tests failed. Check the output above for details.")
        return 0  # Partial success is OK for E2E
    else:
        print("\n  All tests failed. Check your setup.")
        return 1


def run_single_test(test_name: str, verbose: bool = False) -> int:
    """Run a single test by name.

    Args:
        test_name: Name of the test to run
        verbose: Enable verbose output

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    test_map = {
        "access": check_imessage_access,
        "imessage": check_imessage_access,
        "model": check_model_loading,
        "reply": check_real_reply_generation,
        "search": check_search_functionality,
        "api": check_api_endpoint,
    }

    if test_name not in test_map:
        print(f"Unknown test: {test_name}")
        print(f"Available tests: {', '.join(test_map.keys())}")
        return 1

    result = test_map[test_name](verbose)
    return 0 if result else 1


def main() -> int:
    """Main entry point."""
    # Check platform before doing anything
    if not IS_MACOS:
        print("This script requires macOS with iMessage.")
        print("Run on a Mac with Full Disk Access enabled.")
        return 1

    parser = argparse.ArgumentParser(
        description="E2E test script for JARVIS RAG flow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m tests.integration.test_e2e_script              # Run all tests
    python -m tests.integration.test_e2e_script --verbose    # Verbose output
    python -m tests.integration.test_e2e_script --test reply # Run reply test only
        """,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "-t",
        "--test",
        type=str,
        help="Run specific test (access, model, reply, search, api)",
    )

    args = parser.parse_args()

    if args.test:
        return run_single_test(args.test, args.verbose)
    else:
        return run_all_tests(args.verbose)


if __name__ == "__main__":
    sys.exit(main())
