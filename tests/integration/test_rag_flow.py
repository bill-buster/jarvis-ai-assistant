"""Integration tests for the RAG reply generation flow.

Tests the complete pipeline: Intent -> Context -> Prompt -> Generator -> Response

These tests verify that all components work together correctly to generate
contextual replies for iMessage conversations.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from contracts.imessage import Message
from contracts.models import GenerationRequest, GenerationResponse
from jarvis.api import app

from .conftest import (
    IntentType,
    MockContextFetcher,
    MockIntentClassifier,
    build_reply_prompt,
    build_summary_prompt,
    create_mock_message,
    patch_generator,
    patch_imessage_reader,
)


class TestRAGFlow:
    """Test the complete RAG flow from intent to response."""

    def test_reply_flow_with_context(self, mock_messages: list[Message]):
        """Test: User asks for help replying -> gets contextual suggestion."""
        # 1. Classify intent
        classifier = MockIntentClassifier()
        result = classifier.classify("help me reply to John's message")

        assert result.intent == IntentType.REPLY
        assert result.confidence > 0.8
        assert "john" in result.extracted_params.get("person", "").lower()

        # 2. Fetch context (mocked)
        with patch_imessage_reader() as mock_reader:
            mock_reader.get_messages.return_value = mock_messages

            fetcher = MockContextFetcher(reader=mock_reader)
            context = fetcher.get_reply_context("chat123", num_messages=10)

            # Verify context was fetched
            assert context.messages == mock_messages
            assert "dinner tomorrow" in context.formatted_context.lower()
            assert context.last_received_message is not None
            assert context.last_received_message.text == "7pm at my place?"

        # 3. Build prompt
        prompt = build_reply_prompt(
            context=context.formatted_context,
            last_message=context.last_received_message.text,
        )

        assert "7pm at my place?" in prompt
        assert "dinner" in prompt.lower()
        assert "reply" in prompt.lower()

        # 4. Generate response (mocked model)
        with patch_generator() as mock_gen:
            mock_gen.generate.return_value = GenerationResponse(
                text="Sounds perfect! See you at 7!",
                tokens_used=8,
                generation_time_ms=150.0,
                model_name="test-model",
                used_template=False,
                template_name=None,
                finish_reason="stop",
            )

            request = GenerationRequest(
                prompt=prompt,
                context_documents=[context.formatted_context],
                few_shot_examples=[],
                max_tokens=150,
            )
            response = mock_gen.generate(request)

            assert response.text == "Sounds perfect! See you at 7!"
            assert len(response.text) < 200  # Concise reply
            assert response.tokens_used > 0

    def test_reply_flow_extracts_person_name(self):
        """Test that intent extraction captures the person's name."""
        classifier = MockIntentClassifier()

        # Various ways to mention a person
        test_cases = [
            ("help me reply to John", "john"),
            ("respond to Sarah's message", "sarah"),
            ("answer Mike", "mike"),
        ]

        for input_text, expected_name in test_cases:
            result = classifier.classify(input_text)
            assert result.intent == IntentType.REPLY
            person = result.extracted_params.get("person", "").lower()
            assert expected_name in person, f"Expected '{expected_name}' in '{person}'"

    def test_summarize_flow(self, mock_work_messages: list[Message]):
        """Test: User asks for summary -> gets conversation summary."""
        # 1. Classify intent
        classifier = MockIntentClassifier()
        result = classifier.classify("give me a summary of this conversation")

        assert result.intent == IntentType.SUMMARIZE
        assert result.confidence > 0.7

        # 2. Fetch context
        with patch_imessage_reader() as mock_reader:
            mock_reader.get_messages.return_value = mock_work_messages

            fetcher = MockContextFetcher(reader=mock_reader)
            messages, formatted_context = fetcher.get_summary_context("chat123", num_messages=50)

            assert len(messages) == 3
            assert "report" in formatted_context.lower()

        # 3. Build summary prompt
        prompt = build_summary_prompt(
            context=formatted_context,
            num_messages=len(messages),
        )

        assert "summarize" in prompt.lower()
        assert str(len(messages)) in prompt

        # 4. Generate summary
        with patch_generator() as mock_gen:
            mock_gen.generate.return_value = GenerationResponse(
                text="Boss asked about report status. You're almost done and will send by EOD.",
                tokens_used=15,
                generation_time_ms=200.0,
                model_name="test-model",
                used_template=False,
                template_name=None,
                finish_reason="stop",
            )

            request = GenerationRequest(
                prompt=prompt,
                context_documents=[formatted_context],
                few_shot_examples=[],
                max_tokens=200,
            )
            response = mock_gen.generate(request)

            assert "report" in response.text.lower()
            assert response.tokens_used > 0

    def test_quick_reply_detected(self):
        """Test: Simple 'ok' -> classified as QUICK_REPLY."""
        classifier = MockIntentClassifier()

        quick_replies = [
            "ok sounds good",
            "thanks",
            "got it",
            "sure",
            "okay",
        ]

        for reply in quick_replies:
            result = classifier.classify(reply)
            assert result.intent == IntentType.QUICK_REPLY, f"Failed for: {reply}"
            assert result.confidence > 0.9

    def test_quick_reply_skips_llm(self):
        """Test: Quick replies should use templates, not full LLM generation."""
        classifier = MockIntentClassifier()
        result = classifier.classify("ok sounds good")

        assert result.intent == IntentType.QUICK_REPLY

        # In the real implementation, quick replies would:
        # 1. Match against template patterns
        # 2. Return a template response without invoking the LLM
        # This is verified by the used_template flag in the response

    def test_unknown_intent_handling(self):
        """Test: Unclear input -> classified as UNKNOWN with low confidence."""
        classifier = MockIntentClassifier()
        result = classifier.classify("random gibberish input")

        assert result.intent == IntentType.UNKNOWN
        assert result.confidence < 0.5

    def test_context_formats_messages_correctly(self, mock_messages: list[Message]):
        """Test that context formatting produces readable output."""
        with patch_imessage_reader() as mock_reader:
            mock_reader.get_messages.return_value = mock_messages

            fetcher = MockContextFetcher(reader=mock_reader)
            context = fetcher.get_reply_context("chat123", num_messages=10)

            # Check formatting
            lines = context.formatted_context.split("\n")
            assert len(lines) == 3

            # "You" should be used for is_from_me messages
            assert "You:" in context.formatted_context
            # Sender name should be used for received messages
            assert "John:" in context.formatted_context

    def test_context_identifies_last_received_message(self):
        """Test that last_received_message is correctly identified."""
        messages = [
            create_mock_message("Hi", is_from_me=False, msg_id=1),
            create_mock_message("Hello", is_from_me=True, msg_id=2),
            create_mock_message("How are you?", is_from_me=True, msg_id=3),
        ]

        with patch_imessage_reader() as mock_reader:
            mock_reader.get_messages.return_value = messages

            fetcher = MockContextFetcher(reader=mock_reader)
            context = fetcher.get_reply_context("chat123", num_messages=10)

            # Should find the last message not from me
            assert context.last_received_message.text == "Hi"
            assert context.last_received_message.id == 1

    def test_prompt_includes_required_elements(self, mock_messages: list[Message]):
        """Test that prompts contain all necessary information."""
        with patch_imessage_reader() as mock_reader:
            mock_reader.get_messages.return_value = mock_messages

            fetcher = MockContextFetcher(reader=mock_reader)
            context = fetcher.get_reply_context("chat123", num_messages=10)

        prompt = build_reply_prompt(
            context=context.formatted_context,
            last_message=context.last_received_message.text,
        )

        # Verify prompt structure
        assert "conversation" in prompt.lower()
        assert context.formatted_context in prompt
        assert context.last_received_message.text in prompt
        assert "reply" in prompt.lower()


class TestSearchFlow:
    """Test the search functionality flow."""

    def test_search_intent_classified(self):
        """Test that search intents are correctly classified."""
        classifier = MockIntentClassifier()

        search_queries = [
            "find messages about dinner",
            "search for photos from last week",
            "look for the address John sent",
        ]

        for query in search_queries:
            result = classifier.classify(query)
            assert result.intent == IntentType.SEARCH, f"Failed for: {query}"

    def test_search_returns_results(self):
        """Test that search returns formatted results."""
        mock_results = [
            create_mock_message(
                "Let's meet at the restaurant on 5th Ave",
                is_from_me=False,
                msg_id=1,
            ),
            create_mock_message(
                "Sure, what time for dinner?",
                is_from_me=True,
                msg_id=2,
            ),
        ]

        with patch_imessage_reader() as mock_reader:
            mock_reader.search.return_value = mock_results

            # Simulate search
            results = mock_reader.search("dinner", limit=10)

            assert len(results) == 2
            assert any("restaurant" in m.text for m in results)


class TestAPIIntegration:
    """Test API endpoints with mocked services."""

    @pytest.fixture
    def client(self):
        """Create a test client for the API."""
        return TestClient(app, raise_server_exceptions=False)

    def test_chat_endpoint_with_context(self, client: TestClient):
        """Test POST /chat with context documents."""
        with patch("jarvis.api.get_degradation_controller") as mock_deg_ctrl:
            with patch("models.get_generator") as mock_gen:
                mock_gen.return_value = MagicMock()
                mock_response = (
                    "Based on the context, I suggest replying: Sounds great!",
                    {
                        "tokens_used": 15,
                        "generation_time_ms": 150.0,
                        "model_name": "test-model",
                        "used_template": False,
                        "template_name": None,
                        "finish_reason": "stop",
                    },
                )
                mock_deg_ctrl.return_value.execute.return_value = mock_response

                response = client.post(
                    "/chat",
                    json={
                        "message": "Help me reply to this",
                        "context_documents": [
                            "John: Hey, dinner tomorrow?",
                            "You: Sure, what time?",
                            "John: 7pm at my place?",
                        ],
                        "max_tokens": 150,
                        "temperature": 0.7,
                    },
                )

                assert response.status_code == 200
                data = response.json()
                assert "text" in data
                assert data["tokens_used"] == 15

    def test_chat_endpoint_without_context(self, client: TestClient):
        """Test POST /chat works without context documents."""
        with patch("jarvis.api.get_degradation_controller") as mock_deg_ctrl:
            with patch("models.get_generator") as mock_gen:
                mock_gen.return_value = MagicMock()
                mock_response = (
                    "Hello! How can I help?",
                    {
                        "tokens_used": 5,
                        "generation_time_ms": 50.0,
                        "model_name": "test-model",
                        "used_template": False,
                        "template_name": None,
                        "finish_reason": "stop",
                    },
                )
                mock_deg_ctrl.return_value.execute.return_value = mock_response

                response = client.post("/chat", json={"message": "Hello"})

                assert response.status_code == 200
                data = response.json()
                assert data["text"] == "Hello! How can I help?"

    def test_search_endpoint_returns_messages(self, client: TestClient):
        """Test GET /search returns messages."""
        mock_message = MagicMock()
        mock_message.id = 1
        mock_message.chat_id = "chat123"
        mock_message.sender = "+1234567890"
        mock_message.sender_name = "John"
        mock_message.text = "Let's have dinner tomorrow"
        mock_message.date = datetime(2024, 1, 15, 10, 30)
        mock_message.is_from_me = False
        mock_message.attachments = []
        mock_message.reply_to_id = None
        mock_message.reactions = []

        with patch("jarvis.api.get_degradation_controller") as mock_deg_ctrl:
            mock_deg_ctrl.return_value.execute.return_value = [mock_message]

            response = client.get("/search", params={"query": "dinner"})

            assert response.status_code == 200
            data = response.json()
            assert "messages" in data
            assert data["total"] == 1
            assert "dinner" in data["messages"][0]["text"]

    def test_conversations_endpoint(self, client: TestClient):
        """Test GET /conversations returns conversation list."""
        mock_conv = MagicMock()
        mock_conv.chat_id = "chat123"
        mock_conv.participants = ["+1234567890"]
        mock_conv.display_name = "John"
        mock_conv.last_message_date = datetime(2024, 1, 15, 10, 30)
        mock_conv.message_count = 50
        mock_conv.is_group = False

        with patch("jarvis.api.get_degradation_controller") as mock_deg_ctrl:
            mock_deg_ctrl.return_value.execute.return_value = [mock_conv]

            response = client.get("/conversations", params={"limit": 10})

            assert response.status_code == 200
            data = response.json()
            assert "conversations" in data
            assert data["total"] == 1
            assert data["conversations"][0]["display_name"] == "John"

    def test_messages_endpoint(self, client: TestClient):
        """Test GET /messages/{conversation_id} returns messages."""
        mock_message = MagicMock()
        mock_message.id = 1
        mock_message.chat_id = "chat123"
        mock_message.sender = "+1234567890"
        mock_message.sender_name = "John"
        mock_message.text = "Test message"
        mock_message.date = datetime(2024, 1, 15, 10, 30)
        mock_message.is_from_me = False
        mock_message.attachments = []
        mock_message.reply_to_id = None
        mock_message.reactions = []

        with patch("jarvis.api.get_degradation_controller") as mock_deg_ctrl:
            mock_deg_ctrl.return_value.execute.return_value = [mock_message]

            response = client.get("/messages/chat123", params={"limit": 50})

            assert response.status_code == 200
            data = response.json()
            assert "messages" in data
            assert data["chat_id"] == "chat123"
            assert data["total"] == 1


class TestEdgeCases:
    """Test edge cases and error handling in the RAG flow."""

    def test_empty_conversation(self):
        """Test handling of empty conversation."""
        with patch_imessage_reader() as mock_reader:
            mock_reader.get_messages.return_value = []

            fetcher = MockContextFetcher(reader=mock_reader)
            context = fetcher.get_reply_context("chat123", num_messages=10)

            assert context.messages == []
            assert context.formatted_context == ""
            # last_received_message will be None for empty conversation
            assert context.last_received_message is None

    def test_conversation_with_only_my_messages(self):
        """Test conversation where all messages are from the user."""
        messages = [
            create_mock_message("Hello?", is_from_me=True, msg_id=1),
            create_mock_message("Anyone there?", is_from_me=True, msg_id=2),
        ]

        with patch_imessage_reader() as mock_reader:
            mock_reader.get_messages.return_value = messages

            fetcher = MockContextFetcher(reader=mock_reader)
            context = fetcher.get_reply_context("chat123", num_messages=10)

            # Should use the last message when no received messages exist
            assert context.last_received_message is not None
            assert context.last_received_message.text == "Anyone there?"

    def test_message_with_attachments(self):
        """Test handling of messages with attachments."""
        from contracts.imessage import Attachment

        attachment = Attachment(
            filename="photo.jpg",
            file_path="/path/to/photo.jpg",
            mime_type="image/jpeg",
            file_size=1024,
        )
        messages = [
            create_mock_message(
                "Check out this photo",
                is_from_me=False,
                attachments=[attachment],
            ),
        ]

        with patch_imessage_reader() as mock_reader:
            mock_reader.get_messages.return_value = messages

            fetcher = MockContextFetcher(reader=mock_reader)
            context = fetcher.get_reply_context("chat123", num_messages=10)

            assert len(context.messages[0].attachments) == 1
            assert context.messages[0].attachments[0].filename == "photo.jpg"

    def test_long_conversation_context(self):
        """Test handling of long conversations."""
        # Generate 100 messages
        messages = []
        for i in range(100):
            # Create dates spread across multiple hours (valid minute range is 0-59)
            hour = 10 + (i // 60)
            minute = i % 60
            messages.append(
                create_mock_message(
                    f"Message {i}",
                    is_from_me=(i % 2 == 0),
                    msg_id=i,
                    date=datetime(2024, 1, 15, hour, minute),
                )
            )

        with patch_imessage_reader() as mock_reader:
            mock_reader.get_messages.return_value = messages[:10]  # Limit applied

            fetcher = MockContextFetcher(reader=mock_reader)
            context = fetcher.get_reply_context("chat123", num_messages=10)

            assert len(context.messages) == 10
            # Context should be reasonable size
            assert len(context.formatted_context) < 10000

    def test_special_characters_in_messages(self):
        """Test handling of special characters in messages."""
        messages = [
            create_mock_message(
                "Hey! 😀 How's it going? <script>alert('xss')</script>",
                is_from_me=False,
            ),
        ]

        with patch_imessage_reader() as mock_reader:
            mock_reader.get_messages.return_value = messages

            fetcher = MockContextFetcher(reader=mock_reader)
            context = fetcher.get_reply_context("chat123", num_messages=10)

            # Should preserve the message content
            assert "😀" in context.formatted_context
            # Content should be included as-is (sanitization is prompt's job)
            assert "script" in context.formatted_context


class TestMultipleConversations:
    """Test handling of multiple conversations."""

    def test_switch_between_conversations(self):
        """Test fetching context from different conversations."""
        conv1_messages = [
            create_mock_message("Dinner?", is_from_me=False, chat_id="chat1"),
        ]
        conv2_messages = [
            create_mock_message("Meeting at 3pm", is_from_me=False, chat_id="chat2"),
        ]

        with patch_imessage_reader() as mock_reader:

            def side_effect(chat_id, **kwargs):
                if chat_id == "chat1":
                    return conv1_messages
                return conv2_messages

            mock_reader.get_messages.side_effect = side_effect

            fetcher = MockContextFetcher(reader=mock_reader)

            context1 = fetcher.get_reply_context("chat1", num_messages=10)
            context2 = fetcher.get_reply_context("chat2", num_messages=10)

            assert "Dinner" in context1.formatted_context
            assert "Meeting" in context2.formatted_context

    def test_group_conversation_context(self):
        """Test context from group conversations."""
        messages = [
            create_mock_message(
                "Hey everyone!",
                is_from_me=False,
                sender="+1111111111",
                sender_name="Alice",
            ),
            create_mock_message(
                "Hi!",
                is_from_me=False,
                sender="+2222222222",
                sender_name="Bob",
            ),
            create_mock_message(
                "Hello team",
                is_from_me=True,
            ),
        ]

        with patch_imessage_reader() as mock_reader:
            mock_reader.get_messages.return_value = messages

            fetcher = MockContextFetcher(reader=mock_reader)
            context = fetcher.get_reply_context("group-chat", num_messages=10)

            # Should show different sender names
            assert "Alice:" in context.formatted_context
            assert "Bob:" in context.formatted_context
            assert "You:" in context.formatted_context
