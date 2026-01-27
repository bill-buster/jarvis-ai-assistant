"""Tests for MCP server functionality.

Tests the Model Context Protocol server implementation including
tool definitions, handlers, and JSON-RPC protocol handling.
"""

from __future__ import annotations

import json
from io import StringIO
from unittest.mock import MagicMock, patch

from mcp_server.handlers import (
    ToolResult,
    execute_tool,
    handle_get_contact_info,
    handle_list_conversations,
    handle_search_messages,
)
from mcp_server.server import (
    JSONRPCError,
    JSONRPCRequest,
    JSONRPCResponse,
    MCPServer,
    StdioTransport,
)
from mcp_server.tools import get_tool_by_name, get_tool_definitions


class TestToolDefinitions:
    """Tests for MCP tool definitions."""

    def test_get_tool_definitions_returns_list(self) -> None:
        """Test that get_tool_definitions returns a list of tools."""
        tools = get_tool_definitions()
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_all_tools_have_required_fields(self) -> None:
        """Test that all tools have name, description, and inputSchema."""
        tools = get_tool_definitions()
        for tool in tools:
            assert "name" in tool, f"Tool missing name: {tool}"
            assert "description" in tool, f"Tool {tool.get('name')} missing description"
            assert "inputSchema" in tool, f"Tool {tool['name']} missing inputSchema"

    def test_input_schemas_are_valid(self) -> None:
        """Test that all input schemas have required JSON Schema fields."""
        tools = get_tool_definitions()
        for tool in tools:
            schema = tool["inputSchema"]
            assert "type" in schema, f"Tool {tool['name']} schema missing type"
            assert schema["type"] == "object", f"Tool {tool['name']} schema type not object"
            assert "properties" in schema, f"Tool {tool['name']} schema missing properties"

    def test_get_tool_by_name_existing(self) -> None:
        """Test getting a tool by name that exists."""
        tool = get_tool_by_name("search_messages")
        assert tool is not None
        assert tool["name"] == "search_messages"

    def test_get_tool_by_name_nonexistent(self) -> None:
        """Test getting a tool by name that doesn't exist."""
        tool = get_tool_by_name("nonexistent_tool")
        assert tool is None

    def test_search_messages_tool_schema(self) -> None:
        """Test search_messages tool has correct schema."""
        tool = get_tool_by_name("search_messages")
        assert tool is not None
        schema = tool["inputSchema"]
        assert "query" in schema["properties"]
        assert "required" in schema
        assert "query" in schema["required"]

    def test_get_summary_tool_schema(self) -> None:
        """Test get_summary tool has correct schema."""
        tool = get_tool_by_name("get_summary")
        assert tool is not None
        schema = tool["inputSchema"]
        assert "person_name" in schema["properties"]
        assert "chat_id" in schema["properties"]
        assert "num_messages" in schema["properties"]

    def test_generate_reply_tool_schema(self) -> None:
        """Test generate_reply tool has correct schema."""
        tool = get_tool_by_name("generate_reply")
        assert tool is not None
        schema = tool["inputSchema"]
        assert "person_name" in schema["properties"]
        assert "instruction" in schema["properties"]
        assert "num_suggestions" in schema["properties"]


class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_success_result(self) -> None:
        """Test creating a successful result."""
        result = ToolResult(success=True, data={"key": "value"})
        assert result.success is True
        assert result.data == {"key": "value"}
        assert result.error is None

    def test_error_result(self) -> None:
        """Test creating an error result."""
        result = ToolResult(success=False, error="Something went wrong")
        assert result.success is False
        assert result.error == "Something went wrong"
        assert result.data is None


class TestToolHandlers:
    """Tests for MCP tool handlers."""

    def test_execute_tool_unknown_tool(self) -> None:
        """Test executing an unknown tool returns error."""
        result = execute_tool("nonexistent_tool", {})
        assert result.success is False
        assert "Unknown tool" in result.error

    @patch("mcp_server.handlers._check_imessage_access")
    def test_search_messages_no_access(self, mock_check: MagicMock) -> None:
        """Test search_messages when iMessage is not accessible."""
        mock_check.return_value = False
        result = handle_search_messages({"query": "test"})
        assert result.success is False
        assert "Full Disk Access" in result.error

    def test_search_messages_missing_query(self) -> None:
        """Test search_messages with missing query parameter."""
        with patch("mcp_server.handlers._check_imessage_access", return_value=True):
            result = handle_search_messages({})
            assert result.success is False
            assert "required" in result.error.lower()

    @patch("mcp_server.handlers._check_imessage_access")
    def test_list_conversations_no_access(self, mock_check: MagicMock) -> None:
        """Test list_conversations when iMessage is not accessible."""
        mock_check.return_value = False
        result = handle_list_conversations({})
        assert result.success is False
        assert "Full Disk Access" in result.error

    def test_get_contact_info_missing_identifier(self) -> None:
        """Test get_contact_info with missing identifier."""
        result = handle_get_contact_info({})
        assert result.success is False
        assert "required" in result.error.lower()

    def test_get_contact_info_success(self) -> None:
        """Test get_contact_info with valid identifier."""
        with patch(
            "integrations.imessage.parser.normalize_phone_number", return_value="+15551234567"
        ):
            with patch("integrations.imessage.get_contact_avatar", return_value=None):
                result = handle_get_contact_info({"identifier": "5551234567"})
                assert result.success is True
                assert result.data is not None
                assert "identifier" in result.data

    def test_get_contact_info_invalid_phone(self) -> None:
        """Test get_contact_info with invalid phone number."""
        with patch("integrations.imessage.parser.normalize_phone_number", return_value=None):
            result = handle_get_contact_info({"identifier": "invalid"})
            assert result.success is False
            assert "Invalid phone number" in result.error


class TestJSONRPCProtocol:
    """Tests for JSON-RPC 2.0 protocol handling."""

    def test_jsonrpc_response_to_dict_success(self) -> None:
        """Test converting successful response to dict."""
        response = JSONRPCResponse(result={"data": "test"}, id=1)
        d = response.to_dict()
        assert d["jsonrpc"] == "2.0"
        assert d["id"] == 1
        assert d["result"] == {"data": "test"}
        assert "error" not in d

    def test_jsonrpc_response_to_dict_error(self) -> None:
        """Test converting error response to dict."""
        response = JSONRPCResponse(
            error={"code": -32600, "message": "Invalid Request"},
            id=1,
        )
        d = response.to_dict()
        assert d["jsonrpc"] == "2.0"
        assert d["id"] == 1
        assert d["error"]["code"] == -32600
        assert "result" not in d


class TestMCPServer:
    """Tests for the MCP server."""

    def test_server_initialization(self) -> None:
        """Test server initializes correctly."""
        server = MCPServer()
        assert not server._initialized
        assert server._client_info is None

    def test_parse_request_valid(self) -> None:
        """Test parsing a valid JSON-RPC request."""
        server = MCPServer()
        data = '{"jsonrpc":"2.0","method":"ping","id":1}'
        result = server.parse_request(data)
        assert isinstance(result, JSONRPCRequest)
        assert result.method == "ping"
        assert result.id == 1

    def test_parse_request_invalid_json(self) -> None:
        """Test parsing invalid JSON returns error."""
        server = MCPServer()
        result = server.parse_request("not valid json")
        assert isinstance(result, JSONRPCResponse)
        assert result.error is not None
        assert result.error["code"] == JSONRPCError.PARSE_ERROR

    def test_parse_request_missing_method(self) -> None:
        """Test parsing request without method returns error."""
        server = MCPServer()
        result = server.parse_request('{"jsonrpc":"2.0","id":1}')
        assert isinstance(result, JSONRPCResponse)
        assert result.error is not None
        assert result.error["code"] == JSONRPCError.INVALID_REQUEST

    def test_handle_initialize(self) -> None:
        """Test handling initialize request."""
        server = MCPServer()
        request = JSONRPCRequest(
            method="initialize",
            params={"clientInfo": {"name": "test-client", "version": "1.0"}},
            id=1,
        )
        response = server.handle_request(request)
        assert response is not None
        assert response.result is not None
        assert "protocolVersion" in response.result
        assert "capabilities" in response.result
        assert "serverInfo" in response.result
        assert server._initialized is True

    def test_handle_initialized_notification(self) -> None:
        """Test handling initialized notification returns None."""
        server = MCPServer()
        request = JSONRPCRequest(method="initialized", params={})
        response = server.handle_request(request)
        assert response is None  # Notifications don't return responses

    def test_handle_tools_list(self) -> None:
        """Test handling tools/list request."""
        server = MCPServer()
        # Initialize server first (required by Phase 2 security fix)
        init_request = JSONRPCRequest(method="initialize", params={"protocolVersion": "1.0"}, id=0)
        server.handle_request(init_request)
        # Now test tools/list
        request = JSONRPCRequest(method="tools/list", params={}, id=1)
        response = server.handle_request(request)
        assert response is not None
        assert response.result is not None
        assert "tools" in response.result
        assert len(response.result["tools"]) > 0

    def test_handle_ping(self) -> None:
        """Test handling ping request."""
        server = MCPServer()
        request = JSONRPCRequest(method="ping", id=1)
        response = server.handle_request(request)
        assert response is not None
        assert response.result == {}

    def test_handle_unknown_method(self) -> None:
        """Test handling unknown method returns error."""
        server = MCPServer()
        request = JSONRPCRequest(method="unknown/method", id=1)
        response = server.handle_request(request)
        assert response is not None
        assert response.error is not None
        assert response.error["code"] == JSONRPCError.METHOD_NOT_FOUND

    @patch("mcp_server.handlers._check_imessage_access")
    def test_handle_tools_call_no_access(self, mock_check: MagicMock) -> None:
        """Test tools/call when iMessage not accessible."""
        mock_check.return_value = False
        server = MCPServer()
        # Initialize server first (required by Phase 2 security fix)
        init_request = JSONRPCRequest(method="initialize", params={"protocolVersion": "1.0"}, id=0)
        server.handle_request(init_request)
        # Now test tools/call
        request = JSONRPCRequest(
            method="tools/call",
            params={"name": "search_messages", "arguments": {"query": "test"}},
            id=1,
        )
        response = server.handle_request(request)
        assert response is not None
        assert response.result is not None
        assert response.result.get("isError") is True

    def test_handle_tools_call_missing_name(self) -> None:
        """Test tools/call without tool name returns error."""
        server = MCPServer()
        # Initialize server first (required by Phase 2 security fix)
        init_request = JSONRPCRequest(method="initialize", params={"protocolVersion": "1.0"}, id=0)
        server.handle_request(init_request)
        # Now test tools/call
        request = JSONRPCRequest(
            method="tools/call",
            params={"arguments": {}},
            id=1,
        )
        response = server.handle_request(request)
        assert response is not None
        assert response.result.get("isError") is True


class TestStdioTransport:
    """Tests for stdio transport."""

    def test_transport_initialization(self) -> None:
        """Test transport initializes with custom streams."""
        server = MCPServer()
        input_stream = StringIO()
        output_stream = StringIO()
        transport = StdioTransport(server, input_stream, output_stream)
        assert transport.server is server
        assert transport.input_stream is input_stream
        assert transport.output_stream is output_stream

    def test_send_response(self) -> None:
        """Test sending a response writes to output stream."""
        server = MCPServer()
        output_stream = StringIO()
        transport = StdioTransport(server, StringIO(), output_stream)

        response = JSONRPCResponse(result={"test": "data"}, id=1)
        transport.send_response(response)

        output_stream.seek(0)
        line = output_stream.readline()
        parsed = json.loads(line)
        assert parsed["result"] == {"test": "data"}
        assert parsed["id"] == 1

    def test_run_processes_message(self) -> None:
        """Test run processes a single message."""
        server = MCPServer()
        input_data = '{"jsonrpc":"2.0","method":"ping","id":1}\n'
        input_stream = StringIO(input_data)
        output_stream = StringIO()
        transport = StdioTransport(server, input_stream, output_stream)

        transport.run()

        output_stream.seek(0)
        line = output_stream.readline()
        parsed = json.loads(line)
        assert parsed["result"] == {}
        assert parsed["id"] == 1

    def test_run_handles_empty_lines(self) -> None:
        """Test run ignores empty lines."""
        server = MCPServer()
        input_data = '\n\n{"jsonrpc":"2.0","method":"ping","id":1}\n'
        input_stream = StringIO(input_data)
        output_stream = StringIO()
        transport = StdioTransport(server, input_stream, output_stream)

        transport.run()

        output_stream.seek(0)
        line = output_stream.readline()
        parsed = json.loads(line)
        assert parsed["id"] == 1

    def test_run_handles_invalid_json(self) -> None:
        """Test run handles invalid JSON gracefully."""
        server = MCPServer()
        input_data = "not valid json\n"
        input_stream = StringIO(input_data)
        output_stream = StringIO()
        transport = StdioTransport(server, input_stream, output_stream)

        transport.run()

        output_stream.seek(0)
        line = output_stream.readline()
        parsed = json.loads(line)
        assert "error" in parsed
        assert parsed["error"]["code"] == JSONRPCError.PARSE_ERROR
