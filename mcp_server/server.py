"""MCP Server implementation for JARVIS.

Implements the Model Context Protocol (MCP) server that exposes JARVIS
functionality as tools for Claude Code and other MCP clients.

The server communicates via JSON-RPC 2.0 over either:
- stdio (for direct integration with Claude Code)
- HTTP (for network-accessible integration)

Protocol Reference: https://modelcontextprotocol.io/specification
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from typing import Any, TextIO

from mcp_server.handlers import execute_tool
from mcp_server.tools import get_tool_definitions

logger = logging.getLogger(__name__)

# MCP Protocol version
MCP_PROTOCOL_VERSION = "2024-11-05"

# Server info
SERVER_NAME = "jarvis-mcp-server"
SERVER_VERSION = "1.0.0"


@dataclass
class JSONRPCRequest:
    """JSON-RPC 2.0 request."""

    method: str
    params: dict[str, Any] | list[Any] | None = None
    id: int | str | None = None
    jsonrpc: str = "2.0"


@dataclass
class JSONRPCResponse:
    """JSON-RPC 2.0 response."""

    result: Any = None
    error: dict[str, Any] | None = None
    id: int | str | None = None
    jsonrpc: str = "2.0"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d: dict[str, Any] = {"jsonrpc": self.jsonrpc, "id": self.id}
        if self.error is not None:
            d["error"] = self.error
        else:
            d["result"] = self.result
        return d


@dataclass
class JSONRPCError:
    """JSON-RPC 2.0 error codes."""

    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603


class MCPServer:
    """Model Context Protocol server for JARVIS.

    Exposes JARVIS iMessage functionality as MCP tools that can be
    used by Claude Code or other MCP-compatible clients.
    """

    def __init__(self) -> None:
        """Initialize the MCP server."""
        self._initialized = False
        self._client_info: dict[str, Any] | None = None

    def _handle_initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle the initialize request.

        Args:
            params: Initialize parameters from client.

        Returns:
            Server capabilities response.
        """
        self._client_info = params.get("clientInfo")
        self._initialized = True

        logger.info(
            "MCP server initialized. Client: %s",
            self._client_info.get("name") if self._client_info else "unknown",
        )

        return {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {
                "tools": {},  # We support tools
            },
            "serverInfo": {
                "name": SERVER_NAME,
                "version": SERVER_VERSION,
            },
        }

    def _handle_initialized(self, params: dict[str, Any] | None) -> None:
        """Handle the initialized notification.

        Args:
            params: Notification parameters (usually empty).
        """
        logger.info("MCP connection established")

    def _handle_tools_list(self, params: dict[str, Any] | None) -> dict[str, Any]:
        """Handle tools/list request.

        Args:
            params: Request parameters (may include cursor for pagination).

        Returns:
            List of available tools.
        """
        tools = get_tool_definitions()
        return {"tools": tools}

    def _handle_tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle tools/call request.

        Args:
            params: Request parameters including tool name and arguments.

        Returns:
            Tool execution result.
        """
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if not tool_name:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Error: Tool name is required",
                    }
                ],
                "isError": True,
            }

        logger.info("Executing tool: %s", tool_name)
        result = execute_tool(tool_name, arguments)

        if result.success:
            # Format successful result
            if isinstance(result.data, dict):
                text = json.dumps(result.data, indent=2, default=str)
            elif isinstance(result.data, list):
                text = json.dumps(result.data, indent=2, default=str)
            else:
                text = str(result.data)

            return {
                "content": [
                    {
                        "type": "text",
                        "text": text,
                    }
                ],
            }
        else:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Error: {result.error}",
                    }
                ],
                "isError": True,
            }

    def handle_request(self, request: JSONRPCRequest) -> JSONRPCResponse | None:
        """Handle a JSON-RPC request.

        Args:
            request: The incoming JSON-RPC request.

        Returns:
            JSON-RPC response or None for notifications.
        """
        method = request.method
        params = request.params or {}
        if isinstance(params, list):
            params = {}

        try:
            # Method dispatch
            if method == "initialize":
                result = self._handle_initialize(params)
                return JSONRPCResponse(result=result, id=request.id)
            elif method == "initialized":
                self._handle_initialized(params)
                return None  # Notification, no response
            elif method in ("tools/list", "tools/call"):
                # Require initialization for tools methods
                if not self._initialized:
                    return JSONRPCResponse(
                        error={
                            "code": JSONRPCError.INVALID_REQUEST,
                            "message": "Server not initialized. Call 'initialize' first.",
                        },
                        id=request.id,
                    )

                # Handle the specific method
                if method == "tools/list":
                    result = self._handle_tools_list(params)
                else:  # tools/call
                    result = self._handle_tools_call(params)
                return JSONRPCResponse(result=result, id=request.id)
            elif method == "ping":
                result = {}
            elif method == "shutdown":
                logger.info("Shutdown requested")
                result = {}
            else:
                return JSONRPCResponse(
                    error={
                        "code": JSONRPCError.METHOD_NOT_FOUND,
                        "message": f"Method not found: {method}",
                    },
                    id=request.id,
                )

            return JSONRPCResponse(result=result, id=request.id)

        except Exception as e:
            logger.exception("Error handling request: %s", method)
            return JSONRPCResponse(
                error={
                    "code": JSONRPCError.INTERNAL_ERROR,
                    "message": str(e),
                },
                id=request.id,
            )

    def parse_request(self, data: str) -> JSONRPCRequest | JSONRPCResponse:
        """Parse a JSON-RPC request from string.

        Args:
            data: JSON string containing the request.

        Returns:
            Parsed JSONRPCRequest or JSONRPCResponse (error).
        """
        try:
            obj = json.loads(data)
        except json.JSONDecodeError as e:
            return JSONRPCResponse(
                error={
                    "code": JSONRPCError.PARSE_ERROR,
                    "message": f"Parse error: {e}",
                },
            )

        if not isinstance(obj, dict):
            return JSONRPCResponse(
                error={
                    "code": JSONRPCError.INVALID_REQUEST,
                    "message": "Request must be an object",
                },
            )

        if "method" not in obj:
            return JSONRPCResponse(
                error={
                    "code": JSONRPCError.INVALID_REQUEST,
                    "message": "Missing 'method' field",
                },
                id=obj.get("id"),
            )

        return JSONRPCRequest(
            method=obj["method"],
            params=obj.get("params"),
            id=obj.get("id"),
            jsonrpc=obj.get("jsonrpc", "2.0"),
        )


class StdioTransport:
    """Stdio transport for MCP server.

    Reads JSON-RPC messages from stdin and writes responses to stdout.
    Uses newline-delimited JSON format.
    """

    def __init__(
        self,
        server: MCPServer,
        input_stream: TextIO | None = None,
        output_stream: TextIO | None = None,
    ) -> None:
        """Initialize the stdio transport.

        Args:
            server: The MCP server instance.
            input_stream: Input stream (default: sys.stdin).
            output_stream: Output stream (default: sys.stdout).
        """
        self.server = server
        self.input_stream = input_stream or sys.stdin
        self.output_stream = output_stream or sys.stdout
        self._running = False

    def send_response(self, response: JSONRPCResponse) -> None:
        """Send a JSON-RPC response.

        Args:
            response: The response to send.
        """
        try:
            data = json.dumps(response.to_dict())
            self.output_stream.write(data + "\n")
            self.output_stream.flush()
        except Exception as e:
            logger.error("Error sending response: %s", e)

    def run(self) -> None:
        """Run the stdio transport, processing messages until EOF."""
        self._running = True
        logger.info("JARVIS MCP server starting on stdio")

        while self._running:
            try:
                line = self.input_stream.readline()
                if not line:
                    logger.info("EOF received, shutting down")
                    break

                line = line.strip()
                if not line:
                    continue

                # Parse the request
                result = self.server.parse_request(line)

                if isinstance(result, JSONRPCResponse):
                    # Parse error
                    self.send_response(result)
                    continue

                # Handle the request
                response = self.server.handle_request(result)

                # Send response (if not a notification)
                if response is not None:
                    self.send_response(response)

            except KeyboardInterrupt:
                logger.info("Interrupted, shutting down")
                break
            except Exception as e:
                logger.exception("Error processing message")
                self.send_response(
                    JSONRPCResponse(
                        error={
                            "code": JSONRPCError.INTERNAL_ERROR,
                            "message": str(e),
                        },
                    )
                )

        self._running = False

    def stop(self) -> None:
        """Stop the transport."""
        self._running = False


async def run_http_server(host: str = "127.0.0.1", port: int = 8765) -> None:
    """Run the MCP server over HTTP.

    Args:
        host: Host address to bind to.
        port: Port number to bind to.
    """
    try:
        from aiohttp import web  # type: ignore[import-not-found]
    except ImportError:
        logger.error("aiohttp is required for HTTP transport. Install with: pip install aiohttp")
        return

    server = MCPServer()

    async def handle_request(request: web.Request) -> web.Response:
        """Handle HTTP POST request."""
        try:
            data = await request.text()
            parsed = server.parse_request(data)

            if isinstance(parsed, JSONRPCResponse):
                return web.json_response(parsed.to_dict())

            response = server.handle_request(parsed)
            if response is None:
                return web.json_response({})

            return web.json_response(response.to_dict())

        except Exception as e:
            logger.exception("Error handling HTTP request")
            return web.json_response(
                {
                    "jsonrpc": "2.0",
                    "error": {"code": -32603, "message": str(e)},
                    "id": None,
                },
                status=500,
            )

    async def handle_health(request: web.Request) -> web.Response:
        """Handle health check endpoint."""
        return web.json_response({"status": "ok", "server": SERVER_NAME, "version": SERVER_VERSION})

    app = web.Application()
    app.router.add_post("/", handle_request)
    app.router.add_post("/mcp", handle_request)
    app.router.add_get("/health", handle_health)

    runner = web.AppRunner(app)
    await runner.setup()

    site = web.TCPSite(runner, host, port)
    await site.start()

    logger.info("JARVIS MCP server running on http://%s:%d", host, port)
    logger.info("Health check: http://%s:%d/health", host, port)
    logger.info("MCP endpoint: http://%s:%d/mcp", host, port)

    # Keep running until interrupted
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    finally:
        await runner.cleanup()


def run_stdio() -> None:
    """Run the MCP server over stdio."""
    # Configure logging to stderr to avoid interfering with stdio protocol
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )

    server = MCPServer()
    transport = StdioTransport(server)
    transport.run()


def main() -> None:
    """Main entry point for the MCP server."""
    import argparse

    parser = argparse.ArgumentParser(
        description="JARVIS MCP Server - Model Context Protocol server for Claude Code integration"
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport mode (default: stdio)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host address for HTTP transport (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port number for HTTP transport (default: 8765)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            stream=sys.stderr,
        )

    if args.transport == "stdio":
        run_stdio()
    else:
        asyncio.run(run_http_server(args.host, args.port))


if __name__ == "__main__":
    main()
