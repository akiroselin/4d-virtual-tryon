"""Minimal stdio MCP server exposing the fashion-design tools.

Speaks newline-delimited JSON-RPC 2.0 as the MCP stdio transport requires, with
no dependency beyond the project's own requirements. Easy Agent (and any other
MCP client) launches it via ``.mcp.json``:

    python3 -m fashion_agent.mcp_server

stdout carries protocol messages only; diagnostics go to stderr.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
import sys
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fashion_agent.tools import TOOLS, ToolError, call_tool  # noqa: E402

SERVER_INFO = {"name": "fashion-design", "version": "0.1.0"}
SUPPORTED_PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")


def _tool_list() -> list[dict[str, Any]]:
    return [
        {"name": name, "description": description, "inputSchema": schema}
        for name, (_, description, schema) in TOOLS.items()
    ]


def _send(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def _progress_emitter(token: Any):
    if token is None:
        return None

    def emit(done: int, total: int, message: str) -> None:
        _send(
            {
                "jsonrpc": "2.0",
                "method": "notifications/progress",
                "params": {"progressToken": token, "progress": done, "total": total, "message": message},
            }
        )

    return emit


def _tool_result(name: str, arguments: dict[str, Any], progress_token: Any = None) -> dict[str, Any]:
    try:
        result = call_tool(name, arguments, progress=_progress_emitter(progress_token))
    except (ToolError, TypeError, KeyError, FileNotFoundError) as error:
        return {"content": [{"type": "text", "text": f"{type(error).__name__}: {error}"}], "isError": True}
    except Exception as error:  # surface solver failures to the agent instead of killing the server
        traceback.print_exc(file=sys.stderr)
        return {"content": [{"type": "text", "text": f"{type(error).__name__}: {error}"}], "isError": True}
    content: list[dict[str, Any]] = [{"type": "text", "text": json.dumps(result, indent=2, default=str)}]
    artifacts = (result.get("artifacts") or {}) if isinstance(result, dict) else {}
    images = [artifacts["final_png"]] if artifacts.get("final_png") else list(artifacts.get("frames") or [])
    for image in images:
        if Path(image).is_file():
            content.append(
                {
                    "type": "image",
                    "data": base64.b64encode(Path(image).read_bytes()).decode("ascii"),
                    "mimeType": "image/png",
                }
            )
    return {"content": content, "isError": False}


def handle(message: dict[str, Any]) -> dict[str, Any] | None:
    method = message.get("method")
    request_id = message.get("id")
    if request_id is None:
        return None  # notification (e.g. notifications/initialized)
    params = message.get("params") or {}
    if method == "initialize":
        requested = params.get("protocolVersion")
        version = requested if requested in SUPPORTED_PROTOCOL_VERSIONS else SUPPORTED_PROTOCOL_VERSIONS[0]
        result: dict[str, Any] = {
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": SERVER_INFO,
        }
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": _tool_list()}
    elif method == "tools/call":
        result = _tool_result(
            params.get("name", ""),
            params.get("arguments") or {},
            (params.get("_meta") or {}).get("progressToken"),
        )
    else:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as error:
            response: dict[str, Any] | None = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {error}"},
            }
        else:
            response = handle(message)
        if response is not None:
            _send(response)


if __name__ == "__main__":
    main()
