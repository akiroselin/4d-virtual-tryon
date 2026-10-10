from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from fashion_agent.tools import (
    BODY_FIELDS,
    INTENT_FIELDS,
    SIZE_FIELDS,
    TOOLS,
    ToolError,
    call_tool,
    compare_iterations,
    run_tryon,
    tool_field_names,
)

ROOT = Path(__file__).resolve().parents[1]


def test_tool_schemas_mirror_pipeline_dataclasses():
    names = tool_field_names()
    assert set(INTENT_FIELDS) <= set(names["intent"])
    assert set(BODY_FIELDS) <= set(names["body"])
    assert set(SIZE_FIELDS) <= set(names["size"])
    for name, (function, description, schema) in TOOLS.items():
        assert callable(function) and description and schema["type"] == "object", name


def test_parse_brief_allows_explicit_overrides():
    parsed = call_tool(
        "parse_design_brief",
        {"brief": "oversized streetwear cotton t-shirt", "intent": {"fabric": "heavy denim"}},
    )
    assert parsed["intent"]["silhouette"] == "oversized"
    assert parsed["intent"]["fabric"] == "heavy denim"
    assert parsed["material_card"]["density"] > 3.0
    assert parsed["end_to_end_simulation"] is True
    with pytest.raises(ToolError):
        call_tool("parse_design_brief", {"brief": "tee", "intent": {"colour": "red"}})


def test_hypotheses_cover_non_simulated_templates():
    result = call_tool(
        "generate_construction_hypotheses",
        {"intent": {"garment_type": "skirt"}, "body": {"waist_cm": 68, "hip_cm": 94}, "top_k": 2},
    )
    assert len(result["hypotheses"]) == 2
    assert all(item["garment_type"] == "skirt" for item in result["hypotheses"])
    templates = call_tool("list_garment_templates", {})
    assert templates["simulated_garment_types"] == ["t-shirt"]


def test_run_tryon_records_iterations_and_rejects_unsupported(tmp_path):
    with pytest.raises(ToolError):
        run_tryon("bad id!", {"garment_type": "t-shirt"}, sessions_dir=tmp_path)
    with pytest.raises(ToolError):
        run_tryon("dress1", {"garment_type": "dress"}, sessions_dir=tmp_path)

    report = run_tryon(
        "tee1",
        {"garment_type": "t-shirt", "silhouette": "relaxed"},
        steps=4,
        resolution=11,
        notes="baseline",
        sessions_dir=tmp_path,
    )
    assert report["iteration"] == 1
    assert report["fit_validation"]["status"] in {"pass", "fail"}
    assert Path(report["artifacts"]["final_png"]).is_file()
    comparison = compare_iterations("tee1", sessions_dir=tmp_path)
    assert [row["iteration"] for row in comparison["iterations"]] == [1]
    assert comparison["iterations"][0]["notes"] == "baseline"


def test_mcp_server_stdio_handshake_and_tool_call(tmp_path):
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "parse_design_brief", "arguments": {"brief": "slim silk v-neck tee"}, "_meta": {"progressToken": 7}},
        },
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "nope", "arguments": {}}},
    ]
    completed = subprocess.run(
        [sys.executable, str(ROOT / "fashion_agent" / "mcp_server.py")],
        input="".join(json.dumps(request) + "\n" for request in requests),
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=120,
        check=True,
    )
    responses = [json.loads(line) for line in completed.stdout.splitlines()]
    assert [response["id"] for response in responses] == [1, 2, 3, 4]
    assert responses[0]["result"]["protocolVersion"] == "2025-06-18"
    assert {tool["name"] for tool in responses[1]["result"]["tools"]} == set(TOOLS)
    parsed = json.loads(responses[2]["result"]["content"][0]["text"])
    assert parsed["intent"]["neckline"] == "v-neck"
    assert responses[3]["result"]["isError"] is True


def test_outfit_tools_validate_inputs_and_list_committed_scenes():
    scenes = {item["scene"]: item for item in call_tool("list_outfit_scenes", {})["scenes"]}
    committed = "anny-skirt-walk-one-step-2s-contactfix-v10-webgpu"
    assert committed in scenes
    assert scenes[committed]["frames"] > 1
    with pytest.raises(ToolError):
        call_tool("render_outfit", {"scene": "does-not-exist"})
    with pytest.raises(ToolError):
        call_tool("simulate_outfit", {"outfit_id": "bad id!", "shirt_image": "a.png", "skirt_image": "b.png"})
    with pytest.raises(ToolError):
        call_tool("simulate_outfit", {"outfit_id": "ok", "shirt_image": "missing.png", "skirt_image": "missing.png"})
