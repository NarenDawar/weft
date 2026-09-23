from __future__ import annotations
import pytest
from weft.registry import ToolRegistry, UnknownToolError
from weft.types import Tool


def make_registry():
    read = Tool(
        name="read_file",
        description="Read a file",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        fn=lambda path: f"contents of {path}",
    )
    return ToolRegistry([read])


def test_get_returns_registered_tool():
    registry = make_registry()
    assert registry.get("read_file").name == "read_file"


def test_get_raises_for_unknown_tool():
    registry = make_registry()
    with pytest.raises(UnknownToolError):
        registry.get("nope")


def test_tools_returns_all_registered_tools():
    registry = make_registry()
    names = [t.name for t in registry.tools()]
    assert names == ["read_file"]
