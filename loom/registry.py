from __future__ import annotations

from loom.types import Tool


class UnknownToolError(Exception):
    pass


class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        self._tools = {t.name: t for t in tools}

    def tools(self) -> list[Tool]:
        return list(self._tools.values())

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise UnknownToolError(name)
        return self._tools[name]
