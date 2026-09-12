"""Deterministic tests for Model Context Protocol (MCP) Manager."""

from __future__ import annotations

import asyncio
from jarvis.core.mcp import MCPManager, MCPTool


def test_builtin_tools_registration():
    mgr = MCPManager()
    tools = mgr.list_tools()
    names = [t.name for t in tools]

    assert "search_second_brain" in names
    assert "read_second_brain_note" in names
    assert "save_second_brain_note" in names


def test_tool_schema_export_for_llm():
    mgr = MCPManager()
    schemas = mgr.get_tools_schema_for_llm()

    assert len(schemas) >= 3
    fn_names = [s["function"]["name"] for s in schemas]
    assert "search_second_brain" in fn_names


def test_execute_search_second_brain():
    async def _run():
        mgr = MCPManager()
        result = await mgr.execute_tool("search_second_brain", {"query": "arquitetura"})

        assert not result.is_error
        assert isinstance(result.output, list)
        assert len(result.output) > 0
        assert "arquitetura" in result.output[0]["title"]

    asyncio.run(_run())


def test_execute_unknown_tool_fails_closed():
    async def _run():
        mgr = MCPManager()
        result = await mgr.execute_tool("non_existent_tool", {})

        assert result.is_error
        assert "não encontrada" in (result.details or "")

    asyncio.run(_run())
