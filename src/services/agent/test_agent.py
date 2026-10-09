from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

from src.models.base import ChatContext
from src.services.agent import memory
from src.services.agent import service as agent
from src.services.agent import tools as agent_tools
from src.services.agent.mcp_client import McpClient, McpError
from src.services.completions.models import Message
from src.services.research_memory.models import Attempt, MemoryHit


def make_state(**ctx) -> agent.AgentState:
    return agent.AgentState(
        messages=[Message(role="user", content="publish this notebook")],
        model="m",
        api_key="k",
        context=ChatContext(user_id="u", workspace_id="w", document_id="d", chat_id="c", user_token="tok", **ctx),
    )


def test_to_openrouter_tool_strips_schema_key_and_fills_defaults():
    tool = {"name": "list_workspaces", "description": "d", "inputSchema": {"$schema": "x"}}

    assert agent_tools.to_openrouter_tool(tool) == {
        "type": "function",
        "function": {"name": "list_workspaces", "description": "d", "parameters": {"type": "object", "properties": {}}},
    }


def test_system_prompt_names_the_notebook_and_includes_mcp_instructions():
    prompt = agent.system_prompt(make_state(focused_block_ids=["b1"]).context, "MCP RULES")

    assert "notebook d of workspace w" in prompt
    assert "b1" in prompt
    assert "MCP RULES" in prompt
    assert "Sandworm's onchain analytics assistant" in prompt


@pytest.fixture
def envelope():
    env = MagicMock()
    env.thinking = AsyncMock()
    env.text_delta = AsyncMock()
    return env


@pytest.fixture(autouse=True)
def not_cancelled(mocker):
    mocker.patch.object(agent, "is_job_cancelled", new=AsyncMock(return_value=False))


@pytest.fixture(autouse=True)
def offer_every_tool(mocker):
    # The defaults narrow the agent to notebook editing; these tests are about the loop, not that list.
    mocker.patch.object(agent.settings, "AGENT_INCLUDE_TOOLS", "")
    mocker.patch.object(agent.settings, "AGENT_EXCLUDE_TOOLS", "")
    mocker.patch.object(agent.settings, "AGENT_INTENT_ENABLED", False)


def make_mcp(call_result=("done", False)):
    mcp = MagicMock()
    mcp.list_tools = AsyncMock(return_value=[{"name": "publish_notebook", "inputSchema": {"type": "object"}}])
    mcp.instructions = AsyncMock(return_value="")
    mcp.call_tool = AsyncMock(return_value=call_result)
    mcp.aclose = AsyncMock()
    return mcp


@pytest.mark.asyncio
async def test_mcp_client_parses_sse_and_sends_the_token_and_skip_log_header():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["authorization"]
        seen["skip"] = request.headers["x-sandworm-skip-tool-log"]
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text='event: message\ndata: {"result":{"content":[{"type":"text","text":"hi"}],"isError":false},"jsonrpc":"2.0","id":1}\n\n',
        )

    client = McpClient("http://mcp/mcp", "tok", http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    assert await client.call_tool("x", {}) == ("hi", False)
    assert seen == {"auth": "Bearer tok", "skip": "1"}


@pytest.mark.asyncio
async def test_mcp_client_reports_an_expired_token():
    client = McpClient("http://mcp/mcp", "tok", http=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(401))))

    with pytest.raises(McpError, match="expired or invalid"):
        await client.list_tools()


def test_select_tools_include_and_exclude(mocker):
    tools = [{"name": "a"}, {"name": "b"}, {"name": "c"}]

    mocker.patch.object(agent.settings, "AGENT_INCLUDE_TOOLS", "a, b")
    mocker.patch.object(agent.settings, "AGENT_EXCLUDE_TOOLS", "b")
    assert agent_tools.select_tools(tools) == [{"name": "a"}]

    mocker.patch.object(agent.settings, "AGENT_INCLUDE_TOOLS", "")
    mocker.patch.object(agent.settings, "AGENT_EXCLUDE_TOOLS", "")
    assert agent_tools.select_tools(tools) == tools


def test_invalid_arguments_reports_what_is_wrong():
    schema = {"type": "object", "properties": {"notebookId": {"type": "string"}}, "required": ["notebookId"]}

    assert agent_tools.invalid_arguments(schema, {"notebookId": "d"}) is None
    assert "notebookId" in agent_tools.invalid_arguments(schema, {})
    assert "nothing was run" in agent_tools.invalid_arguments(schema, {"notebookId": 5})


@pytest.mark.asyncio
async def test_remember_reads_the_notebook_through_the_mcp_and_indexes_it(mocker):
    index = mocker.patch.object(memory, "index_notebook", new=AsyncMock())
    attempts = mocker.patch.object(memory, "index_attempts", new=AsyncMock())
    mcp = make_mcp(('{"title": "TVL study", "content": "Uniswap TVL fell 12%."}', False))
    state = make_state()

    await memory.remember("w", "d", state.job_id, state.attempts, mcp)

    mcp.call_tool.assert_awaited_once_with("get_notebook", {"notebookId": "d", "workspaceId": "w"})
    index.assert_awaited_once_with("w", "d", "TVL study", "Uniswap TVL fell 12%.")
    attempts.assert_awaited_once_with("w", "d", "TVL study", state.job_id, [])


@pytest.mark.asyncio
async def test_remember_swallows_errors(mocker):
    mocker.patch.object(memory, "index_notebook", new=AsyncMock(side_effect=RuntimeError("boom")))

    await memory.remember("w", "d", "run", [], make_mcp(('{"content": "x"}', False)))


def test_only_actions_are_remembered_not_reads():
    assert memory.attempt_for("list_workspaces", {}, "[]", False) is None
    assert memory.attempt_for("get_notebook", {"notebookId": "d"}, "{}", False) is None
    assert memory.attempt_for("search_tools", {"query": "x"}, "[]", False) is None
    assert memory.attempt_for("get_run_results", {"notebookId": "d"}, "error", True) is not None


def test_an_attempt_records_the_query_and_the_outcome():
    ok = memory.attempt_for(
        "add_cell",
        {"type": "sql", "title": "TVL", "dataSource": "dune", "content": "select  tvl\nfrom  t"},
        '{"cell": {}}',
        False,
    )
    failed = memory.attempt_for("run_notebook", {"notebookId": "d"}, "column tvl does not exist", True)

    assert ok.worked is True and "dataSource=dune" in ok.summary and "select tvl from t" in ok.summary
    assert failed.worked is False and failed.text().startswith("FAILED: run_notebook")


@pytest.mark.asyncio
async def test_attempts_are_remembered_even_when_the_notebook_cannot_be_read(mocker):
    attempts = mocker.patch.object(memory, "index_attempts", new=AsyncMock())
    state = make_state()
    state.attempts.append(Attempt("add_cell", "x", False, "boom"))

    await memory.remember("w", "d", state.job_id, state.attempts, make_mcp(("no access", True)))

    attempts.assert_awaited_once()
    assert attempts.await_args.args[4] == state.attempts


def scripted(mocker, *steps, tokens=1):
    """Stand in for the model: each step is a tool call (name, args) or the final text."""
    seen: list[AgentInfo] = []
    queue = list(steps)

    async def respond(messages, info: AgentInfo) -> ModelResponse:
        seen.append(info)
        step = queue.pop(0) if len(queue) > 1 else queue[0]
        usage = RequestUsage(input_tokens=tokens)
        if isinstance(step, str):
            return ModelResponse(parts=[TextPart(step)], usage=usage)
        name, args = step
        return ModelResponse(parts=[ToolCallPart(name, args)], usage=usage)

    mocker.patch.object(agent, "_make_model", return_value=FunctionModel(respond))
    return seen


async def run(mcp, envelope, state=None):
    return await agent.run_agent(state or make_state(), envelope, mcp, http=httpx.AsyncClient())


@pytest.mark.asyncio
async def test_runs_a_tool_then_streams_the_final_answer(mocker, envelope):
    mcp = make_mcp(("published", False))
    scripted(mocker, ("publish_notebook", {"notebookId": "d"}), "Published. Next: track forks.")

    state = await run(mcp, envelope)

    mcp.call_tool.assert_awaited_once_with("publish_notebook", {"notebookId": "d"})
    assert state.output == "Published. Next: track forks."
    envelope.thinking.assert_awaited_once()
    assert envelope.thinking.await_args.args[0] == "Ran publish_notebook"
    assert "".join(c.args[0] for c in envelope.text_delta.await_args_list) == state.output


@pytest.mark.asyncio
async def test_a_failing_tool_is_reported_to_the_model_not_raised(mocker, envelope):
    mcp = make_mcp()
    mcp.call_tool.side_effect = McpError("expired")
    scripted(mocker, ("publish_notebook", {}), "That failed.")

    state = await run(mcp, envelope)

    assert state.output == "That failed."
    assert envelope.thinking.await_args.args[0] == "Failed publish_notebook"


@pytest.mark.asyncio
async def test_stops_when_the_job_is_cancelled(mocker, envelope):
    mocker.patch.object(agent, "is_job_cancelled", new=AsyncMock(return_value=True))
    seen = scripted(mocker, "never")

    state = await run(make_mcp(), envelope)

    assert seen == [] and state.output is None


@pytest.mark.asyncio
async def test_a_cancel_during_a_run_stops_before_the_next_tool(mocker, envelope):
    mocker.patch.object(agent, "is_job_cancelled", new=AsyncMock(side_effect=[False, True]))
    mcp = make_mcp()
    scripted(mocker, ("publish_notebook", {}), "never")

    state = await run(mcp, envelope)

    mcp.call_tool.assert_not_awaited()
    assert state.output is None


@pytest.mark.asyncio
async def test_gives_up_after_the_step_limit(mocker, envelope):
    mocker.patch.object(agent.settings, "AGENT_MAX_STEPS", 2)
    seen = scripted(mocker, ("publish_notebook", {}))

    state = await run(make_mcp(), envelope)

    assert len(seen) == 2
    assert "ran out of steps" in state.output


@pytest.mark.asyncio
async def test_stops_at_the_token_budget(mocker, envelope):
    mocker.patch.object(agent.settings, "AGENT_MAX_TOKENS", 100)
    seen = scripted(mocker, ("publish_notebook", {}), tokens=150)

    state = await run(make_mcp(), envelope)

    assert len(seen) == 1
    assert "token limit" in state.output


@pytest.mark.asyncio
async def test_stops_at_the_time_limit(mocker, envelope):
    import asyncio

    mocker.patch.object(agent.settings, "AGENT_MAX_SECONDS", 0.05)

    async def slow(messages, info):
        await asyncio.sleep(1)

    mocker.patch.object(agent, "_make_model", return_value=FunctionModel(slow))

    state = await run(make_mcp(), envelope)

    assert "time limit" in state.output


@pytest.mark.asyncio
async def test_tools_run_one_at_a_time_with_the_configured_temperature(mocker, envelope):
    mocker.patch.object(agent.settings, "AGENT_TEMPERATURE", 0.2)
    seen = scripted(mocker, "ok")

    await run(make_mcp(), envelope)

    assert seen[0].model_settings["parallel_tool_calls"] is False
    assert seen[0].model_settings["temperature"] == 0.2


@pytest.mark.asyncio
async def test_a_model_without_tool_support_gets_a_clear_error(mocker, envelope):
    async def refuse(messages, info):
        raise ModelHTTPError(404, "m", body="No endpoints found that support tool use")

    mocker.patch.object(agent, "_make_model", return_value=FunctionModel(refuse))

    with pytest.raises(RuntimeError, match="cannot call tools"):
        await run(make_mcp(), envelope)


@pytest.mark.asyncio
async def test_earlier_messages_are_sent_as_history_and_the_last_one_as_the_prompt(mocker, envelope):
    state = make_state()
    state.messages = [
        Message(role="user", content="hi"),
        Message(role="assistant", content="hello"),
        Message(role="user", content="publish this notebook"),
    ]
    seen = []

    async def respond(messages, info):
        seen.extend(messages)
        return ModelResponse(parts=[TextPart("ok")])

    mocker.patch.object(agent, "_make_model", return_value=FunctionModel(respond))

    await run(make_mcp(), envelope, state)

    texts = [getattr(p, "content", None) for m in seen for p in m.parts]
    assert texts[:3] == ["hi", "hello", "publish this notebook"]


@pytest.mark.asyncio
async def test_added_cell_shows_up_as_a_created_block_card(mocker, envelope):
    envelope.block_generating = AsyncMock()
    envelope.block_ready = AsyncMock()
    mcp = make_mcp(('{"notebookId": "d", "cell": {"id": "c1", "kind": "sql", "title": "TVL"}}', False))
    mcp.list_tools.return_value = [{"name": "add_cell", "inputSchema": {"type": "object"}}]
    scripted(
        mocker,
        ("add_cell", {"type": "sql", "title": "TVL", "content": "select 1", "dataSource": "dune"}),
        "Added.",
    )

    await run(mcp, envelope)

    envelope.block_generating.assert_awaited_once_with("c1", "sql", "TVL")
    envelope.block_ready.assert_awaited_once_with(
        "c1", "sql", "TVL", "select 1", data_source="dune", dataframe_name=None, action="created"
    )


@pytest.mark.asyncio
async def test_a_bad_tool_call_is_fed_back_and_never_reaches_the_server(mocker, envelope):
    mcp = make_mcp()
    mcp.list_tools.return_value = [
        {"name": "publish_notebook", "inputSchema": {"type": "object", "properties": {"notebookId": {"type": "string"}}, "required": ["notebookId"]}}
    ]
    returned = []

    async def respond(messages, info):
        returned.extend(p.content for m in messages for p in m.parts if p.part_kind == "tool-return")
        if not returned:
            return ModelResponse(parts=[ToolCallPart("publish_notebook", {})])
        return ModelResponse(parts=[TextPart("Sorry.")])

    mocker.patch.object(agent, "_make_model", return_value=FunctionModel(respond))

    await run(mcp, envelope)

    mcp.call_tool.assert_not_awaited()
    assert "notebookId" in returned[0]


@pytest.mark.asyncio
async def test_prior_research_goes_into_the_system_prompt(mocker, envelope):
    mocker.patch("src.services.research_memory.service.recall", new=AsyncMock(return_value=[MemoryHit("notebook", "d9", "Old study", "TVL fell", 0.9)]))
    seen = scripted(mocker, "ok")

    await run(make_mcp(), envelope)

    assert "Old study" in seen[0].instructions and "d9" in seen[0].instructions


@pytest.mark.asyncio
async def test_a_failing_memory_never_blocks_the_chat(mocker, envelope):
    mocker.patch("src.services.research_memory.service.recall", new=AsyncMock(side_effect=RuntimeError("qdrant down")))
    scripted(mocker, "still works")

    state = await run(make_mcp(), envelope)

    assert state.output == "still works"


@pytest.mark.asyncio
async def test_tool_calls_become_attempts_with_their_outcome(mocker, envelope):
    mcp = make_mcp()
    mcp.call_tool.side_effect = [("nope", True)]
    scripted(mocker, ("publish_notebook", {}), "done")

    state = await run(mcp, envelope)

    assert [(a.tool, a.worked) for a in state.attempts] == [("publish_notebook", False)]


def test_the_default_agent_gets_the_full_tool_set_minus_browsing_and_destructive_tools():
    from src.config.settings import Settings

    assert Settings.model_fields["AGENT_INCLUDE_TOOLS"].default == ""
    excluded = {n.strip() for n in Settings.model_fields["AGENT_EXCLUDE_TOOLS"].default.split(",")}
    assert {"list_workspaces", "list_projects", "delete_notebook", "publish_notebook", "save_reply"} <= excluded
    assert excluded.isdisjoint({"get_notebook", "add_cell", "update_cell", "run_notebook", "plan_notebook"})
