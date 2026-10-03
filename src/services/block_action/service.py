from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import SystemMessage, HumanMessage
from src.providers.openrouter import make_llm
from src.services.open_data.service import OPEN_DATA_PYTHON_RULES, block_context
from src.services.sandworm_tools.service import SandwormToolsService
from src.util.cache import is_job_cancelled, wait_for_block_result
from src.util.stream_events import StreamEnvelope

from src.services.block_planner.models import BlockPlan, PlannedBlock
from src.services.intent.models import Intent
from .model import DUCKDB, DUNE, GeneratedBlock, SqlDataSource
from src.services.table_output import render_tables
from .prompts import SYSTEM_PROMPTS

log = logging.getLogger("sandworm.block_action")

# How long to wait for a dependency block's real execution result before
# giving up and falling back to generating off its code alone. Node's own
# auto-fix loop can take a few attempts, so this needs headroom beyond a
# single query's runtime.
BLOCK_RESULT_WAIT_TIMEOUT = 60

# Only these block types actually execute (and so ever get a real result
# pushed by Node's publishBlockResult) — waiting on anything else would
# just burn the full timeout for nothing.
EXECUTABLE_TYPES = {"sql", "python"}

# These block types are inserted by the Node side as empty widgets/config —
# it doesn't apply generated freeform text to them (see addBlocks in
# apps/api/.../ai-blocks.ts), so skip the LLM call rather than generate
# content that would just be discarded. pivot_table is handled separately
# (see _select_pivot_config) — it does get real generated content now, just
# not through the generic SYSTEM_PROMPTS[block.type] path below.
NO_CONTENT_TYPES: set[str] = set()

# How many candidate tools to hand the LLM for power_toolbox selection — top-1
# by raw embedding similarity alone isn't reliable enough to auto-select on.
POWER_TOOL_CANDIDATES = 5

# Matches AggregateFunction in packages/types/src/index.ts — kept in sync by
# hand since that's a TS/zod type this Python service can't import directly.
VALID_AGGREGATE_FUNCTIONS = {"sum", "mean", "median", "count", "min", "max"}


_DATAFRAMES_SECTION_RE = re.compile(r"## DATAFRAMES\n\n(.*?)(?=\n\n## |\Z)", re.DOTALL)


def _existing_dataframes(notebook_markdown: str | None) -> str:
    # Node's DuckDB executor already lets *any* query reference *any*
    # dataframe already in this document by name — it self-heals missing
    # globals by reloading from disk (see knownDataframes in
    # duckdb-query.service.ts). The model just needs to be told these exist;
    # docToMarkdown already computed this exact list for the chat context.
    if not notebook_markdown:
        return ""
    match = _DATAFRAMES_SECTION_RE.search(notebook_markdown)
    return match.group(1).strip() if match else ""


_DATAFRAME_NAME_RE = re.compile(r"\*\*([A-Za-z_][A-Za-z0-9_]*)\*\*")


def _dataframe_names_in_text(text: str) -> set[str]:
    return set(_DATAFRAME_NAME_RE.findall(text))


def _references_dataframe(sql: str, known_names: set[str]) -> bool:
    return any(
        re.search(rf"\b{re.escape(name)}\b", sql) for name in known_names
    )


def _user_message(
    block: PlannedBlock,
    intent: Intent,
    prior_blocks: list[GeneratedBlock],
    block_results: dict[int, dict | None] | None = None,
    existing_dataframes: str = "",
) -> str:
    parts: list[str] = [
        f"**Analytical goal:** {intent.goal}",
        f"**Task:** {block.description}",
    ]

    if existing_dataframes and block.type in ("sql", "python"):
        parts.append(
            "**Dataframes already available in this notebook's session** (query "
            "any of these directly by name instead of re-fetching the same data "
            f"— they exist regardless of depends_on):\n{existing_dataframes}"
        )

    if intent.entity.addresses:
        addrs = ", ".join(f"{a.address} ({a.chain})" for a in intent.entity.addresses)
        parts.append(f"**Addresses:** {addrs}")
    if intent.entity.protocol_names:
        parts.append(f"**Protocols:** {', '.join(intent.entity.protocol_names)}")
    if intent.params:
        for k, v in intent.params.items():
            parts.append(f"**{k}:** {v}")

    for idx in block.depends_on:
        if idx >= len(prior_blocks):
            continue
        dep = prior_blocks[idx]
        parts.append(
            f"**Preceding block ({dep.title}):**\n```\n{dep.content[:1500]}\n```"
        )

        # If we actually waited for this dependency and it ran, tell the model
        # what really happened — not just the code we hoped would work. This
        # is the planner's real sensory input: a failed dependency should
        # steer the next block away from assuming clean data, and a successful
        # one confirms the shape/volume of what's actually available.
        outcome = (block_results or {}).get(idx)
        if outcome:
            if outcome.get("outcome") == "error":
                parts.append(
                    f"**Actual result of preceding block:** it FAILED even after "
                    f"auto-fix attempts — error: {outcome.get('summary')}. Account "
                    f"for this: do not assume the data above is available or correct; "
                    f"adapt this block accordingly (e.g. handle missing/empty data)."
                )
            else:
                parts.append(
                    f"**Actual result of preceding block:** it ran successfully — "
                    f"{outcome.get('summary')}."
                )
                columns = outcome.get("columns")
                if columns:
                    col_desc = ", ".join(f"{c['name']} ({c['type']})" for c in columns)
                    parts.append(f"**Actual columns:** {col_desc}")

        if dep.type in EXECUTABLE_TYPES and dep.dataframe_name:
            # The dependency already ran (a Dune pull, a local step, or a
            # python fetch) and its result is sitting in this same session as
            # a DataFrame — point the model at it by name instead of letting
            # it re-fetch the data.
            parts.append(
                f"The result of the preceding block above is already loaded in this "
                f"session as a table/DataFrame named `{dep.dataframe_name}`. Query it "
                f"directly (e.g. `SELECT ... FROM {dep.dataframe_name} ...`) — do not "
                f"re-fetch this data."
            )

    return "\n\n".join(parts)


def _dataframe_name(job_id: str | None, index: int) -> str:
    # A short, valid-identifier, per-job-unique name so a downstream block can
    # reference a preceding SQL block's result without colliding with any
    # dataframeName a human (or an earlier AI run) already has in the doc.
    token = re.sub(r"[^a-zA-Z0-9]", "", job_id or "")[:8] or "adhoc"
    return f"aiq_{token}_{index}"


def _sql_data_source(content: str, known_names: set[str]) -> SqlDataSource:
    # Routing is decided from what the model actually wrote, not from the
    # planner's declared depends_on graph — depends_on only captures the
    # dependency graph the planner thought to declare, but the model is told
    # (in _user_message) it can reference *any* known dataframe by name
    # regardless of depends_on, and it does. Trusting depends_on alone means
    # any block the model routes to a dataframe without a matching depends_on
    # entry gets sent to Trino, which can't resolve a pandas variable as a
    # catalog table. The content is ground truth: if it names a known
    # dataframe, it's a local DuckDB query; otherwise it's a fresh Dune pull.
    return DUCKDB if _references_dataframe(content, known_names) else DUNE


class BlockActionService:
    def __init__(self, api_key: str, model: str, envelope: StreamEnvelope | None = None):
        self.api_key = api_key
        self.llm = make_llm(api_key, model)
        self.envelope = envelope
        self.tools = SandwormToolsService()

    async def generate_blocks(
        self,
        plan: BlockPlan,
        intent: Intent,
        notebook_markdown: str | None = None,
        chat_id: str | None = None,
        open_data: bool = False,
    ) -> list[GeneratedBlock]:
        generated: list[GeneratedBlock] = []
        # Real execution outcomes for already-generated blocks, keyed by their
        # index in plan.blocks — populated as we wait on each one's dependents.
        block_results: dict[int, dict | None] = {}
        existing_dataframes = _existing_dataframes(notebook_markdown)
        # Grows as sql blocks are generated below — depends_on only captures
        # the dependency graph the planner declared, but (per _user_message's
        # "existing dataframes" context) the model is free to reference *any*
        # dataframe by name regardless of depends_on. Routing has to track
        # what the model can actually see, not just the declared graph.
        known_dataframe_names = _dataframe_names_in_text(existing_dataframes)

        job_id = self.envelope.job_id if self.envelope else None

        for index, block in enumerate(plan.blocks):
            # Checked once per block rather than more granularly — a single
            # block's own generation call is short enough that finishing it
            # rather than interrupting mid-call is the simpler, still-responsive
            # choice.
            if chat_id is not None and await is_job_cancelled(chat_id):
                log.info("block generation cancelled before block %d/%d", index + 1, len(plan.blocks))
                return generated

            await self._await_dependency_results(block, generated, block_results)

            generated_block = GeneratedBlock(
                type=block.type,
                title=block.title,
                description=block.description,
                content="",
                depends_on=block.depends_on,
            )

            if block.type in EXECUTABLE_TYPES:
                generated_block.dataframe_name = _dataframe_name(job_id, index)
            elif block.type in ("pivot_table", "visualization") and block.depends_on:
                # Node needs a real dataframeName to wire these up to actual
                # data (see addBlocks on the Node side). A sql or python
                # dependency's dataframe_name is assigned deterministically
                # above, before it even runs; a python block is told to store
                # its result under that name (see below).
                dep_idx = block.depends_on[0]
                if dep_idx < len(generated) and generated[dep_idx].dataframe_name:
                    generated_block.dataframe_name = generated[dep_idx].dataframe_name

            if self.envelope:
                await self.envelope.block_generating(generated_block.id, block.type, block.title)

            if block.type == "power_toolbox":
                content = await self._select_power_tool(block, intent, generated, block_results)
            elif block.type == "pivot_table":
                content = await self._select_pivot_config(block, intent, generated, block_results)
            elif block.type in NO_CONTENT_TYPES:
                content = ""
            else:
                system = SYSTEM_PROMPTS[block.type]
                user = _user_message(block, intent, generated, block_results, existing_dataframes)
                if block.type == "python":
                    # Visualization, pivot_table and follow-up sql blocks read
                    # a python block's output by this name.
                    user = (
                        f"{user}\n\n**Output:** store this block's final table in a pandas "
                        f"DataFrame assigned to a top-level variable named "
                        f"`{generated_block.dataframe_name}`. Later blocks chart, pivot and "
                        f"query it by that name."
                    )
                    if open_data:
                        system += OPEN_DATA_PYTHON_RULES
                        user = f"{user}\n\n{block_context(f'{block.title} {block.description}')}"

                response = await self.llm.ainvoke([
                    SystemMessage(content=system),
                    HumanMessage(content=user),
                ])
                content = re.sub(r"^```(?:\w+)?\s*|\s*```$", "", response.content.strip())
                if block.type == "python":
                    content = render_tables(content)

            if block.type == "sql":
                generated_block.data_source = _sql_data_source(content, known_dataframe_names)
                log.info(
                    "sql block '%s' routed to %s | known_dataframes=%s | sql=%r",
                    block.title, generated_block.data_source,
                    sorted(known_dataframe_names), content[:500],
                )

            # A later sql block that names this dataframe is a local DuckDB
            # query over it, not a Dune pull.
            if block.type in EXECUTABLE_TYPES and generated_block.dataframe_name:
                known_dataframe_names.add(generated_block.dataframe_name)

            generated_block.content = content
            generated.append(generated_block)

            if self.envelope:
                # Only sql/python blocks actually execute ("ran"). A notebook
                # has at most one dashboard_header — generating it always
                # updates that existing block in place (see
                # upsertDashboardHeaderBlock on the Node side), so it's
                # "edited", never "created". Everything else is freshly
                # inserted into the notebook, so "created".
                if block.type in EXECUTABLE_TYPES:
                    action = "ran"
                elif block.type == "dashboard_header":
                    action = "edited"
                else:
                    action = "created"
                await self.envelope.block_ready(
                    generated_block.id, block.type, block.title, content,
                    data_source=generated_block.data_source,
                    dataframe_name=generated_block.dataframe_name,
                    action=action,
                )

            # This block itself may execute (sql/python) — if a later block
            # depends on it, that block's loop iteration will wait on it via
            # _await_dependency_results above. Nothing to do here; the wait
            # is pulled by the dependent, not pushed by this block.

        return generated

    # Waits, one dependency at a time, for the real execution outcome of any
    # not-yet-resolved dependency of `block` that actually runs (sql/python).
    # This is what makes generation "interactive": Node has already inserted
    # and auto-fix-run each preceding block by the time its block_ready event
    # was processed, so by the time we get here the result may already be
    # sitting in Redis — or we block until it lands (up to the timeout), then
    # move on with whatever we got (including nothing, on timeout).
    async def _await_dependency_results(
        self,
        block: PlannedBlock,
        generated: list[GeneratedBlock],
        block_results: dict[int, dict | None],
    ) -> None:
        for dep_idx in block.depends_on:
            if dep_idx in block_results or dep_idx >= len(generated):
                continue
            dep = generated[dep_idx]
            if dep.type not in EXECUTABLE_TYPES:
                continue
            block_results[dep_idx] = await wait_for_block_result(
                dep.id, timeout=BLOCK_RESULT_WAIT_TIMEOUT
            )

    # power_toolbox has no freeform "content" — it needs a real tool_id plus
    # inputs matching that tool's declared schema. Pull the top-N candidates
    # via the same embedding search used for tool selection elsewhere, then
    # let the LLM pick the actual best fit among them and fill in its inputs,
    # in one call.
    async def _select_power_tool(
        self,
        block: PlannedBlock,
        intent: Intent,
        prior_blocks: list[GeneratedBlock],
        block_results: dict[int, dict | None] | None = None,
    ) -> str:
        matches = await self.tools.search(query=block.description, top_k=POWER_TOOL_CANDIDATES, api_key=self.api_key)
        if not matches:
            return json.dumps({"tool_id": None, "inputs": {}})

        def _describe(tool: dict) -> str:
            inputs_schema = tool.get("inputs") or []
            inputs_desc = "\n".join(
                f"  - {i['key']} ({i['type']}{', required' if i.get('required') else ''}): {i['label']}"
                for i in inputs_schema
            ) or "  (no inputs)"
            return f"### {tool['tool_id']}\n{tool['description']}\nInputs:\n{inputs_desc}"

        candidates_desc = "\n\n".join(_describe(m) for m in matches)

        system = (
            "You are picking the best-fitting Sandworm power tool for this task "
            "from the candidates below, then filling in its inputs. Fit is NOT "
            "just about the description reading similarly — for each candidate, "
            "check its actual inputs schema: every input marked required must "
            "have a real, specific value you can derive from the task/intent "
            "below (an address, a date, a number, etc.), not a guess or "
            "placeholder. A tool whose description matches but whose required "
            "inputs can't actually be filled is NOT a fit — reject it and check "
            "the next candidate instead. Return ONLY a JSON object: "
            "{\"tool_id\": \"<chosen tool_id, must be exactly one of the "
            "candidates>\", \"inputs\": {<input key>: <value>, ...}} — no "
            "markdown, no explanation, no input keys outside that tool's own "
            "schema. If NONE of the candidates both fit the task AND have all "
            "required inputs fillable, return {\"tool_id\": null, \"inputs\": "
            "{}} instead of forcing a bad match."
        )
        user = (
            _user_message(block, intent, prior_blocks, block_results)
            + f"\n\n**Candidate tools:**\n{candidates_desc}"
        )

        response = await self.llm.ainvoke([
            SystemMessage(content=system),
            HumanMessage(content=user),
        ])
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.content.strip())
        try:
            result = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return json.dumps({"tool_id": None, "inputs": {}})

        tool_id = result.get("tool_id")
        valid_ids = {m["tool_id"] for m in matches}
        if tool_id not in valid_ids:
            return json.dumps({"tool_id": None, "inputs": {}})

        return json.dumps({"tool_id": tool_id, "inputs": result.get("inputs") or {}})

    # pivot_table has no freeform "content" either — it needs real rows/
    # columns/metrics picked from its dependency's ACTUAL result columns
    # (Node discards anything else — dataframeName is all it accepts as a
    # dependency-derived field, see buildBlockSpec on the Node side). Content
    # here is a JSON config Node parses to build those, not display text.
    async def _select_pivot_config(
        self,
        block: PlannedBlock,
        intent: Intent,
        prior_blocks: list[GeneratedBlock],
        block_results: dict[int, dict | None] | None = None,
    ) -> str:
        empty = json.dumps({"rows": [], "columns": [], "metrics": []})

        if not block.depends_on:
            return empty

        outcome = (block_results or {}).get(block.depends_on[0])
        columns = (outcome or {}).get("columns") if outcome else None
        if not columns:
            # No real column list to reason about (dependency wasn't sql, or
            # didn't run/failed) — leave it for a human to configure rather
            # than guessing column names that may not exist.
            return empty

        columns_by_name = {c["name"]: c for c in columns}
        columns_desc = "\n".join(f"- {c['name']} ({c['type']})" for c in columns)

        system = (
            "You are configuring a pivot table over the real columns listed "
            "below, based on the task. Pick which columns are grouping "
            "dimensions (rows), which are pivot columns (usually none/empty "
            "unless the task genuinely calls for a cross-tab), and which are "
            "aggregated metrics. A metric's aggregateFunction must be exactly "
            "one of: sum, mean, median, count, min, max — pick one that makes "
            "sense for that column's type (never sum/mean a non-numeric "
            "column). Return ONLY a JSON object: {\"rows\": [\"<column "
            "name>\", ...], \"columns\": [\"<column name>\", ...], "
            "\"metrics\": [{\"column\": \"<column name>\", \"aggregateFunction\": "
            "\"<one of the six above>\"}, ...]} — every column name used MUST "
            "be exactly one of the real columns listed below, no invented "
            "ones. rows/columns/metrics may be empty arrays if nothing in the "
            "task calls for them, but at least one metric is expected in the "
            "typical case."
        )
        user = (
            _user_message(block, intent, prior_blocks, block_results)
            + f"\n\n**Real columns available:**\n{columns_desc}"
        )

        response = await self.llm.ainvoke([
            SystemMessage(content=system),
            HumanMessage(content=user),
        ])
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.content.strip())
        try:
            result = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return empty

        # Resolve by name to the REAL known column dict rather than trusting
        # whatever `type` string the LLM might echo back — the ground truth
        # Node already published is what actually matters downstream.
        def _valid_column_refs(names: object) -> list[dict[str, str]]:
            if not isinstance(names, list):
                return []
            return [columns_by_name[n] for n in names if isinstance(n, str) and n in columns_by_name]

        def _valid_metrics(metrics: object) -> list[dict]:
            if not isinstance(metrics, list):
                return []
            out: list[dict] = []
            for m in metrics:
                if not isinstance(m, dict):
                    continue
                col, agg = m.get("column"), m.get("aggregateFunction")
                if col in columns_by_name and agg in VALID_AGGREGATE_FUNCTIONS:
                    out.append({"column": columns_by_name[col], "aggregateFunction": agg})
            return out

        return json.dumps({
            "rows": _valid_column_refs(result.get("rows")),
            "columns": _valid_column_refs(result.get("columns")),
            "metrics": _valid_metrics(result.get("metrics")),
        })
