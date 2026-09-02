from __future__ import annotations

import uuid
from typing import Literal
from pydantic import BaseModel, Field


class BlockActionPart(BaseModel):
    type: Literal["block_action"] = "block_action"
    action: Literal["created", "ran", "edited"]
    blockType: str
    blockTitle: str
    blockId: str


BlockType = Literal[
    "sql",
    "python",
    "visualization",
    "markdown",
    "dashboard_header",
    "pivot_table",
    "rich_text",
    "input",
    "dropdown_input",
    "date_input",
    "power_toolbox",
]

# Named to match apps/api's DATA_SOURCE_QUERY_ENGINE/DATA_SOURCE_DIALECT
# (packages/types/src/index.ts) — same three real data sources that exist.
# SANDWORM_CLOUD is currently unavailable — SandwormCloudQueryService is
# still a mock (no real query execution wired up, disabled in the datasource
# picker; see sandworm-cloud-datasource.service.ts) — so _sql_data_source
# below can never actually produce it today. Listed here anyway so
# SqlDataSource stays an honest, complete picture of the real data sources
# rather than silently 2-of-3 forever.
DUNE: Literal["dune"] = "dune"
DUCKDB: Literal["duckdb"] = "duckdb"
SANDWORM_CLOUD: Literal["sandworm_cloud"] = "sandworm_cloud"
SqlDataSource = Literal["dune", "duckdb", "sandworm_cloud"]


class GeneratedBlock(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: BlockType
    title: str
    description: str
    content: str
    depends_on: list[int] = Field(default_factory=list)
    # Only meaningful for "sql" blocks. DUNE for an initial data pull
    # (Trino-backed, the only source with real chain data today) or DUCKDB
    # when the block instead manipulates a preceding SQL block's
    # already-fetched result locally. dataframe_name is the table/DataFrame
    # name that result is stored under, so a dependent block can reference it.
    data_source: SqlDataSource | None = None
    dataframe_name: str | None = None
