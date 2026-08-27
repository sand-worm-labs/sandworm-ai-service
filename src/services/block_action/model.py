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


class GeneratedBlock(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: BlockType
    title: str
    description: str
    content: str
    depends_on: list[int] = Field(default_factory=list)
    # Only meaningful for "sql" blocks. data_source is "dune" for an initial
    # data pull (Trino-backed, the only source with real chain data today) or
    # "duckdb" when the block instead manipulates a preceding SQL block's
    # already-fetched result locally. dataframe_name is the table/DataFrame
    # name that result is stored under, so a dependent block can reference it.
    data_source: str | None = None
    dataframe_name: str | None = None
