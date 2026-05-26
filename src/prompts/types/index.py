

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ModelTier(StrEnum):
    """
    Controls prompt verbosity and instruction density.

    FRONTIER  claude-3-5+, gpt-4o, gemini-1.5-pro+, gemini-2+
    MID       claude-3-haiku, gpt-4o-mini, gemini-1.5-flash
    LOCAL     llama, mistral, qwen, deepseek, phi — anything sub-7B grade
    """
    FRONTIER = "frontier"
    MID      = "mid"
    LOCAL    = "local"


class ActionType(StrEnum):
    SQL_EDIT         = "sql_edit"
    SQL_FIX          = "sql_fix"
    SQL_GENERATE     = "sql_generate"

    PYTHON_EDIT      = "python_edit"
    PYTHON_FIX       = "python_fix"
    PYTHON_GENERATE  = "python_generate"

    MARKDOWN_EDIT    = "markdown_edit"

    TITLE_GENERATE   = "title_generate"

    VIZ_GENERATE     = "viz_generate"

    DOC_INSIGHTS     = "doc_insights"
    DOC_REARRANGE    = "doc_rearrange"
    DOC_AUDIT        = "doc_audit"

    INTENT_GATE      = "intent_gate"       
    GRIMOIRE_SELECT  = "grimoire_select"   
    GRIMOIRE_PARAMS  = "grimoire_params"   


class MarkdownIntent(StrEnum):
    FIX     = "fix"
    SHORTEN = "shorten"
    EXPAND  = "expand"
    REWRITE = "rewrite"
    CUSTOM  = "custom"


class TitleMode(StrEnum):
    """
    How the title generation was triggered.

    FROM_CONTENT   — no title exists, generate one from document content
    IMPROVE        — user clicked the AI button next to the title field;
                     an existing title is present, suggest better variants
    ITERATE        — user clicked "suggest another"; must differ from
                     all previous_suggestions
    CHAT_DIRECTED  — user described what they want in the chat panel;
                     free-form tone/style preference drives the output
    """
    FROM_CONTENT  = "from_content"
    IMPROVE       = "improve"
    ITERATE       = "iterate"
    CHAT_DIRECTED = "chat_directed"


class TitleTone(StrEnum):
    """
    Default tone the assembler picks when none is specified by the user.
    Passed explicitly when the user requests a specific style via chat.

    ANALYTICAL   "Uniswap v3 LP Fee Revenue — 90d Window"
    PUNCHY       "DEX Metrics 📊"  /  "Pump.Fun Alpha Wallets"
    QUESTION     "Where Are Stablecoins Going?"
    NARRATIVE    "How MEV Bots Drained $4M From This Pool"
    CASUAL       "perps & hyperliquid vibes"
    """
    ANALYTICAL = "analytical"
    PUNCHY     = "punchy"
    QUESTION   = "question"
    NARRATIVE  = "narrative"
    CASUAL     = "casual"


class SQLDialect(StrEnum):
    DUCKDB = "duckdb"
    SQL    = "sql"     



@dataclass
class SchemaColumn:
    name:        str
    type:        str
    description: str = ""


@dataclass
class SchemaTable:
    name:        str
    columns:     list[SchemaColumn] = field(default_factory=list)
    description: str = ""



@dataclass
class GrimoireTool:
    """A parameterized query template retrieved from the Grimoire corpus."""
    id:          str
    name:        str
    description: str
    template:    str           
    category:    str = ""
    params:      list[str] = field(default_factory=list)
    score:       float = 0.0    



@dataclass
class BlockScope:
    """
    Variables available from prior executed blocks in this notebook session.
    Serialized by the NestJS backend before calling the AI sidecar.
    """
    dataframes: list[str] = field(default_factory=list)  
    variables:  list[str] = field(default_factory=list)   
    imports:    list[str] = field(default_factory=list)  



@dataclass
class PromptContext:
    """
    Everything known at request time, passed to the assembler.

    Fields are optional by design — components inspect what they need
    and return None if their required data is absent. The assembler
    silently drops None sections.
    """

    action:     ActionType = ActionType.SQL_EDIT
    model_id:   str        = ""
    model_tier: ModelTier  = ModelTier.FRONTIER

    block_id:      str = ""
    source:        str = ""     
    instructions:  str = ""    
    error_message: str = ""    

    dialect:       SQLDialect        = SQLDialect.DUCKDB
    schema_tables: list[SchemaTable] = field(default_factory=list)

    scope: BlockScope = field(default_factory=BlockScope)

    markdown_intent: MarkdownIntent = MarkdownIntent.CUSTOM

    title_mode:           TitleMode        = TitleMode.FROM_CONTENT
    title_tone:           TitleTone | None = None   
    existing_title:       str              = ""    
    previous_suggestions: list[str]        = field(default_factory=list)
    title_user_context:   str              = ""    

    retrieved_tools: list[GrimoireTool] = field(default_factory=list)

    document_markdown: str       = ""   
    neighbor_blocks:   list[str] = field(default_factory=list) 

    extra: dict[str, Any] = field(default_factory=dict)



@dataclass
class PromptMessages:
    """Final assembled prompt, ready for LiteLLM."""
    system:   str
    user:     str
    action:   ActionType
    model_id: str

    def to_litellm(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user",   "content": self.user},
        ]