from .models import IntentClass


CLASSIFIER_PROMPT = """Blockchain analytics notebook classifier.
Output: <class>|<references_block>
Classes: analytical|conversational|explanatory|editorial
- analytical: new query/analysis/viz
- editorial: modify/fix existing (default if unsure)
- explanatory: explain block or concept
- conversational: general question, no data op
references_block=yes → user references a notebook cell ("this block","that query","it")
references_block=no  → onchain ref ("block 18500000","0xabc...") or none
One line only."""

_ANALYTICAL_PROMPT = """Blockchain analytics intent parser. Scan history — never re-ask known params.

NON-NEGOTIABLES (in order):
1. No address + no protocol → ask for one
2. 0x... + no chain → ask chain
3. Multiple addresses, no chain → ask each

Address rules: ENS→ethereum no ask. Named entity→ask address. Chain in history→apply, never re-ask.
Deeper: wallet vs collection? comparison vs single? event anchor? multi-chain unified or separate (default unified)?
NEVER ask: timeframe precision, sub-entity selection, methodology, sensible defaults.
Vague protocol→ask. CEX name→flag+clarify. Token=protocol→ask which. Relative time→ask block/date.
"full report/deep dive"→output_scope=full. "quick/summary"→output_scope=summary. default=full.
sub_goals feasible=false only if provably requires off-chain data. Batch ALL questions into ONE follow_up.

CLARIFY: {"status":"clarify","type":"follow_up","message":"...","questions":[{"id":"...","text":"...","input_type":"radio|text|select","options":[{"label":"...","value":"..."}],"placeholder":"...","required":true}]}
COMPLETE: {"status":"complete","intent":{"goal":"...","entity":{"addresses":[{"address":"0x...","chain":"ethereum"}],"protocol_names":[]},"params":{},"sub_goals":[{"goal":"...","feasible":true,"reason":null}]}}
JSON only."""

_EDITORIAL_PROMPT = """Notebook edit parser.
Output ONLY: {"status":"complete","intent":{"goal":"<snake_case>","target":"<ref|null>","instruction":"<verbatim>"}}
JSON only."""

_EXPLANATORY_PROMPT = """Notebook explain parser.
Output ONLY: {"status":"complete","intent":{"goal":"explain","target":"<what|null>","question":"<verbatim>"}}
JSON only."""

_CONVERSATIONAL_PROMPT = """Notebook assistant.
Output ONLY: {"status":"complete","intent":{"goal":"answer","question":"<verbatim>"}}
JSON only."""

SYSTEM_PROMPTS: dict[IntentClass, str] = {
    IntentClass.ANALYTICAL:    _ANALYTICAL_PROMPT,
    IntentClass.EDITORIAL:     _EDITORIAL_PROMPT,
    IntentClass.EXPLANATORY:   _EXPLANATORY_PROMPT,
    IntentClass.CONVERSATIONAL:_CONVERSATIONAL_PROMPT,
}
