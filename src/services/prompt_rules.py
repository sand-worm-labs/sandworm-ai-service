# Rules shared by every prompt that writes Python for a notebook cell.

# The notebook renders a DataFrame as a table; print() turns it into plain text.
PYTHON_TABLE_RULE = (
    "Return a table as a DataFrame, never print it: no print(df), and never "
    "print(df.to_string()) or print(df.to_markdown()). End the code with just the "
    "variable on its own line (e.g. `yr`), with no print() or display() around it, "
    "so the notebook renders a real table. One table per block: if there are "
    "several tables, write one block for each, not one block that prints them all. "
    "Turn a Series into a table with .reset_index() or .to_frame(). Give columns clear "
    "names with units, round numbers, and show a date as a column, not as the index. "
    "Do not print titles, sources or notes; those belong in a markdown block."
)

# Mobile comes first for every chart, table and card. Same rule as apps/mcp
# src/tools/notebooks/chart-rules.ts: keep the two in step.
CHART_RESPONSIVE_RULE = (
    "Charts and visualizations must read well on a phone, about 390px wide, before "
    "anything else. Draw every Plotly chart with `from sandworm_theme import chart` and "
    "`chart(fig, title=..., source=...)` instead of fig.show(): the title and legend "
    "become page text above the chart, so they never collide on a small screen. Never "
    "leave a title or legend inside the figure, and never pass a colorscale to a "
    "heatmap: the theme supplies the pink-to-plum scale. "
    "When a rule here conflicts with density, extra series, decoration "
    "or how it looks on a desktop, the phone wins. Never set a chart's width, margins "
    "or legend position: the theme and the renderer size it to the screen. Keep any "
    "height modest, 320 to 420px and never above 480. Show at most 8 to 10 categories "
    "and group the rest as 'Other'. Keep labels and series names short, about 14 "
    "characters. Use at most 5 series and legend entries, and never a dual-axis chart. "
    "Prefer horizontal bars for rankings and long labels. Keep chart titles under "
    "about 60 characters. Put the key numbers in the title or "
    "in a markdown block, not in annotations or text drawn over the plot. Network and "
    "node-link diagrams label only the handful of nodes that matter (about 8). Tables "
    "have at most 6 columns with short headers."
)

# What a notebook is built from. Same rule as COMPONENT_RULES in apps/mcp src/tools/notebooks/chart-rules.ts: keep
# the two in step.
COMPONENT_RULE = (
    "Never hard-code colours on a chart: no hex codes, marker_color, colorway or colour scale, and never red or "
    "green, even for gain and loss. Name every series and let the theme colour it; split in vs out or gain vs loss "
    "into two named traces. POS and NEG colour the text of a figure card only. Build cards, switchable charts and "
    "searchable tables from sandworm_theme, not from HTML or JavaScript you write yourself: a row of findings is "
    "show(card_grid(finding(kicker, big, head, detail, verdict), ...)); a wallet, token or protocol is "
    "profile_card(...) inside card_grid, which carries the padding. To switch between views of one question use "
    "switchable(uid, {\"Button text\": (fig, \"Title\", \"Note\"), ...}) with plain Plotly figures; for a table the "
    "reader can sort, search and page through use data_table(uid, df, [tcol(...), ...])."
)
