# Rules shared by every prompt that writes Python for a notebook cell.

# The notebook renders a DataFrame as a table; print() turns it into plain text.
PYTHON_TABLE_RULE = (
    "Show a table as a DataFrame, never with print(): make it the last expression of "
    "the code so the notebook renders a real table. One table per block: if there are "
    "several tables, write one block for each, not one block that prints them all. "
    "Turn a Series into a table with .reset_index() or .to_frame(). Give columns clear "
    "names with units, round numbers, and show a date as a column, not as the index. "
    "Do not print titles, sources or notes; those belong in a markdown block."
)
