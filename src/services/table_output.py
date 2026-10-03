from __future__ import annotations

import ast

# The notebook renders a DataFrame as a table, but print() turns it into plain
# text. The prompts ask the model not to print tables; this fixes the code when
# it does anyway, so the rule does not depend on the model following it.

# Calls that give back text or a single number, which print() shows correctly.
_TEXT_CALLS = {"str", "repr", "format", "len", "int", "float", "round", "sum", "min", "max", "abs", "bool", "type"}
_TEXT_METHODS = {"format", "join", "strip", "lstrip", "rstrip", "upper", "lower", "title", "replace", "split", "isoformat", "strftime"}
# Methods that turn a table into text. print(df.to_string()) is df, as text.
_TABLE_AS_TEXT = {"to_string", "to_markdown"}


def _is_text(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (str, bytes))
    if isinstance(node, ast.JoinedStr):
        return True
    if isinstance(node, ast.BinOp):
        return _is_text(node.left) or _is_text(node.right)
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name):
            return func.id in _TEXT_CALLS
        if isinstance(func, ast.Attribute):
            return func.attr in _TEXT_METHODS
    return False


def _table_as_text(node: ast.expr) -> ast.expr | None:
    # `x.to_string()` with no arguments: the table `x`.
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _TABLE_AS_TEXT
        and not node.args
        and not node.keywords
    ):
        return node.func.value
    return None


def _offset(lines: list[str], line: int, byte_col: int) -> int:
    # ast columns count UTF-8 bytes; turn one into an index into the whole text.
    before = sum(len(l) for l in lines[: line - 1])
    return before + len(lines[line - 1].encode()[:byte_col].decode())


def render_tables(code: str) -> str:
    """Rewrite print(<table>) as display(<table>), leaving printed text alone."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    lines = code.splitlines(keepends=True)
    edits: list[tuple[int, int, str]] = []

    for stmt in ast.walk(tree):
        if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)):
            continue
        call = stmt.value
        if not (isinstance(call.func, ast.Name) and call.func.id == "print"):
            continue
        if not call.args or call.keywords or any(isinstance(a, ast.Starred) for a in call.args):
            continue

        replacements: list[tuple[int, int, str]] = []
        for arg in call.args:
            inner = _table_as_text(arg)
            if inner is not None:
                text = ast.get_source_segment(code, inner)
                if text is None:
                    break
                start = _offset(lines, arg.lineno, arg.col_offset)
                end = _offset(lines, arg.end_lineno, arg.end_col_offset)
                replacements.append((start, end, text))
            elif _is_text(arg):
                break
        else:
            func = call.func
            start = _offset(lines, func.lineno, func.col_offset)
            end = _offset(lines, func.end_lineno, func.end_col_offset)
            edits.append((start, end, "display"))
            edits.extend(replacements)

    for start, end, text in sorted(edits, reverse=True):
        code = code[:start] + text + code[end:]
    return code
