from __future__ import annotations

import ast

# The notebook renders a DataFrame as a table, but print() turns it into plain
# text. The prompts ask the model not to print tables; this fixes the code when
# it does anyway, so the rule does not depend on the model following it. A table
# on the last line becomes the bare variable; one earlier on becomes display().

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
    # `x.to_string()`, whatever its formatting arguments: the table `x`.
    # With a buffer to write into it gives back nothing, so that one is left.
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _TABLE_AS_TEXT
        and not node.args
        and not any(k.arg in (None, "buf") for k in node.keywords)
    ):
        return node.func.value
    return None


def _has_table(node: ast.expr) -> bool:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _has_table(node.left) or _has_table(node.right)
    if isinstance(node, ast.JoinedStr):
        return any(isinstance(v, ast.FormattedValue) and _table_as_text(v.value) is not None for v in node.values)
    return _table_as_text(node) is not None


def _parts(code: str, node: ast.expr) -> list[tuple[bool, str]] | None:
    # Split what a print() is given into (is a table, source) pieces, taking
    # apart `"text" + df.to_string()` and f"{df.to_string()}".
    inner = _table_as_text(node)
    if inner is not None:
        text = ast.get_source_segment(code, inner)
        return None if text is None else [(True, text)]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add) and _has_table(node):
        left, right = _parts(code, node.left), _parts(code, node.right)
        return None if left is None or right is None else left + right
    if isinstance(node, ast.JoinedStr) and _has_table(node):
        text = [v for v in node.values if not (isinstance(v, ast.FormattedValue) and _table_as_text(v.value) is not None)]
        tables = [_table_as_text(v.value) for v in node.values if v not in text]
        parts = [(True, ast.unparse(t)) for t in tables]
        if any(not (isinstance(v, ast.Constant) and not str(v.value).strip()) for v in text):
            parts.insert(0, (False, ast.unparse(ast.JoinedStr(values=text))))
        return parts
    text = ast.get_source_segment(code, node)
    return None if text is None else [(False, text)]


def _split_statement(code: str, args: list[ast.expr]) -> str | None:
    # print("Totals:", df.to_string()) -> print("Totals:"); display(df)
    calls: list[tuple[bool, list[str]]] = []
    for arg in args:
        parts = _parts(code, arg)
        if parts is None:
            return None
        for is_table, text in parts:
            if calls and calls[-1][0] == is_table:
                calls[-1][1].append(text)
            else:
                calls.append((is_table, [text]))
    return "; ".join(f"{'display' if is_table else 'print'}({', '.join(texts)})" for is_table, texts in calls)


def _offset(lines: list[str], line: int, byte_col: int) -> int:
    # ast columns count UTF-8 bytes; turn one into an index into the whole text.
    before = sum(len(l) for l in lines[: line - 1])
    return before + len(lines[line - 1].encode()[:byte_col].decode())


def render_tables(code: str) -> str:
    """Rewrite print(<table>) as display(<table>), leaving printed text as print()."""
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
            elif _is_text(arg) or _has_table(arg):
                break
        else:
            func = call.func
            start = _offset(lines, func.lineno, func.col_offset)
            end = _offset(lines, func.end_lineno, func.end_col_offset)
            edits.append((start, end, "display"))
            edits.extend(replacements)
            continue

        # Text and a table as text in one call: print the text, show the table.
        if any(_has_table(a) for a in call.args):
            split = _split_statement(code, call.args)
            if split is not None:
                start = _offset(lines, call.lineno, call.col_offset)
                end = _offset(lines, call.end_lineno, call.end_col_offset)
                edits.append((start, end, split))

    for start, end, text in sorted(edits, reverse=True):
        code = code[:start] + text + code[end:]
    return _return_last_table(code) if edits else code


def _return_last_table(code: str) -> str:
    # A table shown by the last line needs no display(): the notebook renders
    # what the code ends with, so `display(yr)` there is just `yr`.
    try:
        last = ast.parse(code).body[-1]
    except (SyntaxError, IndexError):
        return code
    call = last.value if isinstance(last, ast.Expr) else None
    if not (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "display"
        and len(call.args) == 1
        and not call.keywords
        and not isinstance(call.args[0], ast.Starred)
    ):
        return code
    table = ast.get_source_segment(code, call.args[0])
    if table is None:
        return code
    lines = code.splitlines(keepends=True)
    start = _offset(lines, call.lineno, call.col_offset)
    end = _offset(lines, call.end_lineno, call.end_col_offset)
    return code[:start] + table + code[end:]
