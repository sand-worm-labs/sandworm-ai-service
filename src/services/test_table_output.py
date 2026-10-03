from src.services.table_output import render_tables


def test_a_printed_dataframe_becomes_display():
    assert render_tables("print(df)") == "display(df)"
    assert render_tables("print(df.head(10))") == "display(df.head(10))"


def test_printed_series_stats_and_several_tables_become_display():
    assert render_tables("print(top.sort_values('v'), other)") == "display(top.sort_values('v'), other)"


def test_a_table_printed_as_text_is_shown_as_the_table():
    assert render_tables("print(df.to_string())") == "display(df)"
    assert render_tables("print(df.groupby('a').sum().to_markdown())") == "display(df.groupby('a').sum())"


def test_printed_text_is_left_alone():
    code = 'print("Stablecoin supply (USD):")\nprint(f"{n} rows")\nprint("a" + str(n))\nprint(len(df))\nprint("%d rows" % n)\nprint("x".join(parts))'
    assert render_tables(code) == code


def test_a_call_with_text_and_a_table_is_left_alone():
    code = 'print("Top DEXs:", df)'
    assert render_tables(code) == code


def test_print_options_are_left_alone():
    code = "print(df, file=out)\nprint(df, end='')"
    assert render_tables(code) == code


def test_only_the_print_call_changes_and_the_rest_keeps_its_formatting():
    code = "# load\ndf = load()  # raw\nprint(df.head())  # preview\n\nfor n in df:\n    print(n)\n"
    assert render_tables(code) == "# load\ndf = load()  # raw\ndisplay(df.head())  # preview\n\nfor n in df:\n    display(n)\n"


def test_non_ascii_text_before_a_print_does_not_shift_the_edit():
    code = "title = 'Überblick – données'\nprint(df)"
    assert render_tables(code) == "title = 'Überblick – données'\ndisplay(df)"


def test_code_that_does_not_parse_is_returned_unchanged():
    code = "print(df\nx ="
    assert render_tables(code) == code
