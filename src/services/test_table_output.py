from src.services.table_output import render_tables


def test_a_printed_dataframe_becomes_display():
    assert render_tables("print(df)") == "df"
    assert render_tables("print(df.head(10))") == "df.head(10)"


def test_printed_series_stats_and_several_tables_become_display():
    assert render_tables("print(top.sort_values('v'), other)") == "display(top.sort_values('v'), other)"


def test_a_table_printed_as_text_is_shown_as_the_table():
    assert render_tables("print(df.to_string())") == "df"
    assert render_tables("print(df.groupby('a').sum().to_markdown())") == "df.groupby('a').sum()"


def test_the_last_table_is_returned_and_earlier_ones_are_displayed():
    code = "yr = df.groupby('y').sum()\nprint(df.head())\nprint(yr.to_string())\n"
    assert render_tables(code) == "yr = df.groupby('y').sum()\ndisplay(df.head())\nyr\n"
    assert render_tables("print(a, b)") == "display(a, b)"
    assert render_tables("display(yr)") == "display(yr)"


def test_formatting_arguments_do_not_keep_a_table_as_text():
    assert render_tables("print(yr.to_string(index=False))") == "yr"
    assert render_tables("print(df.to_string(buf))\nx = 1") == "display(df.to_string(buf))\nx = 1"


def test_a_table_as_text_next_to_text_is_split_from_it():
    assert render_tables('print("Totals:", yr.to_string())') == 'print("Totals:"); yr'
    assert render_tables('print("By year\\n" + yr.to_string())') == 'print("By year\\n"); yr'
    assert render_tables('print(f"{yr.to_string()}")') == "yr"
    assert render_tables('for k in ks:\n    print(f"{k}:\\n{t[k].to_string()}")') == "for k in ks:\n    print(f'{k}:\\n'); display(t[k])"


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
    assert render_tables(code) == "title = 'Überblick – données'\ndf"


def test_code_that_does_not_parse_is_returned_unchanged():
    code = "print(df\nx ="
    assert render_tables(code) == code
