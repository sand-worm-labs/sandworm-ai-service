from src.services.open_data.service import (
    OPEN_DATA_SOURCES,
    block_context,
    match_open_data,
    open_data_from_prompt,
    planner_context,
    free_plan_reply_note,
    use_open_data,
)


def ids(query: str) -> list[str]:
    return [s["id"] for s in match_open_data(query)]


def test_queries_match_the_apis_that_cover_them():
    assert ids("stablecoin supply over time")[0] == "defillama"
    assert ids("perp funding rates and open interest")[0] == "hyperliquid"
    assert ids("bitcoin hash rate and miner revenue")[0] == "bitcoin"
    assert ids("prediction market odds for the election")[0] == "polymarket"


def test_a_query_with_no_match_falls_back_to_the_general_sources():
    assert ids("zzz qqq") == ["defillama", "coingecko"]


def test_every_source_has_what_the_prompts_print():
    for source in OPEN_DATA_SOURCES:
        assert source["name"] and source["covers"] and source["baseUrl"] and source["endpoints"]


def test_planner_context_lists_each_matched_source_once():
    context = planner_context(["perp funding rates", "open interest on perp markets"])
    assert "Build this from open data only" in context
    assert "a visualization for each chart" in context
    assert context.count("- Hyperliquid:") == 1


def test_block_context_gives_endpoints_and_key_handling():
    context = block_context("token transfers of a wallet address")
    assert "api.etherscan.io" in context
    assert 'os.environ["ETHERSCAN_API_KEY"] (required' in context
    assert "action=tokentx" in context


def test_the_prompt_says_which_data_to_use():
    assert open_data_from_prompt("build a stablecoin notebook with open data") is True
    assert open_data_from_prompt("use public APIs for this") is True
    assert open_data_from_prompt("use our data and tools") is False
    assert open_data_from_prompt("query Dune for daily active wallets") is False
    assert open_data_from_prompt("stablecoin supply by chain") is None
    assert open_data_from_prompt("compare open data with our data") is None


def test_the_latest_prompt_that_names_a_mode_decides():
    assert use_open_data(["use open data", "now add a chart"], sql_available=True) is True
    assert use_open_data(["use open data", "switch to our data"], sql_available=True) is False


def test_without_chain_sql_it_is_always_open_data():
    assert use_open_data(["use our data"], sql_available=False) is True
    assert use_open_data(["using dune, show top pools"], sql_available=False) is True


def test_with_nothing_said_it_is_open_data_only_when_sql_cannot_run():
    assert use_open_data(["stablecoin supply by chain"], sql_available=False) is True
    assert use_open_data(["stablecoin supply by chain"], sql_available=True) is False


def test_free_plan_reply_names_what_was_left_out_and_how_to_upgrade():
    note = free_plan_reply_note(["holders of the token", "every swap on the pool"])
    assert "holders of the token; every swap on the pool" in note
    assert "Settings > Plan" in note
