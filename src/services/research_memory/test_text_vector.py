from __future__ import annotations

import time

from src.services.research_memory import text_vector as tv


def test_filler_words_are_dropped_and_case_is_ignored():
    assert tv.words("The Uniswap TVL is up") == ["uniswap", "tvl"]


def test_addresses_and_snake_case_names_stay_whole_and_their_parts_are_searchable():
    assert "0xabc123def" in tv.words("Wallet 0xAbC123dEf bought")
    words = tv.words("column tvl_usd")
    assert "tvl_usd" in words and "tvl" in words and "usd" in words


def test_the_same_word_always_hashes_to_the_same_place():
    a = tv.document_vector("uniswap liquidity")
    b = tv.document_vector("liquidity uniswap")
    assert a.indices == b.indices and a.values == b.values


def test_a_repeated_word_counts_more_but_with_diminishing_returns():
    once = dict(zip(*(lambda v: (v.indices, v.values))(tv.document_vector("whale"))))
    five = dict(zip(*(lambda v: (v.indices, v.values))(tv.document_vector("whale " * 5))))
    key = next(iter(once))
    assert five[key] > once[key]
    assert five[key] < 5 * once[key]


def test_a_longer_text_weighs_each_word_less_than_a_short_one():
    short = tv.document_vector("whale")
    long = tv.document_vector("whale " + " ".join(f"filler{i}" for i in range(300)))
    assert long.values[long.indices.index(short.indices[0])] < short.values[0]


def test_a_query_counts_each_distinct_word_once():
    vector = tv.query_vector("whale whale whale uniswap")
    assert sorted(vector.values) == [1.0, 1.0]


def test_a_query_matches_the_stored_text_it_shares_words_with():
    doc = tv.document_vector("Uniswap TVL fell 12% after the whale sold")
    hit = set(tv.query_vector("why did uniswap tvl fall").indices) & set(doc.indices)
    miss = set(tv.query_vector("solana nft floor price").indices) & set(doc.indices)
    assert len(hit) == 2 and not miss


def test_it_needs_no_model_and_is_fast():
    text = ("Uniswap v3 pool liquidity 0xabc123 tvl_usd whale outflow arbitrum " * 40).strip()
    start = time.perf_counter()
    for _ in range(200):
        tv.document_vector(text)
    per_text_ms = (time.perf_counter() - start) / 200 * 1000
    assert per_text_ms < 5  # a few hundred words in a few milliseconds at the very most
