"""Phase 3, step 3: the two halves of the BM25 formula, as small pure functions.

    score(chunk, query) = sum over the query terms t of   idf(t) * tf_weight(t, chunk)

    idf(n_docs, doc_freq)  = ln(1 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))

                                          tf * (k1 + 1)
    tf_weight(tf, dl, avgdl, k1, b) = -----------------------------------
                                       tf + k1 * (1 - b + b * dl / avgdl)

    tf = how often the term occurs in the chunk, dl = the chunk's length in tokens,
    avgdl = the average length over all chunks, k1 and b = the two tuning knobs.

Every expected number below was worked out by hand (the working is in the comments).
"""

import math

import pytest

from cairn.retrieval.bm25 import idf, tf_weight


# ------------------------------------------------------------------------- idf
def test_idf_by_hand():
    # idf(2, 1)   = ln(1 + (2 - 1 + 0.5) / (1 + 0.5))   = ln(1 + 1.5 / 1.5)    = ln 2
    assert idf(2, 1) == pytest.approx(math.log(2))
    # idf(10, 1)  = ln(1 + 9.5 / 1.5)                   = ln 7.3333            = 1.99243
    assert idf(10, 1) == pytest.approx(1.99243, abs=1e-5)
    # idf(1, 1)   = ln(1 + 0.5 / 1.5)                   = ln 1.3333            = 0.28768
    assert idf(1, 1) == pytest.approx(0.28768, abs=1e-5)
    # idf(100, 1) = ln(1 + 99.5 / 1.5)                  = ln 67.333            = 4.20966
    assert idf(100, 1) == pytest.approx(4.20966, abs=1e-5)


def test_a_term_in_every_chunk_is_worth_almost_nothing():
    # idf(10, 10) = ln(1 + 0.5 / 10.5) = ln 1.047619 = 0.04652
    assert idf(10, 10) == pytest.approx(0.04652, abs=1e-5)
    assert idf(1000, 1000) < 0.001
    assert idf(1000, 1000) < idf(1000, 500) < idf(1000, 10) < idf(1000, 1)


def test_idf_falls_as_a_term_gets_more_common():
    values = [idf(50, df) for df in range(1, 51)]
    assert values == sorted(values, reverse=True)
    assert len(set(values)) == len(values)


def test_idf_is_never_negative_not_even_for_a_term_in_every_chunk():
    for n in (1, 2, 3, 10, 1000):
        for df in range(1, n + 1):
            assert idf(n, df) > 0


# ------------------------------------------------------------------- tf_weight
def test_a_chunk_of_average_length_by_hand():
    # dl = avgdl, so the length factor (1 - b + b * dl / avgdl) is exactly 1 for any b
    # tf = 1: 1 * 2.2 / (1 + 1.2)      = 1.0
    assert tf_weight(1, 10, 10, k1=1.2, b=0.75) == pytest.approx(1.0)
    # tf = 2: 2 * 2.2 / (2 + 1.2)      = 4.4 / 3.2 = 1.375
    assert tf_weight(2, 10, 10, k1=1.2, b=0.75) == pytest.approx(1.375)


def test_the_defaults_are_k1_1_2_and_b_0_75():
    assert tf_weight(2, 10, 10) == pytest.approx(1.375)
    assert tf_weight(1, 20, 10) == pytest.approx(0.7096774, abs=1e-6)


def test_a_long_chunk_is_penalised_and_a_short_one_is_rewarded():
    # dl = 20, avgdl = 10: length factor = 0.25 + 0.75 * 2    = 1.75
    #   1 * 2.2 / (1 + 1.2 * 1.75) = 2.2 / 3.1                 = 0.709677
    assert tf_weight(1, 20, 10, k1=1.2, b=0.75) == pytest.approx(0.709677, abs=1e-6)
    # dl = 5, avgdl = 10: length factor = 0.25 + 0.75 * 0.5   = 0.625
    #   2.2 / (1 + 1.2 * 0.625) = 2.2 / 1.75                   = 1.257143
    assert tf_weight(1, 5, 10, k1=1.2, b=0.75) == pytest.approx(1.257143, abs=1e-6)


def test_b_zero_switches_length_normalisation_off():
    short = tf_weight(1, 1, 10, k1=1.2, b=0)
    long = tf_weight(1, 1000, 10, k1=1.2, b=0)
    assert short == pytest.approx(1.0)  # the length factor is 1 - 0 + 0 = 1 whatever dl is
    assert long == pytest.approx(short)


def test_b_one_is_full_length_normalisation():
    # dl = 20, avgdl = 10: length factor = 20 / 10 = 2      1 * 2.2 / (1 + 1.2 * 2) = 2.2 / 3.4
    assert tf_weight(1, 20, 10, k1=1.2, b=1) == pytest.approx(0.647059, abs=1e-6)
    # dl = 5: length factor = 0.5                           2.2 / (1 + 0.6) = 1.375
    assert tf_weight(1, 5, 10, k1=1.2, b=1) == pytest.approx(1.375)


def test_a_bigger_b_penalises_a_long_chunk_more():
    weights = [tf_weight(1, 30, 10, k1=1.2, b=b) for b in (0, 0.25, 0.5, 0.75, 1)]
    assert weights == sorted(weights, reverse=True)
    assert len(set(weights)) == len(weights)


def test_repeating_a_term_helps_but_saturates():
    weights = [tf_weight(tf, 10, 10, k1=1.2, b=0.75) for tf in (1, 2, 3, 5, 10, 100, 1000)]
    assert weights == sorted(weights) and len(set(weights)) == len(weights)  # more is better
    assert weights[-1] < 1.2 + 1  # but never above k1 + 1 = 2.2
    assert weights[-1] == pytest.approx(2.19736, abs=1e-5)  # 1000 * 2.2 / 1001.2
    assert weights[-1] / weights[0] < 2.2  # 1000 occurrences are worth < 2.2 times one


def test_k1_zero_means_only_presence_counts():
    # k1 = 0: tf * 1 / (tf + 0) = 1 for every tf and every length
    assert tf_weight(1, 10, 10, k1=0, b=0.75) == pytest.approx(1.0)
    assert tf_weight(7, 10, 10, k1=0, b=0.75) == pytest.approx(1.0)
    assert tf_weight(5, 50, 10, k1=0, b=0.75) == pytest.approx(1.0)


def test_a_bigger_k1_lets_repetition_count_for_longer():
    low = tf_weight(10, 10, 10, k1=0.5, b=0.75) / tf_weight(1, 10, 10, k1=0.5, b=0.75)
    high = tf_weight(10, 10, 10, k1=3.0, b=0.75) / tf_weight(1, 10, 10, k1=3.0, b=0.75)
    assert high > low > 1
