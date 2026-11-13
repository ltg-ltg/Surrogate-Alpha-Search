import numpy as np

from src.experiment.mmr import mmr_select, percentile_rank, unit_feature_vectors


def test_percentile_rank():
    r = percentile_rank(np.array([3.0, 1.0, 2.0, 2.0]))
    np.testing.assert_allclose(r, [1.0, 0.25, 0.625, 0.625])


def test_lambda_zero_is_top_by_relevance():
    U = unit_feature_vectors(np.random.default_rng(0).normal(size=(20, 5)))
    rel = np.linspace(0.05, 1.0, 20)
    picks = mmr_select(np.arange(20), rel, U, 0.0, 5)
    assert picks.tolist() == [19, 18, 17, 16, 15]


def test_mmr_skips_structural_twins():
    # Rows 0 and 1 are identical structures with the top relevance; row 2 is different.
    X = np.array([[1.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    U = unit_feature_vectors(X)
    rel = np.array([1.0, 0.99, 0.6, 0.5])
    picks = mmr_select(np.arange(4), rel, U, 1.0, 2)
    assert picks[0] == 0 and picks[1] != 1


def test_mmr_is_deterministic_and_unique():
    rng = np.random.default_rng(1)
    U = unit_feature_vectors(rng.normal(size=(200, 8)))
    rel = rng.uniform(size=200)
    a = mmr_select(np.arange(200), rel, U, 0.5, 50)
    b = mmr_select(np.arange(200), rel, U, 0.5, 50)
    assert a.tolist() == b.tolist() and len(set(a.tolist())) == 50
