import numpy as np
import pandas as pd
import pytest

from src import backtest, portfolios


def _index(n):
    return pd.date_range("2020-01-01", periods=n, freq="B")


def _random_returns(n=300, seed=0):
    """Cuatro activos correlacionados (un factor común) con volatilidades distintas."""
    rng = np.random.default_rng(seed)
    market = rng.normal(0, 0.01, n)
    data = {c: market * b + rng.normal(0, s, n) for c, b, s in zip("ABCD", [0.5, 1.0, 1.0, 1.5], [0.005, 0.01, 0.02, 0.02])}
    return pd.DataFrame(data, index=_index(n))


def _portfolio_variance(w, returns):
    return float(w @ returns.cov() @ w)


def test_equal_weight():
    w = portfolios.equal_weight(_random_returns())
    assert np.allclose(w, 0.25)


def test_min_variance_is_valid_long_only_portfolio():
    r = _random_returns()
    w = portfolios.min_variance(r)
    assert list(w.index) == list(r.columns)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= -1e-8).all()


def test_min_variance_two_uncorrelated_assets():
    # A y B no están correlacionados y B tiene el doble de volatilidad:
    # la solución exacta es w_A = var_B / (var_A + var_B) = 0.8
    n = 200
    a = 0.01 * np.tile([1, -1], n // 2)
    b = 0.02 * np.tile([1, 1, -1, -1], n // 4)
    w = portfolios.min_variance(pd.DataFrame({"A": a, "B": b}, index=_index(n)))
    assert w["A"] == pytest.approx(0.8, abs=1e-3)
    assert w["B"] == pytest.approx(0.2, abs=1e-3)


def test_min_variance_long_only_constraint_binds():
    # B es casi 3 veces A: sin restricción convendría vender B en corto.
    # Solo largos, lo mejor es no tener B.
    n = 200
    a = 0.01 * np.tile([1, -1], n // 2)
    noise = 0.001 * np.tile([1, 1, -1, -1], n // 4)
    w = portfolios.min_variance(pd.DataFrame({"A": a, "B": 3 * a + noise}, index=_index(n)))
    assert w["B"] == pytest.approx(0.0, abs=1e-3)
    assert w["A"] == pytest.approx(1.0, abs=1e-3)


def test_min_variance_beats_equal_weight_and_single_assets_in_sample():
    r = _random_returns()
    var_mv = _portfolio_variance(portfolios.min_variance(r), r)
    assert var_mv <= _portfolio_variance(portfolios.equal_weight(r), r) + 1e-12
    assert var_mv <= r.var().min() + 1e-12


def test_min_variance_runs_inside_backtest():
    r = _random_returns()
    rets, weights, _ = backtest.walk_forward_backtest(r, portfolios.min_variance, lookback=120, rebalance_every=20)
    assert rets.index.equals(r.index[120:])
    assert np.allclose(weights.sum(axis=1), 1.0)


# ---------- máximo Sharpe ----------

def _sharpe(w, returns):
    return float(w @ returns.mean() / np.sqrt(w @ returns.cov() @ w))


def test_max_sharpe_is_valid_long_only_portfolio():
    r = _random_returns() + 0.001
    w = portfolios.max_sharpe(r)
    assert list(w.index) == list(r.columns)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= -1e-8).all()


def test_max_sharpe_two_uncorrelated_assets():
    # Misma volatilidad, sin correlación, y A rinde el doble que B:
    # la solución exacta es proporcional al retorno, w_A = 2/3
    n = 200
    a = 0.002 + 0.01 * np.tile([1, -1], n // 2)
    b = 0.001 + 0.01 * np.tile([1, 1, -1, -1], n // 4)
    w = portfolios.max_sharpe(pd.DataFrame({"A": a, "B": b}, index=_index(n)))
    assert w["A"] == pytest.approx(2 / 3, abs=2e-3)


def test_max_sharpe_beats_equal_weight_and_single_assets_in_sample():
    r = _random_returns() + 0.001
    best = _sharpe(portfolios.max_sharpe(r), r)
    assert best >= _sharpe(portfolios.equal_weight(r), r) - 1e-9
    for col in r.columns:
        single = pd.Series(0.0, index=r.columns)
        single[col] = 1.0
        assert best >= _sharpe(single, r) - 1e-9


def test_max_sharpe_still_valid_when_all_means_are_negative():
    r = _random_returns() - 0.01
    w = portfolios.max_sharpe(r)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= -1e-8).all()


# ---------- HRP ----------

def test_hrp_is_valid_and_gives_every_asset_some_weight():
    r = _random_returns()
    w = portfolios.hrp(r)
    assert list(w.index) == list(r.columns)
    assert w.sum() == pytest.approx(1.0)
    assert (w > 0).all()


def test_hrp_uncorrelated_assets_gives_inverse_variance_weights():
    # Cuatro activos sin correlación entre sí: HRP se reduce a ponderar por 1/varianza
    n = 200
    patterns = {"A": [1, -1], "B": [1, 1, -1, -1], "C": [1, 1, 1, 1, -1, -1, -1, -1], "D": [1, -1, -1, 1, -1, 1, 1, -1]}
    scales = {"A": 0.01, "B": 0.02, "C": 0.02, "D": 0.04}
    r = pd.DataFrame({c: scales[c] * np.tile(p, n // len(p)) for c, p in patterns.items()}, index=_index(n))
    expected = 1 / r.var()
    expected /= expected.sum()
    assert np.allclose(portfolios.hrp(r), expected, atol=1e-6)


def test_hrp_favors_the_uncorrelated_asset():
    # A, B y C son casi el mismo activo; D es independiente y con la misma volatilidad.
    # Ponderar por 1/varianza daría 25% a cada uno; HRP reconoce el grupo y le da más a D.
    # (La bisección parte la lista en mitades, así que D comparte mitad con uno de los
    # repetidos, que también recibe más; los otros dos quedan por debajo.)
    rng = np.random.default_rng(1)
    n = 500
    common = rng.normal(0, 0.02, n)
    data = {c: common + rng.normal(0, 0.006, n) for c in "ABC"}
    data["D"] = rng.normal(0, 0.021, n)
    w = portfolios.hrp(pd.DataFrame(data, index=_index(n)))
    assert w["D"] > 0.29
    assert w.nsmallest(2).max() < 0.20


def test_hrp_runs_inside_backtest():
    r = _random_returns()
    rets, weights, _ = backtest.walk_forward_backtest(r, portfolios.hrp, lookback=120, rebalance_every=20)
    assert rets.index.equals(r.index[120:])
    assert np.allclose(weights.sum(axis=1), 1.0)
