import numpy as np
import pandas as pd
import pytest

from src import metrics, risk, volatility


def _series(values):
    return pd.Series(values, index=pd.date_range("2020-01-01", periods=len(values), freq="B"))


def test_annualized_return_constant_growth():
    r = _series([0.001] * 252)
    assert metrics.annualized_return(r, periods=252) == pytest.approx(1.001**252 - 1)


def test_max_drawdown_known_case():
    # +10%, -20%, +5%  -> riqueza 1.10, 0.88, 0.924 ; drawdown mínimo = 0.88/1.10 - 1 = -20%
    r = _series([0.10, -0.20, 0.05])
    assert metrics.max_drawdown(r) == pytest.approx(-0.20)


def test_sharpe_zero_vol_returns_nan():
    r = _series([0.001] * 50)
    assert np.isnan(metrics.sharpe_ratio(r))


def test_turnover_two_rebalances():
    w = pd.DataFrame({"A": [0.5, 0.2], "B": [0.5, 0.8]})
    # |0.2-0.5| + |0.8-0.5| = 0.6 ; 0.5 * 0.6 = 0.3
    assert metrics.turnover(w) == pytest.approx(0.3)


def test_var_historical_known_quantile():
    r = _series(np.linspace(-0.05, 0.05, 1001))
    # el cuantil 1% de una rejilla uniforme en [-5%, 5%] es -4.9%
    assert risk.var_historical(r, alpha=0.99) == pytest.approx(0.049, abs=1e-3)


def test_var_parametric_close_to_normal_quantile():
    rng = np.random.default_rng(0)
    r = _series(rng.normal(0, 0.01, 200_000))
    assert risk.var_parametric(r, alpha=0.99) == pytest.approx(0.01 * 2.3263, rel=0.02)


def test_count_exceptions():
    r = _series([-0.03, 0.01, -0.01, -0.05])
    var = _series([0.02, 0.02, 0.02, 0.02])
    assert risk.count_exceptions(r, var) == 2


def test_ewma_volatility_positive_and_same_index():
    rng = np.random.default_rng(1)
    r = _series(rng.normal(0, 0.01, 500))
    vol = volatility.ewma_volatility(r)
    assert vol.index.equals(r.index)
    assert (vol.dropna() > 0).all()
