import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src import risk


def _series(values):
    return pd.Series(values, index=pd.date_range("2020-01-01", periods=len(values), freq="B"))


def _normal(n=2000, seed=0, sigma=0.01):
    return _series(np.random.default_rng(seed).normal(0, sigma, n))


# ---------- CVaR ----------

def test_cvar_historical_known_case():
    # 100 retornos de -1% a -100%: el 5% peor son -96%..-100%, cuyo promedio es -98%
    r = _series(-np.arange(1, 101) / 100)
    assert risk.cvar_historical(r, alpha=0.95) == pytest.approx(0.98, abs=0.005)


def test_cvar_is_at_least_var():
    r = _normal()
    for alpha in (0.95, 0.99):
        assert risk.cvar_historical(r, alpha) >= risk.var_historical(r, alpha)
        assert risk.cvar_parametric(r, alpha) >= risk.var_parametric(r, alpha)


def test_cvar_parametric_close_to_historical_for_normal_data():
    r = _normal(n=200_000)
    assert risk.cvar_parametric(r, 0.99) == pytest.approx(risk.cvar_historical(r, 0.99), rel=0.03)


# ---------- VaR rodante ----------

@pytest.mark.parametrize("method", ["historical", "parametric", "ewma", "fhs"])
def test_rolling_var_has_no_look_ahead(method):
    r = _normal(n=400)
    shocked = r.copy()
    shocked.iloc[300:] = -0.50  # desplome a partir del día 300
    base = risk.rolling_var(r, window=100, method=method)
    after = risk.rolling_var(shocked, window=100, method=method)
    assert base.iloc[:100].isna().all()
    assert base.iloc[101:].notna().all()
    # El VaR hasta el día 300 inclusive no puede cambiar: solo usa datos hasta el día anterior
    assert np.allclose(base.iloc[101:301], after.iloc[101:301])
    assert (after.iloc[302:] > base.iloc[302:]).all()  # después del desplome sí debe subir


def test_rolling_var_historical_matches_static_var_on_previous_window():
    r = _normal(n=400)
    rolled = risk.rolling_var(r, window=100, alpha=0.99, method="historical")
    assert rolled.iloc[250] == pytest.approx(risk.var_historical(r.iloc[150:250], alpha=0.99))


# ---------- Kupiec ----------

def test_kupiec_no_evidence_when_rate_matches():
    out = risk.kupiec_pof_test(n_obs=1000, n_exceptions=10, alpha=0.99)
    assert out["LR"] == pytest.approx(0.0, abs=1e-9)
    assert out["p_value"] == pytest.approx(1.0)
    assert not out["reject_H0_5pct"]


def test_kupiec_matches_binomial_likelihood_ratio():
    n, x, alpha = 1000, 25, 0.99
    expected = -2 * (stats.binom.logpmf(x, n, 1 - alpha) - stats.binom.logpmf(x, n, x / n))
    out = risk.kupiec_pof_test(n, x, alpha)
    assert out["LR"] == pytest.approx(expected)
    assert out["reject_H0_5pct"]  # 25 excepciones cuando se esperaban 10


def test_kupiec_handles_zero_exceptions():
    out = risk.kupiec_pof_test(n_obs=250, n_exceptions=0, alpha=0.99)
    assert np.isfinite(out["LR"])
    assert out["LR"] == pytest.approx(-2 * 250 * np.log(0.99))


def test_backtest_var_counts_and_rate():
    r = _series([-0.03, 0.01, -0.01, -0.05])
    var = _series([0.02, 0.02, 0.02, 0.02])
    out = risk.backtest_var(r, var, alpha=0.99)
    assert out["Días"] == 4
    assert out["Excepciones"] == 2
    assert out["Pérdida media en excepción"] == pytest.approx(0.04)
