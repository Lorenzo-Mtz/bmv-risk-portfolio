import numpy as np
import pandas as pd
import pytest

from src import volatility


def _index(n):
    return pd.date_range("2018-01-01", periods=n, freq="B")


def _garch_series(n=700, seed=0, omega=2e-6, alpha=0.10, beta=0.85):
    """Retornos simulados con un GARCH(1,1) y su volatilidad verdadera."""
    rng = np.random.default_rng(seed)
    r, sigma = np.zeros(n), np.zeros(n)
    var = omega / (1 - alpha - beta)
    for t in range(n):
        sigma[t] = np.sqrt(var)
        r[t] = sigma[t] * rng.standard_normal()
        var = omega + alpha * r[t] ** 2 + beta * var
    return pd.Series(r, index=_index(n)), pd.Series(sigma, index=_index(n))


# ---------- evaluación ----------

def test_qlike_is_zero_for_perfect_forecast_and_positive_otherwise():
    realized = pd.Series([1.0, 2.0, 4.0])
    assert volatility.qlike(realized, realized) == pytest.approx(0.0)
    assert volatility.qlike(realized, realized * 2) > 0
    assert volatility.qlike(realized, realized / 2) > volatility.qlike(realized, realized * 2)  # castiga más subestimar


def test_mse_known_case():
    assert volatility.mse(pd.Series([1.0, 3.0]), pd.Series([2.0, 1.0])) == pytest.approx(2.5)


def test_loss_difference_tstat_sign():
    rng = np.random.default_rng(0)
    better, worse = pd.Series(rng.random(500)), pd.Series(rng.random(500) + 0.5)
    assert volatility.loss_difference_tstat(better, worse) < -2


# ---------- pronósticos sin look-ahead ----------

def _assert_no_look_ahead(forecast_fn, n=420, cut=350):
    r, _ = _garch_series(n)
    shocked = r.copy()
    shocked.iloc[cut:] = -0.20
    base, after = forecast_fn(r), forecast_fn(shocked)
    # El pronóstico del día `cut` solo usa datos hasta `cut - 1`: no puede cambiar
    valid = base.iloc[: cut + 1].dropna().index
    assert len(valid) > 0
    assert np.allclose(base.loc[valid], after.loc[valid])
    assert after.iloc[-1] > base.iloc[-1]  # después del choque sí debe reaccionar


def test_simple_forecasts_have_no_look_ahead():
    _assert_no_look_ahead(lambda r: volatility.rolling_forecast(r, window=100))
    _assert_no_look_ahead(volatility.ewma_forecast)


def test_garch_forecast_has_no_look_ahead():
    _assert_no_look_ahead(lambda r: volatility.garch_forecast(r, min_train=300, refit_every=25))


def test_xgboost_forecast_has_no_look_ahead():
    _assert_no_look_ahead(lambda r: volatility.xgboost_forecast(r, min_train=300, refit_every=25, n_estimators=20)[0])


def test_features_only_use_past_data():
    r, _ = _garch_series(200)
    shocked = r.copy()
    shocked.iloc[150:] = 0.10
    base, after = volatility.make_features(r), volatility.make_features(shocked)
    pd.testing.assert_frame_equal(base.iloc[:150], after.iloc[:150])


# ---------- comportamiento ----------

def test_garch_forecast_shape_and_tracks_true_volatility():
    r, sigma = _garch_series(900)
    forecast = volatility.garch_forecast(r, min_train=500, refit_every=50)
    assert forecast.index.equals(r.index)
    assert forecast.iloc[:500].isna().all()
    assert (forecast.iloc[500:] > 0).all()
    assert np.corrcoef(forecast.iloc[500:], sigma.iloc[500:])[0, 1] > 0.8


def test_xgboost_forecast_shape_and_importance():
    r, _ = _garch_series(500)
    forecast, importance = volatility.xgboost_forecast(r, min_train=300, refit_every=50, n_estimators=20)
    assert forecast.index.equals(r.index)
    assert forecast.iloc[:300].isna().all()
    assert (forecast.iloc[300:] > 0).all()
    assert set(importance.index) == set(volatility.make_features(r).columns)
    assert importance.sum() == pytest.approx(1.0, abs=1e-5)
