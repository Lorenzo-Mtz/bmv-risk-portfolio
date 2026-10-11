import numpy as np
import pandas as pd
import pytest

from src import backtest, portfolios

LOOKBACK, EVERY = 60, 20


def _returns(n=120, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(rng.normal(0.0005, 0.02, (n, 4)), index=idx, columns=list("ABCD"))


def _winner(window):
    """Todo el peso en el activo con mayor retorno medio de la ventana."""
    w = pd.Series(0.0, index=window.columns)
    w[window.mean().idxmax()] = 1.0
    return w


def _run(returns, weight_fn=portfolios.equal_weight, every=EVERY, cost_bps=0.0):
    return backtest.walk_forward_backtest(
        returns, weight_fn, lookback=LOOKBACK, rebalance_every=every, cost_bps=cost_bps
    )


# ---------- piezas ----------

def test_rebalance_positions():
    assert list(backtest.rebalance_positions(300, 252, 21)) == [252, 273, 294]
    assert list(backtest.rebalance_positions(10, 10, 5)) == []


def test_hold_period_known_case():
    idx = pd.date_range("2020-01-01", periods=2, freq="B")
    period = pd.DataFrame({"A": [0.10, 0.00], "B": [0.00, 0.10]}, index=idx)
    rets, w_end = backtest.hold_period(pd.Series({"A": 0.5, "B": 0.5}), period)
    assert rets.index.equals(idx)
    assert rets.iloc[0] == pytest.approx(0.05)
    assert rets.iloc[1] == pytest.approx(1.10 / 1.05 - 1)  # 4.76%, no 5%: los pesos derivaron
    assert w_end["A"] == pytest.approx(0.5)
    assert w_end.sum() == pytest.approx(1.0)


def test_hold_period_weights_drift_toward_winner():
    idx = pd.date_range("2020-01-01", periods=1, freq="B")
    period = pd.DataFrame({"A": [0.10], "B": [0.00]}, index=idx)
    _, w_end = backtest.hold_period(pd.Series({"A": 0.5, "B": 0.5}), period)
    assert w_end["A"] == pytest.approx(0.55 / 1.05)
    assert w_end["B"] == pytest.approx(0.50 / 1.05)


def test_one_way_turnover():
    w_prev = pd.Series({"A": 0.5, "B": 0.5})
    w_new = pd.Series({"A": 0.2, "B": 0.8})
    assert backtest.one_way_turnover(w_new, w_prev) == pytest.approx(0.3)
    assert backtest.one_way_turnover(w_prev, w_prev) == pytest.approx(0.0)


# ---------- forma de la salida ----------

def test_output_shapes_and_dates():
    r = _returns()
    rets, weights, turnover = _run(r)
    assert rets.index.equals(r.index[LOOKBACK:])
    assert weights.index.equals(r.index[[60, 80, 100]])
    assert turnover.index.equals(weights.index)
    assert list(weights.columns) == list(r.columns)
    assert np.allclose(weights.sum(axis=1), 1.0)


def test_invalid_weights_raise():
    with pytest.raises(ValueError):
        _run(_returns(), weight_fn=lambda window: pd.Series(1.0, index=window.columns))


# ---------- look-ahead ----------

def test_window_ends_the_day_before_rebalance():
    r = _returns()
    windows = []

    def spy(window):
        windows.append(window)
        return portfolios.equal_weight(window)

    _, weights, _ = _run(r, weight_fn=spy)
    assert len(windows) == len(weights)
    for window, date in zip(windows, weights.index):
        pos = r.index.get_loc(date)
        assert len(window) == LOOKBACK
        assert window.index[-1] == r.index[pos - 1]


def test_future_returns_do_not_change_weights():
    r = _returns()
    r["A"] += 0.01  # A gana en todas las ventanas
    future = r.copy()
    future.iloc[80, future.columns.get_loc("B")] = 5.0  # +500% de B justo el día del rebalanceo

    rets, weights, _ = _run(r, weight_fn=_winner)
    rets_f, weights_f, _ = _run(future, weight_fn=_winner)

    date = r.index[80]
    assert weights.loc[date, "A"] == 1.0
    assert weights_f.loc[date].equals(weights.loc[date])  # si cambia, la ventana vio el futuro
    assert np.allclose(rets.iloc[:20], rets_f.iloc[:20])


# ---------- retornos y deriva ----------

def test_equal_weight_daily_rebalance_matches_cross_sectional_mean():
    r = _returns()
    rets, _, _ = _run(r, every=1)
    assert np.allclose(rets, r.iloc[LOOKBACK:].mean(axis=1))


def test_buy_and_hold_between_rebalances():
    r = _returns()
    rets, _, _ = _run(r)
    period = r.iloc[60:80]
    expected_growth = (1 + period).prod().mean()  # comprar partes iguales y no tocar
    assert (1 + rets.iloc[:20]).prod() == pytest.approx(expected_growth)
    assert not np.allclose(rets.iloc[:20], period.mean(axis=1))  # sin deriva sería esto


# ---------- costos ----------

def test_costs_only_on_rebalance_days_and_match_turnover():
    r = _returns()
    gross, weights, turnover = _run(r, cost_bps=0.0)
    net, _, turnover_net = _run(r, cost_bps=10.0)
    drag = gross - net

    not_rebalance = ~drag.index.isin(weights.index)
    assert np.allclose(drag[not_rebalance], 0.0)
    assert np.allclose(drag[weights.index], 10.0 / 1e4 * 2 * turnover)
    assert np.allclose(turnover, turnover_net)


def test_turnover_uses_drifted_weights():
    r = _returns()
    _, _, turnover = _run(r)
    assert turnover.iloc[0] == pytest.approx(0.5)  # de efectivo a totalmente invertido

    value = (1 + r.iloc[60:80]).prod() / 4  # posiciones de igual ponderado tras el primer periodo
    w_drift = value / value.sum()
    assert turnover.iloc[1] == pytest.approx(0.5 * (0.25 - w_drift).abs().sum())
    assert turnover.iloc[1] > 0  # con pesos objetivo (sin deriva) daría 0


def test_net_of_costs_matches_backtest_with_costs():
    r = _returns()
    gross, _, turnover = _run(r, weight_fn=_winner, cost_bps=0.0)
    net, _, _ = _run(r, weight_fn=_winner, cost_bps=25.0)
    assert np.allclose(backtest.net_of_costs(gross, turnover, 25.0), net)
