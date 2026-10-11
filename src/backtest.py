"""Backtest walk-forward (Bloque 2).

Convención de tiempo: el retorno de la fila `t` va del cierre de `t-1` al cierre de `t`.
Los pesos que ganan el retorno de `t` se deciden al cierre de `t-1`, así que solo pueden
usar las filas anteriores a `t`.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from . import config

WeightFn = Callable[[pd.DataFrame], pd.Series]


def rebalance_positions(n_obs: int, lookback: int, rebalance_every: int) -> list[int]:
    """Posiciones enteras (para `.iloc`) de las filas en las que se rebalancea.

    La primera es `lookback` (antes no hay una ventana completa) y de ahí avanza en pasos
    de `rebalance_every` hasta el final de los datos.

    Ejemplo: n_obs=300, lookback=252, rebalance_every=21 -> [252, 273, 294].
    """
    return list(range(lookback, n_obs, rebalance_every))



def hold_period(weights: pd.Series, period_returns: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Portafolio que se compra con `weights` y no se toca durante `period_returns`.

    Entre rebalanceos los pesos derivan: lo que sube pesa más y lo que baja pesa menos.
    Por eso el retorno del día NO es `weights @ retornos_del_día` salvo el primer día.

    Devuelve
    --------
    (retornos diarios del portafolio, pesos al cierre del último día)

    Ejemplo con pesos A=0.5, B=0.5:
      día 1: A +10%, B 0%  -> posiciones 0.55 y 0.50, valor 1.05, retorno +5.00%
      día 2: A 0%,  B +10% -> posiciones 0.55 y 0.55, valor 1.10, retorno +4.76% (no +5%)
      pesos finales: A=0.5, B=0.5
    """
    # Valor de cada posición al cierre de cada día, por cada peso invertido al inicio
    positions = (1.0 + period_returns).cumprod() * weights
    value = positions.sum(axis=1)

    # Valor del día anterior; el "ayer" del primer día es lo invertido al inicio
    prev_value = value.shift(1)
    prev_value.iloc[0] = weights.sum()

    returns = value / prev_value - 1.0
    end_weights = positions.iloc[-1] / value.iloc[-1]
    return returns, end_weights


def one_way_turnover(w_new: pd.Series, w_prev: pd.Series) -> float:
    """Rotación de un rebalanceo: 0.5 * sum|w_new - w_prev|.

    `w_prev` son los pesos YA derivados justo antes de rebalancear, no los pesos objetivo
    del rebalanceo anterior. Ejemplo: de (0.5, 0.5) a (0.2, 0.8) la rotación es 0.3.
    """
    return float(0.5* (w_new-w_prev).abs().sum())



def _check_weights(weights: pd.Series, columns: pd.Index) -> pd.Series:
    """Alinea los pesos a `columns` y verifica que sean solo largos y sumen 1."""
    w = pd.Series(weights, dtype=float).reindex(columns).fillna(0.0)
    if (w < -1e-8).any():
        raise ValueError(f"Pesos negativos: {w[w < -1e-8].round(4).to_dict()}")
    if not np.isclose(w.sum(), 1.0, atol=1e-6):
        raise ValueError(f"Los pesos suman {w.sum():.6f}, no 1")
    return w


def walk_forward_backtest(
    simple_returns: pd.DataFrame,
    weight_fn: WeightFn,
    lookback: int = 252,
    rebalance_every: int = 21,
    cost_bps: float = config.TRANSACTION_COST_BPS,
) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """Backtest con rebalanceo periódico, deriva de pesos y costos de transacción.

    Regla anti look-ahead: los pesos que se aplican desde la fila `t` se calculan
    SOLO con las `lookback` filas anteriores a `t` (la fila `t` no entra en la ventana).

    Parámetros
    ----------
    simple_returns : retornos simples diarios (fechas x activos), sin faltantes.
    weight_fn      : función ventana_de_retornos -> pesos (pd.Series, suma 1, solo largos).
    lookback       : tamaño de la ventana de estimación (días).
    rebalance_every: cada cuántos días se rebalancea.
    cost_bps       : costo en puntos base por cada peso operado (compra o venta).

    Devuelve
    --------
    (retornos, pesos, rotación)
      retornos : retornos diarios netos de costos, desde la fila `lookback` hasta el final.
      pesos    : pesos objetivo por fecha de rebalanceo (fechas x activos).
      rotación : rotación de cada rebalanceo (ver `one_way_turnover`).

    Costos
    ------
    costo = cost_bps / 1e4 * sum|w_nuevo - w_previo_derivado| = cost_bps / 1e4 * 2 * rotación,
    y se resta del retorno del día del rebalanceo. Se parte de efectivo (pesos previos en cero),
    así que el primer rebalanceo tiene rotación 0.5 y paga el costo de comprar todo el portafolio.
    """
    if simple_returns.isna().any().any():
        raise ValueError("simple_returns tiene faltantes; pásalo por data.clean_prices primero.")
    n_obs = len(simple_returns)
    if n_obs <= lookback:
        raise ValueError(f"Se necesitan más de {lookback} filas; hay {n_obs}.")

    cost_rate = cost_bps / 1e4
    pieces: list[pd.Series] = []                      # un tramo de retornos por periodo
    weights_hist: dict[pd.Timestamp, pd.Series] = {}
    turnover_hist: dict[pd.Timestamp, float] = {}
    w_prev = pd.Series(0.0, index=simple_returns.columns)  # se parte de efectivo

    for start in rebalance_positions(n_obs, lookback, rebalance_every):
        date = simple_returns.index[start]

        # Ventana de estimación: no incluye la fila `start` (sin look-ahead)
        ventana = simple_returns.iloc[start - lookback:start]

        # Pesos objetivo, validados (solo largos, suman 1)
        w_target = _check_weights(weight_fn(ventana), simple_returns.columns)

        # Rotación contra los pesos ya derivados
        turnover = one_way_turnover(w_target, w_prev)

        # Costo sobre el total operado (compras + ventas)
        costo = cost_rate * 2 * turnover

        # Se mantienen los pesos hasta el siguiente rebalanceo
        period = simple_returns.iloc[start:start + rebalance_every]
        period_returns, w_prev = hold_period(w_target, period)
        period_returns.iloc[0] -= costo  # el costo se paga el día del rebalanceo

        # Resultados de este periodo
        pieces.append(period_returns)
        weights_hist[date] = w_target
        turnover_hist[date] = turnover

    returns = pd.concat(pieces).rename("portafolio")
    weights = pd.DataFrame(weights_hist).T
    turnover = pd.Series(turnover_hist, name="rotación")
    return returns, weights, turnover


def net_of_costs(gross_returns: pd.Series, turnover: pd.Series, cost_bps: float) -> pd.Series:
    """Aplica costos a retornos brutos (de un backtest con `cost_bps=0`).

    Los pesos no dependen del costo, así que basta restar `cost_bps / 1e4 * 2 * rotación`
    en cada fecha de rebalanceo. Da lo mismo que volver a correr el backtest con ese costo.
    """
    net = gross_returns.copy()
    net.loc[turnover.index] -= cost_bps / 1e4 * 2.0 * turnover
    return net
