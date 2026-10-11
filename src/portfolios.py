"""Construcción de portafolios (Bloque 2).

Cada función recibe una ventana de retornos (fechas x activos) y devuelve un
vector de pesos (pd.Series, suma 1, solo largos) indexado por activo.
Esto permite usarlas directamente dentro del backtest walk-forward.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

from . import config

TD = config.TRADING_DAYS


def equal_weight(returns: pd.DataFrame) -> pd.Series:
    n = returns.shape[1]
    return pd.Series(np.full(n, 1.0 / n), index=returns.columns)


def _optimize(objective: Callable[[np.ndarray], float], n: int, max_weight: float = 1.0) -> np.ndarray:
    """Minimiza `objective(w)` con pesos entre 0 y `max_weight` que suman 1 (SLSQP)."""
    if max_weight * n < 1.0:
        raise ValueError(f"max_weight={max_weight} no permite sumar 1 con {n} activos")
    result = minimize(
        objective,
        x0=np.full(n, 1.0 / n),                      # punto de partida: igual ponderado
        method="SLSQP",
        bounds=[(0.0, max_weight)] * n,              # solo largos
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
        # ftol estricto: con la tolerancia por defecto SLSQP se detiene casi en el punto de partida
        options={"ftol": 1e-12, "maxiter": 500},
    )
    if not result.success:
        raise RuntimeError(f"El optimizador no convergió: {result.message}")
    # Limpia residuos numéricos (p. ej. -1e-17) y vuelve a normalizar
    w = np.clip(result.x, 0.0, None)
    return w / w.sum()


def min_variance(returns: pd.DataFrame) -> pd.Series:
    """Portafolio de mínima varianza, solo largos, suma 1.

    Minimiza w' Sigma w. No usa retornos esperados, solo la covarianza muestral.
    """
    # Anualizar no cambia la solución; solo evita trabajar con números del orden de 1e-4
    cov = returns.cov().values * TD
    w = _optimize(lambda w: w @ cov @ w, n=returns.shape[1])
    return pd.Series(w, index=returns.columns)


def max_sharpe(returns: pd.DataFrame, rf_annual: float = 0.0, max_weight: float = 1.0) -> pd.Series:
    """Portafolio de máximo Sharpe, solo largos, suma 1.

    Maximiza (w'mu - rf) / sqrt(w' Sigma w). Es muy sensible al ruido en mu;
    `max_weight` acota el peso por activo (p. ej. 0.15) para limitar la concentración.
    """
    mu = returns.mean().values * TD
    cov = returns.cov().values * TD

    # Si ningún activo le gana a la tasa libre de riesgo en la ventana, el Sharpe máximo es
    # negativo y maximizarlo premiaría más volatilidad; se usa mínima varianza en su lugar.
    if (mu <= rf_annual).all():
        w = _optimize(lambda w: w @ cov @ w, n=len(mu), max_weight=max_weight)
        return pd.Series(w, index=returns.columns)

    def neg_sharpe(w: np.ndarray) -> float:
        return -(w @ mu - rf_annual) / np.sqrt(w @ cov @ w)

    w = _optimize(neg_sharpe, n=len(mu), max_weight=max_weight)
    return pd.Series(w, index=returns.columns)


def _cluster_variance(cov: pd.DataFrame, items: list[str]) -> float:
    """Varianza de un grupo, ponderando sus activos por el inverso de su varianza."""
    sub = cov.loc[items, items].values
    ivp = 1.0 / np.diag(sub)
    ivp /= ivp.sum()
    return float(ivp @ sub @ ivp)


def hrp(returns: pd.DataFrame) -> pd.Series:
    """Hierarchical Risk Parity (López de Prado, 2016).

    No invierte la matriz de covarianza ni usa retornos esperados:
      1. Agrupa los activos por correlación (clustering jerárquico).
      2. Los ordena para que los parecidos queden juntos (quasi-diagonalización).
      3. Reparte el peso de arriba hacia abajo, partiendo la lista ordenada en mitades y
         dando más peso a la mitad con menor varianza (bisección recursiva).
    """
    cov = returns.cov()
    corr = returns.corr()

    # 1. Distancia de correlación: 0 si corr = 1, 1 si corr = -1
    dist = np.sqrt(0.5 * (1.0 - corr.values).clip(min=0.0))
    link = linkage(squareform(dist, checks=False), method="single")

    # 2. Orden de las hojas del dendrograma
    order = [returns.columns[i] for i in leaves_list(link)]

    # 3. Bisección recursiva
    weights = pd.Series(1.0, index=order)
    clusters = [order]
    while clusters:
        # Parte en dos cada grupo que todavía tenga más de un activo
        halves = [c[i:j] for c in clusters for i, j in ((0, len(c) // 2), (len(c) // 2, len(c))) if len(c) > 1]
        for left, right in zip(halves[::2], halves[1::2]):
            var_left = _cluster_variance(cov, left)
            var_right = _cluster_variance(cov, right)
            alpha = 1.0 - var_left / (var_left + var_right)  # la mitad menos riesgosa recibe más
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha
        clusters = halves

    return weights.reindex(returns.columns)
