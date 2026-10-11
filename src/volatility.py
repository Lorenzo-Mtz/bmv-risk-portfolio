"""Pronóstico de volatilidad: EWMA, GARCH y XGBoost (Bloque 4)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def realized_volatility(returns: pd.Series, window: int = 21) -> pd.Series:
    """Volatilidad realizada en ventana móvil (desviación estándar, no anualizada)."""
    return returns.rolling(window).std(ddof=1)


def ewma_volatility(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    """Volatilidad EWMA estilo RiskMetrics (varianza condicional con lambda)."""
    var = (returns**2).ewm(alpha=1.0 - lam, adjust=False).mean()
    return np.sqrt(var)


def qlike(realized_var: pd.Series, forecast_var: pd.Series) -> float:
    """Pérdida QLIKE (robusta para evaluar pronósticos de varianza). Menor es mejor."""
    df = pd.concat([realized_var, forecast_var], axis=1, join="inner").dropna()
    r, f = df.iloc[:, 0], df.iloc[:, 1]
    return float((r / f - np.log(r / f) - 1.0).mean())


def garch_forecast(returns: pd.Series, **kwargs) -> pd.Series:
    """Pronóstico a 1 día con GARCH walk-forward.

    TODO (Bloque 4):
      - Usar `arch.arch_model(returns * 100, vol="GARCH", p=1, q=1)`.
      - Reestimar cada N días con ventana expansiva o móvil (NO ajustar con toda la muestra).
      - Devolver la volatilidad pronosticada para t+1 indexada por la fecha de t+1.
    """
    raise NotImplementedError("Implementar en el Bloque 4")


def xgboost_forecast(returns: pd.Series, **kwargs) -> pd.Series:
    """Pronóstico de volatilidad con XGBoost.

    TODO (Bloque 4):
      - Features: rezagos de |r|, r², volatilidad realizada (5, 21 días), etc.
      - Target: volatilidad realizada futura.
      - Validación walk-forward (TimeSeriesSplit o reentrenamiento periódico).
    """
    raise NotImplementedError("Implementar en el Bloque 4")
