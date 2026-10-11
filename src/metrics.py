"""Métricas de desempeño de estrategias y portafolios."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

TD = config.TRADING_DAYS


def annualized_return(returns: pd.Series, periods: int = TD) -> float:
    """Retorno anualizado geométrico a partir de retornos simples."""
    n = returns.dropna().shape[0]
    if n == 0:
        return np.nan
    growth = (1.0 + returns.dropna()).prod()
    return float(growth ** (periods / n) - 1.0)


def annualized_volatility(returns: pd.Series, periods: int = TD) -> float:
    return float(returns.std(ddof=1) * np.sqrt(periods))


def sharpe_ratio(returns: pd.Series, rf_annual: float = config.RISK_FREE_ANNUAL, periods: int = TD) -> float:
    excess = returns - rf_annual / periods
    vol = excess.std(ddof=1)
    return float(excess.mean() / vol * np.sqrt(periods)) if vol > 0 else np.nan


def sortino_ratio(returns: pd.Series, rf_annual: float = config.RISK_FREE_ANNUAL, periods: int = TD) -> float:
    excess = returns - rf_annual / periods
    downside = excess[excess < 0]
    dd = np.sqrt((downside**2).mean()) if len(downside) else np.nan
    return float(excess.mean() / dd * np.sqrt(periods)) if dd and dd > 0 else np.nan


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Serie de drawdown (valores <= 0) a partir de retornos simples."""
    wealth = (1.0 + returns).cumprod()
    return wealth / wealth.cummax() - 1.0


def max_drawdown(returns: pd.Series) -> float:
    return float(drawdown_series(returns).min())


def turnover(weights: pd.DataFrame) -> float:
    """Rotación promedio por rebalanceo: 0.5 * sum |w_t - w_{t-1}|."""
    diffs = weights.diff().abs().sum(axis=1).iloc[1:]
    return float(0.5 * diffs.mean()) if len(diffs) else np.nan


def summary(returns: pd.Series, rf_annual: float = config.RISK_FREE_ANNUAL) -> pd.Series:
    """Resumen de métricas de una serie de retornos simples diarios."""
    return pd.Series(
        {
            "Retorno anual": annualized_return(returns),
            "Volatilidad anual": annualized_volatility(returns),
            "Sharpe": sharpe_ratio(returns, rf_annual),
            "Sortino": sortino_ratio(returns, rf_annual),
            "Max drawdown": max_drawdown(returns),
        }
    )


def compare(strategies: dict[str, pd.Series], rf_annual: float = config.RISK_FREE_ANNUAL) -> pd.DataFrame:
    """Tabla comparativa de varias estrategias (una por columna)."""
    return pd.DataFrame({name: summary(r, rf_annual) for name, r in strategies.items()})
