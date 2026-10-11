"""VaR, CVaR y backtesting de excepciones (Bloque 3)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import special, stats

from . import volatility


def var_historical(returns: pd.Series, alpha: float = 0.99) -> float:
    """VaR histórico a nivel `alpha`, reportado como pérdida positiva."""
    return float(-np.quantile(returns.dropna(), 1.0 - alpha))


def var_parametric(returns: pd.Series, alpha: float = 0.99) -> float:
    """VaR paramétrico (normal) a nivel `alpha`, reportado como pérdida positiva."""
    mu, sigma = returns.mean(), returns.std(ddof=1)
    return float(-(mu + sigma * stats.norm.ppf(1.0 - alpha)))


def cvar_historical(returns: pd.Series, alpha: float = 0.99) -> float:
    """CVaR / Expected Shortfall histórico: pérdida promedio más allá del VaR."""
    r = returns.dropna()
    threshold = np.quantile(r, 1.0 - alpha)
    return float(-r[r <= threshold].mean())


def cvar_parametric(returns: pd.Series, alpha: float = 0.99) -> float:
    """CVaR bajo normalidad: -mu + sigma * pdf(z) / (1 - alpha), con z el cuantil normal."""
    mu, sigma = returns.mean(), returns.std(ddof=1)
    z = stats.norm.ppf(1.0 - alpha)
    return float(-mu + sigma * stats.norm.pdf(z) / (1.0 - alpha))


def rolling_var(
    returns: pd.Series,
    window: int = 252,
    alpha: float = 0.99,
    method: str = "historical",
    lam: float = 0.94,
) -> pd.Series:
    """VaR pronosticado para cada día usando solo información hasta el día anterior.

    Métodos
    -------
    "historical" : cuantil empírico de los últimos `window` días.
    "parametric" : normal con media y desviación de los últimos `window` días.
    "ewma"       : normal con media cero y volatilidad EWMA (RiskMetrics, lambda = `lam`),
                   que reacciona rápido cuando la volatilidad cambia.
    "fhs"        : simulación histórica filtrada. Divide cada retorno entre la volatilidad EWMA
                   pronosticada para ese día, toma el cuantil empírico de esos retornos
                   estandarizados y lo escala por la volatilidad actual. Combina la reacción
                   rápida del EWMA con las colas reales del histórico.

    Devuelve una pérdida positiva por fecha; los primeros `window` días quedan en NaN.
    """
    r = returns.dropna()
    z = stats.norm.ppf(1.0 - alpha)
    if method == "historical":
        var = -r.rolling(window).quantile(1.0 - alpha)
    elif method == "parametric":
        var = -(r.rolling(window).mean() + z * r.rolling(window).std(ddof=1))
    elif method == "ewma":
        var = -z * volatility.ewma_volatility(r, lam)
        var.iloc[: window - 1] = np.nan  # mismo periodo de arranque que los otros métodos
    elif method == "fhs":
        sigma = volatility.ewma_volatility(r, lam)
        standardized = r / sigma.shift(1)  # cada retorno entre la volatilidad pronosticada el día anterior
        var = -standardized.rolling(window).quantile(1.0 - alpha) * sigma
    else:
        raise ValueError(f"Método desconocido: {method}")
    # shift(1): el VaR del día t se calcula con datos hasta t-1 (sin look-ahead)
    return var.shift(1)


def count_exceptions(returns: pd.Series, var_forecast: pd.Series) -> int:
    """Número de días en que la pérdida real superó el VaR pronosticado.

    `var_forecast` debe ser una pérdida positiva, alineada por fecha y calculada
    con información hasta t-1 (sin look-ahead).
    """
    aligned = pd.concat([returns, var_forecast], axis=1, join="inner").dropna()
    return int((-aligned.iloc[:, 0] > aligned.iloc[:, 1]).sum())


def kupiec_pof_test(n_obs: int, n_exceptions: int, alpha: float = 0.99) -> dict:
    """Prueba de cobertura incondicional de Kupiec (POF).

    H0: la tasa de excepciones es igual a 1 - alpha.

    Compara la verosimilitud de las excepciones observadas bajo la tasa esperada `p = 1 - alpha`
    contra la tasa observada `x / n`. El estadístico LR sigue una chi-cuadrada con 1 grado de libertad.
    Solo revisa CUÁNTAS excepciones hubo, no si llegaron agrupadas.
    """
    p, x, n = 1.0 - alpha, n_exceptions, n_obs
    rate = x / n

    def log_likelihood(q: float) -> float:
        # xlogy(0, 0) = 0, así no falla con cero excepciones o con todas
        return float(special.xlogy(n - x, 1.0 - q) + special.xlogy(x, q))

    lr = -2.0 * (log_likelihood(p) - log_likelihood(rate))
    p_value = float(stats.chi2.sf(lr, df=1))
    return {"LR": lr, "p_value": p_value, "reject_H0_5pct": p_value < 0.05}


def backtest_var(returns: pd.Series, var_forecast: pd.Series, alpha: float = 0.99) -> dict:
    """Resumen del backtesting de un pronóstico de VaR contra los retornos realizados."""
    aligned = pd.concat([returns, var_forecast], axis=1, join="inner").dropna()
    losses, var = -aligned.iloc[:, 0], aligned.iloc[:, 1]
    hits = losses > var
    n_obs, n_exc = len(aligned), int(hits.sum())
    kupiec = kupiec_pof_test(n_obs, n_exc, alpha)
    return {
        "Días": n_obs,
        "Excepciones": n_exc,
        "Esperadas": n_obs * (1.0 - alpha),
        "Tasa": n_exc / n_obs,
        "Kupiec LR": kupiec["LR"],
        "p-value": kupiec["p_value"],
        "Rechaza H0 (5%)": kupiec["reject_H0_5pct"],
        "VaR medio": float(var.mean()),
        "Pérdida media en excepción": float(losses[hits].mean()) if n_exc else np.nan,
    }
