"""Pronóstico de volatilidad: ventana fija, EWMA, GARCH y XGBoost (Bloque 4).

Convención: las funciones `*_forecast` devuelven la volatilidad diaria (sin anualizar)
pronosticada para cada fecha usando solo información hasta el día anterior.
"""
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


# ---------- evaluación ----------

def qlike_losses(realized_var: pd.Series, forecast_var: pd.Series) -> pd.Series:
    """Pérdida QLIKE de cada día: r/f - log(r/f) - 1. Vale 0 si el pronóstico es exacto."""
    df = pd.concat([realized_var, forecast_var], axis=1, join="inner").dropna()
    ratio = df.iloc[:, 0] / df.iloc[:, 1]
    return ratio - np.log(ratio) - 1.0


def qlike(realized_var: pd.Series, forecast_var: pd.Series) -> float:
    """Pérdida QLIKE (robusta para evaluar pronósticos de varianza). Menor es mejor."""
    return float(qlike_losses(realized_var, forecast_var).mean())


def mse_losses(realized_var: pd.Series, forecast_var: pd.Series) -> pd.Series:
    """Error cuadrático de cada día entre la varianza realizada y la pronosticada."""
    df = pd.concat([realized_var, forecast_var], axis=1, join="inner").dropna()
    return (df.iloc[:, 0] - df.iloc[:, 1]) ** 2


def mse(realized_var: pd.Series, forecast_var: pd.Series) -> float:
    """Error cuadrático medio. Menor es mejor; muy sensible a los días extremos."""
    return float(mse_losses(realized_var, forecast_var).mean())


def loss_difference_tstat(losses_a: pd.Series, losses_b: pd.Series) -> float:
    """Estadístico t de la diferencia media de pérdidas (prueba de Diebold-Mariano a un paso).

    Negativo: el modelo A pierde menos que B. |t| > 2 indica una diferencia distinguible del azar.
    """
    d = (losses_a - losses_b).dropna()
    return float(d.mean() / d.std(ddof=1) * np.sqrt(len(d)))


# ---------- pronósticos ----------

def rolling_forecast(returns: pd.Series, window: int = 252) -> pd.Series:
    """Volatilidad de los últimos `window` días, usada como pronóstico del día siguiente."""
    return realized_volatility(returns, window).shift(1)


def ewma_forecast(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    """Volatilidad EWMA calculada hasta el día anterior."""
    return ewma_volatility(returns, lam).shift(1)


def garch_forecast(returns: pd.Series, min_train: int = 1000, refit_every: int = 21) -> pd.Series:
    """Pronóstico a un día con GARCH(1,1) walk-forward.

    Cada `refit_every` días se reestiman los parámetros con todos los datos disponibles hasta
    ese momento (ventana expansiva); nunca se ajusta con toda la muestra. Entre reestimaciones
    los parámetros quedan fijos y la varianza se actualiza cada día con la recursión

        sigma2[t] = omega + alpha * (r[t-1] - mu)^2 + beta * sigma2[t-1]

    Los primeros `min_train` días solo sirven para entrenar y quedan en NaN.
    """
    from arch import arch_model  # import local: no se necesita para el resto del módulo

    r = returns.dropna()
    values = r.to_numpy() * 100.0  # en porcentaje: el optimizador de `arch` es más estable
    out = np.full(len(r), np.nan)

    for start in range(min_train, len(r), refit_every):
        fit = arch_model(values[:start], mean="Constant", vol="GARCH", p=1, q=1).fit(disp="off", show_warning=False)
        mu, omega, alpha, beta = (fit.params[k] for k in ("mu", "omega", "alpha[1]", "beta[1]"))
        sigma2 = fit.conditional_volatility[-1] ** 2  # varianza del último día de entrenamiento

        for t in range(start, min(start + refit_every, len(r))):
            sigma2 = omega + alpha * (values[t - 1] - mu) ** 2 + beta * sigma2
            out[t] = sigma2

    return pd.Series(np.sqrt(out) / 100.0, index=r.index)


def make_features(returns: pd.Series, stock_returns: pd.DataFrame | None = None) -> pd.DataFrame:
    """Variables conocidas al cierre de cada día para pronosticar la varianza del día siguiente.

    Todo se calcula con datos hasta ese día inclusive (rezagos y ventanas hacia atrás).
    `stock_returns` (fechas x emisoras) agrega información que la serie del portafolio no tiene.
    """
    r = returns * 100.0
    sq = r**2
    f = pd.DataFrame(index=r.index)
    for lag in range(5):
        f[f"r2_rezago_{lag}"] = sq.shift(lag)
    f["caida2"] = r.clip(upper=0.0) ** 2             # solo días de pérdida: efecto apalancamiento
    for window in (5, 21, 63):
        f[f"var_{window}d"] = sq.rolling(window).mean()
    f["var_ewma"] = sq.ewm(alpha=0.06, adjust=False).mean()
    if stock_returns is not None:
        s = stock_returns.loc[r.index] * 100.0
        f["var_emisoras"] = (s**2).mean(axis=1)                    # varianza promedio de las emisoras
        f["var_emisoras_21d"] = f["var_emisoras"].rolling(21).mean()
        f["dispersion"] = s.std(axis=1)                            # qué tan distinto se movieron entre sí
    return f


XGB_PARAMS = {
    "objective": "reg:gamma",   # su pérdida es QLIKE y garantiza pronósticos positivos
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "random_state": 0,
}


def xgboost_forecast(
    returns: pd.Series,
    features: pd.DataFrame | None = None,
    min_train: int = 1000,
    refit_every: int = 21,
    **params,
) -> tuple[pd.Series, pd.Series]:
    """Pronóstico a un día con XGBoost walk-forward.

    Cada fila `t` tiene las variables al cierre de `t` y como objetivo el retorno al cuadrado
    de `t+1`. Cada `refit_every` días se reentrena con ventana expansiva, usando solo filas
    cuyo objetivo ya se conoce. Los hiperparámetros (`XGB_PARAMS`) son fijos: no se buscan.

    Devuelve (volatilidad pronosticada, importancia promedio de cada variable).
    """
    import xgboost as xgb  # import local: no se necesita para el resto del módulo

    r = returns.dropna()
    X = (features if features is not None else make_features(r)).loc[r.index]
    target = ((r * 100.0) ** 2).shift(-1).clip(lower=1e-6)  # la pérdida gamma exige objetivo > 0
    complete = X.notna().all(axis=1).to_numpy()

    out = np.full(len(r), np.nan)
    importances = []
    for start in range(min_train, len(r), refit_every):
        # Filas 0..start-2: el objetivo de la fila start-2 es el día start-1, el último ya observado
        train = np.flatnonzero(complete[: start - 1])
        model = xgb.XGBRegressor(**{**XGB_PARAMS, **params}).fit(X.iloc[train], target.iloc[train])

        # El pronóstico de cada día usa las variables del día anterior
        stop = min(start + refit_every, len(r))
        out[start:stop] = model.predict(X.iloc[start - 1 : stop - 1])
        importances.append(model.feature_importances_)

    forecast = pd.Series(np.sqrt(out) / 100.0, index=r.index)
    importance = pd.Series(np.mean(importances, axis=0), index=X.columns).sort_values(ascending=False)
    return forecast, importance
