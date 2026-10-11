"""Descarga, limpieza y caché de precios."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def download_prices(
    tickers: list[str] | None = None,
    benchmark: str | None = config.BENCHMARK,
    start: str = config.START,
    end: str | None = config.END,
    cache_name: str = "prices.parquet",
    force: bool = False,
) -> pd.DataFrame:
    """Descarga precios ajustados de cierre con yfinance y los guarda en caché.

    Devuelve un DataFrame (fechas x tickers). El benchmark, si se pide, y el índice de
    precios de referencia (`config.PRICE_INDEX`) van incluidos como columnas. Los tickers
    que no se pudieron descargar se reportan con una advertencia en lugar de fallar en silencio.
    """
    tickers = list(tickers if tickers is not None else config.TICKERS)
    cache_path: Path = config.DATA_DIR / cache_name

    if cache_path.exists() and not force:
        return pd.read_parquet(cache_path)

    import yfinance as yf  # import local: no es necesario si hay caché

    symbols = tickers + [s for s in (benchmark, config.PRICE_INDEX) if s]
    raw = yf.download(
        symbols,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if raw is None or raw.empty:
        raise RuntimeError("yfinance no devolvió datos. Revisa tu conexión o los símbolos.")

    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    prices = prices.dropna(axis=1, how="all")

    missing = sorted(set(symbols) - set(prices.columns))
    if missing:
        warnings.warn(f"No se pudieron descargar: {missing}. Ajusta config.TICKERS.")

    # Calendario de la BMV: solo las fechas en que cotizó al menos una emisora (Yahoo publica
    # el NAFTRAC con calendario de EE. UU. y trae filas en feriados mexicanos).
    prices = prices.dropna(how="all", subset=[t for t in tickers if t in prices.columns])

    prices = prices.sort_index()
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(cache_path)
    return prices


def clean_prices(
    prices: pd.DataFrame,
    max_missing_frac: float = 0.05,
    max_ffill_days: int = 3,
    exclude: list[str] | None = None,
) -> pd.DataFrame:
    """Limpieza básica sin introducir look-ahead bias.

    - Descarta las emisoras de `exclude` (por defecto `config.EXCLUDED`; usa `[]` para no excluir).
    - Elimina columnas con demasiados faltantes.
    - Rellena huecos cortos hacia adelante (nunca hacia atrás).
    - Elimina filas iniciales sin datos completos.
    """
    exclude = list(exclude if exclude is not None else config.EXCLUDED)
    prices = prices.drop(columns=exclude, errors="ignore")
    keep = prices.columns[prices.isna().mean() <= max_missing_frac]
    out = prices[keep].ffill(limit=max_ffill_days)
    return out.dropna(how="any")


def compute_returns(prices: pd.DataFrame, log: bool = True) -> pd.DataFrame:
    """Retornos diarios (logarítmicos por defecto)."""
    rets = np.log(prices / prices.shift(1)) if log else prices.pct_change()
    return rets.dropna(how="all")


def split_benchmark(
    returns: pd.DataFrame, benchmark: str = config.BENCHMARK
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Separa los retornos de las emisoras de los del benchmark.

    El índice de precios de referencia (`config.PRICE_INDEX`) tampoco es una emisora
    y se quita de las emisoras; sigue disponible en `returns`.
    """
    stocks = returns.drop(columns=[benchmark, config.PRICE_INDEX], errors="ignore")
    return stocks, (returns[benchmark] if benchmark in returns.columns else None)
