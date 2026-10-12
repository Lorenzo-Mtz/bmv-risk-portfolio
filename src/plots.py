"""Estilo y gráficas compartidas por los notebooks."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import ticker

from . import metrics

# Un color fijo por serie en todos los notebooks; el benchmark va en gris neutro.
COLORS = {
    "Igual ponderado": "#2a78d6",
    "Mínima varianza": "#eb6834",
    "HRP": "#1baf7a",
    "Máximo Sharpe": "#eda100",
    "NAFTRAC": "#52514e",
    # Métodos de VaR (notebook 03)
    "Histórico": "#2a78d6",
    "Normal": "#eb6834",
    "EWMA": "#1baf7a",
    "Histórico filtrado": "#eda100",
    # Modelos de volatilidad (notebook 04); EWMA conserva su color
    "Ventana fija": "#2a78d6",
    "GARCH": "#eb6834",
    "XGBoost": "#6250d6",
    "Realizada": "#c3c2b7",
}
INK = "#0b0b0b"
MUTED = "#c3c2b7"


def set_style() -> None:
    plt.rcParams.update({
        "figure.figsize": (10, 4),
        "axes.grid": True,
        "grid.color": "#e6e5e0",
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#c3c2b7",
        "lines.linewidth": 2,
        "legend.frameon": False,
    })


def line_chart(frame: pd.DataFrame, title: str, ax: plt.Axes | None = None) -> plt.Axes:
    """Una línea por columna, con el color fijo de cada serie."""
    ax = ax or plt.subplots(figsize=(11, 4.5))[1]
    for name in frame.columns:
        frame[name].plot(ax=ax, color=COLORS.get(name), label=name)
    ax.set_title(title, loc="left")
    ax.set_xlabel("")
    ax.legend(ncol=len(frame.columns), loc="upper left", bbox_to_anchor=(0, -0.17))
    return ax


def growth_chart(returns: pd.DataFrame, title: str = "Crecimiento de $1", ax: plt.Axes | None = None) -> plt.Axes:
    """Riqueza acumulada de cada columna de retornos simples (escala logarítmica)."""
    ax = line_chart((1.0 + returns).cumprod(), title, ax)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(ticker.LogLocator(subs=(1, 1.5, 2, 3, 5, 7)))
    ax.yaxis.set_major_formatter(lambda v, _: f"${v:g}")
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    return ax


def drawdown_chart(returns: pd.DataFrame, title: str = "Drawdown", ax: plt.Axes | None = None) -> plt.Axes:
    """Caída desde el máximo previo de cada columna de retornos simples."""
    ax = line_chart(returns.apply(metrics.drawdown_series), title, ax)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    return ax


def var_chart(returns: pd.Series, forecasts: pd.DataFrame, title: str) -> plt.Figure:
    """Un panel por método: retornos diarios, la línea de -VaR y las excepciones marcadas."""
    fig, axes = plt.subplots(len(forecasts.columns), 1, figsize=(11, 2.6 * len(forecasts.columns)), sharex=True, sharey=True)
    r = returns.loc[forecasts.dropna().index]
    for ax, name in zip(axes, forecasts.columns):
        var = forecasts[name].loc[r.index]
        hits = -r > var
        ax.plot(r.index, r, color=MUTED, lw=0.6, label="Retorno diario")
        ax.plot(var.index, -var, color=COLORS.get(name), label=f"-VaR ({name})")
        ax.scatter(r.index[hits], r[hits], s=14, color=INK, zorder=3, label=f"Excepciones: {int(hits.sum())}")
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.legend(ncol=3, loc="lower right", fontsize=9)
    axes[0].set_title(title, loc="left")
    fig.tight_layout()
    return fig


def barh_chart(values: pd.Series, title: str, fmt: str = "{:.0%}") -> plt.Axes:
    """Barras horizontales ordenadas de mayor a menor, con el valor en la punta."""
    values = values.sort_values()
    _, ax = plt.subplots(figsize=(8, 0.34 * len(values) + 0.8))
    bars = ax.barh(values.index, values, height=0.55, color=COLORS["Igual ponderado"])
    ax.bar_label(bars, labels=[fmt.format(v) for v in values], padding=4, color=INK, fontsize=9)
    ax.set_title(title, loc="left")
    ax.grid(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])
    return ax
