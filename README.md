# BMV Risk & Portfolio Lab

**ES** — Construcción de portafolios y medición de riesgo de mercado para acciones de la Bolsa Mexicana de Valores (BMV), en Python, con backtesting *walk-forward* y pruebas estadísticas.
**EN** — Portfolio construction and market-risk measurement for Mexican Stock Exchange (BMV) equities, in Python, with walk-forward backtesting and statistical tests.

> **Estado / Status:** bloques 1–3 completos (datos, portafolios, riesgo de mercado). Pronóstico de volatilidad en curso. / Blocks 1–3 complete (data, portfolios, market risk). Volatility forecasting in progress.

---

## Qué responde este proyecto

1. ¿Alguna forma de repartir el capital (igual ponderado, mínima varianza, máximo Sharpe, HRP) mejora de forma creíble a las alternativas simples, una vez descontados los costos?
2. ¿Los modelos habituales de VaR cumplen lo que prometen cuando se evalúan fuera de muestra?

La respuesta corta: **mínima varianza redujo el riesgo; máximo Sharpe fue la peor estrategia; ninguna le gana al índice de forma estadísticamente distinguible; y los VaR de ventana fija fallan al 99%.**

## Resultados principales

### 1. Portafolios

Backtest *walk-forward* del 19-jul-2016 al 8-oct-2026 (10.2 años): ventana de estimación de 252 días, rebalanceo mensual, 10 pb de costo por cada peso operado, retornos netos de costos. Detalle en [`notebooks/02_portafolios.ipynb`](notebooks/02_portafolios.ipynb).

| Estrategia | Retorno anual | Volatilidad | Sharpe | Max drawdown | Rotación anual |
|---|---|---|---|---|---|
| Mínima varianza | 9.3% | 13.4% | 0.73 | −27.8% | 136% |
| HRP | 7.8% | 13.9% | 0.61 | −32.4% | 88% |
| Igual ponderado | 8.2% | 15.6% | 0.59 | −35.5% | 32% |
| Máximo Sharpe | 9.4% | 20.0% | 0.55 | −39.9% | 300% |
| NAFTRAC (benchmark) | 5.7% | 16.0% | 0.43 | −32.4% | – |

- **Mínima varianza cumplió su objetivo:** la menor volatilidad y la menor caída, con el mejor Sharpe. Lo hace concentrándose en unas 10 emisoras efectivas, con las mayores posiciones en consumo básico.
- **Máximo Sharpe, la estrategia "óptima" en teoría, fue la peor fuera de muestra.** Depende de estimar retornos esperados con un año de datos, que son casi puro ruido: termina con 5 emisoras efectivas, 300% de rotación y pierde contra igual ponderado, que no estima nada.
- **No se puede afirmar que le ganen al índice.** El exceso sobre el NAFTRAC no es estadísticamente distinguible de cero en ninguna estrategia (el estadístico *t* más alto es 1.10), y además el universo tiene sesgo de supervivencia.

### 2. Riesgo de mercado

VaR al 99% a un día sobre el portafolio igual ponderado, pronosticado cada día solo con información previa y evaluado del 20-jul-2017 al 8-oct-2026 (2,319 días; 23 excepciones esperadas). Detalle en [`notebooks/03_var.ipynb`](notebooks/03_var.ipynb).

| Método | Excepciones | Kupiec (*p-value*) | Resultado | Excepciones en el desplome de 2020 |
|---|---|---|---|---|
| Histórico, 252 días | 38 | 0.005 | Rechazado | 9 |
| Normal, 252 días | 38 | 0.005 | Rechazado | 11 |
| EWMA normal (λ = 0.94) | 44 | < 0.001 | Rechazado | 3 |
| Histórico filtrado | 33 | 0.054 | No rechazado | 2 |

- **Al 95% los cuatro métodos pasan; al 99% los tres básicos fallan**, y cada uno por una razón distinta: los de ventana fija reaccionan tarde y acumulan excepciones en las crisis; EWMA reacciona rápido pero supone normalidad y se queda corto en la cola.
- **Suponer normalidad cuesta donde más importa:** el CVaR histórico al 99% es 1.54 veces el que predice la normal (3.99% contra 2.59%).
- **El histórico filtrado**, que combina volatilidad EWMA con el cuantil empírico, es el único que no se rechaza en los cinco portafolios evaluados. Se añadió después de ver fallar a los otros tres, así que su resultado debe confirmarse en otro periodo.

### 3. Datos

- **América Móvil (AMXB) se excluyó:** el 41% de sus días tiene retorno exactamente cero en Yahoo Finance, lo que distorsiona su correlación con el resto.
- **El benchmark es el ETF NAFTRAC y no el índice `^MXX`:** el índice es de precios y no incluye dividendos (3.3% anual contra 5.6% del NAFTRAC en la misma ventana), lo que le daría más de 2 puntos de ventaja por año a cualquier portafolio.
- Los retornos tienen colas pesadas y agrupamiento de volatilidad en las 31 emisoras; se rechaza normalidad en todas. Detalle en [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb).

## Universo / Universe

**ES.** Las **35 emisoras de la muestra del S&P/BMV IPC** vigente desde el 21 de septiembre de 2026, fijas en `src/config.py` (fuente: cartera del iShares NAFTRAC al 7 de octubre de 2026 y anuncios de cambio de muestra de S&P Dow Jones Indices). Con la ventana que inicia en 2015 y el filtro de 5 % de faltantes, tres quedan fuera por historia insuficiente (Coca-Cola FEMSA UBL, BanBajío y La Comer) y una por calidad de datos (América Móvil). El análisis usa las **31 restantes**, desde julio de 2015.

**EN.** The **35 constituents of the S&P/BMV IPC** as of the 21 September 2026 rebalance, fixed in `src/config.py`. Three names drop out for insufficient history and one (América Móvil) for data quality, leaving **31 stocks** from July 2015. The benchmark is the iShares NAFTRAC ETF (total return), not the `^MXX` price index.

## Metodología / Methodology

- **Sin *look-ahead bias*:** toda decisión en `t` usa solo información disponible hasta `t−1`. Hay pruebas unitarias que alteran datos futuros y verifican que pesos y pronósticos pasados no cambian.
- **Validación *walk-forward*;** nunca particiones aleatorias en series de tiempo.
- **Costos de transacción y deriva de pesos** entre rebalanceos; la rotación se mide contra los pesos ya derivados.
- **Parámetros fijados de antemano** (ventana de 252 días, rebalanceo mensual, λ = 0.94). No se probaron variantes para escoger la mejor.
- **Los resultados negativos o modestos se reportan tal cual.**
- **51 pruebas unitarias** (`pytest`) con casos de solución conocida.

## Limitaciones / Limitations

- **Sesgo de supervivencia:** el universo es la muestra del IPC de septiembre de 2026 aplicada a toda la ventana desde 2015, no la composición histórica. Las emisoras se eligieron sabiendo que hoy son grandes y líquidas, y quedan fuera las que salieron del índice (p. ej. Elektra, Televisa, Cuervo). Esto favorece a todos los portafolios frente al NAFTRAC; los resultados son una cota optimista.
- **Sharpe con tasa libre de riesgo en cero:** sirve para comparar estrategias entre sí, no como exceso sobre CETES.
- **Calidad de datos:** `yfinance` es adecuado para aprendizaje, no una fuente de grado profesional. Al NAFTRAC le faltan unos 7 días por año (feriados de EE. UU. en los que la BMV operó), que se rellenan con el precio anterior. Los precios de Sigma (antes Alfa) muestran caídas que coinciden con las escisiones de Nemak y Alpek y que no parecen estar ajustadas.
- **Costos simplificados:** un costo proporcional fijo no incluye impacto en precio.
- **Kupiec solo cuenta excepciones;** no revisa si llegan agrupadas.
- Proyecto educativo; **no es asesoría de inversión**.

## Estructura / Structure

```
bmv-risk-portfolio/
├── notebooks/
│   ├── 01_eda.ipynb           # datos, limpieza y hechos estilizados
│   ├── 02_portafolios.ipynb   # estrategias y backtest walk-forward
│   └── 03_var.ipynb           # VaR, CVaR y prueba de Kupiec
├── src/
│   ├── config.py              # universo, benchmark, fechas y parámetros
│   ├── data.py                # descarga, caché y limpieza de precios
│   ├── portfolios.py          # igual ponderado, mínima varianza, máximo Sharpe, HRP
│   ├── backtest.py            # backtest walk-forward con deriva y costos
│   ├── metrics.py             # retorno, volatilidad, Sharpe, Sortino, drawdown
│   ├── risk.py                # VaR, CVaR, VaR rodante y Kupiec
│   ├── volatility.py          # EWMA y pérdida QLIKE
│   └── plots.py               # estilo y gráficas compartidas
├── tests/                     # pruebas unitarias (pytest)
└── data/                      # caché en parquet (ignorado por git)
```

## Cómo reproducirlo / Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest                           # 51 pruebas
jupyter lab                      # ejecutar los notebooks en orden: 01 → 02 → 03
```

La primera ejecución del notebook 01 descarga los precios de Yahoo Finance y los guarda en `data/`.

## Trabajo futuro / Next steps

- Pronóstico de volatilidad: EWMA y GARCH como línea base contra XGBoost, con validación *walk-forward*.
- Prueba de independencia de Christoffersen para las excepciones del VaR.
- CETES a 28 días como tasa libre de riesgo.
- Covarianza con contracción de Ledoit-Wolf.
- Universo con la composición histórica del índice, para eliminar el sesgo de supervivencia.

## Autor / Author

Lorenzo Martínez Ruiz — Ingeniería en Transformación Digital y Negocios, Tec de Monterrey.

## Uso de IA / AI assistance

Este proyecto se desarrolló con asistencia de Claude (Anthropic). La mayor parte del código, las pruebas unitarias y la estructura de los notebooks se generaron con IA. El alcance, las decisiones metodológicas (universo, benchmark, exclusión de datos), la revisión del código y la interpretación de los resultados son del autor; las conclusiones de cada notebook fueron redactadas por el autor y revisadas con IA.
