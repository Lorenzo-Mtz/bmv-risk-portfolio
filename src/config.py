"""Configuración global del proyecto.

Fija el universo y las fechas desde la semana 1 y no los cambies después:
cualquier cambio invalida las comparaciones entre bloques.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# Benchmark: iShares NAFTRAC, el ETF que replica el S&P/BMV IPC. Se usa su precio ajustado,
# que incluye dividendos igual que el de las emisoras (comparación en retorno total).
# OJO: en Yahoo Finance le faltan ~7 días por año (feriados de EE. UU. en los que la BMV
# sí operó); `clean_prices` los rellena hacia adelante (ver notebooks/01_eda.ipynb, sección 4).
BENCHMARK = "NAFTRAC.MX"

# Índice de precios S&P/BMV IPC (sin dividendos). Solo como referencia: no es comparable
# con retornos ajustados y `split_benchmark` lo separa de las emisoras.
PRICE_INDEX = "^MXX"

# Universo: las 35 emisoras de la muestra del S&P/BMV IPC vigente desde el
# 21-sep-2026 (sufijo .MX en Yahoo Finance). Fuente: cartera del iShares NAFTRAC
# al 07-oct-2026 y anuncios de cambio de muestra de S&P DJI (mar-2026 y sep-2026).
# OJO: es la composición ACTUAL aplicada a toda la ventana de estudio, no la
# histórica; eso introduce sesgo de supervivencia (ver Limitaciones en el README).
# IMPORTANTE: los símbolos de yfinance pueden cambiar o fallar. `download_prices`
# reporta los que no se pudieron descargar; ajusta esta lista según el resultado.
TICKERS = [
    "AC.MX",          # Arca Continental
    "ALPEKA.MX",      # Alpek
    "ALSEA.MX",       # Alsea
    "AMXB.MX",        # América Móvil
    "ASURB.MX",       # ASUR
    "BBAJIOO.MX",     # BanBajío (cotiza desde jun-2017)
    "BIMBOA.MX",      # Bimbo
    "BOLSAA.MX",      # Bolsa Mexicana de Valores
    "CEMEXCPO.MX",    # Cemex
    "CHDRAUIB.MX",    # Chedraui
    "FEMSAUBD.MX",    # FEMSA
    "GAPB.MX",        # Grupo Aeroportuario del Pacífico
    "GCARSOA1.MX",    # Grupo Carso
    "GCC.MX",         # GCC
    "GENTERA.MX",     # Gentera
    "GFINBURO.MX",    # Inbursa
    "GFNORTEO.MX",    # Banorte
    "GMEXICOB.MX",    # Grupo México
    "GRUMAB.MX",      # Gruma
    "KIMBERA.MX",     # Kimberly-Clark de México
    "KOFUBL.MX",      # Coca-Cola FEMSA (serie UBL desde abr-2019)
    "LABB.MX",        # Genomma Lab
    "LACOMERUBC.MX",  # La Comer (cotiza desde ene-2016)
    "LIVEPOLC-1.MX",  # Liverpool
    "MEGACPO.MX",     # Megacable
    "OMAB.MX",        # OMA
    "ORBIA.MX",       # Orbia
    "PE&OLES.MX",     # Peñoles
    "PINFRA.MX",      # Promotora y Operadora de Infraestructura
    "Q.MX",           # Quálitas (cotiza desde jul-2015)
    "RA.MX",          # Regional (Banregio)
    "SIGMAFA.MX",     # Sigma Foods (antes Alfa)
    "VESTA.MX",       # Vesta
    "VOLARA.MX",      # Volaris
    "WALMEX.MX",      # Walmart de México
]

# Emisoras excluidas por calidad de datos. `clean_prices` las descarta por defecto,
# así que la exclusión aplica igual en todos los bloques (ver notebooks/01_eda.ipynb, sección 3.1).
EXCLUDED = [
    "AMXB.MX",        # 41% de días con retorno cero en Yahoo Finance (precios repetidos hasta 2023)
]

# Ventana de estudio
START = "2015-01-01"
END = "2026-10-09"  # None = hasta hoy

# Parámetros de trading / cálculo
TRADING_DAYS = 252
TRANSACTION_COST_BPS = 10  # costo por operación en puntos base (ajustable)
RISK_FREE_ANNUAL = 0.0     # reemplazar por CETES 28d (Banxico) en bloques posteriores
