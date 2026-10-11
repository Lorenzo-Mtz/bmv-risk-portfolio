import numpy as np
import pandas as pd

from src import config, data


def _prices():
    idx = pd.date_range("2020-01-01", periods=100, freq="B")
    cols = ["AAA.MX", "BBB.MX", config.EXCLUDED[0], config.BENCHMARK]
    return pd.DataFrame(np.linspace(10, 20, 100)[:, None] * np.ones(len(cols)), index=idx, columns=cols)


def test_clean_prices_drops_excluded_by_default():
    out = data.clean_prices(_prices())
    assert config.EXCLUDED[0] not in out.columns
    assert list(out.columns) == ["AAA.MX", "BBB.MX", config.BENCHMARK]


def test_clean_prices_keeps_all_with_empty_exclude():
    out = data.clean_prices(_prices(), exclude=[])
    assert config.EXCLUDED[0] in out.columns


def test_clean_prices_drops_columns_with_too_many_missing():
    prices = _prices()
    prices.iloc[:10, 0] = np.nan  # 10% de faltantes > umbral de 5%
    out = data.clean_prices(prices, exclude=[])
    assert "AAA.MX" not in out.columns
    assert len(out) == 100


def test_split_benchmark_separates_benchmark_and_price_index():
    rets = _prices().assign(**{config.PRICE_INDEX: 1.0}).pct_change().dropna()
    stocks, bench = data.split_benchmark(rets)
    assert config.BENCHMARK not in stocks.columns
    assert config.PRICE_INDEX not in stocks.columns
    assert bench.name == config.BENCHMARK
