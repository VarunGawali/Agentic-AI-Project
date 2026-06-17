"""
Data layer: load crude oil price history.

Primary source: EIA open data API (free, public domain). Daily WTI (Cushing)
spot prices, series RWTC; Brent is RBRTE.
  Docs: https://www.eia.gov/opendata/  (register for a free API key)

If no API key is supplied (or the network is unavailable), this falls back to a
realistic SYNTHETIC WTI series so the rest of the pipeline can be built and
tested today. The synthetic series is clearly flagged so it is never mistaken
for real data.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd

from hedging_assistant.contracts import PriceHistory, ExposureBook, RiskAppetite

EIA_BASE = "https://api.eia.gov/v2/petroleum/pri/spt/data/"
SERIES = {"WTI": "RWTC", "BRENT": "RBRTE"}


def load_price_history(
    symbol: str = "WTI",
    api_key: str | None = None,
    start: str = "2010-01-01",
) -> PriceHistory:
    """
    Returns a PriceHistory. Uses EIA if an API key is available, else synthetic.
    Set the key via arg or the EIA_API_KEY environment variable.
    """
    api_key = api_key or os.environ.get("EIA_API_KEY")
    if api_key:
        try:
            return _load_from_eia(symbol, api_key, start)
        except Exception as e:  # pragma: no cover - network dependent
            print(f"[data] EIA fetch failed ({e}); using synthetic fallback.")
    else:
        print("[data] No EIA_API_KEY found; using synthetic fallback.")
    return _synthetic_history(symbol, start)


def _load_from_eia(symbol: str, api_key: str, start: str) -> PriceHistory:  # pragma: no cover
    import requests
    series_id = SERIES[symbol.upper()]
    params = {
        "api_key": api_key,
        "frequency": "daily",
        "data[0]": "value",
        "facets[series][]": series_id,
        "start": start,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": 5000,
    }
    r = requests.get(EIA_BASE, params=params, timeout=30)
    r.raise_for_status()
    rows = r.json()["response"]["data"]
    df = pd.DataFrame(rows)
    dates = pd.to_datetime(df["period"]).values
    prices = df["value"].astype(float).values
    return PriceHistory(dates=dates, prices=prices, symbol=symbol.upper())


def _synthetic_history(symbol: str, start: str) -> PriceHistory:
    """
    Realistic-looking WTI: mean-reverting-ish level with volatility clustering
    and occasional jumps, so downstream models have something with structure to
    chew on. NOT real data -- flagged in the symbol.
    """
    rng = np.random.default_rng(42)
    n = 3000
    dates = pd.bdate_range(start=start, periods=n).values
    mu_level = 75.0
    price = 60.0
    prices = []
    vol = 0.02
    for _ in range(n):
        # volatility clustering: vol drifts and spikes
        vol = 0.9 * vol + 0.1 * abs(rng.normal(0, 0.02)) + 0.001
        shock = rng.normal(0, vol)
        # mild mean reversion toward mu_level
        drift = 0.002 * (mu_level - price) / mu_level
        # occasional geopolitical jump
        jump = rng.normal(0, 0.08) if rng.random() < 0.01 else 0.0
        price *= np.exp(drift + shock + jump)
        price = max(price, 5.0)
        prices.append(price)
    return PriceHistory(
        dates=dates,
        prices=np.array(prices),
        symbol=f"{symbol.upper()}_SYNTHETIC",
    )


def make_exposure_book(barrels_per_period: float = 100_000,
                       horizon: int = 6,
                       period_label: str = "month") -> ExposureBook:
    """A simple constant exposure book: same volume needed each period."""
    return ExposureBook(
        volumes=np.full(horizon, float(barrels_per_period)),
        period_label=period_label,
    )


def default_risk_appetite() -> RiskAppetite:
    return RiskAppetite()


if __name__ == "__main__":
    hist = load_price_history()
    print(f"Loaded {len(hist)} prices for {hist.symbol}")
    print(f"  range: {hist.prices.min():.1f} - {hist.prices.max():.1f} USD/bbl")
    print(f"  last:  {hist.prices[-1]:.1f}")
    book = make_exposure_book()
    print(f"Exposure: {book.volumes[0]:,.0f} bbl/{book.period_label} "
          f"x {book.horizon} {book.period_label}s")
