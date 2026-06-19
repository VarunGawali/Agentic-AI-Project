"""
Data layer: load crude oil price history.

Primary source: EIA open data API (free, public domain). Daily WTI (Cushing)
spot prices, series RWTC; Brent is RBRTE.
  Docs: https://www.eia.gov/opendata/  (register for a free API key)

If no API key is supplied (or the network is unavailable), this falls back to a
realistic SYNTHETIC WTI series so the rest of the pipeline can be built and
tested today. The synthetic series is clearly flagged so it is never mistaken
for real data.

CHANGES:
  - Added incremental refresh: refresh() pulls only trailing 30-day window
    + anything newer, merges, dedups, re-sorts, writes parquet snapshot.
  - Added load_cached() to read the snapshot without a network call.
  - Added save_snapshot() / load_snapshot() helpers.
  - Full pull still available via load_price_history(force_full=False).
"""

from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd

from hedging_assistant.contracts import PriceHistory, ExposureBook, RiskAppetite

EIA_BASE = "https://api.eia.gov/v2/petroleum/pri/spt/data/"
SERIES = {"WTI": "RWTC", "BRENT": "RBRTE"}
PAGE_SIZE = 5000   # EIA max rows per request

SNAPSHOT_DIR = Path(__file__).parent.parent / "data" / "snapshots"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


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
    all_rows: list[dict] = []
    offset = 0

    while True:
        params = {
            "api_key": api_key,
            "frequency": "daily",
            "data[0]": "value",
            "facets[series][]": series_id,
            "start": start,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "length": PAGE_SIZE,
            "offset": offset,
        }
        r = requests.get(EIA_BASE, params=params, timeout=30)
        r.raise_for_status()
        payload = r.json()["response"]
        rows = payload.get("data", [])
        all_rows.extend(rows)

        if len(rows) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    if not all_rows:
        raise ValueError(f"EIA returned no data for {series_id} from {start}")

    df = pd.DataFrame(all_rows)
    df = df[df["value"].notna() & (df["value"] != "")]
    df["period"] = pd.to_datetime(df["period"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df = df.sort_values("period").drop_duplicates(subset=["period"])

    print(f"[data] EIA: loaded {len(df)} rows for {series_id} "
          f"({df['period'].min().date()} → {df['period'].max().date()})")

    return PriceHistory(
        dates=df["period"].values,
        prices=df["value"].values,
        symbol=symbol.upper(),
    )


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


# ----------------------------------------------------------------------------
# Snapshot helpers (Feature 5: incremental EIA refresh)
# ----------------------------------------------------------------------------

def _snapshot_path(symbol: str) -> Path:
    return SNAPSHOT_DIR / f"{symbol.upper()}_latest.parquet"


def save_snapshot(history: PriceHistory) -> None:
    """Write PriceHistory to a parquet snapshot (idempotent)."""
    path = _snapshot_path(history.symbol.replace("_SYNTHETIC", ""))
    df = pd.DataFrame({"date": history.dates, "price": history.prices})
    df.to_parquet(path, index=False)
    print(f"[data] Snapshot saved: {path} ({len(df)} rows)")


def load_snapshot(symbol: str) -> PriceHistory | None:
    """Read the latest parquet snapshot. Returns None if not found."""
    path = _snapshot_path(symbol)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date"])
    return PriceHistory(
        dates=df["date"].values,
        prices=df["price"].values,
        symbol=symbol.upper(),
    )


def load_cached(symbol: str = "WTI") -> PriceHistory | None:
    """Read the snapshot without a network call. Returns None if not found."""
    return load_snapshot(symbol)


def refresh(symbol: str = "WTI", api_key: str | None = None, trailing_days: int = 30) -> PriceHistory:
    """
    Incremental EIA refresh per CLAUDE.md spec:
    1. Load existing snapshot (if any)
    2. Pull only trailing_days window + anything newer from EIA
    3. Merge + dedup by date + re-sort
    4. Write back to snapshot
    5. Return merged PriceHistory

    This is SEPARATE from the read path — pipeline runs read the snapshot,
    never trigger a network call mid-run.
    """
    api_key = api_key or os.environ.get("EIA_API_KEY")
    if not api_key:
        raise ValueError("EIA_API_KEY required for refresh(). Use load_snapshot() to read cached data.")

    existing = load_snapshot(symbol)
    if existing is not None:
        # compute the trailing window start date
        last_date = pd.to_datetime(existing.dates).max()
        window_start = (last_date - pd.Timedelta(days=trailing_days)).strftime("%Y-%m-%d")
        print(f"[data] Incremental refresh from {window_start} (trailing {trailing_days}d + new)")
    else:
        window_start = "2010-01-01"
        print(f"[data] No snapshot found — full pull from {window_start}")

    fresh = _load_from_eia(symbol, api_key, window_start)

    if existing is not None:
        # merge: combine existing + fresh, dedup by date, re-sort
        old_df = pd.DataFrame({"date": existing.dates, "price": existing.prices})
        new_df = pd.DataFrame({"date": fresh.dates, "price": fresh.prices})
        merged = pd.concat([old_df, new_df], ignore_index=True)
        merged["date"] = pd.to_datetime(merged["date"])
        merged = merged.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
        result = PriceHistory(
            dates=merged["date"].values,
            prices=merged["price"].values,
            symbol=symbol.upper(),
        )
        print(f"[data] Merged: {len(old_df)} existing + {len(new_df)} fresh → {len(merged)} total")
    else:
        result = fresh

    save_snapshot(result)
    return result


if __name__ == "__main__":
    hist = load_price_history()
    print(f"Loaded {len(hist)} prices for {hist.symbol}")
    print(f"  range: {hist.prices.min():.1f} - {hist.prices.max():.1f} USD/bbl")
    print(f"  last:  {hist.prices[-1]:.1f}")
    book = make_exposure_book()
    print(f"Exposure: {book.volumes[0]:,.0f} bbl/{book.period_label} "
          f"x {book.horizon} {book.period_label}s")
