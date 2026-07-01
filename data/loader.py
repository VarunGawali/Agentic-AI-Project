"""
Data ingestion: load WTI/Brent crude oil price history from EIA.

Cloud behavior:
    1. Download existing CSV from Azure Blob Storage if configured.
    2. Fetch only new rows from EIA after latest saved date.
    3. Append, deduplicate, sort.
    4. Upload updated CSV back to Azure Blob Storage.
    5. Return pandas DataFrame.

Local fallback:
    If Azure Blob env vars are not configured, use data/raw/*.csv locally.
"""

from __future__ import annotations

import logging
import os
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

try:
    from azure.storage.blob import BlobServiceClient
except ImportError:
    BlobServiceClient = None

try:
    from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
except ImportError:
    DefaultAzureCredential = None
    ManagedIdentityCredential = None


def _make_retry_session(
    total: int = 4,
    backoff_factor: float = 1.0,
    status_forcelist: tuple = (429, 500, 502, 503, 504),
) -> requests.Session:
    """
    Build a requests.Session with automatic retry on transient errors.
    """
    session = requests.Session()
    retry = Retry(
        total=total,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


load_dotenv()

EIA_BASE = "https://api.eia.gov/v2/petroleum/pri/spt/data/"

SERIES = {
    "WTI": "RWTC",
    "BRENT": "RBRTE",
}

# NYMEX WTI (Cushing) futures contracts 1-4 — the real forward curve, same EIA
# petroleum dataset / API key as spot, just the `fut` endpoint.
EIA_FUT_BASE = "https://api.eia.gov/v2/petroleum/pri/fut/data/"

FUTURES_SERIES = {
    "WTI": ["RCLC1", "RCLC2", "RCLC3", "RCLC4"],
}

DEFAULT_DATA_DIR = Path("data/raw")


def load_price_history(
    symbol: str = "WTI",
    start: str = "2010-01-01",
    data_dir: str | Path = DEFAULT_DATA_DIR,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Load crude price history.

    In cloud:
        Uses Azure Blob Storage when AZURE_STORAGE_CONNECTION_STRING is set.

    Locally:
        Falls back to data/raw/{symbol}_price_history.csv.

    Behavior:
        1. Load existing cloud/local CSV if available.
        2. Determine fetch_start.
        3. Fetch new EIA rows.
        4. Merge/deduplicate/sort.
        5. Save back to Blob or local CSV.
        6. Return updated DataFrame.
    """

    symbol = symbol.upper()

    if symbol not in SERIES:
        raise ValueError(f"Unsupported symbol: {symbol}. Use WTI or BRENT.")

    api_key = os.environ.get("EIA_API_KEY")

    if not api_key:
        raise ValueError("EIA_API_KEY not found. Set it as an environment variable.")

    existing_df = pd.DataFrame()

    use_blob = _is_blob_configured()

    if use_blob:
        existing_df = _download_from_blob(symbol=symbol)

        if force_refresh or existing_df.empty:
            fetch_start = start
            logger.info("Blob has no existing %s data. Fetching from %s.", symbol, fetch_start)
        else:
            existing_df["date"] = pd.to_datetime(existing_df["date"])
            last_date = existing_df["date"].max()
            fetch_start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

            logger.info("Blob %s data found up to %s.", symbol, last_date.date())
            logger.info("Fetching new rows from %s.", fetch_start)

    else:
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        local_path = data_dir / f"{symbol.lower()}_price_history.csv"

        if force_refresh or not local_path.exists():
            existing_df = pd.DataFrame()
            fetch_start = start
            logger.info("Local %s data missing. Fetching from %s.", symbol, fetch_start)
        else:
            existing_df = pd.read_csv(local_path)
            existing_df["date"] = pd.to_datetime(existing_df["date"])

            last_date = existing_df["date"].max()
            fetch_start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

            logger.info("Local %s data found up to %s.", symbol, last_date.date())
            logger.info("Fetching new rows from %s.", fetch_start)

    new_df = fetch_eia_price_data(
        symbol=symbol,
        api_key=api_key,
        start=fetch_start,
    )

    combined_df = merge_price_data(
        local_df=existing_df,
        new_df=new_df,
        symbol=symbol,
    )

    if use_blob:
        _upload_to_blob(symbol=symbol, df=combined_df)
        logger.info("Uploaded %d rows to Blob for %s.", len(combined_df), symbol)
    else:
        local_path = Path(data_dir) / f"{symbol.lower()}_price_history.csv"
        combined_df.to_csv(local_path, index=False)
        logger.info("Saved %d rows to %s.", len(combined_df), local_path)

    return combined_df


def fetch_eia_price_data(
    symbol: str,
    api_key: str,
    start: str,
) -> pd.DataFrame:
    """
    Fetch EIA crude price data using pagination with automatic retry.
    """

    series_id = SERIES[symbol.upper()]
    all_rows = []
    session = _make_retry_session()

    offset = 0
    page_size = 5000

    while True:
        params = {
            "api_key": api_key,
            "frequency": "daily",
            "data[0]": "value",
            "facets[series][]": series_id,
            "start": start,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "length": page_size,
            "offset": offset,
        }

        response = session.get(EIA_BASE, params=params, timeout=30)
        response.raise_for_status()

        rows = response.json().get("response", {}).get("data", [])

        if not rows:
            break

        all_rows.extend(rows)

        if len(rows) < page_size:
            break

        offset += page_size

    if not all_rows:
        return pd.DataFrame(columns=["date", "price", "symbol", "source"])

    df = pd.DataFrame(all_rows)

    df = df.rename(
        columns={
            "period": "date",
            "value": "price",
        }
    )

    df = df[["date", "price"]]

    df["date"] = pd.to_datetime(df["date"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df = df.dropna(subset=["date", "price"])
    df = df.sort_values("date").drop_duplicates("date", keep="last")

    df["symbol"] = symbol.upper()
    df["source"] = "EIA"

    return df.reset_index(drop=True)


def fetch_eia_futures_curve(
    symbol: str = "WTI",
    api_key: str | None = None,
    lookback_days: int = 10,
) -> list[float]:
    """
    Fetch the latest NYMEX futures curve (contracts 1-4) from EIA.

    Returns the most recent quote for each of RCLC1..RCLC4 as a list of prices
    [F1, F2, F3, F4]. Contracts with no recent data are dropped (shorter list).

    Requires EIA_API_KEY (same key as spot). Raises if unavailable.
    """
    symbol = symbol.upper()
    if symbol not in FUTURES_SERIES:
        raise ValueError(f"No futures series configured for {symbol}.")

    api_key = api_key or os.environ.get("EIA_API_KEY")
    if not api_key:
        raise ValueError("EIA_API_KEY not found; cannot fetch futures curve.")

    session = _make_retry_session()
    start = (pd.Timestamp.utcnow() - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    params = {
        "api_key": api_key,
        "frequency": "daily",
        "data[0]": "value",
        "start": start,
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "length": 200,
    }
    # one facet per contract series
    for series_id in FUTURES_SERIES[symbol]:
        params.setdefault("facets[series][]", [])
    # requests encodes a list value as repeated params
    params["facets[series][]"] = FUTURES_SERIES[symbol]

    response = session.get(EIA_FUT_BASE, params=params, timeout=30)
    response.raise_for_status()
    rows = response.json().get("response", {}).get("data", [])

    # latest value per series
    latest: dict[str, float] = {}
    latest_date: dict[str, str] = {}
    for row in rows:
        sid = row.get("series")
        period = row.get("period", "")
        try:
            val = float(row.get("value"))
        except (TypeError, ValueError):
            continue
        if sid not in latest_date or period > latest_date[sid]:
            latest_date[sid] = period
            latest[sid] = val

    curve = [latest[sid] for sid in FUTURES_SERIES[symbol] if sid in latest]
    if not curve:
        raise ValueError("EIA returned no futures quotes for the requested window.")
    return curve


def load_forward_curve(
    horizon: int,
    symbol: str = "WTI",
    frequency: str = "M",
    spot: float | None = None,
    api_key: str | None = None,
):
    """
    Build a ForwardCurve of length `horizon` from real EIA futures contracts 1-4,
    extrapolating the tail with the front log-slope. Falls back to a flat curve at
    `spot` if the futures fetch fails.

    Returns a hedging_assistant.contracts.ForwardCurve.
    """
    from hedging_assistant.engines.forward_curve import (
        build_curve_from_futures,
        build_parametric_curve,
    )

    if spot is None:
        # anchor at the latest spot if not supplied
        hist = load_price_history(symbol=symbol)
        spot = float(hist["price"].iloc[-1])

    try:
        front = fetch_eia_futures_curve(symbol=symbol, api_key=api_key)
        return build_curve_from_futures(front, spot=spot, horizon=horizon, frequency=frequency)
    except Exception as exc:  # network / key / no data
        logger.warning("Futures curve fetch failed (%s); using flat curve at spot.", exc)
        return build_parametric_curve(spot, horizon, 0.0, frequency, source="spot-fallback")


def merge_price_data(
    local_df: pd.DataFrame,
    new_df: pd.DataFrame,
    symbol: str,
) -> pd.DataFrame:
    """
    Merge existing and newly fetched price data.
    Remove duplicate rows by date.
    """

    frames = []

    if local_df is not None and not local_df.empty:
        frames.append(local_df)

    if new_df is not None and not new_df.empty:
        frames.append(new_df)

    if not frames:
        return pd.DataFrame(columns=["date", "price", "symbol", "source"])

    combined = pd.concat(frames, ignore_index=True)

    combined["date"] = pd.to_datetime(combined["date"])
    combined["price"] = pd.to_numeric(combined["price"], errors="coerce")
    combined["symbol"] = symbol.upper()

    if "source" not in combined.columns:
        combined["source"] = "EIA"

    combined["source"] = combined["source"].fillna("EIA")

    combined = combined.dropna(subset=["date", "price"])
    combined = combined[combined["price"] > 0]
    combined = combined.sort_values("date")
    combined = combined.drop_duplicates("date", keep="last")
    combined = combined.reset_index(drop=True)

    combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")

    return combined


def _is_blob_configured() -> bool:
    """
    Return True when Azure Blob environment variables are available.
    """

    return bool(os.environ.get("AZURE_STORAGE_CONNECTION_STRING"))


def _get_blob_client(symbol: str):
    """
    Get Azure Blob client for the symbol CSV.
    """

    if BlobServiceClient is None:
        raise ImportError(
            "azure-storage-blob is not installed. Add it to requirements.txt."
        )

    container_name = os.environ.get("BLOB_CONTAINER_NAME", "market-data")
    blob_name = os.environ.get(
        f"{symbol.upper()}_BLOB_NAME",
        f"{symbol.lower()}_price_history.csv",
    )

    # Prefer Managed Identity / DefaultAzureCredential when AZURE_STORAGE_ACCOUNT_URL
    # is set. Falls back to connection string for local dev.
    account_url = os.environ.get("AZURE_STORAGE_ACCOUNT_URL")
    if account_url and DefaultAzureCredential is not None:
        credential = DefaultAzureCredential()
        blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)
        logger.debug("Blob client using DefaultAzureCredential (managed identity).")
    else:
        connection_string = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        logger.debug("Blob client using connection string.")

    return blob_service_client.get_blob_client(
        container=container_name,
        blob=blob_name,
    )


def _download_from_blob(symbol: str) -> pd.DataFrame:
    """
    Download existing price CSV from Azure Blob.
    If blob does not exist, return empty DataFrame.
    """

    blob_client = _get_blob_client(symbol)

    try:
        blob_bytes = blob_client.download_blob().readall()
    except Exception as exc:
        message = str(exc).lower()

        if "blobnotfound" in message or "not found" in message:
            return pd.DataFrame(columns=["date", "price", "symbol", "source"])

        raise

    if not blob_bytes:
        return pd.DataFrame(columns=["date", "price", "symbol", "source"])

    return pd.read_csv(BytesIO(blob_bytes))


def _upload_to_blob(symbol: str, df: pd.DataFrame) -> None:
    """
    Upload updated price CSV to Azure Blob.
    """

    blob_client = _get_blob_client(symbol)

    csv_bytes = df.to_csv(index=False).encode("utf-8")

    blob_client.upload_blob(
        csv_bytes,
        overwrite=True,
    )


if __name__ == "__main__":
    df = load_price_history(symbol="WTI", start="2010-01-01")

    print(df.head())
    print(df.tail())

    print(f"Rows loaded: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Latest price: {df['price'].iloc[-1]:.2f} USD/bbl")