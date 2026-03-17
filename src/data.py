from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, Sequence

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def _utc_today() -> datetime:
    return (
        datetime.now(timezone.utc)
        .replace(hour=0, minute=0, second=0, microsecond=0)
        .replace(tzinfo=None)
    )


def _infer_price_column(df: pd.DataFrame) -> str:
    for candidate in ("Adj Close", "Close", "close", "price"):
        if candidate in df.columns:
            return candidate
    lowered = {col.lower(): col for col in df.columns}
    for token in ("adj close", "close", "price", "last"):
        for key, original in lowered.items():
            if token in key:
                return original
    raise ValueError(f"Could not infer price column from columns: {df.columns.tolist()}")


def load_price_series(path: Path, price_col: str | None = None) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date").sort_index()
    col = _infer_price_column(df) if price_col is None else price_col
    return df[col].rename(path.stem)


def load_prices(
    assets: Iterable[str],
    raw_dir: Path,
    price_col: str | None = None,
) -> Dict[str, pd.Series]:
    raw_dir = Path(raw_dir)
    out: Dict[str, pd.Series] = {}
    for asset in assets:
        safe = str(asset).replace("^", "")
        path = raw_dir / f"{safe}.csv"
        if not path.exists():
            logger.warning("Price file not found: %s", path)
            continue
        out[str(asset)] = load_price_series(path, price_col)
    return out


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    if "Ticker" in df.index:
        df = df.drop(index="Ticker")
    return df


def _load_existing(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(csv_path, header=[0, 1], index_col=0)
    except Exception:
        df = pd.read_csv(csv_path, header=0, index_col=0)
    df = _normalize_columns(df)
    df.index = pd.to_datetime(df.index, errors="coerce").tz_localize(None)
    df = df[~df.index.isna()]
    return df.sort_index()


def _save_csv(df: pd.DataFrame, csv_path: Path) -> None:
    df = df.copy()
    df.index.name = "Date"
    df.sort_index().to_csv(csv_path)


def _download_ticker(
    ticker: str,
    csv_path: Path,
    start_date: datetime,
    end_date: datetime,
    existing: pd.DataFrame,
) -> None:
    if start_date >= end_date:
        logger.info("Skip %s: start_date >= end_date", ticker)
        return

    new_data = None
    for attempt in range(1, 6):
        try:
            logger.info(
                "Downloading %s (attempt %d/5) from %s to %s",
                ticker,
                attempt,
                start_date.date().isoformat(),
                end_date.date().isoformat(),
            )
            new_data = yf.download(
                ticker,
                start=start_date.date(),
                end=(end_date + timedelta(days=1)).date(),
                interval="1d",
                auto_adjust=False,
                progress=False,
            )
            break
        except Exception as exc:
            logger.warning("Download failed for %s (attempt %d): %s", ticker, attempt, exc)
            if attempt == 5:
                raise
            time.sleep(2 * attempt)

    if new_data is None or new_data.empty:
        logger.info("No new data for %s", ticker)
        return

    new_data = _normalize_columns(new_data.copy())
    new_data.index = pd.to_datetime(new_data.index).tz_localize(None)

    combined = pd.concat([existing, new_data], axis=0) if not existing.empty else new_data
    combined = combined[~combined.index.duplicated(keep="last")]
    _save_csv(combined, csv_path)
    logger.info("Saved %s (%d rows) to %s", ticker, len(combined), csv_path)


def download_tickers(
    tickers: Sequence[str],
    output_dir: Path,
    start_date: str = "2005-01-01",
    end_date: str | None = None,
) -> None:
    """
    Download (or incrementally update) daily OHLCV data for tickers.

    Writes one CSV per ticker under output_dir. Naming: '^' is stripped.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else _utc_today()

    for ticker in tickers:
        ticker = str(ticker)
        safe = ticker.replace("^", "")
        csv_path = output_dir / f"{safe}.csv"
        existing = _load_existing(csv_path)
        effective_start = start_dt
        if not existing.empty:
            last_date = existing.index.max()
            if last_date is not None:
                effective_start = max(start_dt, last_date + timedelta(days=1))
        _download_ticker(
            ticker,
            csv_path,
            start_date=effective_start,
            end_date=end_dt,
            existing=existing,
        )

