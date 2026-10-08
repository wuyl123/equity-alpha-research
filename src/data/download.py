from massive import RESTClient
import yaml

import time
from datetime import date, timedelta
from pathlib import Path
import os


import pandas as pd
import requests


with open('./configs/base.yaml', 'r', encoding='utf-8') as f:
    base = yaml.safe_load(f)

api_key = base['massive_api_key']

client = RESTClient(api_key, pagination=False)

STOCKS = [
    "AAPL", "MSFT", "NVDA", "IBM",
    "AMZN", "GOOGL", "META", "WMT",
    "JPM", "BAC",
    "JNJ", "UNH",
    "XOM", "CVX",
    "CAT", "HON",
    "NEE", "DUK",
    "AMT", "LIN",
]
TICKERS = STOCKS + ["SPY"]

# 留在近两年范围内；结束于昨天，避免当天未完成的日线。
END = date.today() - timedelta(days=1)
START = END - timedelta(days=700)

# 路径包含日期范围，防止误用其他区间的缓存。
DATA_DIR = Path("data/raw/massive") / f"{START}_{END}"
DATA_DIR.mkdir(parents=True, exist_ok=True)

session = requests.Session()
session.headers["Authorization"] = (
    f"Bearer {os.environ['MASSIVE_API_KEY']}"
)

last_request = 0.0

def fetch_json(url, params):
    """串行限速，并对 429 做有限次数重试。"""
    global last_request

    for attempt in range(4):
        wait = 15 - (time.monotonic() - last_request)
        if wait > 0:
            time.sleep(wait)

        last_request = time.monotonic()
        response = session.get(url, params=params, timeout=30)

        if response.status_code == 429:
            if attempt == 3:
                response.raise_for_status()

            retry_after = response.headers.get("Retry-After", "")
            delay = max(65, float(retry_after)) if retry_after.isdigit() else 65

            print(f"Rate limited; waiting {delay:.0f}s...")
            time.sleep(delay)
            continue

        response.raise_for_status()
        payload = response.json()

        if payload.get("status") == "ERROR":
            raise RuntimeError(payload.get("error", "Massive API error"))

        return payload


def download_daily(ticker):
    path = DATA_DIR / f"{ticker}.csv"

    if path.exists():
        print(f"{ticker}: cached, skipped")
        return

    url = (
        f"https://api.massive.com/v2/aggs/ticker/{ticker}"
        f"/range/1/day/{START}/{END}"
    )
    payload = fetch_json(
        url,
        params={
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
        },
    )

    # 这个日期范围的日线应该一页装下；防止静默遗漏数据。
    if payload.get("next_url"):
        raise RuntimeError(f"{ticker}: additional pages need to be downloaded")

    rows = payload.get("results", [])
    if not rows:
        raise RuntimeError(f"{ticker}: no data returned")

    df = pd.DataFrame(rows).rename(
        columns={
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        }
    )

    # 时间戳是毫秒；转成纽约市场日期。
    df["date"] = (
        pd.to_datetime(df["t"], unit="ms", utc=True)
        .dt.tz_convert("America/New_York")
        .dt.strftime("%Y-%m-%d")
    )
    df["ticker"] = ticker

    df = (
        df[["date", "ticker", "open", "high", "low", "close", "volume"]]
        .sort_values("date")
        .reset_index(drop=True)
    )

    if df["date"].duplicated().any():
        raise ValueError(f"{ticker}: duplicate dates")

    if df[["open", "high", "low", "close", "volume"]].isna().any().any():
        raise ValueError(f"{ticker}: missing OHLCV values")

    # 先写临时文件，再替换，避免中断时留下不完整的缓存。
    temp_path = path.with_suffix(".tmp")
    df.to_csv(temp_path, index=False)
    temp_path.replace(path)

    print(
        f"{ticker}: {len(df)} rows, "
        f"{df['date'].iloc[0]} -> {df['date'].iloc[-1]}"
    )


import json

DIVIDEND_DIR = DATA_DIR / "dividends"
PROCESSED_DIR = Path("data/processed/massive") / f"{START}_{END}"

DIVIDEND_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def download_dividends(ticker):
    path = DIVIDEND_DIR / f"{ticker}.json"

    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    url = "https://api.massive.com/stocks/v1/dividends"
    params = {
        "ticker": ticker,
        "ex_dividend_date.gte": str(START),
        "ex_dividend_date.lte": str(END),
        "sort": "ex_dividend_date.asc",
        "limit": 5000,
    }

    records = []

    while url:
        payload = fetch_json(url, params)
        records.extend(payload.get("results", []))

        # 后续分页也经过 fetch_json，保持限速。
        url = payload.get("next_url")
        params = None

    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(path)

    print(f"{ticker}: {len(records)} dividend records")
    return records

def build_adjusted_data(ticker, records):
    daily_path = DATA_DIR / f"{ticker}.csv"
    df = pd.read_csv(daily_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if df.empty or df["date"].duplicated().any():
        raise ValueError(f"{ticker}: empty data or duplicate dates")

    if df["close"].isna().any() or (df["close"] <= 0).any():
        raise ValueError(f"{ticker}: invalid closing prices")

    # 每个除息日对应的、拆股调整后的现金分红。
    dividends_by_date = {}
    seen_ids = set()

    for record in records:
        record_id = record.get("id")
        if record_id:
            if record_id in seen_ids:
                continue
            seen_ids.add(record_id)

        ex_date = record.get("ex_dividend_date")
        amount = record.get("split_adjusted_cash_amount")

        if not ex_date or amount is None:
            raise ValueError(
                f"{ticker}: missing ex-date or split-adjusted dividend"
            )

        if record.get("currency") != "USD":
            raise ValueError(f"{ticker}: dividend currency is not USD")

        amount = float(amount)
        if amount < 0:
            raise ValueError(f"{ticker}: negative dividend")

        ex_date = pd.Timestamp(ex_date)

        # 同日可能有普通分红和特别分红，需要合计。
        dividends_by_date[ex_date] = (
            dividends_by_date.get(ex_date, 0.0) + amount
        )

    # 不把缺失的除息日日线静默当作无分红。
    missing_dates = set(dividends_by_date) - set(df["date"])
    if missing_dates:
        raise ValueError(
            f"{ticker}: missing daily bars on ex-dates: "
            f"{sorted(missing_dates)}"
        )

    df["dividend"] = (
        df["date"].map(dividends_by_date).fillna(0.0)
    )

    previous_close = df["close"].shift(1)

    df["price_return"] = df["close"] / previous_close - 1
    df["total_return"] = (
        (df["close"] + df["dividend"]) / previous_close - 1
    )

    # 第一天没有前一天价格，收益率保留 NaN；
    # 累计指数以第一天为 100。
    growth = 1 + df["total_return"]
    growth.iloc[0] = 1.0
    df["total_return_index"] = 100 * growth.cumprod()

    # 末日锚定的总收益复权收盘价。
    df["adjusted_close"] = (
        df["total_return_index"]
        / df["total_return_index"].iloc[-1]
        * df["close"].iloc[-1]
    )

    path = PROCESSED_DIR / f"{ticker}.csv"
    temp_path = path.with_suffix(".tmp")
    df.to_csv(temp_path, index=False)
    temp_path.replace(path)

    print(f"{ticker}: processed data saved")
    return df

if __name__ == "__main__":
    failures = []

    for ticker in TICKERS:
        try:
            download_daily(ticker)
            records = download_dividends(ticker)
            build_adjusted_data(ticker, records)
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            failures.append(ticker)
            print(f"{ticker}: FAILED - {exc}")

    print(f"\nFailed tickers: {failures}")