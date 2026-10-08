import pandas as pd


def calculate_features(prices: pd.DataFrame) -> pd.DataFrame:
    result = (
        prices.copy()
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    grouped = result.groupby("ticker")

    for window in (5, 20, 60):
        result[f"ret_{window}"] = (
            result["adjusted_close"]
            / grouped["adjusted_close"].shift(window)
            - 1
        )

    result["vol_20"] = grouped["total_return"].transform(
        lambda s: s.rolling(20, min_periods=20).std()
    )

    return result