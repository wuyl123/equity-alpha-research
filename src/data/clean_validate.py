from pathlib import Path
import pandas as pd

# 替换成你实际的日期区间目录
data_dir = Path("data/processed/massive/2024-11-05_2026-10-06")

files = sorted(data_dir.glob("*.csv"))
if not files:
    raise ValueError(f"No CSV files found in {data_dir}")

prices = pd.concat(
    [pd.read_csv(path, parse_dates=["date"]) for path in files],
    ignore_index=True,
).sort_values(["ticker", "date"]).reset_index(drop=True)

import numpy as np

numeric_columns = [
    "open", "high", "low", "close",
    "volume", "dividend", "adjusted_close",
]

print("股票数量：", prices["ticker"].nunique())
print("重复记录：", prices.duplicated(["ticker", "date"]).sum())
print("缺失值：\n", prices[numeric_columns].isna().sum())
print(
    "非有限数值：\n",
    (~np.isfinite(prices[numeric_columns])).sum(),
)

print(
    prices.groupby("ticker").agg(
        start_date=("date", "min"),
        end_date=("date", "max"),
        rows=("date", "size"),
    )
)

largest_moves = prices["total_return"].abs().nlargest(20).index

print(
    prices.loc[
        largest_moves,
        ["date", "ticker", "close", "dividend", "total_return"],
    ]
)