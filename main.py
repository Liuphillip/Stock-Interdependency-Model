
# Improved Influence-Based Stock Direction Predictor
# Requirements: pip install yfinance pandas numpy

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import timedelta

tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "BRK-B", "TSLA", "UNH", "JPM",
    "V", "LLY", "JNJ", "WMT", "PG", "MA", "XOM", "CVX", "HD", "MRK"
]

print("Downloading data...")
raw_data = yf.download(tickers, start="2023-07-01", end="2025-06-20", group_by='ticker', auto_adjust=True)

clean_data = {}
for ticker in tickers:
    try:
        clean_data[ticker] = raw_data[ticker]['Close']
    except KeyError:
        pass

data = pd.DataFrame(clean_data).dropna(axis=1)
tickers = data.columns.tolist()

window_days = 126
start_eval_date = pd.Timestamp("2025-04-22")
end_eval_date = pd.Timestamp("2025-04-26")
dates = data.index
valid_dates = [d for d in dates if start_eval_date <= d <= end_eval_date]

predictions = []
correct = 0
total = 0

print("Running predictions with correlation filtering and weighting...")
for eval_date in valid_dates:
    idx = dates.get_loc(eval_date)
    if idx < window_days or idx + 1 >= len(dates):
        continue

    train_data = data.iloc[idx - window_days:idx]
    prev_day = data.iloc[idx - 1]
    today = data.iloc[idx]
    next_day = data.iloc[idx + 1]

    log_returns = np.log(train_data / train_data.shift(1)).dropna()
    predicted = today.copy()

    for j in tickers:
        influence_sum = 0
        weight_sum = 0
        for i in tickers:
            if i == j:
                continue
            x = log_returns[i].values
            y = log_returns[j].values
            if np.std(x) == 0 or np.std(y) == 0:
                continue

            r = np.corrcoef(x, y)[0, 1]
            if abs(r) < 0.3:
                continue

            slope, _ = np.polyfit(x, y, 1)
            influence = np.exp(slope)
            delta = today[i] / prev_day[i] - 1
            weight = abs(r)

            influence_sum += weight * influence ** delta
            weight_sum += weight

        if weight_sum > 0:
            predicted[j] = today[j] * (influence_sum / weight_sum)

    row = {"Date": dates[idx + 1].strftime("%Y-%m-%d")}
    for ticker in tickers:
        act_dir = "UP" if next_day[ticker] > today[ticker] else "DOWN"
        pred_dir = "UP" if predicted[ticker] > today[ticker] else "DOWN"
        row[f"{ticker}_Direction"] = pred_dir
        row[f"{ticker}_Correct"] = "✔" if pred_dir == act_dir else "✘"
        correct += pred_dir == act_dir
        total += 1

    predictions.append(row)

df = pd.DataFrame(predictions)
df.to_csv("improved_predictions_july2024.csv", index=False, encoding="utf-8-sig")
print(f"Saved predictions to 'improved_predictions.csv' — Accuracy: {correct}/{total} = {correct/total:.2%}")
