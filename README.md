
# Influence-Based Stock Direction Predictor

This project implements a mutual influence model to predict the next-day direction (up or down) of selected stocks based on historical price relationships.

## How It Works

The core of the system is an influence chart - a matrix that quantifies how the movement of one stock affects others. It operates on the assumption that companies do not move in isolation, but are part of an interconnected system where price changes in one stock can ripple through the market.

### Steps:
1. Log returns are calculated for each stock using historical price data.
2. Correlation coefficients between stock return pairs are computed using a rolling window (default: 126 trading days).
3. An influence matrix is generated based on these correlations, where only strong relationships (above a set threshold) are considered.
4. Each stock's predicted next-day direction is inferred using the influence-weighted signal from other stocks.

The system then compares predicted directions to the actual market movement and records the accuracy.

## Inspiration

This project was inspired by observing the ripple effect companies have on each other. For example, a major development in one tech company often triggers responses in suppliers, competitors, or companies in the same index. The goal was to see if this interdependence could be quantified and used to make directional predictions.

## Output

The script saves results to a CSV file (e.g., `improved_predictions_july2024.csv`), which includes:
- Prediction date
- Predicted direction for each stock
- Whether the prediction was correct
- Overall accuracy

## Requirements

- Python 3
- Packages: `pandas`, `numpy`, `yfinance`

## Disclaimer

This project was developed using ChatGPT to assist in implementation and structure. The idea and overall concept of modeling market relationships through an influence matrix are fully original.

This work is for educational and experimental purposes only. It is not financial advice.
