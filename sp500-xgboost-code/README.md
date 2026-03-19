# S&P 500 XGBoost Portfolio Optimizer

A machine learning-driven investment analysis tool that applies **XGBoost regression models** to S&P 500 stocks to predict returns and construct optimized portfolios across short-term (1-month) and long-term (6-month) horizons.

Built as a graduate-level quantitative research project (ENGR 296 — MSOL Program).

---

## Overview

This project trains XGBoost models on historical S&P 500 price data and fundamental metrics fetched live via `yfinance`, then uses those predictions to construct and evaluate portfolios with configurable budgets.

Three core pipelines are included:

| Pipeline | Horizon | Features | Allocation Strategy |
|---|---|---|---|
| Short-Term Model | 1 month | Technical indicators + dividends | Sequential (top predicted return) |
| Long-Term Model | 6 months | Technical + fundamentals + dividends | Sector-balanced (≤25% per sector) |
| 2-Year Backtest | Monthly walk-forward | Long-term feature set | Equal-weight top-N picks |

---

## Key Features

- **XGBoost regression** trained on 2–3 years of rolling historical data per stock
- **Total return targets** — price appreciation + dividend income combined
- **Fundamental features** for long-term model: P/E ratio, ROE, profit margins, revenue growth, debt/equity, beta, and more
- **Sector-balanced allocation** to prevent concentration risk in long-term portfolios
- **Walk-forward backtesting** — no lookahead bias; features computed as-of each prediction date
- **S&P 500 benchmark comparison** (SPY) with alpha calculation
- **Auto-generated charts**: feature importance, predicted vs. actual scatter plots, portfolio allocation pie charts, sector breakdown bar charts

---

## Project Structure

```
sp500-xgboost-portfolio/
├── stocks_machine_learning_clean.py   # Main script — all three pipelines
├── requirements.txt                    # Python dependencies
├── .gitignore
└── README.md
```

**Generated outputs** (created at runtime, not tracked in git):
```
shortterm_training_data.csv
shortterm_predictions.csv
shortterm_portfolio_allocation.csv
longterm_training_data.csv
longterm_predictions.csv
longterm_portfolio_allocation.csv
longterm_model.pkl                      # Saved model for backtesting
backtest_monthly_summary.csv
backtest_all_picks.csv
shortterm_1_feature_importance.png
shortterm_2_predicted_vs_actual.png
shortterm_3_portfolio_allocation.png
shortterm_4_sector_breakdown.png
longterm_3_portfolio_allocation.png
longterm_4_sector_breakdown.png
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/sp500-xgboost-portfolio.git
cd sp500-xgboost-portfolio
pip install -r requirements.txt
```

Requires Python 3.9+.

---

## Usage

Open `stocks_machine_learning_clean.py` in Jupyter or run it as a script. The file is organized into clearly labeled sections — run them in order or independently.

### Pipeline 1 — Short-Term Model

```python
# Set budget at top of the execution cell (default: $10,000)
SHORTTERM_BUDGET = 10000   # or 50000

# Runs: data collection → model training → portfolio optimization
# Output: top 15 stocks by predicted 1-month return, allocation breakdown
```

### Pipeline 2 — Long-Term Model

```python
LONGTERM_BUDGET = 10000

# Runs: data collection → model training → sector-balanced allocation
# Also saves: longterm_model.pkl (required for backtest)
```

### Pipeline 3 — 2-Year Backtest

```python
# Requires longterm_model.pkl from Pipeline 2
# Runs a walk-forward monthly backtest over 24 months
# Reports: win rate, total return, direction accuracy, alpha vs S&P 500
```

---

## Model Details

### Short-Term Features (17 total)
| Category | Features |
|---|---|
| Momentum | 5-day, 10-day, 20-day returns |
| Technical | RSI, MACD histogram, MACD signal |
| Trend | Price vs SMA-10/20/50, SMA alignment |
| Volume | Volume ratio vs 20-day average |
| Risk | Annualized volatility, up-day ratio |
| Dividend | Yield, payout ratio, growth, has_dividend flag |

### Long-Term Features (25 total)
All short-term features (adapted for longer windows) plus:

| Category | Features |
|---|---|
| Momentum | 3-month, 6-month, 1-year returns |
| Trend | Price vs SMA-50/200, golden/death cross |
| Profitability | Profit margin, operating margin, ROE, ROA |
| Growth | Revenue growth, earnings growth |
| Valuation | P/E ratio, PEG ratio, price-to-book |
| Financial Health | Debt-to-equity, current ratio, market cap, beta |

### XGBoost Hyperparameters
```python
n_estimators=200
max_depth=6
learning_rate=0.1
subsample=0.8
colsample_bytree=0.8
objective='reg:squarederror'
```

---

## Known Limitations

- **Return magnitude overestimation**: The model shows strong directional accuracy but tends to overestimate return magnitudes (~45% average prediction error). Best used as a ranking/screening tool, not for precise return forecasting.
- **Concentration risk**: Without sector constraints, the model allocates heavily to single stocks. The long-term model includes sector caps (≤25% per sector) to mitigate this.
- **Short-term signal decay**: ML-based signals in liquid markets tend to get arbitraged away quickly. The long-term model is more defensible for fundamental-driven investing.
- **Data latency**: `yfinance` fundamental data (P/E, margins, etc.) reflects the most recent reported quarter, not as-of historical dates. This introduces mild lookahead bias in the backtest for fundamental features.

---

## Requirements

See `requirements.txt`. Core dependencies:

- `xgboost`
- `yfinance`
- `pandas`, `numpy`
- `scikit-learn`
- `matplotlib`, `seaborn`
- `python-dateutil`

---

## Academic Context

This project is part of a graduate quantitative research study comparing XGBoost models across two speculative asset markets:
1. **S&P 500 equities** (this repository)
2. **Pokémon TCG secondary market** (separate analysis)

The study examines whether machine learning models can generate alpha in markets with different liquidity profiles, information availability, and arbitrage dynamics.

---

## License

MIT License — free to use, modify, and distribute with attribution.
