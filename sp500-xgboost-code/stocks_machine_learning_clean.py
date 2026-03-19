# ==============================================================================
# # S&P 500 XGBoost Investment Analyzer\n\nThis notebook contains three self-contained pipelines:\n\n1. **Short-Term Model** — predicts 1-month total returns, allocates a configurable budget\n2. **Long-Term Model** — predicts 6-month total returns with fundamental features, sector-balanced allocation, saves model to disk\n3. **2-Year Backtest** — walk-forward monthly backtest using the saved long-term model\n\n**Run order:** Cell 1 → Cell 2 → (Cell 3 or Cell 5) → Cell 7 (after Cell 5)
# ==============================================================================

import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import pickle
import os
import warnings

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("✓ All libraries imported successfully")



# ==============================================================================
# ## Shared Utilities\nFunctions used by every pipeline: ticker list, RSI, MACD, dividend info.
# ==============================================================================

# ============================================================================
# SHARED UTILITIES
# ============================================================================

def get_sp500_tickers():
    """Returns a reliable list of major S&P 500 stocks"""
    tickers = [
        # Mega-cap Tech
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'ORCL', 'ADBE',
        'CRM', 'CSCO', 'ACN', 'AMD', 'INTC', 'QCOM', 'TXN', 'INTU', 'NOW', 'AMAT',
        'PANW', 'MU', 'ADI', 'LRCX', 'KLAC', 'SNPS', 'CDNS', 'MCHP', 'FTNT', 'NXPI',

        # Healthcare
        'UNH', 'JNJ', 'LLY', 'ABBV', 'MRK', 'TMO', 'ABT', 'PFE', 'DHR', 'BMY',
        'AMGN', 'GILD', 'CVS', 'CI', 'MDT', 'ISRG', 'VRTX', 'REGN', 'HUM', 'ZTS',
        'SYK', 'BSX', 'EW', 'IDXX', 'HCA', 'DXCM', 'IQV', 'RMD', 'ALGN', 'BDX',

        # Financials
        'BRK-B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'AXP', 'BLK',
        'C', 'SPGI', 'PGR', 'CB', 'MMC', 'SCHW', 'USB', 'PNC', 'TFC', 'COF',
        'AON', 'FIS', 'MCO', 'CME', 'ICE', 'AFL', 'AIG', 'MET', 'PRU', 'ALL',

        # Consumer Discretionary
        'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'LOW', 'SBUX', 'TJX', 'BKNG', 'CMG',
        'ORLY', 'AZO', 'MAR', 'ABNB', 'GM', 'F', 'YUM', 'DHI', 'LEN', 'HLT',

        # Consumer Staples
        'WMT', 'PG', 'COST', 'KO', 'PEP', 'PM', 'MO', 'MDLZ', 'CL', 'KMB',
        'GIS', 'HSY', 'K', 'CHD', 'CLX', 'SJM', 'KHC', 'TSN', 'HRL', 'CAG',

        # Energy
        'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HAL',
        'PXD', 'WMB', 'KMI', 'OKE', 'FANG', 'DVN', 'HES', 'BKR', 'TRGP', 'MRO',

        # Industrials
        'UPS', 'HON', 'UNP', 'RTX', 'CAT', 'BA', 'GE', 'DE', 'LMT', 'MMM',
        'GD', 'ETN', 'NOC', 'ITW', 'EMR', 'CSX', 'NSC', 'FDX', 'WM',
        'TT', 'PH', 'CARR', 'PCAR', 'JCI', 'ROK', 'ODFL', 'IR', 'VRSK', 'IEX',

        # Communication Services
        'GOOGL', 'META', 'NFLX', 'DIS', 'CMCSA', 'T', 'VZ', 'TMUS', 'EA', 'TTWO',
        'CHTR', 'PARA', 'OMC', 'IPG', 'FOXA', 'MTCH', 'LYV', 'NWSA', 'FOX',

        # Utilities
        'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'ES', 'ED',
        'PEG', 'WEC', 'AWK', 'DTE', 'ETR', 'FE', 'PPL', 'AEE', 'CMS', 'CNP',

        # Materials
        'LIN', 'APD', 'SHW', 'ECL', 'DD', 'NEM', 'DOW', 'PPG', 'FCX', 'ALB',
        'CTVA', 'NUE', 'VMC', 'MLM', 'BALL', 'AVY', 'CE', 'FMC', 'IFF', 'EMN',

        # Real Estate
        'PLD', 'AMT', 'EQIX', 'CCI', 'PSA', 'SPG', 'O', 'WELL', 'DLR', 'AVB',
        'EQR', 'SBAC', 'VTR', 'ARE', 'INVH', 'MAA', 'ESS', 'EXR', 'CBRE', 'BXP'
    ]
    return list(dict.fromkeys(tickers))  # deduplicate while preserving order


def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices):
    """Calculate MACD and signal line"""
    exp1 = prices.ewm(span=12, adjust=False).mean()
    exp2 = prices.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal


def get_dividend_info(ticker):
    """Get dividend yield, rate, payout ratio, and growth for a stock"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        dividend_yield = info.get('dividendYield', 0) or 0
        dividend_rate  = info.get('dividendRate', 0) or 0
        payout_ratio   = info.get('payoutRatio', 0) or 0

        try:
            dividends = stock.dividends
            if len(dividends) > 0:
                recent_divs = dividends.last('2Y')
                if len(recent_divs) >= 2:
                    recent_year   = recent_divs.last('1Y').sum()
                    previous_year = recent_divs.iloc[:-len(recent_divs.last('1Y'))].sum()
                    dividend_growth = (recent_year / previous_year - 1) * 100 if previous_year > 0 else 0
                else:
                    dividend_growth = 0
            else:
                dividend_growth = 0
        except Exception:
            dividend_growth = 0

        return {
            'dividend_yield':  dividend_yield * 100 if dividend_yield else 0,
            'dividend_rate':   dividend_rate,
            'payout_ratio':    payout_ratio * 100 if payout_ratio else 0,
            'dividend_growth': dividend_growth,
            'has_dividend':    1 if dividend_yield and dividend_yield > 0 else 0
        }
    except Exception:
        return {'dividend_yield': 0, 'dividend_rate': 0,
                'payout_ratio': 0, 'dividend_growth': 0, 'has_dividend': 0}

print("✓ Shared utility functions defined")



# ==============================================================================
# ## Pipeline 1 — Short-Term Model (1-Month Horizon)\n\nFeatures: technical indicators + dividends. Target: `forward_total_return_1m`.\n\nSet `SHORTTERM_BUDGET` to `10000` or `50000` before running.
# ==============================================================================

# ============================================================================
# SHORT-TERM MODEL — 1-MONTH HORIZON
# Predicts total return (price + dividends) over the next ~22 trading days
# ============================================================================

def create_shortterm_features(ticker, lookback_days=60):
    """Build a labelled feature row per trading day for short-term prediction."""
    try:
        print(f"  Processing {ticker}...", end='')
        stock = yf.Ticker(ticker)
        hist  = stock.history(period="2y")

        if len(hist) < 150:
            print(" ❌ Not enough data")
            return None

        prices  = hist['Close']
        volumes = hist['Volume']

        try:
            dividends = stock.dividends.reindex(hist.index, fill_value=0)
        except Exception:
            dividends = pd.Series(0, index=hist.index)

        dividend_info = get_dividend_info(ticker)
        data = []

        for i in range(lookback_days, len(hist) - 22):
            wp = prices.iloc[i - lookback_days:i]
            wv = volumes.iloc[i - lookback_days:i]
            cp = prices.iloc[i]

            if wp.isna().any() or wv.isna().any():
                continue

            rsi = calculate_rsi(wp).iloc[-1]
            macd, macd_sig = calculate_macd(wp)
            macd_val  = macd.iloc[-1]
            macd_sigv = macd_sig.iloc[-1]
            macd_hist = macd_val - macd_sigv

            if pd.isna(rsi) or pd.isna(macd_hist):
                continue

            sma10 = wp.rolling(10).mean().iloc[-1]
            sma20 = wp.rolling(20).mean().iloc[-1]
            sma50 = wp.rolling(50).mean().iloc[-1]

            ret5d  = (wp.iloc[-1] / wp.iloc[-6]  - 1) * 100
            ret10d = (wp.iloc[-1] / wp.iloc[-11] - 1) * 100
            ret20d = (wp.iloc[-1] / wp.iloc[-21] - 1) * 100

            volatility   = wp.pct_change().std() * np.sqrt(252) * 100
            avg_vol      = wv.rolling(20).mean().iloc[-1]
            volume_ratio = wv.iloc[-1] / avg_vol if avg_vol > 0 else 1

            # Targets
            fp   = prices.iloc[i + 22]
            fdiv = dividends.iloc[i:i + 22].sum()
            p_ret = (fp / cp - 1) * 100
            d_ret = (fdiv / cp) * 100 if cp > 0 else 0

            fp2w   = prices.iloc[i + 11]
            fdiv2w = dividends.iloc[i:i + 11].sum()
            p2w = (fp2w / cp - 1) * 100
            d2w = (fdiv2w / cp) * 100 if cp > 0 else 0

            data.append({
                'ticker': ticker, 'date': hist.index[i], 'current_price': cp,
                'returns_5d': ret5d, 'returns_10d': ret10d, 'returns_20d': ret20d,
                'rsi': rsi, 'macd_histogram': macd_hist,
                'macd_bullish': int(macd_val > macd_sigv),
                'price_to_sma10': (cp / sma10 - 1) * 100,
                'price_to_sma20': (cp / sma20 - 1) * 100,
                'price_to_sma50': (cp / sma50 - 1) * 100,
                'sma_alignment': int(sma10 > sma20 > sma50),
                'volume_ratio': volume_ratio, 'volatility': volatility,
                'up_day_ratio': (wp.diff().iloc[-20:] > 0).sum() / 20,
                'dividend_yield':   dividend_info['dividend_yield'],
                'dividend_rate':    dividend_info['dividend_rate'],
                'payout_ratio':     dividend_info['payout_ratio'],
                'dividend_growth':  dividend_info['dividend_growth'],
                'has_dividend':     dividend_info['has_dividend'],
                'forward_price_return_2w':    p2w,
                'forward_dividend_return_2w': d2w,
                'forward_total_return_2w':    p2w + d2w,
                'forward_price_return_1m':    p_ret,
                'forward_dividend_return_1m': d_ret,
                'forward_total_return_1m':    p_ret + d_ret,
            })

        df = pd.DataFrame(data)
        print(f" ✓ {len(df)} samples")
        return df

    except Exception as e:
        print(f" ❌ {e}")
        return None


def collect_shortterm_training_data(num_stocks=150):
    """Gather short-term training data across many stocks."""
    print("=" * 80)
    print("STEP 1: COLLECTING SHORT-TERM TRAINING DATA")
    print("=" * 80)

    tickers = get_sp500_tickers()[:num_stocks]
    print(f"\nProcessing {len(tickers)} stocks...")

    all_data, successful = [], 0

    for ticker in tickers:
        df = create_shortterm_features(ticker)
        if df is not None and len(df) > 0:
            all_data.append(df)
            successful += 1

    if not all_data:
        print("\n❌ No data collected!")
        return None

    combined = pd.concat(all_data, ignore_index=True)
    print(f"\nSuccessfully processed: {successful}/{len(tickers)} stocks")
    print(f"Total training samples: {len(combined):,}")
    print(f"Date range: {combined['date'].min()} to {combined['date'].max()}")
    print(f"\nDIVIDEND STATS:")
    print(f"  Stocks with dividends: {combined['has_dividend'].mean()*100:.1f}%")
    print(f"  Avg total return:      {combined['forward_total_return_1m'].mean():.2f}%")
    return combined


def train_shortterm_model(data, target='forward_total_return_1m'):
    """Train XGBoost for 1-month total return prediction."""
    print("\n" + "=" * 80)
    print("STEP 2: TRAINING SHORT-TERM XGBOOST MODEL")
    print("=" * 80)

    feature_cols = [
        'returns_5d', 'returns_10d', 'returns_20d',
        'rsi', 'macd_histogram', 'macd_bullish',
        'price_to_sma10', 'price_to_sma20', 'price_to_sma50',
        'sma_alignment', 'volume_ratio', 'volatility', 'up_day_ratio',
        'dividend_yield', 'payout_ratio', 'dividend_growth', 'has_dividend'
    ]

    X, y = data[feature_cols], data[target]
    mask = ~(X.isna().any(axis=1) | y.isna())
    X, y = X[mask], y[mask]

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    print(f"Training: {len(X_train):,}  |  Test: {len(X_test):,}")

    model = xgb.XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        objective='reg:squarederror', random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    print(f"\nTrain MAE: {mean_absolute_error(y_train, y_pred_train):.3f}%  "
          f"| Train R²: {r2_score(y_train, y_pred_train):.3f}")
    print(f"Test  MAE: {mean_absolute_error(y_test, y_pred_test):.3f}%  "
          f"| Test  R²: {r2_score(y_test, y_pred_test):.3f}")

    # Feature importance chart
    importance = pd.DataFrame({'feature': feature_cols,
                                'importance': model.feature_importances_}
                              ).sort_values('importance', ascending=True)
    div_feats = {'dividend_yield', 'payout_ratio', 'dividend_growth', 'has_dividend'}
    colors = ['gold' if f in div_feats else 'steelblue' for f in importance['feature']]

    plt.figure(figsize=(10, 8))
    plt.barh(importance['feature'], importance['importance'], color=colors)
    plt.xlabel('Importance Score', fontsize=12)
    plt.title('Short-Term Feature Importance\n(Gold = Dividend Features)',
               fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('shortterm_1_feature_importance.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: shortterm_1_feature_importance.png")
    plt.close()

    # Predicted vs Actual
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, yt, yp, title, color in [
        (axes[0], y_train, y_pred_train, 'Training Set', 'blue'),
        (axes[1], y_test,  y_pred_test,  'Test Set',     'green')
    ]:
        ax.scatter(yt, yp, alpha=0.3, s=10, color=color)
        ax.plot([yt.min(), yt.max()], [yt.min(), yt.max()], 'r--', lw=2, label='Perfect')
        ax.set_xlabel('Actual Total Return (%)')
        ax.set_ylabel('Predicted Total Return (%)')
        ax.set_title(title, fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('shortterm_2_predicted_vs_actual.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: shortterm_2_predicted_vs_actual.png")
    plt.close()

    return model, feature_cols


def get_current_shortterm_features(tickers):
    """Fetch current features for the short-term model."""
    print("\n" + "=" * 80)
    print("STEP 3: FETCHING CURRENT SHORT-TERM STOCK DATA")
    print("=" * 80)

    current_data, successful = [], 0

    for ticker in tickers:
        try:
            print(f"  Fetching {ticker}...", end='')
            stock = yf.Ticker(ticker)
            hist  = stock.history(period="3mo")

            if len(hist) < 60:
                print(" ❌ Not enough data")
                continue

            prices, volumes = hist['Close'], hist['Volume']
            cp = prices.iloc[-1]

            rsi = calculate_rsi(prices).iloc[-1]
            macd, macd_sig = calculate_macd(prices)
            macd_val  = macd.iloc[-1]
            macd_sigv = macd_sig.iloc[-1]

            sma10 = prices.rolling(10).mean().iloc[-1]
            sma20 = prices.rolling(20).mean().iloc[-1]
            sma50 = prices.rolling(50).mean().iloc[-1]

            avg_vol = volumes.rolling(20).mean().iloc[-1]
            info = stock.info
            div_info = get_dividend_info(ticker)

            current_data.append({
                'ticker': ticker,
                'sector': info.get('sector', 'Unknown'),
                'current_price': cp,
                'market_cap': info.get('marketCap', 0),
                'returns_5d':  (prices.iloc[-1] / prices.iloc[-6]  - 1) * 100,
                'returns_10d': (prices.iloc[-1] / prices.iloc[-11] - 1) * 100,
                'returns_20d': (prices.iloc[-1] / prices.iloc[-21] - 1) * 100,
                'rsi': rsi,
                'macd_histogram': macd_val - macd_sigv,
                'macd_bullish':   int(macd_val > macd_sigv),
                'price_to_sma10': (cp / sma10 - 1) * 100,
                'price_to_sma20': (cp / sma20 - 1) * 100,
                'price_to_sma50': (cp / sma50 - 1) * 100,
                'sma_alignment':  int(sma10 > sma20 > sma50),
                'volume_ratio':   volumes.iloc[-1] / avg_vol if avg_vol > 0 else 1,
                'volatility':     prices.pct_change().std() * np.sqrt(252) * 100,
                'up_day_ratio':   (prices.diff().iloc[-20:] > 0).sum() / 20,
                'dividend_yield':  div_info['dividend_yield'],
                'dividend_rate':   div_info['dividend_rate'],
                'payout_ratio':    div_info['payout_ratio'],
                'dividend_growth': div_info['dividend_growth'],
                'has_dividend':    div_info['has_dividend'],
            })
            successful += 1
            print(" ✓")
        except Exception as e:
            print(f" ❌ {e}")

    print(f"\nFetched: {successful}/{len(tickers)} stocks")
    return pd.DataFrame(current_data)


def optimize_shortterm_portfolio(model, feature_cols, budget=10000,
                                  num_stocks=150, max_positions=15):
    """Sequential allocation of <budget> across the highest-predicted short-term returns."""
    tickers = get_sp500_tickers()[:num_stocks]
    current_data = get_current_shortterm_features(tickers)

    if current_data.empty:
        print("\n❌ No current data collected!")
        return None, None

    print("\n" + "=" * 80)
    print(f"STEP 4: OPTIMIZING ${budget:,} SHORT-TERM PORTFOLIO")
    print("=" * 80)

    X = current_data[feature_cols]
    current_data = current_data.copy()
    current_data['predicted_total_return']   = model.predict(X)
    current_data['estimated_dividend_return'] = current_data['dividend_yield'] / 12
    current_data['estimated_price_return']    = (current_data['predicted_total_return']
                                                  - current_data['estimated_dividend_return'])
    current_data = current_data.sort_values('predicted_total_return', ascending=False)

    # Top 20 preview
    print("\n" + "=" * 80)
    print("TOP 20 STOCKS BY PREDICTED 1-MONTH TOTAL RETURN")
    print("=" * 80)
    print(f"{'Rank':<6}{'Ticker':<8}{'Total%':<9}{'Price%':<9}{'Div%':<8}{'DivYld%':<10}{'Vol%':<10}{'Sector'}")
    print("-" * 90)
    for i, (_, row) in enumerate(current_data.head(20).iterrows(), 1):
        print(f"{i:<6}{row['ticker']:<8}{row['predicted_total_return']:>8.2f} "
              f"{row['estimated_price_return']:>8.2f} "
              f"{row['estimated_dividend_return']:>7.2f} "
              f"{row['dividend_yield']:>9.2f} "
              f"{row['volatility']:>9.1f}  "
              f"{row['sector']}")

    # Sequential allocation
    allocations, remaining = [], budget
    for _, row in current_data.head(40).iterrows():
        if remaining < row['current_price'] or len(allocations) >= max_positions:
            break
        shares = int(remaining / row['current_price'])
        if shares < 1:
            continue
        inv = shares * row['current_price']
        remaining -= inv
        allocations.append({
            'ticker': row['ticker'], 'sector': row['sector'],
            'shares': shares, 'price': row['current_price'],
            'investment': inv, 'weight': inv / budget * 100,
            'predicted_total_return':    row['predicted_total_return'],
            'predicted_price_return':    row['estimated_price_return'],
            'predicted_dividend_return': row['estimated_dividend_return'],
            'dividend_yield': row['dividend_yield'],
            'volatility':     row['volatility'],
        })

    if not allocations:
        print("\n❌ Could not allocate any stocks!")
        return None, None

    alloc_df     = pd.DataFrame(allocations)
    total_invested = budget - remaining

    p_ret = sum(a['investment'] * a['predicted_total_return']   for a in allocations) / total_invested
    d_ret = sum(a['investment'] * a['predicted_dividend_return'] for a in allocations) / total_invested
    pr_ret = p_ret - d_ret
    port_vol = np.sqrt(sum((a['investment'] / total_invested * a['volatility'])**2
                           for a in allocations))
    sector_breakdown = alloc_df.groupby('sector')['investment'].sum().sort_values(ascending=False)

    print("\n" + "=" * 80)
    print("PORTFOLIO SUMMARY (SHORT-TERM)")
    print("=" * 80)
    print(f"Budget:             ${budget:>10,.2f}")
    print(f"Invested:           ${total_invested:>10,.2f}  ({total_invested/budget*100:.1f}%)")
    print(f"Cash Remaining:     ${remaining:>10,.2f}  ({remaining/budget*100:.1f}%)")
    print(f"Positions:          {len(allocations)}")
    print(f"Portfolio Volatility: {port_vol:.2f}%")
    print(f"\nExpected 1-Month Total Return:  {p_ret:>7.2f}%  (${total_invested*p_ret/100:,.2f})")
    print(f"  ├─ Price:                       {pr_ret:>7.2f}%  (${total_invested*pr_ret/100:,.2f})")
    print(f"  └─ Dividends:                   {d_ret:>7.2f}%  (${total_invested*d_ret/100:,.2f})")
    print(f"Sharpe-like Ratio:  {p_ret/port_vol:.3f}")

    print("\n" + "=" * 80)
    print("SECTOR BREAKDOWN")
    print("=" * 80)
    for sector, amt in sector_breakdown.items():
        print(f"  {sector:<25s}: ${amt:>8,.2f}  ({amt/total_invested*100:.1f}%)")

    # Visualisations
    _plot_portfolio(alloc_df, total_invested, budget, port_vol, sector_breakdown,
                    prefix='shortterm')

    current_data.to_csv('shortterm_predictions.csv', index=False)
    alloc_df.to_csv('shortterm_portfolio_allocation.csv', index=False)
    print("\n✓ Saved: shortterm_predictions.csv, shortterm_portfolio_allocation.csv")

    return current_data, allocations


def _plot_portfolio(alloc_df, total_invested, budget, port_vol, sector_breakdown, prefix):
    """Shared portfolio visualisation helper."""
    # Pie + scatter
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    colors = plt.cm.Set3(np.linspace(0, 1, len(alloc_df)))
    ret_col = 'predicted_total_return' if 'predicted_total_return' in alloc_df.columns else 'predicted_6m_return'

    axes[0].pie(alloc_df['investment'], labels=alloc_df['ticker'],
                autopct=lambda p: f'{p:.1f}%' if p > 3 else '',
                colors=colors, startangle=90)
    axes[0].set_title(f'Portfolio Allocation\n${total_invested:,.0f} invested',
                       fontsize=14, fontweight='bold')

    axes[1].scatter(alloc_df['volatility'], alloc_df[ret_col],
                    s=alloc_df['investment'] / 50, alpha=0.6,
                    c=range(len(alloc_df)), cmap='viridis')
    for _, row in alloc_df.iterrows():
        axes[1].annotate(row['ticker'], (row['volatility'], row[ret_col]), fontsize=8)
    axes[1].set_xlabel('Risk (Volatility %)'); axes[1].set_ylabel('Expected Return (%)')
    axes[1].set_title('Risk vs Return\n(bubble size = investment)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{prefix}_3_portfolio_allocation.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {prefix}_3_portfolio_allocation.png")
    plt.close()

    # Sector bar
    fig, ax = plt.subplots(figsize=(12, 8))
    amounts = sector_breakdown.values
    bars = ax.barh(sector_breakdown.index, amounts, color='steelblue', alpha=0.7)
    ax.set_xlabel('Investment Amount ($)')
    ax.set_title('Sector Diversification', fontsize=14, fontweight='bold')
    for bar, amt in zip(bars, amounts):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                f' ${amt:,.0f} ({amt/total_invested*100:.1f}%)', va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{prefix}_4_sector_breakdown.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {prefix}_4_sector_breakdown.png")
    plt.close()

print("✓ Short-term model functions defined")



# ============================================================================
# RUN SHORT-TERM MODEL  (budget is a variable — change as needed)
# ============================================================================
SHORTTERM_BUDGET = 10000   # ← change to 50000 for the $50k scenario

training_data_st = collect_shortterm_training_data(num_stocks=150)

if training_data_st is not None:
    training_data_st.to_csv('shortterm_training_data.csv', index=False)
    print("\n✓ Training data saved to shortterm_training_data.csv")

    shortterm_model, shortterm_features = train_shortterm_model(training_data_st)

    predictions_st, allocations_st = optimize_shortterm_portfolio(
        shortterm_model, shortterm_features,
        budget=SHORTTERM_BUDGET, num_stocks=150, max_positions=15
    )
else:
    print("❌ Short-term training failed. Check your internet connection / yfinance access.")



# ==============================================================================
# ## Pipeline 2 — Long-Term Model (6-Month Horizon)\n\nFeatures: technical + fundamental metrics + dividends. Target: `forward_total_return_6m`.  \nModel is saved to `longterm_model.pkl` for use by the backtest.
# ==============================================================================

# ============================================================================
# LONG-TERM MODEL — 6-MONTH HORIZON
# Adds fundamental features (P/E, ROE, debt/equity, etc.) to the technical set
# ============================================================================

def create_longterm_features(ticker, lookback_days=120):
    """Build labelled feature rows for 6-month return prediction."""
    try:
        print(f"  Processing {ticker}...", end='')
        stock = yf.Ticker(ticker)
        hist  = stock.history(period="3y")

        if len(hist) < 250:
            print(" ❌ Not enough data")
            return None

        prices, volumes = hist['Close'], hist['Volume']

        try:
            dividends = stock.dividends.reindex(hist.index, fill_value=0)
        except Exception:
            dividends = pd.Series(0, index=hist.index)

        info          = stock.info
        dividend_info = get_dividend_info(ticker)
        data = []

        for i in range(lookback_days, len(hist) - 126):
            wp = prices.iloc[i - lookback_days:i]
            wv = volumes.iloc[i - lookback_days:i]
            cp = prices.iloc[i]

            if wp.isna().any() or wv.isna().any():
                continue

            ret_1m = (wp.iloc[-1] / wp.iloc[-22]  - 1) * 100 if len(wp) >= 22  else 0
            ret_3m = (wp.iloc[-1] / wp.iloc[-63]  - 1) * 100 if len(wp) >= 63  else 0
            ret_6m = (wp.iloc[-1] / wp.iloc[-126] - 1) * 100 if len(wp) >= 126 else 0
            ret_1y = (wp.iloc[-1] / wp.iloc[0]    - 1) * 100 if len(wp) >= 252 else 0

            rsi = calculate_rsi(wp).iloc[-1]
            macd, macd_sig = calculate_macd(wp)
            macd_bullish   = int(macd.iloc[-1] > macd_sig.iloc[-1])

            sma50  = wp.rolling(50).mean().iloc[-1]
            sma200 = wp.rolling(min(200, len(wp))).mean().iloc[-1]

            vol_90d    = wp.pct_change().rolling(90).std().iloc[-1] * np.sqrt(252) * 100
            avg_vol    = wv.rolling(50).mean().iloc[-1]
            vol_ratio  = wv.iloc[-1] / avg_vol if avg_vol > 0 else 1
            up_days_pct = (wp.diff().iloc[-60:] > 0).sum() / 60 if len(wp) >= 60 else 0.5

            def safe(key, default):
                v = info.get(key, default)
                return default if v is None else v

            fp   = prices.iloc[i + 126]
            fdiv = dividends.iloc[i:i + 126].sum()
            p_ret = (fp / cp - 1) * 100
            d_ret = (fdiv / cp) * 100 if cp > 0 else 0

            data.append({
                'ticker': ticker, 'date': hist.index[i], 'current_price': cp,
                'returns_1m': ret_1m, 'returns_3m': ret_3m,
                'returns_6m': ret_6m, 'returns_1y': ret_1y,
                'rsi': rsi, 'macd_bullish': macd_bullish,
                'price_above_sma50':  int(cp > sma50),
                'price_above_sma200': int(cp > sma200),
                'long_term_trend':    int(sma50 > sma200),
                'volatility_90d': vol_90d, 'volume_ratio': vol_ratio,
                'up_days_pct': up_days_pct,
                'profit_margin':    (safe('profitMargins', 0) or 0) * 100,
                'operating_margin': (safe('operatingMargins', 0) or 0) * 100,
                'roe':              (safe('returnOnEquity', 0) or 0) * 100,
                'roa':              (safe('returnOnAssets', 0) or 0) * 100,
                'revenue_growth':   (safe('revenueGrowth', 0) or 0) * 100,
                'earnings_growth':  (safe('earningsGrowth', 0) or 0) * 100,
                'pe_ratio':         safe('trailingPE', 25) or 25,
                'peg_ratio':        safe('pegRatio', 2) or 2,
                'price_to_book':    safe('priceToBook', 3) or 3,
                'debt_to_equity':   safe('debtToEquity', 50) or 50,
                'current_ratio':    safe('currentRatio', 1.5) or 1.5,
                'market_cap_billions': (safe('marketCap', 0) or 0) / 1e9,
                'beta': safe('beta', 1.0) or 1.0,
                'dividend_yield':   dividend_info['dividend_yield'],
                'payout_ratio':     dividend_info['payout_ratio'],
                'dividend_growth':  dividend_info['dividend_growth'],
                'has_dividend':     dividend_info['has_dividend'],
                'forward_price_return_6m':    p_ret,
                'forward_dividend_return_6m': d_ret,
                'forward_total_return_6m':    p_ret + d_ret,
            })

        df = pd.DataFrame(data)
        print(f" ✓ {len(df)} samples")
        return df

    except Exception as e:
        print(f" ❌ {e}")
        return None


def collect_longterm_training_data(num_stocks=150):
    """Gather long-term training data across many stocks."""
    print("=" * 80)
    print("STEP 1: COLLECTING LONG-TERM TRAINING DATA (WITH FUNDAMENTALS)")
    print("=" * 80)

    tickers = get_sp500_tickers()[:num_stocks]
    print(f"\nProcessing {len(tickers)} stocks...")

    all_data, successful = [], 0

    for ticker in tickers:
        df = create_longterm_features(ticker)
        if df is not None and len(df) > 0:
            all_data.append(df)
            successful += 1

    if not all_data:
        print("\n❌ No data collected!")
        return None

    combined = pd.concat(all_data, ignore_index=True)
    print(f"\nSuccessfully processed: {successful}/{len(tickers)} stocks")
    print(f"Total training samples: {len(combined):,}")
    print(f"Avg 6-month return:     {combined['forward_total_return_6m'].mean():.2f}%")
    return combined


def train_longterm_model(data, target='forward_total_return_6m'):
    """Train XGBoost for 6-month total return prediction."""
    print("\n" + "=" * 80)
    print("STEP 2: TRAINING LONG-TERM XGBOOST MODEL")
    print("=" * 80)

    feature_cols = [
        'returns_3m', 'returns_6m', 'returns_1y',
        'price_above_sma50', 'price_above_sma200', 'long_term_trend',
        'volatility_90d', 'up_days_pct',
        'profit_margin', 'operating_margin', 'roe', 'roa',
        'revenue_growth', 'earnings_growth',
        'pe_ratio', 'peg_ratio', 'price_to_book',
        'debt_to_equity', 'current_ratio',
        'market_cap_billions', 'beta',
        'dividend_yield', 'payout_ratio', 'dividend_growth', 'has_dividend'
    ]

    X, y = data[feature_cols], data[target]
    mask = ~(X.isna().any(axis=1) | y.isna() | np.isinf(X).any(axis=1))
    X, y = X[mask], y[mask]

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    print(f"Training: {len(X_train):,}  |  Test: {len(X_test):,}")

    model = xgb.XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        objective='reg:squarederror', random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    print(f"\nTrain MAE: {mean_absolute_error(y_train, y_pred_train):.3f}%  "
          f"| Train R²: {r2_score(y_train, y_pred_train):.3f}")
    print(f"Test  MAE: {mean_absolute_error(y_test, y_pred_test):.3f}%  "
          f"| Test  R²: {r2_score(y_test, y_pred_test):.3f}")

    importance = pd.DataFrame({'feature': feature_cols,
                                'importance': model.feature_importances_}
                              ).sort_values('importance', ascending=False)
    fund_feats = {'profit_margin','operating_margin','roe','roa','revenue_growth',
                  'earnings_growth','pe_ratio','peg_ratio','price_to_book',
                  'debt_to_equity','current_ratio','market_cap_billions','beta'}
    div_feats  = {'dividend_yield','payout_ratio','dividend_growth','has_dividend'}

    print("\nTop 15 Features:")
    for _, row in importance.head(15).iterrows():
        tag = " 📊" if row['feature'] in fund_feats else (" 💰" if row['feature'] in div_feats else " 📈")
        print(f"  {row['feature']:<25s}: {row['importance']:.4f}{tag}")

    return model, feature_cols


def get_current_longterm_features(tickers):
    """Fetch current features for the long-term model."""
    print("\n" + "=" * 80)
    print("STEP 3: FETCHING CURRENT LONG-TERM STOCK DATA")
    print("=" * 80)

    current_data, successful = [], 0

    for ticker in tickers:
        try:
            print(f"  Fetching {ticker}...", end='')
            stock = yf.Ticker(ticker)
            hist  = stock.history(period="1y")

            if len(hist) < 120:
                print(" ❌ Not enough data")
                continue

            prices, volumes = hist['Close'], hist['Volume']
            cp  = prices.iloc[-1]
            info = stock.info
            div_info = get_dividend_info(ticker)

            def safe(key, default):
                v = info.get(key, default)
                return default if v is None else v

            rsi = calculate_rsi(prices).iloc[-1]
            macd, macd_sig = calculate_macd(prices)
            sma50  = prices.rolling(50).mean().iloc[-1]
            sma200 = prices.rolling(min(200, len(prices))).mean().iloc[-1]
            avg_vol = volumes.rolling(50).mean().iloc[-1]

            current_data.append({
                'ticker': ticker,
                'sector': safe('sector', 'Unknown'),
                'current_price': cp,
                'market_cap': safe('marketCap', 0),
                'returns_1m': (prices.iloc[-1] / prices.iloc[-22]  - 1) * 100 if len(prices) >= 22  else 0,
                'returns_3m': (prices.iloc[-1] / prices.iloc[-63]  - 1) * 100 if len(prices) >= 63  else 0,
                'returns_6m': (prices.iloc[-1] / prices.iloc[-126] - 1) * 100 if len(prices) >= 126 else 0,
                'returns_1y': (prices.iloc[-1] / prices.iloc[-252] - 1) * 100 if len(prices) >= 252 else 0,
                'rsi': rsi,
                'macd_bullish':     int(macd.iloc[-1] > macd_sig.iloc[-1]),
                'price_above_sma50':  int(cp > sma50),
                'price_above_sma200': int(cp > sma200),
                'long_term_trend':    int(sma50 > sma200),
                'volatility_90d': prices.pct_change().rolling(90).std().iloc[-1] * np.sqrt(252) * 100,
                'volume_ratio':   volumes.iloc[-1] / avg_vol if avg_vol > 0 else 1,
                'up_days_pct':    (prices.diff().iloc[-60:] > 0).sum() / 60 if len(prices) >= 60 else 0.5,
                'profit_margin':    (safe('profitMargins', 0) or 0) * 100,
                'operating_margin': (safe('operatingMargins', 0) or 0) * 100,
                'roe':              (safe('returnOnEquity', 0) or 0) * 100,
                'roa':              (safe('returnOnAssets', 0) or 0) * 100,
                'revenue_growth':   (safe('revenueGrowth', 0) or 0) * 100,
                'earnings_growth':  (safe('earningsGrowth', 0) or 0) * 100,
                'pe_ratio':         safe('trailingPE', 25) or 25,
                'peg_ratio':        safe('pegRatio', 2) or 2,
                'price_to_book':    safe('priceToBook', 3) or 3,
                'debt_to_equity':   safe('debtToEquity', 50) or 50,
                'current_ratio':    safe('currentRatio', 1.5) or 1.5,
                'market_cap_billions': (safe('marketCap', 0) or 0) / 1e9,
                'beta': safe('beta', 1.0) or 1.0,
                'dividend_yield':  div_info['dividend_yield'],
                'payout_ratio':    div_info['payout_ratio'],
                'dividend_growth': div_info['dividend_growth'],
                'has_dividend':    div_info['has_dividend'],
            })
            successful += 1
            print(" ✓")
        except Exception as e:
            print(f" ❌ {e}")

    print(f"\nFetched: {successful}/{len(tickers)} stocks")
    return pd.DataFrame(current_data)


def optimize_longterm_portfolio(model, feature_cols, budget=10000,
                                 stocks_per_sector=2, max_sector_pct=25):
    """Sector-balanced sequential allocation for the long-term portfolio."""
    tickers = get_sp500_tickers()[:150]
    current_data = get_current_longterm_features(tickers)

    if current_data.empty:
        print("\n❌ No current data collected!")
        return None, None

    print("\n" + "=" * 80)
    print(f"STEP 4: OPTIMIZING ${budget:,} LONG-TERM PORTFOLIO")
    print("=" * 80)

    X = current_data[feature_cols]
    current_data = current_data.copy()
    current_data['predicted_6m_return']       = model.predict(X)
    current_data['estimated_dividend_return']  = current_data['dividend_yield'] / 2
    current_data['estimated_price_return']     = (current_data['predicted_6m_return']
                                                   - current_data['estimated_dividend_return'])
    current_data['quality_score'] = (
        current_data['roe'] * 0.3 +
        current_data['profit_margin'] * 0.3 +
        current_data['revenue_growth'] * 0.2 +
        (100 - current_data['debt_to_equity']) * 0.2
    )

    # Top 20 preview
    print("\n" + "=" * 80)
    print("TOP 20 STOCKS BY PREDICTED 6-MONTH RETURN")
    print("=" * 80)
    print(f"{'Rank':<6}{'Ticker':<8}{'Sector':<22}{'6M Ret%':<10}{'Div%':<8}{'ROE%':<8}")
    print("-" * 65)
    for i, (_, row) in enumerate(current_data.nlargest(20, 'predicted_6m_return').iterrows(), 1):
        print(f"{i:<6}{row['ticker']:<8}{row['sector']:<22}"
              f"{row['predicted_6m_return']:>9.2f} {row['dividend_yield']:>7.2f} {row['roe']:>7.1f}")

    # Sector-balanced allocation
    allocations   = []
    remaining     = budget
    sector_alloc  = {}
    sector_counts = {}
    max_sector_dollars = budget * (max_sector_pct / 100)

    current_data = current_data.sort_values(['sector', 'predicted_6m_return'],
                                             ascending=[True, False])

    for _, row in current_data.iterrows():
        sector = row['sector']
        if sector_counts.get(sector, 0) >= stocks_per_sector:
            continue
        if sector_alloc.get(sector, 0) >= max_sector_dollars:
            continue
        if remaining < row['current_price']:
            continue

        available = min(max_sector_dollars - sector_alloc.get(sector, 0), remaining)
        shares = int(available / row['current_price'])
        if shares < 1:
            continue

        inv = shares * row['current_price']
        remaining -= inv
        sector_alloc[sector]  = sector_alloc.get(sector, 0) + inv
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

        allocations.append({
            'ticker': row['ticker'], 'sector': sector,
            'shares': shares, 'price': row['current_price'],
            'investment': inv, 'weight': inv / budget * 100,
            'predicted_6m_return':       row['predicted_6m_return'],
            'predicted_price_return':    row['estimated_price_return'],
            'predicted_dividend_return': row['estimated_dividend_return'],
            'dividend_yield': row['dividend_yield'],
            'roe': row['roe'], 'profit_margin': row['profit_margin'],
            'pe_ratio': row['pe_ratio'],
            'volatility': row['volatility_90d'],
            'quality_score': row['quality_score'],
        })

    if not allocations:
        print("\n❌ Could not allocate any stocks!")
        return None, None

    alloc_df       = pd.DataFrame(allocations)
    total_invested = budget - remaining

    p_ret    = sum(a['investment'] * a['predicted_6m_return'] for a in allocations) / total_invested
    port_vol = np.sqrt(sum((a['investment'] / total_invested * a['volatility'])**2
                           for a in allocations))
    sector_breakdown = alloc_df.groupby('sector')['investment'].sum().sort_values(ascending=False)

    print("\n" + "=" * 80)
    print("PORTFOLIO SUMMARY (LONG-TERM)")
    print("=" * 80)
    print(f"Invested:            ${total_invested:>10,.2f}  ({total_invested/budget*100:.1f}%)")
    print(f"Positions:           {len(allocations)}")
    print(f"Sectors:             {len(sector_breakdown)}")
    print(f"Expected 6M Return:  {p_ret:.2f}%")
    print(f"Portfolio Volatility:{port_vol:.2f}%")
    print(f"Sharpe-like Ratio:   {p_ret/port_vol:.3f}")

    print("\n" + "=" * 80)
    print("SECTOR BREAKDOWN")
    print("=" * 80)
    for sector, amt in sector_breakdown.items():
        n = len(alloc_df[alloc_df['sector'] == sector])
        print(f"  {sector:<25s}: ${amt:>8,.2f}  ({amt/total_invested*100:.1f}%) — {n} stocks")

    _plot_portfolio(alloc_df, total_invested, budget, port_vol, sector_breakdown,
                    prefix='longterm')

    current_data.to_csv('longterm_predictions.csv', index=False)
    alloc_df.to_csv('longterm_portfolio_allocation.csv', index=False)
    print("\n✓ Saved: longterm_predictions.csv, longterm_portfolio_allocation.csv")

    return current_data, allocations

print("✓ Long-term model functions defined")



# ============================================================================
# RUN LONG-TERM MODEL  (budget is a variable — change as needed)
# ============================================================================
LONGTERM_BUDGET = 10000  # ← adjust as needed

training_data_lt = collect_longterm_training_data(num_stocks=150)

if training_data_lt is not None:
    training_data_lt.to_csv('longterm_training_data.csv', index=False)
    print("\n✓ Training data saved to longterm_training_data.csv")

    longterm_model, longterm_features = train_longterm_model(training_data_lt)

    # Save model for use by the backtest cell below
    with open('longterm_model.pkl', 'wb') as f:
        pickle.dump({'model': longterm_model, 'feature_cols': longterm_features}, f)
    print("✓ Model saved to longterm_model.pkl")

    predictions_lt, allocations_lt = optimize_longterm_portfolio(
        longterm_model, longterm_features, budget=LONGTERM_BUDGET
    )
else:
    print("❌ Long-term training failed. Check your internet connection / yfinance access.")



# ==============================================================================
# ## Pipeline 3 — 2-Year Monthly Backtest\n\n> **Prerequisite:** Run the Long-Term Model cell above first to create `longterm_model.pkl`.\n\nWalk-forward backtest: each of the past 24 months, score all stocks as-of that date, invest in the top 10, record actual results.
# ==============================================================================

# ============================================================================
# 2-YEAR MONTHLY BACKTEST
# Requires: longterm_model.pkl (produced by the long-term model cell above)
# ============================================================================

def backtest_2years_monthly(model, feature_cols, budget=10000, top_n_picks=10):
    """
    Walk-forward backtest over 24 months.
    Each month: score all S&P 500 stocks with the long-term model,
    pick the top N, invest equal-weight, record actual 1-month returns.
    """
    print("=" * 90)
    print(" " * 20 + "2-YEAR MONTHLY BACKTEST (24 MONTHS)")
    print("=" * 90)
    print(f"Strategy : top {top_n_picks} stocks per month | equal-weight ${budget:,}")
    print("=" * 90)

    today          = datetime.now()
    num_months     = 24
    monthly_results, all_picks = [], []
    cumulative_val = budget

    for month_back in range(num_months, 0, -1):
        pred_date = today - relativedelta(months=month_back)
        eval_date = today - relativedelta(months=month_back - 1)

        print(f"\n{'='*90}")
        print(f"MONTH {num_months - month_back + 1}/24 — {pred_date.strftime('%B %Y')}")
        print(f"{'='*90}")

        tickers = get_sp500_tickers()
        month_preds = []

        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist  = stock.history(start=pred_date - timedelta(days=400),
                                      end=eval_date + timedelta(days=5))

                if len(hist) < 150:
                    continue

                pred_idx = eval_idx = None
                for i, date in enumerate(hist.index):
                    if date.date() >= pred_date.date() and pred_idx is None:
                        pred_idx = i
                    if date.date() >= eval_date.date() and eval_idx is None:
                        eval_idx = i
                        break

                if pred_idx is None or eval_idx is None or pred_idx < 120:
                    continue

                p_start = hist['Close'].iloc[pred_idx]
                p_end   = hist['Close'].iloc[eval_idx]

                try:
                    div_paid = (stock.dividends.reindex(hist.index, fill_value=0)
                                    .iloc[pred_idx:eval_idx + 1].sum())
                except Exception:
                    div_paid = 0

                actual_p   = (p_end / p_start - 1) * 100
                actual_d   = (div_paid / p_start) * 100
                actual_tot = actual_p + actual_d

                hp = hist.iloc[:pred_idx + 1]
                prices, volumes = hp['Close'], hp['Volume']

                if len(prices) < 120:
                    continue

                rsi = calculate_rsi(prices).iloc[-1]
                macd, macd_sig = calculate_macd(prices)
                sma50  = prices.rolling(50).mean().iloc[-1]
                sma200 = prices.rolling(min(200, len(prices))).mean().iloc[-1]
                avg_vol = volumes.rolling(50).mean().iloc[-1]
                info = stock.info
                div_info = get_dividend_info(ticker)

                def safe(key, default):
                    v = info.get(key, default)
                    return default if v is None else v

                features = {
                    'returns_1m': (prices.iloc[-1] / prices.iloc[-22]  - 1) * 100 if len(prices) >= 22  else 0,
                    'returns_3m': (prices.iloc[-1] / prices.iloc[-63]  - 1) * 100 if len(prices) >= 63  else 0,
                    'returns_6m': (prices.iloc[-1] / prices.iloc[-126] - 1) * 100 if len(prices) >= 126 else 0,
                    'returns_1y': (prices.iloc[-1] / prices.iloc[-252] - 1) * 100 if len(prices) >= 252 else 0,
                    'rsi': rsi,
                    'macd_bullish':     int(macd.iloc[-1] > macd_sig.iloc[-1]),
                    'price_above_sma50':  int(prices.iloc[-1] > sma50),
                    'price_above_sma200': int(prices.iloc[-1] > sma200),
                    'long_term_trend':    int(sma50 > sma200),
                    'volatility_90d': prices.pct_change().rolling(90).std().iloc[-1] * np.sqrt(252) * 100,
                    'volume_ratio':   volumes.iloc[-1] / avg_vol if avg_vol > 0 else 1,
                    'up_days_pct':    (prices.diff().iloc[-60:] > 0).sum() / 60,
                    'profit_margin':    (safe('profitMargins', 0) or 0) * 100,
                    'operating_margin': (safe('operatingMargins', 0) or 0) * 100,
                    'roe':              (safe('returnOnEquity', 0) or 0) * 100,
                    'roa':              (safe('returnOnAssets', 0) or 0) * 100,
                    'revenue_growth':   (safe('revenueGrowth', 0) or 0) * 100,
                    'earnings_growth':  (safe('earningsGrowth', 0) or 0) * 100,
                    'pe_ratio':         safe('trailingPE', 25) or 25,
                    'peg_ratio':        safe('pegRatio', 2) or 2,
                    'price_to_book':    safe('priceToBook', 3) or 3,
                    'debt_to_equity':   safe('debtToEquity', 50) or 50,
                    'current_ratio':    safe('currentRatio', 1.5) or 1.5,
                    'market_cap_billions': (safe('marketCap', 0) or 0) / 1e9,
                    'beta': safe('beta', 1.0) or 1.0,
                    'dividend_yield':  div_info['dividend_yield'],
                    'payout_ratio':    div_info['payout_ratio'],
                    'dividend_growth': div_info['dividend_growth'],
                    'has_dividend':    div_info['has_dividend'],
                }

                pred = model.predict(pd.DataFrame([features])[feature_cols])[0]

                month_preds.append({
                    'ticker': ticker,
                    'sector': safe('sector', 'Unknown'),
                    'price_start': p_start, 'price_end': p_end,
                    'predicted_return': pred,
                    'actual_return': actual_tot,
                    'price_return': actual_p,
                    'dividend_return': actual_d,
                    'error': abs(pred - actual_tot),
                    'direction_correct': (pred > 0) == (actual_tot > 0),
                })

            except Exception:
                continue

        if not month_preds:
            print("⚠️  No valid predictions this month — skipping")
            continue

        month_df  = pd.DataFrame(month_preds)
        top_picks = month_df.nlargest(top_n_picks, 'predicted_return')

        print(f"\n{'#':<4}{'Ticker':<8}{'Sector':<20}{'Start$':<10}{'End$':<10}"
              f"{'Predicted%':<12}{'ACTUAL%':<12}{'Diff%'}")
        print("-" * 88)
        for i, (_, row) in enumerate(top_picks.iterrows(), 1):
            print(f"{i:<4}{row['ticker']:<8}{row['sector'][:18]:<20}"
                  f"${row['price_start']:>8.2f} ${row['price_end']:>8.2f} "
                  f"{row['predicted_return']:>10.2f}% "
                  f"{row['actual_return']:>10.2f}% "
                  f"{row['actual_return'] - row['predicted_return']:>8.2f}%")

            all_picks.append({
                'month':      num_months - month_back + 1,
                'month_name': pred_date.strftime('%B %Y'),
                'rank': i,
                'ticker': row['ticker'], 'sector': row['sector'],
                'price_start': row['price_start'], 'price_end': row['price_end'],
                'predicted_return': row['predicted_return'],
                'actual_return': row['actual_return'],
                'price_return': row['price_return'],
                'dividend_return': row['dividend_return'],
                'difference': row['actual_return'] - row['predicted_return'],
                'direction_correct': row['direction_correct'],
            })

        inv_per = budget / top_n_picks
        month_val = sum(inv_per * (1 + row['actual_return'] / 100)
                        for _, row in top_picks.iterrows())
        month_pct = (month_val / budget - 1) * 100
        cumulative_val *= (1 + month_pct / 100)

        print(f"\nActual Return: {top_picks['actual_return'].mean():+.2f}%  "
              f"| Dir Acc: {top_picks['direction_correct'].mean()*100:.1f}%  "
              f"| Cumulative: ${cumulative_val:,.2f}")

        monthly_results.append({
            'month':      num_months - month_back + 1,
            'month_name': pred_date.strftime('%B %Y'),
            'date':       pred_date.strftime('%Y-%m-%d'),
            'predicted_return':   top_picks['predicted_return'].mean(),
            'actual_return':      top_picks['actual_return'].mean(),
            'prediction_error':   top_picks['error'].mean(),
            'direction_accuracy': top_picks['direction_correct'].mean() * 100,
            'made_profit':        month_pct > 0,
            'profit_dollars':     month_val - budget,
            'ending_value':       month_val,
            'cumulative_value':   cumulative_val,
        })

    if not monthly_results:
        print("\n❌ No valid monthly results")
        return None

    results_df = pd.DataFrame(monthly_results)
    picks_df   = pd.DataFrame(all_picks)

    total_ret  = (cumulative_val / budget - 1) * 100
    win_rate   = results_df['made_profit'].mean() * 100
    avg_mret   = results_df['actual_return'].mean()
    mae        = results_df['prediction_error'].mean()
    dir_acc    = picks_df['direction_correct'].mean() * 100

    print(f"\n\n{'='*90}")
    print(" " * 30 + "2-YEAR BACKTEST SUMMARY")
    print(f"{'='*90}")
    print(f"Starting Value:          ${budget:>10,.2f}")
    print(f"Final Value:             ${cumulative_val:>10,.2f}")
    print(f"Total Return (24 mo):    {total_ret:>10.2f}%")
    print(f"Annualised Return:       {avg_mret * 12:>10.2f}%")
    print(f"Win Rate:                {win_rate:>10.1f}%")
    print(f"Avg Prediction MAE:      {mae:>10.2f}%")
    print(f"Direction Accuracy:      {dir_acc:>10.1f}%")

    # S&P 500 benchmark
    try:
        spy = yf.Ticker('SPY').history(
            start=datetime.now() - relativedelta(months=24), end=datetime.now())
        if len(spy) > 0:
            spy_ret = (spy['Close'].iloc[-1] / spy['Close'].iloc[0] - 1) * 100
            print(f"\nS&P 500 Return (24 mo):  {spy_ret:>10.2f}%")
            print(f"Alpha:                   {total_ret - spy_ret:>10.2f}%")
            if total_ret > spy_ret:
                print(f"\n✓ Model beat S&P 500 by {total_ret - spy_ret:.2f}%!")
            else:
                print(f"\n✗ Model underperformed S&P 500 by {spy_ret - total_ret:.2f}%")
    except Exception:
        pass

    # Month-by-month table
    print(f"\n{'Month':<8}{'Date':<12}{'Predicted%':<12}{'Actual%':<12}"
          f"{'Profit$':<14}{'Cumulative$':<15}{'Result'}")
    print("-" * 85)
    for _, r in results_df.iterrows():
        tag = "✓ PROFIT" if r['made_profit'] else "✗ LOSS"
        print(f"{r['month']:<8}{r['date']:<12}"
              f"{r['predicted_return']:>10.2f}% {r['actual_return']:>10.2f}% "
              f"${r['profit_dollars']:>12,.2f} ${r['cumulative_value']:>13,.2f}  {tag}")

    results_df.to_csv('backtest_monthly_summary.csv', index=False)
    picks_df.to_csv('backtest_all_picks.csv', index=False)
    print("\n✓ Saved: backtest_monthly_summary.csv, backtest_all_picks.csv")

    return {
        'monthly_summary': results_df, 'all_picks': picks_df,
        'total_return': total_ret, 'final_value': cumulative_val,
        'win_rate': win_rate, 'direction_accuracy': dir_acc, 'mae': mae,
    }

print("✓ Backtest function defined")



# ============================================================================
# RUN 2-YEAR BACKTEST
# Loads longterm_model.pkl — run the long-term model cell first!
# ============================================================================

if not os.path.exists('longterm_model.pkl'):
    print("⚠️  longterm_model.pkl not found.")
    print("Please run the Long-Term Model cell above first to train and save the model.")
else:
    with open('longterm_model.pkl', 'rb') as f:
        saved = pickle.load(f)
    bt_model    = saved['model']
    bt_features = saved['feature_cols']
    print("✓ Loaded longterm_model.pkl")

    backtest_results = backtest_2years_monthly(
        model=bt_model,
        feature_cols=bt_features,
        budget=10000,
        top_n_picks=10
    )


