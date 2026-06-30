import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
import os

# ── 1. Fetch historical data ──────────────────────────────────────────────────
# 2 years of daily data for our 3 stocks
STOCKS = ['AAPL', 'GOOGL', 'MSFT']
print("Fetching historical data...")

all_data = []
for ticker in STOCKS:
    df = yf.Ticker(ticker).history(period="2y", interval="1d")
    df['ticker'] = ticker
    df = df[['Close', 'Volume', 'ticker']].copy()
    df.columns = ['close', 'volume', 'ticker']
    all_data.append(df)

data = pd.concat(all_data)
data = data.reset_index()
print(f"Fetched {len(data)} rows across {len(STOCKS)} stocks.")

# ── 2. Engineer features ──────────────────────────────────────────────────────
# These match what Spark computes live: rolling average and volatility
# We compute per-ticker to avoid mixing AAPL and GOOGL numbers together
print("Engineering features...")

feature_dfs = []
for ticker in STOCKS:
    df = data[data['ticker'] == ticker].copy().sort_values('Date')

    # 5-day rolling average of closing price
    df['rolling_avg_close'] = df['close'].rolling(window=5).mean()

    # 5-day rolling standard deviation (volatility)
    df['volatility'] = df['close'].rolling(window=5).std()

    # Percentage change from yesterday's close
    df['pct_change'] = df['close'].pct_change()

    # Volume Z-score: how unusual is today's volume compared to recent norms?
    df['volume_zscore'] = (
        (df['volume'] - df['volume'].rolling(20).mean())
        / df['volume'].rolling(20).std()
    )

    feature_dfs.append(df)

data = pd.concat(feature_dfs)

# Drop rows with NaN values (they appear at the start of each rolling window)
data = data.dropna()
print(f"{len(data)} rows after dropping NaN from rolling windows.")

# ── 3. Prepare feature matrix ─────────────────────────────────────────────────
features = ['close', 'rolling_avg_close', 'volatility', 'pct_change', 'volume_zscore']
X = data[features].values

# Scale the features so no single one dominates just because of its units
# (e.g., raw close price in hundreds vs pct_change in decimals)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 4. Train the Isolation Forest ─────────────────────────────────────────────
# contamination=0.05 means "assume ~5% of data points are anomalies"
print("Training Isolation Forest model...")
model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)
model.fit(X_scaled)

# Get predictions: -1 = anomaly, 1 = normal
data['anomaly'] = model.predict(X_scaled)
data['anomaly_score'] = model.score_samples(X_scaled)

n_anomalies = (data['anomaly'] == -1).sum()
print(f"Training complete. Flagged {n_anomalies} anomalies out of {len(data)} data points ({n_anomalies/len(data)*100:.1f}%).")

# ── 5. Save the model and scaler ──────────────────────────────────────────────
# We save both — the scaler must be applied to any new data before predicting
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/isolation_forest.pkl")
joblib.dump(scaler, "model/scaler.pkl")
print("Model saved to model/isolation_forest.pkl")
print("Scaler saved to model/scaler.pkl")

# ── 6. Plot anomalies for AAPL (visual sanity check) ─────────────────────────
print("Generating anomaly plot...")
aapl = data[data['ticker'] == 'AAPL'].copy()
normal = aapl[aapl['anomaly'] == 1]
anomalies = aapl[aapl['anomaly'] == -1]

plt.figure(figsize=(14, 5))
plt.plot(normal['Date'], normal['close'], color='steelblue', label='Normal', linewidth=1)
plt.scatter(anomalies['Date'], anomalies['close'], color='red', label='Anomaly', s=40, zorder=5)
plt.title('AAPL — Anomaly Detection (Isolation Forest)', fontsize=13)
plt.xlabel('Date')
plt.ylabel('Closing Price (USD)')
plt.legend()
plt.tight_layout()
plt.savefig("model/aapl_anomalies.png", dpi=150)
plt.close()
print("Plot saved to model/aapl_anomalies.png")
print("\nWeek 3 complete!")