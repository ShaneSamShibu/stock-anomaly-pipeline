import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import joblib
from datetime import datetime, timedelta

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Anomaly Dashboard",
    page_icon="📈",
    layout="wide"
)

# ── Load model and scaler ─────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("model/isolation_forest.pkl")
    scaler = joblib.load("model/scaler.pkl")
    return model, scaler

model, scaler = load_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
ticker = st.sidebar.selectbox(
    "Select Stock",
    ["AAPL", "GOOGL", "MSFT"],
    index=0
)
period = st.sidebar.selectbox(
    "Time Period",
    ["1mo", "3mo", "6mo", "1y", "2y"],
    index=2
)
contamination_info = st.sidebar.info(
    "Model flags ~5% of data points as anomalies based on patterns learned from 2 years of historical data."
)

# ── Fetch and process data ────────────────────────────────────────────────────
@st.cache_data(ttl=300)  # cache for 5 minutes, then auto-refresh
def fetch_and_predict(ticker, period):
    # Fetch data
    df = yf.Ticker(ticker).history(period=period, interval="1d")
    df = df[['Close', 'Volume']].copy()
    df.columns = ['close', 'volume']
    df = df.reset_index()
    df['date'] = pd.to_datetime(df['Date']).dt.date

    # Engineer features (same as training)
    df['rolling_avg_close'] = df['close'].rolling(window=5, min_periods=1).mean()
    df['volatility'] = df['close'].rolling(window=5, min_periods=1).std().fillna(0)
    df['pct_change'] = df['close'].pct_change().fillna(0)
    vol_mean = df['volume'].rolling(20, min_periods=1).mean()
    vol_std = df['volume'].rolling(20, min_periods=1).std().fillna(1)
    df['volume_zscore'] = ((df['volume'] - vol_mean) / vol_std).fillna(0)

    # Run model
    features = ['close', 'rolling_avg_close', 'volatility', 'pct_change', 'volume_zscore']
    X = df[features].values
    X_scaled = scaler.transform(X)
    df['anomaly'] = model.predict(X_scaled)
    df['anomaly_score'] = model.score_samples(X_scaled)
    df['is_anomaly'] = df['anomaly'] == -1

    return df

df = fetch_and_predict(ticker, period)

# ── Main dashboard ────────────────────────────────────────────────────────────
st.title("📈 Real-Time Stock Anomaly Detection")
st.caption(f"Powered by Isolation Forest | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ── Metrics row ───────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Current Price",
        value=f"${df['close'].iloc[-1]:.2f}",
        delta=f"{df['pct_change'].iloc[-1]*100:.2f}%"
    )
with col2:
    st.metric(
        label="Anomalies Detected",
        value=df['is_anomaly'].sum(),
        delta=f"{df['is_anomaly'].mean()*100:.1f}% of period"
    )
with col3:
    st.metric(
        label="Rolling Avg (5d)",
        value=f"${df['rolling_avg_close'].iloc[-1]:.2f}"
    )
with col4:
    st.metric(
        label="Volatility (5d std)",
        value=f"${df['volatility'].iloc[-1]:.2f}"
    )

st.divider()

# ── Price chart with anomalies ────────────────────────────────────────────────
st.subheader(f"{ticker} — Price History with Anomaly Flags")

normal = df[~df['is_anomaly']]
anomalies = df[df['is_anomaly']]

fig = go.Figure()

# Normal price line
fig.add_trace(go.Scatter(
    x=normal['date'],
    y=normal['close'],
    mode='lines',
    name='Normal',
    line=dict(color='steelblue', width=2)
))

# Rolling average line
fig.add_trace(go.Scatter(
    x=df['date'],
    y=df['rolling_avg_close'],
    mode='lines',
    name='5-day Rolling Avg',
    line=dict(color='orange', width=1, dash='dash'),
    opacity=0.7
))

# Anomaly dots
fig.add_trace(go.Scatter(
    x=anomalies['date'],
    y=anomalies['close'],
    mode='markers',
    name='Anomaly',
    marker=dict(color='red', size=10, symbol='circle', line=dict(color='darkred', width=1)),
    hovertemplate='<b>Anomaly</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
))

fig.update_layout(
    height=450,
    hovermode='x unified',
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    xaxis_title='Date',
    yaxis_title='Price (USD)',
    plot_bgcolor='white',
    paper_bgcolor='white',
    xaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
    yaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
)

st.plotly_chart(fig, use_container_width=True)

# ── Anomaly score chart ───────────────────────────────────────────────────────
st.subheader("Anomaly Score Over Time")
st.caption("More negative = more anomalous. The model flags the most negative scores as anomalies.")

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=df['date'],
    y=df['anomaly_score'],
    mode='lines',
    fill='tozeroy',
    name='Anomaly Score',
    line=dict(color='purple', width=1.5),
    fillcolor='rgba(128,0,128,0.1)'
))
fig2.add_hline(
    y=df['anomaly_score'].quantile(0.05),
    line_dash='dash',
    line_color='red',
    annotation_text='Anomaly threshold',
    annotation_position='top right'
)
fig2.update_layout(
    height=250,
    plot_bgcolor='white',
    paper_bgcolor='white',
    xaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
    yaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
    showlegend=False
)
st.plotly_chart(fig2, use_container_width=True)

# ── Anomaly table ─────────────────────────────────────────────────────────────
st.subheader("🚨 Flagged Anomalies")
if len(anomalies) > 0:
    display_df = anomalies[['date', 'close', 'rolling_avg_close', 'volatility', 'anomaly_score']].copy()
    display_df.columns = ['Date', 'Close Price', '5d Avg', 'Volatility', 'Anomaly Score']
    display_df['Close Price'] = display_df['Close Price'].apply(lambda x: f"${x:.2f}")
    display_df['5d Avg'] = display_df['5d Avg'].apply(lambda x: f"${x:.2f}")
    display_df['Volatility'] = display_df['Volatility'].apply(lambda x: f"${x:.2f}")
    display_df['Anomaly Score'] = display_df['Anomaly Score'].apply(lambda x: f"{x:.4f}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)
else:
    st.success("No anomalies detected in the selected period.")

# ── Refresh button ────────────────────────────────────────────────────────────
st.divider()
if st.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

st.caption("Data source: Yahoo Finance | Model: Isolation Forest trained on 2 years of historical data")