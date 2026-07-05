from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import json
import os

# ── Default arguments for all tasks ──────────────────────────────────────────
# These apply to every task in the DAG unless overridden individually
default_args = {
    'owner': 'shane',
    'depends_on_past': False,        # each run is independent
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,                    # retry once if a task fails
    'retry_delay': timedelta(minutes=2),
}

# ── Define the DAG ────────────────────────────────────────────────────────────
dag = DAG(
    'stock_anomaly_pipeline',        # unique name shown in Airflow UI
    default_args=default_args,
    description='Fetch stock data, compute features, detect anomalies',
    schedule_interval=timedelta(hours=1),   # run every hour
    start_date=datetime(2026, 1, 1),
    catchup=False,                   # don't backfill missed runs
    tags=['stocks', 'anomaly', 'ml'],
)

# ── Task 1: Fetch latest stock data ──────────────────────────────────────────
def fetch_data(**context):
    """
    Fetches the latest 5 days of stock data for AAPL, GOOGL, MSFT.
    Saves raw data to a JSON file for the next task to pick up.
    Uses Airflow's XCom (cross-communication) to pass the file path forward.
    """
    STOCKS = ['AAPL', 'GOOGL', 'MSFT']
    all_data = []

    for ticker in STOCKS:
        df = yf.Ticker(ticker).history(period="5d", interval="1h")
        df['ticker'] = ticker
        df = df[['Close', 'Volume', 'ticker']].copy()
        df.columns = ['close', 'volume', 'ticker']
        df = df.reset_index()
        df['datetime'] = df['Datetime'].astype(str)
        df = df[['datetime', 'close', 'volume', 'ticker']]
        all_data.extend(df.to_dict('records'))

    # Save to a shared location all tasks can access
    output_path = '/opt/airflow/model/raw_data.json'
    with open(output_path, 'w') as f:
        json.dump(all_data, f)

    print(f"Fetched {len(all_data)} rows. Saved to {output_path}")
    # Push the path via XCom so the next task knows where to find it
    context['ti'].xcom_push(key='raw_data_path', value=output_path)

# ── Task 2: Compute features ─────────────────────────────────────────────────
def compute_features(**context):
    """
    Loads raw data, computes rolling average, volatility, pct_change,
    and volume z-score — the same features the model was trained on.
    """
    # Pull the file path from the previous task via XCom
    raw_path = context['ti'].xcom_pull(
        task_ids='fetch_data', key='raw_data_path'
    )

    with open(raw_path, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df['close'] = pd.to_numeric(df['close'])
    df['volume'] = pd.to_numeric(df['volume'])

    feature_dfs = []
    for ticker in df['ticker'].unique():
        tdf = df[df['ticker'] == ticker].copy().sort_values('datetime')
        tdf['rolling_avg_close'] = tdf['close'].rolling(window=5, min_periods=1).mean()
        tdf['volatility'] = tdf['close'].rolling(window=5, min_periods=1).std().fillna(0)
        tdf['pct_change'] = tdf['close'].pct_change().fillna(0)
        vol_mean = tdf['volume'].rolling(20, min_periods=1).mean()
        vol_std = tdf['volume'].rolling(20, min_periods=1).std().fillna(1)
        tdf['volume_zscore'] = ((tdf['volume'] - vol_mean) / vol_std).fillna(0)
        feature_dfs.append(tdf)

    features_df = pd.concat(feature_dfs)
    output_path = '/opt/airflow/model/features.json'
    features_df.to_json(output_path, orient='records')

    print(f"Features computed for {len(features_df)} rows.")
    context['ti'].xcom_push(key='features_path', value=output_path)

# ── Task 3: Detect anomalies ──────────────────────────────────────────────────
def detect_anomalies(**context):
    """
    Loads the trained Isolation Forest model and scaler,
    runs predictions on the feature data, flags anomalies.
    """
    features_path = context['ti'].xcom_pull(
        task_ids='compute_features', key='features_path'
    )

    df = pd.read_json(features_path)

    # Load the trained model and scaler
    model = joblib.load('/opt/airflow/model/isolation_forest.pkl')
    scaler = joblib.load('/opt/airflow/model/scaler.pkl')

    feature_cols = ['close', 'rolling_avg_close', 'volatility', 'pct_change', 'volume_zscore']
    X = df[feature_cols].values
    X_scaled = scaler.transform(X)

    df['anomaly'] = model.predict(X_scaled)
    df['anomaly_score'] = model.score_samples(X_scaled)
    df['is_anomaly'] = df['anomaly'] == -1

    n_anomalies = df['is_anomaly'].sum()
    print(f"Detected {n_anomalies} anomalies out of {len(df)} data points.")

    output_path = '/opt/airflow/model/anomaly_results.json'
    df.to_json(output_path, orient='records')
    context['ti'].xcom_push(key='results_path', value=output_path)

# ── Task 4: Save results ──────────────────────────────────────────────────────
def save_results(**context):
    """
    Loads anomaly results, prints a summary, and saves a final report.
    In production this would write to a database or send an alert.
    """
    results_path = context['ti'].xcom_pull(
        task_ids='detect_anomalies', key='results_path'
    )

    df = pd.read_json(results_path)
    anomalies = df[df['is_anomaly'] == True]

    summary = {
        'run_timestamp': datetime.now().isoformat(),
        'total_data_points': len(df),
        'anomalies_detected': len(anomalies),
        'anomaly_rate': round(len(anomalies) / len(df) * 100, 2),
        'anomalies': anomalies[['ticker', 'datetime', 'close', 'anomaly_score']].to_dict('records')
    }

    report_path = '/opt/airflow/model/pipeline_report.json'
    with open(report_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Pipeline complete.")
    print(f"Total points: {summary['total_data_points']}")
    print(f"Anomalies: {summary['anomalies_detected']} ({summary['anomaly_rate']}%)")
    print(f"Report saved to {report_path}")

# ── Wire up the tasks into a DAG ──────────────────────────────────────────────
# This defines the order: fetch → features → anomalies → save
t1 = PythonOperator(task_id='fetch_data',         python_callable=fetch_data,         dag=dag, provide_context=True)
t2 = PythonOperator(task_id='compute_features',   python_callable=compute_features,   dag=dag, provide_context=True)
t3 = PythonOperator(task_id='detect_anomalies',   python_callable=detect_anomalies,   dag=dag, provide_context=True)
t4 = PythonOperator(task_id='save_results',       python_callable=save_results,       dag=dag, provide_context=True)

t1 >> t2 >> t3 >> t4