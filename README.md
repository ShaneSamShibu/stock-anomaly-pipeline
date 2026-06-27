\# Real-Time Stock Data Pipeline (Kafka)



A real-time data pipeline that streams live stock prices using Apache Kafka,

demonstrating production-style data engineering patterns.



\## Architecture

Yahoo Finance API → Producer (Python) → Kafka Topic → Consumer (Python)



\## Tech Stack

\- Apache Kafka (via Docker) — message streaming

\- Python (`yfinance`, `kafka-python`) — data ingestion

\- Docker Compose — container orchestration



\## What it does

\- Producer fetches live OHLCV (Open/High/Low/Close/Volume) data for AAPL, GOOGL,

&#x20; and MSFT every 60 seconds and publishes it to a Kafka topic.

\- Consumer subscribes to that topic and processes incoming messages in real time.

\- Decouples data ingestion from data processing, mirroring real-world

&#x20; streaming architectures (e.g., used at companies like Netflix, Uber).



\## How to run

1\. `docker-compose up -d` — starts Kafka + Zookeeper

2\. `python producer.py` — starts streaming stock data

3\. `python consumer.py` — starts consuming the stream



\## Coming next

\- PySpark consumer for real-time feature engineering

\- Anomaly detection model

\- AWS SageMaker deployment

\- Streamlit dashboard

