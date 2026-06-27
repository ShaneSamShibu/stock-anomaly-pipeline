import json
import time
import yfinance as yf
from kafka import KafkaProducer
from datetime import datetime

# Connect to Kafka
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

TOPIC = 'stock-prices'
STOCKS = ['AAPL', 'GOOGL', 'MSFT']  # add any tickers you want

print("Producer started. Streaming stock data...")

while True:
    for ticker in STOCKS:
        stock = yf.Ticker(ticker)
        data = stock.history(period='1d', interval='1m')

        if not data.empty:
            latest = data.iloc[-1]
            message = {
                'ticker': ticker,
                'timestamp': datetime.now().isoformat(),
                'open': round(latest['Open'], 4),
                'high': round(latest['High'], 4),
                'low': round(latest['Low'], 4),
                'close': round(latest['Close'], 4),
                'volume': int(latest['Volume'])
            }
            producer.send(TOPIC, value=message)
            print(f"Sent: {message}")

    time.sleep(60)  # fetch every 60 seconds