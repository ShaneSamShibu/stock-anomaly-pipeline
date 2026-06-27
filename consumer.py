import json
from kafka import KafkaConsumer

TOPIC = 'stock-prices'

consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='earliest',
    group_id='stock-group'
)

print("Consumer listening...")

for message in consumer:
    data = message.value
    print(f"[{data['timestamp']}] {data['ticker']} | Close: {data['close']} | Volume: {data['volume']}")