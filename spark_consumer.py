import os
os.environ["HADOOP_HOME"] = "C:\\hadoop"
os.environ["PATH"] = "C:\\hadoop\\bin;" + os.environ["PATH"]

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, window, avg, stddev
from pyspark.sql.types import StructType, StringType, DoubleType, LongType, TimestampType

# Define the schema matching our producer's message format
schema = StructType() \
    .add("ticker", StringType()) \
    .add("timestamp", StringType()) \
    .add("open", DoubleType()) \
    .add("high", DoubleType()) \
    .add("low", DoubleType()) \
    .add("close", DoubleType()) \
    .add("volume", LongType())

# Create Spark session, telling it to use the Kafka connector
spark = SparkSession.builder \
    .appName("StockStreamProcessor") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.2") \
    .config("spark.hadoop.io.native.lib.available", "false") \
    .config("spark.driver.host", "127.0.0.1") \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Read the raw stream from Kafka
raw_stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "stock-prices") \
    .option("startingOffsets", "latest") \
    .load()

# Kafka messages arrive as raw bytes in a column called "value" — parse them as JSON
parsed_stream = raw_stream.select(
    from_json(col("value").cast("string"), schema).alias("data")
).select("data.*")

# Just print each parsed row to the console, to prove it's working
query = parsed_stream.writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", "false") \
    .option("checkpointLocation", "C:/Shane/stockprojectkafka/checkpoint") \
    .start()

query.awaitTermination()