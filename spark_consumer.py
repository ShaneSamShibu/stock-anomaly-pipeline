import os
os.environ["HADOOP_HOME"] = "C:\\hadoop"
os.environ["PATH"] = "C:\\hadoop\\bin;" + os.environ["PATH"]

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, window, avg, stddev, to_timestamp
from pyspark.sql.types import StructType, StringType, DoubleType, LongType

# Define the schema matching our producer's message format
schema = StructType() \
    .add("ticker", StringType()) \
    .add("timestamp", StringType()) \
    .add("open", DoubleType()) \
    .add("high", DoubleType()) \
    .add("low", DoubleType()) \
    .add("close", DoubleType()) \
    .add("volume", LongType())

spark = SparkSession.builder \
    .appName("StockStreamProcessor") \
    .master("local[*]") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.2") \
    .config("spark.hadoop.io.native.lib.available", "false") \
    .config("spark.driver.host", "127.0.0.1") \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

raw_stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "stock-prices") \
    .option("startingOffsets", "latest") \
    .load()

# Parse JSON and convert the timestamp string into a real timestamp type
parsed_stream = raw_stream.select(
    from_json(col("value").cast("string"), schema).alias("data")
).select("data.*") \
 .withColumn("event_time", to_timestamp(col("timestamp")))

# Tell Spark: wait up to 2 minutes for late-arriving data before finalizing a window
windowed_stats = parsed_stream \
    .withWatermark("event_time", "2 minutes") \
    .groupBy(
        window(col("event_time"), "5 minutes"),
        col("ticker")
    ) \
    .agg(
        avg("close").alias("rolling_avg_close"),
        stddev("close").alias("volatility")
    )

query = windowed_stats.writeStream \
    .outputMode("update") \
    .format("console") \
    .option("truncate", "false") \
    .option("checkpointLocation", "C:/Shane/stockprojectkafka/checkpoint") \
    .start()

query.awaitTermination()