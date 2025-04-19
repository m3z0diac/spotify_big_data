from pyspark.sql import SparkSession

# Stop any existing Spark session
if 'spark' in locals():
    spark.stop()

# Create a new session with updated MongoDB config
spark = SparkSession.builder \
    .appName("SpotifyMongo") \
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:10.3.0") \
    .config("spark.mongodb.read.connection.uri", "mongodb://localhost:27017") \
    .config("spark.mongodb.write.connection.uri", "mongodb://localhost:27017") \
    .getOrCreate()

# Rest of your code remains the same
df = spark.read.csv("tracks_features.csv", header=True, inferSchema=True)
df.show(5)

df_cleaned = df.dropna(subset=["id", "name", "artists", "danceability", "energy"])
df_cleaned = df_cleaned.withColumnRenamed("name", "track_name")
df_cleaned = df_cleaned.withColumn("duration_min", df_cleaned["duration_ms"] / 60000)

df_final = df_cleaned.select("id", "track_name", "artists", "year", "tempo", "duration_min", "danceability", "energy")

df_final.write \
    .format("mongodb") \
    .mode("overwrite") \
    .option("uri", "mongodb://localhost:27017") \
    .option("database", "spotify") \
    .option("collection", "songs") \
    .save()
