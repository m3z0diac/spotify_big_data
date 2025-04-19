from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, max, col

spark = SparkSession.builder \
    .appName("SpotifyAnalysis") \
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:10.3.0") \
    .config("spark.mongodb.read.connection.uri", "mongodb://localhost:27017") \
    .getOrCreate()

df = spark.read \
    .format("mongodb") \
    .option("database", "spotify") \
    .option("collection", "songs") \
    .load()

df.printSchema()

most_energetic = df.orderBy(col("energy").desc()).select("track_name", "artists", "energy").limit(5)
most_energetic.show()

tempo_energy_corr = df.stat.corr("tempo", "energy")
print(f"\nCorrelation between tempo and energy: {tempo_energy_corr:.2f}")

longest_tracks = df.orderBy(col("duration_min").desc()).select("track_name", "duration_min").limit(5)
longest_tracks.show()

top_years = df.groupBy("year").count().orderBy(col("count").desc()).limit(5)
top_years.show()

spark.stop()

