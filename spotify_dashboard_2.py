from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, count, explode, split, col
import matplotlib.pyplot as plt

# Start Spark session
spark = SparkSession.builder \
    .appName("SpotifyCharts") \
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:10.3.0") \
    .config("spark.mongodb.read.connection.uri", "mongodb://localhost:27017") \
    .getOrCreate()

# Read from MongoDB
df = spark.read \
    .format("mongodb") \
    .option("database", "spotify") \
    .option("collection", "songs") \
    .load()

# Clean and prep
df = df.select("track_name", "artists", "year", "tempo", "duration_min", "danceability", "energy") \
       .dropna()

# === 1. Yearly Music Trends ===
yearly_avg = df.groupBy("year").agg(
    count("*").alias("num_songs"),
    avg("danceability").alias("avg_danceability"),
    avg("energy").alias("avg_energy"),
    avg("tempo").alias("avg_tempo"),
    avg("duration_min").alias("avg_duration_min")
).orderBy("year")

# === 2. Top Artists by Danceability ===
# Convert comma-separated string of artists into separate rows
df_artists = df.withColumn("artist", explode(split(col("artists"), ",\s*")))

# Group by individual artist instead of full string
top_dance = df_artists.groupBy("artist").agg(
    avg("danceability").alias("avg_danceability")
).orderBy("avg_danceability", ascending=False).limit(10)

# Convert to Pandas
yearly_df = yearly_avg.toPandas()
top_dance_df = top_dance.toPandas()

# === CHARTS ===

# Chart 1: Average Features by Year
plt.figure(figsize=(14, 6))
plt.plot(yearly_df["year"], yearly_df["avg_danceability"], label="Danceability")
plt.plot(yearly_df["year"], yearly_df["avg_energy"], label="Energy")
plt.plot(yearly_df["year"], yearly_df["avg_tempo"], label="Tempo")
plt.plot(yearly_df["year"], yearly_df["avg_duration_min"], label="Duration (min)")
plt.title("Average Song Features by Year")
plt.xlabel("Year")
plt.ylabel("Average Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("avg_features_by_year.png")
plt.show()

# Chart 2: Number of Songs per Year
plt.figure(figsize=(14, 6))
plt.bar(yearly_df["year"], yearly_df["num_songs"], color='skyblue')
plt.title("Number of Songs Released Per Year")
plt.xlabel("Year")
plt.ylabel("Number of Songs")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("songs_per_year.png")
plt.show()

# Chart 3: Top Artists by Danceability
plt.figure(figsize=(12, 6))
plt.barh(top_dance_df["artist"], top_dance_df["avg_danceability"], color='green')
plt.title("Top 10 Artists by Average Danceability")
plt.xlabel("Average Danceability")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("top_artists_danceability.png")
plt.show()

# Stop Spark
spark.stop()
